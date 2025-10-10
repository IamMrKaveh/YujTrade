import asyncio
import time
from functools import wraps
from typing import Dict, Tuple, Type, Coroutine, Any, Callable
import random

import numpy as np
import pandas as pd
import talib

from .logger_config import logger
from .core import SignalType, DynamicLevels, MarketAnalysis


class RateLimiter:
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, list] = {}
        self._lock = asyncio.Lock()

    async def wait_if_needed(self, endpoint: str):
        async with self._lock:
            now = time.time()
            reqs = self.requests.setdefault(endpoint, [])
            
            reqs = [t for t in reqs if now - t < self.time_window]
            self.requests[endpoint] = reqs

            if len(reqs) >= self.max_requests:
                oldest_request_time = reqs[0]
                sleep_time = self.time_window - (now - oldest_request_time)
                if sleep_time > 0:
                    logger.debug(f"Rate limit hit for {endpoint}. Sleeping for {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                
                now = time.time()
                self.requests[endpoint] = [t for t in self.requests[endpoint] if now - t < self.time_window]
            
            self.requests[endpoint].append(time.time())


def async_retry(attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, jitter: float = 0.5,
                exceptions: Tuple[Type[Exception], ...] = (Exception,),
                ignore_exceptions: Tuple[Type[Exception], ...] = ()):
    def decorator(func: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Coroutine[Any, Any, Any]]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            for attempt in range(attempts):
                try:
                    return await func(*args, **kwargs)
                except ignore_exceptions as e:
                    raise e
                except exceptions as e:
                    last_exception = e
                    if attempt + 1 == attempts:
                        break
                    actual_delay = current_delay + random.uniform(0, jitter * current_delay)
                    logger.warning(
                        f"Attempt {attempt + 1}/{attempts} for {func.__name__} failed: {e}. "
                        f"Retrying in {actual_delay:.2f}s..."
                    )
                    await asyncio.sleep(actual_delay)
                    current_delay *= backoff
            logger.error(f"Function {func.__name__} failed after {attempts} attempts.")
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Function {func.__name__} failed without a clear exception.")
        return wrapper
    return decorator


def get_support_resistance_strength(data: pd.DataFrame, level: float) -> float:
    if data.empty or 'high' not in data.columns or 'low' not in data.columns or level is None:
        return 0.0
    price_range = data['high'].max() - data['low'].min()
    if price_range == 0: return 1.0
    
    touches = data[(abs(data['high'] - level) / price_range < 0.01) | (abs(data['low'] - level) / price_range < 0.01)]
    
    if touches.empty: return 0.0
    
    volume_score = 1.0
    if 'volume' in touches.columns and not touches['volume'].empty and touches['volume'].mean() > 0:
        avg_volume_at_touch = touches['volume'].mean()
        avg_volume_total = data['volume'].mean()
        if avg_volume_total > 0:
            volume_score = avg_volume_at_touch / avg_volume_total

    recency_score = 0.0
    if isinstance(data.index, pd.DatetimeIndex) and not touches.empty:
        time_span = (data.index.max() - data.index.min()).total_seconds()
        if time_span > 0:
            recency_score = (touches.index.max() - data.index.min()).total_seconds() / time_span

    strength = (len(touches) * 0.5) + (volume_score * 0.3) + (recency_score * 0.2)
    return np.clip(strength, 0, 10)


def calculate_dynamic_levels(data: pd.DataFrame, signal_type: SignalType, volatility: float, market_context: MarketAnalysis) -> DynamicLevels:
    last_close = data['close'].iloc[-1]

    try:
        adx = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        last_adx = adx.iloc[-1] if not adx.empty and not pd.isna(adx.iloc[-1]) else 25
    except Exception:
        last_adx = 25
        
    if last_adx > 40:
        atr_multiplier = 2.0
    elif last_adx < 20:
        atr_multiplier = 1.2
    else:
        atr_multiplier = 1.5

    volatility_multiplier = 1.0 + np.clip(volatility / 10, 0, 1)
    atr = (volatility / 100 * last_close) * volatility_multiplier if volatility > 0 else (data['close'].pct_change().std() * last_close)
    if pd.isna(atr) or atr <= 0:
        atr = last_close * 0.02

    min_rr = 2.0
    entry = float(last_close)

    if signal_type == SignalType.BUY:
        risk = atr_multiplier * atr
        base_stop_loss = entry - risk
        base_take_profit = entry + (risk * min_rr)
    else: 
        risk = atr_multiplier * atr
        base_stop_loss = entry + risk
        base_take_profit = entry - (risk * min_rr)

    stop_loss = base_stop_loss
    take_profit = base_take_profit

    if market_context and market_context.support_levels and market_context.resistance_levels:
        support = market_context.support_levels[0] if market_context.support_levels else None
        resistance = market_context.resistance_levels[0] if market_context.resistance_levels else None
        
        support_strength = get_support_resistance_strength(data, support)
        resistance_strength = get_support_resistance_strength(data, resistance)

        if signal_type == SignalType.BUY:
            if support and support < entry and support > base_stop_loss and support_strength > 3:
                stop_loss = support - (0.5 * atr * (1 / max(1, support_strength/2)))
            if resistance and resistance > entry and resistance < base_take_profit and resistance_strength > 3:
                take_profit = resistance - (0.2 * atr * (1 / max(1, resistance_strength/2)))
        else:
            if resistance and resistance > entry and resistance < base_stop_loss and resistance_strength > 3:
                stop_loss = resistance + (0.5 * atr * (1 / max(1, resistance_strength/2)))
            if support and support < entry and support > base_take_profit and support_strength > 3:
                take_profit = support + (0.2 * atr * (1 / max(1, support_strength/2)))

    final_risk = abs(entry - stop_loss)
    if final_risk > 0:
        current_rr = abs(take_profit - entry) / final_risk
        if current_rr < min_rr:
            if signal_type == SignalType.BUY:
                take_profit = entry + final_risk * min_rr
            else:
                take_profit = entry - final_risk * min_rr

    return DynamicLevels(
        primary_entry=float(entry),
        secondary_entry=float(entry - (0.5 * atr) if signal_type == SignalType.BUY else entry + (0.5 * atr)),
        primary_exit=float(take_profit),
        secondary_exit=float(take_profit + atr if signal_type == SignalType.BUY else take_profit - atr),
        tight_stop=float(stop_loss),
        wide_stop=float(stop_loss - (0.5 * atr) if signal_type == SignalType.BUY else stop_loss + (0.5 * atr)),
        breakeven_point=float(entry + (0.3 * atr) if signal_type == SignalType.BUY else entry - (0.3 * atr)),
        trailing_stop=float(atr * 0.8)
    )


def calculate_risk_reward_ratio(entry: float, stop_loss: float, take_profit: float, signal_type: SignalType, estimated_spread: float = 0.0005) -> float:
    if signal_type == SignalType.BUY:
        effective_entry = entry * (1 + estimated_spread)
        risk = abs(effective_entry - stop_loss)
        reward = abs(take_profit - effective_entry)
    else:
        effective_entry = entry * (1 - estimated_spread)
        risk = abs(stop_loss - effective_entry)
        reward = abs(effective_entry - take_profit)
        
    return reward / risk if risk > 0 else 0.0