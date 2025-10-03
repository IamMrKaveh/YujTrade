import asyncio
import time
from functools import wraps
from typing import Dict, Tuple, Type, Coroutine, Any, Callable
import random

import numpy as np
import pandas as pd

from module.logger_config import logger
from module.core import SignalType, DynamicLevels


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
                oldest_request_time = reqs[0] if reqs else now
                sleep_time = self.time_window - (now - oldest_request_time)
                
                if sleep_time > 0:
                    logger.debug(f"Rate limit hit for {endpoint}. Sleeping for {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                
                now = time.time()
                self.requests[endpoint] = [t for t in self.requests[endpoint] if now - t < self.time_window]
            
            self.requests[endpoint].append(time.time())


def async_retry(attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, jitter: float = 0.5, 
                exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    def decorator(func: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Coroutine[Any, Any, Any]]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            for attempt in range(attempts):
                try:
                    return await func(*args, **kwargs)
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


def calculate_dynamic_levels(data: pd.DataFrame, signal_type: SignalType, volatility: float) -> DynamicLevels:
    last_close = data['close'].iloc[-1]
    
    atr = volatility / 100 * last_close if volatility > 0 else data['close'].pct_change().std() * last_close
    
    if pd.isna(atr) or atr == 0 or (isinstance(atr, (int, float)) and atr == 0):
        atr = last_close * 0.01

    if signal_type == SignalType.BUY:
        primary_entry = float(last_close)
        secondary_entry = float(last_close - 0.5 * atr)
        primary_exit = float(last_close + 2 * atr)
        secondary_exit = float(last_close + 3.5 * atr)
        tight_stop = float(primary_entry - 1.2 * atr)
        wide_stop = float(primary_entry - 2.0 * atr)
        breakeven_point = float(primary_entry + 0.2 * atr)
    else:
        primary_entry = float(last_close)
        secondary_entry = float(last_close + 0.5 * atr)
        primary_exit = float(last_close - 2 * atr)
        secondary_exit = float(last_close - 3.5 * atr)
        tight_stop = float(primary_entry + 1.2 * atr)
        wide_stop = float(primary_entry + 2.0 * atr)
        breakeven_point = float(primary_entry - 0.2 * atr)

    return DynamicLevels(
        primary_entry=primary_entry,
        secondary_entry=secondary_entry,
        primary_exit=primary_exit,
        secondary_exit=secondary_exit,
        tight_stop=tight_stop,
        wide_stop=wide_stop,
        breakeven_point=breakeven_point,
        trailing_stop=float(atr * 0.7)
    )


def calculate_risk_reward_ratio(entry: float, stop_loss: float, take_profit: float, signal_type: SignalType) -> float:
    if signal_type == SignalType.BUY:
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
    else:
        risk = abs(stop_loss - entry)
        reward = abs(entry - take_profit)
    
    return reward / risk if risk > 0 else 0

