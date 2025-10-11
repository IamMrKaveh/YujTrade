import asyncio
import time
from functools import wraps
from typing import Dict, Tuple, Type, Coroutine, Any, Callable
import random

import numpy as np
import pandas as pd
import talib

from .logger_config import logger


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
                    logger.debug(
                        f"Rate limit hit for {endpoint}. Sleeping for {sleep_time:.2f}s"
                    )
                    await asyncio.sleep(sleep_time)

                now = time.time()
                self.requests[endpoint] = [
                    t for t in reqs if now - t < self.time_window
                ]

            self.requests[endpoint].append(time.time())


class IndicatorNormalizer:
    def __init__(self):
        self.expected_ranges = {
            "rsi": (0, 100),
            "macd": (-10, 10),
            "stoch": (0, 100),
            "cci": (-300, 300),
            "williams_r": (-100, 0),
            "mfi": (0, 100),
            "bb": (0, 3),
            "atr": (0, 100),
            "adx": (0, 100),
            "volume": (0, float("inf")),
            "cmf": (-1, 1),
            "obv": (-float("inf"), float("inf")),
            "vwap": (0, float("inf")),
            "aroon": (0, 100),
            "uo": (0, 100),
            "roc": (-50, 50),
            "ad_line": (-float("inf"), float("inf")),
            "force_index": (-float("inf"), float("inf")),
            "vwma": (0, float("inf")),
            "keltner": (0, float("inf")),
            "donchian": (0, float("inf")),
            "trix": (-1, 1),
            "eom": (-0.5, 0.5),
            "std_dev": (0, 10),
            "stochrsi": (0, 100),
            "kst": (-50, 50),
            "mass": (10, 40),
            "corr_coef": (-1, 1),
            "elder_ray": (-10, 10),
            "momentum": (-50, 50),
            "dpo": (-20, 20),
            "choppiness": (0, 100),
            "vortex": (0, 2),
            "awesome": (-10, 10),
            "cmo": (-100, 100),
            "rvi": (-1, 1),
            "pvr": (0, 100),
            "ado": (-float("inf"), float("inf")),
            "pvt": (-float("inf"), float("inf")),
            "bop": (-1, 1),
            "linreg": (0, float("inf")),
            "linreg_slope": (-10, 10),
            "median_price": (0, float("inf")),
            "typical_price": (0, float("inf")),
            "weighted_close": (0, float("inf")),
            "hma": (0, float("inf")),
            "zlema": (0, float("inf")),
            "kama": (0, float("inf")),
            "t3": (0, float("inf")),
            "dema": (0, float("inf")),
            "tema": (0, float("inf")),
            "fisher": (-3, 3),
            "stc": (0, 100),
            "qqe": (0, 100),
            "connors_rsi": (0, 100),
            "smi": (-100, 100),
            "tsi": (-100, 100),
            "gann_hilo": (0, float("inf")),
            "ma_ribbon": (0, 100),
            "fractal": (0, float("inf")),
            "chaikin_vol": (-10, 10),
            "historical_vol": (0, 100),
            "ulcer_index": (0, 50),
            "atr_bands": (0, float("inf")),
            "bbw": (0, 1),
            "volume_osc": (-50, 50),
            "kvo": (-float("inf"), float("inf")),
            "frama": (0, float("inf")),
            "vidya": (0, float("inf")),
            "mama": (0, float("inf")),
            "rmi": (0, 100),
            "rsi2": (0, 100),
            "ppo": (-10, 10),
            "pvo": (-50, 50),
            "nvi": (0, float("inf")),
            "pvi": (0, float("inf")),
            "mfi_bw": (0, float("inf")),
            "ht_dc": (0, 100),
            "ht_trend_mode": (0, 1),
            "er": (0, 1),
            "coppock": (-20, 20),
            "bop_rsi": (0, 100),
            "price_action": (0, 100),
            "market_structure": (0, 100),
            "liquidity_levels": (0, float("inf")),
            "vw_rsi": (0, 100),
            "smc": (0, 100),
            "wyckoff_vsa": (0, 100),
            "adi": (0, float("inf")),
            "tii": (0, 100),
            "order_flow": (-100, 100),
        }

    def normalize_indicator(self, indicator_result, indicator_name: str):
        if (
            not hasattr(indicator_result, "value")
            or indicator_result.value is None
            or pd.isna(indicator_result.value)
        ):
            return indicator_result

        expected_min, expected_max = self.expected_ranges.get(
            indicator_name.lower(), (-1, 1)
        )

        if expected_max == float("inf") and expected_min == -float("inf"):
            return indicator_result
        elif expected_max == float("inf"):
            if indicator_result.value > expected_min:
                normalized_value = min(
                    (indicator_result.value - expected_min) / 10 + 0.5, 1.0
                )
            else:
                normalized_value = -1.0
        elif expected_min == -float("inf"):
            if indicator_result.value < expected_max:
                normalized_value = max(
                    (indicator_result.value - expected_max) / 10 - 0.5, -1.0
                )
            else:
                normalized_value = 1.0
        else:
            if expected_max > expected_min:
                normalized_value = (
                    2
                    * (indicator_result.value - expected_min)
                    / (expected_max - expected_min)
                    - 1
                )
                normalized_value = np.clip(normalized_value, -1.0, 1.0)
            else:
                normalized_value = 0.0

        normalized_result = type(indicator_result)(
            name=indicator_result.name,
            value=float(normalized_value),
            signal_strength=indicator_result.signal_strength,
            interpretation=indicator_result.interpretation,
        )

        return normalized_result


__all__ = [
    "RateLimiter",
    "async_retry",
    "get_support_resistance_strength",
    "calculate_fibonacci_levels",
    "calculate_pivot_points",
    "calculate_dynamic_levels",
    "calculate_risk_reward_ratio",
    "validate_signal_timing",
    "detect_market_regime",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "format_number",
    "calculate_correlation_matrix",
    "IndicatorNormalizer",
]


def async_retry(
    attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    jitter: float = 0.5,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ignore_exceptions: Tuple[Type[Exception], ...] = (),
):
    def decorator(
        func: Callable[..., Coroutine[Any, Any, Any]],
    ) -> Callable[..., Coroutine[Any, Any, Any]]:
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
                    actual_delay = current_delay + random.uniform(
                        0, jitter * current_delay
                    )
                    logger.warning(
                        f"Attempt {attempt + 1}/{attempts} for {func.__name__} failed: {e}. "
                        f"Retrying in {actual_delay:.2f}s..."
                    )
                    await asyncio.sleep(actual_delay)
                    current_delay *= backoff
            logger.error(f"Function {func.__name__} failed after {attempts} attempts.")
            if last_exception:
                raise last_exception
            raise RuntimeError(
                f"Function {func.__name__} failed without a clear exception."
            )

        return wrapper

    return decorator


def get_support_resistance_strength(data: pd.DataFrame, level: float) -> float:
    if (
        data.empty
        or "high" not in data.columns
        or "low" not in data.columns
        or level is None
    ):
        return 0.0

    price_range = data["high"].max() - data["low"].min()
    if price_range == 0:
        return 1.0

    tolerance = price_range * 0.01

    touches = data[
        (abs(data["high"] - level) <= tolerance)
        | (abs(data["low"] - level) <= tolerance)
        | ((data["low"] <= level) & (data["high"] >= level))
    ]

    if touches.empty:
        return 0.0

    touch_count = len(touches)
    touch_score = np.log1p(touch_count) * 2.0

    volume_score = 1.0
    if (
        "volume" in touches.columns
        and not touches["volume"].empty
        and touches["volume"].mean() > 0
    ):
        avg_volume_at_touch = touches["volume"].mean()
        avg_volume_total = data["volume"].mean()
        if avg_volume_total > 0:
            volume_ratio = avg_volume_at_touch / avg_volume_total
            volume_score = np.clip(volume_ratio, 0.5, 2.0)

    recency_score = 0.5
    if isinstance(data.index, pd.DatetimeIndex) and not touches.empty:
        time_span = (data.index.max() - data.index.min()).total_seconds()
        if time_span > 0:
            last_touch = touches.index.max()
            days_since_touch = (data.index.max() - last_touch).total_seconds() / 86400
            recency_score = np.exp(-days_since_touch / 30)

    bounce_score = 0.0
    for idx in touches.index:
        pos = data.index.get_loc(idx)
        if pos < len(data) - 5:
            future_data = data.iloc[pos : pos + 5]
            if level < data.loc[idx, "close"]:
                bounce = (future_data["close"].max() - level) / price_range
            else:
                bounce = (level - future_data["close"].min()) / price_range
            bounce_score += np.clip(bounce, 0, 0.1)

    bounce_score = bounce_score / max(len(touches), 1) * 10

    strength = (
        touch_score * 0.35
        + volume_score * 0.25
        + recency_score * 0.25
        + bounce_score * 0.15
    )

    return np.clip(strength, 0, 10)


def calculate_fibonacci_levels(
    data: pd.DataFrame, lookback: int = 100
) -> Dict[str, float]:
    if len(data) < lookback:
        lookback = len(data)

    recent_data = data.tail(lookback)
    highest_high = recent_data["high"].max()
    lowest_low = recent_data["low"].min()
    diff = highest_high - lowest_low

    if data["close"].iloc[-1] > data["close"].iloc[-lookback]:
        levels = {
            "fib_0": highest_high,
            "fib_236": highest_high - (diff * 0.236),
            "fib_382": highest_high - (diff * 0.382),
            "fib_500": highest_high - (diff * 0.500),
            "fib_618": highest_high - (diff * 0.618),
            "fib_786": highest_high - (diff * 0.786),
            "fib_100": lowest_low,
            "fib_1272": highest_high + (diff * 0.272),
            "fib_1618": highest_high + (diff * 0.618),
        }
    else:
        levels = {
            "fib_0": lowest_low,
            "fib_236": lowest_low + (diff * 0.236),
            "fib_382": lowest_low + (diff * 0.382),
            "fib_500": lowest_low + (diff * 0.500),
            "fib_618": lowest_low + (diff * 0.618),
            "fib_786": lowest_low + (diff * 0.786),
            "fib_100": highest_high,
            "fib_1272": lowest_low - (diff * 0.272),
            "fib_1618": lowest_low - (diff * 0.618),
        }

    return levels


def calculate_pivot_points(data: pd.DataFrame) -> Dict[str, float]:
    if len(data) < 1:
        return {}

    last_candle = data.iloc[-1]
    high = last_candle["high"]
    low = last_candle["low"]
    close = last_candle["close"]

    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)

    woody_pivot = (high + low + 2 * close) / 4
    woody_r1 = 2 * woody_pivot - low
    woody_s1 = 2 * woody_pivot - high

    diff = high - low
    camarilla_r4 = close + diff * 1.1 / 2
    camarilla_r3 = close + diff * 1.1 / 4
    camarilla_r2 = close + diff * 1.1 / 6
    camarilla_r1 = close + diff * 1.1 / 12
    camarilla_s1 = close - diff * 1.1 / 12
    camarilla_s2 = close - diff * 1.1 / 6
    camarilla_s3 = close - diff * 1.1 / 4
    camarilla_s4 = close - diff * 1.1 / 2

    return {
        "pivot": pivot,
        "r1": r1,
        "s1": s1,
        "r2": r2,
        "s2": s2,
        "r3": r3,
        "s3": s3,
        "woody_pivot": woody_pivot,
        "woody_r1": woody_r1,
        "woody_s1": woody_s1,
        "cam_r4": camarilla_r4,
        "cam_r3": camarilla_r3,
        "cam_r2": camarilla_r2,
        "cam_r1": camarilla_r1,
        "cam_s1": camarilla_s1,
        "cam_s2": camarilla_s2,
        "cam_s3": camarilla_s3,
        "cam_s4": camarilla_s4,
    }


def calculate_dynamic_levels(
    data: pd.DataFrame, signal_type, market_context, order_book=None
) -> Dict[str, Any]:
    last_close = data["close"].iloc[-1]

    adx_val = market_context.adx if hasattr(market_context, "adx") else 25

    volatility_multiplier = 1.0 + np.clip(market_context.volatility / 10, 0, 1)

    if market_context.volatility > 0:
        atr = (market_context.volatility / 100 * last_close) * volatility_multiplier
    else:
        atr = data["close"].pct_change().std() * last_close

    if pd.isna(atr) or atr <= 0:
        atr = last_close * 0.02

    min_rr = 2.0 if adx_val > 35 else 1.8

    entry = float(last_close)

    risk = atr * (1.5 if adx_val > 40 else 1.2)

    if signal_type == "buy":
        base_stop_loss = entry - risk
        base_take_profit = entry + (risk * min_rr)
    else:
        base_stop_loss = entry + risk
        base_take_profit = entry - (risk * min_rr)

    fib_levels = calculate_fibonacci_levels(data)
    pivot_levels = calculate_pivot_points(data)

    if signal_type == "buy":
        valid_supports = [
            s
            for s in market_context.support_levels
            if isinstance(s, (int, float)) and s < entry and s > base_stop_loss - risk
        ]
        if valid_supports:
            nearest_support = max(valid_supports)
            if abs(nearest_support - base_stop_loss) < risk * 0.5:
                stop_loss = nearest_support - (risk * 0.1)
            else:
                stop_loss = base_stop_loss
        else:
            stop_loss = base_stop_loss

        valid_resistances = [
            r
            for r in market_context.resistance_levels
            if isinstance(r, (int, float)) and r > entry and r < base_take_profit + risk
        ]
        if valid_resistances:
            nearest_resistance = min(valid_resistances)
            if abs(nearest_resistance - base_take_profit) < risk * 0.5:
                take_profit = nearest_resistance - (risk * 0.1)
            else:
                take_profit = base_take_profit
        else:
            take_profit = base_take_profit
    else:
        valid_resistances = [
            r
            for r in market_context.resistance_levels
            if isinstance(r, (int, float)) and r > entry and r < base_stop_loss + risk
        ]
        if valid_resistances:
            nearest_resistance = min(valid_resistances)
            if abs(nearest_resistance - base_stop_loss) < risk * 0.5:
                stop_loss = nearest_resistance + (risk * 0.1)
            else:
                stop_loss = base_stop_loss
        else:
            stop_loss = base_stop_loss

        valid_supports = [
            s
            for s in market_context.support_levels
            if isinstance(s, (int, float)) and s < entry and s > base_take_profit - risk
        ]
        if valid_supports:
            nearest_support = max(valid_supports)
            if abs(nearest_support - base_take_profit) < risk * 0.5:
                take_profit = nearest_support + (risk * 0.1)
            else:
                take_profit = base_take_profit
        else:
            take_profit = base_take_profit

    if order_book and hasattr(order_book, "bids") and signal_type == "buy":
        major_bid = (
            max([p for p, _ in order_book.bids[:5]]) if order_book.bids else None
        )
        if major_bid and abs(major_bid - stop_loss) / stop_loss < 0.005:
            stop_loss = major_bid * 0.995

    trailing_stop_distance = atr * 0.8
    breakeven_distance = abs(entry - stop_loss) * 0.5

    return {
        "primary_entry": entry,
        "secondary_entry": entry,
        "primary_exit": take_profit,
        "secondary_exit": take_profit,
        "tight_stop": stop_loss,
        "wide_stop": stop_loss,
        "breakeven_point": entry,
        "trailing_stop": trailing_stop_distance,
    }


def calculate_risk_reward_ratio(
    entry_price: float, stop_loss: float, take_profit: float, signal_type
) -> float:
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)

    if risk == 0:
        return 0.0

    commission_per_trade = 0.001
    slippage = 0.0005

    effective_risk = risk * (1 + commission_per_trade + slippage)
    effective_reward = reward * (1 - commission_per_trade - slippage)

    return effective_reward / effective_risk if effective_risk > 0 else 0.0


def validate_signal_timing(
    data: pd.DataFrame, signal_type, min_candles_from_extreme: int = 3
) -> Tuple[bool, str]:
    if len(data) < min_candles_from_extreme + 5:
        return True, "Not enough data for validation"

    recent_data = data.tail(min_candles_from_extreme + 5)
    current_price = data["close"].iloc[-1]

    highest_high = recent_data["high"].max()
    lowest_low = recent_data["low"].min()
    price_range = highest_high - lowest_low

    if price_range == 0:
        return True, "Invalid price range"

    if signal_type == "buy":
        distance_from_high = (highest_high - current_price) / price_range
        if distance_from_high < 0.1:
            return False, "Too close to recent high for BUY signal"

        lowest_index = recent_data["low"].idxmin()
        candles_since_low = (
            len(recent_data) - recent_data.index.get_loc(lowest_index) - 1
        )

        if candles_since_low < min_candles_from_extreme:
            return False, f"Too soon after recent low ({candles_since_low} candles)"

    else:
        distance_from_low = (current_price - lowest_low) / price_range
        if distance_from_low < 0.1:
            return False, "Too close to recent low for SELL signal"

        highest_index = recent_data["high"].idxmax()
        candles_since_high = (
            len(recent_data) - recent_data.index.get_loc(highest_index) - 1
        )

        if candles_since_high < min_candles_from_extreme:
            return False, f"Too soon after recent high ({candles_since_high} candles)"

    last_candle = data.iloc[-1]
    body_size = abs(last_candle["close"] - last_candle["open"])
    total_size = last_candle["high"] - last_candle["low"]

    if total_size > 0:
        body_ratio = body_size / total_size
        if body_ratio < 0.3:
            return False, "Current candle has very long shadows (potential reversal)"

    return True, "Signal timing validated"


def detect_market_regime(data: pd.DataFrame, lookback: int = 50) -> Dict[str, Any]:
    if len(data) < lookback:
        lookback = len(data)

    recent_data = data.tail(lookback)

    try:
        adx = talib.ADX(
            recent_data["high"].to_numpy(),
            recent_data["low"].to_numpy(),
            recent_data["close"].to_numpy(),
            timeperiod=14,
        )
        current_adx = adx[-1] if len(adx) > 0 and not pd.isna(adx[-1]) else 20
    except:
        current_adx = 20

    if current_adx > 25:
        market_type = "trending"
    elif current_adx < 20:
        market_type = "ranging"
    else:
        market_type = "transitional"

    returns = recent_data["close"].pct_change()
    volatility = returns.std() * np.sqrt(252)

    if volatility > returns.std() * 1.5:
        volatility_regime = "high"
    elif volatility < returns.std() * 0.5:
        volatility_regime = "low"
    else:
        volatility_regime = "normal"

    try:
        from .analyzers import MarketConditionAnalyzer

        analyzer = MarketConditionAnalyzer()
        hurst = analyzer._calculate_hurst_exponent(recent_data["close"])

        if hurst:
            if hurst > 0.55:
                trend_persistence = "persistent"
            elif hurst < 0.45:
                trend_persistence = "mean_reverting"
            else:
                trend_persistence = "random_walk"
        else:
            trend_persistence = "unknown"
    except:
        trend_persistence = "unknown"

    if market_type == "trending" and trend_persistence == "persistent":
        recommended_strategy = "trend_following"
    elif market_type == "ranging":
        recommended_strategy = "mean_reversion"
    elif volatility_regime == "high":
        recommended_strategy = "breakout"
    else:
        recommended_strategy = "balanced"

    return {
        "market_type": market_type,
        "adx": round(current_adx, 2),
        "volatility": round(volatility * 100, 2),
        "volatility_regime": volatility_regime,
        "trend_persistence": trend_persistence,
        "hurst_exponent": round(hurst, 3) if hurst else None,
        "recommended_strategy": recommended_strategy,
    }


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    if len(returns) < 2:
        return 0.0

    excess_returns = returns.mean() - (risk_free_rate / 252)
    std_dev = returns.std()

    if std_dev == 0:
        return 0.0

    sharpe = (excess_returns / std_dev) * np.sqrt(252)
    return round(sharpe, 3)


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    if len(returns) < 2:
        return 0.0

    excess_returns = returns.mean() - (risk_free_rate / 252)
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0:
        return float("inf")

    downside_std = downside_returns.std()

    if downside_std == 0:
        return 0.0

    sortino = (excess_returns / downside_std) * np.sqrt(252)
    return round(sortino, 3)


def calculate_max_drawdown(data: pd.DataFrame) -> Tuple[float, int]:
    if len(data) < 2:
        return 0.0, 0

    cumulative = (1 + data["close"].pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    max_dd = drawdown.min()
    max_dd_percentage = abs(max_dd * 100)

    dd_end_idx = drawdown.idxmin()
    dd_start_idx = (cumulative[:dd_end_idx]).expanding().max().idxmax()

    if isinstance(data.index, pd.DatetimeIndex):
        duration = (dd_end_idx - dd_start_idx).days
    else:
        duration = int(
            data.index.get_loc(dd_end_idx) - data.index.get_loc(dd_start_idx)
        )

    return round(max_dd_percentage, 2), duration


def format_number(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}"


def calculate_correlation_matrix(symbols_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not symbols_data:
        return pd.DataFrame()

    closes = pd.DataFrame(
        {symbol: data["close"] for symbol, data in symbols_data.items()}
    )

    returns = closes.pct_change().dropna()
    correlation_matrix = returns.corr()

    return correlation_matrix
