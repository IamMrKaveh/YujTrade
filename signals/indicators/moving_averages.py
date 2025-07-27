import numpy as np
import pandas as pd
from logger_config import logger
from .cache_utils import cached_calculation, NUMBA_AVAILABLE, jit

@jit(nopython=True)
def _fast_sma(prices, period):
    """محاسبه سریع میانگین متحرک"""
    result = np.empty(len(prices))
    result[:period-1] = np.nan
    for i in range(period-1, len(prices)):
        result[i] = np.mean(prices[i-period+1:i+1])
    return result

@jit(nopython=True)
def _fast_ema(prices, alpha):
    """محاسبه سریع میانگین متحرک نمایی"""
    result = np.empty(len(prices))
    result[0] = prices[0]
    for i in range(1, len(prices)):
        result[i] = alpha * prices[i] + (1 - alpha) * result[i-1]
    return result

@cached_calculation('sma')
def calculate_sma(df, column='close', period=20):
    """محاسبه بهینه میانگین متحرک ساده با کشینگ"""
    try:
        if df is None or len(df) < period:
            return None
            
        prices = df[column].values.astype(np.float64)
        if NUMBA_AVAILABLE:
            result = _fast_sma(prices, period)
        else:
            result = df[column].rolling(window=period).mean().values
        
        return pd.Series(result, index=df.index)
    except Exception as e:
        logger.warning(f"Error in SMA calculation: {e}")
        return None


@cached_calculation('ema')
def calculate_ema(df, column='close', period=20):
    """محاسبه میانگین متحرک نمایی با کشینگ"""
    try:
        if df is None or len(df) < period:
            return None
            
        prices = df[column].values.astype(np.float64)
        alpha = 2.0 / (period + 1)
        
        if NUMBA_AVAILABLE:
            result = _fast_ema(prices, alpha)
        else:
            result = df[column].ewm(span=period).mean().values
            
        return pd.Series(result, index=df.index)
    except Exception as e:
        logger.warning(f"Error in EMA calculation: {e}")
        return None


@cached_calculation('kama')
def calculate_kama(df, period=10, fast_sc=2, slow_sc=30):
    """محاسبه Kaufman Adaptive Moving Average با کشینگ"""
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        
        # Efficiency Ratio
        change = abs(close - close.shift(period))
        volatility = abs(close - close.shift(1)).rolling(window=period).sum()
        er = change / volatility
        er = er.fillna(0)
        
        # Smoothing Constants
        fastest_sc = 2.0 / (fast_sc + 1)
        slowest_sc = 2.0 / (slow_sc + 1)
        sc = (er * (fastest_sc - slowest_sc) + slowest_sc) ** 2
        
        # KAMA calculation
        kama = []
        kama.append(close.iloc[0])  # First value
        
        for i in range(1, len(close)):
            kama.append(kama[i-1] + sc.iloc[i] * (close.iloc[i] - kama[i-1]))
        
        return pd.Series(kama, index=df.index)
    except Exception as e:
        logger.warning(f"Error calculating KAMA: {e}")
        return None


@cached_calculation('wma')
def calculate_wma(df, column='close', period=20):
    """محاسبه میانگین متحرک وزنی با کشینگ"""
    try:
        if df is None or len(df) < period:
            return None
            
        prices = df[column]
        weights = np.arange(1, period + 1)
        result = prices.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
        
        return result
    except Exception as e:
        logger.warning(f"Error in WMA calculation: {e}")
        return None


@cached_calculation('vwma')
def calculate_vwma(df, price_column='close', volume_column='volume', period=20):
    """محاسبه میانگین متحرک وزنی حجم با کشینگ"""
    try:
        if df is None or len(df) < period:
            return None
        
        if volume_column not in df.columns:
            logger.warning(f"Volume column '{volume_column}' not found in dataframe")
            return None
            
        prices = df[price_column]
        volumes = df[volume_column]
        
        # Calculate price * volume and rolling sums
        price_volume = prices * volumes
        price_volume_sum = price_volume.rolling(window=period).sum()
        volume_sum = volumes.rolling(window=period).sum()
        
        # Avoid division by zero
        result = price_volume_sum / volume_sum
        return result
    except Exception as e:
        logger.warning(f"Error in VWMA calculation: {e}")
        return None