from logger_config import logger
from numba import jit
import numpy as np
import pandas as pd
from .moving_averages import _fast_sma
from .cache_utils import _cached_indicator_calculation, NUMBA_AVAILABLE

@jit(nopython=True)
def _fast_bollinger_bands(prices, period, std_dev):
    """محاسبه سریع باندهای بولینگر"""
    sma = _fast_sma(prices, period)
    
    # محاسبه انحراف معیار
    std = np.empty(len(prices))
    std[:period-1] = np.nan
    
    for i in range(period-1, len(prices)):
        window = prices[i-period+1:i+1]
        std[i] = np.std(window)
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return sma, upper_band, lower_band

def _calculate_bollinger_bands(df, period, std_dev):
    try:
        if df is None or len(df) < period:
            return None
        
        prices = df['close'].values.astype(np.float64)
        if NUMBA_AVAILABLE:
            sma, upper_band, lower_band = _fast_bollinger_bands(prices, period, std_dev)
        else:
            sma = df['close'].rolling(window=period).mean().values
            std = df['close'].rolling(window=period).std().values
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
        
        return {
            'bb_upper': pd.Series(upper_band, index=df.index),
            'bb_middle': pd.Series(sma, index=df.index),
            'bb_lower': pd.Series(lower_band, index=df.index)
        }
    except Exception as e:
        logger.warning(f"Error in optimized Bollinger Bands calculation: {e}")
        return None
        
def _calculate_optimized_bollinger_bands(df, period=20, std_dev=2):
    """محاسبه بهینه باندهای بولینگر"""
    
    return _cached_indicator_calculation(df, 'bollinger_bands', _calculate_bollinger_bands, period, std_dev)

def _calculate_average_true_range(df, period=14):
    """محاسبه Average True Range"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    except Exception as e:
        logger.warning(f"Error calculating ATR: {e}")
        return None

def _calculate_atr(df, period):
    """Internal ATR calculation function"""
    try:
        if df is None or len(df) < period:
            return None
            
        high = df['high']
        low = df['low']
        close = df['close']
        
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        # Calculate True Range as the maximum of the three values
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR as the rolling mean of True Range
        atr = true_range.rolling(window=period).mean()
        
        return atr
        
    except Exception as e:
        logger.warning(f"Error calculating ATR: {e}")
        return None

def _calculate_standard_deviation(df, period=20):
    """محاسبه Standard Deviation"""
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        std_dev = close.rolling(window=period).std()
        
        return std_dev
    except Exception as e:
        logger.warning(f"Error calculating Standard Deviation: {e}")
        return None

def _calculate_price_std(df, period=20):
    """محاسبه انحراف معیار قیمت"""
    try:
        if df is None or len(df) < period:
            return None
            
        close = df['close']
        std_dev = close.rolling(period).std()
        
        return std_dev
    except Exception:
        return None

def _calculate_market_volatility(df, period=20):
    """محاسبه نوسانات بازار"""
    return _cached_indicator_calculation(df, 'market_volatility', _calculate_market_volatility, period)

