from imports import NUMBA_AVAILABLE, jit, np, pd
from signals.indicators.cache_utils import _cached_indicator_calculation
from logger_config import logger

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

def _calculate_optimized_sma(df, column='close', period=20):
    """محاسبه بهینه میانگین متحرک"""
    
    return _cached_indicator_calculation(df, 'sma', _calculate_sma, column, period)


def _calculate_sma(df, column, period):
    try:
        prices = df[column].values.astype(np.float64)
        if NUMBA_AVAILABLE:
            result = _fast_sma(prices, period)
        else:
            result = df[column].rolling(window=period).mean().values
        
        return pd.Series(result, index=df.index)
    except Exception as e:
        logger.warning(f"Error in optimized SMA calculation: {e}")
        return None


def _calculate_kama(df, period=10, fast_sc=2, slow_sc=30):
    """محاسبه Kaufman Adaptive Moving Average"""
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
