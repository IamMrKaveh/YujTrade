import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List
from logger_config import logger
from .cache_utils import cached_calculation, NUMBA_AVAILABLE, jit

# ==================== Core Calculation Functions ====================

@jit(nopython=True)
def _fast_sma(prices, period):
    """محاسبه سریع میانگین متحرک ساده"""
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
        logger.error(f"Error in SMA calculation: {e}")
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
        logger.error(f"Error in EMA calculation: {e}")
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
        logger.error(f"Error calculating KAMA: {e}")
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
        logger.error(f"Error in WMA calculation: {e}")
        return None

@cached_calculation('vwma')
def calculate_vwma(df, price_column='close', volume_column='volume', period=20):
    """محاسبه میانگین متحرک وزنی حجم با کشینگ"""
    try:
        if df is None or len(df) < period:
            return None
        
        if volume_column not in df.columns:
            logger.error(f"Volume column '{volume_column}' not found in dataframe")
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
        logger.error(f"Error in VWMA calculation: {e}")
        return None

# ==================== Validation Functions ====================

def _validate_dataframe(df: pd.DataFrame) -> bool:
    """اعتبارسنجی DataFrame ورودی"""
    if df is None:
        logger.error("DataFrame is None")
        return False
        
    if df.empty:
        logger.error("DataFrame is empty")
        return False
        
    if 'close' not in df.columns:
        logger.error("'close' column not found in DataFrame")
        return False
        
    return True

def _validate_parameters(params: Dict) -> bool:
    """اعتبارسنجی پارامترهای ورودی"""
    for key, value in params.items():
        if key.endswith('_period') and (not isinstance(value, int) or value <= 0):
            logger.error(f"Invalid period parameter: {key}={value}")
            return False
            
    return True

def _get_required_columns(ma_types: List[str]) -> List[str]:
    """تعیین ستون‌های مورد نیاز برای انواع میانگین متحرک"""
    required_columns = ['close']
    
    if 'vwma' in ma_types:
        required_columns.append('volume')
        
    return required_columns

# ==================== Main Function ====================

def calculate_moving_averages(
    df: pd.DataFrame,
    ma_types: Optional[Union[str, List[str]]] = None,
    sma_period: int = 20,
    ema_period: int = 20,
    kama_period: int = 10,
    kama_fast_sc: int = 2,
    kama_slow_sc: int = 30,
    wma_period: int = 20,
    vwma_period: int = 20,
    price_column: str = 'close',
    volume_column: str = 'volume'
) -> Dict[str, Optional[pd.Series]]:
    """
    محاسبه انواع مختلف میانگین متحرک برای داده‌های قیمت
    
    Args:
        df: DataFrame شامل داده‌های قیمت
        ma_types: انواع میانگین متحرک برای محاسبه ('sma', 'ema', 'kama', 'wma', 'vwma')
        sma_period: دوره زمانی برای SMA
        ema_period: دوره زمانی برای EMA
        kama_period: دوره زمانی برای KAMA
        kama_fast_sc: ضریب سریع KAMA
        kama_slow_sc: ضریب کند KAMA
        wma_period: دوره زمانی برای WMA
        vwma_period: دوره زمانی برای VWMA
        price_column: نام ستون قیمت
        volume_column: نام ستون حجم
        
    Returns:
        Dict شامل نتایج محاسبات میانگین متحرک
    """
    # Default moving average types
    if ma_types is None:
        ma_types = ['sma', 'ema']
    elif isinstance(ma_types, str):
        ma_types = [ma_types]
    
    # Validate inputs
    if not _validate_dataframe(df):
        return {}
    
    parameters = {
        'sma_period': sma_period,
        'ema_period': ema_period,
        'kama_period': kama_period,
        'wma_period': wma_period,
        'vwma_period': vwma_period
    }
    
    if not _validate_parameters(parameters):
        return {}
    
    # Check required columns
    required_columns = _get_required_columns(ma_types)
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return {}
    
    # Calculate moving averages
    results = {}
    
    try:
        if 'sma' in ma_types:
            logger.info(f"Calculating SMA with period {sma_period}")
            results['sma'] = calculate_sma(df, column=price_column, period=sma_period)
            
        if 'ema' in ma_types:
            logger.info(f"Calculating EMA with period {ema_period}")
            results['ema'] = calculate_ema(df, column=price_column, period=ema_period)
            
        if 'kama' in ma_types:
            logger.info(f"Calculating KAMA with period {kama_period}")
            results['kama'] = calculate_kama(
                df, period=kama_period, fast_sc=kama_fast_sc, slow_sc=kama_slow_sc
            )
            
        if 'wma' in ma_types:
            logger.info(f"Calculating WMA with period {wma_period}")
            results['wma'] = calculate_wma(df, column=price_column, period=wma_period)
            
        if 'vwma' in ma_types:
            logger.info(f"Calculating VWMA with period {vwma_period}")
            results['vwma'] = calculate_vwma(
                df, price_column=price_column, volume_column=volume_column, period=vwma_period
            )
    
    except Exception as e:
        logger.error(f"Error in moving averages calculation: {e}")
        return {}
    
    # Log successful calculations
    successful_calculations = [ma_type for ma_type, result in results.items() if result is not None]
    if successful_calculations:
        logger.info(f"Successfully calculated: {', '.join(successful_calculations)}")
    
    failed_calculations = [ma_type for ma_type, result in results.items() if result is None]
    if failed_calculations:
        logger.error(f"Failed calculations: {', '.join(failed_calculations)}")
    
    return results

# ==================== Utility Functions ====================

def get_all_moving_averages(
    df: pd.DataFrame,
    period: int = 20,
    price_column: str = 'close',
    volume_column: str = 'volume'
) -> Dict[str, Optional[pd.Series]]:
    """
    محاسبه تمام انواع میانگین متحرک با یک دوره زمانی یکسان
    
    Args:
        df: DataFrame شامل داده‌های قیمت
        period: دوره زمانی مشترک
        price_column: نام ستون قیمت
        volume_column: نام ستون حجم
        
    Returns:
        Dict شامل تمام انواع میانگین متحرک
    """
    return calculate_moving_averages(
        df=df,
        ma_types=['sma', 'ema', 'kama', 'wma', 'vwma'],
        sma_period=period,
        ema_period=period,
        kama_period=period,
        wma_period=period,
        vwma_period=period,
        price_column=price_column,
        volume_column=volume_column
    )

def get_short_long_ma_signals(
    df: pd.DataFrame,
    ma_type: str = 'sma',
    short_period: int = 10,
    long_period: int = 20,
    price_column: str = 'close'
) -> Dict[str, pd.Series]:
    """
    محاسبه سیگنال‌های خرید/فروش بر اساس تقاطع میانگین متحرک کوتاه و بلند مدت
    
    Args:
        df: DataFrame شامل داده‌های قیمت
        ma_type: نوع میانگین متحرک
        short_period: دوره کوتاه مدت
        long_period: دوره بلند مدت
        price_column: نام ستون قیمت
        
    Returns:
        Dict شامل میانگین‌های کوتاه و بلند مدت و سیگنال‌ها
    """
    if ma_type not in ['sma', 'ema', 'wma']:
        logger.error(f"Unsupported MA type for signals: {ma_type}")
        return {}
    
    # Calculate short and long moving averages
    short_ma = calculate_moving_averages(
        df, [ma_type], **{f'{ma_type}_period': short_period}, price_column=price_column
    )[ma_type]
    
    long_ma = calculate_moving_averages(
        df, [ma_type], **{f'{ma_type}_period': long_period}, price_column=price_column
    )[ma_type]
    
    if short_ma is None or long_ma is None:
        logger.error("Failed to calculate moving averages for signals")
        return {}
    
    # Generate signals
    signals = pd.Series(0, index=df.index)  # 0: hold, 1: buy, -1: sell
    signals[short_ma > long_ma] = 1  # Buy signal
    signals[short_ma < long_ma] = -1  # Sell signal
    
    return {
        f'{ma_type}_short': short_ma,
        f'{ma_type}_long': long_ma,
        'signals': signals
    }