import numpy as np
import pandas as pd
from .cache_utils import cached_calculation, NUMBA_AVAILABLE, jit
from logger_config import logger


@jit() if NUMBA_AVAILABLE else lambda x: x
def _calculate_fib_numba(high_values, low_values, lookback):
    """محاسبه سطوح فیبوناچی با Numba برای بهینه سازی سرعت"""
    n = len(high_values)
    fib_levels = np.zeros((n, 7))  # 7 سطح فیبوناچی
    
    fib_ratios = np.array([0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0])
    
    for i in range(lookback, n):
        start_idx = max(0, i - lookback)
        period_high = np.max(high_values[start_idx:i+1])
        period_low = np.min(low_values[start_idx:i+1])
        
        diff = period_high - period_low
        
        for j in range(7):
            fib_levels[i, j] = period_high - (diff * fib_ratios[j])
    
    return fib_levels

@cached_calculation('fibonacci_levels')
def calculate_fibonacci_levels(df, lookback=50, price_columns=None):
    """
    محاسبه سطوح فیبوناچی با استفاده از کشینگ پیشرفته
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم حاوی داده های قیمت
    lookback : int
        تعداد کندل های قبلی برای محاسبه بالاترین و پایین ترین قیمت
    price_columns : dict
        نام ستون های قیمت {'high': 'high', 'low': 'low'}
    
    Returns:
    --------
    pd.DataFrame
        دیتافریم حاوی سطوح فیبوناچی
    """
    try:
        if df is None or len(df) == 0:
            logger.warning("DataFrame is empty or None")
            return None
        
        # تنظیم نام ستون های پیش فرض
        if price_columns is None:
            price_columns = {'high': 'high', 'low': 'low'}
        
        # بررسی وجود ستون های مورد نیاز
        required_columns = [price_columns['high'], price_columns['low']]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None
        
        # استخراج داده های قیمت
        high_col = price_columns['high']
        low_col = price_columns['low']
        
        high_values = df[high_col].values
        low_values = df[low_col].values
        
        # محاسبه سطوح فیبوناچی
        if NUMBA_AVAILABLE:
            fib_levels = _calculate_fib_numba(high_values, low_values, lookback)
        else:
            fib_levels = _calculate_fib_standard(high_values, low_values, lookback)
        
        # ایجاد DataFrame نتیجه
        fib_columns = ['fib_0', 'fib_23.6', 'fib_38.2', 'fib_50', 'fib_61.8', 'fib_78.6', 'fib_100']
        result_df = pd.DataFrame(fib_levels, columns=fib_columns, index=df.index)
        
        # افزودن اطلاعات اضافی
        result_df['fib_range'] = result_df['fib_0'] - result_df['fib_100']
        result_df['fib_mid'] = (result_df['fib_0'] + result_df['fib_100']) / 2
        
        logger.debug(f"Fibonacci levels calculated successfully for {len(df)} rows with lookback {lookback}")
        return result_df
        
    except Exception as e:
        logger.error(f"Error calculating Fibonacci levels: {e}")
        return None

def _calculate_fib_standard(high_values, low_values, lookback):
    """محاسبه سطوح فیبوناچی بدون Numba"""
    n = len(high_values)
    fib_levels = np.zeros((n, 7))
    
    fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    for i in range(lookback, n):
        start_idx = max(0, i - lookback)
        period_high = np.max(high_values[start_idx:i+1])
        period_low = np.min(low_values[start_idx:i+1])
        
        diff = period_high - period_low
        
        for j, ratio in enumerate(fib_ratios):
            fib_levels[i, j] = period_high - (diff * ratio)
    
    return fib_levels

@cached_calculation('fibonacci_retracements')
def calculate_fibonacci_retracements(df, swing_high_idx, swing_low_idx, price_columns=None):
    """
    محاسبه سطوح بازگشت فیبوناچی بین دو نقطه مشخص
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم حاوی داده های قیمت
    swing_high_idx : int
        ایندکس نقطه بالا
    swing_low_idx : int
        ایندکس نقطه پایین
    price_columns : dict
        نام ستون های قیمت
    
    Returns:
    --------
    dict
        دیکشنری حاوی سطوح بازگشت فیبوناچی
    """
    try:
        if price_columns is None:
            price_columns = {'high': 'high', 'low': 'low'}
        
        high_price = df.iloc[swing_high_idx][price_columns['high']]
        low_price = df.iloc[swing_low_idx][price_columns['low']]
        
        diff = high_price - low_price
        
        retracement_levels = {
            'swing_high': high_price,
            'swing_low': low_price,
            'fib_23.6': low_price + (diff * 0.236),
            'fib_38.2': low_price + (diff * 0.382),
            'fib_50.0': low_price + (diff * 0.5),
            'fib_61.8': low_price + (diff * 0.618),
            'fib_78.6': low_price + (diff * 0.786),
            'range': diff
        }
        
        return retracement_levels
        
    except Exception as e:
        logger.error(f"Error calculating Fibonacci retracements: {e}")
        return None

@cached_calculation('fibonacci_extensions')
def calculate_fibonacci_extensions(df, swing_high_idx, swing_low_idx, current_idx, price_columns=None):
    """
    محاسبه سطوح تمدید فیبوناچی
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم حاوی داده های قیمت
    swing_high_idx : int
        ایندکس نقطه بالا
    swing_low_idx : int
        ایندکس نقطه پایین
    current_idx : int
        ایندکس نقطه فعلی
    price_columns : dict
        نام ستون های قیمت
    
    Returns:
    --------
    dict
        دیکشنری حاوی سطوح تمدید فیبوناچی
    """
    try:
        if price_columns is None:
            price_columns = {'close': 'close'}
        
        high_price = df.iloc[swing_high_idx]['high'] if 'high' in df.columns else df.iloc[swing_high_idx][price_columns['close']]
        low_price = df.iloc[swing_low_idx]['low'] if 'low' in df.columns else df.iloc[swing_low_idx][price_columns['close']]
        current_price = df.iloc[current_idx][price_columns['close']]
        
        diff = high_price - low_price
        
        extension_levels = {
            'current_price': current_price,
            'ext_0': current_price,
            'ext_23.6': current_price + (diff * 0.236),
            'ext_38.2': current_price + (diff * 0.382),
            'ext_61.8': current_price + (diff * 0.618),
            'ext_100': current_price + diff,
            'ext_127.2': current_price + (diff * 1.272),
            'ext_161.8': current_price + (diff * 1.618),
            'ext_261.8': current_price + (diff * 2.618)
        }
        
        return extension_levels
        
    except Exception as e:
        logger.error(f"Error calculating Fibonacci extensions: {e}")
        return None


def _calculate_fibonacci_levels(df, lookback=50):
    """تابع سازگاری - استفاده از calculate_fibonacci_levels توصیه می شود"""
    return calculate_fibonacci_levels(df, lookback)