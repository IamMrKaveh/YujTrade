import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, Tuple, Any
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


@cached_calculation('fibonacci_levels')
def calculate_fibonacci_levels(df: pd.DataFrame, lookback: int = 50, price_columns: Optional[Dict[str, str]] = None) -> Optional[pd.DataFrame]:
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
            logger.error("DataFrame is empty or None")
            return None
        
        if price_columns is None:
            price_columns = {'high': 'high', 'low': 'low'}
        
        required_columns = [price_columns['high'], price_columns['low']]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None
        
        high_col = price_columns['high']
        low_col = price_columns['low']
        
        high_values = df[high_col].values
        low_values = df[low_col].values
        
        if NUMBA_AVAILABLE:
            fib_levels = _calculate_fib_numba(high_values, low_values, lookback)
        else:
            fib_levels = _calculate_fib_standard(high_values, low_values, lookback)
        
        fib_columns = ['fib_0', 'fib_23.6', 'fib_38.2', 'fib_50', 'fib_61.8', 'fib_78.6', 'fib_100']
        result_df = pd.DataFrame(fib_levels, columns=fib_columns, index=df.index)
        
        result_df['fib_range'] = result_df['fib_0'] - result_df['fib_100']
        result_df['fib_mid'] = (result_df['fib_0'] + result_df['fib_100']) / 2
        
        logger.info(f"Fibonacci levels calculated successfully for {len(df)} rows with lookback {lookback}")
        return result_df
        
    except Exception as e:
        logger.error(f"Error calculating Fibonacci levels: {e}")
        return None


@cached_calculation('fibonacci_retracements')
def calculate_fibonacci_retracements(df: pd.DataFrame, swing_high_idx: int, swing_low_idx: int, 
                                   price_columns: Optional[Dict[str, str]] = None) -> Optional[Dict[str, float]]:
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
def calculate_fibonacci_extensions(df: pd.DataFrame, swing_high_idx: int, swing_low_idx: int, 
                                 current_idx: int, price_columns: Optional[Dict[str, str]] = None) -> Optional[Dict[str, float]]:
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


def _find_swing_points(df: pd.DataFrame, lookback: int = 5, price_columns: Optional[Dict[str, str]] = None) -> Tuple[list, list]:
    """
    یافتن نقاط بالا و پایین (Swing Points) در داده های قیمت
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم حاوی داده های قیمت
    lookback : int
        تعداد کندل های قبل و بعد برای تایید swing point
    price_columns : dict
        نام ستون های قیمت
    
    Returns:
    --------
    tuple
        (swing_highs, swing_lows) - لیست ایندکس های نقاط بالا و پایین
    """
    if price_columns is None:
        price_columns = {'high': 'high', 'low': 'low'}
    
    swing_highs = []
    swing_lows = []
    
    high_values = df[price_columns['high']].values
    low_values = df[price_columns['low']].values
    
    for i in range(lookback, len(df) - lookback):
        # بررسی swing high
        is_swing_high = True
        for j in range(i - lookback, i + lookback + 1):
            if j != i and high_values[j] >= high_values[i]:
                is_swing_high = False
                break
        
        if is_swing_high:
            swing_highs.append(i)
        
        # بررسی swing low
        is_swing_low = True
        for j in range(i - lookback, i + lookback + 1):
            if j != i and low_values[j] <= low_values[i]:
                is_swing_low = False
                break
        
        if is_swing_low:
            swing_lows.append(i)
    
    return swing_highs, swing_lows


def _validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    اعتبارسنجی DataFrame و ستون های مورد نیاز
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم برای بررسی
    required_columns : list
        لیست ستون های مورد نیاز
    
    Returns:
    --------
    bool
        True اگر DataFrame معتبر باشد
    """
    if df is None or len(df) == 0:
        logger.error("DataFrame is empty or None")
        return False
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    return True


def fibonacci_complete_analysis(df: pd.DataFrame, 
                              lookback: int = 50,
                              swing_lookback: int = 5,
                              price_columns: Optional[Dict[str, str]] = None,
                              auto_detect_swings: bool = True,
                              swing_high_idx: Optional[int] = None,
                              swing_low_idx: Optional[int] = None,
                              current_idx: Optional[int] = None) -> Dict[str, Any]:
    """
    تابع اصلی برای تحلیل کامل فیبوناچی
    
    این تابع تمام انواع محاسبات فیبوناچی را انجام می‌دهد:
    - سطوح فیبوناچی متحرک
    - سطوح بازگشت فیبوناچی
    - سطوح تمدید فیبوناچی
    - تشخیص خودکار نقاط swing
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم حاوی داده های قیمت (باید شامل ستون های high, low, close باشد)
    lookback : int, default=50
        تعداد کندل های قبلی برای محاسبه سطوح فیبوناچی متحرک
    swing_lookback : int, default=5
        تعداد کندل های قبل و بعد برای تشخیص swing points
    price_columns : dict, optional
        نام ستون های قیمت {'high': 'high', 'low': 'low', 'close': 'close'}
    auto_detect_swings : bool, default=True
        آیا swing points به صورت خودکار تشخیص داده شوند
    swing_high_idx : int, optional
        ایندکس دستی نقطه بالا (اگر auto_detect_swings=False)
    swing_low_idx : int, optional
        ایندکس دستی نقطه پایین (اگر auto_detect_swings=False)
    current_idx : int, optional
        ایندکس نقطه فعلی برای محاسبه extensions (پیش‌فرض: آخرین ردیف)
    
    Returns:
    --------
    dict
        دیکشنری شامل تمام نتایج تحلیل فیبوناچی:
        {
            'fibonacci_levels': pd.DataFrame,      # سطوح فیبوناچی متحرک
            'swing_points': dict,                  # نقاط swing تشخیص داده شده
            'retracements': list,                  # سطوح بازگشت برای هر جفت swing
            'extensions': list,                    # سطوح تمدید برای هر جفت swing
            'summary': dict,                       # خلاصه تحلیل
            'status': str                          # وضعیت انجام عملیات
        }
    
    Examples:
    ---------
    >>> # استفاده ساده با تشخیص خودکار swing points
    >>> result = fibonacci_complete_analysis(df)
    >>> 
    >>> # استفاده با پارامترهای سفارشی
    >>> result = fibonacci_complete_analysis(
    ...     df, 
    ...     lookback=30, 
    ...     swing_lookback=7,
    ...     price_columns={'high': 'High', 'low': 'Low', 'close': 'Close'}
    ... )
    >>> 
    >>> # استفاده با swing points دستی
    >>> result = fibonacci_complete_analysis(
    ...     df,
    ...     auto_detect_swings=False,
    ...     swing_high_idx=100,
    ...     swing_low_idx=50
    ... )
    """
    
    try:
        logger.info("Starting complete Fibonacci analysis")
        
        # تنظیم پیش‌فرض ستون های قیمت
        if price_columns is None:
            price_columns = {'high': 'high', 'low': 'low', 'close': 'close'}
        
        # اعتبارسنجی DataFrame
        required_columns = list(price_columns.values())
        if not _validate_dataframe(df, required_columns):
            return {'status': 'error', 'message': 'DataFrame validation failed'}
        
        # ایجاد دیکشنری نتایج
        results = {
            'fibonacci_levels': None,
            'swing_points': {'highs': [], 'lows': []},
            'retracements': [],
            'extensions': [],
            'summary': {},
            'status': 'success'
        }
        
        # 1. محاسبه سطوح فیبوناچی متحرک
        logger.info("Calculating moving Fibonacci levels")
        results['fibonacci_levels'] = calculate_fibonacci_levels(df, lookback, price_columns)
        
        # 2. تشخیص یا استفاده از swing points
        if auto_detect_swings:
            logger.info("Auto-detecting swing points")
            swing_highs, swing_lows = _find_swing_points(df, swing_lookback, price_columns)
            results['swing_points']['highs'] = swing_highs
            results['swing_points']['lows'] = swing_lows
        else:
            if swing_high_idx is not None and swing_low_idx is not None:
                results['swing_points']['highs'] = [swing_high_idx]
                results['swing_points']['lows'] = [swing_low_idx]
                swing_highs = [swing_high_idx]
                swing_lows = [swing_low_idx]
            else:
                logger.error("Manual swing points not provided, skipping retracement/extension calculations")
                swing_highs, swing_lows = [], []
        
        # 3. محاسبه سطوح بازگشت برای هر جفت swing point
        if swing_highs and swing_lows:
            logger.info("Calculating Fibonacci retracements")
            for high_idx in swing_highs:
                for low_idx in swing_lows:
                    if abs(high_idx - low_idx) > swing_lookback:  # فقط swing points معنادار
                        retracement = calculate_fibonacci_retracements(df, high_idx, low_idx, price_columns)
                        if retracement:
                            retracement['high_idx'] = high_idx
                            retracement['low_idx'] = low_idx
                            results['retracements'].append(retracement)
        
        # 4. محاسبه سطوح تمدید
        if swing_highs and swing_lows:
            logger.info("Calculating Fibonacci extensions")
            current_idx = current_idx or len(df) - 1
            
            for high_idx in swing_highs:
                for low_idx in swing_lows:
                    if abs(high_idx - low_idx) > swing_lookback and current_idx > max(high_idx, low_idx):
                        extension = calculate_fibonacci_extensions(df, high_idx, low_idx, current_idx, price_columns)
                        if extension:
                            extension['high_idx'] = high_idx
                            extension['low_idx'] = low_idx
                            extension['current_idx'] = current_idx
                            results['extensions'].append(extension)
        
        # 5. ایجاد خلاصه تحلیل
        results['summary'] = {
            'total_rows': len(df),
            'lookback_period': lookback,
            'swing_lookback': swing_lookback,
            'swing_highs_count': len(results['swing_points']['highs']),
            'swing_lows_count': len(results['swing_points']['lows']),
            'retracements_count': len(results['retracements']),
            'extensions_count': len(results['extensions']),
            'fibonacci_levels_available': results['fibonacci_levels'] is not None,
            'analysis_method': 'auto_detect' if auto_detect_swings else 'manual_swings'
        }
        
        logger.info(f"Fibonacci analysis completed successfully. Found {len(results['swing_points']['highs'])} swing highs and {len(results['swing_points']['lows'])} swing lows")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in complete Fibonacci analysis: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'fibonacci_levels': None,
            'swing_points': {'highs': [], 'lows': []},
            'retracements': [],
            'extensions': [],
            'summary': {}
        }


# تابع سازگاری برای حفظ کد قدیمی
def _calculate_fibonacci_levels(df: pd.DataFrame, lookback: int = 50) -> Optional[pd.DataFrame]:
    """تابع سازگاری - استفاده از calculate_fibonacci_levels توصیه می شود"""
    return calculate_fibonacci_levels(df, lookback)