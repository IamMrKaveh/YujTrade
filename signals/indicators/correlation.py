import pandas as pd
import numpy as np
from logger_config import logger
from .cache_utils import cached_calculation, NUMBA_AVAILABLE

if NUMBA_AVAILABLE:
    from numba import jit
else:
    def jit():
        def decorator(func):
            return func
        return decorator


@jit(nopython=True, cache=True)
def _fast_rolling_correlation(x, y, window):
    """محاسبه سریع همبستگی غلتان با numba"""
    n = len(x)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        start_idx = i - window + 1
        x_window = x[start_idx:i + 1]
        y_window = y[start_idx:i + 1]
        
        x_mean = np.mean(x_window)
        y_mean = np.mean(y_window)
        
        numerator = np.sum((x_window - x_mean) * (y_window - y_mean))
        x_std = np.sqrt(np.sum((x_window - x_mean) ** 2))
        y_std = np.sqrt(np.sum((y_window - y_mean) ** 2))
        
        if x_std > 0 and y_std > 0:
            result[i] = numerator / (x_std * y_std)
    
    return result

def _create_btc_cache_key(df_sig, btc_sig, period):
    """ایجاد کلید کش برای همبستگی با بیت کوین"""
    return f"btc_corr_{df_sig}_{btc_sig}_{period}"

@cached_calculation("correlation_with_btc")
def _calculate_correlation_with_btc(df, btc_df, period=20, method='pearson', min_periods=None):
    """محاسبه همبستگی با بیت کوین با کشینگ بهینه"""
    try:
        if df is None or btc_df is None or len(df) < period or len(btc_df) < period:
            return None
        
        if min_periods is None:
            min_periods = period // 2
            
        # هم‌تراز کردن داده‌ها بر اساس زمان
        merged = pd.merge(df[['close']], btc_df[['close']], 
                        left_index=True, right_index=True, 
                        suffixes=('', '_btc'), how='inner')
        
        if len(merged) < period:
            return None
            
        # انتخاب روش محاسبه بر اساس در دسترس بودن numba
        if NUMBA_AVAILABLE and method == 'pearson':
            correlation_values = _fast_rolling_correlation(
                merged['close'].values, 
                merged['close_btc'].values, 
                period
            )
            correlation = pd.Series(correlation_values, index=merged.index)
        else:
            # محاسبه همبستگی غلتان با pandas
            correlation = merged['close'].rolling(
                window=period, 
                min_periods=min_periods
            ).corr(merged['close_btc'])
        
        return correlation
        
    except Exception as e:
        logger.error(f"Error calculating BTC correlation: {e}")
        return None

@cached_calculation("rolling_correlation")
def calculate_rolling_correlation(df1, df2, column1='close', column2='close', period=20, min_periods=None):
    """محاسبه همبستگی غلتان بین دو دیتافریم"""
    try:
        if df1 is None or df2 is None or len(df1) < period or len(df2) < period:
            return None
        
        if min_periods is None:
            min_periods = period // 2
            
        # هم‌تراز کردن داده‌ها
        merged = pd.merge(df1[[column1]], df2[[column2]], 
                        left_index=True, right_index=True, 
                        suffixes=('_1', '_2'), how='inner')
        
        if len(merged) < period:
            return None
        
        col1_name = f"{column1}_1" if column1 in df2.columns else column1
        col2_name = f"{column2}_2" if column2 in df1.columns else column2
        
        if NUMBA_AVAILABLE:
            correlation_values = _fast_rolling_correlation(
                merged[col1_name].values, 
                merged[col2_name].values, 
                period
            )
            correlation = pd.Series(correlation_values, index=merged.index)
        else:
            correlation = merged[col1_name].rolling(
                window=period, 
                min_periods=min_periods
            ).corr(merged[col2_name])
        
        return correlation
        
    except Exception as e:
        logger.error(f"Error calculating rolling correlation: {e}")
        return None

@cached_calculation("correlation_matrix")
def calculate_correlation_matrix(dataframes_dict, period=20, method='pearson'):
    """محاسبه ماتریس همبستگی برای چندین دارایی"""
    try:
        if not dataframes_dict or len(dataframes_dict) < 2:
            return None
        
        # استخراج داده‌های قیمت پایانی
        price_data = {}
        for name, df in dataframes_dict.items():
            if df is not None and 'close' in df.columns and len(df) >= period:
                price_data[name] = df['close']
        
        if len(price_data) < 2:
            return None
        
        # ایجاد دیتافریم ترکیبی
        combined_df = pd.DataFrame(price_data)
        combined_df = combined_df.dropna()
        
        if len(combined_df) < period:
            return None
        
        # محاسبه ماتریس همبستگی غلتان
        correlation_matrices = []
        
        for i in range(period - 1, len(combined_df)):
            window_data = combined_df.iloc[i - period + 1:i + 1]
            corr_matrix = window_data.corr(method=method)
            correlation_matrices.append({
                'timestamp': combined_df.index[i],
                'correlation_matrix': corr_matrix
            })
        
        return correlation_matrices
        
    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {e}")
        return None

@cached_calculation("correlation_strength")
def calculate_correlation_strength(correlation_series, threshold=0.7):
    """محاسبه قدرت همبستگی و آمار مربوطه"""
    try:
        if correlation_series is None or len(correlation_series) == 0:
            return None
        
        # حذف مقادیر NaN
        clean_corr = correlation_series.dropna()
        
        if len(clean_corr) == 0:
            return None
        
        stats = {
            'mean_correlation': clean_corr.mean(),
            'std_correlation': clean_corr.std(),
            'max_correlation': clean_corr.max(),
            'min_correlation': clean_corr.min(),
            'current_correlation': clean_corr.iloc[-1] if len(clean_corr) > 0 else None,
            'strong_positive_ratio': (clean_corr > threshold).sum() / len(clean_corr),
            'strong_negative_ratio': (clean_corr < -threshold).sum() / len(clean_corr),
            'weak_correlation_ratio': (abs(clean_corr) < 0.3).sum() / len(clean_corr),
            'correlation_trend': 'increasing' if len(clean_corr) > 10 and clean_corr.tail(5).mean() > clean_corr.head(5).mean() else 'decreasing'
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating correlation strength: {e}")
        return None

def get_correlation_signals(correlation_series, entry_threshold=0.8, exit_threshold=0.3):
    """تولید سیگنال‌های معاملاتی بر اساس همبستگی"""
    try:
        if correlation_series is None or len(correlation_series) == 0:
            return None
        
        signals = pd.Series(0, index=correlation_series.index)
        clean_corr = correlation_series.fillna(0)
        
        # سیگنال خرید: همبستگی قوی مثبت
        signals[clean_corr > entry_threshold] = 1
        
        # سیگنال فروش: همبستگی قوی منفی
        signals[clean_corr < -entry_threshold] = -1
        
        # خروج: همبستگی ضعیف
        signals[abs(clean_corr) < exit_threshold] = 0
        
        return signals
        
    except Exception as e:
        logger.error(f"Error generating correlation signals: {e}")
        return None
