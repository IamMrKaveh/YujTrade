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


def comprehensive_correlation_analysis(
    primary_df,
    comparison_data,
    btc_df=None,
    period=20,
    method='pearson',
    correlation_threshold=0.7,
    signal_entry_threshold=0.8,
    signal_exit_threshold=0.3,
    min_periods=None,
    include_matrix=True,
    include_signals=True,
    include_strength_analysis=True
):
    """
    تابع جامع تحلیل همبستگی که از تمام توابع موجود استفاده می‌کند
    
    Parameters:
    -----------
    primary_df : pd.DataFrame
        دیتافریم اصلی (دارایی اول)
    comparison_data : dict or pd.DataFrame
        دارایی‌های مقایسه - می‌تواند دیتافریم یا دیکشنری از دیتافریم‌ها باشد
    btc_df : pd.DataFrame, optional
        دیتافریم بیت کوین برای تحلیل همبستگی با بازار
    period : int, default=20
        پریود محاسبه همبستگی غلتان
    method : str, default='pearson'
        روش محاسبه همبستگی
    correlation_threshold : float, default=0.7
        آستانه تعیین همبستگی قوی
    signal_entry_threshold : float, default=0.8
        آستانه ورود برای سیگنال‌های معاملاتی
    signal_exit_threshold : float, default=0.3
        آستانه خروج برای سیگنال‌های معاملاتی
    min_periods : int, optional
        حداقل دوره‌های مورد نیاز برای محاسبه
    include_matrix : bool, default=True
        شامل کردن ماتریس همبستگی چندگانه
    include_signals : bool, default=True
        شامل کردن سیگنال‌های معاملاتی
    include_strength_analysis : bool, default=True
        شامل کردن تحلیل قدرت همبستگی
    
    Returns:
    --------
    dict : نتایج کامل تحلیل همبستگی
    """
    
    try:
        logger.info(f"Starting comprehensive correlation analysis with period={period}")
        
        results = {
            'timestamp': pd.Timestamp.now(),
            'period': period,
            'method': method,
            'correlations': {},
            'btc_correlation': None,
            'correlation_matrix': None,
            'strength_analysis': {},
            'signals': {},
            'summary': {}
        }
        
        # بررسی صحت داده‌های ورودی
        if primary_df is None or len(primary_df) < period:
            logger.error("Primary dataframe is insufficient for analysis")
            return results
        
        if min_periods is None:
            min_periods = max(period // 2, 5)
        
        # 1. محاسبه همبستگی با بیت کوین (در صورت وجود)
        if btc_df is not None:
            logger.info("Calculating BTC correlation")
            btc_correlation = _calculate_correlation_with_btc(
                primary_df, btc_df, period, method, min_periods
            )
            results['btc_correlation'] = btc_correlation
            
            if btc_correlation is not None and include_strength_analysis:
                results['strength_analysis']['btc'] = calculate_correlation_strength(
                    btc_correlation, correlation_threshold
                )
            
            if btc_correlation is not None and include_signals:
                results['signals']['btc'] = get_correlation_signals(
                    btc_correlation, signal_entry_threshold, signal_exit_threshold
                )
        
        # 2. محاسبه همبستگی با سایر دارایی‌ها
        if isinstance(comparison_data, pd.DataFrame):
            # تبدیل دیتافریم منفرد به دیکشنری
            comparison_data = {'comparison': comparison_data}
        
        if isinstance(comparison_data, dict):
            logger.info(f"Calculating correlations with {len(comparison_data)} assets")
            
            for asset_name, asset_df in comparison_data.items():
                if asset_df is None or len(asset_df) < period:
                    logger.error(f"Insufficient data for asset: {asset_name}")
                    continue
                
                # محاسبه همبستگی غلتان
                correlation = calculate_rolling_correlation(
                    primary_df, asset_df, 'close', 'close', period, min_periods
                )
                
                if correlation is not None:
                    results['correlations'][asset_name] = correlation
                    
                    # تحلیل قدرت همبستگی
                    if include_strength_analysis:
                        strength = calculate_correlation_strength(
                            correlation, correlation_threshold
                        )
                        results['strength_analysis'][asset_name] = strength
                    
                    # تولید سیگنال‌های معاملاتی
                    if include_signals:
                        signals = get_correlation_signals(
                            correlation, signal_entry_threshold, signal_exit_threshold
                        )
                        results['signals'][asset_name] = signals
            
            # 3. محاسبه ماتریس همبستگی چندگانه
            if include_matrix and len(comparison_data) > 1:
                logger.info("Calculating correlation matrix")
                
                # اضافه کردن دیتافریم اصلی به مجموعه
                all_data = {'primary': primary_df}
                all_data.update(comparison_data)
                
                if btc_df is not None:
                    all_data['btc'] = btc_df
                
                correlation_matrix = calculate_correlation_matrix(
                    all_data, period, method
                )
                results['correlation_matrix'] = correlation_matrix
        
        # 4. ایجاد خلاصه تحلیل
        results['summary'] = _create_analysis_summary(results, correlation_threshold)
        
        logger.info("Comprehensive correlation analysis completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error in comprehensive correlation analysis: {e}")
        return {
            'error': str(e),
            'timestamp': pd.Timestamp.now(),
            'period': period,
            'method': method
        }


def _create_analysis_summary(results, threshold=0.7):
    """ایجاد خلاصه تحلیل همبستگی"""
    try:
        summary = {
            'total_assets_analyzed': len(results.get('correlations', {})),
            'strong_correlations': 0,
            'weak_correlations': 0,
            'negative_correlations': 0,
            'current_correlations': {},
            'trend_analysis': {},
            'trading_signals_count': {}
        }
        
        # تحلیل همبستگی‌های فردی
        for asset_name, strength_data in results.get('strength_analysis', {}).items():
            if strength_data is None:
                continue
            
            current_corr = strength_data.get('current_correlation')
            if current_corr is not None:
                summary['current_correlations'][asset_name] = current_corr
                
                if abs(current_corr) > threshold:
                    summary['strong_correlations'] += 1
                elif abs(current_corr) < 0.3:
                    summary['weak_correlations'] += 1
                
                if current_corr < 0:
                    summary['negative_correlations'] += 1
                
                # ترند همبستگی
                trend = strength_data.get('correlation_trend')
                if trend:
                    summary['trend_analysis'][asset_name] = trend
        
        # تحلیل سیگنال‌های معاملاتی
        for asset_name, signals in results.get('signals', {}).items():
            if signals is not None:
                signal_counts = {
                    'buy_signals': (signals == 1).sum(),
                    'sell_signals': (signals == -1).sum(),
                    'neutral_signals': (signals == 0).sum()
                }
                summary['trading_signals_count'][asset_name] = signal_counts
        
        # اضافه کردن تحلیل بیت کوین
        if results.get('btc_correlation') is not None:
            btc_strength = results.get('strength_analysis', {}).get('btc')
            if btc_strength:
                summary['btc_correlation_strength'] = btc_strength.get('current_correlation')
                summary['btc_trend'] = btc_strength.get('correlation_trend')
        
        return summary
        
    except Exception as e:
        logger.error(f"Error creating analysis summary: {e}")
        return {'error': str(e)}


def quick_correlation_check(df1, df2, period=20, asset_name="Asset"):
    """بررسی سریع همبستگی بین دو دارایی"""
    try:
        correlation = calculate_rolling_correlation(df1, df2, period=period)
        
        if correlation is None:
            return None
        
        strength = calculate_correlation_strength(correlation)
        signals = get_correlation_signals(correlation)
        
        return {
            'asset_name': asset_name,
            'correlation_series': correlation,
            'strength_analysis': strength,
            'signals': signals,
            'current_correlation': correlation.iloc[-1] if len(correlation) > 0 else None
        }
        
    except Exception as e:
        logger.error(f"Error in quick correlation check: {e}")
        return None


def batch_correlation_analysis(primary_df, assets_dict, btc_df=None, period=20):
    """تحلیل همبستگی دسته‌ای برای چندین دارایی"""
    try:
        results = {}
        
        for asset_name, asset_df in assets_dict.items():
            logger.info(f"Analyzing correlation for {asset_name}")
            
            result = quick_correlation_check(
                primary_df, asset_df, period, asset_name
            )
            
            if result is not None:
                results[asset_name] = result
        
        # اضافه کردن تحلیل بیت کوین
        if btc_df is not None:
            btc_result = quick_correlation_check(
                primary_df, btc_df, period, "Bitcoin"
            )
            if btc_result is not None:
                results['Bitcoin'] = btc_result
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch correlation analysis: {e}")
        return {}