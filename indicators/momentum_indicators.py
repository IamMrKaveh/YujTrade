import sys
from numba import jit
import numpy as np
import pandas as pd
from logger_config import logger
from .cache_utils import NUMBA_AVAILABLE, cached_calculation

# Fix numpy compatibility issue
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

# Now import pandas_ta after fixing numpy
try:
    import pandas_ta as ta
except ImportError as e:
    print(f"Error importing pandas_ta: {e}")
    print("Please install with: pip install pandas-ta==0.3.14b")
    sys.exit(1)

# =============================================================================
# CORE CALCULATION FUNCTIONS
# =============================================================================

@jit(nopython=True)
def _fast_rsi_calculation(prices, period):
    """محاسبه سریع RSI"""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    avg_gains = np.empty(len(gains))
    avg_losses = np.empty(len(losses))
    
    # محاسبه اولیه
    avg_gains[period-1] = np.mean(gains[:period])
    avg_losses[period-1] = np.mean(losses[:period])
    
    # محاسبه نمایی
    alpha = 1.0 / period
    for i in range(period, len(gains)):
        avg_gains[i] = alpha * gains[i] + (1 - alpha) * avg_gains[i-1]
        avg_losses[i] = alpha * losses[i] + (1 - alpha) * avg_losses[i-1]
    
    rs = avg_gains[period-1:] / avg_losses[period-1:]
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


@cached_calculation('rsi')
def _calculate_rsi(df, period):
    """محاسبه RSI"""
    try:
        if df is None or len(df) < period + 1:
            return None
        
        if NUMBA_AVAILABLE:
            prices = df['close'].values.astype(np.float64)
            rsi_values = _fast_rsi_calculation(prices, period)
            # Pad with NaN values for consistency
            result = np.full(len(df), np.nan)
            result[period:] = rsi_values
            return pd.Series(result, index=df.index)
        else:
            # Use pandas_ta RSI for accurate calculation
            rsi_result = ta.rsi(df['close'], length=period)
            if rsi_result is not None:
                return rsi_result
            else:
                # Final fallback with correct EMA calculation
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).ewm(span=period, adjust=False).mean()
                loss = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean()
                rs = gain / loss
                result = 100 - (100 / (1 + rs))
                return result
        
    except Exception as e:
        logger.error(f"Error in optimized RSI calculation: {e}")
        return None


@cached_calculation('money_flow_index')
def _calculate_money_flow_index(df, period=14):
    """محاسبه Money Flow Index"""
    try:
        if df is None or len(df) < period + 1:
            return None
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        price_diff = typical_price.diff()
        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)
        
        positive_flow[price_diff > 0] = money_flow[price_diff > 0]
        negative_flow[price_diff < 0] = money_flow[price_diff < 0]
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        money_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    except Exception as e:
        logger.error(f"Error calculating MFI: {e}")
        return None


@cached_calculation('commodity_channel_index')
def _calculate_commodity_channel_index(df, period=20):
    """محاسبه Commodity Channel Index"""
    try:
        if df is None or len(df) < period:
            return None
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    except Exception as e:
        logger.error(f"Error calculating CCI: {e}")
        return None


@cached_calculation('williams_r')
def _calculate_williams_r(df, period=14):
    """محاسبه Williams %R"""
    try:
        if df is None or len(df) < period:
            return None
        
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
        
        return williams_r
    except Exception as e:
        logger.error(f"Error calculating Williams %R: {e}")
        return None


@cached_calculation('ultimate_oscillator')
def _calculate_ultimate_oscillator(df, period1=7, period2=14, period3=28):
    """محاسبه Ultimate Oscillator"""
    try:
        if df is None or len(df) < max(period1, period2, period3):
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        # True Low = minimum of Low or previous Close
        true_low = pd.concat([low, prev_close], axis=1).min(axis=1)
        
        # Buying Pressure = Close - True Low
        buying_pressure = close - true_low
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Average calculations for 3 periods
        bp1 = buying_pressure.rolling(window=period1).sum()
        tr1_sum = true_range.rolling(window=period1).sum()
        
        bp2 = buying_pressure.rolling(window=period2).sum()
        tr2_sum = true_range.rolling(window=period2).sum()
        
        bp3 = buying_pressure.rolling(window=period3).sum()
        tr3_sum = true_range.rolling(window=period3).sum()
        
        # Ultimate Oscillator formula
        uo = 100 * (4 * (bp1 / tr1_sum) + 2 * (bp2 / tr2_sum) + (bp3 / tr3_sum)) / 7
        
        return uo
    except Exception as e:
        logger.error(f"Error calculating Ultimate Oscillator: {e}")
        return None


@cached_calculation('rate_of_change')
def _calculate_rate_of_change(df, period=14):
    """محاسبه Rate of Change (ROC)"""
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        roc = ((close - close.shift(period)) / close.shift(period)) * 100
        
        return roc
    except Exception as e:
        logger.error(f"Error calculating ROC: {e}")
        return None


@cached_calculation('awesome_oscillator')
def _calculate_awesome_oscillator(df, fast_period=5, slow_period=34):
    """محاسبه Awesome Oscillator"""
    try:
        if df is None or len(df) < slow_period:
            return None
        
        median_price = (df['high'] + df['low']) / 2
        
        fast = median_price.rolling(window=fast_period).mean()
        slow = median_price.rolling(window=slow_period).mean()

        ao = fast - slow
        ao = ao.dropna()  # Remove NaN values
        
        if ao.empty:
            logger.error("Awesome Oscillator calculation resulted in empty series")
            return None
        
        logger.info(f"Calculated Awesome Oscillator with {len(ao)} values")
        return ao
    except Exception as e:
        logger.error(f"Error calculating Awesome Oscillator: {e}")
        return None


@cached_calculation('trix')
def _calculate_trix(df, period=14):
    """محاسبه TRIX"""
    try:
        if df is None or len(df) < period * 3:
            return None
        
        close = df['close']
        
        # Triple smoothed EMA
        ema1 = close.ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        ema3 = ema2.ewm(span=period).mean()
        
        # TRIX = Rate of change of triple smoothed EMA
        trix = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 10000
        
        return trix
    except Exception as e:
        logger.error(f"Error calculating TRIX: {e}")
        return None


@cached_calculation('dpo')
def _calculate_dpo(df, period=20):
    """محاسبه Detrended Price Oscillator"""
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        sma = close.rolling(window=period).mean()
        
        # DPO = Close - SMA shifted by (period/2 + 1)
        shift_period = int(period/2) + 1
        dpo = close - sma.shift(shift_period)
        
        dpo = dpo.dropna()  # Remove NaN values
        
        if dpo.empty:
            logger.error("DPO calculation resulted in empty series")
            return None
        
        logger.info(f"Calculated DPO with {len(dpo)} values")
        
        return dpo
    except Exception as e:
        logger.error(f"Error calculating DPO: {e}")
        return None


# =============================================================================
# MAIN TECHNICAL INDICATORS CALCULATOR
# =============================================================================

def calculate_technical_indicators(df, config=None):
    """
    تابع اصلی محاسبه تمام اندیکاتورهای تکنیکال
    
    Args:
        df (pandas.DataFrame): داده‌های قیمت با ستون‌های 'open', 'high', 'low', 'close', 'volume'
        config (dict, optional): پیکربندی پارامترها برای هر اندیکاتور
    
    Returns:
        dict: دیکشنری شامل تمام اندیکاتورهای محاسبه شده
    """
    
    if df is None or df.empty:
        logger.error("DataFrame is None or empty")
        return {}
    
    # تنظیم پیکربندی پیش‌فرض
    default_config = {
        'rsi_period': 14,
        'mfi_period': 14,
        'cci_period': 20,
        'williams_r_period': 14,
        'uo_periods': [7, 14, 28],
        'roc_period': 14,
        'ao_periods': [5, 34],
        'trix_period': 14,
        'dpo_period': 20
    }
    
    if config:
        default_config.update(config)
    
    indicators = {}
    
    try:
        # RSI (Relative Strength Index)
        logger.info("Calculating RSI...")
        indicators['rsi'] = _calculate_rsi(df, default_config['rsi_period'])
        
        # MFI (Money Flow Index)
        logger.info("Calculating MFI...")
        indicators['mfi'] = _calculate_money_flow_index(df, default_config['mfi_period'])
        
        # CCI (Commodity Channel Index)
        logger.info("Calculating CCI...")
        indicators['cci'] = _calculate_commodity_channel_index(df, default_config['cci_period'])
        
        # Williams %R
        logger.info("Calculating Williams %R...")
        indicators['williams_r'] = _calculate_williams_r(df, default_config['williams_r_period'])
        
        # Ultimate Oscillator
        logger.info("Calculating Ultimate Oscillator...")
        uo_periods = default_config['uo_periods']
        indicators['ultimate_oscillator'] = _calculate_ultimate_oscillator(
            df, uo_periods[0], uo_periods[1], uo_periods[2]
        )
        
        # ROC (Rate of Change)
        logger.info("Calculating ROC...")
        indicators['roc'] = _calculate_rate_of_change(df, default_config['roc_period'])
        
        # Awesome Oscillator
        logger.info("Calculating Awesome Oscillator...")
        ao_periods = default_config['ao_periods']
        indicators['awesome_oscillator'] = _calculate_awesome_oscillator(
            df, ao_periods[0], ao_periods[1]
        )
        
        # TRIX
        logger.info("Calculating TRIX...")
        indicators['trix'] = _calculate_trix(df, default_config['trix_period'])
        
        # DPO (Detrended Price Oscillator)
        logger.info("Calculating DPO...")
        indicators['dpo'] = _calculate_dpo(df, default_config['dpo_period'])
        
        # حذف اندیکاتورهای None
        indicators = {k: v for k, v in indicators.items() if v is not None}
        
        logger.info(f"Successfully calculated {len(indicators)} technical indicators")
        
    except Exception as e:
        logger.error(f"Error in calculate_technical_indicators: {e}")
        
    return indicators


def get_indicator_summary(indicators):
    """
    خلاصه‌ای از وضعیت اندیکاتورها
    
    Args:
        indicators (dict): دیکشنری اندیکاتورهای محاسبه شده
        
    Returns:
        dict: خلاصه وضعیت اندیکاتورها
    """
    
    summary = {
        'total_indicators': len(indicators),
        'successful_calculations': 0,
        'failed_calculations': 0,
        'indicator_status': {}
    }
    
    for name, values in indicators.items():
        if values is not None and not values.empty:
            summary['successful_calculations'] += 1
            summary['indicator_status'][name] = {
                'status': 'success',
                'data_points': len(values),
                'last_value': values.iloc[-1] if len(values) > 0 else None
            }
        else:
            summary['failed_calculations'] += 1
            summary['indicator_status'][name] = {
                'status': 'failed',
                'data_points': 0,
                'last_value': None
            }
    
    return summary


def export_indicators_to_dataframe(df, indicators):
    """
    ترکیب داده‌های اصلی با اندیکاتورها در یک DataFrame
    
    Args:
        df (pandas.DataFrame): داده‌های اصلی قیمت
        indicators (dict): دیکشنری اندیکاتورهای محاسبه شده
        
    Returns:
        pandas.DataFrame: DataFrame ترکیبی شامل قیمت‌ها و اندیکاتورها
    """
    
    result_df = df.copy()
    
    for name, values in indicators.items():
        if values is not None and not values.empty:
            # تنظیم اندازه سریز با DataFrame اصلی
            if len(values) != len(result_df):
                # ایجاد سری جدید با اندازه مناسب
                aligned_series = pd.Series(index=result_df.index, dtype=float)
                aligned_series.iloc[-len(values):] = values.values
                result_df[name] = aligned_series
            else:
                result_df[name] = values
    
    return result_df


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_dataframe(df):
    """
    اعتبارسنجی DataFrame ورودی
    
    Args:
        df (pandas.DataFrame): DataFrame برای بررسی
        
    Returns:
        tuple: (is_valid, error_message)
    """
    
    if df is None:
        return False, "DataFrame is None"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    if len(df) < 50:  # حداقل داده مورد نیاز
        return False, f"Insufficient data: {len(df)} rows (minimum 50 required)"
    
    return True, "DataFrame is valid"


def get_available_indicators():
    """
    لیست اندیکاتورهای موجود
    
    Returns:
        list: لیست نام اندیکاتورهای موجود
    """
    
    return [
        'rsi',
        'mfi', 
        'cci',
        'williams_r',
        'ultimate_oscillator',
        'roc',
        'awesome_oscillator',
        'trix',
        'dpo'
    ]


