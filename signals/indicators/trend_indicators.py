from logger_config import logger
from numba import jit
import numpy as np
import pandas as pd
from .moving_averages import _fast_ema
from .volatility_indicators import _calculate_average_true_range
from .cache_utils import cached_calculation


@jit(nopython=True)
def _fast_macd_calculation(prices, fast_period, slow_period, signal_period):
    """محاسبه سریع MACD"""
    fast_alpha = 2.0 / (fast_period + 1)
    slow_alpha = 2.0 / (slow_period + 1)
    signal_alpha = 2.0 / (signal_period + 1)
    
    fast_ema = _fast_ema(prices, fast_alpha)
    slow_ema = _fast_ema(prices, slow_alpha)
    
    macd_line = fast_ema - slow_ema
    signal_line = _fast_ema(macd_line, signal_alpha)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

@cached_calculation('trend_strength')
def _calculate_trend_strength(df, period=20):
    """محاسبه قدرت ترند"""
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        sma = close.rolling(window=period).mean()
        trend_strength = ((close - sma) / sma) * 100
        
        return trend_strength
    except Exception as e:
        logger.warning(f"Error calculating trend strength: {e}")
        return None

@cached_calculation('donchian_channels')
def _calculate_donchian_channels(df, period=20):
    """محاسبه Donchian Channels"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        
        upper_channel = high.rolling(window=period).max()
        lower_channel = low.rolling(window=period).min()
        middle_channel = (upper_channel + lower_channel) / 2
        
        return {
            'donchian_upper': upper_channel,
            'donchian_middle': middle_channel,
            'donchian_lower': lower_channel
        }
    except Exception as e:
        logger.warning(f"Error calculating Keltner Channels: {e}")
        return None

def _calculate_donchian_channels(df, period=20):
    """محاسبه Donchian Channels"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        
        upper_channel = high.rolling(window=period).max()
        lower_channel = low.rolling(window=period).min()
        middle_channel = (upper_channel + lower_channel) / 2
        
        return {
            'donchian_upper': upper_channel,
            'donchian_middle': middle_channel,
            'donchian_lower': lower_channel
        }
    except Exception as e:
        logger.warning(f"Error calculating Donchian Channels: {e}")
        return None

@cached_calculation('supertrend')
def _calculate_supertrend(df, period=10, multiplier=3.0):
    """محاسبه Supertrend with caching"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # محاسبه ATR
        atr = _calculate_average_true_range(df, period)
        if atr is None:
            return None
        
        # محاسبه Basic Upper و Lower Bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Initialize arrays
        final_upper_band = np.zeros(len(df))
        final_lower_band = np.zeros(len(df))
        supertrend = np.zeros(len(df))
        direction = np.zeros(len(df))
        
        # Set initial values
        final_upper_band[0] = upper_band.iloc[0]
        final_lower_band[0] = lower_band.iloc[0]
        direction[0] = 1
        supertrend[0] = final_lower_band[0]
        
        # Calculate Supertrend
        for i in range(1, len(df)):
            # Calculate final upper band
            if upper_band.iloc[i] < final_upper_band[i-1] or close.iloc[i-1] > final_upper_band[i-1]:
                final_upper_band[i] = upper_band.iloc[i]
            else:
                final_upper_band[i] = final_upper_band[i-1]
            
            # Calculate final lower band
            if lower_band.iloc[i] > final_lower_band[i-1] or close.iloc[i-1] < final_lower_band[i-1]:
                final_lower_band[i] = lower_band.iloc[i]
            else:
                final_lower_band[i] = final_lower_band[i-1]
            
            # Calculate direction
            if direction[i-1] == -1 and close.iloc[i] < final_lower_band[i]:
                direction[i] = -1
            elif direction[i-1] == 1 and close.iloc[i] > final_upper_band[i]:
                direction[i] = 1
            elif direction[i-1] == -1 and close.iloc[i] >= final_lower_band[i]:
                direction[i] = 1
            elif direction[i-1] == 1 and close.iloc[i] <= final_upper_band[i]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
            
            # Calculate Supertrend value
            if direction[i] == 1:
                supertrend[i] = final_lower_band[i]
            else:
                supertrend[i] = final_upper_band[i]
        
        return pd.Series(supertrend, index=df.index)
    
    except Exception as e:
        logger.warning(f"Error calculating Supertrend: {e}")
        return None
        
@cached_calculation('aroon_oscillator')
def _calculate_aroon_oscillator(df, period=14):
    try:
        if df is None or len(df) < period:
            return None
        
        if 'high' not in df.columns or 'low' not in df.columns:
            return None
        
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        
        aroon_up = np.full(len(df), np.nan)
        aroon_down = np.full(len(df), np.nan)
        
        for i in range(period - 1, len(df)):
            high_period = high.iloc[i-period+1:i+1].values
            low_period = low.iloc[i-period+1:i+1].values
            
            high_max_idx = np.argmax(high_period)
            low_min_idx = np.argmin(low_period)
            
            periods_since_high = period - 1 - high_max_idx
            periods_since_low = period - 1 - low_min_idx
            
            aroon_up[i] = ((period - periods_since_high) / period) * 100
            aroon_down[i] = ((period - periods_since_low) / period) * 100
        
        aroon_up_series = pd.Series(aroon_up, index=df.index)
        aroon_down_series = pd.Series(aroon_down, index=df.index)
        aroon_oscillator = aroon_up_series - aroon_down_series
        
        return {
            'aroon_up': aroon_up_series,
            'aroon_down': aroon_down_series,
            'aroon_oscillator': aroon_oscillator
        }
    except Exception:
        return None

@cached_calculation('aroon')
def _calculate_aroon(df, period=14):
    """محاسبه Aroon Oscillator"""
    try:
        if df is None or len(df) < period:
            return None
            
        high = df['high']
        low = df['low']
        
        # پیدا کردن موقعیت بالاترین و پایین‌ترین قیمت
        aroon_up = ((period - high.rolling(period).apply(lambda x: period - 1 - x.argmax())) / period) * 100
        aroon_down = ((period - low.rolling(period).apply(lambda x: period - 1 - x.argmin())) / period) * 100
        
        aroon_oscillator = aroon_up - aroon_down
        
        return {
            'aroon_up': aroon_up,
            'aroon_down': aroon_down,
            'aroon_oscillator': aroon_oscillator
        }
        
    except Exception as e:
        logger.warning(f"Error calculating Aroon: {e}")
        return None

@cached_calculation('adx')
def _calculate_adx_internal(df, period):
    """Internal ADX calculation function"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        
        # True Range
        atr = _calculate_average_true_range(df, period)
        if atr is None:
            return None
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = pd.Series(0, index=df.index)
        minus_dm = pd.Series(0, index=df.index)
        
        plus_dm[up_move > down_move] = up_move[up_move > down_move]
        plus_dm[plus_dm < 0] = 0
        
        minus_dm[down_move > up_move] = down_move[down_move > up_move]
        minus_dm[minus_dm < 0] = 0
        
        # Smoothed DM
        plus_dm_smooth = plus_dm.rolling(window=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period).mean()
        
        # DI calculations
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)
        
        # ADX calculation
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = dx.fillna(0)
        adx = dx.rolling(window=period).mean()
        
        return {
            'plus_di': plus_di,
            'minus_di': minus_di,
            'adx': adx
        }
    except Exception as e:
        logger.warning(f"Error calculating ADX: {e}")
        return None

def _calculate_aroon(df, period=14):
    """محاسبه Aroon Oscillator"""
    try:
        if df is None or len(df) < period:
            return None
            
        high = df['high']
        low = df['low']
        
        # پیدا کردن موقعیت بالاترین و پایین‌ترین قیمت
        aroon_up = ((period - high.rolling(period).apply(lambda x: period - 1 - x.argmax())) / period) * 100
        aroon_down = ((period - low.rolling(period).apply(lambda x: period - 1 - x.argmin())) / period) * 100
        
        aroon_oscillator = aroon_up - aroon_down
        
        return {
            'aroon_up': aroon_up,
            'aroon_down': aroon_down,
            'aroon_oscillator': aroon_oscillator
        }
        
    except Exception as e:
        logger.warning(f"Error calculating Aroon: {e}")
        return None

def _calculate_adx_internal(df, period):
    """Internal ADX calculation function"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        
        # True Range
        atr = _calculate_average_true_range(df, period)
        if atr is None:
            return None
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = pd.Series(0, index=df.index)
        minus_dm = pd.Series(0, index=df.index)
        
        plus_dm[up_move > down_move] = up_move[up_move > down_move]
        plus_dm[plus_dm < 0] = 0
        
        minus_dm[down_move > up_move] = down_move[down_move > up_move]
        minus_dm[minus_dm < 0] = 0
        
        # Smoothed DM
        plus_dm_smooth = plus_dm.rolling(window=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period).mean()
        
        # DI calculations
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)
        
        # ADX calculation
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = dx.fillna(0)
        adx = dx.rolling(window=period).mean()
        
        return {
            'plus_di': plus_di,
            'minus_di': minus_di,
            'adx': adx
        }
    except Exception as e:
        logger.warning(f"Error calculating ADX: {e}")
        return None

def _initialize_sar_arrays(df, af, high, low):
    """Initialize arrays for Parabolic SAR calculation"""
    sar = np.zeros(len(df))
    trend = np.zeros(len(df))
    af_val = np.zeros(len(df))
    ep = np.zeros(len(df))
    
    sar[0] = low[0]
    trend[0] = 1
    af_val[0] = af
    ep[0] = high[0]
    
    return sar, trend, af_val, ep

def _handle_uptrend_sar(i, sar, trend, af_val, ep, high, low, af, max_af):
    """Handle Parabolic SAR calculation for uptrend"""
    sar[i] = sar[i-1] + af_val[i-1] * (ep[i-1] - sar[i-1])
    
    if low[i] <= sar[i]:
        # Trend reversal to downtrend
        trend[i] = -1
        sar[i] = ep[i-1]
        ep[i] = low[i]
        af_val[i] = af
    else:
        # Continue uptrend
        trend[i] = 1
        if high[i] > ep[i-1]:
            ep[i] = high[i]
            af_val[i] = min(af_val[i-1] + af, max_af)
        else:
            ep[i] = ep[i-1]
            af_val[i] = af_val[i-1]

def _handle_downtrend_sar(i, sar, trend, af_val, ep, high, low, af, max_af):
    """Handle Parabolic SAR calculation for downtrend"""
    sar[i] = sar[i-1] - af_val[i-1] * (sar[i-1] - ep[i-1])
    
    if high[i] >= sar[i]:
        # Trend reversal to uptrend
        trend[i] = 1
        sar[i] = ep[i-1]
        ep[i] = high[i]
        af_val[i] = af
    else:
        # Continue downtrend
        trend[i] = -1
        if low[i] < ep[i-1]:
            ep[i] = low[i]
            af_val[i] = min(af_val[i-1] + af, max_af)
        else:
            ep[i] = ep[i-1]
            af_val[i] = af_val[i-1]

def _calculate_final_upper_band(upper_band, final_upper_band, close, i):
    """Calculate final upper band for Supertrend"""
    if upper_band.iloc[i] < final_upper_band[i-1] or close.iloc[i-1] > final_upper_band[i-1]:
        return upper_band.iloc[i]
    else:
        return final_upper_band[i-1]

def _calculate_final_lower_band(lower_band, final_lower_band, close, i):
    """Calculate final lower band for Supertrend"""
    if lower_band.iloc[i] > final_lower_band[i-1] or close.iloc[i-1] < final_lower_band[i-1]:
        return lower_band.iloc[i]
    else:
        return final_lower_band[i-1]

def _calculate_direction(direction, close, final_upper_band, final_lower_band, i):
    """Calculate direction for Supertrend"""
    prev_direction = direction[i-1]
    current_close = close.iloc[i]
    
    if prev_direction == -1 and current_close < final_lower_band[i]:
        return -1
    elif prev_direction == 1 and current_close > final_upper_band[i]:
        return 1
    elif prev_direction == -1 and current_close >= final_lower_band[i]:
        return 1
    elif prev_direction == 1 and current_close <= final_upper_band[i]:
        return -1
    else:
        return prev_direction

def _calculate_supertrend_value(direction, final_upper_band, final_lower_band, i):
    """Calculate Supertrend value based on direction"""
    if direction[i] == 1:
        return final_lower_band[i]
    else:
        return final_upper_band[i]

def _initialize_supertrend_arrays(upper_band, lower_band):
    """Initialize arrays for Supertrend calculation"""
    final_upper_band = [upper_band.iloc[0]]
    final_lower_band = [lower_band.iloc[0]]
    supertrend = [None]  # First value is None
    direction = [1]  # Start with uptrend
    
    return final_upper_band, final_lower_band, supertrend, direction

