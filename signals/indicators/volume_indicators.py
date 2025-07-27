from logger_config import logger
import numpy as np
import pandas as pd
from .cache_utils import cached_calculation

@cached_calculation('obv')
def _calculate_obv(df):
    """محاسبه On-Balance Volume"""
    try:
        if df is None or len(df) < 2:
            return None
        
        close = df['close']
        volume = df['volume']
        
        obv = []
        obv.append(0)  # مقدار اولیه
        
        for i in range(1, len(df)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[i-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[i-1] - volume.iloc[i])
            else:
                obv.append(obv[i-1])
        
        return pd.Series(obv, index=df.index)
    except Exception as e:
        logger.warning(f"Error calculating OBV: {e}")
        return None
@cached_calculation('accumulation_distribution')
def _calculate_accumulation_distribution(df):
    """محاسبه Accumulation/Distribution Line"""
    try:
        if df is None or len(df) < 1:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        # Handle division by zero
        mfm = mfm.fillna(0)
        
        # Money Flow Volume
        mfv = mfm * volume
        
        # A/D Line is cumulative sum of MFV
        ad_line = mfv.cumsum()
        
        return ad_line
    except Exception as e:
        logger.warning(f"Error calculating A/D Line: {e}")
        return None
@cached_calculation('ad_line')
def _calculate_ad_line(df):
    """محاسبه Accumulation/Distribution Line"""
    try:
        if df is None or len(df) < 1:
            return None
            
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # محاسبه Money Flow Multiplier
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # در صورت صفر بودن دامنه
        
        # محاسبه Money Flow Volume
        mfv = clv * volume
        
        # محاسبه A/D Line تجمعی
        ad_line = mfv.cumsum()
        
        return ad_line
    except Exception:
        return None
@cached_calculation('chaikin_money_flow')
def _calculate_chaikin_money_flow(df, period=20):
    """محاسبه Chaikin Money Flow"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)
        
        # Money Flow Volume
        mfv = mfm * volume
        
        # CMF = Sum of MFV over period / Sum of Volume over period
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        return cmf
    except Exception as e:
        logger.warning(f"Error calculating CMF: {e}")
        return None
@cached_calculation('volume_price_trend')
def _calculate_volume_price_trend(df):
    """محاسبه Volume Price Trend"""
    try:
        if df is None or len(df) < 2:
            return None
        
        close = df['close']
        volume = df['volume']
        
        # Price change percentage
        price_change_pct = (close - close.shift(1)) / close.shift(1)
        
        # VPT = Previous VPT + Volume * Price Change %
        vpt = (price_change_pct * volume).cumsum()
        
        return vpt
    except Exception as e:
        logger.warning(f"Error calculating VPT: {e}")
        return None
@cached_calculation('ease_of_movement')
def _calculate_ease_of_movement(df, period=14):
    """محاسبه Ease of Movement"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Distance Moved
        distance_moved = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2
        
        # Box Height
        box_height = (volume / 100000) / (high - low)
        
        # 1-Period EMV
        emv_1period = distance_moved / box_height
        emv_1period = emv_1period.replace([np.inf, -np.inf], 0).fillna(0)
        
        # EMV = SMA of 1-Period EMV
        emv = emv_1period.rolling(window=period).mean()
        
        return emv
    except Exception as e:
        logger.warning(f"Error calculating EMV: {e}")
        return None
@cached_calculation('vwap')
def _calculate_vwap(df):
    """محاسبه Volume Weighted Average Price"""
    try:
        if df is None or len(df) < 1:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
        
        # محاسبه Typical Price
        typical_price = (high + low + close) / 3
        
        # محاسبه VWAP
        cumulative_typical_price_volume = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()
        
        vwap = cumulative_typical_price_volume / cumulative_volume
        
        # محاسبه VWAP Bands (انحراف معیار)
        vwap_variance = ((typical_price - vwap) ** 2 * volume).cumsum() / cumulative_volume
        vwap_std = np.sqrt(vwap_variance)
        
        vwap_upper1 = vwap + vwap_std
        vwap_lower1 = vwap - vwap_std
        vwap_upper2 = vwap + 2 * vwap_std
        vwap_lower2 = vwap - 2 * vwap_std
        
        return {
            'vwap': vwap,
            'vwap_upper1': vwap_upper1,
            'vwap_lower1': vwap_lower1,
            'vwap_upper2': vwap_upper2,
            'vwap_lower2': vwap_lower2
        }
    except Exception as e:
        logger.warning(f"Error calculating VWAP: {e}")
        return None

def _check_volume_filter(df, min_volume_ratio):
    """Check if volume meets minimum ratio requirement"""
    if 'volume' not in df.columns or len(df) < 20:
        return True
    
    volume_sma = df['volume'].rolling(window=20).mean().iloc[-1]
    if pd.isna(volume_sma) or volume_sma <= 0:
        return True
    
    last_volume = df.iloc[-1]['volume']
    volume_ratio = last_volume / volume_sma
    return volume_ratio >= min_volume_ratio
