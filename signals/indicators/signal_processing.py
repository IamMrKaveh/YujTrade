import numpy as np
from .volume_indicators import _check_volume_filter

def _ensemble_signal_scoring(signals_dict, weights=None):
    """ترکیب چندین سیگنال با وزن‌دهی"""
    try:
        if not signals_dict:
            return 0
            
        if weights is None:
            weights = dict.fromkeys(signals_dict.keys(), 1)
        
        total_score = 0
        total_weight = 0
        
        for signal_name, signal_value in signals_dict.items():
            if signal_name in weights and signal_value is not None:
                weight = weights[signal_name]
                total_score += signal_value * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    except Exception:
        return 0

def _adaptive_threshold_calculator(df, indicator_values, percentile_low=20, percentile_high=80):
    """محاسبه آستانه‌های تطبیقی"""
    try:
        if df is None or indicator_values is None:
            return {'low': 30, 'high': 70}
            
        # محاسبه آستانه‌ها بر اساس توزیع تاریخی
        low_threshold = np.percentile(indicator_values.dropna(), percentile_low)
        high_threshold = np.percentile(indicator_values.dropna(), percentile_high)
        
        return {
            'low': low_threshold,
            'high': high_threshold
        }
    except Exception:
        return {'low': 30, 'high': 70}

def _extract_signal_type(signal_data):
    """Extract signal type from signal data"""
    if isinstance(signal_data, dict) and 'type' in signal_data:
        return signal_data['type']
    elif isinstance(signal_data, str):
        return signal_data
    return None

def _check_trend_filter(df, signal_data, min_trend_strength):
    """Check if trend strength supports the signal"""
    if len(df) < 10:
        return True
    
    recent_closes = df['close'].tail(10).values
    if len(recent_closes) == 0 or recent_closes[0] == 0:
        return True
    
    trend_strength = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
    signal_type = _extract_signal_type(signal_data)
    
    if signal_type == 'buy' and trend_strength < -min_trend_strength and signal_type == 'sell' and trend_strength > min_trend_strength:
        return False
    
    return True

def _filter_false_signals(df, signal_data, min_volume_ratio=1.2, min_trend_strength=0.1):
    """فیلتر سیگنال‌های کاذب"""
    try:
        if df is None or not signal_data:
            return False
        
        if not _check_volume_filter(df, min_volume_ratio):
            return False
        
        if not _check_trend_filter(df, signal_data, min_trend_strength):
            return False
        
        return True
    except Exception:
        return True
