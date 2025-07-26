from .constants import (
    OVER_SELL, OVER_BUY, BALANCED, STRONG_INFLOW, STRONG_OUTFLOW,
    NATURAL_ZONE, HIGH_VOLUME, MEDIUM_VOLUME, LOW_VOLUME,
    ASCENDING, DESCENDING, NO_TREND
)

def _get_rsi_status(rsi_value):
    """Get RSI status message"""
    if rsi_value < 30:
        return OVER_SELL
    elif rsi_value > 70:
        return OVER_BUY
    else:
        return BALANCED

def _get_stoch_status(stoch_value):
    """Get Stochastic status message"""
    if stoch_value < 20:
        return OVER_SELL
    elif stoch_value > 80:
        return OVER_BUY
    else:
        return BALANCED

def _get_mfi_status(mfi_value):
    """Get MFI status message"""
    if mfi_value < 20:
        return STRONG_OUTFLOW
    elif mfi_value > 80:
        return STRONG_INFLOW
    else:
        return BALANCED

def _get_cci_status(cci_value):
    """Get CCI status message"""
    if cci_value < -100:
        return OVER_SELL
    elif cci_value > 100:
        return OVER_BUY
    else:
        return NATURAL_ZONE

def _get_williams_status(williams_value):
    """Get Williams %R status message"""
    if williams_value < -80:
        return OVER_SELL
    elif williams_value > -20:
        return OVER_BUY
    else:
        return BALANCED

def _get_volume_status(volume_ratio):
    """Get volume status message"""
    if volume_ratio > 2:
        return HIGH_VOLUME
    elif volume_ratio > 1.5:
        return MEDIUM_VOLUME
    else:
        return LOW_VOLUME

def _get_trend_status(trend_direction):
    """Get trend direction status"""
    if trend_direction > 0:
        return ASCENDING
    elif trend_direction < 0:
        return DESCENDING
    else:
        return NO_TREND

