from imports import logger

AVAILABLE_TIMEFRAMES = {
    '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
    '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h',
    '12h': '12h', '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
}

TIMEFRAME_LIMITS = dict.fromkeys(AVAILABLE_TIMEFRAMES, 1000)

def _get_available_timeframes():
    """Get list of available timeframes"""
    return list(AVAILABLE_TIMEFRAMES.keys())

def _validate_timeframe(interval):
    """Validate and normalize timeframe"""
    if interval not in AVAILABLE_TIMEFRAMES:
        logger.warning(f"Invalid timeframe '{interval}'. Using default '1h'")
        logger.info(f"Available timeframes: {', '.join(AVAILABLE_TIMEFRAMES.keys())}")
        return '1h'
    return interval

def _get_optimal_limit(interval, custom_limit=None):
    """Get optimal limit for timeframe"""
    if custom_limit:
        return min(custom_limit, 1000)  # Cap at 1000 for API limits
    return TIMEFRAME_LIMITS.get(interval, 1000)
