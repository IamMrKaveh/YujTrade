from signals.indicators.cache_utils import _cached_indicator_calculation


def _calculate_fibonacci_levels(df, lookback=50):
    """محاسبه سطوح فیبوناچی"""
    return _cached_indicator_calculation(df, 'fibonacci_levels', _calculate_fibonacci_levels, lookback)