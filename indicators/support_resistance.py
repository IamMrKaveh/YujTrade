from logger_config import logger
from .cache_utils import cached_calculation

def _calculate_pivot_points_internal(df):
    """Internal Pivot Points calculation function"""
    try:
        if df is None or len(df) < 1:
            return None
        
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        # Standard Pivot Points
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    except Exception as e:
        logger.error(f"Error calculating Pivot Points: {e}")
        return None

@cached_calculation('pivot_points')
def calculate_pivot_points(df):
    """محاسبه Pivot Points with caching"""
    return _calculate_pivot_points_internal(df)
            
def _calculate_support_resistance_internal(df, window):
    """Internal calculation for Support/Resistance levels"""
    try:
        if df is None or len(df) < window:
            return None
        
        high = df['high']
        low = df['low']
        
        # Local highs and lows
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(df) - window):
            # Check for local high (resistance)
            if high.iloc[i] == high.iloc[i-window:i+window+1].max():
                resistance_levels.append(high.iloc[i])
            
            # Check for local low (support)
            if low.iloc[i] == low.iloc[i-window:i+window+1].min():
                support_levels.append(low.iloc[i])
        
        # Get most significant levels
        resistance_levels = sorted(set(resistance_levels), reverse=True)[:5]
        support_levels = sorted(set(support_levels))[:5]
        
        return {
            'resistance_levels': resistance_levels,
            'support_levels': support_levels
        }
    except Exception as e:
        logger.error(f"Error calculating Support/Resistance: {e}")
        return None

@cached_calculation('support_resistance')
def _calculate_support_resistance(df, window=20):
    """محاسبه سطوح Support و Resistance with caching"""
    try:
        return _calculate_support_resistance_internal(df, window)
    except Exception as e:
        logger.error(f"Error calculating Support/Resistance: {e}")
        return None

def find_swing_points(high, low, swing_strength):
    """Helper function to find swing highs and lows"""
    swing_highs = []
    swing_lows = []
    
    for i in range(swing_strength, len(high) - swing_strength):
        # Swing High
        if high.iloc[i] == high.iloc[i-swing_strength:i+swing_strength+1].max():
            swing_highs.append((i, high.iloc[i]))
        
        # Swing Low
        if low.iloc[i] == low.iloc[i-swing_strength:i+swing_strength+1].min():
            swing_lows.append((i, low.iloc[i]))
    
    return swing_highs, swing_lows

def get_recent_high(swing_highs):
    """Get the most recent significant high"""
    if not swing_highs:
        return None
    return max(swing_highs[-3:], key=lambda x: x[1])[1] if len(swing_highs) >= 3 else swing_highs[-1][1]

def get_recent_low(swing_lows):
    """Get the most recent significant low"""
    if not swing_lows:
        return None
    return min(swing_lows[-3:], key=lambda x: x[1])[1] if len(swing_lows) >= 3 else swing_lows[-1][1]

def _calculate_support_resistance_levels_internal(df, window, min_touches):
    """Internal calculation for detailed Support/Resistance levels"""
    try:
        if df is None or len(df) < window * 2:
            return {
                'resistance_levels': [],
                'support_levels': [],
                'resistance_strength': [],
                'support_strength': []
            }
        
        high = df['high']
        low = df['low']
        
        # پیدا کردن نقاط pivot
        pivot_highs, pivot_lows = _find_pivot_points(high, low, window)
        
        # تجمیع سطوح مشابه
        resistance_clusters = _cluster_levels(pivot_highs)
        support_clusters = _cluster_levels(pivot_lows)
        
        # انتخاب قوی‌ترین سطوح
        strong_resistance = _extract_strong_levels(resistance_clusters, min_touches)
        strong_support = _extract_strong_levels(support_clusters, min_touches)
        
        return {
            'resistance_levels': [level[0] for level in strong_resistance],
            'support_levels': [level[0] for level in strong_support],
            'resistance_strength': [level[1] for level in strong_resistance],
            'support_strength': [level[1] for level in strong_support]
        }
    except Exception as e:
        logger.error(f"Error calculating support/resistance levels: {e}")
        return {
            'resistance_levels': [],
            'support_levels': [],
            'resistance_strength': [],
            'support_strength': []
        }

@cached_calculation('support_resistance_levels')
def calculate_support_resistance_levels(df, window=20, min_touches=3):
    """محاسبه سطوح حمایت و مقاومت دقیق with caching"""
    try:
        return _calculate_support_resistance_levels_internal(df, window, min_touches)
    except Exception as e:
        logger.error(f"Error calculating support/resistance levels: {e}")
        return None

def _find_pivot_points(high, low, window):
    """Helper function to find pivot highs and lows"""
    pivot_highs = []
    pivot_lows = []
    
    for i in range(window, len(high) - window):
        # Pivot High
        if high.iloc[i] == high.iloc[i-window:i+window+1].max():
            pivot_highs.append((i, high.iloc[i]))
        
        # Pivot Low
        if low.iloc[i] == low.iloc[i-window:i+window+1].min():
            pivot_lows.append((i, low.iloc[i]))
    
    return pivot_highs, pivot_lows

def _cluster_levels(levels, tolerance=0.01):
    """Helper function to cluster similar price levels"""
    if not levels:
        return []
    
    levels = sorted(levels, key=lambda x: x[1])
    clusters = []
    current_cluster = [levels[0]]
    
    for level in levels[1:]:
        if abs(level[1] - current_cluster[-1][1]) / current_cluster[-1][1] <= tolerance:
            current_cluster.append(level)
        else:
            clusters.append(current_cluster)
            current_cluster = [level]
    clusters.append(current_cluster)
    
    return clusters

def _extract_strong_levels(clusters, min_touches):
    """Helper function to extract strong levels from clusters"""
    strong_levels = []
    
    for cluster in clusters:
        if len(cluster) >= min_touches:
            # Calculate average price level for the cluster
            avg_price = sum(level[1] for level in cluster) / len(cluster)
            # Return (price, strength) where strength is number of touches
            strong_levels.append((avg_price, len(cluster)))
    
    # Sort by strength (number of touches) in descending order
    strong_levels.sort(key=lambda x: x[1], reverse=True)
    
    return strong_levels

def _find_nearest_support(support_levels, entry_price, default_value):
    """Find nearest support level below entry price"""
    if not support_levels:
        return default_value
    
    valid_supports = [s for s in support_levels if s < entry_price]
    return max(valid_supports, default=default_value)

def _find_nearest_resistance(resistance_levels, entry_price, default_value):
    """Find nearest resistance level above entry price"""
    if not resistance_levels:
        return default_value
    
    valid_resistances = [r for r in resistance_levels if r > entry_price]
    return min(valid_resistances, default=default_value)

