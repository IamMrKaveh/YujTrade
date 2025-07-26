from imports import logger, pd, np
from signals.indicators.support_resistance import _find_swing_points, _get_recent_high, _get_recent_low
from signals.indicators.cache_utils import _cached_indicator_calculation

def _detect_breaks(structure_breaks, current_price, swing_highs, swing_lows):
    """Detect bullish and bearish breaks"""
    # Check if current price breaks recent swing high (bullish break)
    recent_high = _get_recent_high(swing_highs)
    if recent_high and current_price > recent_high:
        structure_breaks.loc[structure_breaks.index[-1], 'bullish_break'] = True
    
    # Check if current price breaks recent swing low (bearish break)
    recent_low = _get_recent_low(swing_lows)
    if recent_low and current_price < recent_low:
        structure_breaks.loc[structure_breaks.index[-1], 'bearish_break'] = True

def _detect_market_structure_breaks(df, swing_strength=5):
    """تشخیص Market Structure Breaks"""
    try:
        if df is None or len(df) < swing_strength * 2:
            return None
        
        high = df['high']
        low = df['low']
        
        structure_breaks = pd.DataFrame(index=df.index)
        structure_breaks['bullish_break'] = False
        structure_breaks['bearish_break'] = False
        
        # Find swing highs and lows
        swing_highs, swing_lows = _find_swing_points(high, low, swing_strength)
        
        # Detect breaks
        current_price = df['close'].iloc[-1]
        _detect_breaks(structure_breaks, current_price, swing_highs, swing_lows)
        
        return structure_breaks
    except Exception as e:
        logger.warning(f"Error detecting Market Structure Breaks: {e}")
        return None

def _calculate_market_structure_score(df, lookback=20):
    """Calculate market structure quality score"""
    try:
        if df is None or len(df) < lookback:
            return 0
        
        recent_data = df.tail(lookback)
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        closes = recent_data['close'].values
        
        # Higher highs and higher lows for uptrend
        higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        
        # Lower highs and lower lows for downtrend
        lower_highs = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
        lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
        
        # Price momentum consistency
        up_moves = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
        down_moves = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i-1])
        
        # Volume trend consistency
        if 'volume' in recent_data.columns:
            volumes = recent_data['volume'].values
            volume_trend = sum(1 for i in range(1, len(volumes)) if volumes[i] > volumes[i-1])
            volume_consistency = volume_trend / (len(volumes) - 1) if len(volumes) > 1 else 0.5
        else:
            volume_consistency = 0.5
        
        # Calculate structure strength
        uptrend_strength = (higher_highs + higher_lows) / (2 * (lookback - 1))
        downtrend_strength = (lower_highs + lower_lows) / (2 * (lookback - 1))
        
        # Momentum consistency
        momentum_consistency = max(up_moves, down_moves) / (len(closes) - 1) if len(closes) > 1 else 0.5
        
        # Final structure score
        if uptrend_strength > downtrend_strength:
            structure_score = (uptrend_strength * 0.4 + momentum_consistency * 0.4 + volume_consistency * 0.2) * 100
        else:
            structure_score = (downtrend_strength * 0.4 + momentum_consistency * 0.4 + volume_consistency * 0.2) * 100
        
        return min(structure_score, 100)
        
    except Exception:
        return 0

def _calculate_market_microstructure_internal(df, period):
    """Internal market microstructure calculation function"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
        
        # Bid-Ask Spread Proxy
        spread_proxy = (high - low) / close
        avg_spread = spread_proxy.rolling(window=period).mean()
        
        # Market Depth Indicator
        price_impact = (high - low) / volume
        market_depth = price_impact.rolling(window=period).mean()
        
        # Order Flow Imbalance
        price_change = close.pct_change()
        volume_weighted_price_change = price_change * volume
        order_flow = volume_weighted_price_change.rolling(window=period).sum()
        
        # Liquidity Score
        liquidity_score = volume / (high - low)
        liquidity_score = liquidity_score.replace([np.inf, -np.inf], 0).fillna(0)
        avg_liquidity = liquidity_score.rolling(window=period).mean()
        
        return {
            'spread_proxy': avg_spread,
            'market_depth': market_depth,
            'order_flow': order_flow,
            'liquidity_score': avg_liquidity
        }
    except Exception as e:
        logger.warning(f"Error calculating market microstructure: {e}")
        return None

def _calculate_market_microstructure(df, period=20):
    """محاسبه ساختار میکرو بازار with caching"""
    return _cached_indicator_calculation(df, 'market_microstructure', _calculate_market_microstructure_internal, period)

def _detect_market_regime(df, lookback=50):
    """تشخیص رژیم بازار"""
    try:
        if df is None or len(df) < lookback:
            return None
            
        close = df['close']
        
        # محاسبه نوسانات
        returns = close.pct_change()
        volatility = returns.rolling(lookback).std() * np.sqrt(252)  # سالانه
        
        # محاسبه ترند
        sma_short = close.rolling(10).mean()
        sma_long = close.rolling(50).mean()
        trend = sma_short - sma_long
        
        # تعیین رژیم بازار
        regime = pd.Series(index=df.index, dtype=str)
        
        for i in range(lookback, len(df)):
            vol = volatility.iloc[i]
            tr = trend.iloc[i]
            
            if vol > volatility.rolling(lookback).quantile(0.75).iloc[i]:
                if tr > 0:
                    regime.iloc[i] = 'Bull_Volatile'
                else:
                    regime.iloc[i] = 'Bear_Volatile'
            else:
                if tr > 0:
                    regime.iloc[i] = 'Bull_Stable'
                else:
                    regime.iloc[i] = 'Bear_Stable'
        
        return regime
    except Exception:
        return None
