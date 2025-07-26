from imports import logger

def _adaptive_position_sizing(capital, risk_percent, entry_price, stop_loss, market_conditions=None):
    """محاسبه اندازه پوزیشن تطبیقی"""
    try:
        base_risk = capital * (risk_percent / 100)
        price_diff = abs(entry_price - stop_loss)
        
        if price_diff == 0:
            return 0
        
        base_position_size = base_risk / price_diff
        
        # تنظیم بر اساس شرایط بازار
        if market_conditions:
            volatility = market_conditions.get('volatility_score', 3)
            trend_strength = market_conditions.get('trend_strength', 50)
            
            # نوسانات بالا = اندازه پوزیشن کمتر
            if volatility > 5:
                size_multiplier = 0.7
            elif volatility < 2:
                size_multiplier = 1.2
            else:
                size_multiplier = 1.0
            
            # ترند قوی = اندازه پوزیشن بیشتر
            if trend_strength > 70:
                size_multiplier *= 1.3
            elif trend_strength < 30:
                size_multiplier *= 0.8
            
            position_size = base_position_size * size_multiplier
        else:
            position_size = base_position_size
        
        # محدودیت حداکثر 10% سرمایه
        max_position_value = capital * 0.1
        max_position_size = max_position_value / entry_price
        
        final_position_size = min(position_size, max_position_size)
        
        return max(final_position_size, 0)
        
    except Exception as e:
        logger.error(f"Error calculating adaptive position size: {e}")
        return 0

def _calculate_position_size_atr(capital, risk_percent, atr_value, atr_multiplier=2):
    """محاسبه اندازه پوزیشن بر اساس ATR"""
    try:
        risk_amount = capital * (risk_percent / 100)
        stop_distance = atr_value * atr_multiplier
        position_size = risk_amount / stop_distance
        
        return min(position_size, capital * 0.1)  # حداکثر 10% سرمایه
    except Exception:
        return 0
