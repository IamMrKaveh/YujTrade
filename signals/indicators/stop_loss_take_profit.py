from signals.indicators.cache_utils import _cached_indicator_calculation
from signals.indicators.support_resistance import _find_nearest_support, _find_nearest_resistance
from signals.indicators.volatility_indicators import _calculate_atr

def _get_default_stops(entry_price, position_type):
    """Get default stop loss and take profit values"""
    if position_type == 'long':
        return {
            'stop_loss': entry_price * 0.95,
            'take_profit': entry_price * 1.1,
            'risk_reward_ratio': 2.0
        }
    else:
        return {
            'stop_loss': entry_price * 1.05,
            'take_profit': entry_price * 0.9,
            'risk_reward_ratio': 2.0
        }

def _calculate_risk_multipliers(trend_data, volatility_data, base_risk):
    """Calculate risk and reward multipliers based on trend and volatility"""
    if trend_data is None or volatility_data is None:
        return base_risk, 2.5
    
    trend_strength = trend_data['strength']
    volatility_score = volatility_data['volatility_score']
    
    # Adjust multipliers based on trend strength
    if trend_strength > 50:
        risk_multiplier = base_risk * (1 + trend_strength / 100)
        reward_multiplier = 3.0 + trend_strength / 50
    else:
        risk_multiplier = base_risk * 0.7
        reward_multiplier = 1.5
    
    # Adjust based on volatility
    if volatility_score > 5:
        risk_multiplier *= 1.5
        reward_multiplier *= 1.3
    elif volatility_score < 2:
        risk_multiplier *= 0.8
        reward_multiplier *= 0.9
    
    return risk_multiplier, reward_multiplier

def _get_atr_value(df, entry_price, risk_multiplier):
    """Get ATR value adjusted by risk multiplier"""
    atr = _calculate_atr(df, 14)
    atr_value = atr.iloc[-1] if atr is not None else entry_price * 0.02
    return atr_value * risk_multiplier

def _calculate_long_stops(entry_price, atr_value, support_resistance):
    """Calculate stop loss and take profit for long positions"""
    stop_loss_atr = entry_price - atr_value
    
    # Adjust stop loss based on support levels
    if (support_resistance and support_resistance.get('support_levels')):
        nearest_support = _find_nearest_support(
            support_resistance['support_levels'], 
            entry_price, 
            stop_loss_atr
        )
        stop_loss = max(stop_loss_atr, nearest_support * 0.98)
    else:
        stop_loss = stop_loss_atr
    
    # Calculate take profit
    risk_amount = entry_price - stop_loss
    take_profit_atr = entry_price + (risk_amount * 2.5)
    
    # Adjust take profit based on resistance levels
    if (support_resistance and support_resistance.get('resistance_levels')):
        nearest_resistance = _find_nearest_resistance(
            support_resistance['resistance_levels'], 
            entry_price, 
            take_profit_atr
        )
        take_profit = min(take_profit_atr, nearest_resistance * 0.98)
    else:
        take_profit = take_profit_atr
    
    return stop_loss, take_profit

def _calculate_short_stops(entry_price, atr_value, support_resistance):
    """Calculate stop loss and take profit for short positions"""
    stop_loss_atr = entry_price + atr_value
    
    # Adjust stop loss based on resistance levels
    if (support_resistance and support_resistance.get('resistance_levels')):
        nearest_resistance = _find_nearest_resistance(
            support_resistance['resistance_levels'], 
            entry_price, 
            stop_loss_atr
        )
        stop_loss = min(stop_loss_atr, nearest_resistance * 1.02)
    else:
        stop_loss = stop_loss_atr
    
    # Calculate take profit
    risk_amount = stop_loss - entry_price
    take_profit_atr = entry_price - (risk_amount * 2.5)
    
    # Adjust take profit based on support levels
    if (support_resistance and support_resistance.get('support_levels')):
        nearest_support = _find_nearest_support(
            support_resistance['support_levels'], 
            entry_price, 
            take_profit_atr
        )
        take_profit = max(take_profit_atr, nearest_support * 1.02)
    else:
        take_profit = take_profit_atr
    
    return stop_loss, take_profit

def _ensure_minimum_risk_reward(entry_price, stop_loss, take_profit, position_type, min_ratio=1.5):
    """Ensure minimum risk-reward ratio"""
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    risk_reward_ratio = reward / risk if risk > 0 else 0
    
    if risk_reward_ratio < min_ratio:
        if position_type == 'long':
            take_profit = entry_price + (risk * min_ratio)
        else:
            take_profit = entry_price - (risk * min_ratio)
        risk_reward_ratio = min_ratio
    
    return take_profit, risk_reward_ratio

def _calculate_dynamic_stops(df, entry_price, position_type='long', base_risk=2.0):
    """محاسبه حد ضرر و حد سود پویا بر اساس ترند و نوسانات"""
    
    return _cached_indicator_calculation(df, 'dynamic_stops', _calculate_dynamic_stops, entry_price, position_type, base_risk)

def _calculate_trailing_stop(df, entry_price, current_price, position_type='long', atr_multiplier=2.0):
    """محاسبه حد ضرر متحرک"""
    try:
        if df is None or len(df) < 14:
            return entry_price * 0.95 if position_type == 'long' else entry_price * 1.05
        
        atr = _calculate_atr(df, 14)
        if atr is None:
            atr_value = abs(current_price - entry_price) * 0.1
        else:
            atr_value = atr.iloc[-1] * atr_multiplier
        
        if position_type == 'long':
            # برای پوزیشن خرید، حد ضرر به سمت بالا حرکت می‌کند
            trailing_stop = current_price - atr_value
            # حد ضرر نمی‌تواند پایین‌تر از قیمت ورود برود
            return max(trailing_stop, entry_price * 0.98)
        else:
            # برای پوزیشن فروش، حد ضرر به سمت پایین حرکت می‌کند
            trailing_stop = current_price + atr_value
            # حد ضرر نمی‌تواند بالاتر از قیمت ورود برود
            return min(trailing_stop, entry_price * 1.02)
            
    except Exception:
        return entry_price * 0.95 if position_type == 'long' else entry_price * 1.05

def _calculate_dynamic_stop_loss(df, entry_price, position_type='long', atr_multiplier=2):
    """محاسبه حد ضرر پویا"""
    try:
        if df is None or len(df) < 14:
            return entry_price * 0.95 if position_type == 'long' else entry_price * 1.05
            
        atr = _calculate_atr(df, 14)
        if atr is None:
            return entry_price * 0.95 if position_type == 'long' else entry_price * 1.05
            
        atr_value = atr.iloc[-1]
        
        if position_type == 'long':
            stop_loss = entry_price - (atr_value * atr_multiplier)
        else:
            stop_loss = entry_price + (atr_value * atr_multiplier)
            
        return stop_loss
    except Exception:
        return entry_price * 0.95 if position_type == 'long' else entry_price * 1.05

def _optimize_risk_reward_ratio(entry_price, target_price, stop_loss, min_ratio=2.0):
    """بهینه‌سازی نسبت ریسک-ریوارد"""
    try:
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
        
        current_ratio = reward / risk if risk > 0 else 0
        
        if current_ratio < min_ratio:
            # تنظیم هدف برای دستیابی به نسبت حداقل
            if entry_price > stop_loss:  # long position
                new_target = entry_price + (risk * min_ratio)
            else:  # short position
                new_target = entry_price - (risk * min_ratio)
            
            return new_target
        
        return target_price
    except Exception:
        return target_price

