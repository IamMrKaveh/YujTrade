from .cache_utils import _cached_indicator_calculation
from .support_resistance import _find_nearest_support, _find_nearest_resistance
from .volatility_indicators import _calculate_atr
from logger_config import logger

def calculate_comprehensive_risk_management(df, entry_price, position_type='long', 
                                            current_price=None, base_risk=2.0, 
                                            atr_multiplier=2.0, min_risk_reward_ratio=1.5,
                                            use_support_resistance=False, support_resistance_data=None):
    """
    تابع جامع مدیریت ریسک که تمام محاسبات حد ضرر و حد سود را انجام می‌دهد
    
    Args:
        df: DataFrame حاوی داده‌های قیمتی
        entry_price: قیمت ورود به پوزیشن
        position_type: نوع پوزیشن ('long' یا 'short')
        current_price: قیمت فعلی (برای trailing stop)
        base_risk: ریسک پایه (پیش‌فرض 2.0)
        atr_multiplier: ضریب ATR (پیش‌فرض 2.0)
        min_risk_reward_ratio: حداقل نسبت ریسک-ریوارد (پیش‌فرض 1.5)
        use_support_resistance: استفاده از سطوح حمایت و مقاومت
        support_resistance_data: داده‌های سطوح حمایت و مقاومت
    
    Returns:
        dict: شامل تمام اطلاعات مدیریت ریسک
    """
    try:
        # محاسبه حدود پویا
        dynamic_stops = _calculate_dynamic_stops(df, entry_price, position_type, base_risk)
        
        # محاسبه حد ضرر پویا با ATR
        dynamic_stop_loss = _calculate_dynamic_stop_loss(df, entry_price, position_type, atr_multiplier)
        
        # محاسبه trailing stop اگر قیمت فعلی موجود باشد
        trailing_stop = None
        if current_price is not None:
            trailing_stop = _calculate_trailing_stop(df, entry_price, current_price, 
                                                    position_type, atr_multiplier)
        
        # محاسبه حدود بر اساس سطوح حمایت و مقاومت
        sr_based_stops = None
        if use_support_resistance and support_resistance_data:
            sr_based_stops = _calculate_stops_with_support_resistance(
                df, entry_price, position_type, support_resistance_data, base_risk
            )
        
        # بهینه‌سازی نسبت ریسک-ریوارد
        optimized_target = _optimize_risk_reward_ratio(
            entry_price, 
            dynamic_stops['take_profit'], 
            dynamic_stops['stop_loss'], 
            min_risk_reward_ratio
        )
        
        # محاسبه آمار نهایی
        final_stop_loss = _get_best_stop_loss(dynamic_stops['stop_loss'], 
                                            dynamic_stop_loss, 
                                            sr_based_stops['stop_loss'] if sr_based_stops else None,
                                            position_type)
        
        final_take_profit = _get_best_take_profit(optimized_target,
                                                sr_based_stops['take_profit'] if sr_based_stops else None,
                                                position_type, entry_price)
        
        # محاسبه نسبت ریسک-ریوارد نهایی
        risk = abs(entry_price - final_stop_loss)
        reward = abs(final_take_profit - entry_price)
        final_risk_reward_ratio = reward / risk if risk > 0 else 0
        
        logger.info(f"Final Stop Loss: {final_stop_loss}, Take Profit: {final_take_profit}, ")
        return {
            'entry_price': entry_price,
            'position_type': position_type,
            'stop_loss': final_stop_loss,
            'take_profit': final_take_profit,
            'risk_reward_ratio': final_risk_reward_ratio,
            'trailing_stop': trailing_stop,
            'risk_amount': risk,
            'reward_amount': reward,
            'calculations': {
                'dynamic_stops': dynamic_stops,
                'dynamic_stop_loss': dynamic_stop_loss,
                'sr_based_stops': sr_based_stops,
                'optimized_target': optimized_target
            },
            'status': 'success'
        }
        
    except Exception as e:
        # در صورت بروز خطا، حدود پیش‌فرض را برگردان
        default_stops = _get_default_stops(entry_price, position_type)
        default_stops['status'] = 'error'
        default_stops['error_message'] = str(e)
        logger.error(f"Error in risk management calculations: {e}")
        return default_stops


def _calculate_stops_with_support_resistance(df, entry_price, position_type, 
                                            support_resistance_data, base_risk):
    """محاسبه حدود با در نظر گیری سطوح حمایت و مقاومت"""
    try:
        # محاسبه ATR برای تعیین ریسک
        risk_multiplier, _ = _calculate_risk_multipliers(None, None, base_risk)
        atr_value = _get_atr_value(df, entry_price, risk_multiplier)
        
        if position_type == 'long':
            stop_loss, take_profit = _calculate_long_stops(
                entry_price, atr_value, support_resistance_data
            )
        else:
            stop_loss, take_profit = _calculate_short_stops(
                entry_price, atr_value, support_resistance_data
            )
        
        logger.info(f"Calculated Stop Loss: {stop_loss}, Take Profit: {take_profit}")
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }

    except Exception as e:
        logger.error(f"Error in stop loss / take profit calculations: {e}")
        return None


def _get_best_stop_loss(dynamic_sl, atr_sl, sr_sl, position_type):
    """انتخاب بهترین حد ضرر از بین گزینه‌های مختلف"""
    stop_losses = [sl for sl in [dynamic_sl, atr_sl, sr_sl] if sl is not None]
    
    if not stop_losses:
        return dynamic_sl
    
    if position_type == 'long':
        # برای long، بهترین stop loss بالاترین مقدار است (کمترین ریسک)
        return max(stop_losses)
    else:
        # برای short، بهترین stop loss پایین‌ترین مقدار است (کمترین ریسک)
        return min(stop_losses)


def _get_best_take_profit(optimized_tp, sr_tp, position_type, entry_price):
    """انتخاب بهترین حد سود"""
    take_profits = [tp for tp in [optimized_tp, sr_tp] if tp is not None]
    
    if not take_profits:
        return optimized_tp
    
    # انتخاب محافظه‌کارانه‌تر (نزدیک‌تر به قیمت ورود)
    if position_type == 'long':
        return min(take_profits)
    else:
        return max(take_profits)


def update_trailing_stop(df, entry_price, current_price, current_trailing_stop, 
                        position_type='long', atr_multiplier=2.0):
    """
    به‌روزرسانی trailing stop بر اساس قیمت فعلی
    
    Args:
        df: DataFrame حاوی داده‌های قیمتی
        entry_price: قیمت ورود
        current_price: قیمت فعلی
        current_trailing_stop: trailing stop فعلی
        position_type: نوع پوزیشن
        atr_multiplier: ضریب ATR
    
    Returns:
        float: trailing stop جدید
    """
    new_trailing_stop = _calculate_trailing_stop(df, entry_price, current_price, 
                                                position_type, atr_multiplier)
    
    if position_type == 'long':
        # trailing stop فقط بالا می‌رود
        return max(new_trailing_stop, current_trailing_stop)
    else:
        # trailing stop فقط پایین می‌آید
        return min(new_trailing_stop, current_trailing_stop)


def validate_risk_parameters(entry_price, stop_loss, take_profit, position_type, 
                            max_risk_percent=5.0, min_risk_reward=1.0):
    """
    اعتبارسنجی پارامترهای ریسک
    
    Args:
        entry_price: قیمت ورود
        stop_loss: حد ضرر
        take_profit: حد سود
        position_type: نوع پوزیشن
        max_risk_percent: حداکثر درصد ریسک
        min_risk_reward: حداقل نسبت ریسک-ریوارد
    
    Returns:
        dict: نتیجه اعتبارسنجی
    """
    warnings = []
    errors = []
    
    # بررسی منطقی بودن حدود
    if position_type == 'long':
        if stop_loss >= entry_price:
            errors.append("Stop loss باید کمتر از قیمت ورود باشد")
        if take_profit <= entry_price:
            errors.append("Take profit باید بیشتر از قیمت ورود باشد")
    else:
        if stop_loss <= entry_price:
            errors.append("Stop loss باید بیشتر از قیمت ورود باشد")
        if take_profit >= entry_price:
            errors.append("Take profit باید کمتر از قیمت ورود باشد")
    
    # محاسبه ریسک و ریوارد
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    risk_percent = (risk / entry_price) * 100
    risk_reward_ratio = reward / risk if risk > 0 else 0
    
    # بررسی حداکثر ریسک
    if risk_percent > max_risk_percent:
        warnings.append(f"ریسک {risk_percent:.2f}% بیش از حد مجاز {max_risk_percent}% است")
    
    # بررسی نسبت ریسک-ریوارد
    if risk_reward_ratio < min_risk_reward:
        warnings.append(f"نسبت ریسک-ریوارد {risk_reward_ratio:.2f} کمتر از حد مجاز {min_risk_reward} است")
    
    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'risk_percent': risk_percent,
        'risk_reward_ratio': risk_reward_ratio
    }


# توابع کمکی موجود از کد قبلی
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
    
    def _dynamic_stops_calc():
        try:
            if df is None or len(df) < 20:
                return _get_default_stops(entry_price, position_type)
            
            # Get volatility data using ATR
            atr = _calculate_atr(df, 14)
            if atr is None or atr.empty:
                return _get_default_stops(entry_price, position_type)
            
            volatility_data = {'volatility_score': atr.iloc[-1] / entry_price * 100}
            trend_data = {'strength': 50}  # Default trend strength
            
            # Calculate risk multipliers
            risk_multiplier, _ = _calculate_risk_multipliers(
                trend_data, volatility_data, base_risk
            )
            
            # Get ATR value
            atr_value = _get_atr_value(df, entry_price, risk_multiplier)
            
            # Calculate stops based on position type
            if position_type == 'long':
                stop_loss, take_profit = _calculate_long_stops(
                    entry_price, atr_value, None
                )
            else:
                stop_loss, take_profit = _calculate_short_stops(
                    entry_price, atr_value, None
                )
            
            # Ensure minimum risk-reward ratio
            take_profit, risk_reward_ratio = _ensure_minimum_risk_reward(
                entry_price, stop_loss, take_profit, position_type
            )
            
            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': risk_reward_ratio
            }
            
        except Exception:
            return _get_default_stops(entry_price, position_type)
    
    return _cached_indicator_calculation(df, 'dynamic_stops', _dynamic_stops_calc)


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