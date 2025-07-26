from imports import logger

def _calculate_dynamic_levels_long(entry_price, volatility_data, symbol):
    """محاسبه سطوح داینامیک برای موقعیت Long"""
    try:
        atr_percentage = volatility_data['atr_percentage']
        volatility_factor = volatility_data['volatility_factor']
        
        # محاسبه target داینامیک (1.5 تا 4 برابر ATR)
        target_multiplier = max(1.5, min(4.0, 2.0 * volatility_factor))
        target_percentage = atr_percentage * target_multiplier
        target = entry_price * (1 + target_percentage / 100)
        
        # محاسبه stop loss داینامیک (0.8 تا 2.5 برابر ATR)
        stop_multiplier = max(0.8, min(2.5, 1.2 * volatility_factor))
        stop_percentage = atr_percentage * stop_multiplier
        stop_loss = entry_price * (1 - stop_percentage / 100)
        
        # اطمینان از Risk/Reward ratio معقول (حداقل 1:1.5)
        risk = entry_price - stop_loss
        reward = target - entry_price
        
        if reward / risk < 1.5:
            # تنظیم target برای بهبود ratio
            target = entry_price + (risk * 1.5)
        
        logger.debug(f"Dynamic Long levels for {symbol}: target={target:.6f} (+{target_percentage:.2f}%), stop_loss={stop_loss:.6f} (-{stop_percentage:.2f}%)")
        
        return target, stop_loss
        
    except Exception as e:
        logger.error(f"Error calculating dynamic Long levels for {symbol}: {e}")
        # fallback به مقادیر ثابت
        return entry_price * 1.03, entry_price * 0.98

def _calculate_dynamic_levels_short(entry_price, volatility_data, symbol):
    """محاسبه سطوح داینامیک برای موقعیت Short"""
    try:
        atr_percentage = volatility_data['atr_percentage']
        volatility_factor = volatility_data['volatility_factor']
        
        # محاسبه target داینامیک (1.5 تا 4 برابر ATR)
        target_multiplier = max(1.5, min(4.0, 2.0 * volatility_factor))
        target_percentage = atr_percentage * target_multiplier
        target = entry_price * (1 - target_percentage / 100)
        
        # محاسبه stop loss داینامیک (0.8 تا 2.5 برابر ATR)
        stop_multiplier = max(0.8, min(2.5, 1.2 * volatility_factor))
        stop_percentage = atr_percentage * stop_multiplier
        stop_loss = entry_price * (1 + stop_percentage / 100)
        
        # اطمینان از Risk/Reward ratio معقول (حداقل 1:1.5)
        risk = stop_loss - entry_price
        reward = entry_price - target
        
        if reward / risk < 1.5:
            # تنظیم target برای بهبود ratio
            target = entry_price - (risk * 1.5)
        
        logger.debug(f"Dynamic Short levels for {symbol}: target={target:.6f} (-{target_percentage:.2f}%), stop_loss={stop_loss:.6f} (+{stop_percentage:.2f}%)")
        
        return target, stop_loss
        
    except Exception as e:
        logger.error(f"Error calculating dynamic Short levels for {symbol}: {e}")
        # fallback به مقادیر ثابت
        return entry_price * 0.97, entry_price * 1.02
