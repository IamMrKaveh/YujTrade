from logger_config import logger
import numpy as np

# Fix numpy compatibility issue
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

def _calculate_combined_score(signal):
    """محاسبه امتیاز ترکیبی برای مرتب‌سازی"""
    if not _validate_signal_for_scoring(signal):
        return 0
    
    symbol = signal.get('symbol', 'Unknown')
    logger.debug(f"Starting combined score calculation for signal: {symbol}")
    
    try:
        base_score = _extract_base_score(signal, symbol)
        strength_bonus = _calculate_strength_bonus(signal, symbol)
        volume_bonus = _calculate_volume_bonus(signal, symbol)
        trend_bonus = _calculate_trend_bonus(signal, symbol)
        major_bonus = _calculate_major_pair_bonus(signal, symbol)
        
        return _combine_and_validate_scores(
            base_score, strength_bonus, volume_bonus, 
            trend_bonus, major_bonus, symbol
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in _calculate_combined_score for {symbol}: {e}")
        return _get_fallback_score(signal)

def _validate_signal_for_scoring(signal):
    """Validate input signal for scoring"""
    if not signal or not isinstance(signal, dict):
        logger.warning("Invalid signal provided to _calculate_combined_score")
        return False
    return True

def _extract_base_score(signal, symbol):
    """Extract and validate base accuracy score"""
    try:
        base_score = float(signal.get('accuracy_score', 0))
        if not np.isfinite(base_score) or base_score < 0:
            logger.warning(f"Invalid base_score for {symbol}: {base_score}, using 0")
            return 0
        return base_score
    except (ValueError, TypeError) as e:
        logger.warning(f"Error converting base_score for {symbol}: {e}, using 0")
        return 0

def _calculate_strength_bonus(signal, symbol):
    """Calculate strength bonus with validation"""
    try:
        strength = float(signal.get('strength', 1))
        if not np.isfinite(strength) or strength < 0:
            logger.debug(f"Invalid strength for {symbol}: {strength}, using 1")
            strength = 1
        return strength * 5
    except (ValueError, TypeError) as e:
        logger.warning(f"Error processing strength for {symbol}: {e}, using default")
        return 5

def _calculate_volume_bonus(signal, symbol):
    """Calculate volume bonus with validation"""
    try:
        volume_ratio = float(signal.get('volume_ratio', 1))
        if not np.isfinite(volume_ratio) or volume_ratio < 0:
            logger.debug(f"Invalid volume_ratio for {symbol}: {volume_ratio}, using 1")
            volume_ratio = 1
        return min(volume_ratio, 3) * 2
    except (ValueError, TypeError) as e:
        logger.warning(f"Error processing volume_ratio for {symbol}: {e}, using default")
        return 2

def _calculate_trend_bonus(signal, symbol):
    """Calculate trend bonus with validation"""
    try:
        trend_direction = float(signal.get('trend_direction', 0))
        if not np.isfinite(trend_direction):
            logger.debug(f"Invalid trend_direction for {symbol}: {trend_direction}, using 0")
            trend_direction = 0
        return abs(trend_direction) * 10
    except (ValueError, TypeError) as e:
        logger.warning(f"Error processing trend_direction for {symbol}: {e}, using default")
        return 0

def _calculate_major_pair_bonus(signal, symbol):
    """Calculate major pair bonus"""
    try:
        major_pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        return 5 if signal.get('symbol') in major_pairs else 0
    except Exception as e:
        logger.warning(f"Error calculating major_bonus for {symbol}: {e}, using 0")
        return 0

def _combine_and_validate_scores(base_score, strength_bonus, volume_bonus, trend_bonus, major_bonus, symbol):
    """Combine all scores and validate the final result"""
    try:
        combined_score = base_score + strength_bonus + volume_bonus + trend_bonus + major_bonus
        
        if not np.isfinite(combined_score):
            logger.error(f"Combined score is not finite for {symbol}: {combined_score}")
            return base_score if np.isfinite(base_score) else 0
        
        final_score = min(combined_score, 150)
        
        logger.debug(f"Combined score calculated for {symbol}: base={base_score:.1f}, "
                    f"strength={strength_bonus:.1f}, volume={volume_bonus:.1f}, "
                    f"trend={trend_bonus:.1f}, major={major_bonus}, final={final_score:.1f}")
        
        return final_score
        
    except Exception as e:
        logger.error(f"Error combining scores for {symbol}: {e}")
        return base_score if np.isfinite(base_score) else 0

def _get_fallback_score(signal):
    """Get fallback score in case of errors"""
    try:
        logger.debug("Starting fallback score calculation")
        
        if not signal:
            logger.warning("Signal is None, returning fallback score 0")
            return 0
        
        if not isinstance(signal, dict):
            logger.warning(f"Signal is not a dictionary (type: {type(signal)}), returning fallback score 0")
            return 0
        
        symbol = signal.get('symbol', 'Unknown')
        accuracy_score = signal.get('accuracy_score', 0)
        
        logger.debug(f"Extracting fallback score for {symbol}: raw_accuracy_score={accuracy_score}")
        
        try:
            fallback_score = float(accuracy_score)
        except (ValueError, TypeError) as e:
            logger.warning(f"Error converting accuracy_score to float for {symbol}: {e}, using 0")
            return 0
        
        if not np.isfinite(fallback_score):
            logger.warning(f"Fallback score is not finite for {symbol}: {fallback_score}, using 0")
            return 0
        
        logger.debug(f"Fallback score calculated successfully for {symbol}: {fallback_score}")
        return fallback_score
        
    except Exception as e:
        logger.error(f"Unexpected error in _get_fallback_score: {e}")
        return 0
