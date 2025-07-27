from logger_config import logger
from .scoring import _calculate_combined_score

import numpy as np

# Fix numpy compatibility issue
if not hasattr(np, 'NaN'):
    np.NaN = np.nan



def _process_and_return_best_signal(all_signals, analysis_stats):
    """پردازش و بازگرداندن بهترین سیگنال"""
    try:
        valid_signals = [sig for sig in all_signals if _validate_signal(sig)]
        
        if not valid_signals:
            _log_no_valid_signals(all_signals)
            return []
        
        # مرتب‌سازی و انتخاب بهترین
        valid_signals.sort(key=_calculate_combined_score, reverse=True)
        best_signal = valid_signals[0]
        
        _log_analysis_complete(analysis_stats, valid_signals, best_signal)
        return [best_signal]
        
    except Exception as e:
        logger.error(f"Error in signal processing: {e}")
        return []

def _log_no_valid_signals(all_signals):
    """لاگ کردن اطلاعات در صورت نبود سیگنال معتبر"""
    logger.info(f"No valid signals found from {len(all_signals)} analyzed signals")
    if all_signals:
        scores = [sig.get('accuracy_score', 0) for sig in all_signals if sig.get('accuracy_score')]
        if scores:
            logger.info(f"Signal scores range: {min(scores):.1f} - {max(scores):.1f}, Average: {sum(scores)/len(scores):.1f}")

def _log_analysis_complete(analysis_stats, valid_signals, best_signal):
    """لاگ کردن تکمیل تحلیل"""
    logger.info(f"Analysis complete. Success: {analysis_stats['successful']}, Failed: {analysis_stats['failed']}, "
                f"Valid signals: {len(valid_signals)}, Best signal: {best_signal.get('symbol', 'Unknown')} "
                f"(Score: {best_signal.get('accuracy_score', 0)})")

def _validate_signal(signal):
    """اعتبارسنجی سیگنال"""
    if not signal or not isinstance(signal, dict):
        logger.debug("Signal validation failed: signal is None or not a dict")
        return False
    
    if not _validate_required_fields(signal):
        logger.debug(f"Signal validation failed: missing required fields for {signal.get('symbol', 'Unknown')}")
        return False
    
    if not _validate_price_logic(signal):
        logger.debug(f"Signal validation failed: invalid price logic for {signal.get('symbol', 'Unknown')}")
        return False
    
    return True
    
def _validate_required_fields(signal):
    """بررسی وجود و صحت فیلدهای ضروری"""
    try:
        required_fields = ['symbol', 'type', 'entry', 'target', 'stop_loss', 'accuracy_score']
        missing_fields = []
        invalid_fields = []
        
        for field in required_fields:
            if field not in signal:
                missing_fields.append(field)
            elif field in ['entry', 'target', 'stop_loss', 'accuracy_score'] and not _validate_numeric_field(signal, field):
                invalid_fields.append(field)
        
        is_valid = len(missing_fields) == 0 and len(invalid_fields) == 0
        
        if not is_valid:
            logger.debug(f"Signal validation failed: missing fields {missing_fields}, invalid fields {invalid_fields}")
            return False
        
        logger.debug(f"Signal validation passed for {signal.get('symbol', 'Unknown')}")
        return True
        
    except Exception as e:
        logger.error(f"Error validating required fields for signal: {e}")
        return False

def _validate_numeric_field(signal, field):
    """بررسی صحت فیلدهای عددی"""
    try:
        if not isinstance(signal, dict):
            logger.error(f"Signal is not a dictionary for field {field}")
            return False
            
        if field not in ['entry', 'target', 'stop_loss', 'accuracy_score']:
            logger.debug(f"Field {field} is not numeric, skipping validation")
            return True
        
        if field not in signal:
            logger.warning(f"Numeric field {field} is missing from signal")
            return False
        
        try:
            value = float(signal[field])
        except (ValueError, TypeError) as e:
            logger.warning(f"Cannot convert field {field} to float: {signal[field]}, error: {e}")
            return False
        
        # Check for valid positive number
        if not np.isfinite(value) or value <= 0:
            logger.debug(f"Numeric field {field} validation failed: value={value} (not finite or not positive)")
            return False
        
        logger.debug(f"Numeric field {field} validation passed: value={value}")
        return True
        
    except Exception as e:
        logger.error(f"Unexpected error validating numeric field {field}: {e}")
        return False
    
def _validate_price_logic(signal):
    """بررسی منطقی قیمت‌ها"""
    try:
        if not signal or not isinstance(signal, dict):
            logger.warning("Signal is None or not a dictionary for price logic validation")
            return False
        
        symbol = signal.get('symbol', 'Unknown')
        signal_type = signal.get('type')
        
        # Check if signal type is valid
        if signal_type not in ['Long', 'Short']:
            logger.warning(f"Invalid signal type '{signal_type}' for {symbol}")
            return False
        
        # Extract and validate price values
        try:
            entry = float(signal['entry'])
            target = float(signal['target'])
            stop_loss = float(signal['stop_loss'])
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error extracting price values for {symbol}: {e}")
            return False
        
        # Check for valid positive numbers
        if not all(price > 0 and np.isfinite(price) for price in [entry, target, stop_loss]):
            logger.warning(f"Invalid price values for {symbol}: entry={entry}, target={target}, stop_loss={stop_loss}")
            return False
        
        # Validate price logic based on signal type
        if signal_type == 'Long':
            is_valid = target > entry > stop_loss
            if not is_valid:
                logger.warning(f"Long signal price logic failed for {symbol}: target({target}) > entry({entry}) > stop_loss({stop_loss})")
        else:  # Short
            is_valid = target < entry < stop_loss
            if not is_valid:
                logger.warning(f"Short signal price logic failed for {symbol}: target({target}) < entry({entry}) < stop_loss({stop_loss})")
        
        if is_valid:
            logger.debug(f"Price logic validation passed for {symbol} ({signal_type})")
        
        return is_valid
        
    except Exception as e:
        logger.error(f"Unexpected error in price logic validation: {e}")
        return False

