from logger_config import logger
import pandas as pd

def _get_nearby_fibonacci_levels(df, current_price):
    """دریافت سطوح فیبوناچی نزدیک"""
    try:
        logger.debug(f"Starting Fibonacci levels calculation for current price: {current_price}")
        
        if not _validate_fibonacci_inputs(df, current_price):
            return None
        
        last_row = _get_last_row_safely(df)
        if last_row is None:
            return None
        
        fibonacci_levels = _extract_fibonacci_levels(df, last_row, current_price)
        
        result = fibonacci_levels if fibonacci_levels else None
        logger.debug(f"Found {len(fibonacci_levels) if fibonacci_levels else 0} nearby Fibonacci levels")
        return result
        
    except Exception as e:
        logger.error(f"Unexpected error in _get_nearby_fibonacci_levels: {e}")
        return None

def _validate_fibonacci_inputs(df, current_price):
    """Validate inputs for Fibonacci calculation"""
    if df is None or len(df) == 0:
        logger.warning("DataFrame is None or empty for Fibonacci calculation")
        return False
        
    if current_price is None or current_price <= 0:
        logger.warning(f"Invalid current price for Fibonacci calculation: {current_price}")
        return False
    
    return True

def _get_last_row_safely(df):
    """Get last row from DataFrame safely"""
    try:
        return df.iloc[-1]
    except (IndexError, KeyError) as e:
        logger.error(f"Error accessing last row for Fibonacci calculation: {e}")
        return None

def _extract_fibonacci_levels(df, last_row, current_price):
    """Extract valid Fibonacci levels near current price"""
    fibonacci_levels = []
    fib_keys = ['fib_236', 'fib_382', 'fib_500', 'fib_618']
    
    for fib_key in fib_keys:
        fib_level = _process_single_fibonacci_key(df, last_row, fib_key, current_price)
        if fib_level:
            fibonacci_levels.append(fib_level)
    
    return fibonacci_levels

def _process_single_fibonacci_key(df, last_row, fib_key, current_price):
    """Process a single Fibonacci key and return formatted level if valid"""
    try:
        if not _is_valid_fibonacci_key(df, last_row, fib_key):
            return None
        
        fib_level = float(last_row[fib_key])
        
        if fib_level <= 0:
            logger.debug(f"Invalid Fibonacci level for {fib_key}: {fib_level}")
            return None
        
        if _is_level_near_current_price(current_price, fib_level, fib_key):
            return f"{fib_key.replace('fib_', 'Fib ')}: {fib_level:.6f}"
        
        return None
        
    except Exception as e:
        logger.warning(f"Error processing Fibonacci key {fib_key}: {e}")
        return None

def _is_valid_fibonacci_key(df, last_row, fib_key):
    """Check if Fibonacci key is valid in DataFrame"""
    return fib_key in df.columns and not pd.isna(last_row.get(fib_key))

def _is_level_near_current_price(current_price, fib_level, fib_key):
    """Check if Fibonacci level is near current price (within 2%)"""
    try:
        price_diff_pct = abs(current_price - fib_level) / current_price * 100
        is_near = price_diff_pct < 2
        
        if is_near:
            logger.debug(f"Added Fibonacci level {fib_key}: {fib_level:.6f} (diff: {price_diff_pct:.2f}%)")
        
        return is_near
        
    except (ZeroDivisionError, TypeError, ValueError) as e:
        logger.warning(f"Error calculating price difference for {fib_key}: {e}")
        return False
