from logger_config import logger
import pandas as pd

def _add_additional_indicators(signal, last_row, df):
    """اضافه کردن شاخص‌های اضافی به صورت ایمن"""
    indicators_added = []
    symbol = signal.get('symbol', 'Unknown')
    
    try:
        logger.debug(f"Starting to add additional indicators for {symbol}")
        
        # Add basic indicators
        _add_basic_indicators(signal, last_row, df, symbol, indicators_added)
        
        # Add volume ratio
        _add_volume_ratio_indicator(signal, last_row, df, symbol, indicators_added)
        
        logger.debug(f"Successfully added {len(indicators_added)} indicators for {symbol}: {indicators_added}")
        
    except Exception as e:
        logger.error(f"Unexpected error in _add_additional_indicators for {symbol}: {e}")

def _add_basic_indicators(signal, last_row, df, symbol, indicators_added):
    """Add basic technical indicators to signal"""
    basic_indicators = [
        ('STOCHk_14_3_3', 'stoch_k', 'Stochastic K'),
        ('mfi', 'mfi', 'MFI'),
        ('cci', 'cci', 'CCI'),
        ('williams_r', 'williams_r', 'Williams %R')
    ]
    
    for column_name, signal_key, display_name in basic_indicators:
        _add_single_indicator(signal, last_row, df, column_name, signal_key, display_name, symbol, indicators_added)

def _add_single_indicator(signal, last_row, df, column_name, signal_key, display_name, symbol, indicators_added):
    """Add a single indicator to signal if valid"""
    try:
        if column_name in df.columns and not pd.isna(last_row.get(column_name)):
            signal[signal_key] = float(last_row[column_name])
            indicators_added.append(signal_key)
            logger.debug(f"Added {display_name} indicator for {symbol}")
    except Exception as e:
        logger.warning(f"Error adding {display_name} for {symbol}: {e}")

def _add_volume_ratio_indicator(signal, last_row, df, symbol, indicators_added):
    """Add volume ratio indicator to signal"""
    try:
        if ('volume_sma' in df.columns and 
            not pd.isna(last_row.get('volume_sma')) and 
            last_row.get('volume_sma', 0) > 0):
            try:
                volume_ratio = last_row['volume'] / last_row['volume_sma']
                signal['volume_ratio'] = float(volume_ratio)
                indicators_added.append('volume_ratio')
                logger.debug(f"Added volume ratio indicator for {symbol}")
            except (ZeroDivisionError, TypeError, ValueError) as e:
                logger.warning(f"Error calculating volume ratio for {symbol}: {e}")
    except Exception as e:
        logger.warning(f"Error processing volume ratio for {symbol}: {e}")
