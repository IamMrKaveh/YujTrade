from logger_config import logger
import pandas as pd

def _calculate_rsi_strength(last_row, signal_type):
    if 'rsi' not in last_row.index or pd.isna(last_row['rsi']):
        return 0
    rsi_value = last_row['rsi']
    if signal_type == 'buy':
        if rsi_value < 20:
            return 3
        elif rsi_value < 25:
            return 2
        elif rsi_value < 30:
            return 1
    else:
        if rsi_value > 80:
            return 3
        elif rsi_value > 75:
            return 2
        elif rsi_value > 70:
            return 1
    return 0

def _calculate_volume_strength(last_row):
    if 'volume_sma' not in last_row.index or pd.isna(last_row['volume_sma']):
        return 0
    try:
        ratio = last_row['volume'] / last_row['volume_sma']
        if ratio > 2:
            return 2
        elif ratio > 1.5:
            return 1
    except Exception as e:
        logger.error(f"Error calculating volume strength: {e}")
        return 0
    return 0

def _calculate_macd_strength(last_row):
    if 'MACD_12_26_9' not in last_row.index or pd.isna(last_row['MACD_12_26_9']):
        return 0
    macd = abs(last_row['MACD_12_26_9'])
    return 1 if macd > 0.001 else 0

def calculate_signal_strength(df, signal_type):
    try:
        if df is None or len(df) == 0:
            return 2
        last_row = df.iloc[-1]
        score = 0
        score += _calculate_rsi_strength(last_row, signal_type)
        score += _calculate_volume_strength(last_row)
        score += _calculate_macd_strength(last_row)
        return min(max(score, 1), 5)
    except Exception:
        return 2
