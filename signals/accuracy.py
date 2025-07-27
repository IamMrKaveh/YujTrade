from logger_config import logger
import pandas as pd

def _calculate_rsi_accuracy_score(signal_data):
    try:
        rsi = signal_data.get('rsi', 50)
        typ = signal_data['type']
        if typ == 'buy':
            if rsi < 20: return 25
            elif rsi < 25: return 20
            elif rsi < 30: return 15
            elif rsi < 35: return 10
        else:
            if rsi > 80: return 25
            elif rsi > 75: return 20
            elif rsi > 70: return 15
            elif rsi > 65: return 10
    except Exception as e:
        logger.error(f"Error calculating RSI accuracy score: {e}")
    return 0

def _calculate_volume_accuracy_score(last_row):
    if 'volume_sma' not in last_row.index or pd.isna(last_row['volume_sma']):
        logger.warning("Volume SMA not available for accuracy score calculation")
        return 0
    try:
        ratio = last_row['volume'] / last_row['volume_sma']
        if ratio > 2.5: return 15
        elif ratio > 2: return 12
        elif ratio > 1.5: return 8
        elif ratio > 1.2: return 5
    except Exception as e:
        logger.error(f"Error calculating volume accuracy score: {e}")
        return 0
    
    return 0

def _calculate_macd_accuracy_score(df, last_row, signal_data):
    try:
        if not {'MACD_12_26_9', 'MACDs_12_26_9'}.issubset(df.columns):
            logger.warning("MACD columns not available for accuracy score calculation")
            return 0
        macd = last_row.get('MACD_12_26_9')
        signal = last_row.get('MACDs_12_26_9')
        if pd.isna(macd) or pd.isna(signal): return 0
        diff = macd - signal
        typ = signal_data['type']
        if typ == 'buy' and diff > 0 and macd > signal:
            return 20
        elif typ == 'sell' and diff < 0 and macd < signal:
            return 20
        elif abs(diff) > 0.001:
            return 10
    except Exception as e:
        logger.error(f"Error calculating MACD accuracy score: {e}")
        return 0
    return 0

def _calculate_trend_accuracy_score(prev_rows, signal_data):
    try:
        if len(prev_rows) < 5: return 0
        trend = 0
        closes = prev_rows['close'].values
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]: trend += 1
            elif closes[i] < closes[i-1]: trend -= 1
        strength = abs(trend) / len(closes)
        if (signal_data['type'] == 'buy' and trend > 0) or (signal_data['type'] == 'sell' and trend < 0) :
            return int(10 * strength)
    except Exception as e:
        logger.error(f"Error calculating trend accuracy score: {e}")
        return 0
    return 0

def calculate_signal_accuracy_score(df, signal_data, symbol):
    try:
        if df is None or len(df) < 50 or not signal_data:
            return 0
        last = df.iloc[-1]
        prev_rows = df.iloc[-10:] if len(df) >= 10 else df
        score = 0
        score += _calculate_rsi_accuracy_score(signal_data)
        score += _calculate_macd_accuracy_score(df, last, signal_data)
        score += _calculate_volume_accuracy_score(last)
        score += _calculate_trend_accuracy_score(prev_rows, signal_data)

        if 'market_structure_score' in df.columns:
            m = last['market_structure_score']
            if m > 70: score += 15
            elif m > 50: score += 10
            elif m > 30: score += 5

        if symbol in ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']:
            score += 5

        dom = abs(signal_data.get('buy_score', 0) - signal_data.get('sell_score', 0))
        score += min(dom * 2, 10)

        score = min(score, 100)
        logger.info(f"Accuracy score for {symbol}: {score}")
        return score
    except Exception as e:
        logger.error(f"Error calculating accuracy score for {symbol}: {e}")
        return 0
