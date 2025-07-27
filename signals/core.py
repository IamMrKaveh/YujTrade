from logger_config import logger
import pandas as pd

from signals.accuracy import calculate_signal_accuracy_score

def _create_signal_result(signal_results, last_row, rsi_value, df=None, symbol=None):
    buy_signals = signal_results['buy_signals']
    sell_signals = signal_results['sell_signals']
    pattern_signals = signal_results.get('pattern_signals', 0)
    
    min_signal_threshold = max(3 - pattern_signals, 1)

    if buy_signals >= min_signal_threshold and buy_signals > sell_signals:
        signal_data = {
            'type': 'buy',
            'strength': min(signal_results['signal_strength'], 5),
            'rsi': rsi_value,
            'macd': last_row.get('MACD_12_26_9', 0),
            'method': 'Advanced_Multi_Indicator_Buy',
            'details': signal_results['signal_details'],
            'buy_score': buy_signals,
            'sell_score': sell_signals,
            'pattern_signals': pattern_signals
        }
        if df is not None and symbol is not None:
            signal_data['accuracy_score'] = calculate_signal_accuracy_score(df, signal_data, symbol)
        return signal_data

    elif sell_signals >= min_signal_threshold and sell_signals > buy_signals:
        signal_data = {
            'type': 'sell',
            'strength': min(signal_results['signal_strength'], 5),
            'rsi': rsi_value,
            'macd': last_row.get('MACD_12_26_9', 0),
            'method': 'Advanced_Multi_Indicator_Sell',
            'details': signal_results['signal_details'],
            'buy_score': buy_signals,
            'sell_score': sell_signals,
            'pattern_signals': pattern_signals
        }
        if df is not None and symbol is not None:
            signal_data['accuracy_score'] = calculate_signal_accuracy_score(df, signal_data, symbol)
        return signal_data

    return None

def check_signals(df, symbol):
    if df is None or len(df) < 2:
        return None

    try:
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        if 'rsi' not in df.columns or pd.isna(last_row['rsi']):
            logger.warning(f"No RSI data for {symbol}")
            return None

        rsi_value = last_row['rsi']
        current_price = last_row['close']

        signal_results = {
            'buy_signals': 0,
            'sell_signals': 0,
            'signal_strength': 0,
            'signal_details': [],
            'pattern_signals': 0
        }

        # _analyze_rsi(last_row, signal_results)
        # _analyze_macd(df, last_row, prev_row, signal_results)
        # _analyze_bollinger_bands(df, last_row, current_price, signal_results)
        # _analyze_oscillators(df, last_row, signal_results)
        # _analyze_trend_indicators(df, last_row, prev_row, current_price, signal_results)
        # _analyze_patterns(df, last_row, signal_results)
        # _analyze_support_resistance(df, last_row, current_price, signal_results)

        return _create_signal_result(signal_results, last_row, rsi_value, df, symbol)

    except Exception as e:
        logger.error(f"Error checking signals for {symbol}: {e}")
        return None
