import logging
import warnings
from datetime import datetime
import pandas as pd
from indicators import *
from market import *
import sys

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
    

# Now import pandas_ta after fixing numpy
try:
    import pandas_ta as ta
except ImportError as e:
    print(f"Error importing pandas_ta: {e}")
    print("Please install with: pip install pandas-ta==0.3.14b")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def _analyze_rsi(last_row, signal_results):
    """Analyze RSI indicator for signals"""
    if 'rsi' not in last_row.index or pd.isna(last_row['rsi']):
        return
    
    rsi_value = last_row['rsi']
    if rsi_value < 30:
        signal_results['buy_signals'] += 2
        signal_results['signal_strength'] += 2
        signal_results['signal_details'].append(f"RSI Oversold: {rsi_value:.1f}")
    elif rsi_value > 70:
        signal_results['sell_signals'] += 2
        signal_results['signal_strength'] += 2
        signal_results['signal_details'].append(f"RSI Overbought: {rsi_value:.1f}")

def _analyze_macd(df, last_row, prev_row, signal_results):
    """Analyze MACD indicator for signals"""
    required_cols = ['MACD_12_26_9', 'MACDs_12_26_9']
    if not all(col in df.columns for col in required_cols):
        return
    
    macd_values = [last_row[col] for col in required_cols]
    prev_macd_values = [prev_row[col] for col in required_cols]
    
    if any(pd.isna(val) for val in macd_values + prev_macd_values):
        return
    
    macd_bullish = (prev_row['MACD_12_26_9'] <= prev_row['MACDs_12_26_9'] and 
                   last_row['MACD_12_26_9'] > last_row['MACDs_12_26_9'])
    
    macd_bearish = (prev_row['MACD_12_26_9'] >= prev_row['MACDs_12_26_9'] and 
                   last_row['MACD_12_26_9'] < last_row['MACDs_12_26_9'])
    
    if macd_bullish:
        signal_results['buy_signals'] += 3
        signal_results['signal_strength'] += 3
        signal_results['signal_details'].append("MACD Bullish Cross")
    elif macd_bearish:
        signal_results['sell_signals'] += 3
        signal_results['signal_strength'] += 3
        signal_results['signal_details'].append("MACD Bearish Cross")

def _analyze_bollinger_bands(df, last_row, current_price, signal_results):
    """Analyze Bollinger Bands for signals"""
    required_cols = ['BBL_20_2.0', 'BBU_20_2.0']
    if not all(col in df.columns for col in required_cols):
        return
    
    bb_values = [last_row[col] for col in required_cols]
    if any(pd.isna(val) for val in bb_values):
        return
    
    if current_price <= last_row['BBL_20_2.0']:
        signal_results['buy_signals'] += 2
        signal_results['signal_strength'] += 2
        signal_results['signal_details'].append("Price at Lower Bollinger Band")
    elif current_price >= last_row['BBU_20_2.0']:
        signal_results['sell_signals'] += 2
        signal_results['signal_strength'] += 2
        signal_results['signal_details'].append("Price at Upper Bollinger Band")

def _analyze_stochastic(df, last_row, signal_results):
    """Analyze Stochastic oscillator"""
    if not all(col in df.columns for col in ['STOCHk_14_3_3', 'STOCHd_14_3_3']):
        return
    
    stoch_k = last_row.get('STOCHk_14_3_3')
    stoch_d = last_row.get('STOCHd_14_3_3')
    
    if pd.isna(stoch_k) or pd.isna(stoch_d):
        return
    
    if stoch_k < 20 and stoch_d < 20:
        signal_results['buy_signals'] += 2
        signal_results['signal_strength'] += 1
        signal_results['signal_details'].append(f"Stochastic Oversold: K={stoch_k:.1f}")
    elif stoch_k > 80 and stoch_d > 80:
        signal_results['sell_signals'] += 2
        signal_results['signal_strength'] += 1
        signal_results['signal_details'].append(f"Stochastic Overbought: K={stoch_k:.1f}")

def _analyze_mfi(df, last_row, signal_results):
    """Analyze Money Flow Index"""
    if 'mfi' not in df.columns or pd.isna(last_row.get('mfi')):
        return
    
    mfi_value = last_row['mfi']
    if mfi_value < 20:
        signal_results['buy_signals'] += 2
        signal_results['signal_strength'] += 1
        signal_results['signal_details'].append(f"MFI Oversold: {mfi_value:.1f}")
    elif mfi_value > 80:
        signal_results['sell_signals'] += 2
        signal_results['signal_strength'] += 1
        signal_results['signal_details'].append(f"MFI Overbought: {mfi_value:.1f}")

def _analyze_cci(df, last_row, signal_results):
    """Analyze Commodity Channel Index"""
    if 'cci' not in df.columns or pd.isna(last_row.get('cci')):
        return
    
    cci_value = last_row['cci']
    if cci_value < -100:
        signal_results['buy_signals'] += 1
        signal_results['signal_details'].append(f"CCI Oversold: {cci_value:.1f}")
    elif cci_value > 100:
        signal_results['sell_signals'] += 1
        signal_results['signal_details'].append(f"CCI Overbought: {cci_value:.1f}")

def _analyze_oscillators(df, last_row, signal_results):
    """Analyze various oscillator indicators"""
    _analyze_stochastic(df, last_row, signal_results)
    _analyze_mfi(df, last_row, signal_results)
    _analyze_cci(df, last_row, signal_results)

def _analyze_trend_indicators(df, last_row, prev_row, current_price, signal_results):
    """Analyze trend-following indicators"""
    # SuperTrend
    if 'supertrend' in df.columns and not pd.isna(last_row.get('supertrend')):
        supertrend_value = last_row['supertrend']
        prev_supertrend = df.iloc[-2].get('supertrend', 0)
        
        if current_price > supertrend_value and prev_row['close'] <= prev_supertrend:
            signal_results['buy_signals'] += 2
            signal_results['signal_strength'] += 2
            signal_results['signal_details'].append("SuperTrend Bullish Signal")
        elif current_price < supertrend_value and prev_row['close'] >= prev_supertrend:
            signal_results['sell_signals'] += 2
            signal_results['signal_strength'] += 2
            signal_results['signal_details'].append("SuperTrend Bearish Signal")
    
    # ADX
    if 'adx' in df.columns and not pd.isna(last_row.get('adx')):
        adx_value = last_row['adx']
        if adx_value > 25:
            signal_results['signal_strength'] += 1
            signal_results['signal_details'].append(f"Strong Trend (ADX: {adx_value:.1f})")

def _analyze_patterns(df, last_row, signal_results):
    """Analyze candlestick patterns"""
    pattern_signals = 0
    
    pattern_configs = [
        ('hammer', 1, 'Bullish Hammer Pattern', 'buy'),
        ('bullish_engulfing', 2, 'Bullish Engulfing Pattern', 'buy'),
        ('morning_star', 2, 'Morning Star Pattern', 'buy'),
        ('bearish_engulfing', 2, 'Bearish Engulfing Pattern', 'sell'),
        ('evening_star', 2, 'Evening Star Pattern', 'sell')
    ]
    
    for pattern_col, strength, description, signal_type in pattern_configs:
        if pattern_col in df.columns and last_row.get(pattern_col, False):
            if signal_type == 'buy':
                signal_results['buy_signals'] += strength
            else:
                signal_results['sell_signals'] += strength
            pattern_signals += strength
            signal_results['signal_details'].append(description)
    
    if 'doji' in df.columns and last_row.get('doji', False):
        signal_results['signal_details'].append("Doji Pattern (Indecision)")
    
    signal_results['pattern_signals'] = pattern_signals

def _analyze_support_resistance(df, last_row, current_price, signal_results):
    """Analyze support and resistance levels"""
    if 'support' in df.columns and 'resistance' in df.columns:
        support_level = last_row.get('support')
        resistance_level = last_row.get('resistance')
        
        if support_level and abs(current_price - support_level) / current_price < 0.005:
            signal_results['buy_signals'] += 1
            signal_results['signal_details'].append(f"Near Support: {support_level:.6f}")
        
        if resistance_level and abs(current_price - resistance_level) / current_price < 0.005:
            signal_results['sell_signals'] += 1
            signal_results['signal_details'].append(f"Near Resistance: {resistance_level:.6f}")

def _create_signal_result(signal_results, last_row, rsi_value):
    """Create the final signal result"""
    buy_signals = signal_results['buy_signals']
    sell_signals = signal_results['sell_signals']
    pattern_signals = signal_results.get('pattern_signals', 0)
    
    min_signal_threshold = max(3 - pattern_signals, 1)
    
    if buy_signals >= min_signal_threshold and buy_signals > sell_signals:
        return {
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
    elif sell_signals >= min_signal_threshold and sell_signals > buy_signals:
        return {
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
        
        # Analyze different indicator groups
        _analyze_rsi(last_row, signal_results)
        _analyze_macd(df, last_row, prev_row, signal_results)
        _analyze_bollinger_bands(df, last_row, current_price, signal_results)
        _analyze_oscillators(df, last_row, signal_results)
        _analyze_trend_indicators(df, last_row, prev_row, current_price, signal_results)
        _analyze_patterns(df, last_row, signal_results)
        _analyze_support_resistance(df, last_row, current_price, signal_results)
        
        return _create_signal_result(signal_results, last_row, rsi_value)
        
    except Exception as e:
        logger.error(f"Error checking signals for {symbol}: {e}")
        return None

def _calculate_rsi_strength(last_row, signal_type):
    """Calculate RSI contribution to signal strength"""
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
    else:  # sell
        if rsi_value > 80:
            return 3
        elif rsi_value > 75:
            return 2
        elif rsi_value > 70:
            return 1
    return 0

def _calculate_volume_strength(last_row):
    """Calculate volume contribution to signal strength"""
    if 'volume_sma' not in last_row.index or pd.isna(last_row['volume_sma']):
        return 0
    
    try:
        volume_ratio = last_row['volume'] / last_row['volume_sma']
        if volume_ratio > 2:
            return 2
        elif volume_ratio > 1.5:
            return 1
    except (ZeroDivisionError, TypeError):
        pass
    return 0

def _calculate_macd_strength(last_row):
    """Calculate MACD contribution to signal strength"""
    if 'MACD_12_26_9' not in last_row.index or pd.isna(last_row['MACD_12_26_9']):
        return 0
    
    macd_value = abs(last_row['MACD_12_26_9'])
    return 1 if macd_value > 0.001 else 0

def calculate_signal_strength(df, signal_type):
    """Calculate signal strength based on multiple factors"""
    try:
        if df is None or len(df) == 0:
            return 2
            
        last_row = df.iloc[-1]
        strength_score = 0
        
        strength_score += _calculate_rsi_strength(last_row, signal_type)
        strength_score += _calculate_volume_strength(last_row)
        strength_score += _calculate_macd_strength(last_row)
        
        # Normalize to 1-5 scale
        return min(max(strength_score, 1), 5)
        
    except Exception:
        return 2  # Default medium strength

def _calculate_rsi_accuracy_score(signal_data):
    """Calculate RSI contribution to accuracy score"""
    rsi_value = signal_data.get('rsi', 50)
    signal_type = signal_data['type']
    
    if signal_type == 'buy':
        if rsi_value < 20:
            return 25
        elif rsi_value < 25:
            return 20
        elif rsi_value < 30:
            return 15
        elif rsi_value < 35:
            return 10
    else:  # sell
        if rsi_value > 80:
            return 25
        elif rsi_value > 75:
            return 20
        elif rsi_value > 70:
            return 15
        elif rsi_value > 65:
            return 10
    return 0

def _calculate_macd_accuracy_score(df, last_row, signal_data):
    """Calculate MACD contribution to accuracy score"""
    if not ('MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns):
        return 0
    
    macd_line = last_row.get('MACD_12_26_9')
    signal_line = last_row.get('MACDs_12_26_9')
    
    if pd.isna(macd_line) or pd.isna(signal_line):
        return 0
    
    macd_histogram = macd_line - signal_line
    signal_type = signal_data['type']
    
    if signal_type == 'buy' and macd_histogram > 0 and macd_line > signal_line:
        return 20
    elif signal_type == 'sell' and macd_histogram < 0 and macd_line < signal_line:
        return 20
    elif abs(macd_histogram) > 0.001:
        return 10
    return 0

def _calculate_volume_accuracy_score(last_row):
    """Calculate volume contribution to accuracy score"""
    if 'volume_sma' not in last_row.index or pd.isna(last_row.get('volume_sma')):
        return 0
    
    try:
        volume_ratio = last_row['volume'] / last_row['volume_sma']
        if volume_ratio > 2.5:
            return 15
        elif volume_ratio > 2:
            return 12
        elif volume_ratio > 1.5:
            return 8
        elif volume_ratio > 1.2:
            return 5
    except (ZeroDivisionError, TypeError):
        pass
    return 0

def _calculate_sma_accuracy_score(df, last_row, signal_data):
    """Calculate SMA trend contribution to accuracy score"""
    if not all(col in df.columns for col in ['sma20', 'sma50', 'sma200']):
        return 0
    
    current_price = last_row['close']
    sma20 = last_row.get('sma20')
    sma50 = last_row.get('sma50')
    sma200 = last_row.get('sma200')
    
    if any(pd.isna(val) for val in [sma20, sma50, sma200]):
        return 0
    
    signal_type = signal_data['type']
    
    if signal_type == 'buy':
        if current_price > sma20 > sma50 > sma200:
            return 15
        elif current_price > sma20 > sma50:
            return 10
        elif current_price > sma20:
            return 5
    else:  # sell
        if current_price < sma20 < sma50 < sma200:
            return 15
        elif current_price < sma20 < sma50:
            return 10
        elif current_price < sma20:
            return 5
    return 0

def _calculate_stochastic_score(df, last_row, signal_type):
    """Calculate Stochastic oscillator score"""
    if not ('STOCHk_14_3_3' in df.columns and 'STOCHd_14_3_3' in df.columns):
        return 0
    
    k_value = last_row.get('STOCHk_14_3_3')
    d_value = last_row.get('STOCHd_14_3_3')
    
    if pd.isna(k_value) or pd.isna(d_value):
        return 0
    
    if signal_type == 'buy':
        if k_value < 20 and d_value < 20:
            return 10
        elif k_value < 30:
            return 5
    else:  # sell
        if k_value > 80 and d_value > 80:
            return 10
        elif k_value > 70:
            return 5
    
    return 0

def _calculate_mfi_score(df, last_row, signal_type):
    """Calculate MFI oscillator score"""
    if 'mfi' not in df.columns or pd.isna(last_row.get('mfi')):
        return 0
    
    mfi_value = last_row['mfi']
    
    if signal_type == 'buy':
        if mfi_value < 20:
            return 8
        elif mfi_value < 30:
            return 4
    else:  # sell
        if mfi_value > 80:
            return 8
        elif mfi_value > 70:
            return 4
    
    return 0

def _calculate_cci_score(df, last_row, signal_type):
    """Calculate CCI oscillator score"""
    if 'cci' not in df.columns or pd.isna(last_row.get('cci')):
        return 0
    
    cci_value = last_row['cci']
    
    if signal_type == 'buy' and cci_value < -100:
        return 5
    if signal_type == 'sell' and cci_value > 100:
        return 5
    
    return 0

def _calculate_oscillator_accuracy_score(df, last_row, signal_data):
    """Calculate oscillator indicators contribution to accuracy score"""
    signal_type = signal_data['type']
    
    score = _calculate_stochastic_score(df, last_row, signal_type)
    score += _calculate_mfi_score(df, last_row, signal_type)
    score += _calculate_cci_score(df, last_row, signal_type)
    
    return score

def _calculate_trend_accuracy_score(prev_rows, signal_data):
    """Calculate trend direction contribution to accuracy score"""
    if len(prev_rows) < 5:
        return 0
    
    trend_direction = 0
    close_prices = prev_rows['close'].values
    
    for i in range(1, len(close_prices)):
        if close_prices[i] > close_prices[i-1]:
            trend_direction += 1
        elif close_prices[i] < close_prices[i-1]:
            trend_direction -= 1
    
    trend_strength = abs(trend_direction) / len(close_prices)
    signal_type = signal_data['type']
    
    if signal_type == 'buy' and trend_direction > 0:
        return int(10 * trend_strength)
    if signal_type == 'sell' and trend_direction < 0:
        return int(10 * trend_strength)
    return 0

def calculate_signal_accuracy_score(df, signal_data, symbol):
    try:
        if df is None or len(df) < 50 or not signal_data:
            return 0
        
        last_row = df.iloc[-1]
        prev_rows = df.iloc[-10:] if len(df) >= 10 else df
        accuracy_score = 0
        
        # Calculate individual component scores
        accuracy_score += _calculate_rsi_accuracy_score(signal_data)
        accuracy_score += _calculate_macd_accuracy_score(df, last_row, signal_data)
        accuracy_score += _calculate_volume_accuracy_score(last_row)
        accuracy_score += _calculate_sma_accuracy_score(df, last_row, signal_data)
        accuracy_score += _calculate_oscillator_accuracy_score(df, last_row, signal_data)
        accuracy_score += _calculate_trend_accuracy_score(prev_rows, signal_data)
        
        # Market structure score
        if 'market_structure_score' in df.columns:
            structure_score = last_row['market_structure_score']
            if structure_score > 70:
                accuracy_score += 15
            elif structure_score > 50:
                accuracy_score += 10
            elif structure_score > 30:
                accuracy_score += 5
        
        # Symbol bonus
        if symbol in ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']:
            accuracy_score += 5
        
        # Signal dominance
        buy_score = signal_data.get('buy_score', 0)
        sell_score = signal_data.get('sell_score', 0)
        signal_dominance = max(buy_score, sell_score) - min(buy_score, sell_score)
        accuracy_score += min(signal_dominance * 2, 10)

        accuracy_score = min(accuracy_score, 100)
        
        logger.info(f"Accuracy score for {symbol}: {accuracy_score}")
        return accuracy_score
        
    except Exception as e:
        logger.error(f"Error calculating accuracy score for {symbol}: {e}")
        return 0
