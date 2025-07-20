import os
import logging
import asyncio
import warnings
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import pandas as pd
import numpy as np
import talib
import ccxt.async_support as ccxt


# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Fix numpy compatibility issue
import sys
if hasattr(np, 'NaN'):
    pass
else:
    np.NaN = np.nan

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

# Telegram bot token
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '7873285133:AAHOo3L7DewCgdVZbnx8Gs4xDJRnHs_R2VI')

# Global exchange instance
exchange = None

def load_symbols():
    """Load symbols from file with error handling"""
    try:
        with open('symbols.txt', 'r', encoding='utf-8') as f:
            symbols = [line.strip().upper() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(symbols)} symbols from symbols.txt")
        return symbols
    except FileNotFoundError:
        logger.error("symbols.txt file not found. Using default symbols.")
        # Create default symbols.txt file
        default_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT', 
                            'DOT/USDT', 'LINK/USDT', 'XRP/USDT', 'LTC/USDT', 'MATIC/USDT']
        try:
            with open('symbols.txt', 'w', encoding='utf-8') as f:
                for symbol in default_symbols:
                    f.write(f"{symbol}\n")
            logger.info("Created default symbols.txt file")
        except Exception as e:
            logger.error(f"Could not create symbols.txt: {e}")
        return default_symbols
    except Exception as e:
        logger.error(f"Error loading symbols: {e}")
        return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']

# Load symbols at startup
SYMBOLS = load_symbols()

async def init_exchange():
    """Initialize exchange connection"""
    global exchange
    if exchange is None:
        try:
            exchange = ccxt.coinex({
                'apiKey': os.getenv('COINEX_API_KEY', ''),
                'secret': os.getenv('COINEX_SECRET', ''),
                'sandbox': False,
                'enableRateLimit': True,
                'timeout': 30000,  # 30 seconds timeout
                'options': {'defaultType': 'spot'}
            })
            logger.info("Exchange initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing exchange: {e}")
            return None
    return exchange

async def get_klines(symbol, interval='1h', limit=300):
    """Fetch klines data with improved error handling"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            exchange = await init_exchange()
            if exchange is None:
                return None
            
            # Validate symbol format
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
            
            # Fetch data with timeout
            ohlcv = await asyncio.wait_for(
                exchange.fetch_ohlcv(symbol, interval, limit=limit),
                timeout=15
            )
            
            if not ohlcv or len(ohlcv) < 50:
                logger.warning(f"Insufficient data for {symbol}: {len(ohlcv) if ohlcv else 0} candles")
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert to numeric types to avoid calculation issues
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any NaN values
            df = df.dropna()
            
            if len(df) < 50:
                logger.warning(f"Insufficient clean data for {symbol}: {len(df)} candles")
                return None
                
            logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
            return df
            
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching klines for {symbol}, attempt {attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching klines for {symbol}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching klines for {symbol}: {e}")
            break  # Don't retry exchange errors
        except Exception as e:
            logger.error(f"Unexpected error fetching klines for {symbol}: {e}")
            break
    
    return None

async def get_current_price(symbol):
    """Fetch current price with improved error handling"""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            exchange = await init_exchange()
            if exchange is None:
                return None
            
            # Validate symbol format
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
                
            ticker = await asyncio.wait_for(
                exchange.fetch_ticker(symbol),
                timeout=10
            )
            
            if ticker and 'last' in ticker and ticker['last'] is not None:
                return float(ticker['last'])
            else:
                logger.warning(f"No valid price data for {symbol}")
                return None
                
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching price for {symbol}, attempt {attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
    
    return None

def safe_indicator_calculation(df, func, *args, **kwargs):
    """Safely calculate indicators with error handling"""
    try:
        result = func(*args, **kwargs)
        if result is not None:
            if isinstance(result, pd.DataFrame):
                return result
            elif isinstance(result, pd.Series):
                return result
        return None
    except Exception as e:
        logger.warning(f"Error calculating indicator: {e}")
        return None

def calculate_fibonacci_levels(df, lookback=100):
    try:
        if df is None or len(df) < lookback:
            return None
        
        recent_data = df.tail(lookback)
        high_price = recent_data['high'].max()
        low_price = recent_data['low'].min()
        
        diff = high_price - low_price
        
        fib_levels = {
            'fib_0': high_price,
            'fib_236': high_price - (diff * 0.236),
            'fib_382': high_price - (diff * 0.382),
            'fib_500': high_price - (diff * 0.5),
            'fib_618': high_price - (diff * 0.618),
            'fib_786': high_price - (diff * 0.786),
            'fib_100': low_price
        }
        
        return fib_levels
    except Exception:
        return None

def calculate_parabolic_sar(df, af=0.02, max_af=0.2):
    try:
        if df is None or len(df) < 5:
            return None
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        sar = np.zeros(len(df))
        trend = np.zeros(len(df))
        af_val = np.zeros(len(df))
        ep = np.zeros(len(df))
        
        sar[0] = low[0]
        trend[0] = 1
        af_val[0] = af
        ep[0] = high[0]
        
        for i in range(1, len(df)):
            if trend[i-1] == 1:
                sar[i] = sar[i-1] + af_val[i-1] * (ep[i-1] - sar[i-1])
                
                if low[i] <= sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = low[i]
                    af_val[i] = af
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af_val[i] = min(af_val[i-1] + af, max_af)
                    else:
                        ep[i] = ep[i-1]
                        af_val[i] = af_val[i-1]
            else:
                sar[i] = sar[i-1] - af_val[i-1] * (sar[i-1] - ep[i-1])
                
                if high[i] >= sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = high[i]
                    af_val[i] = af
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af_val[i] = min(af_val[i-1] + af, max_af)
                    else:
                        ep[i] = ep[i-1]
                        af_val[i] = af_val[i-1]
        
        return pd.Series(sar, index=df.index, name='psar')
    except Exception:
        return None

def calculate_ichimoku(df):
    try:
        if df is None or len(df) < 52:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        
        chikou_span = close.shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    except Exception:
        return None

def calculate_money_flow_index(df, period=14):
    try:
        if df is None or len(df) < period + 1:
            return None
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        price_diff = typical_price.diff()
        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)
        
        positive_flow[price_diff > 0] = money_flow[price_diff > 0]
        negative_flow[price_diff < 0] = money_flow[price_diff < 0]
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        money_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    except Exception:
        return None

def calculate_commodity_channel_index(df, period=20):
    try:
        if df is None or len(df) < period:
            return None
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    except Exception:
        return None

def calculate_williams_r(df, period=14):
    try:
        if df is None or len(df) < period:
            return None
        
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
        
        return williams_r
    except Exception:
        return None

def calculate_indicators(df):
    try:
        if df is None or len(df) < 200:
            logger.warning(f"Insufficient data for indicators: {len(df) if df is not None else 0} candles")
            return None
        
        df = df.copy()
        
        df['sma20'] = safe_indicator_calculation(df, ta.sma, df['close'], length=20)
        df['sma50'] = safe_indicator_calculation(df, ta.sma, df['close'], length=50)
        df['sma200'] = safe_indicator_calculation(df, ta.sma, df['close'], length=200)
        
        df['ema12'] = safe_indicator_calculation(df, ta.ema, df['close'], length=12)
        df['ema26'] = safe_indicator_calculation(df, ta.ema, df['close'], length=26)
        df['ema50'] = safe_indicator_calculation(df, ta.ema, df['close'], length=50)
        
        df['wma20'] = safe_indicator_calculation(df, ta.wma, df['close'], length=20)
        
        rsi = safe_indicator_calculation(df, ta.rsi, df['close'], length=14)
        if rsi is not None:
            df['rsi'] = rsi
        
        try:
            macd_data = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd_data is not None:
                df = df.join(macd_data, how='left')
        except Exception as e:
            logger.warning(f"Error calculating MACD: {e}")
        
        try:
            bbands_data = ta.bbands(df['close'], length=20, std=2)
            if bbands_data is not None:
                df = df.join(bbands_data, how='left')
        except Exception as e:
            logger.warning(f"Error calculating Bollinger Bands: {e}")
        
        try:
            stoch_data = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
            if stoch_data is not None:
                df = df.join(stoch_data, how='left')
        except Exception as e:
            logger.warning(f"Error calculating Stochastic: {e}")
        
        volume_sma = safe_indicator_calculation(df, ta.sma, df['volume'], length=20)
        if volume_sma is not None:
            df['volume_sma'] = volume_sma
        
        mfi = calculate_money_flow_index(df)
        if mfi is not None:
            df['mfi'] = mfi
        
        cci = calculate_commodity_channel_index(df)
        if cci is not None:
            df['cci'] = cci
        
        williams_r = calculate_williams_r(df)
        if williams_r is not None:
            df['williams_r'] = williams_r
        
        psar = calculate_parabolic_sar(df)
        if psar is not None:
            df['psar'] = psar
        
        ichimoku_data = calculate_ichimoku(df)
        if ichimoku_data:
            for key, value in ichimoku_data.items():
                if value is not None:
                    df[key] = value
        
        fib_levels = calculate_fibonacci_levels(df)
        if fib_levels:
            current_price = df['close'].iloc[-1]
            for level_name, level_value in fib_levels.items():
                df[level_name] = level_value
        
        required_indicators = ['rsi', 'sma50', 'volume_sma']
        missing_indicators = [ind for ind in required_indicators if ind not in df.columns or df[ind].isna().all()]
        
        if missing_indicators:
            logger.warning(f"Missing indicators: {missing_indicators}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
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
        
        buy_signals = 0
        sell_signals = 0
        signal_strength = 0
        signal_details = []
        
        if rsi_value < 30:
            buy_signals += 2
            signal_strength += 2
            signal_details.append(f"RSI Oversold: {rsi_value:.1f}")
        elif rsi_value > 70:
            sell_signals += 2
            signal_strength += 2
            signal_details.append(f"RSI Overbought: {rsi_value:.1f}")
        
        if ('MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns and
            not pd.isna(last_row['MACD_12_26_9']) and not pd.isna(last_row['MACDs_12_26_9']) and
            not pd.isna(prev_row['MACD_12_26_9']) and not pd.isna(prev_row['MACDs_12_26_9'])):
            
            macd_bullish = (prev_row['MACD_12_26_9'] <= prev_row['MACDs_12_26_9'] and 
                           last_row['MACD_12_26_9'] > last_row['MACDs_12_26_9'])
            
            macd_bearish = (prev_row['MACD_12_26_9'] >= prev_row['MACDs_12_26_9'] and 
                           last_row['MACD_12_26_9'] < last_row['MACDs_12_26_9'])
            
            if macd_bullish:
                buy_signals += 3
                signal_strength += 3
                signal_details.append("MACD Bullish Cross")
            elif macd_bearish:
                sell_signals += 3
                signal_strength += 3
                signal_details.append("MACD Bearish Cross")
        
        if ('BBL_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns and
            not pd.isna(last_row['BBL_20_2.0']) and not pd.isna(last_row['BBU_20_2.0'])):
            
            if current_price <= last_row['BBL_20_2.0']:
                buy_signals += 2
                signal_strength += 2
                signal_details.append("Price at Lower Bollinger Band")
            elif current_price >= last_row['BBU_20_2.0']:
                sell_signals += 2
                signal_strength += 2
                signal_details.append("Price at Upper Bollinger Band")
        
        if ('STOCHk_14_3_3' in df.columns and 'STOCHd_14_3_3' in df.columns and
            not pd.isna(last_row['STOCHk_14_3_3']) and not pd.isna(last_row['STOCHd_14_3_3'])):
            
            stoch_k = last_row['STOCHk_14_3_3']
            stoch_d = last_row['STOCHd_14_3_3']
            
            if stoch_k < 20 and stoch_d < 20:
                buy_signals += 2
                signal_strength += 1
                signal_details.append(f"Stochastic Oversold: K={stoch_k:.1f}")
            elif stoch_k > 80 and stoch_d > 80:
                sell_signals += 2
                signal_strength += 1
                signal_details.append(f"Stochastic Overbought: K={stoch_k:.1f}")
        
        if 'mfi' in df.columns and not pd.isna(last_row['mfi']):
            mfi_value = last_row['mfi']
            
            if mfi_value < 20:
                buy_signals += 2
                signal_strength += 1
                signal_details.append(f"MFI Oversold: {mfi_value:.1f}")
            elif mfi_value > 80:
                sell_signals += 2
                signal_strength += 1
                signal_details.append(f"MFI Overbought: {mfi_value:.1f}")
        
        if 'cci' in df.columns and not pd.isna(last_row['cci']):
            cci_value = last_row['cci']
            
            if cci_value < -100:
                buy_signals += 1
                signal_details.append(f"CCI Oversold: {cci_value:.1f}")
            elif cci_value > 100:
                sell_signals += 1
                signal_details.append(f"CCI Overbought: {cci_value:.1f}")
        
        if 'williams_r' in df.columns and not pd.isna(last_row['williams_r']):
            wr_value = last_row['williams_r']
            
            if wr_value < -80:
                buy_signals += 1
                signal_details.append(f"Williams %R Oversold: {wr_value:.1f}")
            elif wr_value > -20:
                sell_signals += 1
                signal_details.append(f"Williams %R Overbought: {wr_value:.1f}")
        
        if 'psar' in df.columns and not pd.isna(last_row['psar']):
            psar_value = last_row['psar']
            
            if current_price > psar_value and prev_row['close'] <= df.iloc[-2]['psar']:
                buy_signals += 2
                signal_strength += 1
                signal_details.append("PSAR Bullish Signal")
            elif current_price < psar_value and prev_row['close'] >= df.iloc[-2]['psar']:
                sell_signals += 2
                signal_strength += 1
                signal_details.append("PSAR Bearish Signal")
        
        if ('tenkan_sen' in df.columns and 'kijun_sen' in df.columns and
            not pd.isna(last_row['tenkan_sen']) and not pd.isna(last_row['kijun_sen'])):
            
            if (last_row['tenkan_sen'] > last_row['kijun_sen'] and 
                prev_row['tenkan_sen'] <= prev_row['kijun_sen']):
                buy_signals += 2
                signal_strength += 1
                signal_details.append("Ichimoku Bullish Cross")
            elif (last_row['tenkan_sen'] < last_row['kijun_sen'] and 
                  prev_row['tenkan_sen'] >= prev_row['kijun_sen']):
                sell_signals += 2
                signal_strength += 1
                signal_details.append("Ichimoku Bearish Cross")
        
        if ('sma20' in df.columns and 'sma50' in df.columns and
            not pd.isna(last_row['sma20']) and not pd.isna(last_row['sma50'])):
            
            if current_price > last_row['sma20'] > last_row['sma50']:
                buy_signals += 1
                signal_details.append("Price Above SMA20 & SMA50")
            elif current_price < last_row['sma20'] < last_row['sma50']:
                sell_signals += 1
                signal_details.append("Price Below SMA20 & SMA50")
        
        if 'volume_sma' in df.columns and not pd.isna(last_row['volume_sma']):
            volume_ratio = last_row['volume'] / last_row['volume_sma']
            if volume_ratio > 1.5:
                signal_strength += 1
                signal_details.append(f"High Volume: {volume_ratio:.1f}x")
        
        fibonacci_support_resistance = []
        fib_keys = ['fib_236', 'fib_382', 'fib_500', 'fib_618']
        for fib_key in fib_keys:
            if fib_key in df.columns:
                fib_level = last_row[fib_key]
                price_diff_pct = abs(current_price - fib_level) / current_price * 100
                if price_diff_pct < 1:
                    fibonacci_support_resistance.append(f"{fib_key}: {fib_level:.6f}")
        
        if fibonacci_support_resistance:
            signal_details.extend(fibonacci_support_resistance)
        
        min_signal_threshold = 3
        
        if buy_signals >= min_signal_threshold and buy_signals > sell_signals:
            return {
                'type': 'buy',
                'strength': min(signal_strength, 5),
                'rsi': rsi_value,
                'macd': last_row.get('MACD_12_26_9', 0),
                'method': 'Multi_Indicator_Buy',
                'details': signal_details,
                'buy_score': buy_signals,
                'sell_score': sell_signals
            }
        elif sell_signals >= min_signal_threshold and sell_signals > buy_signals:
            return {
                'type': 'sell',
                'strength': min(signal_strength, 5),
                'rsi': rsi_value,
                'macd': last_row.get('MACD_12_26_9', 0),
                'method': 'Multi_Indicator_Sell',
                'details': signal_details,
                'buy_score': buy_signals,
                'sell_score': sell_signals
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error checking signals for {symbol}: {e}")
        return None

def calculate_signal_strength(df, signal_type):
    """Calculate signal strength based on multiple factors"""
    try:
        if df is None or len(df) == 0:
            return 2
            
        last_row = df.iloc[-1]
        strength_score = 0
        
        # RSI strength
        if 'rsi' in df.columns and not pd.isna(last_row['rsi']):
            rsi_value = last_row['rsi']
            if signal_type == 'buy':
                if rsi_value < 20:
                    strength_score += 3
                elif rsi_value < 25:
                    strength_score += 2
                elif rsi_value < 30:
                    strength_score += 1
            else:  # sell
                if rsi_value > 80:
                    strength_score += 3
                elif rsi_value > 75:
                    strength_score += 2
                elif rsi_value > 70:
                    strength_score += 1
        
        # Volume strength
        if 'volume_sma' in df.columns and not pd.isna(last_row['volume_sma']):
            try:
                volume_ratio = last_row['volume'] / last_row['volume_sma']
                if volume_ratio > 2:
                    strength_score += 2
                elif volume_ratio > 1.5:
                    strength_score += 1
            except (ZeroDivisionError, TypeError):
                pass
        
        # MACD strength
        if ('MACD_12_26_9' in df.columns and 
            not pd.isna(last_row['MACD_12_26_9'])):
            macd_value = abs(last_row['MACD_12_26_9'])
            if macd_value > 0.001:  # Strong MACD signal
                strength_score += 1
        
        # Normalize to 1-5 scale
        return min(max(strength_score, 1), 5)
        
    except Exception:
        return 2  # Default medium strength

def calculate_signal_accuracy_score(df, signal_data, symbol):
    try:
        if df is None or len(df) < 50 or not signal_data:
            return 0
        
        last_row = df.iloc[-1]
        prev_rows = df.iloc[-10:] if len(df) >= 10 else df
        accuracy_score = 0
        
        rsi_value = signal_data.get('rsi', 50)
        if signal_data['type'] == 'buy':
            if rsi_value < 20:
                accuracy_score += 25
            elif rsi_value < 25:
                accuracy_score += 20
            elif rsi_value < 30:
                accuracy_score += 15
            elif rsi_value < 35:
                accuracy_score += 10
        else:
            if rsi_value > 80:
                accuracy_score += 25
            elif rsi_value > 75:
                accuracy_score += 20
            elif rsi_value > 70:
                accuracy_score += 15
            elif rsi_value > 65:
                accuracy_score += 10
        
        if ('MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns and
            not pd.isna(last_row.get('MACD_12_26_9')) and not pd.isna(last_row.get('MACDs_12_26_9'))):
            
            macd_line = last_row['MACD_12_26_9']
            signal_line = last_row['MACDs_12_26_9']
            macd_histogram = macd_line - signal_line
            
            if signal_data['type'] == 'buy' and macd_histogram > 0 and macd_line > signal_line:
                accuracy_score += 20
            elif signal_data['type'] == 'sell' and macd_histogram < 0 and macd_line < signal_line:
                accuracy_score += 20
            elif abs(macd_histogram) > 0.001:
                accuracy_score += 10
        
        if 'volume_sma' in df.columns and not pd.isna(last_row.get('volume_sma')):
            try:
                volume_ratio = last_row['volume'] / last_row['volume_sma']
                if volume_ratio > 2.5:
                    accuracy_score += 15
                elif volume_ratio > 2:
                    accuracy_score += 12
                elif volume_ratio > 1.5:
                    accuracy_score += 8
                elif volume_ratio > 1.2:
                    accuracy_score += 5
            except (ZeroDivisionError, TypeError):
                pass
        
        if all(col in df.columns for col in ['sma20', 'sma50', 'sma200']):
            current_price = last_row['close']
            sma20 = last_row.get('sma20')
            sma50 = last_row.get('sma50')
            sma200 = last_row.get('sma200')
            
            if not any(pd.isna(val) for val in [sma20, sma50, sma200]):
                if signal_data['type'] == 'buy':
                    if current_price > sma20 > sma50 > sma200:
                        accuracy_score += 15
                    elif current_price > sma20 > sma50:
                        accuracy_score += 10
                    elif current_price > sma20:
                        accuracy_score += 5
                else:
                    if current_price < sma20 < sma50 < sma200:
                        accuracy_score += 15
                    elif current_price < sma20 < sma50:
                        accuracy_score += 10
                    elif current_price < sma20:
                        accuracy_score += 5
        
        if 'STOCHk_14_3_3' in df.columns and 'STOCHd_14_3_3' in df.columns:
            k_value = last_row.get('STOCHk_14_3_3')
            d_value = last_row.get('STOCHd_14_3_3')
            
            if not pd.isna(k_value) and not pd.isna(d_value):
                if signal_data['type'] == 'buy' and k_value < 20 and d_value < 20:
                    accuracy_score += 10
                elif signal_data['type'] == 'sell' and k_value > 80 and d_value > 80:
                    accuracy_score += 10
                elif signal_data['type'] == 'buy' and k_value < 30:
                    accuracy_score += 5
                elif signal_data['type'] == 'sell' and k_value > 70:
                    accuracy_score += 5
        
        if 'mfi' in df.columns and not pd.isna(last_row.get('mfi')):
            mfi_value = last_row['mfi']
            if signal_data['type'] == 'buy' and mfi_value < 20:
                accuracy_score += 8
            elif signal_data['type'] == 'sell' and mfi_value > 80:
                accuracy_score += 8
            elif signal_data['type'] == 'buy' and mfi_value < 30:
                accuracy_score += 4
            elif signal_data['type'] == 'sell' and mfi_value > 70:
                accuracy_score += 4
        
        if 'cci' in df.columns and not pd.isna(last_row.get('cci')):
            cci_value = last_row['cci']
            if signal_data['type'] == 'buy' and cci_value < -100:
                accuracy_score += 5
            elif signal_data['type'] == 'sell' and cci_value > 100:
                accuracy_score += 5
        
        if len(prev_rows) >= 5:
            trend_direction = 0
            close_prices = prev_rows['close'].values
            
            for i in range(1, len(close_prices)):
                if close_prices[i] > close_prices[i-1]:
                    trend_direction += 1
                elif close_prices[i] < close_prices[i-1]:
                    trend_direction -= 1
            
            trend_strength = abs(trend_direction) / len(close_prices)
            
            if signal_data['type'] == 'buy' and trend_direction > 0:
                accuracy_score += int(10 * trend_strength)
            elif signal_data['type'] == 'sell' and trend_direction < 0:
                accuracy_score += int(10 * trend_strength)
        
        if symbol in ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']:
            accuracy_score += 5
        
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

async def analyze_market():
    """تحلیل بازار و بازگرداندن بهترین سیگنال"""
    all_signals = []
    successful_analyses = 0
    failed_analyses = 0
    
    logger.info(f"Starting market analysis for {len(SYMBOLS)} symbols")
    
    # پردازش نمادها به صورت دسته‌ای
    batch_size = 5
    for i in range(0, len(SYMBOLS), batch_size):
        batch_symbols = SYMBOLS[i:i+batch_size]
        
        for symbol in batch_symbols:
            try:
                # تاخیر برای رعایت محدودیت نرخ
                await asyncio.sleep(1)
                
                logger.info(f"Analyzing {symbol}...")
                
                df = await get_klines(symbol)
                if df is None:
                    failed_analyses += 1
                    continue
                
                df = calculate_indicators(df)
                if df is None:
                    failed_analyses += 1
                    continue
                
                signal_data = check_signals(df, symbol)
                if signal_data:
                    current_price = await get_current_price(symbol)
                    if current_price is not None:
                        # محاسبه امتیاز دقت
                        accuracy_score = calculate_signal_accuracy_score(df, signal_data, symbol)
                        
                        if accuracy_score >= 40:  # حداقل امتیاز قابل قبول
                            if signal_data['type'] == 'buy':
                                entry = current_price
                                target = entry * 1.05  # 5% هدف
                                stop_loss = entry * 0.96  # 4% حد ضرر
                                signal_type = 'Long'
                            else:  # sell
                                entry = current_price
                                target = entry * 0.95  # 5% هدف
                                stop_loss = entry * 1.04  # 4% حد ضرر
                                signal_type = 'Short'
                            
                            all_signals.append({
                                'symbol': symbol,
                                'type': signal_type,
                                'entry': entry,
                                'target': target,
                                'stop_loss': stop_loss,
                                'strength': signal_data['strength'],
                                'accuracy_score': accuracy_score,
                                'rsi': signal_data['rsi'],
                                'macd': signal_data['macd'],
                                'method': signal_data.get('method', 'Unknown'),
                                'timestamp': datetime.now().strftime('%H:%M:%S')
                            })
                            
                            logger.info(f"High accuracy signal found for {symbol}: {signal_type} (Score: {accuracy_score})")
                
                successful_analyses += 1
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                failed_analyses += 1
        
        # تاخیر کوتاه بین دسته‌ها
        if i + batch_size < len(SYMBOLS):
            await asyncio.sleep(2)
    
    # انتخاب بهترین سیگنال
    best_signal = None
    if all_signals:
        # مرتب‌سازی بر اساس امتیاز دقت
        all_signals.sort(key=lambda x: x['accuracy_score'], reverse=True)
        best_signal = all_signals[0]  # بهترین سیگنال
        
        logger.info(f"Best signal selected: {best_signal['symbol']} with accuracy score: {best_signal['accuracy_score']}")
    
    logger.info(f"Analysis complete. Success: {successful_analyses}, Failed: {failed_analyses}, "
                f"Total signals: {len(all_signals)}, Best signal: {best_signal['symbol'] if best_signal else 'None'}")
    
    return [best_signal] if best_signal else []

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """مدیریت دستور /start با ارائه بهترین سیگنال"""
    try:
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        logger.info(f"User {username} ({user_id}) started analysis")
        
        await update.message.reply_text(
            "🔍 در حال تحلیل بازار برای یافتن بهترین فرصت معاملاتی...\n"
            "⏳ این کار ممکن است چند دقیقه طول بکشد."
        )
        
        # تنظیم timeout برای کل تحلیل
        try:
            signals = await asyncio.wait_for(analyze_market(), timeout=1800)  # حداکثر 30 دقیقه
        except asyncio.TimeoutError:
            await update.message.reply_text(
                "⏱️ تحلیل بیش از حد زمان برد. لطفا دوباره تلاش کنید."
            )
            return
        
        if signals and len(signals) > 0:
            sig = signals[0]  # بهترین سیگنال
            
            # تعیین emoji و رنگ بر اساس نوع سیگنال
            emoji = '📈' if sig['type'] == 'Long' else '📉'
            type_color = '🟢' if sig['type'] == 'Long' else '🔴'
            
            # محاسبه درصد سود/ضرر
            if sig['type'] == 'Long':
                profit_pct = ((sig['target'] - sig['entry']) / sig['entry']) * 100
                loss_pct = ((sig['entry'] - sig['stop_loss']) / sig['entry']) * 100
            else:
                profit_pct = ((sig['entry'] - sig['target']) / sig['entry']) * 100
                loss_pct = ((sig['stop_loss'] - sig['entry']) / sig['entry']) * 100
            
            # ساختار پیام بهینه‌شده
            message = f"🎯 *بهترین فرصت معاملاتی یافت شده*\n"
            message += f"{'='*30}\n\n"
            
            message += f"{emoji} *{sig['type']} {sig['symbol']}* {type_color}\n"
            message += f"🏆 **امتیاز دقت: {sig['accuracy_score']}/100**\n\n"
            
            message += f"📊 **جزئیات معاملاتی:**\n"
            message += f"💰 قیمت ورودی: `{sig['entry']:.6f}`\n"
            message += f"🎯 هدف قیمت: `{sig['target']:.6f}` (+{profit_pct:.1f}%)\n"
            message += f"🛑 حد ضرر: `{sig['stop_loss']:.6f}` (-{loss_pct:.1f}%)\n\n"
            
            message += f"📈 **تحلیل تکنیکال:**\n"
            message += f"• RSI: `{sig['rsi']:.1f}`\n"
            message += f"• MACD: `{sig['macd']:.6f}`\n"
            message += f"• روش تحلیل: `{sig['method']}`\n"
            message += f"• قدرت سیگنال: {'⭐' * sig['strength']}\n\n"
            
            message += f"⏰ زمان تولید سیگنال: `{sig['timestamp']}`\n\n"
            
        else:
            message = (
                "❌ متأسفانه در حال حاضر هیچ سیگنال معاملاتی با دقت بالا یافت نشد.\n\n"
                "🔍 **دلایل احتمالی:**\n"
                "• بازار در حالت تثبیت قرار دارد\n"
                "• شرایط تکنیکال مناسب معاملاتی وجود ندارد\n"
                "• همه سیگنال‌ها دارای ریسک بالا هستند\n\n"
                "💡 **پیشنهاد:**\n"
                "• 30-60 دقیقه دیگر مجدداً تلاش کنید\n"
                "• در انتظار شکل‌گیری الگوهای تکنیکال باشید\n"
                "• از معاملات پر ریسک خودداری کنید\n\n"
                "🔄 برای تحلیل مجدد /start را ارسال کنید."
            )
        
        await update.message.reply_text(message, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in start command: {e}")
        await update.message.reply_text(
            "❌ خطایی در تحلیل بازار رخ داد. لطفا دوباره تلاش کنید.\n"
            f"جزئیات خطا: {str(e)[:100]}..."
        )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command"""
    try:
        # Test exchange connection
        exchange_status = "❌ قطع"
        try:
            test_exchange = await init_exchange()
            if test_exchange:
                await test_exchange.fetch_ticker('BTC/USDT')
                exchange_status = "✅ متصل"
        except:
            pass
        
        message = "🤖 *وضعیت ربات:*\n\n"
        message += "🟢 ربات فعال است\n"
        message += f"📈 تعداد نمادها: `{len(SYMBOLS)}`\n"
        message += f"🔗 صرافی CoinEx: {exchange_status}\n"
        message += f"⏰ آخرین بررسی: `{datetime.now().strftime('%H:%M:%S')}`\n"
        message += f"🐍 Python: `{sys.version.split()[0]}`\n\n"
        message += "💡 *دستورات موجود:*\n"
        message += "`/start` - تحلیل بازار\n"
        message += "`/status` - وضعیت ربات\n"
        message += "`/symbols` - نمایش نمادها\n"
        message += "`/help` - راهنما"
        
        await update.message.reply_text(message, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in status command: {e}")
        await update.message.reply_text("خطایی در نمایش وضعیت رخ داد.")

async def show_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /symbols command"""
    try:
        message = f"📋 *نمادهای تحت نظارت:* ({len(SYMBOLS)} نماد)\n\n"
        
        # Group symbols in rows of 3
        for i in range(0, len(SYMBOLS), 3):
            row_symbols = SYMBOLS[i:i+3]
            message += " | ".join([f"`{symbol}`" for symbol in row_symbols]) + "\n"
        
        message += f"\n💡 برای تغییر نمادها، فایل `symbols.txt` را ویرایش کنید."
        
        await update.message.reply_text(message, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in symbols command: {e}")
        await update.message.reply_text("خطایی در نمایش نمادها رخ داد.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command"""
    try:
        message = (
            "🤖 *راهنمای ربات تحلیل تکنیکال*\n\n"
            "این ربات با استفاده از اندیکاتورهای تکنیکال، فرصت‌های معاملاتی را شناسایی می‌کند.\n\n"
            "📋 *دستورات:*\n"
            "`/start` - شروع تحلیل بازار\n"
            "`/status` - نمایش وضعیت ربات\n"
            "`/symbols` - لیست نمادهای تحت نظارت\n"
            "`/help` - نمایش این راهنما\n\n"
            "📊 *اندیکاتورهای استفاده شده:*\n"
            "• RSI (Relative Strength Index)\n"
            "• MACD (Moving Average Convergence Divergence)\n"
            "• SMA (Simple Moving Average)\n"
            "• Volume Analysis\n\n"
            "⚠️ *هشدار مهم:*\n"
            "این سیگنال‌ها صرفاً جهت اطلاع‌رسانی هستند و توصیه سرمایه‌گذاری محسوب نمی‌شوند. "
            "لطفاً قبل از هر معامله، تحلیل‌های خود را انجام دهید."
        )
        
        await update.message.reply_text(message, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in help command: {e}")
        await update.message.reply_text("خطایی در نمایش راهنما رخ داد.")

async def close_exchange():
    """Close exchange connection"""
    global exchange
    if exchange:
        try:
            await exchange.close()
            logger.info("Exchange connection closed")
        except:
            pass
        exchange = None

def main() -> None:
    """Run the bot with improved error handling"""
    try:
        if not BOT_TOKEN:
            logger.error("BOT_TOKEN not found. Please set TELEGRAM_BOT_TOKEN environment variable.")
            print("Please set the TELEGRAM_BOT_TOKEN environment variable!")
            return
        
        if not SYMBOLS:
            logger.error("No symbols loaded. Please check symbols.txt file.")
            print("Please create a symbols.txt file with trading pairs!")
            return
        
        logger.info("Starting Telegram Trading Bot...")
        logger.info(f"Loaded {len(SYMBOLS)} symbols for analysis")
        
        application = ApplicationBuilder().token(BOT_TOKEN).build()
        
        # Add command handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("status", status))
        application.add_handler(CommandHandler("symbols", show_symbols))
        application.add_handler(CommandHandler("help", help_command))
        
        logger.info("Bot is ready and polling...")
        print("✅ Bot started successfully! Press Ctrl+C to stop.")
        
        # Run with graceful shutdown
        application.run_polling(
            drop_pending_updates=True,
            allowed_updates=['message'],
            stop_signals=[],  # Handle shutdown manually
        )
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        print("Bot stopped.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Fatal error: {e}")
    finally:
        # Cleanup
        try:
            asyncio.run(close_exchange())
        except:
            pass

if __name__ == '__main__':
    main()