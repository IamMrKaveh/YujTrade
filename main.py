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
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'token')

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

# اندیکاتورهای مومنتوم اضافی
def calculate_ultimate_oscillator(df, period1=7, period2=14, period3=28):
    """محاسبه Ultimate Oscillator"""
    try:
        if df is None or len(df) < max(period1, period2, period3):
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        # True Low = minimum of Low or previous Close
        true_low = pd.concat([low, prev_close], axis=1).min(axis=1)
        
        # Buying Pressure = Close - True Low
        buying_pressure = close - true_low
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Average calculations for 3 periods
        bp1 = buying_pressure.rolling(window=period1).sum()
        tr1_sum = true_range.rolling(window=period1).sum()
        
        bp2 = buying_pressure.rolling(window=period2).sum()
        tr2_sum = true_range.rolling(window=period2).sum()
        
        bp3 = buying_pressure.rolling(window=period3).sum()
        tr3_sum = true_range.rolling(window=period3).sum()
        
        # Ultimate Oscillator formula
        uo = 100 * (4 * (bp1 / tr1_sum) + 2 * (bp2 / tr2_sum) + (bp3 / tr3_sum)) / 7
        
        return uo
    except Exception as e:
        logger.warning(f"Error calculating Ultimate Oscillator: {e}")
        return None

def calculate_rate_of_change(df, period=14):
    """محاسبه Rate of Change (ROC)"""
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        roc = ((close - close.shift(period)) / close.shift(period)) * 100
        
        return roc
    except Exception as e:
        logger.warning(f"Error calculating ROC: {e}")
        return None

def calculate_awesome_oscillator(df, fast_period=5, slow_period=34):
    try:
        if df is None or len(df) < slow_period:
            return None
        
        median_price = (df['high'] + df['low']) / 2
        
        fast = median_price.rolling(window=fast_period).mean()
        slow = median_price.rolling(window=slow_period).mean()

        ao = fast - slow
        ao = ao.fillna(0)  # Fill NaN with 0
        
        logger.info(f"Calculated Awesome Oscillator with {len(ao)} values")
        return ao
    except Exception as e:
        logger.warning(f"Error calculating Awesome Oscillator: {e}")
        return None

def calculate_trix(df, period=14):
    """محاسبه TRIX"""
    try:
        if df is None or len(df) < period * 3:
            return None
        
        close = df['close']
        
        # Triple smoothed EMA
        ema1 = close.ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        ema3 = ema2.ewm(span=period).mean()
        
        # TRIX = Rate of change of triple smoothed EMA
        trix = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 10000
        
        return trix
    except Exception as e:
        logger.warning(f"Error calculating TRIX: {e}")
        return None

def calculate_dpo(df, period=20):
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        sma = close.rolling(window=period).mean()
        
        shift_period = int(period/2) + 1
        dpo = close - sma.shift(shift_period)
        
        dpo = dpo.fillna(0)  # Fill NaN with 0
        
        logger.info(f"Calculated DPO with {len(dpo)} values")
        
        return dpo
    except Exception as e:
        logger.warning(f"Error calculating DPO: {e}")
        return None

# ===== اندیکاتورهای حجم پیشرفته =====

def calculate_obv(df):
    """محاسبه On-Balance Volume"""
    try:
        if df is None or len(df) < 2:
            return None
        
        close = df['close']
        volume = df['volume']
        
        obv = []
        obv.append(0)  # مقدار اولیه
        
        for i in range(1, len(df)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[i-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[i-1] - volume.iloc[i])
            else:
                obv.append(obv[i-1])
        
        return pd.Series(obv, index=df.index)
    except Exception as e:
        logger.warning(f"Error calculating OBV: {e}")
        return None

def calculate_accumulation_distribution(df):
    """محاسبه Accumulation/Distribution Line"""
    try:
        if df is None or len(df) < 1:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        # Handle division by zero
        mfm = mfm.fillna(0)
        
        # Money Flow Volume
        mfv = mfm * volume
        
        # A/D Line is cumulative sum of MFV
        ad_line = mfv.cumsum()
        
        return ad_line
    except Exception as e:
        logger.warning(f"Error calculating A/D Line: {e}")
        return None

def calculate_ad_line(df):
    """محاسبه Accumulation/Distribution Line"""
    try:
        if df is None or len(df) < 1:
            return None
            
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # محاسبه Money Flow Multiplier
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # در صورت صفر بودن دامنه
        
        # محاسبه Money Flow Volume
        mfv = clv * volume
        
        # محاسبه A/D Line تجمعی
        ad_line = mfv.cumsum()
        
        return ad_line
    except Exception:
        return None

def calculate_chaikin_money_flow(df, period=20):
    """محاسبه Chaikin Money Flow"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)
        
        # Money Flow Volume
        mfv = mfm * volume
        
        # CMF = Sum of MFV over period / Sum of Volume over period
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        return cmf
    except Exception as e:
        logger.warning(f"Error calculating CMF: {e}")
        return None

def calculate_volume_price_trend(df):
    """محاسبه Volume Price Trend"""
    try:
        if df is None or len(df) < 2:
            return None
        
        close = df['close']
        volume = df['volume']
        
        # Price change percentage
        price_change_pct = (close - close.shift(1)) / close.shift(1)
        
        # VPT = Previous VPT + Volume * Price Change %
        vpt = (price_change_pct * volume).cumsum()
        
        return vpt
    except Exception as e:
        logger.warning(f"Error calculating VPT: {e}")
        return None

def calculate_ease_of_movement(df, period=14):
    """محاسبه Ease of Movement"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Distance Moved
        distance_moved = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2
        
        # Box Height
        box_height = (volume / 100000) / (high - low)
        
        # 1-Period EMV
        emv_1period = distance_moved / box_height
        emv_1period = emv_1period.replace([np.inf, -np.inf], 0).fillna(0)
        
        # EMV = SMA of 1-Period EMV
        emv = emv_1period.rolling(window=period).mean()
        
        return emv
    except Exception as e:
        logger.warning(f"Error calculating EMV: {e}")
        return None

# ===== اندیکاتورهای نوسان =====

def calculate_average_true_range(df, period=14):
    """محاسبه Average True Range"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    except Exception as e:
        logger.warning(f"Error calculating ATR: {e}")
        return None

def calculate_atr(df, period=14):
    """محاسبه Average True Range"""
    try:
        if df is None or len(df) < period:
            return None
            
        high = df['high']
        low = df['low']
        close = df['close']
        
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    except Exception:
        return None

def calculate_keltner_channels(df, period=20, multiplier=2):
    """محاسبه Keltner Channels"""
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        atr = calculate_average_true_range(df, period)
        
        if atr is None:
            return None
        
        middle_line = close.rolling(window=period).mean()
        upper_channel = middle_line + (multiplier * atr)
        lower_channel = middle_line - (multiplier * atr)
        
        return {
            'keltner_upper': upper_channel,
            'keltner_middle': middle_line,
            'keltner_lower': lower_channel
        }
    except Exception as e:
        logger.warning(f"Error calculating Keltner Channels: {e}")
        return None

def calculate_donchian_channels(df, period=20):
    """محاسبه Donchian Channels"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        
        upper_channel = high.rolling(window=period).max()
        lower_channel = low.rolling(window=period).min()
        middle_channel = (upper_channel + lower_channel) / 2
        
        return {
            'donchian_upper': upper_channel,
            'donchian_middle': middle_channel,
            'donchian_lower': lower_channel
        }
    except Exception as e:
        logger.warning(f"Error calculating Donchian Channels: {e}")
        return None

def calculate_standard_deviation(df, period=20):
    """محاسبه Standard Deviation"""
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        std_dev = close.rolling(window=period).std()
        
        return std_dev
    except Exception as e:
        logger.warning(f"Error calculating Standard Deviation: {e}")
        return None

def calculate_price_std(df, period=20):
    """محاسبه انحراف معیار قیمت"""
    try:
        if df is None or len(df) < period:
            return None
            
        close = df['close']
        std_dev = close.rolling(period).std()
        
        return std_dev
    except Exception:
        return None

# ===== اندیکاتورهای ترند پیشرفته =====

def calculate_supertrend(df, period=10, multiplier=3.0):
    """محاسبه Supertrend"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # محاسبه ATR
        atr = calculate_average_true_range(df, period)
        if atr is None:
            return None
        
        # محاسبه Basic Bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Final Bands
        final_upper_band = []
        final_lower_band = []
        supertrend = []
        direction = []
        
        for i in range(len(df)):
            if i == 0:
                final_upper_band.append(upper_band.iloc[i])
                final_lower_band.append(lower_band.iloc[i])
                supertrend.append(0)
                direction.append(1)
            else:
                # Final Upper Band
                if upper_band.iloc[i] < final_upper_band[i-1] or close.iloc[i-1] > final_upper_band[i-1]:
                    final_upper_band.append(upper_band.iloc[i])
                else:
                    final_upper_band.append(final_upper_band[i-1])
                
                # Final Lower Band
                if lower_band.iloc[i] > final_lower_band[i-1] or close.iloc[i-1] < final_lower_band[i-1]:
                    final_lower_band.append(lower_band.iloc[i])
                else:
                    final_lower_band.append(final_lower_band[i-1])
                
                # Direction and Supertrend
                if direction[i-1] == -1 and close.iloc[i] < final_lower_band[i]:
                    direction.append(-1)
                elif direction[i-1] == 1 and close.iloc[i] > final_upper_band[i]:
                    direction.append(1)
                elif direction[i-1] == -1 and close.iloc[i] >= final_lower_band[i]:
                    direction.append(1)
                elif direction[i-1] == 1 and close.iloc[i] <= final_upper_band[i]:
                    direction.append(-1)
                else:
                    direction.append(direction[i-1])
                
                if direction[i] == 1:
                    supertrend.append(final_lower_band[i])
                else:
                    supertrend.append(final_upper_band[i])
        
        return pd.Series(supertrend, index=df.index)
    except Exception as e:
        logger.warning(f"Error calculating Supertrend: {e}")
        return None

def calculate_aroon_oscillator(df, period=14):
    """محاسبه Aroon Oscillator"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        
        aroon_up = []
        aroon_down = []
        
        for i in range(len(df)):
            if i < period - 1:
                aroon_up.append(np.nan)
                aroon_down.append(np.nan)
            else:
                # محاسبه Aroon Up
                high_period = high.iloc[i-period+1:i+1]
                periods_since_high = period - 1 - high_period.idxmax()
                aroon_up_val = ((period - periods_since_high) / period) * 100
                aroon_up.append(aroon_up_val)
                
                # محاسبه Aroon Down
                low_period = low.iloc[i-period+1:i+1]
                periods_since_low = period - 1 - low_period.idxmin()
                aroon_down_val = ((period - periods_since_low) / period) * 100
                aroon_down.append(aroon_down_val)
        
        aroon_up = pd.Series(aroon_up, index=df.index).fillna(0)  # Fill NaN with 0
        aroon_down = pd.Series(aroon_down, index=df.index).fillna(0)  # Fill NaN with 0
        aroon_oscillator = aroon_up - aroon_down
        
        return {
            'aroon_up': aroon_up,
            'aroon_down': aroon_down,
            'aroon_oscillator': aroon_oscillator
        }
    except Exception as e:
        logger.warning(f"Error calculating Aroon Oscillator: {e}")
        return None

def calculate_aroon(df, period=14):
    """محاسبه Aroon Oscillator"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        
        # پیدا کردن موقعیت بالاترین و پایین‌ترین قیمت
        aroon_up = ((period - high.rolling(period).apply(lambda x: period - 1 - x.argmax())) / period) * 100
        aroon_down = ((period - low.rolling(period).apply(lambda x: period - 1 - x.argmin())) / period) * 100
        
        aroon_oscillator = aroon_up.fillna(0) - aroon_down.fillna(0)  # Fill NaN with 0
        
        return {
            'aroon_up': aroon_up,
            'aroon_down': aroon_down,
            'aroon_oscillator': aroon_oscillator
        }
    except Exception:
        return None

def calculate_adx(df, period=14):
    """محاسبه Average Directional Index (ADX)"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        
        # True Range
        atr = calculate_average_true_range(df, period)
        if atr is None:
            return None
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = pd.Series(0, index=df.index)
        minus_dm = pd.Series(0, index=df.index)
        
        plus_dm[up_move > down_move] = up_move[up_move > down_move]
        plus_dm[plus_dm < 0] = 0
        
        minus_dm[down_move > up_move] = down_move[down_move > up_move]
        minus_dm[minus_dm < 0] = 0
        
        # Smoothed DM
        plus_dm_smooth = plus_dm.rolling(window=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period).mean()
        
        # DI calculations
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)
        
        # ADX calculation
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = dx.fillna(0)
        adx = dx.rolling(window=period).mean()
        
        return {
            'plus_di': plus_di,
            'minus_di': minus_di,
            'adx': adx
        }
    except Exception as e:
        logger.warning(f"Error calculating ADX: {e}")
        return None

def calculate_kama(df, period=10, fast_sc=2, slow_sc=30):
    """محاسبه Kaufman Adaptive Moving Average"""
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        
        # Efficiency Ratio
        change = abs(close - close.shift(period))
        volatility = abs(close - close.shift(1)).rolling(window=period).sum()
        er = change / volatility
        er = er.fillna(0)
        
        # Smoothing Constants
        fastest_sc = 2.0 / (fast_sc + 1)
        slowest_sc = 2.0 / (slow_sc + 1)
        sc = (er * (fastest_sc - slowest_sc) + slowest_sc) ** 2
        
        # KAMA calculation
        kama = []
        kama.append(close.iloc[0])  # First value
        
        for i in range(1, len(close)):
            kama.append(kama[i-1] + sc.iloc[i] * (close.iloc[i] - kama[i-1]))
        
        return pd.Series(kama, index=df.index)
    except Exception as e:
        logger.warning(f"Error calculating KAMA: {e}")
        return None

# الگوهای کندل استیک
def detect_hammer_doji_patterns(df):
    """تشخیص الگوهای Hammer و Doji"""
    try:
        if df is None or len(df) < 3:
            return None
        
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # محاسبه اجزای کندل
        body = abs(close - open_price)
        upper_shadow = high - pd.concat([open_price, close], axis=1).max(axis=1)
        lower_shadow = pd.concat([open_price, close], axis=1).min(axis=1) - low
        total_range = high - low
        
        patterns = pd.DataFrame(index=df.index)
        
        # Hammer Pattern
        hammer_condition = (
            (lower_shadow >= 2 * body) &
            (upper_shadow <= 0.1 * total_range) &
            (body <= 0.3 * total_range)
        )
        patterns['hammer'] = hammer_condition
        
        # Doji Pattern
        doji_condition = (body <= 0.1 * total_range)
        patterns['doji'] = doji_condition
        
        # Shooting Star Pattern
        shooting_star_condition = (
            (upper_shadow >= 2 * body) &
            (lower_shadow <= 0.1 * total_range) &
            (body <= 0.3 * total_range)
        )
        patterns['shooting_star'] = shooting_star_condition
        
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting Hammer/Doji patterns: {e}")
        return None

def detect_engulfing_patterns(df):
    """تشخیص الگوهای Engulfing"""
    try:
        if df is None or len(df) < 2:
            return None
        
        open_price = df['open']
        close = df['close']
        
        patterns = pd.DataFrame(index=df.index)
        patterns['bullish_engulfing'] = False
        patterns['bearish_engulfing'] = False
        
        for i in range(1, len(df)):
            prev_open = open_price.iloc[i-1]
            prev_close = close.iloc[i-1]
            curr_open = open_price.iloc[i]
            curr_close = close.iloc[i]
            
            # Bullish Engulfing
            if (prev_close < prev_open and  # Previous red candle
                curr_close > curr_open and  # Current green candle
                curr_open < prev_close and  # Current opens below previous close
                curr_close > prev_open):    # Current closes above previous open
                patterns.iloc[i, patterns.columns.get_loc('bullish_engulfing')] = True
            
            # Bearish Engulfing
            if (prev_close > prev_open and  # Previous green candle
                curr_close < curr_open and  # Current red candle
                curr_open > prev_close and  # Current opens above previous close
                curr_close < prev_open):    # Current closes below previous open
                patterns.iloc[i, patterns.columns.get_loc('bearish_engulfing')] = True
        
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting Engulfing patterns: {e}")
        return None

def detect_star_patterns(df):
    """تشخیص الگوهای Morning/Evening Star"""
    try:
        if df is None or len(df) < 3:
            return None
        
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        patterns = pd.DataFrame(index=df.index)
        patterns['morning_star'] = False
        patterns['evening_star'] = False
        
        for i in range(2, len(df)):
            # Morning Star Pattern
            first_red = close.iloc[i-2] < open_price.iloc[i-2]  # First candle is red
            small_body = abs(close.iloc[i-1] - open_price.iloc[i-1]) < abs(close.iloc[i-2] - open_price.iloc[i-2]) * 0.3  # Small middle candle
            gap_down = high.iloc[i-1] < low.iloc[i-2]  # Gap down
            third_green = close.iloc[i] > open_price.iloc[i]  # Third candle is green
            closes_into_first = close.iloc[i] > (open_price.iloc[i-2] + close.iloc[i-2]) / 2  # Closes well into first candle
            
            if first_red and small_body and gap_down and third_green and closes_into_first:
                patterns.iloc[i, patterns.columns.get_loc('morning_star')] = True
            
            # Evening Star Pattern
            first_green = close.iloc[i-2] > open_price.iloc[i-2]  # First candle is green
            gap_up = low.iloc[i-1] > high.iloc[i-2]  # Gap up
            third_red = close.iloc[i] < open_price.iloc[i]  # Third candle is red
            closes_into_first = close.iloc[i] < (open_price.iloc[i-2] + close.iloc[i-2]) / 2  # Closes well into first candle
            
            if first_green and small_body and gap_up and third_red and closes_into_first:
                patterns.iloc[i, patterns.columns.get_loc('evening_star')] = True
        
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting Star patterns: {e}")
        return None

def detect_morning_evening_star(df):
    """تشخیص الگوهای Morning/Evening Star"""
    try:
        if df is None or len(df) < 3:
            return None
            
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # محاسبه بدنه کندل‌ها
        body = abs(close - open_price)
        body_1 = body.shift(1)
        body_2 = body.shift(2)
        
        # Morning Star Pattern
        morning_star = ((close.shift(2) < open_price.shift(2)) &  # کندل نزولی
                       (body_1 < body_2 * 0.3) &  # کندل کوچک میانی
                       (close > open_price) &  # کندل صعودی
                       (close > (close.shift(2) + open_price.shift(2)) / 2))
        
        # Evening Star Pattern
        evening_star = ((close.shift(2) > open_price.shift(2)) &  # کندل صعودی
                       (body_1 < body_2 * 0.3) &  # کندل کوچک میانی
                       (close < open_price) &  # کندل نزولی
                       (close < (close.shift(2) + open_price.shift(2)) / 2))
        
        return {
            'morning_star': morning_star,
            'evening_star': evening_star
        }
    except Exception:
        return None

# اندیکاتورهای مارکت استراکچر
def calculate_pivot_points(df):
    """محاسبه Pivot Points"""
    try:
        if df is None or len(df) < 1:
            return None
        
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        # Standard Pivot Points
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    except Exception as e:
        logger.warning(f"Error calculating Pivot Points: {e}")
        return None

def calculate_support_resistance(df, window=20):
    """محاسبه سطوح Support و Resistance"""
    try:
        if df is None or len(df) < window:
            return None
        
        high = df['high']
        low = df['low']
        
        # Local highs and lows
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(df) - window):
            # Check for local high (resistance)
            if high.iloc[i] == high.iloc[i-window:i+window+1].max():
                resistance_levels.append(high.iloc[i])
            
            # Check for local low (support)
            if low.iloc[i] == low.iloc[i-window:i+window+1].min():
                support_levels.append(low.iloc[i])
        
        # Get most significant levels
        resistance_levels = sorted(set(resistance_levels), reverse=True)[:5]
        support_levels = sorted(set(support_levels))[:5]
        
        return {
            'resistance_levels': resistance_levels,
            'support_levels': support_levels
        }
    except Exception as e:
        logger.warning(f"Error calculating Support/Resistance: {e}")
        return None

def detect_market_structure_breaks(df, swing_strength=5):
    """تشخیص Market Structure Breaks"""
    try:
        if df is None or len(df) < swing_strength * 2:
            return None
        
        high = df['high']
        low = df['low']
        
        structure_breaks = pd.DataFrame(index=df.index)
        structure_breaks['bullish_break'] = False
        structure_breaks['bearish_break'] = False
        
        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(swing_strength, len(df) - swing_strength):
            # Swing High
            if high.iloc[i] == high.iloc[i-swing_strength:i+swing_strength+1].max():
                swing_highs.append((i, high.iloc[i]))
            
            # Swing Low
            if low.iloc[i] == low.iloc[i-swing_strength:i+swing_strength+1].min():
                swing_lows.append((i, low.iloc[i]))
        
        # Detect breaks
        current_price = df['close'].iloc[-1]
        
        # Check if current price breaks recent swing high (bullish break)
        if swing_highs:
            recent_high = max(swing_highs[-3:], key=lambda x: x[1])[1] if len(swing_highs) >= 3 else swing_highs[-1][1]
            if current_price > recent_high:
                structure_breaks.iloc[-1, structure_breaks.columns.get_loc('bullish_break')] = True
        
        # Check if current price breaks recent swing low (bearish break)
        if swing_lows:
            recent_low = min(swing_lows[-3:], key=lambda x: x[1])[1] if len(swing_lows) >= 3 else swing_lows[-1][1]
            if current_price < recent_low:
                structure_breaks.iloc[-1, structure_breaks.columns.get_loc('bearish_break')] = True
        
        return structure_breaks
    except Exception as e:
        logger.warning(f"Error detecting Market Structure Breaks: {e}")
        return None

# ===== فیلترهای اضافی =====

def calculate_correlation_with_btc(df, btc_df, period=20):
    """محاسبه همبستگی با بیت کوین"""
    try:
        if df is None or btc_df is None or len(df) < period or len(btc_df) < period:
            return None
            
        # هم‌تراز کردن داده‌ها بر اساس زمان
        merged = pd.merge(df[['close']], btc_df[['close']], 
                         left_index=True, right_index=True, 
                         suffixes=('', '_btc'), how='inner')
        
        if len(merged) < period:
            return None
            
        # محاسبه همبستگی غلتان
        correlation = merged['close'].rolling(period).corr(merged['close_btc'])
        
        return correlation
    except Exception:
        return None

def detect_market_regime(df, lookback=50):
    """تشخیص رژیم بازار"""
    try:
        if df is None or len(df) < lookback:
            return None
            
        close = df['close']
        
        # محاسبه نوسانات
        returns = close.pct_change()
        volatility = returns.rolling(lookback).std() * np.sqrt(252)  # سالانه
        
        # محاسبه ترند
        sma_short = close.rolling(10).mean()
        sma_long = close.rolling(50).mean()
        trend = sma_short - sma_long
        
        # تعیین رژیم بازار
        regime = pd.Series(index=df.index, dtype=str)
        
        for i in range(lookback, len(df)):
            vol = volatility.iloc[i]
            tr = trend.iloc[i]
            
            if vol > volatility.rolling(lookback).quantile(0.75).iloc[i]:
                if tr > 0:
                    regime.iloc[i] = 'Bull_Volatile'
                else:
                    regime.iloc[i] = 'Bear_Volatile'
            else:
                if tr > 0:
                    regime.iloc[i] = 'Bull_Stable'
                else:
                    regime.iloc[i] = 'Bear_Stable'
        
        return regime
    except Exception:
        return None

# ===== ابزارهای ریسک منجمنت =====

def calculate_position_size_atr(capital, risk_percent, entry_price, atr_value, atr_multiplier=2):
    """محاسبه اندازه پوزیشن بر اساس ATR"""
    try:
        risk_amount = capital * (risk_percent / 100)
        stop_distance = atr_value * atr_multiplier
        position_size = risk_amount / stop_distance
        
        return min(position_size, capital * 0.1)  # حداکثر 10% سرمایه
    except Exception:
        return 0

def calculate_dynamic_stop_loss(df, entry_price, position_type='long', atr_multiplier=2):
    """محاسبه حد ضرر پویا"""
    try:
        if df is None or len(df) < 14:
            return entry_price * 0.95 if position_type == 'long' else entry_price * 1.05
            
        atr = calculate_atr(df)
        if atr is None:
            return entry_price * 0.95 if position_type == 'long' else entry_price * 1.05
            
        atr_value = atr.iloc[-1]
        
        if position_type == 'long':
            stop_loss = entry_price - (atr_value * atr_multiplier)
        else:
            stop_loss = entry_price + (atr_value * atr_multiplier)
            
        return stop_loss
    except Exception:
        return entry_price * 0.95 if position_type == 'long' else entry_price * 1.05

def optimize_risk_reward_ratio(entry_price, target_price, stop_loss, min_ratio=2.0):
    """بهینه‌سازی نسبت ریسک-ریوارد"""
    try:
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
        
        current_ratio = reward / risk if risk > 0 else 0
        
        if current_ratio < min_ratio:
            # تنظیم هدف برای دستیابی به نسبت حداقل
            if entry_price > stop_loss:  # long position
                new_target = entry_price + (risk * min_ratio)
            else:  # short position
                new_target = entry_price - (risk * min_ratio)
            
            return new_target
        
        return target_price
    except Exception:
        return target_price

# ===== تکنیک‌های بهبود دقت =====

def ensemble_signal_scoring(signals_dict, weights=None):
    """ترکیب چندین سیگنال با وزن‌دهی"""
    try:
        if not signals_dict:
            return 0
            
        if weights is None:
            weights = {key: 1 for key in signals_dict.keys()}
        
        total_score = 0
        total_weight = 0
        
        for signal_name, signal_value in signals_dict.items():
            if signal_name in weights and signal_value is not None:
                weight = weights[signal_name]
                total_score += signal_value * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    except Exception:
        return 0

def adaptive_threshold_calculator(df, indicator_values, percentile_low=20, percentile_high=80):
    """محاسبه آستانه‌های تطبیقی"""
    try:
        if df is None or indicator_values is None:
            return {'low': 30, 'high': 70}
            
        # محاسبه آستانه‌ها بر اساس توزیع تاریخی
        low_threshold = np.percentile(indicator_values.dropna(), percentile_low)
        high_threshold = np.percentile(indicator_values.dropna(), percentile_high)
        
        return {
            'low': low_threshold,
            'high': high_threshold
        }
    except Exception:
        return {'low': 30, 'high': 70}

def filter_false_signals(df, signal_data, min_volume_ratio=1.2, min_trend_strength=0.1):
    """فیلتر سیگنال‌های کاذب"""
    try:
        if df is None or not signal_data:
            return False
            
        last_row = df.iloc[-1]
        
        # فیلتر حجم
        if 'volume_sma' in df.columns and not pd.isna(last_row['volume_sma']):
            volume_ratio = last_row['volume'] / last_row['volume_sma']
            if volume_ratio < min_volume_ratio:
                return False
        
        # فیلتر قدرت ترند
        if len(df) >= 10:
            recent_closes = df['close'].tail(10).values
            trend_strength = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
            
            if signal_data['type'] == 'buy' and trend_strength < -min_trend_strength:
                return False
            elif signal_data['type'] == 'sell' and trend_strength > min_trend_strength:
                return False
        
        return True
    except Exception:
        return True

def calculate_indicators(df):
    try:
        if df is None or len(df) < 200:
            logger.warning(f"Insufficient data for indicators: {len(df) if df is not None else 0} candles")
            return None
        
        df = df.copy()
        
        # Simple Moving Averages
        df['sma20'] = safe_indicator_calculation(df, ta.sma, df['close'], length=20)
        df['sma50'] = safe_indicator_calculation(df, ta.sma, df['close'], length=50)
        df['sma200'] = safe_indicator_calculation(df, ta.sma, df['close'], length=200)
        
        # Exponential Moving Averages
        df['ema12'] = safe_indicator_calculation(df, ta.ema, df['close'], length=12)
        df['ema26'] = safe_indicator_calculation(df, ta.ema, df['close'], length=26)
        df['ema50'] = safe_indicator_calculation(df, ta.ema, df['close'], length=50)
        
        # Weighted Moving Average
        df['wma20'] = safe_indicator_calculation(df, ta.wma, df['close'], length=20)
        
        # RSI
        rsi = safe_indicator_calculation(df, ta.rsi, df['close'], length=14)
        if rsi is not None:
            df['rsi'] = rsi
        
        # MACD
        try:
            macd_data = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd_data is not None:
                df = df.join(macd_data, how='left')
        except Exception as e:
            logger.warning(f"Error calculating MACD: {e}")
        
        # Bollinger Bands
        try:
            bbands_data = ta.bbands(df['close'], length=20, std=2)
            if bbands_data is not None:
                df = df.join(bbands_data, how='left')
        except Exception as e:
            logger.warning(f"Error calculating Bollinger Bands: {e}")
        
        # Stochastic
        try:
            stoch_data = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
            if stoch_data is not None:
                df = df.join(stoch_data, how='left')
        except Exception as e:
            logger.warning(f"Error calculating Stochastic: {e}")
        
        # Volume indicators
        volume_sma = safe_indicator_calculation(df, ta.sma, df['volume'], length=20)
        if volume_sma is not None:
            df['volume_sma'] = volume_sma
        
        # Basic indicators
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
        
        # Ichimoku
        ichimoku_data = calculate_ichimoku(df)
        if ichimoku_data:
            for key, value in ichimoku_data.items():
                if value is not None:
                    df[key] = value
        
        # Fibonacci levels
        fib_levels = calculate_fibonacci_levels(df)
        if fib_levels:
            for level_name, level_value in fib_levels.items():
                df[level_name] = level_value
        
        # Advanced momentum indicators
        uo = calculate_ultimate_oscillator(df)
        if uo is not None:
            df['ultimate_oscillator'] = uo
        
        roc = calculate_rate_of_change(df)
        if roc is not None:
            df['roc'] = roc
        
        ao = calculate_awesome_oscillator(df)
        if ao is not None:
            df['awesome_oscillator'] = ao
        
        trix = calculate_trix(df)
        if trix is not None:
            df['trix'] = trix
        
        dpo = calculate_dpo(df)
        if dpo is not None:
            df['dpo'] = dpo
        
        # Advanced volume indicators
        obv = calculate_obv(df)
        if obv is not None:
            df['obv'] = obv
        
        ad = calculate_accumulation_distribution(df)
        if ad is not None:
            df['ad'] = ad
        
        cmf = calculate_chaikin_money_flow(df)
        if cmf is not None:
            df['cmf'] = cmf
        
        vpt = calculate_volume_price_trend(df)
        if vpt is not None:
            df['vpt'] = vpt
        
        eom = calculate_ease_of_movement(df)
        if eom is not None:
            df['eom'] = eom
        
        # Volatility indicators
        atr = calculate_average_true_range(df)
        if atr is not None:
            df['atr'] = atr
        
        keltner = calculate_keltner_channels(df)
        if keltner:
            for key, value in keltner.items():
                if value is not None:
                    df[key] = value
        
        donchian = calculate_donchian_channels(df)
        if donchian:
            for key, value in donchian.items():
                if value is not None:
                    df[key] = value
        
        std_dev = calculate_standard_deviation(df)
        if std_dev is not None:
            df['std_dev'] = std_dev
        
        # Advanced trend indicators
        supertrend = calculate_supertrend(df)
        if supertrend:
            for key, value in supertrend.items():
                if value is not None:
                    df[key] = value
        
        aroon = calculate_aroon_oscillator(df)
        if aroon is not None:
            df['aroon'] = aroon
        
        adx = calculate_adx(df)
        if adx is not None:
            df['adx'] = adx
        
        kama = calculate_kama(df)
        if kama is not None:
            df['kama'] = kama
        
        # Candlestick patterns
        hammer_doji = detect_hammer_doji_patterns(df)
        if hammer_doji:
            for key, value in hammer_doji.items():
                if value is not None:
                    df[key] = value
        
        engulfing = detect_engulfing_patterns(df)
        if engulfing:
            for key, value in engulfing.items():
                if value is not None:
                    df[key] = value
        
        star = detect_star_patterns(df)
        if star:
            for key, value in star.items():
                if value is not None:
                    df[key] = value
        
        # Market structure
        pivot = calculate_pivot_points(df)
        if pivot:
            for key, value in pivot.items():
                if value is not None:
                    df[key] = value
        
        support_resistance = calculate_support_resistance(df)
        if support_resistance:
            for key, value in support_resistance.items():
                if value is not None:
                    df[key] = value
        
        structure_breaks = detect_market_structure_breaks(df)
        if structure_breaks:
            for key, value in structure_breaks.items():
                if value is not None:
                    df[key] = value
        
        # Check required indicators
        required_indicators = ['rsi', 'sma50', 'volume_sma', 'atr']
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
        
        # RSI Analysis
        if rsi_value < 30:
            buy_signals += 2
            signal_strength += 2
            signal_details.append(f"RSI Oversold: {rsi_value:.1f}")
        elif rsi_value > 70:
            sell_signals += 2
            signal_strength += 2
            signal_details.append(f"RSI Overbought: {rsi_value:.1f}")
        
        # MACD Analysis
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
        
        # Bollinger Bands Analysis
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
        
        # Stochastic Analysis
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
        
        # MFI Analysis
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
        
        # CCI Analysis
        if 'cci' in df.columns and not pd.isna(last_row['cci']):
            cci_value = last_row['cci']
            
            if cci_value < -100:
                buy_signals += 1
                signal_details.append(f"CCI Oversold: {cci_value:.1f}")
            elif cci_value > 100:
                sell_signals += 1
                signal_details.append(f"CCI Overbought: {cci_value:.1f}")
        
        # Williams %R Analysis
        if 'williams_r' in df.columns and not pd.isna(last_row['williams_r']):
            wr_value = last_row['williams_r']
            
            if wr_value < -80:
                buy_signals += 1
                signal_details.append(f"Williams %R Oversold: {wr_value:.1f}")
            elif wr_value > -20:
                sell_signals += 1
                signal_details.append(f"Williams %R Overbought: {wr_value:.1f}")
        
        # Ultimate Oscillator Analysis
        if 'ultimate_oscillator' in df.columns and not pd.isna(last_row['ultimate_oscillator']):
            uo_value = last_row['ultimate_oscillator']
            
            if uo_value < 30:
                buy_signals += 1
                signal_details.append(f"Ultimate Oscillator Oversold: {uo_value:.1f}")
            elif uo_value > 70:
                sell_signals += 1
                signal_details.append(f"Ultimate Oscillator Overbought: {uo_value:.1f}")
        
        # Rate of Change Analysis
        if 'roc' in df.columns and not pd.isna(last_row['roc']):
            roc_value = last_row['roc']
            
            if roc_value > 5:
                buy_signals += 1
                signal_details.append(f"Strong Positive ROC: {roc_value:.1f}%")
            elif roc_value < -5:
                sell_signals += 1
                signal_details.append(f"Strong Negative ROC: {roc_value:.1f}%")
        
        # Awesome Oscillator Analysis
        if 'awesome_oscillator' in df.columns and not pd.isna(last_row['awesome_oscillator']) and not pd.isna(prev_row['awesome_oscillator']):
            ao_current = last_row['awesome_oscillator']
            ao_prev = prev_row['awesome_oscillator']
            
            if ao_current > 0 and ao_prev <= 0:
                buy_signals += 2
                signal_details.append("Awesome Oscillator Bullish Cross")
            elif ao_current < 0 and ao_prev >= 0:
                sell_signals += 2
                signal_details.append("Awesome Oscillator Bearish Cross")
        
        # Chaikin Money Flow Analysis
        if 'cmf' in df.columns and not pd.isna(last_row['cmf']):
            cmf_value = last_row['cmf']
            
            if cmf_value > 0.1:
                buy_signals += 1
                signal_details.append(f"Strong Money Inflow: {cmf_value:.3f}")
            elif cmf_value < -0.1:
                sell_signals += 1
                signal_details.append(f"Strong Money Outflow: {cmf_value:.3f}")
        
        # SuperTrend Analysis
        if 'supertrend' in df.columns and not pd.isna(last_row['supertrend']):
            supertrend_value = last_row['supertrend']
            
            if current_price > supertrend_value and prev_row['close'] <= df.iloc[-2].get('supertrend', 0):
                buy_signals += 2
                signal_strength += 2
                signal_details.append("SuperTrend Bullish Signal")
            elif current_price < supertrend_value and prev_row['close'] >= df.iloc[-2].get('supertrend', 0):
                sell_signals += 2
                signal_strength += 2
                signal_details.append("SuperTrend Bearish Signal")
        
        # ADX Trend Strength Analysis
        if 'adx' in df.columns and not pd.isna(last_row['adx']):
            adx_value = last_row['adx']
            
            if adx_value > 25:
                signal_strength += 1
                signal_details.append(f"Strong Trend (ADX: {adx_value:.1f})")
        
        # Aroon Oscillator Analysis
        if 'aroon' in df.columns and not pd.isna(last_row['aroon']):
            aroon_value = last_row['aroon']
            
            if aroon_value > 50:
                buy_signals += 1
                signal_details.append(f"Bullish Aroon: {aroon_value:.1f}")
            elif aroon_value < -50:
                sell_signals += 1
                signal_details.append(f"Bearish Aroon: {aroon_value:.1f}")
        
        # Candlestick Pattern Analysis
        pattern_signals = 0
        if 'hammer' in df.columns and last_row.get('hammer', False):
            buy_signals += 1
            pattern_signals += 1
            signal_details.append("Bullish Hammer Pattern")
        
        if 'doji' in df.columns and last_row.get('doji', False):
            signal_details.append("Doji Pattern (Indecision)")
        
        if 'bullish_engulfing' in df.columns and last_row.get('bullish_engulfing', False):
            buy_signals += 2
            pattern_signals += 2
            signal_details.append("Bullish Engulfing Pattern")
        
        if 'bearish_engulfing' in df.columns and last_row.get('bearish_engulfing', False):
            sell_signals += 2
            pattern_signals += 2
            signal_details.append("Bearish Engulfing Pattern")
        
        if 'morning_star' in df.columns and last_row.get('morning_star', False):
            buy_signals += 2
            pattern_signals += 2
            signal_details.append("Morning Star Pattern")
        
        if 'evening_star' in df.columns and last_row.get('evening_star', False):
            sell_signals += 2
            pattern_signals += 2
            signal_details.append("Evening Star Pattern")
        
        # Support/Resistance Analysis
        if 'support' in df.columns and 'resistance' in df.columns:
            support_level = last_row.get('support')
            resistance_level = last_row.get('resistance')
            
            if support_level and abs(current_price - support_level) / current_price < 0.005:
                buy_signals += 1
                signal_details.append(f"Near Support: {support_level:.6f}")
            
            if resistance_level and abs(current_price - resistance_level) / current_price < 0.005:
                sell_signals += 1
                signal_details.append(f"Near Resistance: {resistance_level:.6f}")
        
        # Market Structure Break Analysis
        if 'bullish_break' in df.columns and last_row.get('bullish_break', False):
            buy_signals += 2
            signal_strength += 1
            signal_details.append("Bullish Structure Break")
        
        if 'bearish_break' in df.columns and last_row.get('bearish_break', False):
            sell_signals += 2
            signal_strength += 1
            signal_details.append("Bearish Structure Break")
        
        # PSAR Analysis
        if 'psar' in df.columns and not pd.isna(last_row['psar']):
            psar_value = last_row['psar']
            
            if current_price > psar_value and prev_row['close'] <= df.iloc[-2].get('psar', 0):
                buy_signals += 2
                signal_strength += 1
                signal_details.append("PSAR Bullish Signal")
            elif current_price < psar_value and prev_row['close'] >= df.iloc[-2].get('psar', 0):
                sell_signals += 2
                signal_strength += 1
                signal_details.append("PSAR Bearish Signal")
        
        # Ichimoku Analysis
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
        
        # SMA Trend Analysis
        if ('sma20' in df.columns and 'sma50' in df.columns and
            not pd.isna(last_row['sma20']) and not pd.isna(last_row['sma50'])):
            
            if current_price > last_row['sma20'] > last_row['sma50']:
                buy_signals += 1
                signal_details.append("Price Above SMA20 & SMA50")
            elif current_price < last_row['sma20'] < last_row['sma50']:
                sell_signals += 1
                signal_details.append("Price Below SMA20 & SMA50")
        
        # Volume Analysis
        if 'volume_sma' in df.columns and not pd.isna(last_row['volume_sma']):
            volume_ratio = last_row['volume'] / last_row['volume_sma']
            if volume_ratio > 1.5:
                signal_strength += 1
                signal_details.append(f"High Volume: {volume_ratio:.1f}x")
        
        # Fibonacci Support/Resistance
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
        
        # Signal threshold with pattern bonus
        min_signal_threshold = 3 - pattern_signals  # Reduce threshold if strong patterns found
        min_signal_threshold = max(min_signal_threshold, 1)  # Minimum threshold of 1
        
        if buy_signals >= min_signal_threshold and buy_signals > sell_signals:
            return {
                'type': 'buy',
                'strength': min(signal_strength, 5),
                'rsi': rsi_value,
                'macd': last_row.get('MACD_12_26_9', 0),
                'method': 'Advanced_Multi_Indicator_Buy',
                'details': signal_details,
                'buy_score': buy_signals,
                'sell_score': sell_signals,
                'pattern_signals': pattern_signals
            }
        elif sell_signals >= min_signal_threshold and sell_signals > buy_signals:
            return {
                'type': 'sell',
                'strength': min(signal_strength, 5),
                'rsi': rsi_value,
                'macd': last_row.get('MACD_12_26_9', 0),
                'method': 'Advanced_Multi_Indicator_Sell',
                'details': signal_details,
                'buy_score': buy_signals,
                'sell_score': sell_signals,
                'pattern_signals': pattern_signals
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error checking signals for {symbol}: {e}")
        return None

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

# اندیکاتورهای مومنتوم اضافی
def calculate_ultimate_oscillator(df, period1=7, period2=14, period3=28):
    """محاسبه Ultimate Oscillator"""
    try:
        if df is None or len(df) < max(period1, period2, period3):
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        # True Low = minimum of Low or previous Close
        true_low = pd.concat([low, prev_close], axis=1).min(axis=1)
        
        # Buying Pressure = Close - True Low
        buying_pressure = close - true_low
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Average calculations for 3 periods
        bp1 = buying_pressure.rolling(window=period1).sum()
        tr1_sum = true_range.rolling(window=period1).sum()
        
        bp2 = buying_pressure.rolling(window=period2).sum()
        tr2_sum = true_range.rolling(window=period2).sum()
        
        bp3 = buying_pressure.rolling(window=period3).sum()
        tr3_sum = true_range.rolling(window=period3).sum()
        
        # Ultimate Oscillator formula
        uo = 100 * (4 * (bp1 / tr1_sum) + 2 * (bp2 / tr2_sum) + (bp3 / tr3_sum)) / 7
        
        return uo
    except Exception as e:
        logger.warning(f"Error calculating Ultimate Oscillator: {e}")
        return None

def calculate_rate_of_change(df, period=14):
    """محاسبه Rate of Change (ROC)"""
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        roc = ((close - close.shift(period)) / close.shift(period)) * 100
        
        return roc
    except Exception as e:
        logger.warning(f"Error calculating ROC: {e}")
        return None

def calculate_awesome_oscillator(df, fast_period=5, slow_period=34):
    """محاسبه Awesome Oscillator"""
    try:
        if df is None or len(df) < slow_period:
            return None
        
        median_price = (df['high'] + df['low']) / 2
        
        fast = median_price.rolling(window=fast_period).mean()
        slow = median_price.rolling(window=slow_period).mean()

        ao = fast - slow
        ao = ao.dropna()  # Remove NaN values
        
        if ao.empty:
            logger.warning("Awesome Oscillator calculation resulted in empty series")
            return None
        
        logger.info(f"Calculated Awesome Oscillator with {len(ao)} values")
        return ao
    except Exception as e:
        logger.warning(f"Error calculating Awesome Oscillator: {e}")
        return None

def calculate_trix(df, period=14):
    """محاسبه TRIX"""
    try:
        if df is None or len(df) < period * 3:
            return None
        
        close = df['close']
        
        # Triple smoothed EMA
        ema1 = close.ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        ema3 = ema2.ewm(span=period).mean()
        
        # TRIX = Rate of change of triple smoothed EMA
        trix = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 10000
        
        return trix
    except Exception as e:
        logger.warning(f"Error calculating TRIX: {e}")
        return None

def calculate_dpo(df, period=20):
    """محاسبه Detrended Price Oscillator"""
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        sma = close.rolling(window=period).mean()
        
        # DPO = Close - SMA shifted by (period/2 + 1)
        shift_period = int(period/2) + 1
        dpo = close - sma.shift(shift_period)
        
        dpo = dpo.dropna()  # Remove NaN values
        
        if dpo.empty:
            logger.warning("DPO calculation resulted in empty series")
            return None
        
        logger.info(f"Calculated DPO with {len(dpo)} values")
        
        return dpo
    except Exception as e:
        logger.warning(f"Error calculating DPO: {e}")
        return None

# ===== اندیکاتورهای حجم پیشرفته =====

def calculate_obv(df):
    """محاسبه On-Balance Volume"""
    try:
        if df is None or len(df) < 2:
            return None
        
        close = df['close']
        volume = df['volume']
        
        obv = []
        obv.append(0)  # مقدار اولیه
        
        for i in range(1, len(df)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[i-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[i-1] - volume.iloc[i])
            else:
                obv.append(obv[i-1])
        
        return pd.Series(obv, index=df.index)
    except Exception as e:
        logger.warning(f"Error calculating OBV: {e}")
        return None

def calculate_accumulation_distribution(df):
    """محاسبه Accumulation/Distribution Line"""
    try:
        if df is None or len(df) < 1:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        # Handle division by zero
        mfm = mfm.fillna(0)
        
        # Money Flow Volume
        mfv = mfm * volume
        
        # A/D Line is cumulative sum of MFV
        ad_line = mfv.cumsum()
        
        return ad_line
    except Exception as e:
        logger.warning(f"Error calculating A/D Line: {e}")
        return None

def calculate_ad_line(df):
    """محاسبه Accumulation/Distribution Line"""
    try:
        if df is None or len(df) < 1:
            return None
            
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # محاسبه Money Flow Multiplier
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # در صورت صفر بودن دامنه
        
        # محاسبه Money Flow Volume
        mfv = clv * volume
        
        # محاسبه A/D Line تجمعی
        ad_line = mfv.cumsum()
        
        return ad_line
    except Exception:
        return None

def calculate_chaikin_money_flow(df, period=20):
    """محاسبه Chaikin Money Flow"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)
        
        # Money Flow Volume
        mfv = mfm * volume
        
        # CMF = Sum of MFV over period / Sum of Volume over period
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        return cmf
    except Exception as e:
        logger.warning(f"Error calculating CMF: {e}")
        return None

def calculate_volume_price_trend(df):
    """محاسبه Volume Price Trend"""
    try:
        if df is None or len(df) < 2:
            return None
        
        close = df['close']
        volume = df['volume']
        
        # Price change percentage
        price_change_pct = (close - close.shift(1)) / close.shift(1)
        
        # VPT = Previous VPT + Volume * Price Change %
        vpt = (price_change_pct * volume).cumsum()
        
        return vpt
    except Exception as e:
        logger.warning(f"Error calculating VPT: {e}")
        return None

def calculate_ease_of_movement(df, period=14):
    """محاسبه Ease of Movement"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Distance Moved
        distance_moved = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2
        
        # Box Height
        box_height = (volume / 100000) / (high - low)
        
        # 1-Period EMV
        emv_1period = distance_moved / box_height
        emv_1period = emv_1period.replace([np.inf, -np.inf], 0).fillna(0)
        
        # EMV = SMA of 1-Period EMV
        emv = emv_1period.rolling(window=period).mean()
        
        return emv
    except Exception as e:
        logger.warning(f"Error calculating EMV: {e}")
        return None

# ===== اندیکاتورهای نوسان =====

def calculate_average_true_range(df, period=14):
    """محاسبه Average True Range"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    except Exception as e:
        logger.warning(f"Error calculating ATR: {e}")
        return None

def calculate_atr(df, period=14):
    """محاسبه Average True Range"""
    try:
        if df is None or len(df) < period:
            return None
            
        high = df['high']
        low = df['low']
        close = df['close']
        
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    except Exception:
        return None

def calculate_keltner_channels(df, period=20, multiplier=2):
    """محاسبه Keltner Channels"""
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        atr = calculate_average_true_range(df, period)
        
        if atr is None:
            return None
        
        middle_line = close.rolling(window=period).mean()
        upper_channel = middle_line + (multiplier * atr)
        lower_channel = middle_line - (multiplier * atr)
        
        return {
            'keltner_upper': upper_channel,
            'keltner_middle': middle_line,
            'keltner_lower': lower_channel
        }
    except Exception as e:
        logger.warning(f"Error calculating Keltner Channels: {e}")
        return None

def calculate_donchian_channels(df, period=20):
    """محاسبه Donchian Channels"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        
        upper_channel = high.rolling(window=period).max()
        lower_channel = low.rolling(window=period).min()
        middle_channel = (upper_channel + lower_channel) / 2
        
        return {
            'donchian_upper': upper_channel,
            'donchian_middle': middle_channel,
            'donchian_lower': lower_channel
        }
    except Exception as e:
        logger.warning(f"Error calculating Donchian Channels: {e}")
        return None

def calculate_standard_deviation(df, period=20):
    """محاسبه Standard Deviation"""
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        std_dev = close.rolling(window=period).std()
        
        return std_dev
    except Exception as e:
        logger.warning(f"Error calculating Standard Deviation: {e}")
        return None

def calculate_price_std(df, period=20):
    """محاسبه انحراف معیار قیمت"""
    try:
        if df is None or len(df) < period:
            return None
            
        close = df['close']
        std_dev = close.rolling(period).std()
        
        return std_dev
    except Exception:
        return None

# ===== اندیکاتورهای ترند پیشرفته =====

def calculate_supertrend(df, period=10, multiplier=3.0):
    """محاسبه Supertrend"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # محاسبه ATR
        atr = calculate_average_true_range(df, period)
        if atr is None:
            return None
        
        # محاسبه Basic Bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Final Bands
        final_upper_band = []
        final_lower_band = []
        supertrend = []
        direction = []
        
        for i in range(len(df)):
            if i == 0:
                final_upper_band.append(upper_band.iloc[i])
                final_lower_band.append(lower_band.iloc[i])
                supertrend.append(0)
                direction.append(1)
            else:
                # Final Upper Band
                if upper_band.iloc[i] < final_upper_band[i-1] or close.iloc[i-1] > final_upper_band[i-1]:
                    final_upper_band.append(upper_band.iloc[i])
                else:
                    final_upper_band.append(final_upper_band[i-1])
                
                # Final Lower Band
                if lower_band.iloc[i] > final_lower_band[i-1] or close.iloc[i-1] < final_lower_band[i-1]:
                    final_lower_band.append(lower_band.iloc[i])
                else:
                    final_lower_band.append(final_lower_band[i-1])
                
                # Direction and Supertrend
                if direction[i-1] == -1 and close.iloc[i] < final_lower_band[i]:
                    direction.append(-1)
                elif direction[i-1] == 1 and close.iloc[i] > final_upper_band[i]:
                    direction.append(1)
                elif direction[i-1] == -1 and close.iloc[i] >= final_lower_band[i]:
                    direction.append(1)
                elif direction[i-1] == 1 and close.iloc[i] <= final_upper_band[i]:
                    direction.append(-1)
                else:
                    direction.append(direction[i-1])
                
                if direction[i] == 1:
                    supertrend.append(final_lower_band[i])
                else:
                    supertrend.append(final_upper_band[i])
        
        return pd.Series(supertrend, index=df.index)
    except Exception as e:
        logger.warning(f"Error calculating Supertrend: {e}")
        return None

def calculate_aroon_oscillator(df, period=14):
    """محاسبه Aroon Oscillator"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        
        aroon_up = []
        aroon_down = []
        
        for i in range(len(df)):
            if i < period - 1:
                aroon_up.append(np.nan)
                aroon_down.append(np.nan)
            else:
                # محاسبه Aroon Up
                high_period = high.iloc[i-period+1:i+1]
                periods_since_high = period - 1 - high_period.idxmax()
                aroon_up_val = ((period - periods_since_high) / period) * 100
                aroon_up.append(aroon_up_val)
                
                # محاسبه Aroon Down
                low_period = low.iloc[i-period+1:i+1]
                periods_since_low = period - 1 - low_period.idxmin()
                aroon_down_val = ((period - periods_since_low) / period) * 100
                aroon_down.append(aroon_down_val)
        
        aroon_up = pd.Series(aroon_up, index=df.index)
        aroon_down = pd.Series(aroon_down, index=df.index)
        aroon_oscillator = aroon_up - aroon_down
        
        return {
            'aroon_up': aroon_up,
            'aroon_down': aroon_down,
            'aroon_oscillator': aroon_oscillator
        }
    except Exception as e:
        logger.warning(f"Error calculating Aroon Oscillator: {e}")
        return None

def calculate_aroon(df, period=14):
    """محاسبه Aroon Oscillator"""
    try:
        if df is None or len(df) < period:
            return None
            
        high = df['high']
        low = df['low']
        
        # پیدا کردن موقعیت بالاترین و پایین‌ترین قیمت
        aroon_up = ((period - high.rolling(period).apply(lambda x: period - 1 - x.argmax())) / period) * 100
        aroon_down = ((period - low.rolling(period).apply(lambda x: period - 1 - x.argmin())) / period) * 100
        
        aroon_oscillator = aroon_up - aroon_down
        
        return {
            'aroon_up': aroon_up,
            'aroon_down': aroon_down,
            'aroon_oscillator': aroon_oscillator
        }
    except Exception:
        return None

def calculate_adx(df, period=14):
    """محاسبه Average Directional Index (ADX)"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        
        # True Range
        atr = calculate_average_true_range(df, period)
        if atr is None:
            return None
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = pd.Series(0, index=df.index)
        minus_dm = pd.Series(0, index=df.index)
        
        plus_dm[up_move > down_move] = up_move[up_move > down_move]
        plus_dm[plus_dm < 0] = 0
        
        minus_dm[down_move > up_move] = down_move[down_move > up_move]
        minus_dm[minus_dm < 0] = 0
        
        # Smoothed DM
        plus_dm_smooth = plus_dm.rolling(window=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period).mean()
        
        # DI calculations
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)
        
        # ADX calculation
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = dx.fillna(0)
        adx = dx.rolling(window=period).mean()
        
        return {
            'plus_di': plus_di,
            'minus_di': minus_di,
            'adx': adx
        }
    except Exception as e:
        logger.warning(f"Error calculating ADX: {e}")
        return None

def calculate_kama(df, period=10, fast_sc=2, slow_sc=30):
    """محاسبه Kaufman Adaptive Moving Average"""
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        
        # Efficiency Ratio
        change = abs(close - close.shift(period))
        volatility = abs(close - close.shift(1)).rolling(window=period).sum()
        er = change / volatility
        er = er.fillna(0)
        
        # Smoothing Constants
        fastest_sc = 2.0 / (fast_sc + 1)
        slowest_sc = 2.0 / (slow_sc + 1)
        sc = (er * (fastest_sc - slowest_sc) + slowest_sc) ** 2
        
        # KAMA calculation
        kama = []
        kama.append(close.iloc[0])  # First value
        
        for i in range(1, len(close)):
            kama.append(kama[i-1] + sc.iloc[i] * (close.iloc[i] - kama[i-1]))
        
        return pd.Series(kama, index=df.index)
    except Exception as e:
        logger.warning(f"Error calculating KAMA: {e}")
        return None

# الگوهای کندل استیک
def detect_hammer_doji_patterns(df):
    """تشخیص الگوهای Hammer و Doji"""
    try:
        if df is None or len(df) < 3:
            return None
        
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # محاسبه اجزای کندل
        body = abs(close - open_price)
        upper_shadow = high - pd.concat([open_price, close], axis=1).max(axis=1)
        lower_shadow = pd.concat([open_price, close], axis=1).min(axis=1) - low
        total_range = high - low
        
        patterns = pd.DataFrame(index=df.index)
        
        # Hammer Pattern
        hammer_condition = (
            (lower_shadow >= 2 * body) &
            (upper_shadow <= 0.1 * total_range) &
            (body <= 0.3 * total_range)
        )
        patterns['hammer'] = hammer_condition
        
        # Doji Pattern
        doji_condition = (body <= 0.1 * total_range)
        patterns['doji'] = doji_condition
        
        # Shooting Star Pattern
        shooting_star_condition = (
            (upper_shadow >= 2 * body) &
            (lower_shadow <= 0.1 * total_range) &
            (body <= 0.3 * total_range)
        )
        patterns['shooting_star'] = shooting_star_condition
        
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting Hammer/Doji patterns: {e}")
        return None

def detect_engulfing_patterns(df):
    """تشخیص الگوهای Engulfing"""
    try:
        if df is None or len(df) < 2:
            return None
        
        open_price = df['open']
        close = df['close']
        
        patterns = pd.DataFrame(index=df.index)
        patterns['bullish_engulfing'] = False
        patterns['bearish_engulfing'] = False
        
        for i in range(1, len(df)):
            prev_open = open_price.iloc[i-1]
            prev_close = close.iloc[i-1]
            curr_open = open_price.iloc[i]
            curr_close = close.iloc[i]
            
            # Bullish Engulfing
            if (prev_close < prev_open and  # Previous red candle
                curr_close > curr_open and  # Current green candle
                curr_open < prev_close and  # Current opens below previous close
                curr_close > prev_open):    # Current closes above previous open
                patterns.iloc[i, patterns.columns.get_loc('bullish_engulfing')] = True
            
            # Bearish Engulfing
            if (prev_close > prev_open and  # Previous green candle
                curr_close < curr_open and  # Current red candle
                curr_open > prev_close and  # Current opens above previous close
                curr_close < prev_open):    # Current closes below previous open
                patterns.iloc[i, patterns.columns.get_loc('bearish_engulfing')] = True
        
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting Engulfing patterns: {e}")
        return None

def detect_star_patterns(df):
    """تشخیص الگوهای Morning/Evening Star"""
    try:
        if df is None or len(df) < 3:
            return None
        
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        patterns = pd.DataFrame(index=df.index)
        patterns['morning_star'] = False
        patterns['evening_star'] = False
        
        for i in range(2, len(df)):
            # Morning Star Pattern
            first_red = close.iloc[i-2] < open_price.iloc[i-2]  # First candle is red
            small_body = abs(close.iloc[i-1] - open_price.iloc[i-1]) < abs(close.iloc[i-2] - open_price.iloc[i-2]) * 0.3  # Small middle candle
            gap_down = high.iloc[i-1] < low.iloc[i-2]  # Gap down
            third_green = close.iloc[i] > open_price.iloc[i]  # Third candle is green
            closes_into_first = close.iloc[i] > (open_price.iloc[i-2] + close.iloc[i-2]) / 2  # Closes well into first candle
            
            if first_red and small_body and gap_down and third_green and closes_into_first:
                patterns.iloc[i, patterns.columns.get_loc('morning_star')] = True
            
            # Evening Star Pattern
            first_green = close.iloc[i-2] > open_price.iloc[i-2]  # First candle is green
            gap_up = low.iloc[i-1] > high.iloc[i-2]  # Gap up
            third_red = close.iloc[i] < open_price.iloc[i]  # Third candle is red
            closes_into_first = close.iloc[i] < (open_price.iloc[i-2] + close.iloc[i-2]) / 2  # Closes well into first candle
            
            if first_green and small_body and gap_up and third_red and closes_into_first:
                patterns.iloc[i, patterns.columns.get_loc('evening_star')] = True
        
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting Star patterns: {e}")
        return None

def detect_morning_evening_star(df):
    """تشخیص الگوهای Morning/Evening Star"""
    try:
        if df is None or len(df) < 3:
            return None
            
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # محاسبه بدنه کندل‌ها
        body = abs(close - open_price)
        body_1 = body.shift(1)
        body_2 = body.shift(2)
        
        # Morning Star Pattern
        morning_star = ((close.shift(2) < open_price.shift(2)) &  # کندل نزولی
                       (body_1 < body_2 * 0.3) &  # کندل کوچک میانی
                       (close > open_price) &  # کندل صعودی
                       (close > (close.shift(2) + open_price.shift(2)) / 2))
        
        # Evening Star Pattern
        evening_star = ((close.shift(2) > open_price.shift(2)) &  # کندل صعودی
                       (body_1 < body_2 * 0.3) &  # کندل کوچک میانی
                       (close < open_price) &  # کندل نزولی
                       (close < (close.shift(2) + open_price.shift(2)) / 2))
        
        return {
            'morning_star': morning_star,
            'evening_star': evening_star
        }
    except Exception:
        return None

# اندیکاتورهای مارکت استراکچر
def calculate_pivot_points(df):
    """محاسبه Pivot Points"""
    try:
        if df is None or len(df) < 1:
            return None
        
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        # Standard Pivot Points
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    except Exception as e:
        logger.warning(f"Error calculating Pivot Points: {e}")
        return None

def calculate_support_resistance(df, window=20):
    """محاسبه سطوح Support و Resistance"""
    try:
        if df is None or len(df) < window:
            return None
        
        high = df['high']
        low = df['low']
        
        # Local highs and lows
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(df) - window):
            # Check for local high (resistance)
            if high.iloc[i] == high.iloc[i-window:i+window+1].max():
                resistance_levels.append(high.iloc[i])
            
            # Check for local low (support)
            if low.iloc[i] == low.iloc[i-window:i+window+1].min():
                support_levels.append(low.iloc[i])
        
        # Get most significant levels
        resistance_levels = sorted(set(resistance_levels), reverse=True)[:5]
        support_levels = sorted(set(support_levels))[:5]
        
        return {
            'resistance_levels': resistance_levels,
            'support_levels': support_levels
        }
    except Exception as e:
        logger.warning(f"Error calculating Support/Resistance: {e}")
        return None

def detect_market_structure_breaks(df, swing_strength=5):
    """تشخیص Market Structure Breaks"""
    try:
        if df is None or len(df) < swing_strength * 2:
            return None
        
        high = df['high']
        low = df['low']
        
        structure_breaks = pd.DataFrame(index=df.index)
        structure_breaks['bullish_break'] = False
        structure_breaks['bearish_break'] = False
        
        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(swing_strength, len(df) - swing_strength):
            # Swing High
            if high.iloc[i] == high.iloc[i-swing_strength:i+swing_strength+1].max():
                swing_highs.append((i, high.iloc[i]))
            
            # Swing Low
            if low.iloc[i] == low.iloc[i-swing_strength:i+swing_strength+1].min():
                swing_lows.append((i, low.iloc[i]))
        
        # Detect breaks
        current_price = df['close'].iloc[-1]
        
        # Check if current price breaks recent swing high (bullish break)
        if swing_highs:
            recent_high = max(swing_highs[-3:], key=lambda x: x[1])[1] if len(swing_highs) >= 3 else swing_highs[-1][1]
            if current_price > recent_high:
                structure_breaks.iloc[-1, structure_breaks.columns.get_loc('bullish_break')] = True
        
        # Check if current price breaks recent swing low (bearish break)
        if swing_lows:
            recent_low = min(swing_lows[-3:], key=lambda x: x[1])[1] if len(swing_lows) >= 3 else swing_lows[-1][1]
            if current_price < recent_low:
                structure_breaks.iloc[-1, structure_breaks.columns.get_loc('bearish_break')] = True
        
        return structure_breaks
    except Exception as e:
        logger.warning(f"Error detecting Market Structure Breaks: {e}")
        return None

# ===== فیلترهای اضافی =====

def calculate_correlation_with_btc(df, btc_df, period=20):
    """محاسبه همبستگی با بیت کوین"""
    try:
        if df is None or btc_df is None or len(df) < period or len(btc_df) < period:
            return None
            
        # هم‌تراز کردن داده‌ها بر اساس زمان
        merged = pd.merge(df[['close']], btc_df[['close']], 
                         left_index=True, right_index=True, 
                         suffixes=('', '_btc'), how='inner')
        
        if len(merged) < period:
            return None
            
        # محاسبه همبستگی غلتان
        correlation = merged['close'].rolling(period).corr(merged['close_btc'])
        
        return correlation
    except Exception:
        return None

def detect_market_regime(df, lookback=50):
    """تشخیص رژیم بازار"""
    try:
        if df is None or len(df) < lookback:
            return None
            
        close = df['close']
        
        # محاسبه نوسانات
        returns = close.pct_change()
        volatility = returns.rolling(lookback).std() * np.sqrt(252)  # سالانه
        
        # محاسبه ترند
        sma_short = close.rolling(10).mean()
        sma_long = close.rolling(50).mean()
        trend = sma_short - sma_long
        
        # تعیین رژیم بازار
        regime = pd.Series(index=df.index, dtype=str)
        
        for i in range(lookback, len(df)):
            vol = volatility.iloc[i]
            tr = trend.iloc[i]
            
            if vol > volatility.rolling(lookback).quantile(0.75).iloc[i]:
                if tr > 0:
                    regime.iloc[i] = 'Bull_Volatile'
                else:
                    regime.iloc[i] = 'Bear_Volatile'
            else:
                if tr > 0:
                    regime.iloc[i] = 'Bull_Stable'
                else:
                    regime.iloc[i] = 'Bear_Stable'
        
        return regime
    except Exception:
        return None

# ===== ابزارهای ریسک منجمنت =====

def calculate_position_size_atr(capital, risk_percent, entry_price, atr_value, atr_multiplier=2):
    """محاسبه اندازه پوزیشن بر اساس ATR"""
    try:
        risk_amount = capital * (risk_percent / 100)
        stop_distance = atr_value * atr_multiplier
        position_size = risk_amount / stop_distance
        
        return min(position_size, capital * 0.1)  # حداکثر 10% سرمایه
    except Exception:
        return 0

def calculate_dynamic_stop_loss(df, entry_price, position_type='long', atr_multiplier=2):
    """محاسبه حد ضرر پویا"""
    try:
        if df is None or len(df) < 14:
            return entry_price * 0.95 if position_type == 'long' else entry_price * 1.05
            
        atr = calculate_atr(df)
        if atr is None:
            return entry_price * 0.95 if position_type == 'long' else entry_price * 1.05
            
        atr_value = atr.iloc[-1]
        
        if position_type == 'long':
            stop_loss = entry_price - (atr_value * atr_multiplier)
        else:
            stop_loss = entry_price + (atr_value * atr_multiplier)
            
        return stop_loss
    except Exception:
        return entry_price * 0.95 if position_type == 'long' else entry_price * 1.05

def optimize_risk_reward_ratio(entry_price, target_price, stop_loss, min_ratio=2.0):
    """بهینه‌سازی نسبت ریسک-ریوارد"""
    try:
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
        
        current_ratio = reward / risk if risk > 0 else 0
        
        if current_ratio < min_ratio:
            # تنظیم هدف برای دستیابی به نسبت حداقل
            if entry_price > stop_loss:  # long position
                new_target = entry_price + (risk * min_ratio)
            else:  # short position
                new_target = entry_price - (risk * min_ratio)
            
            return new_target
        
        return target_price
    except Exception:
        return target_price

# ===== تکنیک‌های بهبود دقت =====

def ensemble_signal_scoring(signals_dict, weights=None):
    """ترکیب چندین سیگنال با وزن‌دهی"""
    try:
        if not signals_dict:
            return 0
            
        if weights is None:
            weights = {key: 1 for key in signals_dict.keys()}
        
        total_score = 0
        total_weight = 0
        
        for signal_name, signal_value in signals_dict.items():
            if signal_name in weights and signal_value is not None:
                weight = weights[signal_name]
                total_score += signal_value * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    except Exception:
        return 0

def adaptive_threshold_calculator(df, indicator_values, percentile_low=20, percentile_high=80):
    """محاسبه آستانه‌های تطبیقی"""
    try:
        if df is None or indicator_values is None:
            return {'low': 30, 'high': 70}
            
        # محاسبه آستانه‌ها بر اساس توزیع تاریخی
        low_threshold = np.percentile(indicator_values.dropna(), percentile_low)
        high_threshold = np.percentile(indicator_values.dropna(), percentile_high)
        
        return {
            'low': low_threshold,
            'high': high_threshold
        }
    except Exception:
        return {'low': 30, 'high': 70}

def filter_false_signals(df, signal_data, min_volume_ratio=1.2, min_trend_strength=0.1):
    """فیلتر سیگنال‌های کاذب"""
    try:
        if df is None or not signal_data:
            return False
            
        last_row = df.iloc[-1]
        
        # فیلتر حجم
        if 'volume_sma' in df.columns and not pd.isna(last_row['volume_sma']):
            volume_ratio = last_row['volume'] / last_row['volume_sma']
            if volume_ratio < min_volume_ratio:
                return False
        
        # فیلتر قدرت ترند
        if len(df) >= 10:
            recent_closes = df['close'].tail(10).values
            trend_strength = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
            
            if signal_data['type'] == 'buy' and trend_strength < -min_trend_strength:
                return False
            elif signal_data['type'] == 'sell' and trend_strength > min_trend_strength:
                return False
        
        return True
    except Exception:
        return True

def calculate_indicators(df, btc_df=None):
    """
    Calculate technical indicators for a given DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame with OHLCV data.
        btc_df (pd.DataFrame, optional): Bitcoin OHLCV data for correlation calculation.
    Returns:
        pd.DataFrame: DataFrame with calculated indicators or None if errors occur.
    """
    try:
        if df is None or len(df) < 200:
            logger.warning(f"Insufficient data for indicators: {len(df) if df is not None else 0} candles")
            return None
        
        df = df.copy()
        
        # Simple Moving Averages
        df['sma20'] = safe_indicator_calculation(df, ta.sma, df['close'], length=20)
        df['sma50'] = safe_indicator_calculation(df, ta.sma, df['close'], length=50)
        df['sma200'] = safe_indicator_calculation(df, ta.sma, df['close'], length=200)
        
        # Exponential Moving Averages
        df['ema12'] = safe_indicator_calculation(df, ta.ema, df['close'], length=12)
        df['ema26'] = safe_indicator_calculation(df, ta.ema, df['close'], length=26)
        df['ema50'] = safe_indicator_calculation(df, ta.ema, df['close'], length=50)
        
        # Weighted Moving Average
        df['wma20'] = safe_indicator_calculation(df, ta.wma, df['close'], length=20)
        
        # RSI
        rsi = safe_indicator_calculation(df, ta.rsi, df['close'], length=14)
        if rsi is not None:
            df['rsi'] = rsi.fillna(50)  # Fill NaN with neutral value
        
        # MACD
        try:
            macd_data = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd_data is not None:
                df = df.join(macd_data.fillna(0), how='left')  # Fill NaN with 0
        except Exception as e:
            logger.warning(f"Error calculating MACD: {e}")
        
        # Bollinger Bands
        try:
            bbands_data = ta.bbands(df['close'], length=20, std=2)
            if bbands_data is not None:
                df = df.join(bbands_data.fillna(0), how='left')  # Fill NaN with 0
        except Exception as e:
            logger.warning(f"Error calculating Bollinger Bands: {e}")
        
        # Stochastic
        try:
            stoch_data = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
            if stoch_data is not None:
                df = df.join(stoch_data.fillna(50), how='left')  # Fill NaN with 50
        except Exception as e:
            logger.warning(f"Error calculating Stochastic: {e}")
        
        # Volume indicators
        volume_sma = safe_indicator_calculation(df, ta.sma, df['volume'], length=20)
        if volume_sma is not None:
            df['volume_sma'] = volume_sma.fillna(0)  # Fill NaN with 0
        
        # Basic indicators
        mfi = calculate_money_flow_index(df)
        if mfi is not None:
            df['mfi'] = mfi.fillna(50)  # Fill NaN with 50
        
        cci = calculate_commodity_channel_index(df)
        if cci is not None:
            df['cci'] = cci.fillna(0)  # Fill NaN with 0
        
        williams_r = calculate_williams_r(df)
        if williams_r is not None:
            df['williams_r'] = williams_r.fillna(-50)  # Fill NaN with -50
        
        psar = calculate_parabolic_sar(df)
        if psar is not None:
            df['psar'] = psar.fillna(df['close'])  # Fill NaN with close price
        
        # Ichimoku
        ichimoku_data = calculate_ichimoku(df)
        if ichimoku_data:
            for key, value in ichimoku_data.items():
                if value is not None:
                    df[key] = value.fillna(0)  # Fill NaN with 0
        
        # Fibonacci levels
        fib_levels = calculate_fibonacci_levels(df)
        if fib_levels:
            for level_name, level_value in fib_levels.items():
                df[level_name] = level_value
        
        # Advanced momentum indicators
        uo = calculate_ultimate_oscillator(df)
        if uo is not None:
            df['ultimate_oscillator'] = uo.fillna(50)  # Fill NaN with 50
        
        roc = calculate_rate_of_change(df)
        if roc is not None:
            df['roc'] = roc.fillna(0)  # Fill NaN with 0
        
        ao = calculate_awesome_oscillator(df)
        if ao is not None:
            df['awesome_oscillator'] = ao.fillna(0)  # Fill NaN with 0
        
        trix = calculate_trix(df)
        if trix is not None:
            df['trix'] = trix.fillna(0)  # Fill NaN with 0
        
        dpo = calculate_dpo(df)
        if dpo is not None:
            df['dpo'] = dpo.fillna(0)  # Fill NaN with 0
        
        # Advanced volume indicators
        obv = calculate_obv(df)
        if obv is not None:
            df['obv'] = obv.fillna(0)  # Fill NaN with 0
        
        ad = calculate_accumulation_distribution(df)
        if ad is not None:
            df['ad'] = ad.fillna(0)  # Fill NaN with 0
        
        cmf = calculate_chaikin_money_flow(df)
        if cmf is not None:
            df['cmf'] = cmf.fillna(0)  # Fill NaN with 0
        
        vpt = calculate_volume_price_trend(df)
        if vpt is not None:
            df['vpt'] = vpt.fillna(0)  # Fill NaN with 0
        
        eom = calculate_ease_of_movement(df)
        if eom is not None:
            df['eom'] = eom.fillna(0)  # Fill NaN with 0
        
# Volatility indicators
        atr = calculate_average_true_range(df)
        if atr is not None:
            df['atr'] = atr
        
        keltner = calculate_keltner_channels(df)
        if keltner is not None:
            for key, value in keltner.items():
                df[key] = value.fillna(0)
        
        donchian = calculate_donchian_channels(df)
        if donchian is not None:
            for key, value in donchian.items():
                df[key] = value.fillna(0)
        
        std_dev = calculate_standard_deviation(df)
        if std_dev is not None:
            df['std_dev'] = std_dev.fillna(0)
        
        # Advanced trend indicators
        supertrend = calculate_supertrend(df)
        if supertrend is not None:
            df['supertrend'] = supertrend.fillna(0)
        
        # Aroon Oscillator
        aroon_osc = calculate_aroon_oscillator(df)
        if aroon_osc is not None:
            df['aroon_up'] = aroon_osc['aroon_up'].fillna(0)
            df['aroon_down'] = aroon_osc['aroon_down'].fillna(0)
            df['aroon_oscillator'] = aroon_osc['aroon_oscillator'].fillna(0)
        
        adx = calculate_adx(df)
        if adx is not None:
            df['adx'] = adx['adx'].fillna(0)
            df['plus_di'] = adx['plus_di'].fillna(0)
            df['minus_di'] = adx['minus_di'].fillna(0)
        
        kama = calculate_kama(df)
        if kama is not None:
            df['kama'] = kama.fillna(0)
        
        # Candlestick patterns
        hammer_doji = detect_hammer_doji_patterns(df)
        if hammer_doji is not None:
            df['hammer'] = hammer_doji['hammer'].fillna(False)
            df['doji'] = hammer_doji['doji'].fillna(False)
            df['shooting_star'] = hammer_doji['shooting_star'].fillna(False)
        
        engulfing = detect_engulfing_patterns(df)
        if engulfing is not None:
            df['bullish_engulfing'] = engulfing['bullish_engulfing'].fillna(False)
            df['bearish_engulfing'] = engulfing['bearish_engulfing'].fillna(False)
        
        star = detect_star_patterns(df)
        if star is not None:
            df['morning_star'] = star['morning_star'].fillna(False)
            df['evening_star'] = star['evening_star'].fillna(False)
        
        # Market structure
        pivot = calculate_pivot_points(df)
        if pivot is not None:
            for key, value in pivot.items():
                df[key] = value
        
        support_resistance = calculate_support_resistance(df)
        if support_resistance is not None:
            df['support'] = support_resistance['support_levels'][0] if support_resistance['support_levels'] else None
            df['resistance'] = support_resistance['resistance_levels'][0] if support_resistance['resistance_levels'] else None
        
        structure_breaks = detect_market_structure_breaks(df)
        if structure_breaks is not None:
            df['bullish_break'] = structure_breaks['bullish_break'].fillna(False)
            df['bearish_break'] = structure_breaks['bearish_break'].fillna(False)
        
        # New indicators
        # Accumulation/Distribution Line
        ad_line = calculate_ad_line(df)
        if ad_line is not None:
            df['ad_line'] = ad_line.fillna(0)  # Fill NaN with 0
        
        # Correlation with BTC (if btc_df is provided)
        if btc_df is not None:
            btc_corr = calculate_correlation_with_btc(df, btc_df)
            if btc_corr is not None:
                df['btc_correlation'] = btc_corr.fillna(0)  # Fill NaN with 0
        
        # Market regime
        market_regime = detect_market_regime(df)
        if market_regime is not None:
            df['market_regime'] = market_regime.fillna('neutral')  # Fill NaN with 'neutral'
        
        # Check required indicators
        required_indicators = ['rsi', 'sma50', 'volume_sma', 'atr']
        missing_indicators = [ind for ind in required_indicators if ind not in df.columns or df[ind].isnull().all()]
        
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
        
        # RSI Analysis
        if rsi_value < 30:
            buy_signals += 2
            signal_strength += 2
            signal_details.append(f"RSI Oversold: {rsi_value:.1f}")
        elif rsi_value > 70:
            sell_signals += 2
            signal_strength += 2
            signal_details.append(f"RSI Overbought: {rsi_value:.1f}")
        
        # MACD Analysis
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
        
        # Bollinger Bands Analysis
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
        
        # Stochastic Analysis
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
        
        # MFI Analysis
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
        
        # CCI Analysis
        if 'cci' in df.columns and not pd.isna(last_row['cci']):
            cci_value = last_row['cci']
            
            if cci_value < -100:
                buy_signals += 1
                signal_details.append(f"CCI Oversold: {cci_value:.1f}")
            elif cci_value > 100:
                sell_signals += 1
                signal_details.append(f"CCI Overbought: {cci_value:.1f}")
        
        # Williams %R Analysis
        if 'williams_r' in df.columns and not pd.isna(last_row['williams_r']):
            wr_value = last_row['williams_r']
            
            if wr_value < -80:
                buy_signals += 1
                signal_details.append(f"Williams %R Oversold: {wr_value:.1f}")
            elif wr_value > -20:
                sell_signals += 1
                signal_details.append(f"Williams %R Overbought: {wr_value:.1f}")
        
        # Ultimate Oscillator Analysis
        if 'ultimate_oscillator' in df.columns and not pd.isna(last_row['ultimate_oscillator']):
            uo_value = last_row['ultimate_oscillator']
            
            if uo_value < 30:
                buy_signals += 1
                signal_details.append(f"Ultimate Oscillator Oversold: {uo_value:.1f}")
            elif uo_value > 70:
                sell_signals += 1
                signal_details.append(f"Ultimate Oscillator Overbought: {uo_value:.1f}")
        
        # Rate of Change Analysis
        if 'roc' in df.columns and not pd.isna(last_row['roc']):
            roc_value = last_row['roc']
            
            if roc_value > 5:
                buy_signals += 1
                signal_details.append(f"Strong Positive ROC: {roc_value:.1f}%")
            elif roc_value < -5:
                sell_signals += 1
                signal_details.append(f"Strong Negative ROC: {roc_value:.1f}%")
        
        # Awesome Oscillator Analysis
        if 'awesome_oscillator' in df.columns and not pd.isna(last_row['awesome_oscillator']) and not pd.isna(prev_row['awesome_oscillator']):
            ao_current = last_row['awesome_oscillator']
            ao_prev = prev_row['awesome_oscillator']
            
            if ao_current > 0 and ao_prev <= 0:
                buy_signals += 2
                signal_details.append("Awesome Oscillator Bullish Cross")
            elif ao_current < 0 and ao_prev >= 0:
                sell_signals += 2
                signal_details.append("Awesome Oscillator Bearish Cross")
        
        # Chaikin Money Flow Analysis
        if 'cmf' in df.columns and not pd.isna(last_row['cmf']):
            cmf_value = last_row['cmf']
            
            if cmf_value > 0.1:
                buy_signals += 1
                signal_details.append(f"Strong Money Inflow: {cmf_value:.3f}")
            elif cmf_value < -0.1:
                sell_signals += 1
                signal_details.append(f"Strong Money Outflow: {cmf_value:.3f}")
        
        # SuperTrend Analysis
        if 'supertrend' in df.columns and not pd.isna(last_row['supertrend']):
            supertrend_value = last_row['supertrend']
            
            if current_price > supertrend_value and prev_row['close'] <= df.iloc[-2].get('supertrend', 0):
                buy_signals += 2
                signal_strength += 2
                signal_details.append("SuperTrend Bullish Signal")
            elif current_price < supertrend_value and prev_row['close'] >= df.iloc[-2].get('supertrend', 0):
                sell_signals += 2
                signal_strength += 2
                signal_details.append("SuperTrend Bearish Signal")
        
        # ADX Trend Strength Analysis
        if 'adx' in df.columns and not pd.isna(last_row['adx']):
            adx_value = last_row['adx']
            
            if adx_value > 25:
                signal_strength += 1
                signal_details.append(f"Strong Trend (ADX: {adx_value:.1f})")
        
        # Aroon Oscillator Analysis
        if 'aroon' in df.columns and not pd.isna(last_row['aroon']):
            aroon_value = last_row['aroon']
            
            if aroon_value > 50:
                buy_signals += 1
                signal_details.append(f"Bullish Aroon: {aroon_value:.1f}")
            elif aroon_value < -50:
                sell_signals += 1
                signal_details.append(f"Bearish Aroon: {aroon_value:.1f}")
        
        # Candlestick Pattern Analysis
        pattern_signals = 0
        if 'hammer' in df.columns and last_row.get('hammer', False):
            buy_signals += 1
            pattern_signals += 1
            signal_details.append("Bullish Hammer Pattern")
        
        if 'doji' in df.columns and last_row.get('doji', False):
            signal_details.append("Doji Pattern (Indecision)")
        
        if 'bullish_engulfing' in df.columns and last_row.get('bullish_engulfing', False):
            buy_signals += 2
            pattern_signals += 2
            signal_details.append("Bullish Engulfing Pattern")
        
        if 'bearish_engulfing' in df.columns and last_row.get('bearish_engulfing', False):
            sell_signals += 2
            pattern_signals += 2
            signal_details.append("Bearish Engulfing Pattern")
        
        if 'morning_star' in df.columns and last_row.get('morning_star', False):
            buy_signals += 2
            pattern_signals += 2
            signal_details.append("Morning Star Pattern")
        
        if 'evening_star' in df.columns and last_row.get('evening_star', False):
            sell_signals += 2
            pattern_signals += 2
            signal_details.append("Evening Star Pattern")
        
        # Support/Resistance Analysis
        if 'support' in df.columns and 'resistance' in df.columns:
            support_level = last_row.get('support')
            resistance_level = last_row.get('resistance')
            
            if support_level and abs(current_price - support_level) / current_price < 0.005:
                buy_signals += 1
                signal_details.append(f"Near Support: {support_level:.6f}")
            
            if resistance_level and abs(current_price - resistance_level) / current_price < 0.005:
                sell_signals += 1
                signal_details.append(f"Near Resistance: {resistance_level:.6f}")
        
        # Market Structure Break Analysis
        if 'bullish_break' in df.columns and last_row.get('bullish_break', False):
            buy_signals += 2
            signal_strength += 1
            signal_details.append("Bullish Structure Break")
        
        if 'bearish_break' in df.columns and last_row.get('bearish_break', False):
            sell_signals += 2
            signal_strength += 1
            signal_details.append("Bearish Structure Break")
        
        # PSAR Analysis
        if 'psar' in df.columns and not pd.isna(last_row['psar']):
            psar_value = last_row['psar']
            
            if current_price > psar_value and prev_row['close'] <= df.iloc[-2].get('psar', 0):
                buy_signals += 2
                signal_strength += 1
                signal_details.append("PSAR Bullish Signal")
            elif current_price < psar_value and prev_row['close'] >= df.iloc[-2].get('psar', 0):
                sell_signals += 2
                signal_strength += 1
                signal_details.append("PSAR Bearish Signal")
        
        # Ichimoku Analysis
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
        
        # SMA Trend Analysis
        if ('sma20' in df.columns and 'sma50' in df.columns and
            not pd.isna(last_row['sma20']) and not pd.isna(last_row['sma50'])):
            
            if current_price > last_row['sma20'] > last_row['sma50']:
                buy_signals += 1
                signal_details.append("Price Above SMA20 & SMA50")
            elif current_price < last_row['sma20'] < last_row['sma50']:
                sell_signals += 1
                signal_details.append("Price Below SMA20 & SMA50")
        
        # Volume Analysis
        if 'volume_sma' in df.columns and not pd.isna(last_row['volume_sma']):
            volume_ratio = last_row['volume'] / last_row['volume_sma']
            if volume_ratio > 1.5:
                signal_strength += 1
                signal_details.append(f"High Volume: {volume_ratio:.1f}x")
        
        # Fibonacci Support/Resistance
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
        
        # Signal threshold with pattern bonus
        min_signal_threshold = 3 - pattern_signals  # Reduce threshold if strong patterns found
        min_signal_threshold = max(min_signal_threshold, 1)  # Minimum threshold of 1
        
        if buy_signals >= min_signal_threshold and buy_signals > sell_signals:
            return {
                'type': 'buy',
                'strength': min(signal_strength, 5),
                'rsi': rsi_value,
                'macd': last_row.get('MACD_12_26_9', 0),
                'method': 'Advanced_Multi_Indicator_Buy',
                'details': signal_details,
                'buy_score': buy_signals,
                'sell_score': sell_signals,
                'pattern_signals': pattern_signals
            }
        elif sell_signals >= min_signal_threshold and sell_signals > buy_signals:
            return {
                'type': 'sell',
                'strength': min(signal_strength, 5),
                'rsi': rsi_value,
                'macd': last_row.get('MACD_12_26_9', 0),
                'method': 'Advanced_Multi_Indicator_Sell',
                'details': signal_details,
                'buy_score': buy_signals,
                'sell_score': sell_signals,
                'pattern_signals': pattern_signals
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
            
            # ساختار پیام بهینه‌شده با شاخص های اضافی
            message = f"🎯 *بهترین فرصت معاملاتی یافت شده*\n"
            message += f"{'='*30}\n\n"
            
            message += f"{emoji} *{sig['type']} {sig['symbol']}* {type_color}\n"
            message += f"🏆 **امتیاز دقت: {sig['accuracy_score']}/100**\n\n"
            
            message += f"📊 **جزئیات معاملاتی:**\n"
            message += f"💰 قیمت ورودی: `{sig['entry']:.6f}`\n"
            message += f"🎯 هدف قیمت: `{sig['target']:.6f}` (+{profit_pct:.1f}%)\n"
            message += f"🛑 حد ضرر: `{sig['stop_loss']:.6f}` (-{loss_pct:.1f}%)\n\n"
            
            message += f"📈 **تحلیل تکنیکال پیشرفته:**\n"
            message += f"• RSI: `{sig['rsi']:.1f}` "
            if sig['rsi'] < 30:
                message += "🟢 (فروش بیش از حد)"
            elif sig['rsi'] > 70:
                message += "🔴 (خرید بیش از حد)"
            else:
                message += "🟡 (متعادل)"
            message += "\n"
            
            message += f"• MACD: `{sig['macd']:.6f}` "
            if sig['macd'] > 0:
                message += "🟢 (صعودی)"
            else:
                message += "🔴 (نزولی)"
            message += "\n"
            
            # نمایش شاخص‌های اضافی اگر در سیگنال موجود باشند
            if 'stoch_k' in sig:
                message += f"• Stochastic K: `{sig['stoch_k']:.1f}` "
                if sig['stoch_k'] < 20:
                    message += "🟢 (فروش بیش از حد)"
                elif sig['stoch_k'] > 80:
                    message += "🔴 (خرید بیش از حد)"
                else:
                    message += "🟡 (متعادل)"
                message += "\n"
            
            if 'mfi' in sig:
                message += f"• MFI: `{sig['mfi']:.1f}` "
                if sig['mfi'] < 20:
                    message += "🟢 (جریان پول خروجی قوی)"
                elif sig['mfi'] > 80:
                    message += "🔴 (جریان پول ورودی قوی)"
                else:
                    message += "🟡 (متعادل)"
                message += "\n"
            
            if 'cci' in sig:
                message += f"• CCI: `{sig['cci']:.1f}` "
                if sig['cci'] < -100:
                    message += "🟢 (فروش بیش از حد)"
                elif sig['cci'] > 100:
                    message += "🔴 (خرید بیش از حد)"
                else:
                    message += "🟡 (محدوده طبیعی)"
                message += "\n"
            
            if 'williams_r' in sig:
                message += f"• Williams %R: `{sig['williams_r']:.1f}` "
                if sig['williams_r'] < -80:
                    message += "🟢 (فروش بیش از حد)"
                elif sig['williams_r'] > -20:
                    message += "🔴 (خرید بیش از حد)"
                else:
                    message += "🟡 (متعادل)"
                message += "\n"
            
            if 'volume_ratio' in sig:
                message += f"• نسبت حجم: `{sig['volume_ratio']:.1f}x` "
                if sig['volume_ratio'] > 2:
                    message += "🟢 (حجم بالا)"
                elif sig['volume_ratio'] > 1.5:
                    message += "🟡 (حجم متوسط)"
                else:
                    message += "⚪ (حجم پایین)"
                message += "\n"
            
            message += f"• روش تحلیل: `{sig['method']}`\n"
            message += f"• قدرت سیگنال: {'⭐' * sig['strength']} ({sig['strength']}/5)\n"
            
            if 'trend_direction' in sig:
                message += f"• جهت ترند: "
                if sig['trend_direction'] > 0:
                    message += "🟢 صعودی"
                elif sig['trend_direction'] < 0:
                    message += "🔴 نزولی"
                else:
                    message += "🟡 بغل"
                message += f" (قدرت: {abs(sig['trend_direction']):.1f})\n"
            
            # نمایش سطوح فیبوناچی اگر موجود باشند
            if 'fibonacci_levels' in sig and sig['fibonacci_levels']:
                message += f"\n🎯 **سطوح فیبوناچی نزدیک:**\n"
                for level in sig['fibonacci_levels']:
                    message += f"• {level}\n"
            
            # نمایش امتیازهای خرید و فروش
            if 'buy_score' in sig and 'sell_score' in sig:
                message += f"\n🎯 **امتیاز سیگنال‌ها:**\n"
                message += f"• امتیاز خرید: `{sig['buy_score']}`\n"
                message += f"• امتیاز فروش: `{sig['sell_score']}`\n"
                message += f"• برتری: {sig['buy_score'] - sig['sell_score']:+d} به نفع {'خرید' if sig['buy_score'] > sig['sell_score'] else 'فروش'}\n"
            
            message += f"\n⏰ زمان تولید سیگنال: `{sig['timestamp']}`\n"
            
            # اضافه کردن هشدار ریسک
            message += f"\n⚠️ **مدیریت ریسک:**\n"
            message += f"• نسبت سود به ضرر: {profit_pct/loss_pct:.1f}:1\n"
            message += f"• احتمال موفقیت: {sig['accuracy_score']}%\n"
            message += f"• ریسک توصیه شده: حداکثر 2-3% از سرمایه\n"
            
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

async def analyze_market():
    """تحلیل بازار و بازگرداندن بهترین سیگنال با اطلاعات تکمیلی"""
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
                            
                            # اضافه کردن اطلاعات تکمیلی از DataFrame
                            last_row = df.iloc[-1]
                            prev_rows = df.iloc[-10:] if len(df) >= 10 else df
                            
                            # محاسبه جهت ترند
                            trend_direction = 0
                            if len(prev_rows) >= 5:
                                close_prices = prev_rows['close'].values
                                for j in range(1, len(close_prices)):
                                    if close_prices[j] > close_prices[j-1]:
                                        trend_direction += 1
                                    elif close_prices[j] < close_prices[j-1]:
                                        trend_direction -= 1
                                trend_direction = trend_direction / len(close_prices)
                            
                            # اضافه کردن شاخص‌های تکمیلی
                            enhanced_signal = {
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
                                'timestamp': datetime.now().strftime('%H:%M:%S'),
                                'buy_score': signal_data.get('buy_score', 0),
                                'sell_score': signal_data.get('sell_score', 0),
                                'trend_direction': trend_direction
                            }
                            
                            # اضافه کردن شاخص‌های اضافی اگر موجود باشند
                            if 'STOCHk_14_3_3' in df.columns and not pd.isna(last_row['STOCHk_14_3_3']):
                                enhanced_signal['stoch_k'] = last_row['STOCHk_14_3_3']
                            
                            if 'mfi' in df.columns and not pd.isna(last_row['mfi']):
                                enhanced_signal['mfi'] = last_row['mfi']
                            
                            if 'cci' in df.columns and not pd.isna(last_row['cci']):
                                enhanced_signal['cci'] = last_row['cci']
                            
                            if 'williams_r' in df.columns and not pd.isna(last_row['williams_r']):
                                enhanced_signal['williams_r'] = last_row['williams_r']
                            
                            if 'volume_sma' in df.columns and not pd.isna(last_row['volume_sma']):
                                try:
                                    volume_ratio = last_row['volume'] / last_row['volume_sma']
                                    enhanced_signal['volume_ratio'] = volume_ratio
                                except (ZeroDivisionError, TypeError):
                                    pass
                            
                            # اضافه کردن سطوح فیبوناچی نزدیک
                            fibonacci_levels = []
                            fib_keys = ['fib_236', 'fib_382', 'fib_500', 'fib_618']
                            for fib_key in fib_keys:
                                if fib_key in df.columns:
                                    fib_level = last_row[fib_key]
                                    price_diff_pct = abs(current_price - fib_level) / current_price * 100
                                    if price_diff_pct < 2:  # اگر کمتر از 2% فاصله داشته باشد
                                        fibonacci_levels.append(f"{fib_key.replace('fib_', 'Fib ')}: {fib_level:.6f}")
                            
                            if fibonacci_levels:
                                enhanced_signal['fibonacci_levels'] = fibonacci_levels
                            
                            all_signals.append(enhanced_signal)
                            
                            logger.info(f"Enhanced signal found for {symbol}: {signal_type} (Score: {accuracy_score})")
                
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
        message += "📊 *شاخص‌های تحلیلی:*\n"
        message += "• RSI (قدرت نسبی)\n"
        message += "• MACD (همگرایی واگرایی)\n"
        message += "• Stochastic (نوسانگر)\n"
        message += "• MFI (شاخص جریان پول)\n"
        message += "• CCI (شاخص کانال کالا)\n"
        message += "• Williams %R\n"
        message += "• Fibonacci Levels\n"
        message += "• Volume Analysis\n\n"
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
        
        message += f"\n💡 برای تغییر نمادها، فایل `symbols.txt` را ویرایش کنید.\n"
        message += f"🎯 هر نماد با {len(['RSI', 'MACD', 'Stochastic', 'MFI', 'CCI', 'Williams %R', 'Fibonacci', 'Volume'])} شاخص تکنیکال تحلیل می‌شود."
        
        await update.message.reply_text(message, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in symbols command: {e}")
        await update.message.reply_text("خطایی در نمایش نمادها رخ داد.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command"""
    try:
        message = (
            "🤖 *راهنمای ربات تحلیل تکنیکال پیشرفته*\n\n"
            "این ربات با استفاده از 8+ اندیکاتور تکنیکال، بهترین فرصت‌های معاملاتی را شناسایی می‌کند.\n\n"
            "📋 *دستورات:*\n"
            "`/start` - شروع تحلیل بازار\n"
            "`/status` - نمایش وضعیت ربات\n"
            "`/symbols` - لیست نمادهای تحت نظارت\n"
            "`/help` - نمایش این راهنما\n\n"
            "📊 *اندیکاتورهای تحلیلی:*\n"
            "🔹 **RSI** - شناسایی مناطق فروش/خرید بیش از حد\n"
            "🔹 **MACD** - تشخیص تغییر روند بازار\n"
            "🔹 **Stochastic** - نوسانگر قدرتمند برای ورود/خروج\n"
            "🔹 **MFI** - تحلیل جریان پول هوشمند\n"
            "🔹 **CCI** - شاخص قدرت روند\n"
            "🔹 **Williams %R** - تایید سیگنال‌های اصلی\n"
            "🔹 **Fibonacci** - سطوح حمایت و مقاومت\n"
            "🔹 **Volume Analysis** - تحلیل حجم معاملات\n\n"
            "🎯 *ویژگی‌های خاص:*\n"
            "• سیستم امتیازدهی پیشرفته (0-100)\n"
            "• تحلیل چندگانه شاخص‌ها\n"
            "• محاسبه نسبت سود به ضرر\n"
            "• شناسایی قدرت روند\n"
            "• تشخیص سطوح فیبوناچی\n"
            "• مدیریت ریسک هوشمند\n\n"
            "⚠️ *هشدار مهم:*\n"
            "این سیگنال‌ها صرفاً جهت اطلاع‌رسانی هستند و توصیه سرمایه‌گذاری محسوب نمی‌شوند. "
            "لطفاً قبل از هر معامله، تحلیل‌های خود را انجام دهید.\n\n"
            "💰 *مدیریت ریسک:*\n"
            "• حداکثر 2-3% از سرمایه را ریسک کنید\n"
            "• همیشه Stop Loss تعیین کنید\n"
            "• از سیگنال‌های بالای 60 امتیاز استفاده کنید"
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