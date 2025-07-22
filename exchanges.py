# Global exchange instance
import os
import logging
import asyncio
import warnings
import pandas as pd
import ccxt.async_support as ccxt

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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
