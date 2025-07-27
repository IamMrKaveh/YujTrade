import os
import ccxt.async_support as ccxt
from logger_config import logger

exchange = None

BTC = 'BTC/USDT'
ETH = 'ETH/USDT'
BNB = 'BNB/USDT'
ADA = 'ADA/USDT'
SOL = 'SOL/USDT'

def _validate_symbol_format(symbol):
    """Validate and format symbol"""
    if '/' not in symbol:
        return f"{symbol}/USDT"
    return symbol

def load_symbols():
    default_symbols = [BTC, ETH, BNB, ADA, SOL]
    try:
        with open('symbols.txt', 'r', encoding='utf-8') as f:
            symbols = [line.strip().upper() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(symbols)} symbols from symbols.txt")
        return symbols
    except FileNotFoundError:
        logger.error("symbols.txt not found. Using default symbols.")
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
        return default_symbols
    
SYMBOLS = load_symbols()

def init_exchange():
    if exchange is None:
        try:
            exchange = ccxt.coinex({
                'apiKey': os.getenv('COINEX_API_KEY', ''),
                'secret': os.getenv('COINEX_SECRET', ''),
                'sandbox': False,
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {'defaultType': 'spot'}
            })
            logger.info("Exchange initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing exchange: {e}")
            return None
    return exchange

async def close_exchange():
    """Close exchange connection"""
    global exchange
    if exchange:
        try:
            await exchange.close()
            logger.info("Exchange connection closed")
        except Exception as e:
            logger.error(f"Error closing exchange connection: {e}")
            
        exchange = None
