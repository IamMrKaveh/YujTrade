import os
import ccxt.async_support as ccxt
from logger_config import logger

exchange = None

async def init_exchange():
    global exchange
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
            # حتما load_markets را await کن تا مطمئن باشی exchange آماده است
            await exchange.load_markets()
            logger.info("Exchange initialized successfully (async)")
        except Exception as e:
            logger.error(f"Error initializing exchange: {e}")
            exchange = None
            return None
    else:
        logger.info("Exchange already initialized")
    return exchange

async def close_exchange():
    """Close exchange connection"""
    global exchange
    if exchange:
        try:
            if hasattr(exchange, 'close') and callable(exchange.close):
                await exchange.close()
            logger.info("Exchange connection closed")
        except Exception as e:
            logger.error(f"Error closing exchange connection: {e}")
        exchange = None