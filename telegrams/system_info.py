from exchange.exchange_config import SYMBOLS, init_exchange
from imports import logger, asyncio, datetime, sys

async def _test_exchange_connection():
    """Test exchange connection and return status and error"""
    try:
        logger.debug("Testing exchange connection...")
        test_exchange = await init_exchange()
        
        if not test_exchange:
            logger.error("Failed to initialize exchange")
            return "❌ خطا در اتصال", None
            
        logger.debug("Exchange initialized, testing ticker fetch...")
        ticker = await test_exchange.fetch_ticker('BTC/USDT')
        
        if ticker:
            logger.info("Exchange connection test successful")
            return "✅ متصل", None
        else:
            logger.warning("Exchange connection established but ticker fetch returned None")
            return "⚠️ مشکل در دریافت داده", None
            
    except asyncio.TimeoutError:
        logger.error("Exchange connection timeout")
        return "⏳ تایم‌اوت", "Connection timeout"
    except Exception as e:
        logger.error(f"Exchange connection error: {e}", exc_info=True)
        return "❌ خطا", str(e)

def _get_system_info():
    """Get system information safely"""
    def safe_get(func, default="نامشخص"):
        try:
            return func()
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return default
    
    symbols_count = safe_get(lambda: len(SYMBOLS) if 'SYMBOLS' in globals() else 0, 0)
    current_time = safe_get(lambda: datetime.now().strftime('%H:%M:%S'))
    python_version = safe_get(lambda: sys.version.split()[0])
    
    return symbols_count, current_time, python_version