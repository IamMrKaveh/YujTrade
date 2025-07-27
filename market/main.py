from exchange.exchange_config import SYMBOLS, init_exchange
from logger_config import logger
from market.batch_processing import _process_all_symbols_in_batches
from market.signal_processing import _process_and_return_best_signal



exchange = None

async def analyze_market():
    """تحلیل بازار و بازگرداندن بهترین سیگنال با اطلاعات تکمیلی"""
    try:
        logger.info("Starting market analysis")
        
        if not _validate_symbols():
            logger.error("Symbol validation failed")
            return []
        
        if not _ensure_exchange_connection():
            logger.error("Exchange connection failed")
            return []
        
        analysis_stats = {'successful': 0, 'failed': 0}
        
        # Process all symbols in batches
        all_signals = await _process_all_symbols_in_batches(analysis_stats)
        
        # Process and return the best signal
        result = _process_and_return_best_signal(all_signals, analysis_stats)
        
        logger.info(f"Market analysis completed successfully, returning {len(result)} signals")
        return result
        
    except Exception as e:
        logger.error(f"Critical error in analyze_market: {e}")
        logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")
        return []

def _validate_symbols():
    """بررسی وجود نمادها"""
    if not SYMBOLS or len(SYMBOLS) == 0:
        logger.warning("No symbols available for analysis")
        logger.debug("Symbol validation failed - no symbols found")
        return False
    
    logger.info(f"Starting market analysis for {len(SYMBOLS)} symbols")
    logger.debug(f"Symbol validation completed successfully with {len(SYMBOLS)} symbols")
    return True

def _ensure_exchange_connection():
    """بررسی و تضمین اتصال به صرافی"""
    if not exchange:
        try:
            exchange = init_exchange()
            if not exchange:
                logger.error("Exchange initialization returned None")
                logger.debug("Exchange connection validation failed - initialization returned None")
                return False
            logger.info("Exchange initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            logger.debug(f"Exchange connection validation failed with exception: {e}")
            return False
    return True