import asyncio
import ccxt.async_support as ccxt
from exchange.exchange_config import _validate_symbol_format, init_exchange
from logger_config import logger

async def get_current_price(symbol):
    """Fetch current price with improved error handling"""
    max_retries = 2
    symbol = _validate_symbol_format(symbol)
    
    for attempt in range(max_retries):
        try:
            exchange = init_exchange()
            if exchange is None:
                logger.error("Failed to initialize exchange")
                return None
                
            ticker = await _fetch_ticker_with_timeout(exchange, symbol)
            return _extract_price_from_ticker(ticker, symbol)
                
        except (ccxt.NetworkError, Exception) as e:
            if isinstance(e, (ccxt.ExchangeError, ValueError)):
                logger.error(f"Non-retryable error for {symbol}: {e}")
                break
            
            should_retry = await _handle_price_fetch_error(e, symbol, attempt, max_retries)
            if not should_retry:
                break
    
    logger.error(f"Failed to fetch price for {symbol} after {max_retries} attempts")
    return None

async def _fetch_ticker_with_timeout(exchange, symbol):
    """Fetch ticker data with timeout"""
    return await asyncio.wait_for(
        exchange.fetch_ticker(symbol),
        timeout=10
    )

def _extract_price_from_ticker(ticker, symbol):
    """Extract price from ticker data"""
    if ticker and 'last' in ticker and ticker['last'] is not None:
        return float(ticker['last'])
    logger.warning(f"No valid price data for {symbol}")
    return None

async def _handle_price_fetch_error(error, symbol, attempt, max_retries):
    """Handle errors during price fetching"""
    if isinstance(error, asyncio.TimeoutError):
        logger.warning(f"Timeout fetching price for {symbol}, attempt {attempt + 1}")
    else:
        logger.error(f"Error fetching price for {symbol}: {error}")
    
    if attempt < max_retries - 1:
        await asyncio.sleep(1)
        return True
    return False
