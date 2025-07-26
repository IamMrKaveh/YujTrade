from exchange.exchange_config import _validate_symbol_format, init_exchange
from exchange.timeframes import _get_optimal_limit, _validate_timeframe
from imports import logger, ccxt, asyncio, pd

async def get_klines(symbol, interval='1h', limit=None):
    """
    Fetch klines data with improved error handling and multiple timeframe support
    
    Args:
        symbol (str): Trading symbol (e.g., 'BTC')
        interval (str): Timeframe ('1m', '5m', '15m', '30m', '1h', '4h', '1d', etc.)
        limit (int): Number of candles to fetch (None for default based on timeframe)
    
    Returns:
        pandas.DataFrame: OHLCV data with timestamp index, or None if failed
    """
    max_retries = 3
    symbol = _validate_symbol_format(symbol)
    interval = _validate_timeframe(interval)
    limit = _get_optimal_limit(interval, limit)
    
    logger.info(f"Fetching {limit} candles for {symbol} on {interval} timeframe")
    
    for attempt in range(max_retries):
        try:
            exchange = init_exchange()
            result = await _fetch_klines_attempt(exchange, symbol, interval, limit)
            if result is not None:
                return result
                
        except (asyncio.TimeoutError, ccxt.NetworkError) as e:
            if _should_retry_error(e, attempt, max_retries):
                await _handle_retryable_error(e, symbol, interval, attempt)
                continue
            break
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching klines for {symbol} ({interval}): {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected error fetching klines for {symbol} ({interval}): {e}")
            break
    
    return None

async def _fetch_ohlcv_data(exchange, symbol, interval, limit):
    """Fetch OHLCV data with timeout"""
    return await asyncio.wait_for(
        exchange.fetch_ohlcv(symbol, interval, limit=limit),
        timeout=15
    )

def _process_ohlcv_data(symbol, ohlcv, interval):
    """Process and validate OHLCV data"""
    if not _is_sufficient_data(ohlcv):
        logger.warning(f"Insufficient data for {symbol} ({interval}): {len(ohlcv) if ohlcv else 0} candles")
        return None
        
    df = _create_dataframe_from_ohlcv(ohlcv)
    
    if not _is_sufficient_data(df):
        logger.warning(f"Insufficient clean data for {symbol} ({interval}): {len(df)} candles")
        return None
        
    logger.info(f"Successfully fetched {len(df)} candles for {symbol} ({interval})")
    return df

async def _fetch_klines_attempt(exchange, symbol, interval, limit):
    """Single attempt to fetch klines data"""
    if exchange is None:
        return None
    
    ohlcv = await _fetch_ohlcv_data(exchange, symbol, interval, limit)
    
    return _process_ohlcv_data(symbol, ohlcv, interval)

async def _get_multiple_timeframes(symbol, intervals=['1h', '4h', '1d'], limit=None):
    """
    Fetch klines data for multiple timeframes simultaneously
    
    Args:
        symbol (str): Trading symbol
        intervals (list): List of timeframes to fetch
        limit (int): Number of candles per timeframe
    
    Returns:
        dict: Dictionary with timeframes as keys and DataFrames as values
    """
    symbol = _validate_symbol_format(symbol)
    
    # Validate all intervals
    valid_intervals = []
    for interval in intervals:
        validated = _validate_timeframe(interval)
        if validated not in valid_intervals:
            valid_intervals.append(validated)
    
    logger.info(f"Fetching multiple timeframes for {symbol}: {', '.join(valid_intervals)}")
    
    # Create tasks for concurrent fetching
    tasks = []
    for interval in valid_intervals:
        task = get_klines(symbol, interval, limit)
        tasks.append((interval, task))
    
    # Execute all tasks concurrently
    results = {}
    for interval, task in tasks:
        try:
            result = await task
            if result is not None:
                results[interval] = result
            else:
                logger.warning(f"Failed to fetch data for {symbol} ({interval})")
        except Exception as e:
            logger.error(f"Error fetching {interval} data for {symbol}: {e}")
    
    return results

def _should_retry_error(error, attempt, max_retries):
    """Determine if error should trigger a retry"""
    if isinstance(error, ccxt.ExchangeError):
        # Allow retries for temporary exchange errors
        retryable_errors = (
            ccxt.DDoSProtection,
            ccxt.ExchangeNotAvailable,
            ccxt.RequestTimeout,
            ccxt.NetworkError
        )
        if not isinstance(error, retryable_errors):
            return False
    return attempt < max_retries - 1

async def _handle_retryable_error(error, symbol, interval, attempt):
    """Handle retryable errors with appropriate logging and delay"""
    error_type = "Timeout" if isinstance(error, asyncio.TimeoutError) else "Network error"
    logger.warning(f"{error_type} fetching klines for {symbol} ({interval}), attempt {attempt + 1}")
    await asyncio.sleep(2 ** attempt)

def _create_dataframe_from_ohlcv(ohlcv):
    """Create and clean DataFrame from OHLCV data"""
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Convert to numeric types
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df.dropna()

def _is_sufficient_data(data, min_candles=10):
    """Check if data has sufficient candles (reduced minimum for shorter timeframes)"""
    return data is not None and len(data) >= min_candles
