from logger_config import logger
import hashlib

NUMBA_AVAILABLE = True

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available. Using standard calculations.")
    
    # Define dummy jit decorator if numba is not available
    def jit():
        def decorator(func):
            return func
        return decorator


# Cache for storing calculated indicators
_indicator_cache = {}
_cache_max_size = 1000

def _get_dataframe_hash(df, columns=None):
    """Create a hash for DataFrame to use as cache key"""
    try:
        if df is None or len(df) == 0:
            return None
        
        # Use specific columns or all numeric columns
        if columns is None:
            columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check if columns exist in dataframe
        available_columns = [col for col in columns if col in df.columns]
        if not available_columns:
            return None
        
        # Get the last few rows to create a representative hash
        sample_size = min(50, len(df))
        sample_df = df[available_columns].tail(sample_size)
        
        # Create more robust hash using multiple factors
        hash_components = [
            str(len(df)),  # Total length
            str(sample_df.iloc[0].values.tobytes()) if len(sample_df) > 0 else '',  # First row
            str(sample_df.iloc[-1].values.tobytes()) if len(sample_df) > 0 else '',  # Last row
            str(sample_df.shape),  # Shape info
        ]
        
        # Add hash of values if sample is manageable
        if len(sample_df) > 0:
            hash_components.append(str(hash(tuple(sample_df.values.flatten().tobytes()))))
        
        hash_string = "_".join(hash_components)
        return hashlib.md5(hash_string.encode()).hexdigest()[:16]  # Shorter hash
    except Exception:
        return None

def _clean_cache():
    """Clean cache if it gets too large using LRU strategy"""
    global _indicator_cache
    if len(_indicator_cache) > _cache_max_size:
        # Keep only the most recent entries (simple LRU approximation)
        items = list(_indicator_cache.items())
        # Remove oldest 40% of entries
        remove_count = len(items) // 5 * 2
        _indicator_cache = dict(items[remove_count:])

def _cached_simple_calculation(df_hash, indicator_name, *args, **kwargs):
    """Create a simple cache key for calculations"""
    # Handle both args and kwargs for better cache key generation
    args_str = "_".join(str(arg) for arg in args)
    kwargs_str = "_".join(f"{k}:{v}" for k, v in sorted(kwargs.items()) if k != 'df')
    
    components = [indicator_name, df_hash, args_str]
    if kwargs_str:
        components.append(kwargs_str)
    
    return "_".join(filter(None, components))

def _cached_indicator_calculation(df, indicator_name, calculation_func, *args, **kwargs):
    """Generic cached calculation for indicators with improved error handling"""
    try:
        # Quick validation
        if df is None or len(df) == 0:
            return None
            
        df_hash = _get_dataframe_hash(df)
        if df_hash is None:
            # Call the function directly without caching to avoid recursion
            try:
                return calculation_func(df, *args, **kwargs)
            except Exception:
                return None
        
        # Create cache key
        cache_key = _cached_simple_calculation(df_hash, indicator_name, *args, **kwargs)
        
        # Check cache with thread safety consideration
        if cache_key in _indicator_cache:
            cached_result = _indicator_cache[cache_key]
            # Validate cached result
            if cached_result is not None:
                return cached_result
        
        # Calculate new result - call the function directly to avoid recursion
        try:
            result = calculation_func(df, *args, **kwargs)
        except Exception as calc_error:
            logger.warning(f"Calculation error for {indicator_name}: {calc_error}")
            return None
        
        # Cache valid results only
        if result is not None and hasattr(result, '__len__'):
            _indicator_cache[cache_key] = result
            _clean_cache()
        
        return result
    except Exception as e:
        logger.warning(f"Cache error for {indicator_name}: {e}")
        # Call the function directly without caching to avoid recursion
        try:
            return calculation_func(df, *args, **kwargs)
        except Exception:
            return None

def _get_cache_key_sma(args, values_hash):
    """Generate cache key for SMA"""
    period = args[0] if args else 20
    return f"sma_{period}_{values_hash}"

def _get_cache_key_ema(args, values_hash):
    """Generate cache key for EMA"""
    period = args[0] if args else 20
    return f"ema_{period}_{values_hash}"

def _get_cache_key_rsi(args, values_hash):
    """Generate cache key for RSI"""
    period = args[0] if args else 14
    return f"rsi_{period}_{values_hash}"

def _get_cache_key_stdev(args, values_hash):
    """Generate cache key for STDEV"""
    period = args[0] if args else 20
    return f"stdev_{period}_{values_hash}"

def _get_cache_key_macd(args, values_hash):
    """Generate cache key for MACD"""
    fast_period = args[0] if len(args) > 0 else 12
    slow_period = args[1] if len(args) > 1 else 26
    signal_period = args[2] if len(args) > 2 else 9
    return f"macd_{fast_period}_{slow_period}_{signal_period}_{values_hash}"

def _get_cache_key_bollinger_bands(args, values_hash):
    """Generate cache key for Bollinger Bands"""
    period = args[0] if len(args) > 0 else 20
    std_dev = args[1] if len(args) > 1 else 2
    return f"bollinger_{period}_{std_dev}_{values_hash}"

def _get_cache_key_atr(args, values_hash):
    """Generate cache key for ATR"""
    period = args[0] if len(args) > 0 else 14
    return f"atr_{period}_{values_hash}"

def _get_cache_key_keltner_channels(args, values_hash):
    """Generate cache key for Keltner Channels"""
    period = args[0] if len(args) > 0 else 20
    multiplier = args[1] if len(args) > 1 else 2
    return f"keltner_{period}_{multiplier}_{values_hash}"

def _get_cache_key_supertrend(args, values_hash):
    """Generate cache key for Supertrend"""
    period = args[0] if len(args) > 0 else 10
    multiplier = args[1] if len(args) > 1 else 3.0
    return f"supertrend_{period}_{multiplier}_{values_hash}"

def _get_cache_key_adx(args, values_hash):
    """Generate cache key for ADX"""
    period = args[0] if len(args) > 0 else 14
    return f"adx_{period}_{values_hash}"

def _get_cache_key_generic(calculation_type, args, values_hash):
    """Generate generic cache key"""
    # Mapping calculation types to their cache key generators
    cache_key_generators = {
        'sma': _get_cache_key_sma,
        'ema': _get_cache_key_ema,
        'rsi': _get_cache_key_rsi,
        'stdev': _get_cache_key_stdev,
        'macd': _get_cache_key_macd,
        'bollinger_bands': _get_cache_key_bollinger_bands,
        'atr': _get_cache_key_atr,
        'keltner_channels': _get_cache_key_keltner_channels,
        'supertrend': _get_cache_key_supertrend,
        'adx': _get_cache_key_adx,
    }
    
    try:
        # Get the appropriate cache key generator
        key_generator = cache_key_generators.get(calculation_type)
        
        if key_generator:
            return key_generator(args, values_hash)
        else:
            return f"calc_{calculation_type}_{values_hash}"
            
    except Exception as e:
        logger.warning(f"Error generating cache key for {calculation_type}: {e}")
        return f"calc_{calculation_type}_{values_hash}_error"
