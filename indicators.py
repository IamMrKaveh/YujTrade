import logging
import sys
import pandas as pd
import numpy as np
from numba import jit
import warnings

# Suppress pkg_resources deprecation warning from pandas_ta
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")

if not hasattr(np, 'NaN'):
    np.NaN = np.nan
    
# Now import pandas_ta after fixing numpy
try:
    import pandas_ta as ta
except ImportError as e:
    print(f"Error importing pandas_ta: {e}")
    print("Please install with: pip install pandas-ta")
    sys.exit(1)
    

# 2. کش کردن محاسبات
from functools import lru_cache
import hashlib

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
        
        # Get the last few rows to create a representative hash
        sample_size = min(50, len(df))
        sample_df = df[columns].tail(sample_size)
        
        # Create hash from values
        hash_string = str(sample_df.values.tobytes()) + str(len(df))
        return hashlib.md5(hash_string.encode()).hexdigest()
    except Exception:
        return None

def _clean_cache():
    """Clean cache if it gets too large"""
    global _indicator_cache
    if len(_indicator_cache) > _cache_max_size:
        # Remove oldest half of entries
        items = list(_indicator_cache.items())
        _indicator_cache = dict(items[len(items)//2:])

def cached_indicator_calculation(df, indicator_name, calculation_func, *args, **kwargs):
    """Generic cached calculation for indicators"""
    try:
        df_hash = _get_dataframe_hash(df)
        if df_hash is None:
            return calculation_func(df, *args, **kwargs)
        
        # Create cache key using cached_simple_calculation
        cache_key = cached_simple_calculation(df_hash, indicator_name, *args)
        
        # Check cache
        if cache_key in _indicator_cache:
            return _indicator_cache[cache_key]
        
        # Calculate and cache
        result = calculation_func(df, *args, **kwargs)
        
        if result is not None:
            _indicator_cache[cache_key] = result
            _clean_cache()
        
        return result
    except Exception:
        return calculation_func(df, *args, **kwargs)

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
    fast = args[0] if len(args) > 0 else 12
    slow = args[1] if len(args) > 1 else 26
    signal = args[2] if len(args) > 2 else 9
    return f"macd_{fast}_{slow}_{signal}_{values_hash}"

def _get_cache_key_bollinger_bands(args, values_hash):
    """Generate cache key for Bollinger Bands"""
    period = args[0] if len(args) > 0 else 20
    std_dev = args[1] if len(args) > 1 else 2
    return f"bb_{period}_{std_dev}_{values_hash}"

def _get_cache_key_atr(args, values_hash):
    """Generate cache key for ATR"""
    period = args[0] if args else 14
    return f"atr_{period}_{values_hash}"

def _get_cache_key_generic(calculation_type, args, values_hash):
    """Generate generic cache key"""
    args_str = '_'.join(str(arg) for arg in args) if args else 'default'
    return f"{calculation_type}_{args_str}_{values_hash}"

@lru_cache(maxsize=128)
def cached_simple_calculation(values_hash, calculation_type, *args):
    """Cache for simple numeric calculations"""
    # Mapping calculation types to their cache key generators
    cache_key_generators = {
        'sma': _get_cache_key_sma,
        'ema': _get_cache_key_ema,
        'rsi': _get_cache_key_rsi,
        'stdev': _get_cache_key_stdev,
        'macd': _get_cache_key_macd,
        'bollinger_bands': _get_cache_key_bollinger_bands,
        'atr': _get_cache_key_atr
    }
    
    try:
        # Get the appropriate cache key generator
        key_generator = cache_key_generators.get(calculation_type)
        
        if key_generator:
            return key_generator(args, values_hash)
        else:
            return _get_cache_key_generic(calculation_type, args, values_hash)
            
    except Exception as e:
        # Return a simple cache key if there's an error
        return f"calc_{calculation_type}_{values_hash}_{str(e)}"

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Set up logger
logger = logging.getLogger(__name__)

try:
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available. Using standard calculations.")
    
    # Define dummy jit decorator if numba is not available
    def jit():
        def decorator(func):
            return func
        return decorator

@jit(nopython=True)
def fast_sma(prices, period):
    """محاسبه سریع میانگین متحرک"""
    result = np.empty(len(prices))
    result[:period-1] = np.nan
    for i in range(period-1, len(prices)):
        result[i] = np.mean(prices[i-period+1:i+1])
    return result

@jit(nopython=True)
def fast_ema(prices, alpha):
    """محاسبه سریع میانگین متحرک نمایی"""
    result = np.empty(len(prices))
    result[0] = prices[0]
    for i in range(1, len(prices)):
        result[i] = alpha * prices[i] + (1 - alpha) * result[i-1]
    return result

@jit(nopython=True)
def fast_rsi_calculation(prices, period):
    """محاسبه سریع RSI"""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    avg_gains = np.empty(len(gains))
    avg_losses = np.empty(len(losses))
    
    # محاسبه اولیه
    avg_gains[period-1] = np.mean(gains[:period])
    avg_losses[period-1] = np.mean(losses[:period])
    
    # محاسبه نمایی
    alpha = 1.0 / period
    for i in range(period, len(gains)):
        avg_gains[i] = alpha * gains[i] + (1 - alpha) * avg_gains[i-1]
        avg_losses[i] = alpha * losses[i] + (1 - alpha) * avg_losses[i-1]
    
    rs = avg_gains[period-1:] / avg_losses[period-1:]
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

@jit(nopython=True)
def fast_bollinger_bands(prices, period, std_dev):
    """محاسبه سریع باندهای بولینگر"""
    sma = fast_sma(prices, period)
    
    # محاسبه انحراف معیار
    std = np.empty(len(prices))
    std[:period-1] = np.nan
    
    for i in range(period-1, len(prices)):
        window = prices[i-period+1:i+1]
        std[i] = np.std(window)
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return sma, upper_band, lower_band

@jit(nopython=True)
def fast_macd_calculation(prices, fast_period, slow_period, signal_period):
    """محاسبه سریع MACD"""
    fast_alpha = 2.0 / (fast_period + 1)
    slow_alpha = 2.0 / (slow_period + 1)
    signal_alpha = 2.0 / (signal_period + 1)
    
    fast_ema = fast_ema(prices, fast_alpha)
    slow_ema = fast_ema(prices, slow_alpha)

def _calculate_sma(df, column, period):
    try:
        if df is None or len(df) < period:
            return None
        
        prices = df[column].values.astype(np.float64)
        if NUMBA_AVAILABLE:
            result = fast_sma(prices, period)
        else:
            result = df[column].rolling(window=period).mean().values
        
        return pd.Series(result, index=df.index)
    except Exception as e:
        logger.warning(f"Error in optimized SMA calculation: {e}")
        return None

def calculate_optimized_sma(df, column='close', period=20):
    """محاسبه بهینه میانگین متحرک"""
    
    return cached_indicator_calculation(df, 'sma', _calculate_sma, column, period)

def _calculate_rsi(df, period):
    try:
        if df is None or len(df) < period + 1:
            return None
        
        prices = df['close'].values.astype(np.float64)
        if NUMBA_AVAILABLE:
            rsi_values = fast_rsi_calculation(prices, period)
            # Pad with NaN values for consistency
            result = np.full(len(df), np.nan)
            result[period:] = rsi_values
        else:
            # Fallback to pandas calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            result = 100 - (100 / (1 + rs)).values
        
        return pd.Series(result, index=df.index)
    except Exception as e:
        logger.warning(f"Error in optimized RSI calculation: {e}")
        return None

def calculate_optimized_rsi(df, period=14):
    """محاسبه بهینه RSI"""  
    
    return cached_indicator_calculation(df, 'rsi', _calculate_rsi, period)

def _calculate_bollinger_bands(df, period, std_dev):
    try:
        if df is None or len(df) < period:
            return None
        
        prices = df['close'].values.astype(np.float64)
        if NUMBA_AVAILABLE:
            sma, upper_band, lower_band = fast_bollinger_bands(prices, period, std_dev)
        else:
            sma = df['close'].rolling(window=period).mean().values
            std = df['close'].rolling(window=period).std().values
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
        
        return {
            'bb_upper': pd.Series(upper_band, index=df.index),
            'bb_middle': pd.Series(sma, index=df.index),
            'bb_lower': pd.Series(lower_band, index=df.index)
        }
    except Exception as e:
        logger.warning(f"Error in optimized Bollinger Bands calculation: {e}")
        return None
        
def calculate_optimized_bollinger_bands(df, period=20, std_dev=2):
    """محاسبه بهینه باندهای بولینگر"""
    
    return cached_indicator_calculation(df, 'bollinger_bands', _calculate_bollinger_bands, period, std_dev)

# ===== اندیکاتورهای ترند و نوسان =====

def _calculate_trend_strength(df, period):
    """محاسبه قدرت ترند"""
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        # محاسبه ADX برای قدرت ترند
        adx_data = calculate_adx(df, period)
        if adx_data is None:
            return None
        
        # محاسبه زاویه خط ترند
        prices = close.tail(period).values
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        angle = np.arctan(slope) * 180 / np.pi
        
        # ترکیب ADX و زاویه برای تعیین قدرت ترند
        adx_value = adx_data['adx'].iloc[-1]
        trend_strength = min(adx_value + abs(angle) * 2, 100)
        
        return {
            'strength': trend_strength,
            'direction': 'up' if slope > 0 else 'down',
            'adx': adx_value,
            'angle': angle
        }
    except Exception:
        return None

def _calculate_market_volatility(df, period):
    """محاسبه نوسانات بازار"""
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        # انحراف معیار قیمت
        price_std = close.rolling(period).std()
        
        # Average True Range
        atr = calculate_atr(df, period)
        
        # نوسانات درون روزی
        daily_range = (high - low) / close
        avg_daily_range = daily_range.rolling(period).mean()
        
        # شاخص نوسان ترکیبی
        if atr is not None and price_std is not None:
            volatility_score = (price_std.iloc[-1] / close.iloc[-1] * 100 + 
                              atr.iloc[-1] / close.iloc[-1] * 100 + 
                              avg_daily_range.iloc[-1] * 100) / 3
        else:
            volatility_score = avg_daily_range.iloc[-1] * 100
        
        return {
            'volatility_score': volatility_score,
            'price_std': price_std.iloc[-1] if price_std is not None else None,
            'atr': atr.iloc[-1] if atr is not None else None,
            'daily_range': avg_daily_range.iloc[-1]
        }
    except Exception:
        return None

def calculate_market_volatility(df, period=20):
    """محاسبه نوسانات بازار"""
    return cached_indicator_calculation(df, 'market_volatility', _calculate_market_volatility, period)

def calculate_trend_strength(df, period=20):
    """محاسبه قدرت ترند"""
    return cached_indicator_calculation(df, 'trend_strength', _calculate_trend_strength, period)

def _get_default_stops(entry_price, position_type):
    """Get default stop loss and take profit values"""
    if position_type == 'long':
        return {
            'stop_loss': entry_price * 0.95,
            'take_profit': entry_price * 1.1,
            'risk_reward_ratio': 2.0
        }
    else:
        return {
            'stop_loss': entry_price * 1.05,
            'take_profit': entry_price * 0.9,
            'risk_reward_ratio': 2.0
        }

def _calculate_risk_multipliers(trend_data, volatility_data, base_risk):
    """Calculate risk and reward multipliers based on trend and volatility"""
    if trend_data is None or volatility_data is None:
        return base_risk, 2.5
    
    trend_strength = trend_data['strength']
    volatility_score = volatility_data['volatility_score']
    
    # Adjust multipliers based on trend strength
    if trend_strength > 50:
        risk_multiplier = base_risk * (1 + trend_strength / 100)
        reward_multiplier = 3.0 + trend_strength / 50
    else:
        risk_multiplier = base_risk * 0.7
        reward_multiplier = 1.5
    
    # Adjust based on volatility
    if volatility_score > 5:
        risk_multiplier *= 1.5
        reward_multiplier *= 1.3
    elif volatility_score < 2:
        risk_multiplier *= 0.8
        reward_multiplier *= 0.9
    
    return risk_multiplier, reward_multiplier

def _get_atr_value(df, entry_price, risk_multiplier):
    """Get ATR value adjusted by risk multiplier"""
    atr = calculate_atr(df)
    atr_value = atr.iloc[-1] if atr is not None else entry_price * 0.02
    return atr_value * risk_multiplier

def _find_nearest_support(support_levels, entry_price, default_value):
    """Find nearest support level below entry price"""
    if not support_levels:
        return default_value
    
    valid_supports = [s for s in support_levels if s < entry_price]
    return max(valid_supports, default=default_value)

def _find_nearest_resistance(resistance_levels, entry_price, default_value):
    """Find nearest resistance level above entry price"""
    if not resistance_levels:
        return default_value
    
    valid_resistances = [r for r in resistance_levels if r > entry_price]
    return min(valid_resistances, default=default_value)

def _calculate_long_stops(entry_price, atr_value, support_resistance):
    """Calculate stop loss and take profit for long positions"""
    stop_loss_atr = entry_price - atr_value
    
    # Adjust stop loss based on support levels
    if (support_resistance and support_resistance.get('support_levels')):
        nearest_support = _find_nearest_support(
            support_resistance['support_levels'], 
            entry_price, 
            stop_loss_atr
        )
        stop_loss = max(stop_loss_atr, nearest_support * 0.98)
    else:
        stop_loss = stop_loss_atr
    
    # Calculate take profit
    risk_amount = entry_price - stop_loss
    take_profit_atr = entry_price + (risk_amount * 2.5)
    
    # Adjust take profit based on resistance levels
    if (support_resistance and support_resistance.get('resistance_levels')):
        nearest_resistance = _find_nearest_resistance(
            support_resistance['resistance_levels'], 
            entry_price, 
            take_profit_atr
        )
        take_profit = min(take_profit_atr, nearest_resistance * 0.98)
    else:
        take_profit = take_profit_atr
    
    return stop_loss, take_profit

def _calculate_short_stops(entry_price, atr_value, support_resistance):
    """Calculate stop loss and take profit for short positions"""
    stop_loss_atr = entry_price + atr_value
    
    # Adjust stop loss based on resistance levels
    if (support_resistance and support_resistance.get('resistance_levels')):
        nearest_resistance = _find_nearest_resistance(
            support_resistance['resistance_levels'], 
            entry_price, 
            stop_loss_atr
        )
        stop_loss = min(stop_loss_atr, nearest_resistance * 1.02)
    else:
        stop_loss = stop_loss_atr
    
    # Calculate take profit
    risk_amount = stop_loss - entry_price
    take_profit_atr = entry_price - (risk_amount * 2.5)
    
    # Adjust take profit based on support levels
    if (support_resistance and support_resistance.get('support_levels')):
        nearest_support = _find_nearest_support(
            support_resistance['support_levels'], 
            entry_price, 
            take_profit_atr
        )
        take_profit = max(take_profit_atr, nearest_support * 1.02)
    else:
        take_profit = take_profit_atr
    
    return stop_loss, take_profit

def _ensure_minimum_risk_reward(entry_price, stop_loss, take_profit, position_type, min_ratio=1.5):
    """Ensure minimum risk-reward ratio"""
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    risk_reward_ratio = reward / risk if risk > 0 else 0
    
    if risk_reward_ratio < min_ratio:
        if position_type == 'long':
            take_profit = entry_price + (risk * min_ratio)
        else:
            take_profit = entry_price - (risk * min_ratio)
        risk_reward_ratio = min_ratio
    
    return take_profit, risk_reward_ratio

def _calculate_dynamic_stops(df, entry_price, position_type, base_risk):
    """محاسبه حد ضرر و حد سود پویا بر اساس ترند و نوسانات"""
    try:
        if df is None or len(df) < 20:
            return _get_default_stops(entry_price, position_type)
        
        # Get market analysis data
        trend_data = calculate_trend_strength(df)
        volatility_data = calculate_market_volatility(df)
        support_resistance = calculate_support_resistance_levels(df)
        
        # Calculate risk multipliers
        risk_multiplier, reward_multiplier = _calculate_risk_multipliers(
            trend_data, volatility_data, base_risk
        )
        
        # Get ATR value
        atr_value = _get_atr_value(df, entry_price, risk_multiplier)
        
        # Calculate stops based on position type
        if position_type == 'long':
            stop_loss, take_profit = _calculate_long_stops(
                entry_price, atr_value, support_resistance
            )
        else:
            stop_loss, take_profit = _calculate_short_stops(
                entry_price, atr_value, support_resistance
            )
        
        # Ensure minimum risk-reward ratio
        take_profit, risk_reward_ratio = _ensure_minimum_risk_reward(
            entry_price, stop_loss, take_profit, position_type
        )
        
        return {
            'stop_loss': round(stop_loss, 8),
            'take_profit': round(take_profit, 8),
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'atr_value': round(atr_value, 8)
        }
        
    except Exception:
        return _get_default_stops(entry_price, position_type)

def calculate_dynamic_stops(df, entry_price, position_type='long', base_risk=2.0):
    """محاسبه حد ضرر و حد سود پویا بر اساس ترند و نوسانات"""
    
    return cached_indicator_calculation(df, 'dynamic_stops', _calculate_dynamic_stops, entry_price, position_type, base_risk)

def calculate_trailing_stop(df, entry_price, current_price, position_type='long', atr_multiplier=2.0):
    """محاسبه حد ضرر متحرک"""
    try:
        if df is None or len(df) < 14:
            return entry_price * 0.95 if position_type == 'long' else entry_price * 1.05
        
        atr = calculate_atr(df)
        if atr is None:
            atr_value = abs(current_price - entry_price) * 0.1
        else:
            atr_value = atr.iloc[-1] * atr_multiplier
        
        if position_type == 'long':
            # برای پوزیشن خرید، حد ضرر به سمت بالا حرکت می‌کند
            trailing_stop = current_price - atr_value
            # حد ضرر نمی‌تواند پایین‌تر از قیمت ورود برود
            return max(trailing_stop, entry_price * 0.98)
        else:
            # برای پوزیشن فروش، حد ضرر به سمت پایین حرکت می‌کند
            trailing_stop = current_price + atr_value
            # حد ضرر نمی‌تواند بالاتر از قیمت ورود برود
            return min(trailing_stop, entry_price * 1.02)
            
    except Exception:
        return entry_price * 0.95 if position_type == 'long' else entry_price * 1.05

def adaptive_position_sizing(capital, risk_percent, entry_price, stop_loss, market_conditions=None):
    """محاسبه اندازه پوزیشن تطبیقی"""
    try:
        base_risk = capital * (risk_percent / 100)
        price_diff = abs(entry_price - stop_loss)
        
        if price_diff == 0:
            return 0
        
        base_position_size = base_risk / price_diff
        
        # تنظیم بر اساس شرایط بازار
        if market_conditions:
            volatility = market_conditions.get('volatility_score', 3)
            trend_strength = market_conditions.get('trend_strength', 50)
            
            # نوسانات بالا = اندازه پوزیشن کمتر
            if volatility > 5:
                size_multiplier = 0.7
            elif volatility < 2:
                size_multiplier = 1.2
            else:
                size_multiplier = 1.0
            
            # ترند قوی = اندازه پوزیشن بیشتر
            if trend_strength > 70:
                size_multiplier *= 1.3
            elif trend_strength < 30:
                size_multiplier *= 0.8
            
            position_size = base_position_size * size_multiplier
        else:
            position_size = base_position_size
        
        # محدودیت حداکثر 10% سرمایه
        max_position_value = capital * 0.1
        max_position_size = max_position_value / entry_price
        
        final_position_size = min(position_size, max_position_size)
        
        return max(final_position_size, 0)
        
    except Exception:
        logger.error(f"Error calculating adaptive position size: {e}")
        return 0
        
def _calculate_fibonacci_levels(df, lookback):
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

def calculate_fibonacci_levels(df, lookback=50):
    """محاسبه سطوح فیبوناچی"""
    return cached_indicator_calculation(df, 'fibonacci_levels', _calculate_fibonacci_levels, lookback)

def _initialize_sar_arrays(df, af, high, low):
    """Initialize arrays for Parabolic SAR calculation"""
    sar = np.zeros(len(df))
    trend = np.zeros(len(df))
    af_val = np.zeros(len(df))
    ep = np.zeros(len(df))
    
    sar[0] = low[0]
    trend[0] = 1
    af_val[0] = af
    ep[0] = high[0]
    
    return sar, trend, af_val, ep

def _handle_uptrend_sar(i, sar, trend, af_val, ep, high, low, af, max_af):
    """Handle Parabolic SAR calculation for uptrend"""
    sar[i] = sar[i-1] + af_val[i-1] * (ep[i-1] - sar[i-1])
    
    if low[i] <= sar[i]:
        # Trend reversal to downtrend
        trend[i] = -1
        sar[i] = ep[i-1]
        ep[i] = low[i]
        af_val[i] = af
    else:
        # Continue uptrend
        trend[i] = 1
        if high[i] > ep[i-1]:
            ep[i] = high[i]
            af_val[i] = min(af_val[i-1] + af, max_af)
        else:
            ep[i] = ep[i-1]
            af_val[i] = af_val[i-1]

def _handle_downtrend_sar(i, sar, trend, af_val, ep, high, low, af, max_af):
    """Handle Parabolic SAR calculation for downtrend"""
    sar[i] = sar[i-1] - af_val[i-1] * (sar[i-1] - ep[i-1])
    
    if high[i] >= sar[i]:
        # Trend reversal to uptrend
        trend[i] = 1
        sar[i] = ep[i-1]
        ep[i] = high[i]
        af_val[i] = af
    else:
        # Continue downtrend
        trend[i] = -1
        if low[i] < ep[i-1]:
            ep[i] = low[i]
            af_val[i] = min(af_val[i-1] + af, max_af)
        else:
            ep[i] = ep[i-1]
            af_val[i] = af_val[i-1]

def _calculate_parabolic_sar(df, af, max_af):
    try:
        if df is None or len(df) < 5:
            return None
        
        high = df['high'].values
        low = df['low'].values
        
        sar, trend, af_val, ep = _initialize_sar_arrays(df, af, high, low)
        
        for i in range(1, len(df)):
            if trend[i-1] == 1:
                _handle_uptrend_sar(i, sar, trend, af_val, ep, high, low, af, max_af)
            else:
                _handle_downtrend_sar(i, sar, trend, af_val, ep, high, low, af, max_af)
        
        return pd.Series(sar, index=df.index, name='psar')
    except Exception:
        return None

def calculate_parabolic_sar(df, af=0.02, max_af=0.2):
    """محاسبه Parabolic SAR"""
    return cached_indicator_calculation(df, 'parabolic_sar', _calculate_parabolic_sar, af, max_af)

def _calculate_ichimoku(df):
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

def calculate_ichimoku(df):
    return cached_indicator_calculation(df, 'ichimoku', _calculate_ichimoku)

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

def _calculate_final_upper_band(upper_band, final_upper_band, close, i):
    """Calculate final upper band for Supertrend"""
    if upper_band.iloc[i] < final_upper_band[i-1] or close.iloc[i-1] > final_upper_band[i-1]:
        return upper_band.iloc[i]
    else:
        return final_upper_band[i-1]

def _calculate_final_lower_band(lower_band, final_lower_band, close, i):
    """Calculate final lower band for Supertrend"""
    if lower_band.iloc[i] > final_lower_band[i-1] or close.iloc[i-1] < final_lower_band[i-1]:
        return lower_band.iloc[i]
    else:
        return final_lower_band[i-1]

def _calculate_direction(direction, close, final_upper_band, final_lower_band, i):
    """Calculate direction for Supertrend"""
    prev_direction = direction[i-1]
    current_close = close.iloc[i]
    
    if prev_direction == -1 and current_close < final_lower_band[i]:
        return -1
    elif prev_direction == 1 and current_close > final_upper_band[i]:
        return 1
    elif prev_direction == -1 and current_close >= final_lower_band[i]:
        return 1
    elif prev_direction == 1 and current_close <= final_upper_band[i]:
        return -1
    else:
        return prev_direction

def _calculate_supertrend_value(direction, final_upper_band, final_lower_band, i):
    """Calculate Supertrend value based on direction"""
    if direction[i] == 1:
        return final_lower_band[i]
    else:
        return final_upper_band[i]

def _initialize_supertrend_arrays(upper_band, lower_band):
    """Initialize arrays for Supertrend calculation"""
    final_upper_band = [upper_band.iloc[0]]
    final_lower_band = [lower_band.iloc[0]]
    supertrend = [0]
    direction = [1]
    return final_upper_band, final_lower_band, supertrend, direction

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
        
        # Initialize arrays
        final_upper_band, final_lower_band, supertrend, direction = _initialize_supertrend_arrays(upper_band, lower_band)
        
        # Calculate for each period
        for i in range(1, len(df)):
            # Calculate final bands
            final_upper = _calculate_final_upper_band(upper_band, final_upper_band, close, i)
            final_lower = _calculate_final_lower_band(lower_band, final_lower_band, close, i)
            
            final_upper_band.append(final_upper)
            final_lower_band.append(final_lower)
            
            # Calculate direction
            new_direction = _calculate_direction(direction, close, final_upper_band, final_lower_band, i)
            direction.append(new_direction)
            
            # Calculate Supertrend value
            supertrend_value = _calculate_supertrend_value(direction, final_upper_band, final_lower_band, i)
            supertrend.append(supertrend_value)
        
        return pd.Series(supertrend, index=df.index)
    except Exception as e:
        logger.warning(f"Error calculating Supertrend: {e}")
        return None

def calculate_aroon_oscillator(df, period=14):
    try:
        if df is None or len(df) < period:
            return None
        
        if 'high' not in df.columns or 'low' not in df.columns:
            return None
        
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        
        aroon_up = np.full(len(df), np.nan)
        aroon_down = np.full(len(df), np.nan)
        
        for i in range(period - 1, len(df)):
            high_period = high.iloc[i-period+1:i+1].values
            low_period = low.iloc[i-period+1:i+1].values
            
            high_max_idx = np.argmax(high_period)
            low_min_idx = np.argmin(low_period)
            
            periods_since_high = period - 1 - high_max_idx
            periods_since_low = period - 1 - low_min_idx
            
            aroon_up[i] = ((period - periods_since_high) / period) * 100
            aroon_down[i] = ((period - periods_since_low) / period) * 100
        
        aroon_up_series = pd.Series(aroon_up, index=df.index)
        aroon_down_series = pd.Series(aroon_down, index=df.index)
        aroon_oscillator = aroon_up_series - aroon_down_series
        
        return {
            'aroon_up': aroon_up_series,
            'aroon_down': aroon_down_series,
            'aroon_oscillator': aroon_oscillator
        }
    except Exception:
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

def _find_swing_points(high, low, swing_strength):
    """Helper function to find swing highs and lows"""
    swing_highs = []
    swing_lows = []
    
    for i in range(swing_strength, len(high) - swing_strength):
        # Swing High
        if high.iloc[i] == high.iloc[i-swing_strength:i+swing_strength+1].max():
            swing_highs.append((i, high.iloc[i]))
        
        # Swing Low
        if low.iloc[i] == low.iloc[i-swing_strength:i+swing_strength+1].min():
            swing_lows.append((i, low.iloc[i]))
    
    return swing_highs, swing_lows

def _get_recent_high(swing_highs):
    """Get the most recent significant high"""
    if not swing_highs:
        return None
    return max(swing_highs[-3:], key=lambda x: x[1])[1] if len(swing_highs) >= 3 else swing_highs[-1][1]

def _get_recent_low(swing_lows):
    """Get the most recent significant low"""
    if not swing_lows:
        return None
    return min(swing_lows[-3:], key=lambda x: x[1])[1] if len(swing_lows) >= 3 else swing_lows[-1][1]

def _detect_breaks(structure_breaks, current_price, swing_highs, swing_lows):
    """Detect bullish and bearish breaks"""
    # Check if current price breaks recent swing high (bullish break)
    recent_high = _get_recent_high(swing_highs)
    if recent_high and current_price > recent_high:
        structure_breaks.iloc[-1, structure_breaks.columns.get_loc('bullish_break')] = True
    
    # Check if current price breaks recent swing low (bearish break)
    recent_low = _get_recent_low(swing_lows)
    if recent_low and current_price < recent_low:
        structure_breaks.iloc[-1, structure_breaks.columns.get_loc('bearish_break')] = True

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
        swing_highs, swing_lows = _find_swing_points(high, low, swing_strength)
        
        # Detect breaks
        current_price = df['close'].iloc[-1]
        _detect_breaks(structure_breaks, current_price, swing_highs, swing_lows)
        
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

def calculate_position_size_atr(capital, risk_percent, atr_value, atr_multiplier=2):
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
    
def calculate_market_microstructure(df, period=20):
    """محاسبه ساختار میکرو بازار"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
        
        # Bid-Ask Spread Proxy
        spread_proxy = (high - low) / close
        avg_spread = spread_proxy.rolling(window=period).mean()
        
        # Market Depth Indicator
        price_impact = (high - low) / volume
        market_depth = price_impact.rolling(window=period).mean()
        
        # Order Flow Imbalance
        price_change = close.pct_change()
        volume_weighted_price_change = price_change * volume
        order_flow = volume_weighted_price_change.rolling(window=period).sum()
        
        # Liquidity Score
        liquidity_score = volume / (high - low)
        liquidity_score = liquidity_score.replace([np.inf, -np.inf], 0).fillna(0)
        avg_liquidity = liquidity_score.rolling(window=period).mean()
        
        return {
            'spread_proxy': avg_spread,
            'market_depth': market_depth,
            'order_flow': order_flow,
            'liquidity_score': avg_liquidity
        }
    except Exception as e:
        logger.warning(f"Error calculating market microstructure: {e}")
        return None

def _find_pivot_points(high, low, window):
    """Helper function to find pivot highs and lows"""
    pivot_highs = []
    pivot_lows = []
    
    for i in range(window, len(high) - window):
        # Pivot High
        if high.iloc[i] == high.iloc[i-window:i+window+1].max():
            pivot_highs.append((i, high.iloc[i]))
        
        # Pivot Low
        if low.iloc[i] == low.iloc[i-window:i+window+1].min():
            pivot_lows.append((i, low.iloc[i]))
    
    return pivot_highs, pivot_lows

def _cluster_levels(levels, tolerance=0.01):
    """Helper function to cluster similar price levels"""
    if not levels:
        return []
    
    levels = sorted(levels, key=lambda x: x[1])
    clusters = []
    current_cluster = [levels[0]]
    
    for level in levels[1:]:
        if abs(level[1] - current_cluster[-1][1]) / current_cluster[-1][1] <= tolerance:
            current_cluster.append(level)
        else:
            clusters.append(current_cluster)
            current_cluster = [level]
    clusters.append(current_cluster)
    
    return clusters

def _extract_strong_levels(clusters, min_touches):
    """Helper function to extract strong levels from clusters"""
    strong_levels = []
    for cluster in clusters:
        if len(cluster) >= min_touches:
            avg_price = sum(level[1] for level in cluster) / len(cluster)
            strength = len(cluster)
            strong_levels.append((avg_price, strength))
    
    return sorted(strong_levels, key=lambda x: x[1], reverse=True)[:5]

def calculate_support_resistance_levels(df, window=20, min_touches=3):
    """محاسبه سطوح حمایت و مقاومت دقیق"""
    try:
        if df is None or len(df) < window * 2:
            return None
        
        high = df['high']
        low = df['low']
        
        # پیدا کردن نقاط pivot
        pivot_highs, pivot_lows = _find_pivot_points(high, low, window)
        
        # تجمیع سطوح مشابه
        resistance_clusters = _cluster_levels(pivot_highs)
        support_clusters = _cluster_levels(pivot_lows)
        
        # انتخاب قوی‌ترین سطوح
        strong_resistance = _extract_strong_levels(resistance_clusters, min_touches)
        strong_support = _extract_strong_levels(support_clusters, min_touches)
        
        return {
            'resistance_levels': [level[0] for level in strong_resistance],
            'support_levels': [level[0] for level in strong_support],
            'resistance_strength': [level[1] for level in strong_resistance],
            'support_strength': [level[1] for level in strong_support]
        }
    except Exception as e:
        logger.warning(f"Error calculating support/resistance levels: {e}")
        return None

def detect_dark_cloud_cover(df):
    """تشخیص الگوی Dark Cloud Cover"""
    try:
        if df is None or len(df) < 2:
            return None
        
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        patterns = pd.Series(False, index=df.index)
        
        for i in range(1, len(df)):
            # کندل اول: صعودی قوی
            first_bullish = close.iloc[i-1] > open_price.iloc[i-1]
            first_body = close.iloc[i-1] - open_price.iloc[i-1]
            
            # کندل دوم: نزولی
            second_bearish = close.iloc[i] < open_price.iloc[i]
            second_body = open_price.iloc[i] - close.iloc[i]
            
            # شرایط Dark Cloud Cover
            opens_above_prev_high = open_price.iloc[i] > high.iloc[i-1]  # باز شدن بالای کندل قبل
            closes_into_first_body = (close.iloc[i] < (open_price.iloc[i-1] + close.iloc[i-1]) / 2)  # بسته شدن در نیمه پایین کندل اول
            significant_penetration = second_body > first_body * 0.5  # نفوذ قابل توجه
            
            if (first_bullish and second_bearish and opens_above_prev_high and 
                closes_into_first_body and significant_penetration):
                patterns.iloc[i] = True
        
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting Dark Cloud Cover: {e}")
        return None

def detect_piercing_line(df):
    """تشخیص الگوی Piercing Line"""
    try:
        if df is None or len(df) < 2:
            return None
        
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        patterns = pd.Series(False, index=df.index)
        
        for i in range(1, len(df)):
            # کندل اول: نزولی قوی
            first_bearish = close.iloc[i-1] < open_price.iloc[i-1]
            first_body = open_price.iloc[i-1] - close.iloc[i-1]
            
            # کندل دوم: صعودی
            second_bullish = close.iloc[i] > open_price.iloc[i]
            second_body = close.iloc[i] - open_price.iloc[i]
            
            # شرایط Piercing Line
            opens_below_prev_low = open_price.iloc[i] < low.iloc[i-1]  # باز شدن زیر کندل قبل
            closes_into_first_body = (close.iloc[i] > (open_price.iloc[i-1] + close.iloc[i-1]) / 2)  # بسته شدن در نیمه بالای کندل اول
            significant_penetration = second_body > first_body * 0.5  # نفوذ قابل توجه
            
            if (first_bearish and second_bullish and opens_below_prev_low and 
                closes_into_first_body and significant_penetration):
                patterns.iloc[i] = True
        
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting Piercing Line: {e}")
        return None

def detect_harami_patterns(df):
    """تشخیص الگوهای Harami"""
    try:
        if df is None or len(df) < 2:
            return None
        
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        patterns = pd.DataFrame(index=df.index)
        patterns['bullish_harami'] = False
        patterns['bearish_harami'] = False
        
        for i in range(1, len(df)):
            # محاسبه اندازه بدنه کندل‌ها
            first_body_size = abs(close.iloc[i-1] - open_price.iloc[i-1])
            second_body_size = abs(close.iloc[i] - open_price.iloc[i])
            
            # کندل دوم باید در داخل کندل اول باشد
            first_max = max(open_price.iloc[i-1], close.iloc[i-1])
            first_min = min(open_price.iloc[i-1], close.iloc[i-1])
            second_max = max(open_price.iloc[i], close.iloc[i])
            second_min = min(open_price.iloc[i], close.iloc[i])
            
            is_inside = (second_max < first_max and second_min > first_min)
            is_smaller = second_body_size < first_body_size * 0.7  # کندل دوم کوچک‌تر
            
            # Bullish Harami
            first_bearish = close.iloc[i-1] < open_price.iloc[i-1]  # کندل اول نزولی
            second_bullish = close.iloc[i] > open_price.iloc[i]     # کندل دوم صعودی
            
            if first_bearish and second_bullish and is_inside and is_smaller:
                patterns.iloc[i, patterns.columns.get_loc('bullish_harami')] = True
            
            # Bearish Harami
            first_bullish = close.iloc[i-1] > open_price.iloc[i-1]  # کندل اول صعودی
            second_bearish = close.iloc[i] < open_price.iloc[i]     # کندل دوم نزولی
            
            if first_bullish and second_bearish and is_inside and is_smaller:
                patterns.iloc[i, patterns.columns.get_loc('bearish_harami')] = True
        
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting Harami patterns: {e}")
        return None

def calculate_vwap(df):
    """محاسبه Volume Weighted Average Price"""
    try:
        if df is None or len(df) < 1:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
        
        # محاسبه Typical Price
        typical_price = (high + low + close) / 3
        
        # محاسبه VWAP
        cumulative_typical_price_volume = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()
        
        vwap = cumulative_typical_price_volume / cumulative_volume
        
        # محاسبه VWAP Bands (انحراف معیار)
        vwap_variance = ((typical_price - vwap) ** 2 * volume).cumsum() / cumulative_volume
        vwap_std = np.sqrt(vwap_variance)
        
        vwap_upper1 = vwap + vwap_std
        vwap_lower1 = vwap - vwap_std
        vwap_upper2 = vwap + 2 * vwap_std
        vwap_lower2 = vwap - 2 * vwap_std
        
        return {
            'vwap': vwap,
            'vwap_upper1': vwap_upper1,
            'vwap_lower1': vwap_lower1,
            'vwap_upper2': vwap_upper2,
            'vwap_lower2': vwap_lower2
        }
    except Exception as e:
        logger.warning(f"Error calculating VWAP: {e}")
        return None

def _check_volume_filter(df, min_volume_ratio):
    """Check if volume meets minimum ratio requirement"""
    if 'volume' not in df.columns or len(df) < 20:
        return True
    
    volume_sma = df['volume'].rolling(window=20).mean().iloc[-1]
    if pd.isna(volume_sma) or volume_sma <= 0:
        return True
    
    last_volume = df.iloc[-1]['volume']
    volume_ratio = last_volume / volume_sma
    return volume_ratio >= min_volume_ratio

def _extract_signal_type(signal_data):
    """Extract signal type from signal data"""
    if isinstance(signal_data, dict) and 'type' in signal_data:
        return signal_data['type']
    elif isinstance(signal_data, str):
        return signal_data
    return None

def _check_trend_filter(df, signal_data, min_trend_strength):
    """Check if trend strength supports the signal"""
    if len(df) < 10:
        return True
    
    recent_closes = df['close'].tail(10).values
    if len(recent_closes) == 0 or recent_closes[0] == 0:
        return True
    
    trend_strength = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
    signal_type = _extract_signal_type(signal_data)
    
    if signal_type == 'buy' and trend_strength < -min_trend_strength:
        return False
    elif signal_type == 'sell' and trend_strength > min_trend_strength:
        return False
    
    return True

def filter_false_signals(df, signal_data, min_volume_ratio=1.2, min_trend_strength=0.1):
    """فیلتر سیگنال‌های کاذب"""
    try:
        if df is None or not signal_data:
            return False
        
        if not _check_volume_filter(df, min_volume_ratio):
            return False
        
        if not _check_trend_filter(df, signal_data, min_trend_strength):
            return False
        
        return True
    except Exception:
        return True

def calculate_market_structure_score(df, lookback=20):
    """Calculate market structure quality score"""
    try:
        if df is None or len(df) < lookback:
            return 0
        
        recent_data = df.tail(lookback)
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        closes = recent_data['close'].values
        
        # Higher highs and higher lows for uptrend
        higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        
        # Lower highs and lower lows for downtrend
        lower_highs = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
        lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
        
        # Price momentum consistency
        up_moves = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
        down_moves = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i-1])
        
        # Volume trend consistency
        if 'volume' in recent_data.columns:
            volumes = recent_data['volume'].values
            volume_trend = sum(1 for i in range(1, len(volumes)) if volumes[i] > volumes[i-1])
            volume_consistency = volume_trend / (len(volumes) - 1) if len(volumes) > 1 else 0.5
        else:
            volume_consistency = 0.5
        
        # Calculate structure strength
        uptrend_strength = (higher_highs + higher_lows) / (2 * (lookback - 1))
        downtrend_strength = (lower_highs + lower_lows) / (2 * (lookback - 1))
        
        # Momentum consistency
        momentum_consistency = max(up_moves, down_moves) / (len(closes) - 1) if len(closes) > 1 else 0.5
        
        # Final structure score
        if uptrend_strength > downtrend_strength:
            structure_score = (uptrend_strength * 0.4 + momentum_consistency * 0.4 + volume_consistency * 0.2) * 100
        else:
            structure_score = (downtrend_strength * 0.4 + momentum_consistency * 0.4 + volume_consistency * 0.2) * 100
        
        return min(structure_score, 100)
        
    except Exception:
        return 0
    
def safe_indicator_calculation(func, *args, **kwargs):
    """Safely calculate indicators with error handling"""
    try:
        result = func(*args, **kwargs)
        if result is not None:
            if isinstance(result, pd.DataFrame) or isinstance(result, pd.Series):
                return result
        return None
    except Exception as e:
        logger.warning(f"Error calculating indicator: {e}")
        return None

def _add_basic_indicators(df):
    """Add basic moving averages and momentum indicators"""
    # Simple Moving Averages
    df['sma20'] = safe_indicator_calculation(ta.sma, df['close'], length=20)
    df['sma50'] = safe_indicator_calculation(ta.sma, df['close'], length=50)
    df['sma200'] = safe_indicator_calculation(ta.sma, df['close'], length=200)
    
    # Exponential Moving Averages
    df['ema12'] = safe_indicator_calculation(ta.ema, df['close'], length=12)
    df['ema26'] = safe_indicator_calculation(ta.ema, df['close'], length=26)
    df['ema50'] = safe_indicator_calculation(ta.ema, df['close'], length=50)
    
    # Weighted Moving Average
    df['wma20'] = safe_indicator_calculation(ta.wma, df['close'], length=20)
    
    # RSI
    rsi = safe_indicator_calculation(ta.rsi, df['close'], length=14)
    if rsi is not None:
        df['rsi'] = rsi.fillna(50)

def _add_advanced_indicators(df):
    """Add MACD, Bollinger Bands, and Stochastic"""
    # MACD
    try:
        macd_data = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd_data is not None:
            df = df.join(macd_data.fillna(0), how='left')
    except Exception as e:
        logger.warning(f"Error calculating MACD: {e}")
    
    # Bollinger Bands
    try:
        bbands_data = ta.bbands(df['close'], length=20, std=2)
        if bbands_data is not None:
            df = df.join(bbands_data.fillna(0), how='left')
    except Exception as e:
        logger.warning(f"Error calculating Bollinger Bands: {e}")
    
    # Stochastic
    try:
        stoch_data = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if stoch_data is not None:
            df = df.join(stoch_data.fillna(50), how='left')
    except Exception as e:
        logger.warning(f"Error calculating Stochastic: {e}")

def _add_momentum_indicators(df):
    """Add momentum-based indicators"""
    # Volume SMA
    volume_sma = safe_indicator_calculation(ta.sma, df['volume'], length=20)
    if volume_sma is not None:
        df['volume_sma'] = volume_sma.fillna(0)
    
    # Basic momentum indicators
    mfi = safe_indicator_calculation(calculate_money_flow_index, df)
    if mfi is not None:
        df['mfi'] = mfi.fillna(50)
    
    cci = safe_indicator_calculation(calculate_commodity_channel_index, df)
    if cci is not None:
        df['cci'] = cci.fillna(0)
    
    williams_r = safe_indicator_calculation(calculate_williams_r, df)
    if williams_r is not None:
        df['williams_r'] = williams_r.fillna(-50)
    
    # Advanced momentum indicators
    uo = safe_indicator_calculation(calculate_ultimate_oscillator, df)
    if uo is not None:
        df['ultimate_oscillator'] = uo.fillna(50)
    
    roc = safe_indicator_calculation(calculate_rate_of_change, df)
    if roc is not None:
        df['roc'] = roc.fillna(0)
    
    ao = safe_indicator_calculation(calculate_awesome_oscillator, df)
    if ao is not None:
        df['awesome_oscillator'] = ao.fillna(0)
    
    trix = safe_indicator_calculation(calculate_trix, df)
    if trix is not None:
        df['trix'] = trix.fillna(0)
    
    dpo = safe_indicator_calculation(calculate_dpo, df)
    if dpo is not None:
        df['dpo'] = dpo.fillna(0)

def _add_trend_indicators(df):
    """Add trend-based indicators"""
    psar = safe_indicator_calculation(calculate_parabolic_sar, df)
    if psar is not None:
        df['psar'] = psar.fillna(df['close'])
    
    supertrend = safe_indicator_calculation(calculate_supertrend, df)
    if supertrend is not None:
        df['supertrend'] = supertrend.fillna(0)
    
    # Ichimoku
    ichimoku_data = safe_indicator_calculation(calculate_ichimoku, df)
    if ichimoku_data:
        for key, value in ichimoku_data.items():
            if value is not None:
                df[key] = value.fillna(0)
    
    # Fibonacci levels
    fib_levels = safe_indicator_calculation(calculate_fibonacci_levels, df)
    if fib_levels:
        for level_name, level_value in fib_levels.items():
            df[level_name] = level_value
    
    # Trend strength indicators
    aroon_osc = safe_indicator_calculation(calculate_aroon_oscillator, df)
    if aroon_osc is not None:
        df['aroon_up'] = aroon_osc['aroon_up'].fillna(0)
        df['aroon_down'] = aroon_osc['aroon_down'].fillna(0)
        df['aroon'] = aroon_osc['aroon_oscillator'].fillna(0)
    
    adx = safe_indicator_calculation(calculate_adx, df)
    if adx is not None:
        df['adx'] = adx['adx'].fillna(0)
        df['plus_di'] = adx['plus_di'].fillna(0)
        df['minus_di'] = adx['minus_di'].fillna(0)
    
    kama = safe_indicator_calculation(calculate_kama, df)
    if kama is not None:
        df['kama'] = kama.fillna(0)

def _add_volume_indicators(df):
    """Add volume-based indicators"""
    obv = safe_indicator_calculation(calculate_obv, df)
    if obv is not None:
        df['obv'] = obv.fillna(0)
    
    ad = safe_indicator_calculation(calculate_accumulation_distribution, df)
    if ad is not None:
        df['ad'] = ad.fillna(0)
    
    cmf = safe_indicator_calculation(calculate_chaikin_money_flow, df)
    if cmf is not None:
        df['cmf'] = cmf.fillna(0)
    
    vpt = safe_indicator_calculation(calculate_volume_price_trend, df)
    if vpt is not None:
        df['vpt'] = vpt.fillna(0)
    
    eom = safe_indicator_calculation(calculate_ease_of_movement, df)
    if eom is not None:
        df['eom'] = eom.fillna(0)
    
    ad_line = safe_indicator_calculation(calculate_ad_line, df)
    if ad_line is not None:
        df['ad_line'] = ad_line.fillna(0)
    
    vwap_data = safe_indicator_calculation(calculate_vwap, df)
    if vwap_data is not None:
        df['vwap'] = vwap_data.fillna(0)

def _add_volatility_indicators(df):
    """Add volatility-based indicators"""
    atr = safe_indicator_calculation(calculate_average_true_range, df)
    if atr is not None:
        df['atr'] = atr.fillna(0)
    
    keltner = safe_indicator_calculation(calculate_keltner_channels, df)
    if keltner is not None:
        for key, value in keltner.items():
            df[key] = value.fillna(0)
    
    donchian = safe_indicator_calculation(calculate_donchian_channels, df)
    if donchian is not None:
        for key, value in donchian.items():
            df[key] = value.fillna(0)
    
    std_dev = safe_indicator_calculation(calculate_standard_deviation, df)
    if std_dev is not None:
        df['std_dev'] = std_dev.fillna(0)

def _add_candlestick_patterns(df):
    """Add candlestick pattern indicators"""
    hammer_doji = safe_indicator_calculation(detect_hammer_doji_patterns, df)
    if hammer_doji is not None:
        df['hammer'] = hammer_doji['hammer'].fillna(False)
        df['doji'] = hammer_doji['doji'].fillna(False)
        df['shooting_star'] = hammer_doji['shooting_star'].fillna(False)
    
    engulfing = safe_indicator_calculation(detect_engulfing_patterns, df)
    if engulfing is not None:
        df['bullish_engulfing'] = engulfing['bullish_engulfing'].fillna(False)
        df['bearish_engulfing'] = engulfing['bearish_engulfing'].fillna(False)
    
    star = safe_indicator_calculation(detect_star_patterns, df)
    if star is not None:
        df['morning_star'] = star['morning_star'].fillna(False)
        df['evening_star'] = star['evening_star'].fillna(False)
    
    harami = safe_indicator_calculation(detect_harami_patterns, df)
    if harami is not None:
        df['bullish_harami'] = harami['bullish_harami'].fillna(False)
        df['bearish_harami'] = harami['bearish_harami'].fillna(False)
    
    piercing = safe_indicator_calculation(detect_piercing_line, df)
    if piercing is not None:
        df['piercing_line'] = piercing.fillna(False)
    
    dark_cloud = safe_indicator_calculation(detect_dark_cloud_cover, df)
    if dark_cloud is not None:
        df['dark_cloud_cover'] = dark_cloud.fillna(False)

def _add_pivot_and_structure_data(df):
    """Add pivot points and structure break data"""
    pivot = safe_indicator_calculation(calculate_pivot_points, df)
    if pivot is not None:
        for key, value in pivot.items():
            df[key] = value
    
    structure_breaks = safe_indicator_calculation(detect_market_structure_breaks, df)
    if structure_breaks is not None:
        df['bullish_break'] = structure_breaks['bullish_break'].fillna(False)
        df['bearish_break'] = structure_breaks['bearish_break'].fillna(False)

def _add_support_resistance_data(df):
    """Add support and resistance level data"""
    support_resistance = safe_indicator_calculation(calculate_support_resistance, df)
    if support_resistance is not None:
        df['support'] = support_resistance['support_levels'][0] if support_resistance['support_levels'] else None
        df['resistance'] = support_resistance['resistance_levels'][0] if support_resistance['resistance_levels'] else None
    
    sr_levels = safe_indicator_calculation(calculate_support_resistance_levels, df)
    if sr_levels is not None:
        for key, value in sr_levels.items():
            df[key] = value

def _add_correlation_and_regime_data(df, btc_df):
    """Add correlation with BTC and market regime data"""
    if btc_df is not None:
        btc_corr = safe_indicator_calculation(calculate_correlation_with_btc, df, btc_df)
        if btc_corr is not None:
            df['btc_correlation'] = btc_corr.fillna(0)
    
    market_regime = safe_indicator_calculation(detect_market_regime, df)
    if market_regime is not None:
        df['market_regime'] = market_regime.fillna('neutral')

def _add_market_structure_indicators(df, btc_df=None):
    """Add market structure and correlation indicators"""
    # Add pivot points and structure breaks
    _add_pivot_and_structure_data(df)
    
    # Add support and resistance levels
    _add_support_resistance_data(df)
    
    # Market structure score
    structure_score = safe_indicator_calculation(calculate_market_structure_score, df)
    df['market_structure_score'] = structure_score if structure_score is not None else 50
    
    # Add correlation and market regime data
    _add_correlation_and_regime_data(df, btc_df)
    
    # Market microstructure
    microstructure = safe_indicator_calculation(calculate_market_microstructure, df)
    if microstructure is not None:
        for key, value in microstructure.items():
            df[key] = value.fillna(0)

def calculate_indicators(df, btc_df=None):
    """
    Calculate comprehensive technical indicators for a given DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame with OHLCV data.
        btc_df (pd.DataFrame, optional): Bitcoin OHLCV data for correlation calculation.
    Returns:
        pd.DataFrame: DataFrame with calculated indicators or None if errors occur.
    """
    try:
        if df is None or len(df) < 50:
            logger.warning(f"Insufficient data for indicators: {len(df) if df is not None else 0} candles")
            return None
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns. Required: {required_columns}, Available: {list(df.columns)}")
            return None
        
        df_result = df.copy()
        
        # Add basic moving averages
        _add_basic_indicators(df_result)
        
        # Add advanced indicators (MACD, Bollinger Bands, Stochastic)
        _add_advanced_indicators(df_result)
        
        # Add momentum indicators
        _add_momentum_indicators(df_result)
        
        # Add trend indicators
        _add_trend_indicators(df_result)
        
        # Add volume indicators
        _add_volume_indicators(df_result)
        
        # Add volatility indicators
        _add_volatility_indicators(df_result)
        
        # Add candlestick patterns
        _add_candlestick_patterns(df_result)
        
        # Add market structure indicators
        _add_market_structure_indicators(df_result, btc_df)
        
        # Fill any remaining NaN values with appropriate defaults
        numeric_columns = df_result.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in ['rsi', 'mfi', 'ultimate_oscillator', 'williams_r']:
                df_result[col] = df_result[col].fillna(50)
            elif col in ['market_structure_score']:
                df_result[col] = df_result[col].fillna(50)
            elif col in ['btc_correlation']:
                df_result[col] = df_result[col].fillna(0)
            else:
                df_result[col] = df_result[col].fillna(method='ffill').fillna(0)
        
        # Fill boolean columns
        bool_columns = df_result.select_dtypes(include=[bool]).columns
        for col in bool_columns:
            df_result[col] = df_result[col].fillna(False)
        
        logger.info(f"Successfully calculated {len([col for col in df_result.columns if col not in df.columns])} indicators for {len(df_result)} candles")
        return df_result
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None
