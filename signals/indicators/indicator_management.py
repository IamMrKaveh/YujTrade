import sys
from logger_config import logger
import numpy as np
import pandas as pd

# Fix numpy compatibility issue
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

# Now import pandas_ta after fixing numpy
try:
    import pandas_ta as ta
except ImportError as e:
    print(f"Error importing pandas_ta: {e}")
    print("Please install with: pip install pandas-ta==0.3.14b")
    sys.exit(1)


from .candlestick_patterns import _detect_dark_cloud_cover, _detect_engulfing_patterns, _detect_hammer_doji_patterns, _detect_harami_patterns, _detect_piercing_line, _detect_star_patterns
from .correlation import _calculate_correlation_with_btc
from .fibonacci import _calculate_fibonacci_levels
from .market_structure import _calculate_market_microstructure, _calculate_market_structure_score, _detect_market_regime, _detect_market_structure_breaks
from .momentum_indicators import _calculate_awesome_oscillator, _calculate_commodity_channel_index, _calculate_dpo, _calculate_money_flow_index, _calculate_rate_of_change, _calculate_trix, _calculate_ultimate_oscillator, _calculate_williams_r
from .moving_averages import _calculate_kama
from .support_resistance import _calculate_pivot_points, _calculate_support_resistance, _calculate_support_resistance_levels
from .trend_indicators import _calculate_adx, _calculate_aroon_oscillator, _calculate_donchian_channels, _calculate_ichimoku, _calculate_keltner_channels, _calculate_parabolic_sar, _calculate_supertrend
from .volatility_indicators import _calculate_average_true_range, _calculate_standard_deviation
from .volume_indicators import _calculate_accumulation_distribution, _calculate_ad_line, _calculate_chaikin_money_flow, _calculate_ease_of_movement, _calculate_obv, _calculate_volume_price_trend, _calculate_vwap



def _safe_indicator_calculation(func, *args, **kwargs):
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
    df['sma20'] = _safe_indicator_calculation(ta.sma, df['close'], length=20)
    df['sma50'] = _safe_indicator_calculation(ta.sma, df['close'], length=50)
    df['sma200'] = _safe_indicator_calculation(ta.sma, df['close'], length=200)
    
    # Exponential Moving Averages
    df['ema12'] = _safe_indicator_calculation(ta.ema, df['close'], length=12)
    df['ema26'] = _safe_indicator_calculation(ta.ema, df['close'], length=26)
    df['ema50'] = _safe_indicator_calculation(ta.ema, df['close'], length=50)
    
    # Weighted Moving Average
    df['wma20'] = _safe_indicator_calculation(ta.wma, df['close'], length=20)
    
    # RSI
    rsi = _safe_indicator_calculation(ta.rsi, df['close'], length=14)
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
    volume_sma = _safe_indicator_calculation(ta.sma, df['volume'], length=20)
    if volume_sma is not None:
        df['volume_sma'] = volume_sma.fillna(0)
    
    # Basic momentum indicators
    mfi = _safe_indicator_calculation(_calculate_money_flow_index, df)
    if mfi is not None:
        df['mfi'] = mfi.fillna(50)
    
    cci = _safe_indicator_calculation(_calculate_commodity_channel_index, df)
    if cci is not None:
        df['cci'] = cci.fillna(0)
    
    williams_r = _safe_indicator_calculation(_calculate_williams_r, df)
    if williams_r is not None:
        df['williams_r'] = williams_r.fillna(-50)
    
    # Advanced momentum indicators
    uo = _safe_indicator_calculation(_calculate_ultimate_oscillator, df)
    if uo is not None:
        df['ultimate_oscillator'] = uo.fillna(50)

    roc = _safe_indicator_calculation(_calculate_rate_of_change, df)
    if roc is not None:
        df['roc'] = roc.fillna(0)

    ao = _safe_indicator_calculation(_calculate_awesome_oscillator, df)
    if ao is not None:
        df['awesome_oscillator'] = ao.fillna(0)
    
    trix = _safe_indicator_calculation(_calculate_trix, df)
    if trix is not None:
        df['trix'] = trix.fillna(0)
    
    dpo = _safe_indicator_calculation(_calculate_dpo, df)
    if dpo is not None:
        df['dpo'] = dpo.fillna(0)

def _add_trend_indicators(df):
    """Add trend-based indicators"""
    psar = _safe_indicator_calculation(_calculate_parabolic_sar, df)
    if psar is not None:
        df['psar'] = psar.fillna(df['close'])

    supertrend = _safe_indicator_calculation(_calculate_supertrend, df)
    if supertrend is not None:
        df['supertrend'] = supertrend.fillna(0)
    
    # Ichimoku
    ichimoku_data = _safe_indicator_calculation(_calculate_ichimoku, df)
    if ichimoku_data:
        for key, value in ichimoku_data.items():
            if value is not None:
                df[key] = value.fillna(0)
    
    # Fibonacci levels
    fib_levels = _safe_indicator_calculation(_calculate_fibonacci_levels, df)
    if fib_levels:
        for level_name, level_value in fib_levels.items():
            df[level_name] = level_value
    
    # Trend strength indicators
    aroon_osc = _safe_indicator_calculation(_calculate_aroon_oscillator, df)
    if aroon_osc is not None:
        df['aroon_up'] = aroon_osc['aroon_up'].fillna(0)
        df['aroon_down'] = aroon_osc['aroon_down'].fillna(0)
        df['aroon'] = aroon_osc['aroon_oscillator'].fillna(0)
    
    adx = _safe_indicator_calculation(_calculate_adx, df)
    if adx is not None:
        df['adx'] = adx['adx'].fillna(0)
        df['plus_di'] = adx['plus_di'].fillna(0)
        df['minus_di'] = adx['minus_di'].fillna(0)
    
    kama = _safe_indicator_calculation(_calculate_kama, df)
    if kama is not None:
        df['kama'] = kama.fillna(0)

def _add_volume_indicators(df):
    """Add volume-based indicators"""
    obv = _safe_indicator_calculation(_calculate_obv, df)
    if obv is not None:
        df['obv'] = obv.fillna(0)
    
    ad = _safe_indicator_calculation(_calculate_accumulation_distribution, df)
    if ad is not None:
        df['ad'] = ad.fillna(0)

    cmf = _safe_indicator_calculation(_calculate_chaikin_money_flow, df)
    if cmf is not None:
        df['cmf'] = cmf.fillna(0)

    vpt = _safe_indicator_calculation(_calculate_volume_price_trend, df)
    if vpt is not None:
        df['vpt'] = vpt.fillna(0)

    eom = _safe_indicator_calculation(_calculate_ease_of_movement, df)
    if eom is not None:
        df['eom'] = eom.fillna(0)

    ad_line = _safe_indicator_calculation(_calculate_ad_line, df)
    if ad_line is not None:
        df['ad_line'] = ad_line.fillna(0)
    
    vwap_data = _safe_indicator_calculation(_calculate_vwap, df)
    if vwap_data is not None:
        df['vwap'] = vwap_data.fillna(0)

def _add_volatility_indicators(df):
    """Add volatility-based indicators"""
    atr = _safe_indicator_calculation(_calculate_average_true_range, df)
    if atr is not None:
        df['atr'] = atr.fillna(0)
    
    keltner = _safe_indicator_calculation(_calculate_keltner_channels, df)
    if keltner is not None:
        for key, value in keltner.items():
            df[key] = value.fillna(0)
    
    donchian = _safe_indicator_calculation(_calculate_donchian_channels, df)
    if donchian is not None:
        for key, value in donchian.items():
            df[key] = value.fillna(0)
    
    std_dev = _safe_indicator_calculation(_calculate_standard_deviation, df)
    if std_dev is not None:
        df['std_dev'] = std_dev.fillna(0)

def _add_candlestick_patterns(df):
    """Add candlestick pattern indicators"""
    hammer_doji = _safe_indicator_calculation(_detect_hammer_doji_patterns, df)
    if hammer_doji is not None:
        df['hammer'] = hammer_doji['hammer'].fillna(False)
        df['doji'] = hammer_doji['doji'].fillna(False)
        df['shooting_star'] = hammer_doji['shooting_star'].fillna(False)
    
    engulfing = _safe_indicator_calculation(_detect_engulfing_patterns, df)
    if engulfing is not None:
        df['bullish_engulfing'] = engulfing['bullish_engulfing'].fillna(False)
        df['bearish_engulfing'] = engulfing['bearish_engulfing'].fillna(False)
    
    star = _safe_indicator_calculation(_detect_star_patterns, df)
    if star is not None:
        df['morning_star'] = star['morning_star'].fillna(False)
        df['evening_star'] = star['evening_star'].fillna(False)
    
    harami = _safe_indicator_calculation(_detect_harami_patterns, df)
    if harami is not None:
        df['bullish_harami'] = harami['bullish_harami'].fillna(False)
        df['bearish_harami'] = harami['bearish_harami'].fillna(False)
    
    piercing = _safe_indicator_calculation(_detect_piercing_line, df)
    if piercing is not None:
        df['piercing_line'] = piercing.fillna(False)
    
    dark_cloud = _safe_indicator_calculation(_detect_dark_cloud_cover, df)
    if dark_cloud is not None:
        df['dark_cloud_cover'] = dark_cloud.fillna(False)

def _add_pivot_and_structure_data(df):
    """Add pivot points and structure break data"""
    pivot = _safe_indicator_calculation(_calculate_pivot_points, df)
    if pivot is not None:
        for key, value in pivot.items():
            df[key] = value
    
    structure_breaks = _safe_indicator_calculation(_detect_market_structure_breaks, df)
    if structure_breaks is not None:
        df['bullish_break'] = structure_breaks['bullish_break'].fillna(False)
        df['bearish_break'] = structure_breaks['bearish_break'].fillna(False)

def _add_support_resistance_data(df):
    """Add support and resistance level data"""
    support_resistance = _safe_indicator_calculation(_calculate_support_resistance, df)
    if support_resistance is not None:
        df['support'] = support_resistance['support_levels'][0] if support_resistance['support_levels'] else None
        df['resistance'] = support_resistance['resistance_levels'][0] if support_resistance['resistance_levels'] else None
    
    sr_levels = _safe_indicator_calculation(_calculate_support_resistance_levels, df)
    if sr_levels is not None:
        for key, value in sr_levels.items():
            df[key] = value

def _add_correlation_and_regime_data(df, btc_df):
    """Add correlation with BTC and market regime data"""
    if btc_df is not None:
        btc_corr = _safe_indicator_calculation(_calculate_correlation_with_btc, df, btc_df)
        if btc_corr is not None:
            df['btc_correlation'] = btc_corr.fillna(0)
    
    market_regime = _safe_indicator_calculation(_detect_market_regime, df)
    if market_regime is not None:
        df['market_regime'] = market_regime.fillna('neutral')

def _add_market_structure_indicators(df, btc_df=None):
    """Add market structure and correlation indicators"""
    # Add pivot points and structure breaks
    _add_pivot_and_structure_data(df)
    
    # Add support and resistance levels
    _add_support_resistance_data(df)
    
    # Market structure score
    structure_score = _safe_indicator_calculation(_calculate_market_structure_score, df)
    df['market_structure_score'] = structure_score if structure_score is not None else 50
    
    # Add correlation and market regime data
    _add_correlation_and_regime_data(df, btc_df)
    
    # Market microstructure
    microstructure = _safe_indicator_calculation(_calculate_market_microstructure, df)
    if microstructure is not None:
        for key, value in microstructure.items():
            df[key] = value.fillna(0)

def _create_custom_strategy():
    """Create a custom pandas_ta strategy with all indicators"""
    try:
        from pandas_ta import Strategy
        
        # Create custom strategy
        custom_strategy = Strategy(
            name="comprehensive_indicators",
            description="Comprehensive technical analysis indicators",
            ta=[
                # Moving Averages
                {"kind": "sma", "length": 20},
                {"kind": "sma", "length": 50},
                {"kind": "sma", "length": 200},
                {"kind": "ema", "length": 12},
                {"kind": "ema", "length": 26},
                {"kind": "ema", "length": 50},
                {"kind": "wma", "length": 20},
                
                # Momentum Indicators
                {"kind": "rsi", "length": 14},
                {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
                {"kind": "stoch", "k": 14, "d": 3},
                {"kind": "mfi", "length": 14},
                {"kind": "cci", "length": 20},
                {"kind": "willr", "length": 14},
                
                # Volatility Indicators
                {"kind": "bbands", "length": 20, "std": 2},
                {"kind": "atr", "length": 14},
                
                # Volume Indicators
                {"kind": "obv"},
                {"kind": "ad"},
                {"kind": "cmf", "length": 20},
                {"kind": "vwap"},
                
                # Trend Indicators
                {"kind": "adx", "length": 14},
                {"kind": "psar"},
                {"kind": "supertrend", "length": 10, "multiplier": 3.0},
                
                # Candlestick Patterns
                {"kind": "cdl_pattern", "name": "all"},
            ]
        )
        return custom_strategy
    except Exception as e:
        logger.warning(f"Error creating custom strategy: {e}")
        return None

def _validate_data(df):
    """Validate input data for indicator calculation"""
    if df is None or len(df) < 20:
        logger.warning(f"Insufficient data for indicators: {len(df) if df is not None else 0} candles")
        return False
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Missing required columns. Required: {required_columns}, Available: {list(df.columns)}")
        return False
    
    return True

def _apply_pandas_ta_strategy(df):
    """Apply pandas_ta strategy to calculate indicators"""
    try:
        custom_strategy = _create_custom_strategy()
        if custom_strategy is None:
            return _fallback_indicator_calculation(df)
        
        # Apply strategy
        df.ta.strategy(custom_strategy)
        return df
    except Exception as e:
        logger.warning(f"Error applying pandas_ta strategy: {e}")
        return _fallback_indicator_calculation(df)

def _fallback_indicator_calculation(df):
    """Fallback method using individual indicator calculations"""
    df_result = df.copy()
    
    # Calculate essential indicators manually
    df_result['sma20'] = _safe_indicator_calculation(ta.sma, df['close'], length=20)
    df_result['sma50'] = _safe_indicator_calculation(ta.sma, df['close'], length=50)
    df_result['ema12'] = _safe_indicator_calculation(ta.ema, df['close'], length=12)
    df_result['ema26'] = _safe_indicator_calculation(ta.ema, df['close'], length=26)
    df_result['rsi'] = _safe_indicator_calculation(ta.rsi, df['close'], length=14)
    
    # Add MACD
    macd_data = _safe_indicator_calculation(ta.macd, df['close'], fast=12, slow=26, signal=9)
    if macd_data is not None:
        df_result = df_result.join(macd_data, how='left')
    
    # Add Bollinger Bands
    bbands_data = _safe_indicator_calculation(ta.bbands, df['close'], length=20, std=2)
    if bbands_data is not None:
        df_result = df_result.join(bbands_data, how='left')
    
    return df_result

def _fill_nan_values(df):
    """Fill NaN values with appropriate defaults"""
    try:
        # Fill numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        fillna_map = {
            'rsi': 50, 'mfi': 50, 'ultimate_oscillator': 50,
            'williams_r': -50, 'willr_14': -50,
            'adx': 25, 'plus_di': 25, 'minus_di': 25
        }
        
        for col in numeric_columns:
            default_value = fillna_map.get(col, 0)
            df[col] = df[col].fillna(default_value)
        
        # Fill boolean columns
        bool_columns = df.select_dtypes(include=[bool]).columns
        for col in bool_columns:
            df[col] = df[col].fillna(False)
            
    except Exception as e:
        logger.warning(f"Error filling NaN values: {e}")

def calculate_indicators(df, btc_df=None):
    """
    Calculate comprehensive technical indicators using pandas_ta strategy.
    Args:
        df (pd.DataFrame): Input DataFrame with OHLCV data.
        btc_df (pd.DataFrame, optional): Bitcoin OHLCV data for correlation calculation.
    Returns:
        pd.DataFrame: DataFrame with calculated indicators or None if errors occur.
    """
    try:
        if not _validate_data(df):
            return df if df is not None else None
        
        df_result = df.copy()
        
        # Apply pandas_ta strategy for main indicators
        df_result = _apply_pandas_ta_strategy(df_result)
        
        # Add custom indicators not available in pandas_ta
        _add_market_structure_indicators(df_result, btc_df)
        
        # Fill NaN values
        _fill_nan_values(df_result)
        
        logger.info(f"Successfully calculated indicators for {len(df_result)} candles")
        return df_result
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return df if df is not None else None
    
