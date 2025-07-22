import logging
import warnings
from datetime import datetime
import pandas as pd
from indicators import *
from market import *
import sys

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
    

# Now import pandas_ta after fixing numpy
try:
    import pandas_ta as ta
except ImportError as e:
    print(f"Error importing pandas_ta: {e}")
    print("Please install with: pip install pandas-ta==0.3.14b")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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

def calculate_indicators(df, btc_df=None):
    """
    Calculate technical indicators for a given DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame with OHLCV data.
        btc_df (pd.DataFrame, optional): Bitcoin OHLCV data for correlation calculation.
    Returns:
        pd.DataFrame: DataFrame with calculated indicators or None if errors occur.
    """
    try:
        if df is None or len(df) < 200:
            logger.warning(f"Insufficient data for indicators: {len(df) if df is not None else 0} candles")
            return None
        
        df = df.copy()
        
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
        
        # Volume indicators
        volume_sma = safe_indicator_calculation(ta.sma, df['volume'], length=20)
        if volume_sma is not None:
            df['volume_sma'] = volume_sma.fillna(0)
        
        # Import all indicators from indicators.py
        
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
        
        # Trend indicators
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
        
        # Volume indicators
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
        
        vwap = safe_indicator_calculation(calculate_vwap, df)
        if vwap is not None:
            df['vwap'] = vwap.fillna(0)
        
        # Volatility indicators
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
        
        # Candlestick patterns
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
        
        # Market structure
        pivot = safe_indicator_calculation(calculate_pivot_points, df)
        if pivot is not None:
            for key, value in pivot.items():
                df[key] = value
        
        support_resistance = safe_indicator_calculation(calculate_support_resistance, df)
        if support_resistance is not None:
            df['support'] = support_resistance['support_levels'][0] if support_resistance['support_levels'] else None
            df['resistance'] = support_resistance['resistance_levels'][0] if support_resistance['resistance_levels'] else None
        
        structure_breaks = safe_indicator_calculation(detect_market_structure_breaks, df)
        if structure_breaks is not None:
            df['bullish_break'] = structure_breaks['bullish_break'].fillna(False)
            df['bearish_break'] = structure_breaks['bearish_break'].fillna(False)
        
        # Market structure score
        structure_score = safe_indicator_calculation(calculate_market_structure_score, df)
        if structure_score is not None:
            df['market_structure_score'] = structure_score
        else:
            df['market_structure_score'] = 50  # Default neutral score
        
        # Correlation with BTC (if btc_df is provided)
        if btc_df is not None:
            btc_corr = safe_indicator_calculation(calculate_correlation_with_btc, df, btc_df)
            if btc_corr is not None:
                df['btc_correlation'] = btc_corr.fillna(0)
        
        # Market regime
        market_regime = safe_indicator_calculation(detect_market_regime, df)
        if market_regime is not None:
            df['market_regime'] = market_regime.fillna('neutral')
        
        # Support and resistance levels
        sr_levels = safe_indicator_calculation(calculate_support_resistance_levels, df)
        if sr_levels is not None:
            for key, value in sr_levels.items():
                df[key] = value
        
        # Market microstructure
        microstructure = safe_indicator_calculation(calculate_market_microstructure, df)
        if microstructure is not None:
            for key, value in microstructure.items():
                df[key] = value.fillna(0)
        
        # Check required indicators
        required_indicators = ['rsi', 'sma50', 'volume_sma', 'atr', 'market_structure_score']
        missing_indicators = [ind for ind in required_indicators if ind not in df.columns or df[ind].isnull().all()]
        
        if missing_indicators:
            logger.warning(f"Missing indicators: {missing_indicators}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

def check_signals(df, symbol):
    if df is None or len(df) < 2:
        return None
    
    try:
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        if 'rsi' not in df.columns or pd.isna(last_row['rsi']):
            logger.warning(f"No RSI data for {symbol}")
            return None
        
        rsi_value = last_row['rsi']
        current_price = last_row['close']
        
        buy_signals = 0
        sell_signals = 0
        signal_strength = 0
        signal_details = []
        
        # RSI Analysis
        if rsi_value < 30:
            buy_signals += 2
            signal_strength += 2
            signal_details.append(f"RSI Oversold: {rsi_value:.1f}")
        elif rsi_value > 70:
            sell_signals += 2
            signal_strength += 2
            signal_details.append(f"RSI Overbought: {rsi_value:.1f}")
        
        # MACD Analysis
        if ('MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns and
            not pd.isna(last_row['MACD_12_26_9']) and not pd.isna(last_row['MACDs_12_26_9']) and
            not pd.isna(prev_row['MACD_12_26_9']) and not pd.isna(prev_row['MACDs_12_26_9'])):
            
            macd_bullish = (prev_row['MACD_12_26_9'] <= prev_row['MACDs_12_26_9'] and 
                           last_row['MACD_12_26_9'] > last_row['MACDs_12_26_9'])
            
            macd_bearish = (prev_row['MACD_12_26_9'] >= prev_row['MACDs_12_26_9'] and 
                           last_row['MACD_12_26_9'] < last_row['MACDs_12_26_9'])
            
            if macd_bullish:
                buy_signals += 3
                signal_strength += 3
                signal_details.append("MACD Bullish Cross")
            elif macd_bearish:
                sell_signals += 3
                signal_strength += 3
                signal_details.append("MACD Bearish Cross")
        
        # Bollinger Bands Analysis
        if ('BBL_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns and
            not pd.isna(last_row['BBL_20_2.0']) and not pd.isna(last_row['BBU_20_2.0'])):
            
            if current_price <= last_row['BBL_20_2.0']:
                buy_signals += 2
                signal_strength += 2
                signal_details.append("Price at Lower Bollinger Band")
            elif current_price >= last_row['BBU_20_2.0']:
                sell_signals += 2
                signal_strength += 2
                signal_details.append("Price at Upper Bollinger Band")
        
        # Stochastic Analysis
        if ('STOCHk_14_3_3' in df.columns and 'STOCHd_14_3_3' in df.columns and
            not pd.isna(last_row['STOCHk_14_3_3']) and not pd.isna(last_row['STOCHd_14_3_3'])):
            
            stoch_k = last_row['STOCHk_14_3_3']
            stoch_d = last_row['STOCHd_14_3_3']
            
            if stoch_k < 20 and stoch_d < 20:
                buy_signals += 2
                signal_strength += 1
                signal_details.append(f"Stochastic Oversold: K={stoch_k:.1f}")
            elif stoch_k > 80 and stoch_d > 80:
                sell_signals += 2
                signal_strength += 1
                signal_details.append(f"Stochastic Overbought: K={stoch_k:.1f}")
        
        # MFI Analysis
        if 'mfi' in df.columns and not pd.isna(last_row['mfi']):
            mfi_value = last_row['mfi']
            
            if mfi_value < 20:
                buy_signals += 2
                signal_strength += 1
                signal_details.append(f"MFI Oversold: {mfi_value:.1f}")
            elif mfi_value > 80:
                sell_signals += 2
                signal_strength += 1
                signal_details.append(f"MFI Overbought: {mfi_value:.1f}")
        
        # CCI Analysis
        if 'cci' in df.columns and not pd.isna(last_row['cci']):
            cci_value = last_row['cci']
            
            if cci_value < -100:
                buy_signals += 1
                signal_details.append(f"CCI Oversold: {cci_value:.1f}")
            elif cci_value > 100:
                sell_signals += 1
                signal_details.append(f"CCI Overbought: {cci_value:.1f}")
        
        # Williams %R Analysis
        if 'williams_r' in df.columns and not pd.isna(last_row['williams_r']):
            wr_value = last_row['williams_r']
            
            if wr_value < -80:
                buy_signals += 1
                signal_details.append(f"Williams %R Oversold: {wr_value:.1f}")
            elif wr_value > -20:
                sell_signals += 1
                signal_details.append(f"Williams %R Overbought: {wr_value:.1f}")
        
        # Ultimate Oscillator Analysis
        if 'ultimate_oscillator' in df.columns and not pd.isna(last_row['ultimate_oscillator']):
            uo_value = last_row['ultimate_oscillator']
            
            if uo_value < 30:
                buy_signals += 1
                signal_details.append(f"Ultimate Oscillator Oversold: {uo_value:.1f}")
            elif uo_value > 70:
                sell_signals += 1
                signal_details.append(f"Ultimate Oscillator Overbought: {uo_value:.1f}")
        
        # Rate of Change Analysis
        if 'roc' in df.columns and not pd.isna(last_row['roc']):
            roc_value = last_row['roc']
            
            if roc_value > 5:
                buy_signals += 1
                signal_details.append(f"Strong Positive ROC: {roc_value:.1f}%")
            elif roc_value < -5:
                sell_signals += 1
                signal_details.append(f"Strong Negative ROC: {roc_value:.1f}%")
        
        # Awesome Oscillator Analysis
        if 'awesome_oscillator' in df.columns and not pd.isna(last_row['awesome_oscillator']) and not pd.isna(prev_row['awesome_oscillator']):
            ao_current = last_row['awesome_oscillator']
            ao_prev = prev_row['awesome_oscillator']
            
            if ao_current > 0 and ao_prev <= 0:
                buy_signals += 2
                signal_details.append("Awesome Oscillator Bullish Cross")
            elif ao_current < 0 and ao_prev >= 0:
                sell_signals += 2
                signal_details.append("Awesome Oscillator Bearish Cross")
        
        # Chaikin Money Flow Analysis
        if 'cmf' in df.columns and not pd.isna(last_row['cmf']):
            cmf_value = last_row['cmf']
            
            if cmf_value > 0.1:
                buy_signals += 1
                signal_details.append(f"Strong Money Inflow: {cmf_value:.3f}")
            elif cmf_value < -0.1:
                sell_signals += 1
                signal_details.append(f"Strong Money Outflow: {cmf_value:.3f}")
        
        # SuperTrend Analysis
        if 'supertrend' in df.columns and not pd.isna(last_row['supertrend']):
            supertrend_value = last_row['supertrend']
            
            if current_price > supertrend_value and prev_row['close'] <= df.iloc[-2].get('supertrend', 0):
                buy_signals += 2
                signal_strength += 2
                signal_details.append("SuperTrend Bullish Signal")
            elif current_price < supertrend_value and prev_row['close'] >= df.iloc[-2].get('supertrend', 0):
                sell_signals += 2
                signal_strength += 2
                signal_details.append("SuperTrend Bearish Signal")
        
        # ADX Trend Strength Analysis
        if 'adx' in df.columns and not pd.isna(last_row['adx']):
            adx_value = last_row['adx']
            
            if adx_value > 25:
                signal_strength += 1
                signal_details.append(f"Strong Trend (ADX: {adx_value:.1f})")
        
        # Aroon Oscillator Analysis
        if 'aroon' in df.columns and not pd.isna(last_row['aroon']):
            aroon_value = last_row['aroon']
            
            if aroon_value > 50:
                buy_signals += 1
                signal_details.append(f"Bullish Aroon: {aroon_value:.1f}")
            elif aroon_value < -50:
                sell_signals += 1
                signal_details.append(f"Bearish Aroon: {aroon_value:.1f}")
        
        # Candlestick Pattern Analysis
        pattern_signals = 0
        if 'hammer' in df.columns and last_row.get('hammer', False):
            buy_signals += 1
            pattern_signals += 1
            signal_details.append("Bullish Hammer Pattern")
        
        if 'doji' in df.columns and last_row.get('doji', False):
            signal_details.append("Doji Pattern (Indecision)")
        
        if 'bullish_engulfing' in df.columns and last_row.get('bullish_engulfing', False):
            buy_signals += 2
            pattern_signals += 2
            signal_details.append("Bullish Engulfing Pattern")
        
        if 'bearish_engulfing' in df.columns and last_row.get('bearish_engulfing', False):
            sell_signals += 2
            pattern_signals += 2
            signal_details.append("Bearish Engulfing Pattern")
        
        if 'morning_star' in df.columns and last_row.get('morning_star', False):
            buy_signals += 2
            pattern_signals += 2
            signal_details.append("Morning Star Pattern")
        
        if 'evening_star' in df.columns and last_row.get('evening_star', False):
            sell_signals += 2
            pattern_signals += 2
            signal_details.append("Evening Star Pattern")
        
        # Support/Resistance Analysis
        if 'support' in df.columns and 'resistance' in df.columns:
            support_level = last_row.get('support')
            resistance_level = last_row.get('resistance')
            
            if support_level and abs(current_price - support_level) / current_price < 0.005:
                buy_signals += 1
                signal_details.append(f"Near Support: {support_level:.6f}")
            
            if resistance_level and abs(current_price - resistance_level) / current_price < 0.005:
                sell_signals += 1
                signal_details.append(f"Near Resistance: {resistance_level:.6f}")
        
        # Market Structure Break Analysis
        if 'bullish_break' in df.columns and last_row.get('bullish_break', False):
            buy_signals += 2
            signal_strength += 1
            signal_details.append("Bullish Structure Break")
        
        if 'bearish_break' in df.columns and last_row.get('bearish_break', False):
            sell_signals += 2
            signal_strength += 1
            signal_details.append("Bearish Structure Break")
        
        # PSAR Analysis
        if 'psar' in df.columns and not pd.isna(last_row['psar']):
            psar_value = last_row['psar']
            
            if current_price > psar_value and prev_row['close'] <= df.iloc[-2].get('psar', 0):
                buy_signals += 2
                signal_strength += 1
                signal_details.append("PSAR Bullish Signal")
            elif current_price < psar_value and prev_row['close'] >= df.iloc[-2].get('psar', 0):
                sell_signals += 2
                signal_strength += 1
                signal_details.append("PSAR Bearish Signal")
        
        # Ichimoku Analysis
        if ('tenkan_sen' in df.columns and 'kijun_sen' in df.columns and
            not pd.isna(last_row['tenkan_sen']) and not pd.isna(last_row['kijun_sen'])):
            
            if (last_row['tenkan_sen'] > last_row['kijun_sen'] and 
                prev_row['tenkan_sen'] <= prev_row['kijun_sen']):
                buy_signals += 2
                signal_strength += 1
                signal_details.append("Ichimoku Bullish Cross")
            elif (last_row['tenkan_sen'] < last_row['kijun_sen'] and 
                prev_row['tenkan_sen'] >= prev_row['kijun_sen']):
                sell_signals += 2
                signal_strength += 1
                signal_details.append("Ichimoku Bearish Cross")
        
        # SMA Trend Analysis
        if ('sma20' in df.columns and 'sma50' in df.columns and
            not pd.isna(last_row['sma20']) and not pd.isna(last_row['sma50'])):
            
            if current_price > last_row['sma20'] > last_row['sma50']:
                buy_signals += 1
                signal_details.append("Price Above SMA20 & SMA50")
            elif current_price < last_row['sma20'] < last_row['sma50']:
                sell_signals += 1
                signal_details.append("Price Below SMA20 & SMA50")
        
        # Volume Analysis
        if 'volume_sma' in df.columns and not pd.isna(last_row['volume_sma']):
            volume_ratio = last_row['volume'] / last_row['volume_sma']
            if volume_ratio > 1.5:
                signal_strength += 1
                signal_details.append(f"High Volume: {volume_ratio:.1f}x")
        
        # Fibonacci Support/Resistance
        fibonacci_support_resistance = []
        fib_keys = ['fib_236', 'fib_382', 'fib_500', 'fib_618']
        for fib_key in fib_keys:
            if fib_key in df.columns:
                fib_level = last_row[fib_key]
                price_diff_pct = abs(current_price - fib_level) / current_price * 100
                if price_diff_pct < 1:
                    fibonacci_support_resistance.append(f"{fib_key}: {fib_level:.6f}")
        
        if fibonacci_support_resistance:
            signal_details.extend(fibonacci_support_resistance)
        
        # Signal threshold with pattern bonus
        min_signal_threshold = 3 - pattern_signals  # Reduce threshold if strong patterns found
        min_signal_threshold = max(min_signal_threshold, 1)  # Minimum threshold of 1
        
        if buy_signals >= min_signal_threshold and buy_signals > sell_signals:
            return {
                'type': 'buy',
                'strength': min(signal_strength, 5),
                'rsi': rsi_value,
                'macd': last_row.get('MACD_12_26_9', 0),
                'method': 'Advanced_Multi_Indicator_Buy',
                'details': signal_details,
                'buy_score': buy_signals,
                'sell_score': sell_signals,
                'pattern_signals': pattern_signals
            }
        elif sell_signals >= min_signal_threshold and sell_signals > buy_signals:
            return {
                'type': 'sell',
                'strength': min(signal_strength, 5),
                'rsi': rsi_value,
                'macd': last_row.get('MACD_12_26_9', 0),
                'method': 'Advanced_Multi_Indicator_Sell',
                'details': signal_details,
                'buy_score': buy_signals,
                'sell_score': sell_signals,
                'pattern_signals': pattern_signals
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error checking signals for {symbol}: {e}")
        return None

def calculate_signal_strength(df, signal_type):
    """Calculate signal strength based on multiple factors"""
    try:
        if df is None or len(df) == 0:
            return 2
            
        last_row = df.iloc[-1]
        strength_score = 0
        
        # RSI strength
        if 'rsi' in df.columns and not pd.isna(last_row['rsi']):
            rsi_value = last_row['rsi']
            if signal_type == 'buy':
                if rsi_value < 20:
                    strength_score += 3
                elif rsi_value < 25:
                    strength_score += 2
                elif rsi_value < 30:
                    strength_score += 1
            else:  # sell
                if rsi_value > 80:
                    strength_score += 3
                elif rsi_value > 75:
                    strength_score += 2
                elif rsi_value > 70:
                    strength_score += 1
        
        # Volume strength
        if 'volume_sma' in df.columns and not pd.isna(last_row['volume_sma']):
            try:
                volume_ratio = last_row['volume'] / last_row['volume_sma']
                if volume_ratio > 2:
                    strength_score += 2
                elif volume_ratio > 1.5:
                    strength_score += 1
            except (ZeroDivisionError, TypeError):
                pass
        
        # MACD strength
        if ('MACD_12_26_9' in df.columns and 
            not pd.isna(last_row['MACD_12_26_9'])):
            macd_value = abs(last_row['MACD_12_26_9'])
            if macd_value > 0.001:  # Strong MACD signal
                strength_score += 1
        
        # Normalize to 1-5 scale
        return min(max(strength_score, 1), 5)
        
    except Exception:
        return 2  # Default medium strength

def calculate_signal_accuracy_score(df, signal_data, symbol):
    try:
        if df is None or len(df) < 50 or not signal_data:
            return 0
        
        last_row = df.iloc[-1]
        prev_rows = df.iloc[-10:] if len(df) >= 10 else df
        accuracy_score = 0
        
        rsi_value = signal_data.get('rsi', 50)
        if signal_data['type'] == 'buy':
            if rsi_value < 20:
                accuracy_score += 25
            elif rsi_value < 25:
                accuracy_score += 20
            elif rsi_value < 30:
                accuracy_score += 15
            elif rsi_value < 35:
                accuracy_score += 10
        else:
            if rsi_value > 80:
                accuracy_score += 25
            elif rsi_value > 75:
                accuracy_score += 20
            elif rsi_value > 70:
                accuracy_score += 15
            elif rsi_value > 65:
                accuracy_score += 10
        
        if ('MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns and
            not pd.isna(last_row.get('MACD_12_26_9')) and not pd.isna(last_row.get('MACDs_12_26_9'))):
            
            macd_line = last_row['MACD_12_26_9']
            signal_line = last_row['MACDs_12_26_9']
            macd_histogram = macd_line - signal_line
            
            if signal_data['type'] == 'buy' and macd_histogram > 0 and macd_line > signal_line:
                accuracy_score += 20
            elif signal_data['type'] == 'sell' and macd_histogram < 0 and macd_line < signal_line:
                accuracy_score += 20
            elif abs(macd_histogram) > 0.001:
                accuracy_score += 10
        
        if 'volume_sma' in df.columns and not pd.isna(last_row.get('volume_sma')):
            try:
                volume_ratio = last_row['volume'] / last_row['volume_sma']
                if volume_ratio > 2.5:
                    accuracy_score += 15
                elif volume_ratio > 2:
                    accuracy_score += 12
                elif volume_ratio > 1.5:
                    accuracy_score += 8
                elif volume_ratio > 1.2:
                    accuracy_score += 5
            except (ZeroDivisionError, TypeError):
                pass
        
        # Market structure score analysis
        if 'market_structure_score' in df.columns:
            structure_score = last_row['market_structure_score']
            if structure_score > 70:
                accuracy_score += 15
            elif structure_score > 50:
                accuracy_score += 10
            elif structure_score > 30:
                accuracy_score += 5
        
        if all(col in df.columns for col in ['sma20', 'sma50', 'sma200']):
            current_price = last_row['close']
            sma20 = last_row.get('sma20')
            sma50 = last_row.get('sma50')
            sma200 = last_row.get('sma200')
            
            if not any(pd.isna(val) for val in [sma20, sma50, sma200]):
                if signal_data['type'] == 'buy':
                    if current_price > sma20 > sma50 > sma200:
                        accuracy_score += 15
                    elif current_price > sma20 > sma50:
                        accuracy_score += 10
                    elif current_price > sma20:
                        accuracy_score += 5
                else:
                    if current_price < sma20 < sma50 < sma200:
                        accuracy_score += 15
                    elif current_price < sma20 < sma50:
                        accuracy_score += 10
                    elif current_price < sma20:
                        accuracy_score += 5
        
        if 'STOCHk_14_3_3' in df.columns and 'STOCHd_14_3_3' in df.columns:
            k_value = last_row.get('STOCHk_14_3_3')
            d_value = last_row.get('STOCHd_14_3_3')
            
            if not pd.isna(k_value) and not pd.isna(d_value):
                if signal_data['type'] == 'buy' and k_value < 20 and d_value < 20:
                    accuracy_score += 10
                elif signal_data['type'] == 'sell' and k_value > 80 and d_value > 80:
                    accuracy_score += 10
                elif signal_data['type'] == 'buy' and k_value < 30:
                    accuracy_score += 5
                elif signal_data['type'] == 'sell' and k_value > 70:
                    accuracy_score += 5
        
        if 'mfi' in df.columns and not pd.isna(last_row.get('mfi')):
            mfi_value = last_row['mfi']
            if signal_data['type'] == 'buy' and mfi_value < 20:
                accuracy_score += 8
            elif signal_data['type'] == 'sell' and mfi_value > 80:
                accuracy_score += 8
            elif signal_data['type'] == 'buy' and mfi_value < 30:
                accuracy_score += 4
            elif signal_data['type'] == 'sell' and mfi_value > 70:
                accuracy_score += 4
        
        if 'cci' in df.columns and not pd.isna(last_row.get('cci')):
            cci_value = last_row['cci']
            if signal_data['type'] == 'buy' and cci_value < -100:
                accuracy_score += 5
            elif signal_data['type'] == 'sell' and cci_value > 100:
                accuracy_score += 5
        
        if len(prev_rows) >= 5:
            trend_direction = 0
            close_prices = prev_rows['close'].values
            
            for i in range(1, len(close_prices)):
                if close_prices[i] > close_prices[i-1]:
                    trend_direction += 1
                elif close_prices[i] < close_prices[i-1]:
                    trend_direction -= 1
            
            trend_strength = abs(trend_direction) / len(close_prices)
            
            if signal_data['type'] == 'buy' and trend_direction > 0:
               accuracy_score += int(10 * trend_strength)
            elif signal_data['type'] == 'sell' and trend_direction < 0:
               accuracy_score += int(10 * trend_strength)

        if symbol in ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']:
            accuracy_score += 5
        
        buy_score = signal_data.get('buy_score', 0)
        sell_score = signal_data.get('sell_score', 0)
        signal_dominance = max(buy_score, sell_score) - min(buy_score, sell_score)
        accuracy_score += min(signal_dominance * 2, 10)

        accuracy_score = min(accuracy_score, 100)
        
        logger.info(f"Accuracy score for {symbol}: {accuracy_score}")
        return accuracy_score
        
    except Exception as e:
        logger.error(f"Error calculating accuracy score for {symbol}: {e}")
        return 0
