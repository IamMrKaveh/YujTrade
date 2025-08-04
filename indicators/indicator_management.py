from logger_config import logger
import pandas as pd

from .candlestick_patterns import detect_all_candlestick_patterns
from .correlation import comprehensive_correlation_analysis, quick_correlation_check
from .fibonacci import calculate_fibonacci_levels
from .market_structure import analyze_market_structure
from .momentum_indicators import calculate_technical_indicators
from .moving_averages import calculate_moving_averages
from .support_resistance import calculate_pivot_points, calculate_support_resistance_levels
from .trend_indicators import (
    calculate_adx_internal, calculate_aroon, calculate_donchian_channels,
    calculate_supertrend, calculate_trend_strength
)
from .volatility_indicators import calculate_atr, calculate_bollinger_bands
from .volume_indicators import calculate_ad_line, calculate_chaikin_money_flow, calculate_obv, calculate_vwap


class IndicatorConfig:
    @staticmethod
    def calculate_indicators(df, config=None):
        """
        Calculate all available technical indicators on the given DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame containing OHLCV data
            config (dict, optional): Configuration parameters for indicators
            
        Returns:
            dict: Dictionary containing indicators and trading signals
        """
        if df is None or df.empty:
            logger.error("DataFrame is None or empty")
            return {}
        
        # Default configuration
        default_config = {
            'use_btc_correlation': False,
            'btc_df': None,
            'ma_types': ['sma', 'ema', 'wma'],
            'sma_period': 20,
            'ema_period': 20,
            'wma_period': 20,
            'fibonacci_lookback': 50,
            'support_resistance_window': 20,
            'support_resistance_min_touches': 3,
            'trend_adx_period': 14,
            'volatility_bollinger_period': 20,
            'volatility_bollinger_std': 2,
            'volatility_atr_period': 14,
            'volume_cmf_period': 20,
            'market_structure_swing_strength': 5,
        }
        
        # Update default config with provided config
        if config:
            default_config.update(config)
        
        # Initialize results dictionary
        indicators = {}
        
        try:
            # Calculate moving averages
            logger.info("Calculating moving averages...")
            ma_results = calculate_moving_averages(
                df, 
                ma_types=default_config['ma_types'],
                sma_period=default_config['sma_period'],
                ema_period=default_config['ema_period'],
                wma_period=default_config['wma_period']
            )
            indicators.update(ma_results)
            
            # Calculate candlestick patterns
            logger.info("Detecting candlestick patterns...")
            candlestick_patterns = detect_all_candlestick_patterns(df)
            indicators['candlestick_patterns'] = candlestick_patterns
            
            # Calculate Fibonacci levels
            logger.info("Calculating Fibonacci levels...")
            fibonacci_levels = calculate_fibonacci_levels(df, default_config['fibonacci_lookback'])
            indicators['fibonacci_levels'] = fibonacci_levels
            
            # Calculate pivot points and support/resistance
            logger.info("Calculating support/resistance levels...")
            pivot_points = calculate_pivot_points(df)
            indicators['pivot_points'] = pivot_points
            
            support_resistance = calculate_support_resistance_levels(
                df, 
                window=default_config['support_resistance_window'],
                min_touches=default_config['support_resistance_min_touches']
            )
            indicators['support_resistance'] = support_resistance
            
            # Calculate trend indicators
            logger.info("Calculating trend indicators...")
            adx_results = calculate_adx_internal(df, default_config['trend_adx_period'])
            if adx_results:
                indicators.update(adx_results)
            
            aroon_results = calculate_aroon(df)
            if aroon_results:
                indicators.update(aroon_results)
            
            donchian_results = calculate_donchian_channels(df)
            if donchian_results:
                indicators.update(donchian_results)
            
            supertrend_results = calculate_supertrend(df)
            if supertrend_results:
                indicators['supertrend'] = supertrend_results
            
            trend_strength_results = calculate_trend_strength(df)
            if trend_strength_results:
                indicators['trend_strength'] = trend_strength_results
            
            # Calculate volatility indicators
            logger.info("Calculating volatility indicators...")
            atr = calculate_atr(df, default_config['volatility_atr_period'])
            indicators['atr'] = atr
            
            bollinger_results = calculate_bollinger_bands(
                df, 
                default_config['volatility_bollinger_period'], 
                default_config['volatility_bollinger_std']
            )
            if bollinger_results:
                indicators.update(bollinger_results)
            
            # Calculate volume indicators
            logger.info("Calculating volume indicators...")
            if 'volume' in df.columns:
                obv = calculate_obv(df)
                indicators['obv'] = obv
                
                ad_line = calculate_ad_line(df)
                indicators['ad_line'] = ad_line
                
                cmf = calculate_chaikin_money_flow(df, default_config['volume_cmf_period'])
                indicators['cmf'] = cmf
                
                vwap_results = calculate_vwap(df)
                if vwap_results:
                    indicators.update(vwap_results)
            
            # Calculate momentum indicators
            logger.info("Calculating momentum indicators...")
            momentum_indicators = calculate_technical_indicators(df)
            indicators.update(momentum_indicators)
            
            # Calculate market structure
            logger.info("Analyzing market structure...")
            market_structure = analyze_market_structure(
                df, 
                swing_strength=default_config['market_structure_swing_strength']
            )
            indicators['market_structure'] = market_structure
            
            # Calculate correlation if BTC data is provided
            if default_config['use_btc_correlation'] and default_config['btc_df'] is not None:
                logger.info("Calculating BTC correlation...")
                btc_correlation = quick_correlation_check(df, default_config['btc_df'])
                indicators['btc_correlation'] = btc_correlation
            
            # Process the results to create trading signals
            signals = IndicatorConfig._generate_trading_signals(indicators, df)
            
            # Log successful calculations
            logger.info(f"Successfully calculated {len(indicators)} indicator groups")
            
            return {
                'indicators': indicators,
                'timestamp': pd.Timestamp.now(),
                'signals': signals
            }
            
        except Exception as e:
            logger.error(f"Error in calculate_indicators: {e}")
            return {
                'indicators': indicators,
                'timestamp': pd.Timestamp.now(),
                'signals': {},
                'error': str(e)
            }

    @staticmethod
    def _generate_trading_signals(indicators, df):
        """
        Generate trading signals based on calculated indicators
        
        Args:
            indicators (dict): Dictionary of calculated indicators
            df (pd.DataFrame): Original price DataFrame
        
        Returns:
            dict: Trading signals with entry and exit points
        """
        signals = {
            'buy': [],
            'sell': [],
            'strength': 0,
            'entry_points': [],
            'exit_points': []
        }
        
        try:
            if not indicators or df.empty:
                return signals
            
            current_price = df['close'].iloc[-1]
            
            # Check moving averages crossovers
            if 'sma' in indicators and 'ema' in indicators:
                sma = indicators['sma']
                ema = indicators['ema']
                
                if not sma.empty and not ema.empty and len(sma) > 1 and len(ema) > 1:
                    # Bullish crossover: EMA crosses above SMA
                    if ema.iloc[-2] <= sma.iloc[-2] and ema.iloc[-1] > sma.iloc[-1]:
                        signals['buy'].append({
                            'indicator': 'ma_crossover',
                            'confidence': 70,
                            'entry': current_price,
                            'exit': current_price * 1.05
                        })
                    
                    # Bearish crossover: EMA crosses below SMA
                    if ema.iloc[-2] >= sma.iloc[-2] and ema.iloc[-1] < sma.iloc[-1]:
                        signals['sell'].append({
                            'indicator': 'ma_crossover',
                            'confidence': 70,
                            'entry': current_price,
                            'exit': current_price * 0.95
                        })
            
            # Check RSI for overbought/oversold conditions
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                
                if not isinstance(rsi, pd.Series):
                    rsi = pd.Series(rsi)
                    
                if not rsi.empty and len(rsi) > 1:
                    # Oversold condition: RSI crosses above 30 from below
                    if rsi.iloc[-2] < 30 and rsi.iloc[-1] >= 30:
                        signals['buy'].append({
                            'indicator': 'rsi_oversold',
                            'confidence': 65,
                            'entry': current_price,
                            'exit': current_price * 1.04
                        })
                    
                    # Overbought condition: RSI crosses below 70 from above
                    if rsi.iloc[-2] > 70 and rsi.iloc[-1] <= 70:
                        signals['sell'].append({
                            'indicator': 'rsi_overbought',
                            'confidence': 65,
                            'entry': current_price,
                            'exit': current_price * 0.96
                        })
            
            # Check candlestick patterns
            if 'candlestick_patterns' in indicators:
                patterns = indicators['candlestick_patterns']
                
                # Bullish patterns
                bullish_patterns = ['bullish_engulfing', 'hammer', 'morning_star', 'piercing_line']
                for pattern in bullish_patterns:
                    if pattern in patterns and patterns[pattern].iloc[-1]:
                        signals['buy'].append({
                            'indicator': f'candlestick_{pattern}',
                            'confidence': 75 if pattern == 'morning_star' else 60,
                            'entry': current_price,
                            'exit': current_price * 1.03
                        })
                
                # Bearish patterns
                bearish_patterns = ['bearish_engulfing', 'shooting_star', 'evening_star', 'dark_cloud_cover']
                for pattern in bearish_patterns:
                    if pattern in patterns and patterns[pattern].iloc[-1]:
                        signals['sell'].append({
                            'indicator': f'candlestick_{pattern}',
                            'confidence': 75 if pattern == 'evening_star' else 60,
                            'entry': current_price,
                            'exit': current_price * 0.97
                        })
            
            # Calculate overall signal strength and determine best entry/exit points
            if signals['buy']:
                buy_confidence = [signal['confidence'] for signal in signals['buy']]
                max_confidence_idx = buy_confidence.index(max(buy_confidence))
                signals['strength'] = max(buy_confidence) / 100
                signals['entry_points'] = [signals['buy'][max_confidence_idx]['entry']]
                signals['exit_points'] = [signals['buy'][max_confidence_idx]['exit']]
            elif signals['sell']:
                sell_confidence = [signal['confidence'] for signal in signals['sell']]
                max_confidence_idx = sell_confidence.index(max(sell_confidence))
                signals['strength'] = -max(sell_confidence) / 100
                signals['entry_points'] = [signals['sell'][max_confidence_idx]['entry']]
                signals['exit_points'] = [signals['sell'][max_confidence_idx]['exit']]
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
        
        return signals