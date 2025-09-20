import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from module.analyzers import (DynamicLevelCalculator, MarketConditionAnalyzer,
                              PatternAnalyzer)
from module.constants import (MULTI_TF_CONFIRMATION_MAP,
                              MULTI_TF_CONFIRMATION_WEIGHTS)
from module.core import (IndicatorResult, SignalType, TradingSignal,
                         TrendDirection, TrendStrength)
from module.indicators import (ADXIndicator, ATRIndicator,
                               BollingerBandsIndicator, CCIIndicator,
                               ChaikinMoneyFlowIndicator, IchimokuIndicator,
                               MACDIndicator, MovingAverageIndicator,
                               OBVIndicator, RSIIndicator, StochasticIndicator,
                               SuperTrendIndicator, VolumeIndicator,
                               WilliamsRIndicator)
from module.logger_config import logger
from module.lstm import LSTMModelManager


class MultiTimeframeAnalyzer:
    def __init__(self, exchange_manager, indicators, cache_ttl=300):
        self.exchange_manager = exchange_manager
        self.indicators = indicators
        self._cache = {}
        self._cache_expiry = {}
        self.cache_ttl = cache_ttl
        
    def _get_cache(self, key):
        if key in self._cache and time.time() < self._cache_expiry.get(key, 0):
            return self._cache[key]
        return None

    def _set_cache(self, key, value):
        self._cache[key] = value
        self._cache_expiry[key] = time.time() + self.cache_ttl

    async def is_direction_aligned(self, symbol: str, exec_tf: str, threshold: float = 0.6) -> bool:
        try:
            confirm_tfs = MULTI_TF_CONFIRMATION_MAP.get(exec_tf, [])
            if not confirm_tfs:
                return True
            base_df = await self.exchange_manager.fetch_ohlcv_data(symbol, exec_tf)
            if base_df.empty or len(base_df) < 50:
                return False

            self._cache = getattr(self, '_cache', {})
            base_key = (symbol, exec_tf)
            if self._get_cache(base_key):
                base_signals = self._get_cache(base_key)
            else:
                base_signals = self._analyze_indicators(base_df)
                self._set_cache(base_key, base_signals)

            if not base_signals:
                return False

            total_score = 0
            total_weight = 0
            for tf in confirm_tfs:
                try:
                    confirm_df = await self.exchange_manager.fetch_ohlcv_data(symbol, tf)
                    if confirm_df.empty or len(confirm_df) < 50:
                        continue
                    confirm_key = (symbol, tf)
                    if self._get_cache(confirm_key):
                        confirm_signals = self._get_cache(confirm_key)
                    else:
                        confirm_signals = self._analyze_indicators(confirm_df)
                        self._set_cache(confirm_key, confirm_signals)
                    if not confirm_signals:
                        continue

                    weight = MULTI_TF_CONFIRMATION_WEIGHTS.get(exec_tf, {}).get(tf, 1.0)
                    matches = sum(1 for k in base_signals
                                if k in confirm_signals and self._signals_match(base_signals[k], confirm_signals[k]))
                    score = matches / len(base_signals) if base_signals else 0
                    total_score += score * weight
                    total_weight += weight
                except Exception as e:
                    logger.warning(f"Error analyzing timeframe {tf} for {symbol}: {e}")
                    continue

            if total_weight == 0:
                return False
            avg_score = total_score / total_weight
            return avg_score >= threshold
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis for {symbol}: {e}")
            return False

    def _signals_match(self, signal1: str, signal2: str) -> bool:
        bullish_signals = ['bullish', 'oversold', 'bullish_crossover', 'bullish_above_ma', 'buy_pressure', 'bullish_engulfing', 'price_above_cloud']
        bearish_signals = ['bearish', 'overbought', 'bearish_crossover', 'bearish_below_ma', 'sell_pressure', 'bearish_engulfing', 'price_below_cloud']
        
        signal1_is_bullish = any(pattern in signal1.lower() for pattern in bullish_signals)
        signal2_is_bullish = any(pattern in signal2.lower() for pattern in bullish_signals)
        signal1_is_bearish = any(pattern in signal1.lower() for pattern in bearish_signals)
        signal2_is_bearish = any(pattern in signal2.lower() for pattern in bearish_signals)
        
        return (signal1_is_bullish and signal2_is_bullish) or (signal1_is_bearish and signal2_is_bearish)

    def _analyze_indicators(self, df: pd.DataFrame) -> Dict[str, str]:
        signals = {}
        
        with ThreadPoolExecutor(max_workers=min(len(self.indicators), 4)) as executor:
            futures = {executor.submit(ind.calculate, df): name for name, ind in self.indicators.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    if hasattr(result, 'interpretation') and result.interpretation:
                        signals[name] = result.interpretation
                except Exception as e:
                    logger.warning(f"Error calculating {name}: {e}")
        return signals

class SignalGenerator:
    def __init__(self, sentiment_fetcher=None, onchain_fetcher=None, lstm_model_manager: Optional[LSTMModelManager] = None, multi_tf_analyzer=None, config=None):
        self.indicators = {
            'sma_20': MovingAverageIndicator(20, "sma"),
            'sma_50': MovingAverageIndicator(50, "sma"),
            'ema_12': MovingAverageIndicator(12, "ema"),
            'ema_26': MovingAverageIndicator(26, "ema"),
            'rsi': RSIIndicator(),
            'macd': MACDIndicator(),
            'bb': BollingerBandsIndicator(),
            'stoch': StochasticIndicator(),
            'volume': VolumeIndicator(),
            'atr': ATRIndicator(),
            'ichimoku': IchimokuIndicator(),
            'williams_r': WilliamsRIndicator(),
            'cci': CCIIndicator(),
            'supertrend': SuperTrendIndicator(),
            'adx': ADXIndicator(),
            'cmf': ChaikinMoneyFlowIndicator(),
            'obv': OBVIndicator()
        }
        self.market_analyzer = MarketConditionAnalyzer()
        self.level_calculator = DynamicLevelCalculator()
        self.sentiment_fetcher = sentiment_fetcher
        self.onchain_fetcher = onchain_fetcher
        self.lstm_model_manager = lstm_model_manager
        self.multi_tf_analyzer = multi_tf_analyzer
        self.config = config or {'min_confidence_score': 60}
    
    def _safe_dataframe(self, df):
        if df is None or df.empty:
            return pd.DataFrame()
        
        try:
            df_copy = df.copy()
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df_copy.columns:
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            
            df_copy = df_copy.replace([np.inf, -np.inf], np.nan)
            
            initial_len = len(df_copy)
            df_copy = df_copy.dropna(subset=numeric_cols, how='any')
            
            if len(df_copy) < initial_len * 0.8:
                logger.warning(f"Lost {initial_len - len(df_copy)} rows due to invalid data")
            
            for col in ['open', 'high', 'low', 'close']:
                if col in df_copy.columns:
                    df_copy = df_copy[df_copy[col] > 0]
            
            if 'volume' in df_copy.columns:
                df_copy = df_copy[df_copy['volume'] >= 0]
            
            invalid_ohlc = (
                (df_copy['high'] < df_copy['low']) |
                (df_copy['high'] < df_copy['open']) |
                (df_copy['high'] < df_copy['close']) |
                (df_copy['low'] > df_copy['open']) |
                (df_copy['low'] > df_copy['close'])
            )
            
            if invalid_ohlc.any():
                df_copy = df_copy[~invalid_ohlc]
                logger.warning(f"Removed {invalid_ohlc.sum()} invalid OHLC candles")
            
            return df_copy.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error in _safe_dataframe: {e}")
            return pd.DataFrame()
    
    def _evaluate_buy_signal(self,
                            indicators: Dict[str, IndicatorResult],
                            data: pd.DataFrame,
                            symbol: str,
                            timeframe: str,
                            market_analysis: "MarketAnalysis",
                            patterns: List[str],
                            advanced_patterns: List[str],
                            vp: List[Tuple[float,float]],
                            vwap: float,
                            sentiment_fg: Optional[int],
                            sentiment_news: int,
                            onchain_active: Optional[int],
                            onchain_volume: Optional[int]
                            ) -> Optional[TradingSignal]:
        try:
            if data.empty or 'close' not in data.columns:
                logger.warning("Invalid or empty data for buy signal evaluation")
                return None
            
            score = 0
            reasons = []
            current_price = data['close'].iloc[-1]
            
            if 'rsi' in indicators and indicators['rsi'].interpretation == "oversold":
                score += 25
                reasons.append("RSI oversold condition")
                
            if 'macd' in indicators and indicators['macd'].interpretation == "bullish_crossover":
                score += 20
                reasons.append("MACD bullish crossover")
                
            if ('sma_20' in indicators and 'sma_50' in indicators and 
                not pd.isna(indicators['sma_20'].value) and not pd.isna(indicators['sma_50'].value) and
                indicators['sma_20'].value > indicators['sma_50'].value):
                score += 15
                reasons.append("Price above SMA trend")
                
            if 'bb' in indicators and indicators['bb'].interpretation == "near_lower_band":
                score += 15
                reasons.append("Price near Bollinger lower band")
                
            if 'stoch' in indicators and indicators['stoch'].interpretation == "oversold":
                score += 10
                reasons.append("Stochastic oversold")
                
            if 'volume' in indicators and indicators['volume'].interpretation == "high_volume":
                score += 10
                reasons.append("High volume confirmation")
                
            if market_analysis.trend == TrendDirection.BULLISH:
                score += 15
                reasons.append("Overall bullish trend")
                
            if market_analysis.trend_strength == TrendStrength.STRONG:
                score += 10
                reasons.append("Strong trend momentum")
                
            if market_analysis.volume_confirmation:
                score += 8
                reasons.append("Volume trend confirmation")
                
            if market_analysis.trend_acceleration > 0.5:
                score += 7
                reasons.append("Positive trend acceleration")
                
            if 'bullish_engulfing' in patterns:
                score += 15
                reasons.append("Bullish Engulfing pattern")
                
            if 'double_bottom' in patterns:
                score += 15
                reasons.append("Double Bottom pattern")
                
            if 'inverse_head_and_shoulders' in patterns:
                score += 15
                reasons.append("Inverse Head & Shoulders pattern")
                
            if 'flag' in advanced_patterns:
                score += 8
                reasons.append("Flag pattern")
                
            if 'triangle' in advanced_patterns or 'wedge' in advanced_patterns:
                score += 7
                reasons.append("Triangle/Wedge consolidation")
                
            if vwap and not pd.isna(vwap) and vwap > 0 and current_price < vwap:
                score += 5
                reasons.append("Price below VWAP - potential mean reversion")
                
            if sentiment_fg is not None:
                if sentiment_fg < 40:
                    score += 5
                    reasons.append("Fear & Greed indicates fear - favorable for buys")
                elif sentiment_fg > 70:
                    score -= 5
                    reasons.append("High greed - caution")
                    
            if sentiment_news > 0:
                score += 3
                reasons.append("Positive news sentiment")
                
            if onchain_active and onchain_active > 1000:
                score += 2
                reasons.append("Healthy on-chain activity")
                
            if score >= 80:
                try:
                    dynamic_levels = self.level_calculator.calculate_dynamic_levels(data, SignalType.BUY, market_analysis)
                    
                    return TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        entry_price=dynamic_levels.primary_entry,
                        exit_price=dynamic_levels.primary_exit,
                        stop_loss=dynamic_levels.tight_stop,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        confidence_score=score,
                        reasons=reasons,
                        risk_reward_ratio=self._calculate_risk_reward(
                            dynamic_levels.primary_entry, 
                            dynamic_levels.primary_exit, 
                            dynamic_levels.tight_stop
                        ),
                        predicted_profit=((dynamic_levels.primary_exit - dynamic_levels.primary_entry) / dynamic_levels.primary_entry) * 100,
                        volume_analysis=self.market_analyzer.volume_analyzer.analyze_volume_pattern(data),
                        market_context=self._create_market_context(market_analysis),
                        dynamic_levels={
                            'primary_entry': dynamic_levels.primary_entry,
                            'secondary_entry': dynamic_levels.secondary_entry,
                            'primary_exit': dynamic_levels.primary_exit,
                            'secondary_exit': dynamic_levels.secondary_exit,
                            'tight_stop': dynamic_levels.tight_stop,
                            'wide_stop': dynamic_levels.wide_stop,
                            'breakeven_point': dynamic_levels.breakeven_point,
                            'trailing_stop': dynamic_levels.trailing_stop
                        }
                    )
                except Exception as e:
                    logger.error(f"Error calculating dynamic levels: {e}")
                    return None

            return None
        except Exception as e:
            logger.error(f"Error evaluating buy signal: {e}")
            return None
        
    def _evaluate_sell_signal(self,
                            indicators: Dict[str, IndicatorResult],
                            data: pd.DataFrame,
                            symbol: str,
                            timeframe: str,
                            market_analysis: "MarketAnalysis",
                            patterns: List[str],
                            advanced_patterns: List[str],
                            vp: List[Tuple[float,float]],
                            vwap: float,
                            sentiment_fg: Optional[int],
                            sentiment_news: int,
                            onchain_active: Optional[int],
                            onchain_volume: Optional[int]
                            ) -> Optional[TradingSignal]:
        try:
            if data.empty or 'close' not in data.columns:
                logger.warning("Invalid or empty data for sell signal evaluation")
                return None
            
            score = 0
            reasons = []
            current_price = data['close'].iloc[-1]
            
            if 'rsi' in indicators and indicators['rsi'].interpretation == "overbought":
                score += 25
                reasons.append("RSI overbought condition")
                
            if 'macd' in indicators and indicators['macd'].interpretation == "bearish_crossover":
                score += 20
                reasons.append("MACD bearish crossover")
                
            if ('sma_20' in indicators and 'sma_50' in indicators and 
                not pd.isna(indicators['sma_20'].value) and not pd.isna(indicators['sma_50'].value) and
                indicators['sma_20'].value < indicators['sma_50'].value):
                score += 15
                reasons.append("Price below SMA trend")
                
            if 'bb' in indicators and indicators['bb'].interpretation == "near_upper_band":
                score += 15
                reasons.append("Price near Bollinger upper band")
                
            if 'stoch' in indicators and indicators['stoch'].interpretation == "overbought":
                score += 10
                reasons.append("Stochastic overbought")
                
            if 'volume' in indicators and indicators['volume'].interpretation == "high_volume":
                score += 10
                reasons.append("High volume confirmation")
                
            if market_analysis.trend == TrendDirection.BEARISH:
                score += 15
                reasons.append("Overall bearish trend")
                
            if market_analysis.trend_strength == TrendStrength.STRONG:
                score += 10
                reasons.append("Strong trend momentum")
                
            if market_analysis.volume_confirmation:
                score += 8
                reasons.append("Volume trend confirmation")
                
            if market_analysis.trend_acceleration < -0.5:
                score += 7
                reasons.append("Negative trend acceleration")
                
            if 'bearish_engulfing' in patterns:
                score += 15
                reasons.append("Bearish Engulfing pattern")
                
            if 'double_top' in patterns:
                score += 15
                reasons.append("Double Top pattern")
                
            if 'head_and_shoulders' in patterns:
                score += 15
                reasons.append("Head & Shoulders pattern")
                
            if 'flag' in advanced_patterns:
                score += 6
                reasons.append("Flag breakdown")
                
            if vwap and not pd.isna(vwap) and vwap > 0 and current_price > vwap:
                score += 5
                reasons.append("Price above VWAP - potential mean reversion")
                
            if sentiment_fg is not None:
                if sentiment_fg > 70:
                    score += 5
                    reasons.append("Greed indicated - favorable for sells")
                elif sentiment_fg < 30:
                    score -= 5
                    reasons.append("Extreme fear - caution on sells")
                    
            if sentiment_news < 0:
                score += 3
                reasons.append("Negative news sentiment")
                
            if onchain_active and onchain_active < 500:
                score += 2
                reasons.append("Low on-chain activity - weakness")
                
            if score >= 80:
                try:
                    dynamic_levels = self.level_calculator.calculate_dynamic_levels(data, SignalType.SELL, market_analysis)
                    
                    return TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        entry_price=dynamic_levels.primary_entry,
                        exit_price=dynamic_levels.primary_exit,
                        stop_loss=dynamic_levels.tight_stop,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        confidence_score=score,
                        reasons=reasons,
                        risk_reward_ratio=self._calculate_risk_reward(
                            dynamic_levels.primary_entry, 
                            dynamic_levels.primary_exit, 
                            dynamic_levels.tight_stop
                        ),
                        predicted_profit=((dynamic_levels.primary_entry - dynamic_levels.primary_exit) / dynamic_levels.primary_entry) * 100,
                        volume_analysis=self.market_analyzer.volume_analyzer.analyze_volume_pattern(data),
                        market_context=self._create_market_context(market_analysis),
                        dynamic_levels={
                            'primary_entry': dynamic_levels.primary_entry,
                            'secondary_entry': dynamic_levels.secondary_entry,
                            'primary_exit': dynamic_levels.primary_exit,
                            'secondary_exit': dynamic_levels.secondary_exit,
                            'tight_stop': dynamic_levels.tight_stop,
                            'wide_stop': dynamic_levels.wide_stop,
                            'breakeven_point': dynamic_levels.breakeven_point,
                            'trailing_stop': dynamic_levels.trailing_stop
                        }
                    )
                except Exception as e:
                    logger.error(f"Error calculating dynamic levels: {e}")
                    return None
                    
            return None
        except Exception as e:
            logger.error(f"Error evaluating sell signal: {e}")
            return None
            
    def _calculate_risk_reward(self, entry: float, exit: float, stop_loss: float) -> float:
        if pd.isna(entry) or pd.isna(exit) or pd.isna(stop_loss) or entry == stop_loss:
            return 0
            
        potential_profit = abs(exit - entry)
        potential_loss = abs(entry - stop_loss)
        return potential_profit / potential_loss if potential_loss > 0 else 0
        
    def _create_market_context(self, market_analysis: "MarketAnalysis") -> Dict[str, Any]:
        return {
            'trend': market_analysis.trend.value,
            'trend_strength': market_analysis.trend_strength.value,
            'volatility': market_analysis.volatility,
            'momentum_score': market_analysis.momentum_score,
            'market_condition': market_analysis.market_condition.value,
            'volume_trend': market_analysis.volume_trend,
            'trend_acceleration': market_analysis.trend_acceleration,
            'volume_confirmation': market_analysis.volume_confirmation
        }
        
    def optimize_params(self, train: pd.DataFrame) -> Dict[str, Any]:
        return {}
        
    def test_params(self, test: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    async def generate_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[TradingSignal]:
        if not symbol or not isinstance(symbol, str) or not symbol.strip():
            logger.error("Invalid symbol provided")
            return []
            
        if not timeframe or not isinstance(timeframe, str) or not timeframe.strip():
            logger.error("Invalid timeframe provided")
            return []
            
        logger.info(f"🔄 Generating signals for {symbol} on {timeframe}")
        
        try:
            if data is None:
                logger.warning(f"⚠️ No data provided for {symbol} on {timeframe}")
                return []
                
            if not isinstance(data, pd.DataFrame):
                logger.error(f"⚠️ Invalid data type for {symbol}: {type(data)}")
                return []
            
            if data.empty:
                logger.warning(f"⚠️ Empty dataframe for {symbol} on {timeframe}")
                return []
            
            data = self._safe_dataframe(data)
            
            if data.empty:
                logger.warning(f"⚠️ No valid data after cleaning for {symbol} on {timeframe}")
                return []
            
            if len(data) < 100:
                logger.warning(f"⚠️ Insufficient data for {symbol} on {timeframe}: {len(data)} candles (need 100+)")
                return []
        
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"❌ Missing required columns for {symbol}: {missing_columns}")
                return []
            
            if 'timestamp' in data.columns:
                try:
                    data = data.sort_values('timestamp').reset_index(drop=True)
                except Exception as e:
                    logger.warning(f"Could not sort by timestamp for {symbol}: {e}")
            
        except Exception as e:
            logger.error(f"❌ Data validation failed for {symbol}: {e}")
            return []
        
        indicator_results = {}
        failed_indicators = []
        critical_indicators = ['rsi', 'macd', 'volume', 'sma_20', 'bb']
        
        for name, indicator in self.indicators.items():
            try:
                if not hasattr(indicator, 'calculate') or not callable(indicator.calculate):
                    logger.warning(f"Invalid indicator {name} - no calculate method")
                    failed_indicators.append(name)
                    continue
                    
                data_copy = data.copy()
                result = indicator.calculate(data_copy)
                
                if result is None:
                    logger.warning(f"Indicator {name} returned None for {symbol}")
                    failed_indicators.append(name)
                    continue
                
                if not hasattr(result, 'value'):
                    logger.warning(f"Indicator {name} result missing 'value' attribute for {symbol}")
                    failed_indicators.append(name)
                    continue
                    
                if not hasattr(result, 'signal_strength'):
                    logger.warning(f"Indicator {name} result missing 'signal_strength' attribute for {symbol}")
                    failed_indicators.append(name)
                    continue
                    
                if not hasattr(result, 'interpretation'):
                    logger.warning(f"Indicator {name} result missing 'interpretation' attribute for {symbol}")
                    failed_indicators.append(name)
                    continue
                
                if (result.value is not None and
                    not pd.isna(result.value) and 
                    not np.isinf(result.value) and
                    isinstance(result.interpretation, str) and
                    result.interpretation.strip()):
                    
                    indicator_results[name] = result
                    logger.debug(f"✅ {name}: {result.value:.4f} ({result.interpretation})")
                else:
                    logger.warning(f"⚠️ Invalid result from {name} for {symbol}: value={getattr(result, 'value', None)}")
                    failed_indicators.append(name)
                    
            except Exception as e:
                logger.warning(f"⚠️ Error calculating {name} for {symbol}: {str(e)[:100]}")
                failed_indicators.append(name)
                continue
        
        missing_critical = [ind for ind in critical_indicators if ind not in indicator_results]
        if missing_critical:
            logger.warning(f"⚠️ Critical indicators failed for {symbol}: {', '.join(missing_critical)}")
            return []
        
        success_rate = len(indicator_results) / len(self.indicators)
        if success_rate < 0.6:
            logger.warning(f"⚠️ Too many indicators failed for {symbol}: {len(failed_indicators)}/{len(self.indicators)} failed")
            return []

        if failed_indicators:
            logger.info(f"ℹ️ Some indicators failed for {symbol}: {', '.join(failed_indicators)}")
        
        try:
            if not hasattr(self, 'market_analyzer') or self.market_analyzer is None:
                logger.error(f"Market analyzer not available for {symbol}")
                return []
                
            market_analysis = self.market_analyzer.analyze_market_condition(data)
            
            if market_analysis is None:
                logger.error(f"Market analysis returned None for {symbol}")
                return []
                
            if not hasattr(market_analysis, 'trend') or not hasattr(market_analysis, 'trend_strength'):
                logger.error(f"Invalid market analysis result for {symbol}")
                return []
                
            logger.debug(f"📊 Market analysis for {symbol}: trend={market_analysis.trend.value}, strength={market_analysis.trend_strength.value}")
        except Exception as e:
            logger.error(f"❌ Market analysis failed for {symbol}: {e}")
            return []
        
        patterns = []
        advanced_patterns = []
        
        try:
            if hasattr(PatternAnalyzer, 'detect_patterns') and callable(PatternAnalyzer.detect_patterns):
                patterns = PatternAnalyzer.detect_patterns(data)
                if not isinstance(patterns, list):
                    patterns = []
            
            if hasattr(PatternAnalyzer, 'detect_flag') and callable(PatternAnalyzer.detect_flag):
                if PatternAnalyzer.detect_flag(data):
                    advanced_patterns.append("flag")
                    
            if hasattr(PatternAnalyzer, 'detect_wedge') and callable(PatternAnalyzer.detect_wedge):
                if PatternAnalyzer.detect_wedge(data):
                    advanced_patterns.append("wedge")
                    
            if hasattr(PatternAnalyzer, 'detect_triangle') and callable(PatternAnalyzer.detect_triangle):
                if PatternAnalyzer.detect_triangle(data):
                    advanced_patterns.append("triangle")
                
            if patterns or advanced_patterns:
                logger.debug(f"🔍 Patterns detected for {symbol}: {patterns + advanced_patterns}")
                
        except Exception as e:
            logger.warning(f"⚠️ Pattern detection error for {symbol}: {e}")
            patterns = []
            advanced_patterns = []
        
        try:
            vp = []
            vwap = None
            
            if (hasattr(self.market_analyzer, 'volume_analyzer') and 
                self.market_analyzer.volume_analyzer is not None):
                
                if hasattr(self.market_analyzer.volume_analyzer, 'volume_profile'):
                    vp = self.market_analyzer.volume_analyzer.volume_profile(data)
                    if not isinstance(vp, list):
                        vp = []
                
                if hasattr(self.market_analyzer.volume_analyzer, 'vwap'):
                    vwap = self.market_analyzer.volume_analyzer.vwap(data)
            
            if vwap is None or pd.isna(vwap) or np.isinf(vwap) or vwap <= 0:
                if 'close' in data.columns and not data.empty:
                    vwap = data['close'].iloc[-1]
                    logger.warning(f"⚠️ Invalid VWAP for {symbol}, using last close price")
                else:
                    vwap = 0
                    logger.warning(f"⚠️ Could not calculate VWAP for {symbol}")
                
        except Exception as e:
            logger.warning(f"⚠️ Volume analysis error for {symbol}: {e}")
            vp = []
            vwap = data['close'].iloc[-1] if not data.empty and 'close' in data.columns else 0
        
        sentiment_fg = None
        sentiment_news_score = 0
        
        if (self.sentiment_fetcher and 
            hasattr(self.sentiment_fetcher, 'fetch_fear_greed') and
            callable(self.sentiment_fetcher.fetch_fear_greed)):
            try:
                sentiment_fg = await self.sentiment_fetcher.fetch_fear_greed()
                if sentiment_fg is not None and (not isinstance(sentiment_fg, (int, float)) or 
                                                sentiment_fg < 0 or sentiment_fg > 100):
                    sentiment_fg = None
                    
                if (hasattr(self.sentiment_fetcher, 'cryptopanic_key') and 
                    self.sentiment_fetcher.cryptopanic_key and
                    hasattr(self.sentiment_fetcher, 'fetch_news') and
                    callable(self.sentiment_fetcher.fetch_news) and
                    hasattr(self.sentiment_fetcher, 'score_news') and
                    callable(self.sentiment_fetcher.score_news)):
                    
                    try:
                        base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
                        news = await self.sentiment_fetcher.fetch_news([base_symbol])
                        if isinstance(news, list):
                            sentiment_news_score = self.sentiment_fetcher.score_news(news)
                            if not isinstance(sentiment_news_score, (int, float)):
                                sentiment_news_score = 0
                    except Exception as e:
                        logger.warning(f"News sentiment error for {symbol}: {e}")
                        sentiment_news_score = 0
                        
            except Exception as e:
                logger.warning(f"⚠️ Sentiment analysis error for {symbol}: {e}")
        
        onchain_active = None
        onchain_volume = None
        
        if (self.onchain_fetcher and 
            hasattr(self.onchain_fetcher, 'web3') and 
            self.onchain_fetcher.web3):
            try:
                if (hasattr(self.onchain_fetcher, 'active_addresses') and
                    callable(self.onchain_fetcher.active_addresses)):
                    onchain_active = await self.onchain_fetcher.active_addresses()
                    if onchain_active is not None and (not isinstance(onchain_active, (int, float)) or 
                                                        onchain_active < 0):
                        onchain_active = None
                        
                if (hasattr(self.onchain_fetcher, 'transaction_volume') and
                    callable(self.onchain_fetcher.transaction_volume)):
                    onchain_volume = await self.onchain_fetcher.transaction_volume()
                    if onchain_volume is not None and (not isinstance(onchain_volume, (int, float)) or 
                                                        onchain_volume < 0):
                        onchain_volume = None
                        
            except Exception as e:
                logger.warning(f"⚠️ On-chain analysis error: {e}")
        
        signals = []
        
        try:
            buy_signal = self._evaluate_buy_signal(
                indicator_results, data, symbol, timeframe, market_analysis, 
                patterns, advanced_patterns, vp, vwap, sentiment_fg, 
                sentiment_news_score, onchain_active, onchain_volume
            )
            
            if (buy_signal and 
                hasattr(buy_signal, 'confidence_score') and
                isinstance(buy_signal.confidence_score, (int, float)) and
                buy_signal.confidence_score >= self.config.get('min_confidence_score', 60)):
                
                if self.multi_tf_analyzer and hasattr(self.multi_tf_analyzer, 'is_direction_aligned'):
                    try:
                        is_aligned = await self.multi_tf_analyzer.is_direction_aligned(symbol, timeframe)
                        if is_aligned:
                            buy_signal.confidence_score += 5
                            if hasattr(buy_signal, 'reasons') and isinstance(buy_signal.reasons, list):
                                buy_signal.reasons.append("Multi-timeframe confirmation")
                            logger.info(f"🟢 BUY signal for {symbol} on {timeframe} - Confidence: {buy_signal.confidence_score:.0f}")
                            signals.append(buy_signal)
                        else:
                            logger.debug(f"❌ BUY signal rejected for {symbol} - No multi-timeframe alignment")
                    except Exception as e:
                        logger.warning(f"⚠️ Multi-timeframe check failed for {symbol}: {e}")
                        signals.append(buy_signal)
                else:
                    signals.append(buy_signal)
                    
        except Exception as e:
            logger.error(f"❌ Buy signal evaluation failed for {symbol}: {e}")
        
        try:
            sell_signal = self._evaluate_sell_signal(
                indicator_results, data, symbol, timeframe, market_analysis, 
                patterns, advanced_patterns, vp, vwap, sentiment_fg, 
                sentiment_news_score, onchain_active, onchain_volume
            )
            
            if (sell_signal and 
                hasattr(sell_signal, 'confidence_score') and
                isinstance(sell_signal.confidence_score, (int, float)) and
                sell_signal.confidence_score >= self.config.get('min_confidence_score', 60)):
                
                if self.multi_tf_analyzer and hasattr(self.multi_tf_analyzer, 'is_direction_aligned'):
                    try:
                        is_aligned = await self.multi_tf_analyzer.is_direction_aligned(symbol, timeframe)
                        if is_aligned:
                            sell_signal.confidence_score += 5
                            if hasattr(sell_signal, 'reasons') and isinstance(sell_signal.reasons, list):
                                sell_signal.reasons.append("Multi-timeframe confirmation")
                            logger.info(f"🔴 SELL signal for {symbol} on {timeframe} - Confidence: {sell_signal.confidence_score:.0f}")
                            signals.append(sell_signal)
                        else:
                            logger.debug(f"❌ SELL signal rejected for {symbol} - No multi-timeframe alignment")
                    except Exception as e:
                        logger.warning(f"⚠️ Multi-timeframe check failed for {symbol}: {e}")
                        signals.append(sell_signal)
                else:
                    signals.append(sell_signal)
                    
        except Exception as e:
            logger.error(f"❌ Sell signal evaluation failed for {symbol}: {e}")
        
        lstm_prediction = None
        if self.lstm_model_manager:
            try:
                prediction = await self.lstm_model_manager.predict_async(symbol, timeframe, data)
                if (prediction is not None and 
                    len(prediction) > 0 and 
                    not np.isnan(prediction[0]) and
                    not np.isinf(prediction[0]) and
                    prediction[0] > 0):
                    lstm_prediction = prediction[0]
                    logger.debug(f"🧠 LSTM prediction for {symbol}-{timeframe}: {lstm_prediction:.4f}")
                else:
                    logger.debug(f"ℹ️ LSTM model for {symbol}-{timeframe} not ready or returned invalid prediction.")
                    
            except Exception as e:
                logger.error(f"❌ LSTM prediction error for {symbol}-{timeframe}: {e}")
        
        if lstm_prediction is not None and signals and 'close' in data.columns and not data.empty:
            try:
                current_price = data['close'].iloc[-1]
                if (isinstance(current_price, (int, float)) and 
                    current_price > 0 and 
                    not pd.isna(current_price) and
                    not np.isinf(current_price)):
                    
                    price_change_percent = ((lstm_prediction - current_price) / current_price) * 100
                    
                    if abs(price_change_percent) > 0.5:
                        for signal in signals:
                            if (hasattr(signal, 'signal_type') and 
                                hasattr(signal, 'confidence_score') and
                                hasattr(signal, 'reasons') and
                                isinstance(signal.reasons, list)):
                                
                                lstm_boost = min(abs(price_change_percent) * 2, 15)
                                
                                if (hasattr(signal.signal_type, 'value') or 
                                    hasattr(signal.signal_type, 'name')):
                                    
                                    signal_type_str = (getattr(signal.signal_type, 'value', None) or 
                                                    getattr(signal.signal_type, 'name', str(signal.signal_type)))
                                    
                                    if signal_type_str.lower() == 'buy' and price_change_percent > 0:
                                        signal.confidence_score += lstm_boost
                                        signal.reasons.append(f"LSTM predicts {price_change_percent:.2f}% price increase")
                                        
                                    elif signal_type_str.lower() == 'sell' and price_change_percent < 0:
                                        signal.confidence_score += lstm_boost
                                        signal.reasons.append(f"LSTM predicts {abs(price_change_percent):.2f}% price decrease")
                                    
                                    signal.confidence_score = min(signal.confidence_score, 100)
                        
            except Exception as e:
                logger.error(f"❌ Error applying LSTM prediction to signals for {symbol}: {e}")
        
        final_signals = []
        for signal in signals:
            try:
                if (signal and
                    hasattr(signal, 'confidence_score') and
                    hasattr(signal, 'risk_reward_ratio') and
                    hasattr(signal, 'reasons') and
                    isinstance(signal.confidence_score, (int, float)) and
                    isinstance(signal.risk_reward_ratio, (int, float)) and
                    isinstance(signal.reasons, list)):
                    
                    if (signal.confidence_score >= 50 and
                        signal.risk_reward_ratio >= 1.0 and
                        len(signal.reasons) >= 3):
                        
                        final_signals.append(signal)
                        logger.debug(f"✅ Final signal approved for {symbol}: {signal.signal_type.value} (confidence: {signal.confidence_score:.0f})")
                    else:
                        logger.debug(f"❌ Signal rejected for {symbol}: confidence={signal.confidence_score:.0f}, rr={signal.risk_reward_ratio:.2f}, reasons={len(signal.reasons)}")
                else:
                    logger.warning(f"Invalid signal object for {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Error validating signal for {symbol}: {e}")
                continue
        
        if final_signals:
            logger.info(f"🎯 Generated {len(final_signals)} high-quality signal(s) for {symbol} on {timeframe}")
        else:
            logger.debug(f"ℹ️ No qualifying signals for {symbol} on {timeframe}")
        
        return final_signals

class SignalRanking:
    @staticmethod
    def rank_signals(signals: List[TradingSignal]) -> List[TradingSignal]:
        def signal_score(signal: TradingSignal) -> float:
            base_score = signal.confidence_score
            rr_bonus = min(signal.risk_reward_ratio * 10, 20)
            profit_bonus = min(abs(signal.predicted_profit) * 2, 15)
            volume_bonus = 0
            if signal.volume_analysis.get('volume_ratio', 1) > 1.5:
                volume_bonus = 10
            trend_bonus = 0
            trend_strength = signal.market_context.get('trend_strength', 'weak')
            if trend_strength == 'strong':
                trend_bonus = 15
            elif trend_strength == 'moderate':
                trend_bonus = 10
            acceleration_bonus = 0
            trend_acceleration = abs(signal.market_context.get('trend_acceleration', 0))
            if trend_acceleration > 1:
                acceleration_bonus = 8
            volume_confirmation_bonus = 0
            if signal.market_context.get('volume_confirmation', False):
                volume_confirmation_bonus = 5
            return base_score + rr_bonus + profit_bonus + volume_bonus + trend_bonus + acceleration_bonus + volume_confirmation_bonus
        return sorted(signals, key=signal_score, reverse=True)