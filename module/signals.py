import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

from module.analyzers import (MarketConditionAnalyzer,
                              PatternAnalyzer)
from module.constants import (MULTI_TF_CONFIRMATION_MAP,
                              MULTI_TF_CONFIRMATION_WEIGHTS)
from module.core import (DerivativesAnalysis, DynamicLevels, FundamentalAnalysis,
                         IndicatorResult, MarketAnalysis, OrderBookAnalysis,
                         SignalType, TradingSignal, TrendDirection,
                         TrendStrength)
from module.indicators import (ADXIndicator, ATRIndicator,
                               BollingerBandsIndicator, CCIIndicator,
                               ChaikinMoneyFlowIndicator, IchimokuIndicator,
                               MACDIndicator, MovingAverageIndicator,
                               OBVIndicator, ParabolicSARIndicator,
                               RSIIndicator, SqueezeMomentumIndicator,
                               StochasticIndicator, SuperTrendIndicator,
                               VolumeIndicator, WilliamsRIndicator,
                               VWAPIndicator, MoneyFlowIndexIndicator, AroonIndicator,
                               UltimateOscillatorIndicator, ROCIndicator, ADLineIndicator,
                               ForceIndexIndicator, VWMAIndicator, KeltnerChannelsIndicator,
                               DonchianChannelsIndicator, TRIXIndicator, EaseOfMovementIndicator,
                               StandardDeviationIndicator)
from module.logger_config import logger
from module.lstm import LSTMModelManager
from module.sentiment import (CoinGeckoFetcher, ExchangeManager,
                              MarketIndicesFetcher, NewsFetcher,
                              OnChainFetcher)


class MultiTimeframeAnalyzer:
    def __init__(self, exchange_manager, indicators, cache_ttl=300):
        self.exchange_manager = exchange_manager
        self.indicators = indicators
        self._cache = {}
        self._cache_expiry = {}
        self.cache_ttl = cache_ttl
        self._lock = asyncio.Lock()

    async def _get_cache(self, key):
        async with self._lock:
            if key in self._cache and time.time() < self._cache_expiry.get(key, 0):
                return self._cache[key]
        return None

    async def _set_cache(self, key, value):
        async with self._lock:
            self._cache[key] = value
            self._cache_expiry[key] = time.time() + self.cache_ttl

    async def is_direction_aligned(self, symbol: str, exec_tf: str, threshold: float = 0.6) -> bool:
        try:
            confirm_tfs = MULTI_TF_CONFIRMATION_MAP.get(exec_tf, [])
            if not confirm_tfs:
                return True
            
            base_key = (symbol, exec_tf)
            base_signals = await self._get_cache(base_key)
            if not base_signals:
                base_df = await self.exchange_manager.fetch_ohlcv_data(symbol, exec_tf)
                if base_df.empty or len(base_df) < 50:
                    return False
                base_signals = self._analyze_indicators(base_df)
                await self._set_cache(base_key, base_signals)

            if not base_signals:
                return False

            tasks = []
            for tf in confirm_tfs:
                tasks.append(self._get_confirmation_score(symbol, tf, base_signals, exec_tf))
            
            results = await asyncio.gather(*tasks)
            
            total_score = sum(r[0] for r in results)
            total_weight = sum(r[1] for r in results)

            if total_weight == 0:
                return False
            avg_score = total_score / total_weight
            return avg_score >= threshold
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis for {symbol}: {e}")
            return False

    async def _get_confirmation_score(self, symbol, tf, base_signals, exec_tf):
        try:
            confirm_key = (symbol, tf)
            confirm_signals = await self._get_cache(confirm_key)
            if not confirm_signals:
                confirm_df = await self.exchange_manager.fetch_ohlcv_data(symbol, tf)
                if confirm_df.empty or len(confirm_df) < 50:
                    return 0, 0
                confirm_signals = self._analyze_indicators(confirm_df)
                await self._set_cache(confirm_key, confirm_signals)

            if not confirm_signals:
                return 0, 0

            weight = MULTI_TF_CONFIRMATION_WEIGHTS.get(exec_tf, {}).get(tf, 1.0)
            matches = sum(1 for k in base_signals
                        if k in confirm_signals and self._signals_match(base_signals[k], confirm_signals[k]))
            score = matches / len(base_signals) if base_signals else 0
            return score * weight, weight
        except Exception as e:
            logger.warning(f"Error analyzing timeframe {tf} for {symbol}: {e}")
            return 0, 0


    def _signals_match(self, signal1: str, signal2: str) -> bool:
        bullish_signals = ['bullish', 'oversold', 'above', 'buy', 'up', 'accumulation', 'bull_power', 'upward']
        bearish_signals = ['bearish', 'overbought', 'below', 'sell', 'down', 'distribution', 'bear_power', 'downward']

        s1_lower = signal1.lower()
        s2_lower = signal2.lower()

        signal1_is_bullish = any(p in s1_lower for p in bullish_signals)
        signal2_is_bullish = any(p in s2_lower for p in bullish_signals)
        signal1_is_bearish = any(p in s1_lower for p in bearish_signals)
        signal2_is_bearish = any(p in s2_lower for p in bearish_signals)

        return (signal1_is_bullish and signal2_is_bullish) or (signal1_is_bearish and signal2_is_bearish)

    def _analyze_indicators(self, df: pd.DataFrame) -> Dict[str, str]:
        signals = {}
        if df.empty or not self.indicators:
            return signals

        for name, ind in self.indicators.items():
            try:
                result = ind.calculate(df)
                if hasattr(result, 'interpretation') and result.interpretation:
                    signals[name] = result.interpretation
            except Exception as e:
                logger.warning(f"Error calculating {name}: {e}")
        return signals

class SignalGenerator:
    def __init__(self, exchange_manager: ExchangeManager, news_fetcher: Optional[NewsFetcher] = None,
                 onchain_fetcher: Optional[OnChainFetcher] = None,
                 coingecko_fetcher: Optional[CoinGeckoFetcher] = None,
                 market_indices_fetcher: Optional[MarketIndicesFetcher] = None,
                 lstm_model_manager: Optional[LSTMModelManager] = None,
                 multi_tf_analyzer: Optional[MultiTimeframeAnalyzer] = None,
                 config: Optional[Dict] = None):
        self.exchange_manager = exchange_manager
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
            'obv': OBVIndicator(),
            'squeeze': SqueezeMomentumIndicator(),
            'psar': ParabolicSARIndicator(),
            'vwap': VWAPIndicator(),
            'mfi': MoneyFlowIndexIndicator(),
            'aroon': AroonIndicator(),
            'uo': UltimateOscillatorIndicator(),
            'roc': ROCIndicator(),
            'ad_line': ADLineIndicator(),
            'force_index': ForceIndexIndicator(),
            'vwma': VWMAIndicator(),
            'keltner': KeltnerChannelsIndicator(),
            'donchian': DonchianChannelsIndicator(),
            'trix': TRIXIndicator(),
            'eom': EaseOfMovementIndicator(),
            'std_dev': StandardDeviationIndicator()
        }
        self.market_analyzer = MarketConditionAnalyzer()
        self.news_fetcher = news_fetcher
        self.onchain_fetcher = onchain_fetcher
        self.coingecko_fetcher = coingecko_fetcher
        self.market_indices_fetcher = market_indices_fetcher
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
            df_copy = df_copy.dropna(subset=numeric_cols, how='any')

            for col in ['open', 'high', 'low', 'close']:
                if col in df_copy.columns:
                    df_copy = df_copy[df_copy[col] > 0]

            if 'volume' in df_copy.columns:
                df_copy = df_copy[df_copy['volume'] >= 0]

            invalid_ohlc = (df_copy['high'] < df_copy['low'])
            if invalid_ohlc.any():
                df_copy = df_copy[~invalid_ohlc]

            return df_copy.reset_index(drop=True)

        except Exception as e:
            logger.error(f"Error in _safe_dataframe: {e}")
            return pd.DataFrame()

    def calculate_dynamic_levels(self, data: pd.DataFrame, signal_type: SignalType, market_analysis: MarketAnalysis) -> DynamicLevels:
        last_close = data['close'].iloc[-1]
        atr = market_analysis.volatility / 100 * last_close if market_analysis.volatility > 0 else data['close'].pct_change().std() * last_close
        if atr == 0 or pd.isna(atr):
            atr = last_close * 0.01

        if signal_type == SignalType.BUY:
            primary_entry = last_close
            secondary_entry = last_close - 0.5 * atr
            primary_exit = last_close + 2 * atr
            secondary_exit = last_close + 3 * atr
            tight_stop = primary_entry - 1 * atr
            wide_stop = primary_entry - 1.5 * atr
        else: # SELL
            primary_entry = last_close
            secondary_entry = last_close + 0.5 * atr
            primary_exit = last_close - 2 * atr
            secondary_exit = last_close - 3 * atr
            tight_stop = primary_entry + 1 * atr
            wide_stop = primary_entry + 1.5 * atr

        return DynamicLevels(
            primary_entry=primary_entry,
            secondary_entry=secondary_entry,
            primary_exit=primary_exit,
            secondary_exit=secondary_exit,
            tight_stop=tight_stop,
            wide_stop=wide_stop,
            breakeven_point=primary_entry + 0.2 * atr if signal_type == SignalType.BUY else primary_entry - 0.2 * atr,
            trailing_stop=atr * 0.5
        )

    def _evaluate_buy_signal(self,
                            indicators: Dict[str, IndicatorResult],
                            market_analysis: MarketAnalysis,
                            patterns: List[str],
                            sentiment_data: Dict[str, Any],
                            data: pd.DataFrame,
                            symbol: str,
                            timeframe: str
                            ) -> Optional[TradingSignal]:
        try:
            if data.empty or 'close' not in data.columns:
                return None

            score = 0
            reasons = []

            rsi_res = indicators.get('rsi')
            if rsi_res and rsi_res.interpretation == "oversold":
                score += 20; reasons.append(f"RSI oversold ({rsi_res.value:.2f})")
            
            macd_res = indicators.get('macd')
            if macd_res and macd_res.interpretation == "bullish_crossover":
                score += 15; reasons.append("MACD bullish crossover")

            bb_res = indicators.get('bb')
            if bb_res and bb_res.interpretation == "near_lower_band":
                score += 10; reasons.append("Price near Bollinger Lower Band")
            
            stoch_res = indicators.get('stoch')
            if stoch_res and stoch_res.interpretation == "oversold":
                score += 10; reasons.append(f"Stochastic oversold ({stoch_res.value:.2f})")

            mfi_res = indicators.get('mfi')
            if mfi_res and mfi_res.interpretation == "oversold":
                score += 15; reasons.append(f"MFI oversold ({mfi_res.value:.2f})")
            
            aroon_res = indicators.get('aroon')
            if aroon_res and 'uptrend' in aroon_res.interpretation:
                score += 5; reasons.append("Aroon indicates uptrend")
            
            vwap_res = indicators.get('vwap')
            if vwap_res and 'above' in vwap_res.interpretation:
                score += 5; reasons.append("Price above VWAP")
            
            bullish_patterns = [p for p in patterns if 'bullish' in p or 'Up' in p or 'Morning' in p or 'Hammer' in p or 'Soldiers' in p]
            if bullish_patterns:
                score += 10; reasons.append(f"Bullish pattern: {bullish_patterns[0]}")

            if 'bullish_divergence' in patterns:
                score += 15; reasons.append("Bullish divergence detected")
            if market_analysis.trend == TrendDirection.BULLISH:
                score += 10
                if market_analysis.trend_strength == TrendStrength.STRONG: score += 5
                reasons.append(f"Aligned with market trend ({market_analysis.trend.value} - {market_analysis.trend_strength.value})")

            btc_d = sentiment_data.get('BTC.D')
            if btc_d is not None and btc_d < 45:
                score += 5; reasons.append("Altcoin season (low BTC.D)")

            dxy = sentiment_data.get('DXY')
            if dxy is not None and dxy < 100:
                score += 5; reasons.append("Weak US Dollar (DXY)")

            fear_greed = sentiment_data.get('FEAR.GREED')
            if fear_greed is not None and fear_greed < 30:
                score += 10; reasons.append("Market in Extreme Fear")

            if score >= self.config.get('min_confidence_score', 60):
                dynamic_levels = self.calculate_dynamic_levels(data, SignalType.BUY, market_analysis)
                return self._create_trading_signal(symbol, timeframe, SignalType.BUY, score, reasons, dynamic_levels, data, market_analysis)

            return None
        except Exception as e:
            logger.error(f"Error evaluating buy signal for {symbol}: {e}")
            return None

    def _evaluate_sell_signal(self,
                             indicators: Dict[str, IndicatorResult],
                             market_analysis: MarketAnalysis,
                             patterns: List[str],
                             sentiment_data: Dict[str, Any],
                             data: pd.DataFrame,
                             symbol: str,
                             timeframe: str
                             ) -> Optional[TradingSignal]:
        try:
            if data.empty or 'close' not in data.columns:
                return None

            score = 0
            reasons = []

            rsi_res = indicators.get('rsi')
            if rsi_res and rsi_res.interpretation == "overbought":
                score += 20; reasons.append(f"RSI overbought ({rsi_res.value:.2f})")
            
            macd_res = indicators.get('macd')
            if macd_res and macd_res.interpretation == "bearish_crossover":
                score += 15; reasons.append("MACD bearish crossover")

            bb_res = indicators.get('bb')
            if bb_res and bb_res.interpretation == "near_upper_band":
                score += 10; reasons.append("Price near Bollinger Upper Band")

            stoch_res = indicators.get('stoch')
            if stoch_res and stoch_res.interpretation == "overbought":
                score += 10; reasons.append(f"Stochastic overbought ({stoch_res.value:.2f})")
            
            mfi_res = indicators.get('mfi')
            if mfi_res and mfi_res.interpretation == "overbought":
                score += 15; reasons.append(f"MFI overbought ({mfi_res.value:.2f})")
            
            aroon_res = indicators.get('aroon')
            if aroon_res and 'downtrend' in aroon_res.interpretation:
                score += 5; reasons.append("Aroon indicates downtrend")
            
            vwap_res = indicators.get('vwap')
            if vwap_res and 'below' in vwap_res.interpretation:
                score += 5; reasons.append("Price below VWAP")

            bearish_patterns = [p for p in patterns if 'bearish' in p or 'Down' in p or 'Evening' in p or 'Hanging' in p or 'Crows' in p]
            if bearish_patterns:
                score += 10; reasons.append(f"Bearish pattern: {bearish_patterns[0]}")

            if 'bearish_divergence' in patterns:
                score += 15; reasons.append("Bearish divergence detected")
            if market_analysis.trend == TrendDirection.BEARISH:
                score += 10
                if market_analysis.trend_strength == TrendStrength.STRONG: score += 5
                reasons.append(f"Aligned with market trend ({market_analysis.trend.value} - {market_analysis.trend_strength.value})")

            btc_d = sentiment_data.get('BTC.D')
            if btc_d is not None and btc_d > 55:
                score += 5; reasons.append("Bitcoin dominance is high")

            dxy = sentiment_data.get('DXY')
            if dxy is not None and dxy > 105:
                score += 5; reasons.append("Strong US Dollar (DXY)")

            fear_greed = sentiment_data.get('FEAR.GREED')
            if fear_greed is not None and fear_greed > 75:
                score += 10; reasons.append("Market in Extreme Greed")

            if score >= self.config.get('min_confidence_score', 60):
                dynamic_levels = self.calculate_dynamic_levels(data, SignalType.SELL, market_analysis)
                return self._create_trading_signal(symbol, timeframe, SignalType.SELL, score, reasons, dynamic_levels, data, market_analysis)

            return None
        except Exception as e:
            logger.error(f"Error evaluating sell signal for {symbol}: {e}")
            return None

    def _create_trading_signal(self, symbol, timeframe, signal_type, score, reasons, dynamic_levels, data, market_analysis):
        entry = dynamic_levels.primary_entry
        exit_price = dynamic_levels.primary_exit
        stop_loss = dynamic_levels.tight_stop

        rr = self._calculate_risk_reward(entry, exit_price, stop_loss)
        profit = ((exit_price - entry) / entry) * 100 if signal_type == SignalType.BUY else ((entry - exit_price) / entry) * 100

        market_context_dict = self._create_market_context(market_analysis)

        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            entry_price=entry,
            exit_price=exit_price,
            stop_loss=stop_loss,
            timestamp=datetime.now(),
            timeframe=timeframe,
            confidence_score=score,
            reasons=reasons,
            risk_reward_ratio=rr,
            predicted_profit=profit,
            volume_analysis=self.market_analyzer.volume_analyzer.analyze_volume_pattern(data),
            market_context=market_context_dict,
            dynamic_levels=vars(dynamic_levels),
            derivatives_analysis=market_analysis.derivatives_analysis,
            order_book_analysis=market_analysis.order_book_analysis,
            fundamental_analysis=market_analysis.fundamental_analysis
        )

    def _calculate_risk_reward(self, entry: float, exit: float, stop_loss: float) -> float:
        if pd.isna(entry) or pd.isna(exit) or pd.isna(stop_loss) or entry == stop_loss:
            return 0

        potential_profit = abs(exit - entry)
        potential_loss = abs(entry - stop_loss)
        return potential_profit / potential_loss if potential_loss > 0 else float('inf')

    def _create_market_context(self, market_analysis: "MarketAnalysis") -> Dict[str, Any]:
        return {
            'trend': market_analysis.trend.value if hasattr(market_analysis.trend, 'value') else market_analysis.trend,
            'trend_strength': market_analysis.trend_strength.value if hasattr(market_analysis.trend_strength, 'value') else market_analysis.trend_strength,
            'volatility': market_analysis.volatility,
            'momentum_score': market_analysis.momentum_score,
            'hurst_exponent': market_analysis.hurst_exponent,
        }

    def optimize_params(self, train: pd.DataFrame) -> Dict[str, Any]:
        return {}

    def test_params(self, test: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    async def generate_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[TradingSignal]:
        data = self._safe_dataframe(data)
        if data.empty or len(data) < 100:
            logger.warning(f"Insufficient data for {symbol} on {timeframe}")
            return []

        indicator_results = {name: indicator.calculate(data) for name, indicator in self.indicators.items()}

        derivatives_data, order_book_data, fundamental_data, sentiment_data = await self._gather_context_data(symbol)

        market_analysis = self.market_analyzer.analyze_market_condition(
            data,
            derivatives_data=derivatives_data,
            order_book_analysis=order_book_data,
            fundamental_analysis=fundamental_data
        )

        patterns = PatternAnalyzer.detect_patterns(data)
        rsi_series = ta.rsi(data['close'])
        if rsi_series is not None and not rsi_series.empty:
            patterns.extend(PatternAnalyzer.detect_divergence(data, rsi_series))

        signals = []
        buy_signal = self._evaluate_buy_signal(indicator_results, market_analysis, patterns, sentiment_data, data, symbol, timeframe)
        if buy_signal:
            signals.append(buy_signal)

        sell_signal = self._evaluate_sell_signal(indicator_results, market_analysis, patterns, sentiment_data, data, symbol, timeframe)
        if sell_signal:
            signals.append(sell_signal)

        if signals and self.multi_tf_analyzer:
            is_aligned = await self.multi_tf_analyzer.is_direction_aligned(symbol, timeframe)
            if not is_aligned:
                logger.info(f"Signal for {symbol} on {timeframe} rejected due to multi-timeframe misalignment.")
                return []
            else:
                for signal in signals:
                    signal.confidence_score = min(100, signal.confidence_score + 5)
                    signal.reasons.append("Multi-timeframe confirmation")
        
        if self.lstm_model_manager:
            prediction = await self.lstm_model_manager.predict_async(symbol, timeframe, data)
            if prediction is not None and len(prediction) > 0:
                pred_price = prediction[0]
                last_close = data['close'].iloc[-1]
                for signal in signals:
                    if signal.signal_type == SignalType.BUY and pred_price > last_close * 1.005:
                        signal.confidence_score = min(100, signal.confidence_score + 10)
                        signal.reasons.append(f"LSTM Prediction Bullish ({pred_price:.4f})")
                    elif signal.signal_type == SignalType.SELL and pred_price < last_close * 0.995:
                        signal.confidence_score = min(100, signal.confidence_score + 10)
                        signal.reasons.append(f"LSTM Prediction Bearish ({pred_price:.4f})")

        return signals

    async def _gather_context_data(self, symbol: str):
        tasks = {
            "derivatives": self.exchange_manager.fetch_derivatives_data(symbol),
            "order_book": self.exchange_manager.fetch_order_book(symbol),
            "fundamentals": self.coingecko_fetcher.get_fundamental_data(symbol.split('/')[0]) if self.coingecko_fetcher else asyncio.sleep(0, result=None),
            "indices": self.market_indices_fetcher.get_all_indices() if self.market_indices_fetcher else asyncio.sleep(0, result={}),
            "fear_greed": self.news_fetcher.fetch_fear_greed() if self.news_fetcher else asyncio.sleep(0, result=None),
            "hash_rate": self.onchain_fetcher.get_hash_rate() if self.onchain_fetcher else asyncio.sleep(0, result=None),
            "gas_fees": self.onchain_fetcher.get_eth_gas_fees() if self.onchain_fetcher else asyncio.sleep(0, result=None),
        }
        
        results = await asyncio.gather(*tasks.values())
        derivatives_data, order_book_data, fundamental_data, indices, fear_greed, hash_rate, gas_fees = results
        
        sentiment_data = indices or {}
        sentiment_data['FEAR.GREED'] = fear_greed
        sentiment_data['HASH_RATE'] = hash_rate
        sentiment_data['GAS_FEES'] = gas_fees
        
        return derivatives_data, order_book_data, fundamental_data, sentiment_data


class SignalRanking:
    @staticmethod
    def rank_signals(signals: List[TradingSignal]) -> List[TradingSignal]:
        def signal_score(signal: TradingSignal) -> float:
            base_score = signal.confidence_score
            rr_bonus = min(signal.risk_reward_ratio * 10, 20)
            profit_bonus = min(abs(signal.predicted_profit) * 2, 15)
            
            volume_bonus = 0
            if signal.volume_analysis and signal.volume_analysis.get('volume_ratio', 1) > 1.5:
                volume_bonus = 10
            
            trend_bonus = 0
            if signal.market_context:
                trend_strength = signal.market_context.get('trend_strength')
                if trend_strength == TrendStrength.STRONG.value:
                    trend_bonus = 15
                elif trend_strength == TrendStrength.MODERATE.value:
                    trend_bonus = 10
            
            return base_score + rr_bonus + profit_bonus + volume_bonus + trend_bonus
        return sorted(signals, key=signal_score, reverse=True)