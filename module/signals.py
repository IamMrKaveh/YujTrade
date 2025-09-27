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
from module.core import (DynamicLevels, FundamentalAnalysis,
                         IndicatorResult, MarketAnalysis,
                         SignalType, TradingSignal, TrendDirection,
                         TrendStrength)
from module.data_sources import MarketIndicesFetcher, NewsFetcher
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
from module.market import MarketDataProvider


class MultiTimeframeAnalyzer:
    def __init__(self, market_data_provider: MarketDataProvider, indicators, cache_ttl=300):
        self.market_data_provider = market_data_provider
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
                base_df = await self.market_data_provider.fetch_ohlcv_data(symbol, exec_tf)
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
                confirm_df = await self.market_data_provider.fetch_ohlcv_data(symbol, tf)
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

    INDICATOR_WEIGHTS = {
        'rsi': 15, 'macd': 12, 'stoch': 10, 'mfi': 12, 'cci': 8, 'williams_r': 8,
        'bb': 8, 'supertrend': 10, 'psar': 8, 'ichimoku': 10,
        'volume': 7, 'cmf': 7, 'obv': 5,
        'trend': 15, 'strength': 10, 'divergence': 15, 'pattern': 10,
        'fear_greed': 8, 'btc_dominance': 5, 'dxy': 5,
        'lstm': 15, 'multi_tf': 10
    }
    
    def __init__(self, market_data_provider: MarketDataProvider, news_fetcher: Optional[NewsFetcher] = None,
                 market_indices_fetcher: Optional[MarketIndicesFetcher] = None,
                 lstm_model_manager: Optional[LSTMModelManager] = None,
                 multi_tf_analyzer: Optional[MultiTimeframeAnalyzer] = None,
                 config: Optional[Dict] = None):
        self.market_data_provider = market_data_provider
        self.indicators = {
            'sma_20': MovingAverageIndicator(20, "sma"), 'sma_50': MovingAverageIndicator(50, "sma"),
            'ema_12': MovingAverageIndicator(12, "ema"), 'ema_26': MovingAverageIndicator(26, "ema"),
            'rsi': RSIIndicator(), 'macd': MACDIndicator(), 'bb': BollingerBandsIndicator(),
            'stoch': StochasticIndicator(), 'volume': VolumeIndicator(), 'atr': ATRIndicator(),
            'ichimoku': IchimokuIndicator(), 'williams_r': WilliamsRIndicator(), 'cci': CCIIndicator(),
            'supertrend': SuperTrendIndicator(), 'adx': ADXIndicator(), 'cmf': ChaikinMoneyFlowIndicator(),
            'obv': OBVIndicator(), 'squeeze': SqueezeMomentumIndicator(), 'psar': ParabolicSARIndicator(),
            'vwap': VWAPIndicator(), 'mfi': MoneyFlowIndexIndicator(), 'aroon': AroonIndicator(),
            'uo': UltimateOscillatorIndicator(), 'roc': ROCIndicator(), 'ad_line': ADLineIndicator(),
            'force_index': ForceIndexIndicator(), 'vwma': VWMAIndicator(), 'keltner': KeltnerChannelsIndicator(),
            'donchian': DonchianChannelsIndicator(), 'trix': TRIXIndicator(), 'eom': EaseOfMovementIndicator(),
            'std_dev': StandardDeviationIndicator()
        }
        self.market_analyzer = MarketConditionAnalyzer()
        self.news_fetcher = news_fetcher
        self.market_indices_fetcher = market_indices_fetcher
        self.lstm_model_manager = lstm_model_manager
        self.multi_tf_analyzer = multi_tf_analyzer
        self.config = config or {'min_confidence_score': 75}

    def _safe_dataframe(self, df):
        if df is None or df.empty: return pd.DataFrame()
        try:
            df_copy = df.copy()
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df_copy.columns: df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            df_copy = df_copy.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_cols, how='any')
            for col in ['open', 'high', 'low', 'close']:
                if col in df_copy.columns: df_copy = df_copy[df_copy[col] > 0]
            if 'volume' in df_copy.columns: df_copy = df_copy[df_copy['volume'] >= 0]
            if 'high' in df_copy.columns and 'low' in df_copy.columns and (df_copy['high'] < df_copy['low']).any():
                df_copy = df_copy[~(df_copy['high'] < df_copy['low'])]
            if not isinstance(df_copy.index, pd.DatetimeIndex): df_copy = df_copy.reset_index(drop=True)
            return df_copy
        except Exception as e:
            logger.error(f"Error in _safe_dataframe: {e}")
            return pd.DataFrame()

    def calculate_dynamic_levels(self, data: pd.DataFrame, market_analysis: MarketAnalysis, signal_type: SignalType) -> DynamicLevels:
        last_close = data['close'].iloc[-1]
        atr = market_analysis.volatility / 100 * last_close if market_analysis.volatility > 0 else data['close'].pct_change().std() * last_close
        if pd.isna(atr) or atr == 0: atr = last_close * 0.01

        if signal_type == SignalType.BUY:
            primary_entry, secondary_entry = last_close, last_close - 0.5 * atr
            primary_exit, secondary_exit = last_close + 2 * atr, last_close + 3.5 * atr
            tight_stop, wide_stop = primary_entry - 1.2 * atr, primary_entry - 2.0 * atr
        else:
            primary_entry, secondary_entry = last_close, last_close + 0.5 * atr
            primary_exit, secondary_exit = last_close - 2 * atr, last_close - 3.5 * atr
            tight_stop, wide_stop = primary_entry + 1.2 * atr, primary_entry + 2.0 * atr

        return DynamicLevels(
            primary_entry=primary_entry, secondary_entry=secondary_entry,
            primary_exit=primary_exit, secondary_exit=secondary_exit,
            tight_stop=tight_stop, wide_stop=wide_stop,
            breakeven_point=primary_entry + 0.2 * atr if signal_type == SignalType.BUY else primary_entry - 0.2 * atr,
            trailing_stop=atr * 0.7
        )

    def _calculate_confidence_score(self, signal_type: SignalType, indicators: Dict[str, IndicatorResult], market_analysis: MarketAnalysis, patterns: List, sentiment: Dict) -> Tuple[float, List[str]]:
        score, total_weight = 0.0, 0.0
        reasons = []

        is_buy = signal_type == SignalType.BUY

        def add_reason(source: str, interpretation: str, points: float, strength: float = -1):
            strength_str = f", strength: {strength:.0f}%" if strength >= 0 else ""
            reasons.append(f"{source} ({interpretation}{strength_str}) -> {points:+.1f} pts")

        for name, res in indicators.items():
            weight = self.INDICATOR_WEIGHTS.get(name.split('_')[0].lower(), 0)
            if weight == 0: continue

            total_weight += weight
            points = 0
            
            is_bullish = any(s in res.interpretation for s in ['bullish', 'oversold', 'above', 'buy', 'up', 'accumulation', 'bull_power', 'upward'])
            is_bearish = any(s in res.interpretation for s in ['bearish', 'overbought', 'below', 'sell', 'down', 'distribution', 'bear_power', 'downward'])

            if (is_buy and is_bullish) or (not is_buy and is_bearish):
                points = (res.signal_strength / 100) * weight
            elif (is_buy and is_bearish) or (not is_buy and is_bullish):
                points = - (res.signal_strength / 100) * weight * 1.2
            
            if points != 0:
                score += points
                add_reason(res.name, res.interpretation, points, res.signal_strength)

        trend_weight = self.INDICATOR_WEIGHTS.get('trend', 15)
        total_weight += trend_weight
        trend_aligned = (is_buy and market_analysis.trend == TrendDirection.BULLISH) or \
                        (not is_buy and market_analysis.trend == TrendDirection.BEARISH)
        
        if trend_aligned:
            adx_res = indicators.get('adx')
            adx_strength = adx_res.signal_strength / 100 if adx_res else 0
            points = trend_weight * (0.5 + 0.5 * adx_strength)
            score += points
            add_reason("Trend Align", f"{market_analysis.trend.value} ({market_analysis.trend_strength.value})", points)
        elif market_analysis.trend != TrendDirection.SIDEWAYS:
            points = -trend_weight * 0.75
            score += points
            add_reason("Trend Misalign", f"{market_analysis.trend.value}", points)

        pattern_weight = self.INDICATOR_WEIGHTS.get('pattern', 10)
        for p in patterns:
            total_weight += pattern_weight
            points = 0
            if is_buy and 'bullish' in p:
                points = pattern_weight
            elif not is_buy and 'bearish' in p:
                points = pattern_weight
            
            if points > 0:
                score += points
                add_reason("Pattern", p, points)

        if total_weight == 0: return 0.0, reasons
        
        final_score = (score / total_weight) * 100
        return min(max(final_score, 0), 100), reasons

    def _create_trading_signal(self, symbol, timeframe, signal_type, score, reasons, dynamic_levels, data, market_analysis):
        entry = dynamic_levels.primary_entry
        exit_price = dynamic_levels.primary_exit
        stop_loss = dynamic_levels.tight_stop

        rr = self._calculate_risk_reward(entry, exit_price, stop_loss)
        profit = ((exit_price - entry) / entry) * 100 if signal_type == SignalType.BUY else ((entry - exit_price) / entry) * 100

        market_context_dict = self._create_market_context(market_analysis)

        return TradingSignal(
            symbol=symbol, signal_type=signal_type, entry_price=entry, exit_price=exit_price, stop_loss=stop_loss,
            timestamp=datetime.now(), timeframe=timeframe, confidence_score=score, reasons=reasons,
            risk_reward_ratio=rr, predicted_profit=profit,
            volume_analysis=self.market_analyzer.volume_analyzer.analyze_volume_pattern(data),
            market_context=market_context_dict, dynamic_levels=vars(dynamic_levels),
            fundamental_analysis=market_analysis.fundamental_analysis
        )

    def _calculate_risk_reward(self, entry: float, exit_price: float, stop_loss: float) -> float:
        if pd.isna(entry) or pd.isna(exit_price) or pd.isna(stop_loss) or entry == stop_loss: return 0
        potential_profit = abs(exit_price - entry)
        potential_loss = abs(entry - stop_loss)
        return potential_profit / potential_loss if potential_loss > 0 else float('inf')

    def _create_market_context(self, market_analysis: "MarketAnalysis") -> Dict[str, Any]:
        return {
            'trend': market_analysis.trend.value, 'trend_strength': market_analysis.trend_strength.value,
            'volatility': market_analysis.volatility, 'momentum_score': market_analysis.momentum_score,
            'hurst_exponent': market_analysis.hurst_exponent,
        }

    async def generate_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[TradingSignal]:
        data = self._safe_dataframe(data)
        if data.empty or len(data) < 100:
            logger.warning(f"Insufficient data for {symbol} on {timeframe}")
            return []

        indicator_results = {name: indicator.calculate(data) for name, indicator in self.indicators.items()}
        sentiment_data = await self._gather_context_data(symbol)
        market_analysis = self.market_analyzer.analyze_market_condition(data)
        patterns = PatternAnalyzer.detect_patterns(data)
        
        signals = []
        for signal_type in [SignalType.BUY, SignalType.SELL]:
            score, reasons = self._calculate_confidence_score(signal_type, indicator_results, market_analysis, patterns, sentiment_data)
            
            if score >= self.config.get('min_confidence_score', 0):
                dynamic_levels = self.calculate_dynamic_levels(data, market_analysis, signal_type)
                trading_signal = self._create_trading_signal(symbol, timeframe, signal_type, score, reasons, dynamic_levels, data, market_analysis)
                
                if self.multi_tf_analyzer:
                    is_aligned = await self.multi_tf_analyzer.is_direction_aligned(symbol, timeframe)
                    if is_aligned:
                        trading_signal.confidence_score = min(100, score + self.INDICATOR_WEIGHTS['multi_tf'])
                        trading_signal.reasons.append("Confirm: Multi-Timeframe Alignment")
                    else:
                        trading_signal.confidence_score *= 0.8
                        trading_signal.reasons.append("Warning: Multi-Timeframe Misalignment")

                if self.lstm_model_manager:
                    await self.lstm_model_manager.train_model_if_needed(symbol, timeframe, data)
                    prediction = await self.lstm_model_manager.predict_async(symbol, timeframe, data)
                    if prediction is not None and len(prediction) > 0:
                        pred_price, last_close = prediction[0], data['close'].iloc[-1]
                        if (signal_type == SignalType.BUY and pred_price > last_close) or \
                           (signal_type == SignalType.SELL and pred_price < last_close):
                            trading_signal.confidence_score = min(100, trading_signal.confidence_score + self.INDICATOR_WEIGHTS['lstm'])
                            trading_signal.reasons.append(f"Confirm: LSTM Prediction ({pred_price:.4f})")
                
                signals.append(trading_signal)
        return signals

    async def _gather_context_data(self, symbol: str):
        tasks = {
            "indices": self.market_indices_fetcher.get_all_indices() if self.market_indices_fetcher else asyncio.sleep(0, result={}),
            "fear_greed": self.news_fetcher.fetch_fear_greed() if self.news_fetcher else asyncio.sleep(0, result=None),
        }
        results = await asyncio.gather(*tasks.values())
        indices, fear_greed = results
        sentiment_data = indices or {}
        sentiment_data['FEAR.GREED'] = fear_greed
        return sentiment_data


class SignalRanking:
    @staticmethod
    def rank_signals(signals: List[TradingSignal]) -> List[TradingSignal]:
        def signal_score(signal: TradingSignal) -> float:
            base_score = signal.confidence_score
            
            rr = signal.risk_reward_ratio or 0
            rr_bonus = 0
            if rr >= 3.0: rr_bonus = 15
            elif rr >= 2.0: rr_bonus = 10
            elif rr >= 1.5: rr_bonus = 5
            
            trend_bonus = 0
            mc = signal.market_context
            if mc:
                is_aligned = (signal.signal_type == SignalType.BUY and mc.get('trend') == TrendDirection.BULLISH.value) or \
                             (signal.signal_type == SignalType.SELL and mc.get('trend') == TrendDirection.BEARISH.value)
                if is_aligned:
                    trend_bonus = 10
                    if mc.get('trend_strength') == TrendStrength.STRONG.value:
                        trend_bonus += 5
            
            final_score = base_score + rr_bonus + trend_bonus
            return final_score

        return sorted(signals, key=signal_score, reverse=True)