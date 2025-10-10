import asyncio
from enum import Enum
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import redis.asyncio as redis

from .analyzers import MarketConditionAnalyzer, PatternAnalyzer
from .constants import (
    DEFAULT_INDICATOR_WEIGHTS, LONG_TERM_CONFIG,
    MULTI_TF_CONFIRMATION_MAP, MULTI_TF_CONFIRMATION_WEIGHTS
)
from .core import (
    MarketAnalysis, MarketCondition, SignalType,
    TradingSignal, TrendDirection, TrendStrength
)

from .data.binance import BinanceFetcher
from .data.marketindices import MarketIndicesFetcher
from .data.messari import MessariFetcher
from .data.news import NewsFetcher

from .indicators.indicators import get_all_indicators
from .logger_config import logger
from .market import MarketDataProvider
from .models import ModelManager
from .analysis_engine import AnalysisEngine


class MultiTimeframeAnalyzer:
    def __init__(self, market_data_provider: MarketDataProvider, indicators, redis_client: Optional[redis.Redis] = None, cache_ttl: Optional[Dict[str, int]] = None):
        self.market_data_provider = market_data_provider
        self.indicators = indicators
        self.market_analyzer = MarketConditionAnalyzer()
        self.redis = redis_client
        self.cache_ttl_map = cache_ttl or {
            "1h": 120, "4h": 300, "1d": 600, "1w": 1800, "1M": 3600
        }
        self._lock = asyncio.Lock()

    async def _get_cache(self, key: str) -> Optional[Dict]:
        if not self.redis:
            return None
        try:
            cached_data_str = await self.redis.get(key)
            if cached_data_str:
                cached_data = json.loads(cached_data_str)
                
                cached_timestamp = datetime.fromisoformat(cached_data.get('timestamp', '1970-01-01T00:00:00'))
                ttl = self.cache_ttl_map.get(cached_data.get('timeframe', '1h'), 600)
                if (datetime.now() - cached_timestamp).total_seconds() > ttl:
                    return None

                data_hash = cached_data.get('data_hash')
                
                deserialized_data = cached_data['analysis']
                deserialized_data['trend'] = TrendDirection(deserialized_data['trend'])
                deserialized_data['trend_strength'] = TrendStrength(deserialized_data['trend_strength'])
                
                return {'analysis': deserialized_data, 'data_hash': data_hash}
        except Exception as e:
            logger.warning(f"Redis GET or validation failed for {key}: {e}")
        return None

    async def _set_cache(self, key: str, analysis_dict: Dict, data_hash: str, ttl: int):
        if not self.redis:
            return
        try:
            serializable_value = analysis_dict.copy()
            for k, v in serializable_value.items():
                if isinstance(v, (TrendDirection, TrendStrength, MarketCondition)):
                    serializable_value[k] = v.value
            
            payload = {
                'timestamp': datetime.now().isoformat(),
                'data_hash': data_hash,
                'analysis': serializable_value
            }
            await self.redis.set(key, json.dumps(payload), ex=ttl)
        except Exception as e:
            logger.warning(f"Redis SET failed for {key}: {e}")

    async def fetch_and_analyze_batch(self, symbol: str, timeframes: List[str]) -> Dict[str, MarketAnalysis]:
        tasks = {}
        for tf in timeframes:
            limit = LONG_TERM_CONFIG['min_data_points'].get(tf, 200)
            tasks[tf] = self.market_data_provider.fetch_ohlcv_data(symbol, tf, limit=limit)

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        analyses: Dict[str, MarketAnalysis] = {}
        for tf, result in zip(tasks.keys(), results):
            if isinstance(result, pd.DataFrame) and not result.empty:
                analysis = self.market_analyzer.analyze_market_condition(result)
                analyses[tf] = analysis
                ttl = self.cache_ttl_map.get(tf, 600)
                analysis_dict = {k: v.value if isinstance(v, Enum) else v for k, v in analysis.__dict__.items()}
                
                data_hash = pd.util.hash_pandas_object(result).sum()
                await self._set_cache(f"mtf_analysis:{symbol}:{tf}", analysis_dict, str(data_hash), ttl)
            elif isinstance(result, Exception):
                logger.warning(f"Failed to fetch data for {symbol}-{tf} in batch: {result}")
        return analyses

    async def get_confirmation_score(self, symbol: str, exec_tf: str, base_trend: TrendDirection, data: pd.DataFrame) -> Tuple[float, float]:
        confirm_tfs = MULTI_TF_CONFIRMATION_MAP.get(exec_tf, [])
        if not confirm_tfs:
            return 1.0, 1.0

        all_needed_tfs = list(set(confirm_tfs))

        analyses: Dict[str, MarketAnalysis] = {}
        tfs_to_fetch = []
        
        current_data_hash = str(pd.util.hash_pandas_object(data).sum())

        for tf in all_needed_tfs:
            cache_key = f"mtf_analysis:{symbol}:{tf}"
            cached = await self._get_cache(cache_key)
            if cached and cached.get('data_hash') == current_data_hash:
                try:
                    analyses[tf] = MarketAnalysis(**cached['analysis'])
                except (TypeError, KeyError) as e:
                    logger.warning(f"Cached data for {cache_key} is invalid ({e}). Refetching.")
                    tfs_to_fetch.append(tf)
            else:
                tfs_to_fetch.append(tf)

        if tfs_to_fetch:
            new_analyses = await self.fetch_and_analyze_batch(symbol, tfs_to_fetch)
            analyses.update(new_analyses)

        total_score, total_weight = 0.0, 0.0
        for tf in confirm_tfs:
            confirm_analysis = analyses.get(tf)
            if confirm_analysis:
                weight = self._get_dynamic_weight(exec_tf, tf, confirm_analysis)

                alignment = 0.0
                if confirm_analysis.trend == base_trend:
                    alignment = 1.0
                elif confirm_analysis.trend != TrendDirection.SIDEWAYS:
                    alignment = -0.5

                strength_multiplier = {TrendStrength.STRONG: 1.2, TrendStrength.MODERATE: 1.0, TrendStrength.WEAK: 0.7}
                persistence_factor = np.clip(confirm_analysis.hurst_exponent, 0.4, 0.6) if confirm_analysis.hurst_exponent else 0.5
                
                alignment_score = alignment * strength_multiplier.get(confirm_analysis.trend_strength, 1.0) * (persistence_factor * 2)

                total_score += alignment_score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0, 0.0

        final_score = total_score / total_weight if total_weight > 0 else 0.0
        return final_score, total_weight

    def _get_dynamic_weight(self, exec_tf: str, confirm_tf: str, analysis: MarketAnalysis) -> float:
        base_weight = MULTI_TF_CONFIRMATION_WEIGHTS.get(exec_tf, {}).get(confirm_tf, 1.0)
        strength_multiplier = {TrendStrength.STRONG: 1.2, TrendStrength.MODERATE: 1.0, TrendStrength.WEAK: 0.8}
        volatility_multiplier = 1.0 / (1.0 + analysis.volatility / 10)
        
        final_weight = base_weight * strength_multiplier.get(analysis.trend_strength, 1.0) * volatility_multiplier
        return np.clip(final_weight, 0.5, 1.5)


class SignalGenerator:
    def __init__(self, market_data_provider: MarketDataProvider,
                news_fetcher: Optional[NewsFetcher] = None,
                market_indices_fetcher: Optional[MarketIndicesFetcher] = None,
                model_manager: Optional[ModelManager] = None,
                multi_tf_analyzer: Optional[MultiTimeframeAnalyzer] = None,
                config: Optional[Dict] = None,
                binance_fetcher: Optional[BinanceFetcher] = None,
                messari_fetcher: Optional[MessariFetcher] = None,
                analysis_engine: Optional[AnalysisEngine] = None):
        self.market_data_provider = market_data_provider
        self.news_fetcher = news_fetcher
        self.market_indices_fetcher = market_indices_fetcher
        self.model_manager = model_manager
        self.multi_tf_analyzer = multi_tf_analyzer
        self.config = config or {'min_confidence_score': 75}
        self.binance_fetcher = binance_fetcher
        self.messari_fetcher = messari_fetcher
        self.analysis_engine = analysis_engine
        self.indicators = get_all_indicators()
        self.market_analyzer = MarketConditionAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()
        self.indicator_weights = self.config.get('indicator_weights', DEFAULT_INDICATOR_WEIGHTS)
        self.indicator_cache: Dict[str, Tuple[float, Any]] = {}

    async def generate_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[TradingSignal]:
        if not self.analysis_engine:
            logger.error("AnalysisEngine is not initialized in SignalGenerator.")
            return []
            
        signal = await self.analysis_engine.run_full_analysis(symbol, timeframe, data)
        
        if signal:
            return [signal]
        return []


class SignalRanking:
    _historical_performance: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def update_performance(signal: TradingSignal, success: bool):
        key = f"{signal.symbol}-{signal.timeframe}"
        if key not in SignalRanking._historical_performance:
            SignalRanking._historical_performance[key] = {'success': 0, 'total': 0}
        
        SignalRanking._historical_performance[key]['total'] += 1
        if success:
            SignalRanking._historical_performance[key]['success'] += 1

    @staticmethod
    def get_performance_factor(signal: TradingSignal) -> float:
        key = f"{signal.symbol}-{signal.timeframe}"
        perf = SignalRanking._historical_performance.get(key)
        if perf and perf['total'] > 5:
            return (perf['success'] / perf['total']) * 1.2
        return 1.0

    @staticmethod
    def calculate_signal_score(signal: TradingSignal) -> float:
        weights = {
            'confidence': 0.40,
            'rr': 0.25,
            'trend': 0.25,
            'timeframe': 0.10
        }

        base_score = signal.confidence_score

        rr = signal.risk_reward_ratio or 0
        rr_bonus = np.clip(rr, 0, 5) * 20

        trend_bonus = 0
        if signal.market_context:
            mc = signal.market_context
            trend_val = mc.get('trend')
            is_aligned = (signal.signal_type == SignalType.BUY and trend_val == 'bullish') or \
                            (signal.signal_type == SignalType.SELL and trend_val == 'bearish')
            if is_aligned:
                trend_bonus = 70
                strength_val = mc.get('trend_strength')
                if strength_val == 'strong':
                    trend_bonus += 30
            elif trend_val != 'sideways':
                trend_bonus = 20

        tf_weight = LONG_TERM_CONFIG['timeframe_priority_weights'].get(signal.timeframe, 0.5)
        tf_bonus = tf_weight * 100

        final_score = (base_score * weights['confidence'] +
                       rr_bonus * weights['rr'] +
                       trend_bonus * weights['trend'] +
                       tf_bonus * weights['timeframe'])
        
        performance_factor = SignalRanking.get_performance_factor(signal)
        final_score *= performance_factor
        
        return final_score

    @staticmethod
    def rank_signals(signals: List[TradingSignal]) -> List[TradingSignal]:
        if not signals:
            return []
            
        unique_signals = {f"{s.symbol}-{s.timeframe}": s for s in signals}.values()

        sorted_signals = sorted(unique_signals, key=lambda s: SignalRanking.calculate_signal_score(s), reverse=True)

        return sorted_signals