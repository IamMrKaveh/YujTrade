import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import redis

from config.settings import ConfigManager
from config.logger import logger
from common.core import SignalType, TrendDirection, MarketAnalysis
from common.constants import MULTI_TF_CONFIRMATION_MAP, MULTI_TF_CONFIRMATION_WEIGHTS
from common.exceptions import InsufficientDataError
from data.data_provider import MarketDataProvider
from analysis.market_analyzer import MarketConditionAnalyzer
from common.cache import CacheKeyBuilder, CacheCategory


class MultiTimeframeAnalyzer:
    def __init__(
        self,
        market_data_provider: MarketDataProvider,
        redis_client: Optional[redis.Redis] = None,
        config_manager: Optional[ConfigManager] = None,
    ):
        self.data_provider = market_data_provider
        self.redis = redis_client
        self.config = config_manager or ConfigManager()
        self.analyzer = MarketConditionAnalyzer()
        self.cache_ttl = 3600  # 1 hour

    async def get_confirmation_score(
        self,
        symbol: str,
        base_timeframe: str,
        signal_type: SignalType,
        base_data: pd.DataFrame,
    ) -> Tuple[float, Dict[str, MarketAnalysis], bool]:
        higher_timeframes = MULTI_TF_CONFIRMATION_MAP.get(base_timeframe, [])
        if not higher_timeframes:
            return 1.0, {}, False

        weights = MULTI_TF_CONFIRMATION_WEIGHTS.get(base_timeframe, {})
        total_score = 0.0
        total_weight = 0.0
        analyses: Dict[str, MarketAnalysis] = {}
        direction_conflict = False

        for tf in higher_timeframes:
            try:
                higher_tf_data = await self._get_higher_tf_data(symbol, tf)
                if higher_tf_data is None or higher_tf_data.empty:
                    continue

                analysis = self.analyzer.analyze_market_condition(higher_tf_data)
                analyses[tf] = analysis

                score, conflict = self._calculate_single_tf_score(analysis, signal_type)
                if conflict:
                    direction_conflict = True

                weight = weights.get(tf, 1.0)
                total_score += score * weight
                total_weight += weight

            except InsufficientDataError:
                logger.warning(
                    f"Insufficient data for multi-timeframe analysis on {symbol}-{tf}"
                )
            except Exception as e:
                logger.error(
                    f"Error in multi-timeframe analysis for {symbol}-{tf}: {e}"
                )

        final_score = (total_score / total_weight) if total_weight > 0 else 1.0
        return final_score, analyses, direction_conflict

    async def _get_higher_tf_data(
        self, symbol: str, timeframe: str
    ) -> Optional[pd.DataFrame]:
        cache_key = CacheKeyBuilder.mtf_analysis_key(symbol, timeframe)
        if self.redis:
            try:
                cached_data = await self.redis.get(cache_key)
                if cached_data:
                    return pd.read_json(cached_data, orient="split")
            except Exception as e:
                logger.warning(f"Redis GET for MTF data failed: {e}")

        data = await self.data_provider.fetch_ohlcv_data(symbol, timeframe, limit=200)
        if data is not None and not data.empty and self.redis:
            try:
                await self.redis.set(
                    cache_key, data.to_json(orient="split"), ex=self.cache_ttl
                )
            except Exception as e:
                logger.warning(f"Redis SET for MTF data failed: {e}")

        return data

    def _calculate_single_tf_score(
        self, analysis: MarketAnalysis, signal_type: SignalType
    ) -> Tuple[float, bool]:
        score = 0.5  # Neutral starting point
        conflict = False

        is_bullish_signal = signal_type == SignalType.BUY
        is_bearish_signal = signal_type == SignalType.SELL

        # Trend alignment
        if is_bullish_signal and analysis.trend == TrendDirection.BULLISH:
            score += 0.3
        elif is_bearish_signal and analysis.trend == TrendDirection.BEARISH:
            score += 0.3
        elif (is_bullish_signal and analysis.trend == TrendDirection.BEARISH) or (
            is_bearish_signal and analysis.trend == TrendDirection.BULLISH
        ):
            score -= 0.4
            conflict = True

        # Trend strength
        if analysis.trend_strength.value == "strong":
            score += 0.15
        elif analysis.trend_strength.value == "weak":
            score -= 0.1

        # Volume confirmation
        if analysis.volume_confirmation:
            score += 0.1

        return np.clip(score, 0, 1), conflict
