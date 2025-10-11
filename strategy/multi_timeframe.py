import asyncio
from datetime import datetime
from enum import Enum
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import redis
import talib

from module.analyzers import MarketConditionAnalyzer

from ..common.constants import (
    DEFAULT_INDICATOR_WEIGHTS, LONG_TERM_CONFIG,
    MULTI_TF_CONFIRMATION_MAP, MULTI_TF_CONFIRMATION_WEIGHTS
)
from ..common.core import (
    MarketAnalysis, MarketCondition, SignalType,
    TradingSignal, TrendDirection, TrendStrength
)

from .data.binance import BinanceFetcher
from .data.marketindices import MarketIndicesFetcher
from .data.messari import MessariFetcher
from .data.news import NewsFetcher

from .indicators.indicators import get_all_indicators
from ..config.logger import logger
from .market import MarketDataProvider
from .models import ModelManager
from ..common.exceptions import InvalidSymbolError


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

    async def _get_cache(self, key: str, current_data_hash: str, config_version: str = "1.0") -> Optional[Dict]:
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
                cached_config_version = cached_data.get('config_version', '1.0')
                if data_hash != current_data_hash or cached_config_version != config_version:
                    logger.debug(f"Cache invalidated for {key} due to data or config change")
                    return None
                
                deserialized_data = cached_data['analysis']
                deserialized_data['trend'] = TrendDirection(deserialized_data['trend'])
                deserialized_data['trend_strength'] = TrendStrength(deserialized_data['trend_strength'])
                
                return {'analysis': deserialized_data, 'data_hash': data_hash}
        except Exception as e:
            logger.warning(f"Redis GET or validation failed for {key}: {e}")
        return None

    async def _set_cache(self, key: str, analysis_dict: Dict, data_hash: str, ttl: int, timeframe: str, config_version: str = "1.0"):
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
                'timeframe': timeframe,
                'config_version': config_version,
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
                
                data_hash = str(pd.util.hash_pandas_object(result).sum())
                config_hash = str(hash(str(DEFAULT_INDICATOR_WEIGHTS)))
                await self._set_cache(f"mtf_analysis:{symbol}:{tf}", analysis_dict, data_hash, ttl, tf, config_hash)
            elif isinstance(result, Exception):
                logger.warning(f"Failed to fetch data for {symbol}-{tf} in batch: {result}")
        return analyses

    async def get_confirmation_score(self, symbol: str, exec_tf: str, base_trend: TrendDirection, data: pd.DataFrame) -> Tuple[float, float, bool]:
        confirm_tfs = MULTI_TF_CONFIRMATION_MAP.get(exec_tf, [])
        if not confirm_tfs:
            return 1.0, 1.0, False

        all_needed_tfs = list(set(confirm_tfs))

        analyses: Dict[str, MarketAnalysis] = {}
        tfs_to_fetch = []
        
        current_data_hash = str(pd.util.hash_pandas_object(data).sum())
        config_hash = str(hash(str(DEFAULT_INDICATOR_WEIGHTS)))

        for tf in all_needed_tfs:
            cache_key = f"mtf_analysis:{symbol}:{tf}"
            cached = await self._get_cache(cache_key, current_data_hash, config_hash)
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
        direction_conflict = False
        
        for tf in confirm_tfs:
            confirm_analysis = analyses.get(tf)
            if not confirm_analysis:
                continue
                
            weight = MULTI_TF_CONFIRMATION_WEIGHTS.get(exec_tf, {}).get(tf, 1.0)
            
            alignment = 0.0
            if confirm_analysis.trend == base_trend:
                alignment = 1.0
            elif confirm_analysis.trend != TrendDirection.SIDEWAYS:
                alignment = -0.5
                direction_conflict = True

            strength_multiplier = {TrendStrength.STRONG: 1.2, TrendStrength.MODERATE: 1.0, TrendStrength.WEAK: 0.7}
            persistence_factor = np.clip(confirm_analysis.hurst_exponent, 0.4, 0.6) if confirm_analysis.hurst_exponent else 0.5
            
            alignment_score = alignment * strength_multiplier.get(confirm_analysis.trend_strength, 1.0) * persistence_factor

            total_score += alignment_score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0, 0.0, direction_conflict

        final_score = total_score / total_weight if total_weight > 0 else 0.0
        return final_score, total_weight, direction_conflict



