from enum import Enum
from typing import List, Optional


class CacheCategory(Enum):
    OHLCV = "ohlcv"
    MACRO = "macro"
    NEWS = "news"
    SENTIMENT = "sentiment"
    FEAR_GREED = "fear_greed"
    TRENDING = "trending"
    DERIVATIVES = "derivatives"
    FUNDAMENTAL = "fundamental"
    METRICS = "metrics"
    PROFILE = "profile"
    GENERIC = "generic"
    INDICES = "indices"
    YIELD = "yield"
    COIN_LIST = "coin_list"


class CacheKeyBuilder:
    """A centralized builder for creating consistent cache keys."""

    PREFIX = "cache"

    @staticmethod
    def build(
        category: CacheCategory,
        source: str,
        parts: List[str],
        symbol: Optional[str] = None,
    ) -> str:
        """
        Constructs a standardized cache key.

        Example:
        >>> CacheKeyBuilder.build(CacheCategory.OHLCV, "binance", ["1h", "1000"], "BTC/USDT")
        'cache:ohlcv:binance:btc_usdt:1h:1000'
        """
        safe_symbol = (
            symbol.lower().replace("/", "_").replace("\\", "") if symbol else "global"
        )
        key_parts = [
            CacheKeyBuilder.PREFIX,
            category.value,
            source.lower(),
            safe_symbol,
        ] + [str(p).lower() for p in parts]
        return ":".join(key_parts)

    @staticmethod
    def ohlcv_key(source: str, symbol: str, timeframe: str, limit: int) -> str:
        return CacheKeyBuilder.build(
            CacheCategory.OHLCV, source, [timeframe, str(limit)], symbol
        )

    @staticmethod
    def generic_key(source: str, identifier: str, symbol: Optional[str] = None) -> str:
        return CacheKeyBuilder.build(
            CacheCategory.GENERIC, source, [identifier], symbol
        )

    @staticmethod
    def mtf_analysis_key(symbol: str, timeframe: str) -> str:
        return CacheKeyBuilder.build(
            CacheCategory.METRICS, "internal", ["mtf_analysis", timeframe], symbol
        )

    @staticmethod
    def derivatives_key(source: str, metric: str, symbol: str) -> str:
        """Creates a cache key for derivatives data."""
        return CacheKeyBuilder.build(
            CacheCategory.DERIVATIVES, source, [metric], symbol
        )

    @staticmethod
    def macro_key(source: str, identifier: str) -> str:
        """Creates a cache key for macro-economic data."""
        return CacheKeyBuilder.build(CacheCategory.MACRO, source, [identifier])

    @staticmethod
    def indices_key(source: str, identifier: str) -> str:
        """Creates a cache key for market indices."""
        return CacheKeyBuilder.build(CacheCategory.INDICES, source, [identifier])
