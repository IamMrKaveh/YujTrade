import json
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
import redis
from config.logger import logger
from utils.circuit_breaker import CircuitBreaker
from common.exceptions import APIRateLimitError, NetworkError, InvalidSymbolError
from common.utils import RateLimiter, async_retry
from data.sources.base_fetcher import BaseFetcher
from common.cache import CacheKeyBuilder, CacheCategory


coindesk_rate_limiter = RateLimiter(max_requests=20, time_window=60)


class CoinDeskFetcher(BaseFetcher):
    def __init__(
        self,
        api_key: str,
        redis_client: Optional[redis.Redis] = None,
        session: Optional[aiohttp.ClientSession] = None,
        **kwargs,
    ):
        super().__init__(redis_client=redis_client, session=session, **kwargs)
        self.api_key = api_key
        self.base_url = "https://data-api.coindesk.com"
        self.source_name = "coindesk"

    @async_retry(
        attempts=3,
        delay=5,
        exceptions=(NetworkError, APIRateLimitError),
        ignore_exceptions=(InvalidSymbolError,),
    )
    async def _fetch(self, endpoint: str, params: Optional[Dict] = None):
        self._check_if_closed()
        headers = {"x-api-key": self.api_key}
        url = f"{self.base_url}/{endpoint}"
        await coindesk_rate_limiter.wait_if_needed(endpoint)

        async def _do_fetch():
            session = await self._get_session()
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 429:
                    raise APIRateLimitError("CoinDesk rate limit exceeded")
                if response.status >= 400:
                    text = await response.text()
                    if "not found" in text.lower():
                        raise InvalidSymbolError(
                            f"Invalid symbol for CoinDesk: {params.get('symbol') if params else endpoint}"
                        )
                    raise NetworkError(
                        f"HTTP Error {response.status} for {url}: {text}"
                    )
                return await response.json()

        return await self.circuit_breaker.call(_do_fetch)

    async def get_historical_ohlc(
        self, symbol: str, timeframe: str, limit: int
    ) -> Optional[pd.DataFrame]:
        self._check_if_closed()
        try:
            timeframe_lower = timeframe.lower()
            asset_id = symbol.upper().replace("/USDT", "-USD")

            if (
                "d" in timeframe_lower
                or "w" in timeframe_lower
                or "m" in timeframe_lower
            ):
                endpoint = f"spot/v1/historical/ohlcv/{asset_id}/d"
                params = {"limit": limit}
            else:
                endpoint = f"spot/v1/historical/ohlcv/{asset_id}/h"
                params = {"time_frame": timeframe, "limit": limit}

            data = await self._fetch(endpoint, params)

            if not data or "data" not in data or not data["data"]:
                return None

            ohlc_data = data["data"]
            if not ohlc_data:
                return None

            df = pd.DataFrame(ohlc_data)
            df.rename(
                columns={
                    "ts": "timestamp",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                },
                inplace=True,
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df.astype(float).sort_index()

            return df.tail(limit)
        except InvalidSymbolError:
            logger.warning(f"Invalid symbol for CoinDesk OHLC: {symbol}")
            return None
        except Exception as e:
            logger.error(
                f"Error fetching historical OHLC from CoinDesk for {symbol}: {e}"
            )
            return None

    async def get_news(self, symbols: List[str]) -> List[Dict]:
        self._check_if_closed()
        try:
            all_news = []
            news_url = "https://api.coindesk.com/v1/news"
            params = {"slug": ",".join(s.lower().replace("-usdt", "") for s in symbols)}
            headers = {"X-CoinDesk-API-Key": self.api_key}

            session = await self._get_session()
            async with session.get(
                news_url, params=params, headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and "data" in data:
                        all_news.extend(data["data"])

            unique_news = list({item["id"]: item for item in all_news}.values())
            return unique_news
        except Exception as e:
            logger.error(f"Error fetching news from CoinDesk: {e}")
            return []

    async def get_bitcoin_price_index(self) -> Optional[Dict[str, Any]]:
        self._check_if_closed()
        cache_key = CacheKeyBuilder.indices_key(self.source_name, "bpi")
        cache_ttl = self.config_manager.get_cache_ttl("indices")

        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        try:
            session = await self._get_session()
            async with session.get(
                "https://api.coindesk.com/v1/bpi/currentprice.json"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if self.redis:
                        await self.redis.set(cache_key, json.dumps(data), ex=cache_ttl)
                    return data
            return None
        except Exception as e:
            logger.error(f"Error fetching Bitcoin Price Index from CoinDesk: {e}")
            return None

    async def get_latest_news(self, limit: int = 20) -> List[Dict]:
        self._check_if_closed()
        cache_key = CacheKeyBuilder.build(
            CacheCategory.NEWS, self.source_name, ["latest", str(limit)]
        )
        cache_ttl = self.config_manager.get_cache_ttl("news")

        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")

        try:
            news_url = "https://api.coindesk.com/v1/news/latest"
            params = {"limit": limit}
            headers = {"X-CoinDesk-API-Key": self.api_key}

            session = await self._get_session()
            async with session.get(
                news_url, params=params, headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    news_items = data.get("data", [])
                    if self.redis and news_items:
                        await self.redis.set(
                            cache_key, json.dumps(news_items), ex=cache_ttl
                        )
                    return news_items
            return []
        except Exception as e:
            logger.error(f"Error fetching latest news: {e}")
            return []
