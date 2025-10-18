import json
from typing import Dict, List, Optional

from common.utils import RateLimiter
from common.cache import CacheKeyBuilder, CacheCategory
from data.sources.base_fetcher import BaseFetcher


cryptopanic_rate_limiter = RateLimiter(max_requests=20, time_window=60)


class CryptoPanicFetcher(BaseFetcher):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = "https://cryptopanic.com/api/v1"
        self.source_name = "cryptopanic"

    async def fetch_posts(
        self, currencies: Optional[List[str]] = None, kind: Optional[str] = None
    ) -> List[Dict]:
        url = f"{self.base_url}/posts/"
        params = {"auth_token": self.api_key, "public": "true"}
        if currencies:
            params["currencies"] = ",".join(currencies)
        if kind:
            params["kind"] = kind

        data = await self._fetch_json(
            url, params, limiter=cryptopanic_rate_limiter, endpoint_name="posts"
        )
        return data.get("results", [])

    async def fetch_media(self) -> List[Dict]:
        cache_key = CacheKeyBuilder.build(
            CacheCategory.GENERIC, self.source_name, ["media"]
        )
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"{self.base_url}/media/"
        params = {"auth_token": self.api_key}

        data = await self._fetch_json(
            url, params, limiter=cryptopanic_rate_limiter, endpoint_name="media"
        )
        media_list = data.get("results", [])
        if self.redis and media_list:
            await self.redis.set(cache_key, json.dumps(media_list), ex=86400)
        return media_list
