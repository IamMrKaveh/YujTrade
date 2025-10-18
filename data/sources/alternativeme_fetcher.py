import json
from typing import List, Dict, Any

from data.sources.base_fetcher import BaseFetcher
from common.utils import RateLimiter
from common.cache import CacheKeyBuilder, CacheCategory


alternative_me_rate_limiter = RateLimiter(max_requests=20, time_window=60)


class AlternativeMeFetcher(BaseFetcher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.source_name = "alternative.me"

    async def fetch_fear_greed(self, limit: int = 1) -> List[Dict[str, Any]]:
        cache_key = CacheKeyBuilder.build(
            CacheCategory.FEAR_GREED, self.source_name, [str(limit)]
        )
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"https://api.alternative.me/fng/?limit={limit}"
        data = await self._fetch_json(
            url, limiter=alternative_me_rate_limiter, endpoint_name="fng"
        )

        results = data.get("data", [])
        if results and self.redis:
            await self.redis.set(cache_key, json.dumps(results), ex=3600)
        return results
    