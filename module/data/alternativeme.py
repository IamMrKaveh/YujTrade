import json
from .base import BaseFetcher
from module.utils import RateLimiter


alternative_me_rate_limiter = RateLimiter(max_requests=20, time_window=60)


class AlternativeMeFetcher(BaseFetcher):
    async def fetch_fear_greed(self, limit: int = 1):
        cache_key = f"cache:fear_greed:{limit}"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"https://api.alternative.me/fng/?limit={limit}"
        data = await self._fetch_json(url, limiter=alternative_me_rate_limiter, endpoint_name="fng")
        
        results = data.get("data", [])
        if results and self.redis:
            await self.redis.set(cache_key, json.dumps(results), ex=3600)
        return results
