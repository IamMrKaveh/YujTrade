from typing import Optional
import aiohttp
import redis
from module.logger_config import logger
from module.circuit_breaker import CircuitBreaker
from module.exceptions import APIRateLimitError, NetworkError
from module.utils import RateLimiter, async_retry


class BaseFetcher:
    def __init__(self, redis_client: Optional[redis.Redis] = None, session: Optional[aiohttp.ClientSession] = None):
        self.redis = redis_client
        self._owns_session = session is None
        self.session = session
        self._is_closed = False
        self.circuit_breaker = CircuitBreaker()
        self.class_name = self.__class__.__name__

    def _check_if_closed(self):
        if self._is_closed:
            raise RuntimeError(f"{self.class_name} has been closed and cannot be used")

    async def _get_session(self) -> aiohttp.ClientSession:
        self._check_if_closed()
        if self.session and not self.session.closed:
            return self.session
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self.session

    async def close(self):
        if self._is_closed:
            return
        self._is_closed = True
        if self._owns_session and self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    @async_retry(attempts=3, delay=5, exceptions=(NetworkError, APIRateLimitError))
    async def _fetch_json(self, url, params=None, headers=None, limiter: Optional[RateLimiter] = None, endpoint_name: str = "default"):
        self._check_if_closed()
        if limiter:
            await limiter.wait_if_needed(endpoint_name)

        async def _do_fetch():
            session = await self._get_session()
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 429:
                    raise APIRateLimitError(f"Rate limit exceeded for {url}")
                if response.status != 200:
                    logger.error(f"Error fetching from {url}: Status {response.status}, Body: {await response.text()}")
                    raise NetworkError(f"HTTP Error {response.status} for {url}")
                return await response.json()

        return await self.circuit_breaker.call(_do_fetch)