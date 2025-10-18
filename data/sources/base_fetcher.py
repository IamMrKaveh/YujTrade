from typing import Optional
import aiohttp
import redis.asyncio as redis
from config.logger import logger
from utils.circuit_breaker import CircuitBreaker
from common.exceptions import (
    APIRateLimitError,
    NetworkError,
    ObjectClosedError,
    InvalidSymbolError,
)
from common.utils import RateLimiter, async_retry
from config.settings import ConfigManager


class BaseFetcher:
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        session: Optional[aiohttp.ClientSession] = None,
        config_manager: Optional[ConfigManager] = None,
    ):
        self.redis = redis_client
        self.session = session
        self.config_manager = config_manager or ConfigManager()
        self._is_closed = False
        self.circuit_breaker = CircuitBreaker()
        self.class_name = self.__class__.__name__

    def _check_if_closed(self):
        if self._is_closed:
            raise ObjectClosedError(
                f"{self.class_name} has been closed and cannot be used"
            )

    async def _get_session(self) -> aiohttp.ClientSession:
        self._check_if_closed()
        if self.session is None or self.session.closed:
            # This should ideally not be hit if ResourceManager is used correctly
            raise RuntimeError(
                f"Session for {self.class_name} is not available. It should be provided by ResourceManager."
            )
        return self.session

    async def close(self):
        # This method is now a no-op as the session is managed externally
        # by ResourceManager. This prevents accidental closing of the shared session.
        if self._is_closed:
            return
        self._is_closed = True
        logger.debug(
            f"Fetcher {self.class_name} instance closed (shared session not affected)."
        )

    @async_retry(
        attempts=3,
        delay=5,
        exceptions=(NetworkError, APIRateLimitError),
        ignore_exceptions=(InvalidSymbolError,),
    )
    async def _fetch_json(
        self,
        url,
        params=None,
        headers=None,
        limiter: Optional[RateLimiter] = None,
        endpoint_name: str = "default",
    ):
        self._check_if_closed()
        if limiter:
            await limiter.wait_if_needed(endpoint_name)

        async def _do_fetch():
            session = await self._get_session()
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 429:
                    raise APIRateLimitError(f"Rate limit exceeded for {url}")
                if response.status >= 400:
                    try:
                        error_text = await response.text()
                    except Exception:
                        error_text = "Could not read response body."

                    if "Invalid symbol" in error_text:
                        raise InvalidSymbolError(f"Invalid symbol for {url}")

                    logger.error(
                        f"Error fetching from {url}: Status {response.status}, Body: {error_text}"
                    )
                    response.raise_for_status()
                return await response.json()

        return await self.circuit_breaker.call(_do_fetch)
