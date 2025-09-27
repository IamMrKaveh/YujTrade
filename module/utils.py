import asyncio
import time
from functools import wraps
from typing import Dict, Tuple, Type

from module.logger_config import logger


class RateLimiter:
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, list] = {}
        self._lock = asyncio.Lock()
    
    async def wait_if_needed(self, endpoint: str):
        async with self._lock:
            now = time.time()
            reqs = self.requests.setdefault(endpoint, [])
            # Filter out old requests
            reqs = [t for t in reqs if now - t < self.time_window]
            if len(reqs) >= self.max_requests:
                # Find the oldest request to calculate sleep time
                oldest_request_time = reqs[0]
                sleep_time = self.time_window - (now - oldest_request_time)
                if sleep_time > 0:
                    logger.debug(f"Rate limit hit for {endpoint}. Sleeping for {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                # After sleeping, the window has passed for all previous requests
                self.requests[endpoint] = []
            
            self.requests[endpoint].append(time.time())

def async_retry(attempts: int = 3, delay: int = 5, exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """
    A decorator for retrying an async function if it raises a specified exception.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1}/{attempts} for {func.__name__} failed: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
            logger.error(f"Function {func.__name__} failed after {attempts} attempts.")
            if last_exception:
                raise last_exception
        return wrapper
    return decorator