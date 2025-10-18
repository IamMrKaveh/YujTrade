import asyncio
from contextlib import contextmanager, asynccontextmanager
from typing import Optional, AsyncGenerator, Any, Generator

import aiohttp
import redis.asyncio as redis
import tensorflow as tf

from config.logger import logger
from config.settings import SecretsManager


class ResourceManager:
    """A centralized class to manage shared resources like sessions and DB connections."""

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._redis_client: Optional[redis.Redis] = None
        self._lock = asyncio.Lock()
        self._redis_connection_pool: Optional[redis.ConnectionPool] = None

    async def get_session(self) -> aiohttp.ClientSession:
        """Provides a shared aiohttp.ClientSession."""
        async with self._lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30)
                )
                logger.info("AIOHTTP session created.")
            return self._session

    async def get_redis_client(self) -> Optional[redis.Redis]:
        """Provides a shared Redis client instance."""
        async with self._lock:
            if self._redis_client is None:
                if not all(
                    [
                        SecretsManager.REDIS_HOST,
                        SecretsManager.REDIS_PORT,
                        SecretsManager.REDIS_TOKEN,
                    ]
                ):
                    logger.warning("Redis not configured. Caching will be disabled.")
                    return None
                try:
                    redis_url = f"redis://default:{SecretsManager.REDIS_TOKEN}@{SecretsManager.REDIS_HOST}:{SecretsManager.REDIS_PORT}"
                    self._redis_connection_pool = redis.ConnectionPool.from_url(
                        redis_url, decode_responses=True, max_connections=50
                    )
                    self._redis_client = redis.Redis(
                        connection_pool=self._redis_connection_pool
                    )
                    await self._redis_client.ping()
                    logger.info("Redis connection pool created successfully.")
                except Exception as e:
                    logger.error(
                        f"Redis connection failed: {e}. Caching will be disabled."
                    )
                    self._redis_client = None
                    if self._redis_connection_pool:
                        await self._redis_connection_pool.disconnect()
                        self._redis_connection_pool = None
            return self._redis_client

    async def cleanup(self):
        """Cleans up all managed resources."""
        logger.info("Cleaning up ResourceManager...")
        async with self._lock:
            cleanup_tasks = []
            if self._session and not self._session.closed:
                cleanup_tasks.append(self._session.close())
                logger.info("AIOHTTP session closed.")

            if self._redis_client:
                cleanup_tasks.append(self._redis_client.close())
                logger.info("Redis client closed.")

            if self._redis_connection_pool:
                cleanup_tasks.append(self._redis_connection_pool.disconnect())
                logger.info("Redis connection pool disconnected.")

            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

            self._session = None
            self._redis_client = None
            self._redis_connection_pool = None
        logger.info("ResourceManager cleaned up successfully.")

    async def __aenter__(self):
        await self.get_session()
        await self.get_redis_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()


@contextmanager
def managed_tf_session() -> Generator[None, None, None]:
    """
    A context manager to handle TensorFlow session clearing.
    """
    try:
        yield
    finally:
        tf.keras.backend.clear_session()
        logger.debug("TensorFlow session cleared.")


@asynccontextmanager
async def managed_resource(resource: Any) -> AsyncGenerator[Any, None]:
    """
    A generic async context manager for resources that have an async cleanup method.
    Example: aiohttp.ClientSession, DB Connections.
    """
    try:
        yield resource
    finally:
        if hasattr(resource, "close") and asyncio.iscoroutinefunction(resource.close):
            await resource.close()
        elif hasattr(resource, "shutdown") and asyncio.iscoroutinefunction(
            resource.shutdown
        ):
            await resource.shutdown()
        elif hasattr(resource, "cleanup") and asyncio.iscoroutinefunction(
            resource.cleanup
        ):
            await resource.cleanup()
