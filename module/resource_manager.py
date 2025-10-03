import asyncio
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator, Any

import tensorflow as tf
from module.logger_config import logger


@asynccontextmanager
async def managed_tf_session() -> AsyncGenerator[None, None]:
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
        if hasattr(resource, 'close') and asyncio.iscoroutinefunction(resource.close):
            await resource.close()
        elif hasattr(resource, 'shutdown') and asyncio.iscoroutinefunction(resource.shutdown):
            await resource.shutdown()
        elif hasattr(resource, 'cleanup') and asyncio.iscoroutinefunction(resource.cleanup):
            await resource.cleanup()
            
