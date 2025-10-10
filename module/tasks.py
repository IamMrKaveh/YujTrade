import asyncio
from typing import Set

import redis.asyncio as redis

from .background_tasks import BackgroundTaskManager
from .config import Config, ConfigManager
from .logger_config import logger
from .market import MarketDataProvider
from .trading_service import TradingService


class TaskServiceContainer:
    _instance = None
    _lock = asyncio.Lock()

    def __init__(self):
        raise RuntimeError("Call instance() instead")

    @classmethod
    async def instance(cls):
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls.__new__(cls)
                await cls._instance._initialize()
        return cls._instance

    async def _initialize(self):
        logger.info("Initializing TaskServiceContainer...")
        self.config_manager = ConfigManager()
        self.bot_token = Config.TELEGRAM_BOT_TOKEN
        self.background_tasks = BackgroundTaskManager()
        
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is not configured.")

        try:
            redis_url = f"redis://default:{Config.REDIS_TOKEN}@{Config.REDIS_HOST}:{Config.REDIS_PORT}"
            
            self.redis_client = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                max_connections=20
            )
            await self.redis_client.ping()
            logger.info("Task Service: Redis connection pool created successfully.")
        except Exception as e:
            logger.error(f"Task Service: Redis connection failed: {e}. Caching will be disabled.")
            self.redis_client = None

        shared_session = None

        from module.data.binance import BinanceFetcher
        binance_fetcher = BinanceFetcher(
            redis_client=self.redis_client,
            session=shared_session
        )

        from module.data.marketindices import MarketIndicesFetcher
        market_indices_fetcher = MarketIndicesFetcher(
            redis_client=self.redis_client,
            session=shared_session
        )

        self.market_data_provider = MarketDataProvider(
            redis_client=self.redis_client,
            binance_fetcher=binance_fetcher,
            market_indices_fetcher=market_indices_fetcher,
            session=shared_session
        )
        
        self.trading_service = TradingService(
            market_data_provider=self.market_data_provider,
            config_manager=self.config_manager
        )
        
        logger.info("TaskServiceContainer initialized successfully.")
    
    async def cleanup(self):
        logger.info("Cleaning up TaskServiceContainer...")
        
        if self.background_tasks:
            await self.background_tasks.cancel_all()

        cleanup_tasks = []
        
        if hasattr(self, 'trading_service') and self.trading_service:
            cleanup_tasks.append(self.trading_service.cleanup())
        
        if hasattr(self, 'redis_client') and self.redis_client:
            async def close_redis():
                try:
                    await self.redis_client.aclose()
                    if hasattr(self.redis_client, 'connection_pool'):
                        await self.redis_client.connection_pool.disconnect()
                except Exception as e:
                    logger.warning(f"Error closing Redis: {e}")
            
            cleanup_tasks.append(close_redis())
        
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        logger.info("TaskServiceContainer cleaned up successfully.")


async def run_full_analysis_task(chat_id: int, message_id: int):
    try:
        logger.info(f"Task 'run_full_analysis_task' started for chat_id: {chat_id}")
        container = await TaskServiceContainer.instance()
        signals = await container.trading_service.run_analysis_for_all_symbols()
        logger.info(f"Task 'run_full_analysis_task' finished for chat_id: {chat_id}. Generated {len(signals)} signals.")
    except Exception as e:
        logger.error(f"Error in run_full_analysis_task: {e}", exc_info=True)


async def run_quick_scan_task(chat_id: int, message_id: int):
    try:
        logger.info(f"Task 'run_quick_scan_task' started for chat_id: {chat_id}")
        container = await TaskServiceContainer.instance()
        signals = await container.trading_service.run_analysis_for_all_symbols()
        logger.info(f"Task 'run_quick_scan_task' finished for chat_id: {chat_id}. Generated {len(signals)} signals.")
    except Exception as e:
        logger.error(f"Error in run_quick_scan_task: {e}", exc_info=True)