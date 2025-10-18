import asyncio
from typing import Set

from utils.background_manager import BackgroundTaskManager
from config.settings import ConfigManager, SecretsManager
from config.logger import logger
from data.data_provider import MarketDataProvider
from services.trading_service import TradingService
from utils.resource_manager import ResourceManager


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
        self.bot_token = SecretsManager.TELEGRAM_BOT_TOKEN
        self.background_tasks = BackgroundTaskManager()

        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is not configured.")

        self.resource_manager = ResourceManager()
        await self.resource_manager.get_session()
        await self.resource_manager.get_redis_client()

        self.market_data_provider = MarketDataProvider(
            resource_manager=self.resource_manager, config_manager=self.config_manager
        )
        await self.market_data_provider.initialize()

        self.trading_service = TradingService(
            market_data_provider=self.market_data_provider,
            config_manager=self.config_manager,
            resource_manager=self.resource_manager,
        )

        logger.info("TaskServiceContainer initialized successfully.")

    async def cleanup(self):
        logger.info("Cleaning up TaskServiceContainer...")

        if self.background_tasks:
            await self.background_tasks.cancel_all()

        cleanup_tasks = []

        if hasattr(self, "trading_service") and self.trading_service:
            cleanup_tasks.append(self.trading_service.cleanup())

        if hasattr(self, "market_data_provider") and self.market_data_provider:
            cleanup_tasks.append(self.market_data_provider.close())

        if hasattr(self, "resource_manager") and self.resource_manager:
            cleanup_tasks.append(self.resource_manager.cleanup())

        await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        logger.info("TaskServiceContainer cleaned up successfully.")


async def run_full_analysis_task(chat_id: int, message_id: int):
    try:
        logger.info(f"Task 'run_full_analysis_task' started for chat_id: {chat_id}")
        container = await TaskServiceContainer.instance()
        signals = await container.trading_service.run_analysis_for_all_symbols()
        logger.info(
            f"Task 'run_full_analysis_task' finished for chat_id: {chat_id}. Generated {len(signals)} signals."
        )
    except Exception as e:
        logger.error(f"Error in run_full_analysis_task: {e}", exc_info=True)


async def run_quick_scan_task(chat_id: int, message_id: int):
    try:
        logger.info(f"Task 'run_quick_scan_task' started for chat_id: {chat_id}")
        container = await TaskServiceContainer.instance()

        signals = await container.trading_service.run_analysis_for_all_symbols()
        logger.info(
            f"Task 'run_quick_scan_task' finished for chat_id: {chat_id}. Generated {len(signals)} signals."
        )
    except Exception as e:
        logger.error(f"Error in run_quick_scan_task: {e}", exc_info=True)
