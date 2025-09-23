import asyncio
import time
from typing import Optional

from celery.signals import task_failure, task_postrun, task_prerun
from telegram.ext import Application

from module.celery_app import celery_app
from module.config import Config, ConfigManager
from module.logger_config import logger
from module.monitoring import CELERY_TASKS_TOTAL, CELERY_TASK_DURATION_SECONDS
from module.sentiment import (
    CoinGeckoFetcher,
    ExchangeManager,
    MarketIndicesFetcher,
    NewsFetcher,
    OnChainFetcher,
)
from module.signals import MultiTimeframeAnalyzer, SignalGenerator, SignalRanking
from module.telegram import TradingBotService
from module.lstm import LSTMModelManager


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
        self.loop = asyncio.get_running_loop()
        self.config_manager = ConfigManager()
        
        # Initialize Redis client once
        try:
            self.redis_client = redis.Redis(host=Config.REDIS_HOST, port=Config.REDIS_PORT, decode_responses=True)
            await self.redis_client.ping()
            logger.info("Task Service: Redis connection successful.")
        except Exception as e:
            logger.error(f"Task Service: Redis connection failed: {e}. Caching will be disabled.")
            self.redis_client = None

        self.exchange_manager = ExchangeManager(redis_client=self.redis_client)
        self.news_fetcher = NewsFetcher(Config.CRYPTOPANIC_KEY, redis_client=self.redis_client) if Config.CRYPTOPANIC_KEY else None
        self.coingecko_fetcher = CoinGeckoFetcher(redis_client=self.redis_client)
        self.market_indices_fetcher = MarketIndicesFetcher(Config.ALPHA_VANTAGE_KEY, redis_client=self.redis_client)
        self.onchain_fetcher = OnChainFetcher(glassnode_api_key=Config.GLASSNODE_API_KEY) if Config.GLASSNODE_API_KEY else None
        
        self.lstm_manager = LSTMModelManager(model_path='lstm-model')
        await self.lstm_manager.initialize_models()
        
        self.signal_generator = SignalGenerator(
            exchange_manager=self.exchange_manager,
            news_fetcher=self.news_fetcher,
            onchain_fetcher=self.onchain_fetcher,
            coingecko_fetcher=self.coingecko_fetcher,
            market_indices_fetcher=self.market_indices_fetcher,
            lstm_model_manager=self.lstm_manager,
            multi_tf_analyzer=None,
            config=self.config_manager.config,
        )
        
        multi_tf_analyzer = MultiTimeframeAnalyzer(self.exchange_manager, self.signal_generator.indicators)
        self.signal_generator.multi_tf_analyzer = multi_tf_analyzer
        
        self.trading_service = TradingBotService(
            config=self.config_manager,
            exchange_manager=self.exchange_manager,
            signal_generator=self.signal_generator,
            signal_ranking=SignalRanking(),
        )
        
        # Telegram app is built and set from the main process
        self.telegram_app = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
        self.trading_service.set_telegram_app(self.telegram_app)
        
        logger.info("TaskServiceContainer initialized.")

    async def cleanup(self):
        if self.lstm_manager:
            self.lstm_manager.cleanup()
        if self.redis_client:
            await self.redis_client.close()
        logger.info("TaskServiceContainer cleaned up.")


async def _get_services():
    container = await TaskServiceContainer.instance()
    return container.trading_service, container.loop


@celery_app.task(name="tasks.run_full_analysis_task")
def run_full_analysis_task(chat_id: int, message_id: int):
    logger.info(f"Celery task 'run_full_analysis_task' started for chat_id: {chat_id}")
    trading_service, loop = asyncio.run(_get_services())
    loop.run_until_complete(trading_service.run_comprehensive_analysis_task(chat_id, message_id))
    logger.info(f"Celery task 'run_full_analysis_task' finished for chat_id: {chat_id}")


@celery_app.task(name="tasks.run_quick_scan_task")
def run_quick_scan_task(chat_id: int, message_id: int):
    logger.info(f"Celery task 'run_quick_scan_task' started for chat_id: {chat_id}")
    trading_service, loop = asyncio.run(_get_services())
    loop.run_until_complete(trading_service.run_find_best_signals_task("1h", chat_id, message_id))
    logger.info(f"Celery task 'run_quick_scan_task' finished for chat_id: {chat_id}")


@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, **kwargs):
    # Using a simple time recording method as backend might not be immediately available
    task.start_time = time.time()


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, retval=None, state=None, **kwargs):
    CELERY_TASKS_TOTAL.labels(task_name=task.name, status=state).inc()
    if hasattr(task, 'start_time'):
        duration = time.time() - task.start_time
        CELERY_TASK_DURATION_SECONDS.labels(task_name=task.name).observe(duration)


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, **kwargs):
    logger.error(f"Celery task {task_id} ({sender.name}) failed: {exception}")