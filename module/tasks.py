import asyncio
from typing import Optional

import redis.asyncio as redis
from telegram.ext import Application

from module.config import Config, ConfigManager
from module.data_sources import (BinanceFetcher, CoinDeskFetcher,
                                 GlassnodeFetcher, MarketIndicesFetcher,
                                 NewsFetcher)
from module.logger_config import logger
from module.lstm import LSTMModelManager
from module.market import MarketDataProvider
from module.signals import MultiTimeframeAnalyzer, SignalGenerator, SignalRanking
from module.telegram import TradingBotService


class TaskServiceContainer:
    _instance = None
    _lock = asyncio.Lock()

    def __init__(self):
        raise RuntimeError("Call instance() instead")

    @classmethod
    async def instance(cls):
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.__new__(cls)
                    await cls._instance._initialize()
        return cls._instance

    async def _initialize(self):
        logger.info("Initializing TaskServiceContainer...")
        self.config_manager = ConfigManager()
        
        try:
            self.redis_client = redis.Redis.from_url(f"rediss://default:{Config.REDIS_TOKEN}@{Config.REDIS_HOST}:{Config.REDIS_PORT}", decode_responses=True)
            await self.redis_client.ping()
            logger.info("Task Service: Redis connection successful.")
        except Exception as e:
            logger.error(f"Task Service: Redis connection failed: {e}. Caching will be disabled.")
            self.redis_client = None

        binance_fetcher = BinanceFetcher(redis_client=self.redis_client)
        coindesk_fetcher = CoinDeskFetcher(api_key=Config.COINDESK_API_KEY, redis_client=self.redis_client) if Config.COINDESK_API_KEY else None
        glassnode_fetcher = GlassnodeFetcher(api_key=Config.GLASSNODE_API_KEY, redis_client=self.redis_client) if Config.GLASSNODE_API_KEY else None
        
        self.market_data_provider = MarketDataProvider(
            redis_client=self.redis_client,
            coindesk_fetcher=coindesk_fetcher,
            binance_fetcher=binance_fetcher
        )
        
        self.news_fetcher = NewsFetcher(Config.CRYPTOPANIC_KEY, coindesk_fetcher=coindesk_fetcher, redis_client=self.redis_client) if Config.CRYPTOPANIC_KEY else None
        self.market_indices_fetcher = MarketIndicesFetcher(
            alpha_vantage_key=Config.ALPHA_VANTAGE_KEY,
            coingecko_key=Config.COINGECKO_KEY,
            glassnode_fetcher=glassnode_fetcher,
            redis_client=self.redis_client
        )
        
        self.lstm_manager = LSTMModelManager(model_path='lstm-model')
        await self.lstm_manager.initialize_models()
        
        self.signal_generator = SignalGenerator(
            market_data_provider=self.market_data_provider,
            news_fetcher=self.news_fetcher,
            market_indices_fetcher=self.market_indices_fetcher,
            lstm_model_manager=self.lstm_manager,
            multi_tf_analyzer=None,
            config=self.config_manager.config,
        )
        
        multi_tf_analyzer = MultiTimeframeAnalyzer(self.market_data_provider, self.signal_generator.indicators)
        self.signal_generator.multi_tf_analyzer = multi_tf_analyzer
        
        self.trading_service = TradingBotService(
            config=self.config_manager,
            market_data_provider=self.market_data_provider,
            signal_generator=self.signal_generator,
            signal_ranking=SignalRanking(),
        )
        
        app_builder = Application.builder().token(Config.TELEGRAM_BOT_TOKEN)
        app_builder.connect_timeout(30).read_timeout(30).write_timeout(30)
        self.telegram_app = app_builder.build()
        self.trading_service.set_telegram_app(self.telegram_app)
        
        logger.info("TaskServiceContainer initialized.")

    async def cleanup(self):
        logger.info("Cleaning up TaskServiceContainer...")
        if self.lstm_manager:
            self.lstm_manager.cleanup()
        if self.redis_client:
            await self.redis_client.aclose()
        if self.market_data_provider:
            await self.market_data_provider.close()
        logger.info("TaskServiceContainer cleaned up.")


async def run_full_analysis_task(chat_id: int, message_id: int):
    try:
        logger.info(f"Task 'run_full_analysis_task' started for chat_id: {chat_id}")
        container = await TaskServiceContainer.instance()
        await container.trading_service.run_comprehensive_analysis_task(chat_id, message_id)
        logger.info(f"Task 'run_full_analysis_task' finished for chat_id: {chat_id}")
    except Exception as e:
        logger.error(f"Error in run_full_analysis_task: {e}", exc_info=True)


async def run_quick_scan_task(chat_id: int, message_id: int):
    try:
        logger.info(f"Task 'run_quick_scan_task' started for chat_id: {chat_id}")
        container = await TaskServiceContainer.instance()
        await container.trading_service.run_find_best_signals_task("1m", chat_id, message_id)
        logger.info(f"Task 'run_quick_scan_task' finished for chat_id: {chat_id}")
    except Exception as e:
        logger.error(f"Error in run_quick_scan_task: {e}", exc_info=True)