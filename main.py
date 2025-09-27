import asyncio
import platform
import signal
import sys
import traceback
import warnings
from typing import Set

import redis.asyncio as redis
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update

from module.config import Config, ConfigManager
from module.data_sources import (BinanceFetcher, CoinDeskFetcher,
                                GlassnodeFetcher, MarketIndicesFetcher,
                                NewsFetcher)
from module.logger_config import logger
from module.lstm import LSTMModelManager
from module.market import MarketDataProvider
from module.signals import MultiTimeframeAnalyzer, SignalGenerator, SignalRanking
from module.tasks import TaskServiceContainer
from module.telegram import TelegramBotHandler, TradingBotService

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.runtime_version")
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


background_tasks: Set[asyncio.Task] = set()

async def main():
    stop_event = asyncio.Event()
    bot_instance = None
    scheduler = None
    redis_client = None
    application = None
    lstm_manager = None
    market_data_provider = None
    task_container = None

    def signal_handler():
        if not stop_event.is_set():
            logger.info("Shutdown signal received")
            stop_event.set()

    loop = asyncio.get_running_loop()
    if platform.system() != "Windows":
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

    config_manager = ConfigManager()

    try:
        redis_client = redis.Redis.from_url( f"rediss://default:{Config.REDIS_TOKEN}@{Config.REDIS_HOST}:{Config.REDIS_PORT}",
                                            decode_responses=True)
        await redis_client.ping()
        logger.info("Redis connection successful.")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}. Caching will be disabled.")
        redis_client = None

    try:
        bot_token = Config.TELEGRAM_BOT_TOKEN
        if not bot_token:
            raise ValueError("Bot token is required")

        binance_fetcher = BinanceFetcher(redis_client=redis_client)
        coindesk_fetcher = CoinDeskFetcher(api_key=Config.COINDESK_API_KEY, redis_client=redis_client) if Config.COINDESK_API_KEY else None
        glassnode_fetcher = GlassnodeFetcher(api_key=Config.GLASSNODE_API_KEY, redis_client=redis_client) if Config.GLASSNODE_API_KEY else None
        
        market_data_provider = MarketDataProvider(
            redis_client=redis_client, 
            coindesk_fetcher=coindesk_fetcher,
            binance_fetcher=binance_fetcher
        )
        
        news_fetcher = NewsFetcher(Config.CRYPTOPANIC_KEY, coindesk_fetcher=coindesk_fetcher, redis_client=redis_client) if Config.CRYPTOPANIC_KEY else None
        market_indices_fetcher = MarketIndicesFetcher(
            alpha_vantage_key=Config.ALPHA_VANTAGE_KEY,
            coingecko_key=Config.COINGECKO_KEY,
            glassnode_fetcher=glassnode_fetcher,
            redis_client=redis_client
        )
        
        lstm_manager = LSTMModelManager(model_path='MLM')
        await lstm_manager.initialize_models()

        signal_generator = SignalGenerator(
            market_data_provider=market_data_provider,
            news_fetcher=news_fetcher,
            market_indices_fetcher=market_indices_fetcher,
            lstm_model_manager=lstm_manager,
            multi_tf_analyzer=None,
            config=config_manager.config,
        )

        multi_tf_analyzer = MultiTimeframeAnalyzer(market_data_provider, signal_generator.indicators)
        signal_generator.multi_tf_analyzer = multi_tf_analyzer

        trading_service = TradingBotService(
            config=config_manager,
            market_data_provider=market_data_provider,
            signal_generator=signal_generator,
            signal_ranking=SignalRanking(),
        )

        bot_instance = TelegramBotHandler(
            bot_token=bot_token,
            config_manager=config_manager,
            trading_service=trading_service,
        )

        await bot_instance.initialize()
        application = bot_instance.create_application(background_tasks)
        
        trading_service.set_telegram_app(application)


        scheduler = AsyncIOScheduler()
        if config_manager.get("enable_scheduled_analysis", True):
            schedule_cron = config_manager.get("schedule_hour", "*/4")
            scheduler.add_job(
                bot_instance.run_scheduled_analysis, "cron", hour=schedule_cron
            )
            scheduler.start()
            logger.info(f"APScheduler started for periodic analysis with schedule: 'hour={schedule_cron}'.")

        logger.info("Bot is ready and waiting for commands...")
        await application.initialize()
        await application.start()
        await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        await stop_event.wait()

    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped by user")
    except asyncio.CancelledError:
        logger.info("Main task cancelled, shutting down.")
    except Exception as e:
        logger.error(f"Bot crashed with error: {e}\n{traceback.format_exc()}")
    finally:
        logger.info("Starting cleanup...")

        if scheduler and scheduler.running:
            scheduler.shutdown(wait=False)
        
        if application and application.updater and application.updater.running:
            await application.updater.stop()
        
        if application:
            await application.stop()
            await application.shutdown()

        logger.info(f"Cancelling {len(background_tasks)} background tasks.")
        for task in list(background_tasks):
            task.cancel()
        
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        
        if bot_instance:
            await bot_instance.cleanup()
        
        if market_data_provider:
            await market_data_provider.close()
        
        if lstm_manager:
            lstm_manager.cleanup()
        
        try:
            task_container = await TaskServiceContainer.instance()
            if task_container:
                await task_container.cleanup()
        except (RuntimeError, NameError):
            pass

        if redis_client:
            await redis_client.aclose()
            
        logger.info("Bot shutdown completed successfully")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Application terminated by user.")