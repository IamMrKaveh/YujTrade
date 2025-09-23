import asyncio
import platform
import signal
import sys
import traceback
import warnings

import redis.asyncio as redis
import sentry_sdk
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from prometheus_client import start_http_server
from telegram import Update

from module.config import Config, ConfigManager
from module.logger_config import logger
from module.monitoring import ERRORS_TOTAL, set_app_info
from module.sentiment import (
    CoinGeckoFetcher,
    ExchangeManager,
    MarketIndicesFetcher,
    NewsFetcher,
    OnChainFetcher,
)
from module.signals import MultiTimeframeAnalyzer, SignalGenerator, SignalRanking
from module.telegram import TelegramBotHandler, TradingBotService
from module.lstm import LSTMModelManager

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.runtime_version")


def setup_sentry(dsn: str, app_version: str):
    if dsn:
        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=1.0,
            profiles_sample_rate=1.0,
            release=f"trading-bot@{app_version}",
            environment="production",
        )
        logger.info("Sentry initialized successfully.")
    else:
        logger.warning("Sentry DSN not found. Sentry is disabled.")


def start_prometheus_server(port: int):
    try:
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start Prometheus server: {e}")
        ERRORS_TOTAL.labels(module="main", function="start_prometheus_server").inc()


async def main():
    stop_event = asyncio.Event()
    bot_instance = None
    scheduler = None
    redis_client = None
    application = None
    lstm_manager = None
    exchange_manager = None

    def signal_handler():
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    if platform.system() != "Windows":
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

    config_manager = ConfigManager()
    setup_sentry(Config.SENTRY_DSN, config_manager.get("app_version"))
    set_app_info("TradingBot", config_manager.get("app_version"))
    start_prometheus_server(Config.PROMETHEUS_PORT)

    try:
        redis_client = redis.Redis(host=Config.REDIS_HOST, port=Config.REDIS_PORT, decode_responses=True)
        await redis_client.ping()
        logger.info("Redis connection successful.")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}. Caching will be disabled.")
        ERRORS_TOTAL.labels(module="main", function="redis_connection").inc()
        redis_client = None

    try:
        bot_token = Config.TELEGRAM_BOT_TOKEN
        if not bot_token:
            raise ValueError("Bot token is required")

        exchange_manager = ExchangeManager(redis_client=redis_client)
        await exchange_manager.init_database()
        
        news_fetcher = NewsFetcher(Config.CRYPTOPANIC_KEY, redis_client=redis_client) if Config.CRYPTOPANIC_KEY else None
        coingecko_fetcher = CoinGeckoFetcher(redis_client=redis_client)
        market_indices_fetcher = MarketIndicesFetcher(Config.ALPHA_VANTAGE_KEY, redis_client=redis_client)
        onchain_fetcher = OnChainFetcher(glassnode_api_key=Config.GLASSNODE_API_KEY) if Config.GLASSNODE_API_KEY else None
        
        lstm_manager = LSTMModelManager(model_path='lstm-model')
        await lstm_manager.initialize_models()

        signal_generator = SignalGenerator(
            exchange_manager=exchange_manager,
            news_fetcher=news_fetcher,
            onchain_fetcher=onchain_fetcher,
            coingecko_fetcher=coingecko_fetcher,
            market_indices_fetcher=market_indices_fetcher,
            lstm_model_manager=lstm_manager,
            multi_tf_analyzer=None,
            config=config_manager.config,
        )

        multi_tf_analyzer = MultiTimeframeAnalyzer(exchange_manager, signal_generator.indicators)
        signal_generator.multi_tf_analyzer = multi_tf_analyzer

        trading_service = TradingBotService(
            config=config_manager,
            exchange_manager=exchange_manager,
            signal_generator=signal_generator,
            signal_ranking=SignalRanking(),
        )

        bot_instance = TelegramBotHandler(
            bot_token=bot_token,
            config_manager=config_manager,
            trading_service=trading_service,
        )

        await bot_instance.initialize()
        application = bot_instance.create_application()
        
        # Share the application instance with the trading service for task callbacks
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
    except Exception as e:
        logger.error(f"Bot crashed with error: {e}\n{traceback.format_exc()}")
        sentry_sdk.capture_exception(e)
        ERRORS_TOTAL.labels(module="main", function="main_loop").inc()
    finally:
        logger.info("Starting cleanup...")
        if scheduler and scheduler.running:
            scheduler.shutdown(wait=False)
        if application:
            await application.updater.stop()
            await application.stop()
            await application.shutdown()
        if bot_instance:
            await bot_instance.cleanup()
        if exchange_manager:
            await exchange_manager.close()
        if lstm_manager:
            lstm_manager.cleanup()
        if redis_client:
            await redis_client.close()
        logger.info("Bot shutdown completed successfully")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())