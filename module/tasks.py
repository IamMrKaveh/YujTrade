import asyncio

from celery.signals import task_failure, task_postrun, task_prerun

from celery_app import celery_app
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
from module.telegram import TelegramBotHandler, TradingBotService


@celery_app.task(name="tasks.run_full_analysis_task")
def run_full_analysis_task(chat_id: int, message_id: int):
    """
    Celery task to run the full analysis and send results to Telegram.
    """
    logger.info(f"Celery task started for chat_id: {chat_id}")

    # Re-initialize components needed for the task
    config_manager = ConfigManager()
    exchange_manager = ExchangeManager()
    news_fetcher = NewsFetcher(Config.CRYPTOPANIC_KEY) if Config.CRYPTOPANIC_KEY else None
    coingecko_fetcher = CoinGeckoFetcher()
    market_indices_fetcher = MarketIndicesFetcher(Config.ALPHA_VANTAGE_KEY)
    onchain_fetcher = OnChainFetcher(glassnode_api_key=Config.GLASSNODE_API_KEY)
    multi_tf_analyzer = MultiTimeframeAnalyzer(exchange_manager, {})
    signal_generator = SignalGenerator(
        exchange_manager=exchange_manager,
        news_fetcher=news_fetcher,
        onchain_fetcher=onchain_fetcher,
        coingecko_fetcher=coingecko_fetcher,
        market_indices_fetcher=market_indices_fetcher,
        multi_tf_analyzer=multi_tf_analyzer,
        config=config_manager.config,
    )
    trading_service = TradingBotService(
        config=config_manager,
        exchange_manager=exchange_manager,
        signal_generator=signal_generator,
        signal_ranking=SignalRanking(),
    )
    bot_handler = TelegramBotHandler(
        bot_token=Config.TELEGRAM_BOT_TOKEN,
        config_manager=config_manager,
        trading_service=trading_service,
    )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # This is a blocking call that runs the async code and waits for it to complete.
    loop.run_until_complete(bot_handler.run_full_analysis_from_task(chat_id, message_id))
    logger.info(f"Celery task finished for chat_id: {chat_id}")


@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, **kwargs):
    # Store start time in a global dict or task metadata
    task.app.backend.set(f"start_time:{task_id}", asyncio.run(asyncio.sleep(0, result=True)))


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, retval=None, state=None, **kwargs):
    CELERY_TASKS_TOTAL.labels(task_name=task.name, status=state).inc()
    start_time_bytes = task.app.backend.get(f"start_time:{task_id}")
    if start_time_bytes:
        start_time = float(start_time_bytes.decode())
        duration = asyncio.run(asyncio.sleep(0, result=True)) - start_time
        CELERY_TASK_DURATION_SECONDS.labels(task_name=task.name).observe(duration)
        task.app.backend.delete(f"start_time:{task_id}")


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, **kwargs):
    logger.error(f"Celery task {task_id} ({sender.name}) failed: {exception}")