import asyncio
import platform
import signal
import sys
import traceback
import warnings

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update

from module.config import ConfigManager
from module.logger_config import logger
from module.tasks import TaskServiceContainer
from module.telegram import TelegramBotHandler

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.runtime_version")
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

async def main():
    stop_event = asyncio.Event()

    def signal_handler():
        if not stop_event.is_set():
            logger.info("Shutdown signal received")
            stop_event.set()

    loop = asyncio.get_running_loop()
    if platform.system() != "Windows":
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

    task_container = None
    scheduler = None
    application = None
    
    try:
        # Initialize the service container which holds all shared services
        task_container = await TaskServiceContainer.instance()
        config_manager = task_container.config_manager

        if not task_container.trading_service.telegram_app:
            raise RuntimeError("Telegram application failed to initialize within the service container.")
        
        application = task_container.trading_service.telegram_app

        bot_handler = TelegramBotHandler(
            bot_token=task_container.bot_token,
            config_manager=config_manager,
            trading_service=task_container.trading_service,
            background_tasks=task_container.background_tasks
        )
        
        # Register handlers with the application from the container
        bot_handler._register_handlers()

        scheduler = AsyncIOScheduler()
        if config_manager.get("enable_scheduled_analysis", True):
            schedule_cron = config_manager.get("schedule_hour", "*/4")
            scheduler.add_job(
                bot_handler.run_scheduled_analysis, "cron", hour=schedule_cron
            )
            scheduler.start()
            logger.info(f"APScheduler started for periodic analysis with schedule: 'hour={schedule_cron}'.")

        logger.info("Bot is ready and waiting for commands...")
        
        # The application is already built inside the container, just start it
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
        
        # Cleanup is now handled by the TaskServiceContainer
        if task_container:
            await task_container.cleanup()
            
        logger.info("Bot shutdown completed successfully")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Application terminated by user.")