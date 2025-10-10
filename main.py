import asyncio
import platform
import signal
import sys
import traceback
import warnings
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update
from telegram.ext import Application

from module.config import ConfigManager
from module.logger_config import logger
from module.tasks import TaskServiceContainer
from module.telegram import TelegramBotHandler

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.runtime_version")
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


async def initialize_bot(task_container: TaskServiceContainer) -> Optional[TelegramBotHandler]:
    max_retries = 3
    retry_delay = 5
    bot_handler: Optional[TelegramBotHandler] = None

    for attempt in range(max_retries):
        try:
            bot_handler = TelegramBotHandler(
                bot_token=task_container.bot_token,
                config_manager=task_container.config_manager,
                trading_service=task_container.trading_service,
                background_tasks=task_container.background_tasks,
            )
            application = bot_handler.application
            await application.initialize()
            await application.start()
            await application.updater.start_polling(
                allowed_updates=Update.ALL_TYPES, drop_pending_updates=True
            )
            logger.info("Telegram bot initialized and polling started successfully.")
            return bot_handler
        except Exception as e:
            if bot_handler and bot_handler.application:
                await bot_handler.application.stop()
                await bot_handler.application.shutdown()
            
            if attempt < max_retries - 1:
                logger.warning(f"Failed to initialize bot (attempt {attempt + 1}/{max_retries}): {e}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"Failed to initialize bot after {max_retries} attempts: {e}")
                return None


def initialize_scheduler(config_manager: ConfigManager, bot_handler: TelegramBotHandler) -> Optional[AsyncIOScheduler]:
    if not config_manager.get("enable_scheduled_analysis", True):
        return None

    scheduler = AsyncIOScheduler()
    schedule_cron = config_manager.get("schedule_hour", "*/4")
    
    try:
        scheduler.add_job(bot_handler.run_scheduled_analysis, "cron", hour=schedule_cron)
        scheduler.start()
        logger.info(f"APScheduler started for long-term analysis with schedule: 'hour={schedule_cron}'.")
        return scheduler
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
        return None


async def main():
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    
    # Capture the main task to cancel it gracefully
    main_task = asyncio.current_task()

    def signal_handler():
        logger.info("Shutdown signal received")
        if not stop_event.is_set():
            stop_event.set()
            # Gently cancel the main task to allow the finally block to run
            if main_task:
                main_task.cancel()

    if platform.system() != "Windows":
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

    task_container = None
    bot_handler = None
    scheduler = None
    
    try:
        task_container = await TaskServiceContainer.instance()
        bot_handler = await initialize_bot(task_container)

        if not bot_handler:
            raise RuntimeError("Bot could not be initialized. Exiting.")

        scheduler = initialize_scheduler(task_container.config_manager, bot_handler)
        
        logger.info("Bot is ready and waiting for commands...")
        await stop_event.wait()

    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped by user or system.")
    except RuntimeError as e:
        logger.error(str(e))
    except asyncio.CancelledError:
        # This is expected on shutdown, so we log it as info.
        logger.info("Main task cancelled, initiating shutdown.")
    except Exception as e:
        logger.critical(f"Bot crashed with unhandled error: {e}\n{traceback.format_exc()}")
    finally:
        logger.info("Starting graceful shutdown...")

        if scheduler and scheduler.running:
            try:
                scheduler.shutdown(wait=False)
                logger.info("Scheduler shut down.")
            except Exception as e:
                logger.warning(f"Error shutting down scheduler: {e}")
        
        if bot_handler and bot_handler.application and bot_handler.application.updater and bot_handler.application.updater.running:
            try:
                await bot_handler.application.updater.stop()
                logger.info("Updater stopped polling.")
            except Exception as e:
                logger.warning(f"Error stopping updater: {e}")
        
        if bot_handler and bot_handler.application:
            try:
                await bot_handler.application.stop()
                await bot_handler.application.shutdown()
                logger.info("Telegram application shut down.")
            except Exception as e:
                logger.warning(f"Error shutting down application: {e}")
        
        if task_container:
            await task_container.cleanup()
            
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Application terminated by user.")