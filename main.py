import asyncio
import sys
import warnings
import numpy as np
from telegram.ext import ApplicationBuilder, CommandHandler
from sklearn import logger

from exchange.exchange_config import SYMBOLS, close_exchange
from telegrams.constants import BOT_TOKEN
from telegrams.handlers import help_command, show_symbols, start, status

if not hasattr(np, 'NaN'):
    np.NaN = np.nan

# Now import pandas_ta after fixing numpy
try:
    import pandas_ta as ta
except ImportError as e:
    print(f"Error importing pandas_ta: {e}")
    print("Please install with: pip install pandas-ta")
    sys.exit(1)


warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def main() -> None:
    """Run the bot with improved error handling"""
    try:
        if not BOT_TOKEN:
            logger.error("BOT_TOKEN not found. Please set TELEGRAM_BOT_TOKEN environment variable.")
            print("Please set the TELEGRAM_BOT_TOKEN environment variable!")
            return
        
        if not SYMBOLS:
            logger.error("No symbols loaded. Please check symbols.txt file.")
            print("Please create a symbols.txt file with trading pairs!")
            return
        
        logger.info("Starting Telegram Trading Bot...")
        logger.info(f"Loaded {len(SYMBOLS)} symbols for analysis")
        
        application = ApplicationBuilder().token(BOT_TOKEN).build()
        
        # Add command handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("status", status))
        application.add_handler(CommandHandler("symbols", show_symbols))
        application.add_handler(CommandHandler("help", help_command))
        
        logger.info("Bot is ready and polling...")
        print("✅ Bot started successfully! Press Ctrl+C to stop.")
        
        # Run with graceful shutdown
        application.run_polling(
            drop_pending_updates=True,
            allowed_updates=['message'],
            stop_signals=[],  # Handle shutdown manually
        )
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        print("Bot stopped.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Fatal error: {e}")
    finally:
        # Cleanup
        try:
            asyncio.run(close_exchange())
        except Exception as e:
            logger.error(f"Error closing exchange: {e}")
            print(f"Error closing exchange: {e}")
        logger.info("Bot shutdown complete")

if __name__ == '__main__':
    main()