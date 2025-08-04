# import asyncio
# import warnings
# from telegram.ext import ApplicationBuilder, CommandHandler
# from logger_config import logger

# from services.coinex_api import close_exchange
# from config.constants import BOT_TOKEN, SYMBOLS
# from handlers.bot_handlers import TelegramBotHandlers


# warnings.filterwarnings('ignore', category=UserWarning)
# warnings.filterwarnings('ignore', category=FutureWarning)

# def main() -> None:
#     """Run the bot with improved error handling"""
#     try:
#         if not BOT_TOKEN:
#             logger.error("BOT_TOKEN not found. Please set TELEGRAM_BOT_TOKEN environment variable.")
#             print("Please set the TELEGRAM_BOT_TOKEN environment variable!")
#             return
        
#         if not SYMBOLS:
#             logger.error("No symbols loaded. Please check symbols.txt file.")
#             print("Please create a symbols.txt file with trading pairs!")
#             return
        
#         logger.info("Starting Telegram Trading Bot...")
#         logger.info(f"Loaded {len(SYMBOLS)} symbols for analysis")
        
#         application = ApplicationBuilder().token(BOT_TOKEN).build()
        
#         # Add command handlers
#         application.add_handler(CommandHandler("start", TelegramBotHandlers.start))
#         # application.add_handler(CommandHandler("start", TelegramBotHandlers.start_command))
#         # application.add_handler(CommandHandler("status", status))
#         # application.add_handler(CommandHandler("symbols", show_symbols))
#         # application.add_handler(CommandHandler("help", help_command))
        
#         logger.info("Bot is ready and polling...")
#         print("✅ Bot started successfully! Press Ctrl+C to stop.")
        
#         # Run with graceful shutdown
#         application.run_polling(
#             drop_pending_updates=True,
#             allowed_updates=['message'],
#             stop_signals=[],  # Handle shutdown manually
#         )
        
#     except KeyboardInterrupt:
#         logger.info("Bot stopped by user")
#         print("Bot stopped.")
#     except Exception as e:
#         logger.error(f"Fatal error: {e}")
#         print(f"Fatal error: {e}")
#     finally:
#         # Cleanup
#         try:
#             asyncio.run(close_exchange())
#         except Exception as e:
#             logger.error(f"Error closing exchange: {e}")
#             print(f"Error closing exchange: {e}")
#         logger.info("Bot shutdown complete")

# if __name__ == '__main__':
#     main()
























import asyncio
import pandas as pd
import ccxt.async_support as ccxt
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# --- وارد کردن ماژول‌های پروژه ---
from logger_config import logger
from indicators.indicator_management import IndicatorConfig

from services.coinex_api import close_exchange
from config.constants import BOT_TOKEN, SYMBOLS

TIME_FRAMES = [
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "2d",
    "3d",
    "5d",
    "1w",
    "2w",
    "1M",
    "3M",
]

async def fetch_ohlcv(exchange, symbol, timeframe, limit=1000):
    """
    دریافت کندل‌های OHLCV از صرافی و تبدیل آن به DataFrame.
    """
    try:
        # 3. تلاش برای دریافت 1000 کندل (اگر موجود نباشد، ccxt حداقل ممکن را برمی‌گرداند)
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv:
            logger.warning(f"No data returned for {symbol} on {timeframe}")
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"Error fetching ohlcv for {symbol} on {timeframe}: {e}")
        return pd.DataFrame()

async def find_best_signal_for_timeframe(exchange, timeframe):
    """
    بهترین سیگنال را برای یک تایم فریم مشخص در میان تمام نمادها پیدا می‌کند.
    """
    best_signal = None
    max_profit = 0.0

    for symbol in SYMBOLS:
        logger.info(f"Processing {symbol} for timeframe {timeframe}...")
        df = await fetch_ohlcv(exchange, symbol, timeframe)

        if df.empty or len(df) < 50: # حداقل کندل برای محاسبات
            logger.warning(f"Not enough data for {symbol} on {timeframe}. Skipping.")
            continue

        # 4. پاس دادن داده‌ها برای پردازش اندیکاتورها
        analysis_result = IndicatorConfig.calculate_indicators(df)

        if 'signals' in analysis_result and analysis_result['signals']:
            signals = analysis_result['signals']
            
            # 5. بهترین سیگنال را بر اساس بیشترین سود دریافت کن
            # بررسی سیگنال های خرید
            for signal in signals.get('buy', []):
                profit = (signal['exit'] / signal['entry']) - 1
                if profit > max_profit:
                    max_profit = profit
                    best_signal = {
                        'symbol': symbol,
                        'type': 'خرید (Buy)',
                        'profit': profit * 100,
                        'entry': signal['entry'],
                        'exit': signal['exit'],
                        'indicator': signal.get('indicator', 'N/A')
                    }

            # بررسی سیگنال های فروش
            for signal in signals.get('sell', []):
                # برای فروش، سود زمانی است که قیمت ورود بالاتر از خروج باشد
                profit = (signal['entry'] / signal['exit']) - 1
                if profit > max_profit:
                    max_profit = profit
                    best_signal = {
                        'symbol': symbol,
                        'type': 'فروش (Sell)',
                        'profit': profit * 100,
                        'entry': signal['entry'],
                        'exit': signal['exit'],
                        'indicator': signal.get('indicator', 'N/A')
                    }
    
    return best_signal

# 1. تعریف دستور start برای تلگرام
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """دستور /start را مدیریت می‌کند."""
    await update.message.reply_text('در حال بررسی سیگنال‌ها... لطفاً صبر کنید. 🧐')
    
    exchange = ccxt.binance() # یا هر صرافی دیگری که پشتیبانی می‌شود

    try:
        for timeframe in TIME_FRAMES:
            await update.message.reply_text(f'🔍 در حال جستجوی بهترین سیگنال برای تایم فریم {timeframe}...')
            
            best_signal = await find_best_signal_for_timeframe(exchange, timeframe)

            # 6. نمایش بهترین سیگنال در یک پیام ساده
            if best_signal:
                message = (
                    f"🚀 **بهترین سیگنال برای تایم فریم {timeframe}**\n\n"
                    f"📈 **نماد:** `{best_signal['symbol']}`\n"
                    f"📊 **نوع سیگنال:** {best_signal['type']}\n"
                    f"💰 **سود احتمالی:** `{best_signal['profit']:.2f}%`\n"
                    f"🟢 **نقطه ورود:** `{best_signal['entry']}`\n"
                    f"🔴 **نقطه خروج:** `{best_signal['exit']}`\n"
                    f"⚙️ **بر اساس:** `{best_signal['indicator']}`"
                )
                await update.message.reply_text(message, parse_mode='Markdown')
            else:
                message = f"⚠️ برای تایم فریم `{timeframe}` هیچ سیگنال سودآوری یافت نشد."
                await update.message.reply_text(message, parse_mode='Markdown')
                
    except Exception as e:
        logger.error(f"An error occurred during the start command: {e}")
        await update.message.reply_text(f"خطایی رخ داد: {e}")
    finally:
        await exchange.close()

def main() -> None:
    """ربات را اجرا می‌کند."""
    logger.info("Starting bot...")
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))

    application.run_polling()

if __name__ == "__main__":
    main()