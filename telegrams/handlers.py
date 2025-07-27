import asyncio
from telegram import Update
from telegram.ext import ContextTypes

from exchange.exchange_config import SYMBOLS
from logger_config import logger
from market.main import analyze_market
from .background import _background_analysis
from .constants import ERROR_MESSAGE, WAIT_MESSAGE
from .message_builder import _build_signal_message, _build_status_message, _send_error_message, _send_status_message
from .system_info import _get_system_info, _test_exchange_connection

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """مدیریت دستور /start با ارائه بهترین سیگنال"""
    try:
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        logger.info(f"User {username} ({user_id}) started analysis")
        
        # Send immediate response to user
        await update.message.reply_text(WAIT_MESSAGE)
        
        # Run analysis in background task to avoid blocking the bot
        _ = asyncio.create_task(_background_analysis(update, user_id, username))
        
    except Exception as e:
        logger.error(f"Error in start command: {e}")
        await update.message.reply_text(ERROR_MESSAGE + f"جزئیات خطا: {str(e)}")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command with improved error handling and logging"""
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    
    try:
        logger.info(f"Status command requested by user {username} ({user_id})")
        
        exchange_status, exchange_error = await _test_exchange_connection()
        symbols_count, current_time, python_version = _get_system_info()
        
        message = _build_status_message(exchange_status, exchange_error, symbols_count, current_time, python_version)
        await _send_status_message(update, message, username)
        
    except Exception as e:
        logger.error(f"Critical error in status command for user {username} ({user_id}): {e}", exc_info=True)
        await _send_error_message(update, e)

async def show_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /symbols command"""
    try:
        message = f"📋 *نمادهای تحت نظارت:* ({len(SYMBOLS)} نماد)\n\n"
        
        # Group symbols in rows of 3
        for i in range(0, len(SYMBOLS), 3):
            row_symbols = SYMBOLS[i:i+3]
            message += " | ".join([f"`{symbol}`" for symbol in row_symbols]) + "\n"
        
        message += "\n💡 برای تغییر نمادها، فایل `symbols.txt` را ویرایش کنید.\n"
        message += f"🎯 هر نماد با {len(['RSI', 'MACD', 'Stochastic', 'MFI', 'CCI', 'Williams %R', 'Fibonacci', 'Volume'])} شاخص تکنیکال تحلیل می‌شود."
        
        await update.message.reply_text(message, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in symbols command: {e}")
        await update.message.reply_text("خطایی در نمایش نمادها رخ داد.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command"""
    try:
        message = (
            "🤖 *راهنمای ربات تحلیل تکنیکال پیشرفته*\n\n"
            "این ربات با استفاده از 8+ اندیکاتور تکنیکال، بهترین فرصت‌های معاملاتی را شناسایی می‌کند.\n\n"
            "📋 *دستورات:*\n"
            "`/start` - شروع تحلیل بازار\n"
            "`/status` - نمایش وضعیت ربات\n"
            "`/symbols` - لیست نمادهای تحت نظارت\n"
            "`/help` - نمایش این راهنما\n\n"
            "📊 *اندیکاتورهای تحلیلی:*\n"
            "🔹 **RSI** - شناسایی مناطق فروش/خرید بیش از حد\n"
            "🔹 **MACD** - تشخیص تغییر روند بازار\n"
            "🔹 **Stochastic** - نوسانگر قدرتمند برای ورود/خروج\n"
            "🔹 **MFI** - تحلیل جریان پول هوشمند\n"
            "🔹 **CCI** - شاخص قدرت روند\n"
            "🔹 **Williams %R** - تایید سیگنال‌های اصلی\n"
            "🔹 **Fibonacci** - سطوح حمایت و مقاومت\n"
            "🔹 **Volume Analysis** - تحلیل حجم معاملات\n\n"
            "🎯 *ویژگی‌های خاص:*\n"
            "• سیستم امتیازدهی پیشرفته (0-100)\n"
            "• تحلیل چندگانه شاخص‌ها\n"
            "• محاسبه نسبت سود به ضرر\n"
            "• شناسایی قدرت روند\n"
            "• تشخیص سطوح فیبوناچی\n"
            "• مدیریت ریسک هوشمند\n\n"
            "⚠️ *هشدار مهم:*\n"
            "این سیگنال‌ها صرفاً جهت اطلاع‌رسانی هستند و توصیه سرمایه‌گذاری محسوب نمی‌شوند. "
            "لطفاً قبل از هر معامله، تحلیل‌های خود را انجام دهید.\n\n"
            "💰 *مدیریت ریسک:*\n"
            "• حداکثر 2-3% از سرمایه را ریسک کنید\n"
            "• همیشه Stop Loss تعیین کنید\n"
            "• از سیگنال‌های بالای 60 امتیاز استفاده کنید"
        )
        
        await update.message.reply_text(message, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in help command: {e}")
        await update.message.reply_text("خطایی در نمایش راهنما رخ داد.")
