import os
import logging
import asyncio
import warnings
from datetime import datetime
from telegram import Update
from telegram.ext import ContextTypes
from market import analyze_market
import sys

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Now import pandas_ta after fixing numpy
try:
    import pandas_ta as ta
except ImportError as e:
    print(f"Error importing pandas_ta: {e}")
    print("Please install with: pip install pandas-ta==0.3.14b")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Telegram bot token
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'token')

OVER_BUY = "🔴 (خرید بیش از حد)"

OVER_SELL = "🟢 (فروش بیش از حد)"

BALANCED = "🟡 (متعادل)"
NATURAL_ZONE = "🟡 (محدوده طبیعی)"

ASCENDING = "⬆️ (صعودی)"
DESCENDING = "⬇️ (نزولی)"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """مدیریت دستور /start با ارائه بهترین سیگنال"""
    try:
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        logger.info(f"User {username} ({user_id}) started analysis")
        
        await update.message.reply_text(
            "🔍 در حال تحلیل بازار برای یافتن بهترین فرصت معاملاتی...\n"
            "⏳ این کار ممکن است چند دقیقه طول بکشد."
        )
        
        # تنظیم timeout برای کل تحلیل
        try:
            # Check if analyze_market is async or sync
            result = analyze_market()
            if asyncio.iscoroutine(result):
                signals = await asyncio.wait_for(result, timeout=1800)  # حداکثر 30 دقیقه
            else:
                signals = result
        except asyncio.TimeoutError:
            await update.message.reply_text(
                "⏱️ تحلیل بیش از حد زمان برد. لطفا دوباره تلاش کنید."
            )
            return
        
        if signals and len(signals) > 0:
            sig = signals[0]  # بهترین سیگنال
            
            # تعیین emoji و رنگ بر اساس نوع سیگنال
            emoji = '📈' if sig['type'] == 'Long' else '📉'
            type_color = '🟢' if sig['type'] == 'Long' else '🔴'
            
            # محاسبه درصد سود/ضرر
            if sig['type'] == 'Long':
                profit_pct = ((sig['target'] - sig['entry']) / sig['entry']) * 100
                loss_pct = ((sig['entry'] - sig['stop_loss']) / sig['entry']) * 100
            else:
                profit_pct = ((sig['entry'] - sig['target']) / sig['entry']) * 100
                loss_pct = ((sig['stop_loss'] - sig['entry']) / sig['entry']) * 100
            
            # ساختار پیام بهینه‌شده با شاخص های اضافی
            message = "🎯 *بهترین فرصت معاملاتی یافت شده*\n"
            message += f"{'='*30}\n\n"
            
            message += f"{emoji} *{sig['type']} {sig['symbol']}* {type_color}\n"
            message += f"🏆 **امتیاز دقت: {sig['accuracy_score']}/100**\n\n"
            
            message += "📊 **جزئیات معاملاتی:**\n"
            message += f"💰 قیمت ورودی: `{sig['entry']:.6f}`\n"
            message += f"🎯 هدف قیمت: `{sig['target']:.6f}` (+{profit_pct:.1f}%)\n"
            message += f"🛑 حد ضرر: `{sig['stop_loss']:.6f}` (-{loss_pct:.1f}%)\n\n"
            
            message += "📈 **تحلیل تکنیکال پیشرفته:**\n"
            message += f"• RSI: `{sig['rsi']:.1f}` "
            if sig['rsi'] < 30:
                message += OVER_SELL
            elif sig['rsi'] > 70:
                message += OVER_BUY
            else:
                message += BALANCED
            message += "\n"
            
            message += f"• MACD: `{sig['macd']:.6f}` "
            if sig['macd'] > 0:
                message += ASCENDING
            else:
                message += DESCENDING
            message += "\n"
            
            # نمایش شاخص‌های اضافی اگر در سیگنال موجود باشند
            if 'stoch_k' in sig:
                message += f"• Stochastic K: `{sig['stoch_k']:.1f}` "
                if sig['stoch_k'] < 20:
                    message += OVER_SELL
                elif sig['stoch_k'] > 80:
                    message += OVER_BUY
                else:
                    message += BALANCED
                message += "\n"
            
            if 'mfi' in sig:
                message += f"• MFI: `{sig['mfi']:.1f}` "
                if sig['mfi'] < 20:
                    message += "🟢 (جریان پول خروجی قوی)"
                elif sig['mfi'] > 80:
                    message += "🔴 (جریان پول ورودی قوی)"
                else:
                    message += BALANCED
                message += "\n"
            
            if 'cci' in sig:
                message += f"• CCI: `{sig['cci']:.1f}` "
                if sig['cci'] < -100:
                    message += OVER_SELL
                elif sig['cci'] > 100:
                    message += OVER_BUY
                else:
                    message += NATURAL_ZONE
                message += "\n"
            
            if 'williams_r' in sig:
                message += f"• Williams %R: `{sig['williams_r']:.1f}` "
                if sig['williams_r'] < -80:
                    message += OVER_SELL
                elif sig['williams_r'] > -20:
                    message += OVER_BUY
                else:
                    message += BALANCED
                message += "\n"
            
            if 'volume_ratio' in sig:
                message += f"• نسبت حجم: `{sig['volume_ratio']:.1f}x` "
                if sig['volume_ratio'] > 2:
                    message += "🟢 (حجم بالا)"
                elif sig['volume_ratio'] > 1.5:
                    message += "🟡 (حجم متوسط)"
                else:
                    message += "⚪ (حجم پایین)"
                message += "\n"
            
            message += f"• روش تحلیل: `{sig['method']}`\n"
            message += f"• قدرت سیگنال: {'⭐' * sig['strength']} ({sig['strength']}/5)\n"
            
            if 'trend_direction' in sig:
                message += "• جهت ترند: "
                if sig['trend_direction'] > 0:
                    message += ASCENDING
                elif sig['trend_direction'] < 0:
                    message += DESCENDING
                else:
                    message += "🟡 بغل"
                message += f" (قدرت: {abs(sig['trend_direction']):.1f})\n"
            
            # نمایش سطوح فیبوناچی اگر موجود باشند
            if 'fibonacci_levels' in sig and sig['fibonacci_levels']:
                message += "\n🎯 **سطوح فیبوناچی نزدیک:**\n"
                for level in sig['fibonacci_levels']:
                    message += f"• {level}\n"
            
            # نمایش امتیازهای خرید و فروش
            if 'buy_score' in sig and 'sell_score' in sig:
                message += "\n🎯 **امتیاز سیگنال‌ها:**\n"
                message += f"• امتیاز خرید: `{sig['buy_score']}`\n"
                message += f"• امتیاز فروش: `{sig['sell_score']}`\n"
                message += f"• برتری: {sig['buy_score'] - sig['sell_score']:+d} به نفع {'خرید' if sig['buy_score'] > sig['sell_score'] else 'فروش'}\n"
            
            message += f"\n⏰ زمان تولید سیگنال: `{sig['timestamp']}`\n"
            
        else:
            message = (
                "❌ متأسفانه در حال حاضر هیچ سیگنال معاملاتی با دقت بالا یافت نشد.\n\n"
                "🔍 **دلایل احتمالی:**\n"
                "• بازار در حالت تثبیت قرار دارد\n"
                "• شرایط تکنیکال مناسب معاملاتی وجود ندارد\n"
                "• همه سیگنال‌ها دارای ریسک بالا هستند\n\n"
                "💡 **پیشنهاد:**\n"
                "• 30-60 دقیقه دیگر مجدداً تلاش کنید\n"
                "• در انتظار شکل‌گیری الگوهای تکنیکال باشید\n"
                "• از معاملات پر ریسک خودداری کنید\n\n"
                "🔄 برای تحلیل مجدد /start را ارسال کنید."
            )
        
        await update.message.reply_text(message, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in start command: {e}")
        await update.message.reply_text(
            "❌ خطایی در تحلیل بازار رخ داد. لطفا دوباره تلاش کنید.\n"
            f"جزئیات خطا: {str(e)[:100]}..."
        )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command"""
    try:
        # Test exchange connection
        exchange_status = "❌ قطع"
        try:
            test_exchange = await init_exchange()
            if test_exchange:
                await test_exchange.fetch_ticker('BTC/USDT')
                exchange_status = "✅ متصل"
        except:
            pass
        
        message = "🤖 *وضعیت ربات:*\n\n"
        message += "🟢 ربات فعال است\n"
        message += f"📈 تعداد نمادها: `{len(SYMBOLS)}`\n"
        message += f"🔗 صرافی CoinEx: {exchange_status}\n"
        message += f"⏰ آخرین بررسی: `{datetime.now().strftime('%H:%M:%S')}`\n"
        message += f"🐍 Python: `{sys.version.split()[0]}`\n\n"
        message += "📊 *شاخص‌های تحلیلی:*\n"
        message += "• RSI (قدرت نسبی)\n"
        message += "• MACD (همگرایی واگرایی)\n"
        message += "• Stochastic (نوسانگر)\n"
        message += "• MFI (شاخص جریان پول)\n"
        message += "• CCI (شاخص کانال کالا)\n"
        message += "• Williams %R\n"
        message += "• Fibonacci Levels\n"
        message += "• Volume Analysis\n\n"
        message += "💡 *دستورات موجود:*\n"
        message += "`/start` - تحلیل بازار\n"
        message += "`/status` - وضعیت ربات\n"
        message += "`/symbols` - نمایش نمادها\n"
        message += "`/help` - راهنما"
        
        await update.message.reply_text(message, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in status command: {e}")
        await update.message.reply_text("خطایی در نمایش وضعیت رخ داد.")

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
