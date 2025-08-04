# handlers/bot_handlers.py

import asyncio
import logging
from typing import Dict
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ParseMode

from telegram.ext import (
    CallbackContext, CommandHandler, MessageHandler, 
    CallbackQueryHandler, Filters, ConversationHandler,
    ContextTypes
)

from models.user import User, UserRole, UserStatus
from models.notification import Notification, NotificationType, NotificationPriority
from services.telegram_service import TelegramService
from services.signal_service import SignalService
from config.constants import ERROR_MESSAGE, MESSAGE_TEMPLATES, NO_SIGNAL_FOUND, TELEGRAM_COLORS, TIMEFRAME_PERSIAN_NAMES, WAIT_MESSAGE
from utils.helpers import format_price, format_percentage, get_persian_date
from utils.validators import validate_user_input

logger = logging.getLogger(__name__)

# حالت‌های مکالمه
(SELECTING_TIMEFRAMES, SELECTING_SYMBOLS, SETTING_CONFIDENCE, 
SETTING_NOTIFICATIONS) = range(4)

class TelegramBotHandlers:
    """کلاس مدیریت هندلرهای تلگرام"""
    
    def __init__(self, telegram_service: TelegramService, signal_service: SignalService):
        self.telegram_service = telegram_service
        self.signal_service = signal_service
        self.user_sessions: Dict[int, Dict] = {}
        
    def initialize(self) -> None:
        """راه‌اندازی هندلرها"""
        logger.info("🤖 راه‌اندازی هندلرهای تلگرام...")
        
        # ثبت هندلرها
        handlers = [
            CommandHandler("start", self.start_command),
            CommandHandler("help", self.help_command),
            CommandHandler("status", self.status_command),
            CommandHandler("settings", self.settings_command),
            CommandHandler("signals", self.signals_command),
            CommandHandler("subscribe", self.subscribe_command),
            CommandHandler("unsubscribe", self.unsubscribe_command),
            CommandHandler("stats", self.stats_command),
            CallbackQueryHandler(self.button_callback),
            MessageHandler(Filters.text & ~Filters.command, self.handle_text_message),
            MessageHandler(Filters.document, self.handle_document),
        ]
        
        # تنظیمات مکالمه
        conversation_handler = ConversationHandler(
            entry_points=[CommandHandler("setup", self.setup_command)],
            states={
                SELECTING_TIMEFRAMES: [CallbackQueryHandler(self.select_timeframes)],
                SELECTING_SYMBOLS: [CallbackQueryHandler(self.select_symbols)],
                SETTING_CONFIDENCE: [MessageHandler(Filters.text, self.set_confidence)],
                SETTING_NOTIFICATIONS: [CallbackQueryHandler(self.set_notifications)],
            },
            fallbacks=[CommandHandler("cancel", self.cancel_setup)]
        )
        handlers.append(conversation_handler)
        
        # افزودن هندلرها به تلگرام سرویس
        for handler in handlers:
            self.telegram_service.application.add_handler(handler)
            
        logger.info("✅ هندلرهای تلگرام راه‌اندازی شدند")
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command with improved performance"""
        user_info = self._extract_user_info(update)

        try:
            logger.info(f"User {user_info['username']} ({user_info['id']}) started analysis")

            await self._send_safe_message(update, WAIT_MESSAGE)

            # Start background analysis without blocking
            result = asyncio.create_task(analyze_market())
            
            # Wait for the analysis to complete
            analysis_result = await result

            # Check if the result is empty
            if not analysis_result:
                await self._send_safe_message(update, NO_SIGNAL_FOUND)
            else:
                # Send the analysis result
                await self._send_safe_message(update, analysis_result, parse_mode='Markdown')

        except Exception as e:
            logger.error(f"Error in start command for user {user_info['username']} ({user_info['id']}): {e}", exc_info=True)
            await self._handle_command_error(update, e, "start command")


    def _extract_user_info(self, update: Update) -> dict:
        """Extract user information safely"""
        return {
            'id': update.effective_user.id,
            'username': update.effective_user.username or "Unknown"
        }

    async def _send_safe_message(self, update: Update, message: str, **kwargs) -> bool:
        """Send message with fallback for markdown errors"""
        try:
            await update.message.reply_text(message, **kwargs)
            return True
        except Exception as e:
            logger.error(f"Failed to send formatted message: {e}")
            # Fallback to plain text
            plain_message = message.replace('*', '').replace('`', '').replace('_', '')
            try:
                await update.message.reply_text(plain_message)
                return True
            except Exception as fallback_e:
                logger.error(f"Failed to send fallback message: {fallback_e}")
                return False

    async def _handle_command_error(self, update: Update, error: Exception,
                                    command: str, username: str = "Unknown") -> None:
        """Handle command errors consistently"""
        logger.error(f"Error in {command} for user {username}: {error}", exc_info=True)
        error_msg = f"{ERROR_MESSAGE}جزئیات خطا: {str(error)[:100]}"
        await self._send_safe_message(update, error_msg)

    
    
    
    async def start_command(self, update: Update, context: CallbackContext) -> None:
        """هندلر دستور /start"""
        try:
            user_data = update.effective_user
            user = await self.get_or_create_user(user_data)
            
            # ایجاد کیبورد inline
            keyboard = [
                [
                    InlineKeyboardButton("📊 تنظیمات", callback_data="settings"),
                    InlineKeyboardButton("📈 سیگنال‌ها", callback_data="signals"),
                ],
                [
                    InlineKeyboardButton("ℹ️ راهنما", callback_data="help"),
                    InlineKeyboardButton("📊 آمار", callback_data="stats"),
                ],
                [
                    InlineKeyboardButton("💎 اشتراک VIP", callback_data="subscribe"),
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            welcome_text = MESSAGE_TEMPLATES["WELCOME"]
            
            await update.message.reply_text(
                welcome_text,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup
            )
            
            # ثبت فعالیت کاربر
            await self.log_user_activity(user, "start_command")
            
        except Exception as e:
            logger.error(f"خطا در start_command: {e}")
            await self.send_error_message(update, "خطا در اجرای دستور start")
    
    async def help_command(self, update: Update, context: CallbackContext) -> None:
        """هندلر دستور /help"""
        help_text = """
📚 <b>راهنمای استفاده از ربات YujTrade</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 <b>دستورات اصلی:</b>
/start - شروع مجدد ربات
/help - نمایش این راهنما  
/status - نمایش وضعیت سیستم
/settings - تنظیمات شخصی
/signals - مشاهده سیگنال‌های اخیر
/subscribe - خرید اشتراک VIP
/stats - آمار شخصی

🔹 <b>ویژگی‌های کلیدی:</b>
• سیگنال‌های معاملاتی دقیق
• تحلیل فنی پیشرفته
• پشتیبانی از تمام تایم فریم‌ها
• مدیریت ریسک هوشمند
• هشدارهای شخصی‌سازی شده

🔹 <b>انواع اشتراک:</b>
• رایگان: 5 سیگنال در روز
• VIP: نامحدود + ویژگی‌های اضافی

━━━━━━━━━━━━━━━━━━━━━━━━━━━
💬 <b>پشتیبانی:</b> @YujTradeSupport
"""
        
        await update.effective_chat.send_message(
            help_text,
            parse_mode=ParseMode.HTML
        )
    
    async def status_command(self, update: Update, context: CallbackContext) -> None:
        """هندلر دستور /status"""
        try:
            # دریافت وضعیت سیستم
            system_status = await self.signal_service.get_system_status()
            
            status_text = f"""
⚡️ <b>وضعیت سیستم YujTrade</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔋 <b>وضعیت:</b> {TELEGRAM_COLORS.get('SUCCESS', '✅')} فعال
📊 <b>تعداد نمادها:</b> {system_status.get('symbols_count', 0)}
⏰ <b>تایم فریم‌های فعال:</b> {len(system_status.get('active_timeframes', []))}
📈 <b>سیگنال‌های امروز:</b> {system_status.get('signals_today', 0)}
👥 <b>کاربران فعال:</b> {system_status.get('active_users', 0)}

🕐 <b>آخرین به‌روزرسانی:</b> {get_persian_date(datetime.now())}
━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            
            await update.effective_chat.send_message(
                status_text,
                parse_mode=ParseMode.HTML
            )
            
        except Exception as e:
            logger.error(f"خطا در status_command: {e}")
            await self.send_error_message(update, "خطا در دریافت وضعیت سیستم")
    
    async def settings_command(self, update: Update, context: CallbackContext) -> None:
        """هندلر دستور /settings"""
        try:
            user_data = update.effective_user
            user = await self.get_or_create_user(user_data)
            
            # ایجاد کیبورد تنظیمات
            keyboard = [
                [
                    InlineKeyboardButton("⏰ تایم فریم‌ها", callback_data="settings_timeframes"),
                    InlineKeyboardButton("💰 نمادها", callback_data="settings_symbols"),
                ],
                [
                    InlineKeyboardButton("📊 حد اطمینان", callback_data="settings_confidence"),
                    InlineKeyboardButton("🔔 اعلان‌ها", callback_data="settings_notifications"),
                ],
                [
                    InlineKeyboardButton("🌐 زبان", callback_data="settings_language"),
                    InlineKeyboardButton("🎨 تم", callback_data="settings_theme"),
                ],
                [
                    InlineKeyboardButton("🔙 بازگشت", callback_data="main_menu"),
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            settings_text = f"""
⚙️ <b>تنظیمات شخصی</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━

👤 <b>نام:</b> {user.display_name}
🏷️ <b>نقش:</b> {user.role.value}
📊 <b>حد اطمینان:</b> {user.preferences.min_confidence_score}%
⏰ <b>تایم فریم‌های انتخابی:</b> {len(user.preferences.preferred_timeframes)}
💰 <b>نمادهای دنبال شده:</b> {len(user.preferences.watchlist)}
🌐 <b>زبان:</b> {user.preferences.language}

━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡️ برای تغییر تنظیمات از دکمه‌های زیر استفاده کنید
"""
            
            await update.effective_chat.send_message(
                settings_text,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"خطا در settings_command: {e}")
            await self.send_error_message(update, "خطا در نمایش تنظیمات")
    
    async def signals_command(self, update: Update, context: CallbackContext) -> None:
        """هندلر دستور /signals"""
        try:
            user_data = update.effective_user
            user = await self.get_or_create_user(user_data)
            
            # دریافت سیگنال‌های اخیر
            recent_signals = await self.signal_service.get_recent_signals(
                user_id=user.id,
                limit=10
            )
            
            if not recent_signals:
                await update.effective_chat.send_message(
                    "📭 سیگنال اخیری یافت نشد",
                    parse_mode=ParseMode.HTML
                )
                return
            
            signals_text = "📈 <b>سیگنال‌های اخیر</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            for signal in recent_signals:
                color = TELEGRAM_COLORS.get(signal.type.upper(), "🔵")
                signals_text += f"""
{color} <b>{signal.symbol}</b> - {signal.type.upper()}
⏰ {TIMEFRAME_PERSIAN_NAMES.get(signal.timeframe, signal.timeframe)}
📊 اطمینان: {signal.metrics.confidence_score:.1f}%
💰 قیمت: ${format_price(signal.entry_price)}
🕐 {get_persian_date(signal.created_at)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            
            # کیبورد برای اقدامات
            keyboard = [
                [
                    InlineKeyboardButton("🔄 به‌روزرسانی", callback_data="refresh_signals"),
                    InlineKeyboardButton("⚙️ تنظیمات", callback_data="settings"),
                ],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.effective_chat.send_message(
                signals_text,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"خطا در signals_command: {e}")
            await self.send_error_message(update, "خطا در دریافت سیگنال‌ها")
    
    async def setup_command(self, update: Update, context: CallbackContext) -> int:
        """شروع فرآیند تنظیمات اولیه"""
        keyboard = [
            [
                InlineKeyboardButton("1m", callback_data="tf_1m"),
                InlineKeyboardButton("5m", callback_data="tf_5m"),
                InlineKeyboardButton("15m", callback_data="tf_15m"),
            ],
            [
                InlineKeyboardButton("1h", callback_data="tf_1h"),
                InlineKeyboardButton("4h", callback_data="tf_4h"), 
                InlineKeyboardButton("1d", callback_data="tf_1d"),
            ],
            [
                InlineKeyboardButton("✅ تمام", callback_data="tf_all"),
                InlineKeyboardButton("🔙 بازگشت", callback_data="tf_done"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "⏰ <b>انتخاب تایم فریم‌ها</b>\n\nتایم فریم‌های مورد نظر خود را انتخاب کنید:",
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
        
        return SELECTING_TIMEFRAMES
    
    async def button_callback(self, update: Update, context: CallbackContext) -> None:
        """هندلر callback query ها"""
        query = update.callback_query
        await query.answer()
        
        try:
            data = query.data
            
            if data == "settings":
                await self.settings_command(update, context)
            elif data == "signals":
                await self.signals_command(update, context)
            elif data == "help":
                await self.help_command(update, context)
            elif data == "stats":
                await self.stats_callback(query, context)
            elif data == "subscribe":
                await self.subscribe_callback(query, context)
            elif data.startswith("settings_"):
                await self.handle_settings_callback(query, context, data)
            elif data.startswith("tf_"):
                await self.handle_timeframe_callback(query, context, data)
            elif data == "refresh_signals":
                await self.refresh_signals_callback(query, context)
            else:
                await query.edit_message_text("⚠️ دستور نامشخص")
                
        except Exception as e:
            logger.error(f"خطا در button_callback: {e}")
            await query.edit_message_text("❌ خطا در پردازش درخواست")
    
    async def handle_text_message(self, update: Update, context: CallbackContext) -> None:
        """پردازش پیام‌های متنی"""
        try:
            text = update.message.text.strip()
            
            # پردازش کامندهای غیر رسمی
            if text.startswith("/"):
                await update.message.reply_text(
                    "❓ دستور ناشناخته. برای مشاهده دستورات از /help استفاده کنید."
                )
                return
            
            # پردازش متن آزاد
            await self.process_free_text(update, context, text)
            
        except Exception as e:
            logger.error(f"خطا در handle_text_message: {e}")
    
    async def get_or_create_user(self, user_data) -> User:
        """دریافت یا ایجاد کاربر"""
        try:
            user = await self.telegram_service.user_repository.get_by_telegram_id(
                str(user_data.id)
            )
            
            if not user:
                user = User(
                    telegram_id=str(user_data.id),
                    username=user_data.username,
                    first_name=user_data.first_name,
                    last_name=user_data.last_name,
                    role=UserRole.FREE,
                    status=UserStatus.ACTIVE
                )
                user = await self.telegram_service.user_repository.create(user)
                logger.info(f"کاربر جدید ایجاد شد: {user.display_name}")
            
            # به‌روزرسانی آخرین فعالیت
            user.last_activity = datetime.now()
            await self.telegram_service.user_repository.update(user)
            
            return user
            
        except Exception as e:
            logger.error(f"خطا در get_or_create_user: {e}")
            raise
    
    async def send_error_message(self, update: Update, message: str) -> None:
        """ارسال پیام خطا"""
        error_text = MESSAGE_TEMPLATES["ERROR"].format(
            error_message=message,
            timestamp=get_persian_date(datetime.now())
        )
        
        try:
            if update.message:
                await update.message.reply_text(error_text, parse_mode=ParseMode.HTML)
            elif update.callback_query:
                await update.callback_query.edit_message_text(error_text, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"خطا در ارسال پیام خطا: {e}")
    
    async def log_user_activity(self, user: User, activity: str) -> None:
        """ثبت فعالیت کاربر"""
        try:
            # ثبت در لاگ
            logger.info(f"فعالیت کاربر {user.display_name}: {activity}")
            
            # ثبت در متادیتا کاربر
            user.add_metadata(f"last_{activity}", datetime.now().isoformat())
            user.last_activity = datetime.now()
            
            await self.telegram_service.user_repository.update(user)
            
        except Exception as e:
            logger.error(f"خطا در ثبت فعالیت کاربر: {e}")
    
    async def stats_callback(self, query, context: CallbackContext) -> None:
        """نمایش آمار کاربر"""
        try:
            user_data = query.from_user
            user = await self.get_or_create_user(user_data)
            
            stats_text = f"""
📊 <b>آمار شخصی</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 <b>سیگنال‌های دریافتی:</b> {user.statistics.total_signals_received}
✅ <b>سیگنال‌های دنبال شده:</b> {user.statistics.signals_followed}
🎯 <b>معاملات موفق:</b> {user.statistics.successful_trades}
❌ <b>معاملات ناموفق:</b> {user.statistics.failed_trades}
📊 <b>نرخ موفقیت:</b> {user.statistics.win_rate:.1f}%
💰 <b>سود/ضرر کل:</b> {user.statistics.total_pnl}
🔥 <b>برد متوالی:</b> {user.statistics.consecutive_wins}

📅 <b>عضویت:</b> {get_persian_date(user.created_at)}
⏰ <b>آخرین فعالیت:</b> {get_persian_date(user.last_activity) if user.last_activity else 'نامشخص'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            
            keyboard = [
                [InlineKeyboardButton("🔄 به‌روزرسانی", callback_data="refresh_stats")],
                [InlineKeyboardButton("🔙 بازگشت", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                stats_text,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"خطا در stats_callback: {e}")
            await query.edit_message_text("❌ خطا در دریافت آمار")
    
    async def subscribe_callback(self, query, context: CallbackContext) -> None:
        """مدیریت اشتراک"""
        subscription_text = """
💎 <b>اشتراک VIP یوج‌ترید</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 <b>ویژگی‌های VIP:</b>
• سیگنال‌های نامحدود
• تحلیل‌های اختصاصی  
• اولویت در پشتیبانی
• دسترسی به API
• گزارش‌های تفصیلی
• بدون تبلیغات

💰 <b>قیمت:</b>
• 1 ماهه: $29.99
• 3 ماهه: $79.99 (11% تخفیف)
• سالانه: $299.99 (17% تخفیف)

━━━━━━━━━━━━━━━━━━━━━━━━━━━
💳 برای خرید با پشتیبانی تماس بگیرید
"""
        
        keyboard = [
            [
                InlineKeyboardButton("💳 خرید 1 ماهه", callback_data="buy_1m"),
                InlineKeyboardButton("💎 خرید 3 ماهه", callback_data="buy_3m"),
            ],
            [
                InlineKeyboardButton("👑 خرید سالانه", callback_data="buy_1y"),
            ],
            [
                InlineKeyboardButton("💬 پشتیبانی", url="https://t.me/YujTradeSupport"),
                InlineKeyboardButton("🔙 بازگشت", callback_data="main_menu"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            subscription_text,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
    
    def shutdown(self) -> None:
        """خاموش کردن هندلرها"""
        logger.info("🛑 خاموش کردن هندلرهای تلگرام...")
        
        # پاکسازی session ها
        self.user_sessions.clear()
        
        logger.info("✅ هندلرهای تلگرام خاموش شدند")
