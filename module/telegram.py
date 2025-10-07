import asyncio
from datetime import datetime
import os
from collections import defaultdict

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from .background_tasks import BackgroundTaskManager
from .config import ConfigManager
from .core import TradingSignal
from .logger_config import logger
from .trading_service import TradingService


def escape_markdown_v2(text: str) -> str:
    escape_chars = r"_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{char}" if char in escape_chars else char for char in str(text))


class TelegramBotHandler:
    def __init__(
        self,
        bot_token: str,
        config_manager: ConfigManager,
        trading_service: TradingService,
        background_tasks: BackgroundTaskManager,
    ):
        self.config_manager = config_manager
        self.trading_service = trading_service
        self.background_tasks = background_tasks

        admin_chat_id_raw = (
            os.getenv("ADMIN_CHAT_ID")
            or self.config_manager.get("admin_chat_id")
            or self.config_manager.get("telegram_chat_id")
        )
        self.admin_chat_id = str(admin_chat_id_raw).strip() if admin_chat_id_raw else None

        if not self.admin_chat_id:
            logger.error("CRITICAL: Admin Chat ID is not configured in environment or config!")
        else:
            logger.info(f"Admin Chat ID configured as: {self.admin_chat_id}")

        request = HTTPXRequest(
            http_version="1.1",
            connection_pool_size=10,
            connect_timeout=10.0,
            read_timeout=10.0,
            write_timeout=10.0,
            pool_timeout=10.0,
        )

        self.application = ApplicationBuilder().token(bot_token).request(request).build()
        self._register_handlers()

    def _register_handlers(self):
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("status", self.status))
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_admin(update.effective_chat.id):
            return
        keyboard = [
            [InlineKeyboardButton("⚡ Quick Analyze (1h)", callback_data="quick_analyze")],
            [InlineKeyboardButton("📊 Full Analyze (Long-term: 1h-1M)", callback_data="full_analyze")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        welcome_text = (
            "Welcome to the Long-term Trading Signal Bot! 🤖\n\n"
            "Choose your analysis type:\n"
            "• *Quick Analyze*: Only 1h timeframe ⚡\n"
            "• *Full Analyze*: All long-term timeframes (1h, 4h, 1d, 1w, 1M) 📊"
        )
        await update.message.reply_text(
            escape_markdown_v2(welcome_text),
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN_V2,
        )

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        if not self._is_admin(query.message.chat_id):
            return
        if query.data == "quick_analyze":
            await self.quick_analyze(update, context)
        elif query.data == "full_analyze":
            await self.full_analyze(update, context)

    async def quick_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        text = "Starting Quick Analyze (1h timeframe)... ⏳"
        await query.edit_message_text(escape_markdown_v2(text), parse_mode=ParseMode.MARKDOWN_V2)

        async def analysis_task():
            symbols = self.config_manager.get("symbols", [])
            signals = []
            for symbol in symbols:
                try:
                    signal = await self.trading_service.analyze_symbol(symbol, "1h")
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol} on 1h: {e}")
            await self.send_signals_to_telegram(
                signals,
                str(query.message.chat_id),
                "✅ Quick Analyze completed.",
                "❌ No signals found on 1h timeframe.",
            )

        self.background_tasks.create_task(analysis_task())

    async def full_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        text = "Starting Full Analyze (all timeframes)... ⏳"
        await query.edit_message_text(escape_markdown_v2(text), parse_mode=ParseMode.MARKDOWN_V2)

        async def analysis_task():
            signals = await self.trading_service.run_analysis_for_all_symbols()
            await self.send_signals_to_telegram(
                signals,
                str(query.message.chat_id),
                "✅ Full Analyze completed.",
                "❌ No signals found across all timeframes.",
            )

        self.background_tasks.create_task(analysis_task())

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self._is_admin(update.effective_chat.id):
            text = "Bot is operational. ✅"
            await update.message.reply_text(escape_markdown_v2(text), parse_mode=ParseMode.MARKDOWN_V2)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self._is_admin(update.effective_chat.id):
            text = "Please use /start to choose analysis type. 🤖"
            await update.message.reply_text(escape_markdown_v2(text), parse_mode=ParseMode.MARKDOWN_V2)

    def _is_admin(self, chat_id: int) -> bool:
        incoming_chat_id = str(chat_id).strip()
        if not self.admin_chat_id:
            logger.error("Admin chat ID is not configured!")
            return False
        if incoming_chat_id == self.admin_chat_id:
            logger.info(f"Admin access granted for chat ID: {incoming_chat_id}")
            return True
        logger.warning(f"Unauthorized access attempt from chat ID: {incoming_chat_id} (Expected: {self.admin_chat_id})")
        return False

    async def run_scheduled_analysis(self):
        logger.info("Running scheduled analysis for all symbols...")
        signals = await self.trading_service.run_analysis_for_all_symbols()
        await self.send_signals_to_telegram(
            signals,
            self.admin_chat_id,
            "Scheduled analysis completed. ⏰",
            "No new signals found from scheduled analysis. 🤷",
        )

    async def send_signals_to_telegram(
        self,
        signals: list[TradingSignal],
        chat_id: str,
        summary_text: str,
        no_signals_text: str,
    ):
        if not chat_id:
            logger.warning("Telegram chat ID not configured.")
            return

        min_confidence = self.config_manager.get("min_confidence_score", 0)
        max_signals_per_tf = self.config_manager.get("max_signals_per_timeframe", 1)

        confident_signals = [s for s in signals if s.confidence_score >= min_confidence]

        signals_by_timeframe = defaultdict(list)
        for signal in confident_signals:
            signals_by_timeframe[signal.timeframe].append(signal)

        filtered_signals = []
        for tf, tf_signals in signals_by_timeframe.items():
            sorted_signals = sorted(tf_signals, key=lambda s: s.confidence_score, reverse=True)
            filtered_signals.extend(sorted_signals[:max_signals_per_tf])
        
        filtered_signals.sort(key=lambda s: s.confidence_score, reverse=True)

        if filtered_signals:
            summary = f"{summary_text} Found {len(filtered_signals)} signal(s). 🎯"
            await self.application.bot.send_message(
                chat_id,
                escape_markdown_v2(summary),
                parse_mode=ParseMode.MARKDOWN_V2,
            )
            for signal in filtered_signals:
                message = self.format_signal_message(signal)
                try:
                    await self.application.bot.send_message(chat_id, message, parse_mode=ParseMode.MARKDOWN_V2)
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Failed to send signal message: {e}")
        else:
            await self.application.bot.send_message(
                chat_id,
                escape_markdown_v2(no_signals_text),
                parse_mode=ParseMode.MARKDOWN_V2,
            )

    def format_signal_message(self, signal: TradingSignal) -> str:
        signal_type_str = signal.signal_type.value.upper()
        signal_emoji = "📈" if signal_type_str == "BUY" else "📉"
        signal_type = escape_markdown_v2(f"{signal_emoji} {signal_type_str}")
        symbol = escape_markdown_v2(signal.symbol)
        timeframe = escape_markdown_v2(signal.timeframe)
        header = f"*{signal_type} Signal for {symbol} on {timeframe}*"

        confidence_score_str = escape_markdown_v2(f"{signal.confidence_score:.4f}%")
        entry_price_str = escape_markdown_v2(f"{signal.entry_price:.10f}")
        exit_price_str = escape_markdown_v2(f"{signal.exit_price:.10f}")
        stop_loss_str = escape_markdown_v2(f"{signal.stop_loss:.10f}")
        risk_reward_ratio_str = escape_markdown_v2(f"{signal.risk_reward_ratio:.4f}")

        main_info = (
            f" Date Created: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n"
            f"🎯 Confidence: `{confidence_score_str}`\n"
            f"➡️ Entry: `{entry_price_str}`\n"
            f"✅ Take Profit: `{exit_price_str}`\n"
            f"🛑 Stop Loss: `{stop_loss_str}`\n"
            f"⚖️ Risk/Reward Ratio: `{risk_reward_ratio_str}`"
        )

        ctx = signal.market_context
        trend = escape_markdown_v2(str(ctx.get("trend", "N/A")))
        condition = escape_markdown_v2(str(ctx.get("market_condition", "N/A")))
        vol_trend = escape_markdown_v2(str(ctx.get("volume_trend", "N/A")))
        volatility_str = escape_markdown_v2(f"{ctx.get('volatility', 0):.2f}%")

        market_info = (
            f"\n*Market Context: 🌐*\n"
            f"Trend: `{trend}`\n"
            f"Condition: `{condition}`\n"
            f"Volatility: `{volatility_str}`\n"
            f"Volume Trend: `{vol_trend}`"
        )

        reasons_list = [f"• {escape_markdown_v2(r)}" for r in signal.reasons]
        reasons = "\n*Analysis Reasons: 🧠*\n" + "\n".join(reasons_list)

        return f"{header}\n\n{main_info}{market_info}{reasons}"