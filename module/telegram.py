import asyncio
import io
import time
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

from module.config import Config, ConfigManager
from module.constants import TIME_FRAMES
from module.core import SignalType, TradingSignal
from module.logger_config import logger
from module.monitoring import SIGNALS_GENERATED_TOTAL
from module.signals import SignalGenerator, SignalRanking


class TradingBotService:
    def __init__(
        self,
        config: ConfigManager,
        exchange_manager,
        signal_generator: SignalGenerator,
        signal_ranking: SignalRanking,
    ):
        self.config = config
        self.exchange_manager = exchange_manager
        self.signal_generator = signal_generator
        self.signal_ranking = signal_ranking
        self._initialized = False
        self.telegram_app: Optional[Application] = None

    def set_telegram_app(self, app: Application):
        self.telegram_app = app

    async def initialize(self):
        if self._initialized:
            return
        if self.exchange_manager:
            await self.exchange_manager.init_database()
        self._initialized = True

    async def analyze_symbol(self, symbol: str, timeframe: str) -> List[TradingSignal]:
        try:
            data = await self.exchange_manager.fetch_ohlcv_data(symbol, timeframe, limit=300)
            if data is None or data.empty or len(data) < 100:
                logger.warning(f"Insufficient data for {symbol} on {timeframe}")
                return []

            all_signals = await self.signal_generator.generate_signals(data, symbol, timeframe)
            min_confidence = self.config.get("min_confidence_score", 60)
            qualified_signals = [s for s in all_signals if s.confidence_score >= min_confidence]

            for signal in qualified_signals:
                SIGNALS_GENERATED_TOTAL.labels(
                    symbol=signal.symbol, timeframe=signal.timeframe, signal_type=signal.signal_type.value
                ).inc()
            return qualified_signals
        except Exception as e:
            logger.error(f"Analysis failed for {symbol} on {timeframe}: {e}", exc_info=True)
            return []

    async def find_best_signals_for_timeframe(self, timeframe: str) -> List[TradingSignal]:
        symbols = self.config.get("symbols", [])
        tasks = [self.analyze_symbol(symbol, timeframe) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        all_signals = [signal for res in results for signal in res]
        ranked_signals = self.signal_ranking.rank_signals(all_signals)
        return ranked_signals[: self.config.get("max_signals_per_timeframe", 3)]

    async def get_comprehensive_analysis(self) -> Dict[str, List[TradingSignal]]:
        timeframes = self.config.get("timeframes", TIME_FRAMES)
        analysis_results = {}
        for tf in timeframes:
            analysis_results[tf] = await self.find_best_signals_for_timeframe(tf)
        return analysis_results
        
    async def run_comprehensive_analysis_task(self, chat_id: int, message_id: int):
        if not self.telegram_app:
            logger.error("Telegram application not set in TradingBotService.")
            return

        timeframes = self.config.get("timeframes", TIME_FRAMES)
        total_signals_found = 0

        try:
            await self.telegram_app.bot.edit_message_text(
                chat_id=chat_id, message_id=message_id, text="🚀 Starting full analysis..."
            )
        except BadRequest:
            logger.warning("Could not edit initial message, it may have been deleted.")

        for timeframe in timeframes:
            await self.telegram_app.bot.send_message(
                chat_id=chat_id, text=f"⏳ Analyzing {timeframe} timeframe..."
            )
            
            signals = await self.find_best_signals_for_timeframe(timeframe)
            
            if signals:
                total_signals_found += len(signals)
                await self.telegram_app.bot.send_message(
                    chat_id, f"--- Top Signals for {timeframe} ---"
                )
                for signal in signals:
                    await self.send_signal_to_chat(chat_id, signal)
            else:
                await self.telegram_app.bot.send_message(
                    chat_id=chat_id, text=f"✅ No strong signals found for {timeframe}."
                )
            await asyncio.sleep(1)

        final_message = "🏁 Full analysis complete."
        if total_signals_found == 0:
            final_message += "\n\nNo strong signals found in the current market conditions across all timeframes."
        
        await self.telegram_app.bot.send_message(chat_id=chat_id, text=final_message)


    async def run_find_best_signals_task(self, timeframe: str, chat_id: int, message_id: int):
        if not self.telegram_app:
            logger.error("Telegram application not set in TradingBotService.")
            return
        
        try:
            await self.telegram_app.bot.edit_message_text(
                chat_id=chat_id, message_id=message_id, text=f"⏳ Running quick scan for {timeframe} timeframe..."
            )
        except BadRequest:
             logger.warning(f"Could not edit message, it might have been deleted.")

        signals = await self.find_best_signals_for_timeframe(timeframe)
        
        final_message = f"⚡ Quick scan for {timeframe} complete."
        if signals:
            await self.telegram_app.bot.send_message(chat_id, f"--- Top Signals for {timeframe} ---")
            for signal in signals:
                await self.send_signal_to_chat(chat_id, signal)
        else:
            final_message = f"No strong signals found on the {timeframe} timeframe right now."

        try:
            await self.telegram_app.bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=final_message)
        except BadRequest:
            await self.telegram_app.bot.send_message(chat_id=chat_id, text=final_message)

    async def cleanup(self):
        if hasattr(self, "exchange_manager") and self.exchange_manager and self._initialized:
            await self.exchange_manager.close()
        self._initialized = False
        if self.signal_generator and self.signal_generator.lstm_model_manager:
            self.signal_generator.lstm_model_manager.cleanup()
            
    def format_signal_message(self, signal: TradingSignal) -> str:
        signal_emoji = "🟢" if signal.signal_type == SignalType.BUY else "🔴"
        message = f"{signal_emoji} *{signal.symbol} ({signal.timeframe}) - {signal.signal_type.value.upper()}*\n\n"
        
        message += f"✨ *Confidence:* {signal.confidence_score:.2f}%\n"
        message += f"📈 *Predicted Profit:* {signal.predicted_profit:.2f}%\n"
        message += f"⚖️ *Risk/Reward Ratio:* {signal.risk_reward_ratio:.2f}\n\n"

        if signal.market_context:
            mc = signal.market_context
            trend = mc.get('trend', 'N/A').replace('_', ' ').title()
            strength = mc.get('trend_strength', 'N/A').title()
            volatility = mc.get('volatility', 0)
            message += "*Market Context:*\n"
            message += f"  - *Trend:* {trend} ({strength})\n"
            message += f"  - *Volatility:* {volatility:.2f}%\n\n"

        if signal.dynamic_levels:
            dl = signal.dynamic_levels
            message += "*Dynamic Levels:*\n"
            message += f"  - *Entry:* `{dl.get('primary_entry', 0):.4f}`\n"
            message += f"  - *Take Profit:* `{dl.get('primary_exit', 0):.4f}`\n"
            message += f"  - *Stop Loss:* `{dl.get('tight_stop', 0):.4f}`\n\n"

        if signal.reasons:
            message += "*Key Reasons:*\n"
            for reason in signal.reasons[:5]:
                message += f"  • _{reason}_\n"
        
        return message

    async def send_signal_to_chat(self, chat_id: int, signal: TradingSignal):
        if not self.telegram_app:
            logger.error("Cannot send message, Telegram app is not available.")
            return
            
        message = self.format_signal_message(signal)
        try:
            await self.telegram_app.bot.send_message(chat_id, text=message, parse_mode=ParseMode.MARKDOWN_V2)
        except Exception as e:
            logger.error(f"Failed to send signal message for {signal.symbol}: {e}. Retrying without markdown.")
            plain_message = message.replace('*', '').replace('_', '').replace('`', '').replace('.', r'\.')
            try:
                await self.telegram_app.bot.send_message(chat_id, text=plain_message)
            except Exception as e2:
                logger.error(f"Failed to send plain text message for {signal.symbol}: {e2}")


class TelegramBotHandler:
    def __init__(self, bot_token: str, config_manager: ConfigManager, trading_service: TradingBotService):
        self.bot_token = bot_token
        self.config = config_manager
        self.trading_service = trading_service
        self.application = None

    async def initialize(self):
        await self.trading_service.initialize()

    def create_application(self):
        app_builder = Application.builder().token(self.bot_token)
        app_builder.post_init(self.post_init)
        self.application = app_builder.build()
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        return self.application
    
    async def post_init(self, application: Application):
        await application.bot.set_my_commands([
            ('start', 'Start the bot and see main menu'),
        ])

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            [
                InlineKeyboardButton("🚀 Full Analysis", callback_data="full_analysis"),
                InlineKeyboardButton("⚡ Quick Scan (1h)", callback_data="quick_scan"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Welcome to the Trading Bot! Choose an option:", reply_markup=reply_markup)

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        from module.tasks import run_full_analysis_task, run_quick_scan_task
        query = update.callback_query
        await query.answer()

        if query.data == "full_analysis":
            await query.edit_message_text("✅ Full analysis request queued. You will receive updates as each timeframe is processed.")
            run_full_analysis_task.delay(query.message.chat_id, query.message.message_id)
        elif query.data == "quick_scan":
            await query.edit_message_text("✅ Quick scan request queued. This will be processed shortly.")
            run_quick_scan_task.delay(query.message.chat_id, query.message.message_id)

    async def run_scheduled_analysis(self):
        logger.info("Running scheduled analysis...")
        chat_id = Config.TELEGRAM_CHAT_ID
        if not chat_id:
            logger.warning("TELEGRAM_CHAT_ID not set. Cannot send scheduled analysis.")
            return

        try:
            results = await self.trading_service.get_comprehensive_analysis()
            found_signals = False
            for timeframe, signals in results.items():
                if signals:
                    found_signals = True
                    await self.application.bot.send_message(chat_id, f"--- Scheduled Signals for {timeframe} ---")
                    for signal in signals:
                        await self.trading_service.send_signal_to_chat(chat_id, signal)
            
            if not found_signals:
                await self.application.bot.send_message(chat_id, "No strong signals found in scheduled analysis.")
            else:
                await self.application.bot.send_message(chat_id, "Scheduled analysis complete.")
        except Exception as e:
            logger.error(f"Error during scheduled analysis: {e}")
            try:
                await self.application.bot.send_message(chat_id, f"An error occurred during scheduled analysis: {e}")
            except Exception as bot_e:
                logger.error(f"Failed to send error message to Telegram: {bot_e}")

    async def cleanup(self):
        if self.trading_service:
            await self.trading_service.cleanup()