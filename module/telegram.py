import asyncio
import io
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

from module.config import Config, ConfigManager
from module.constants import TIME_FRAMES
from module.core import SignalType, TradingSignal
from module.logger_config import logger
from module.monitoring import SIGNALS_GENERATED_TOTAL
from module.signals import SignalGenerator, SignalRanking
from module.tasks import run_full_analysis_task


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
        tasks = {tf: self.find_best_signals_for_timeframe(tf) for tf in timeframes}
        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results))

    async def cleanup(self):
        if hasattr(self, "exchange_manager") and self.exchange_manager and self._initialized:
            await self.exchange_manager.close()
        self._initialized = False
        if self.signal_generator and self.signal_generator.lstm_model_manager:
            self.signal_generator.lstm_model_manager.cleanup()


class TelegramBotHandler:
    def __init__(self, bot_token: str, config_manager: ConfigManager, trading_service: TradingBotService):
        self.bot_token = bot_token
        self.config = config_manager
        self.trading_service = trading_service
        self.application = None
        self._scheduled_analysis_task = None

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
        query = update.callback_query
        await query.answer()

        if query.data == "full_analysis":
            await query.edit_message_text("✅ Analysis request queued. This may take a few minutes. You will be notified when it's complete.")
            run_full_analysis_task.delay(query.message.chat_id, query.message.message_id)
        elif query.data == "quick_scan":
            await self.run_quick_scan(query)

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
                        await self.send_signal_to_chat(chat_id, signal)
            
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


    async def run_full_analysis_from_task(self, chat_id: int, message_id: int):
        try:
            await self.application.bot.edit_message_text(
                chat_id=chat_id, message_id=message_id, text="⏳ Analysis in progress..."
            )
        except Exception as e:
            logger.warning(f"Could not edit message, it might have been deleted: {e}")

        results = await self.trading_service.get_comprehensive_analysis()
        found_signals = False
        for timeframe, signals in results.items():
            if signals:
                found_signals = True
                await self.application.bot.send_message(chat_id, f"--- Signals for {timeframe} ---")
                for signal in signals:
                    await self.send_signal_to_chat(chat_id, signal)
        
        final_message = "Full analysis complete."
        if not found_signals:
            final_message = "No strong signals found in the current market conditions."
        
        try:
            await self.application.bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=final_message)
        except Exception:
            await self.application.bot.send_message(chat_id=chat_id, text=final_message)

    def format_signal_message(self, signal: TradingSignal) -> str:
        signal_emoji = "🟢" if signal.signal_type == SignalType.BUY else "🔴"
        message = f"{signal_emoji} *{signal.symbol} ({signal.timeframe}) - {signal.signal_type.value.upper()}*\n\n"
        message += f"✨ *Confidence:* {signal.confidence_score:.2f}%\n"
        message += f"🎯 *Entry:* `{signal.entry_price:.4f}`\n"
        message += f"💰 *Take Profit:* `{signal.exit_price:.4f}`\n"
        message += f"🛑 *Stop Loss:* `{signal.stop_loss:.4f}`\n"
        message += f"⚖️ *Risk/Reward Ratio:* {signal.risk_reward_ratio:.2f}\n"
        message += f"📈 *Predicted Profit:* {signal.predicted_profit:.2f}%\n\n"
        
        if signal.reasons:
            message += "*Key Reasons:*\n"
            for reason in signal.reasons[:5]:
                message += f"• _{reason}_\n"
        return message

    async def send_signal_to_chat(self, chat_id: int, signal: TradingSignal):
        message = self.format_signal_message(signal)
        try:
            await self.application.bot.send_message(chat_id, text=message, parse_mode=ParseMode.MARKDOWN_V2)
        except Exception as e:
            logger.error(f"Failed to send signal message for {signal.symbol}: {e}. Retrying without markdown.")
            plain_message = message.replace('*', '').replace('_', '').replace('`', '')
            await self.application.bot.send_message(chat_id, text=plain_message)


    async def run_quick_scan(self, query):
        await query.edit_message_text("⏳ Running quick scan for 1h timeframe...")
        signals = await self.trading_service.find_best_signals_for_timeframe("1h")
        if signals:
            await query.edit_message_text("--- Top Signals for 1h ---")
            for signal in signals:
                await self.send_signal_to_chat(query.message.chat_id, signal)
        else:
            await query.edit_message_text("No strong signals found on the 1h timeframe right now.")


    async def cleanup(self):
        if self.trading_service:
            await self.trading_service.cleanup()