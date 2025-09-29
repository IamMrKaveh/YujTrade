import asyncio
import io
import time
import re
from typing import Dict, List, Optional, Set

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
from module.market import MarketDataProvider
from module.signals import SignalGenerator, SignalRanking

def escape_markdown_v2(text: str) -> str:
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

class TradingBotService:
    def __init__(
        self,
        config: ConfigManager,
        market_data_provider: MarketDataProvider,
        signal_generator: SignalGenerator,
        signal_ranking: SignalRanking,
    ):
        self.config = config
        self.market_data_provider = market_data_provider
        self.signal_generator = signal_generator
        self.signal_ranking = signal_ranking
        self._initialized = False
        self.telegram_app: Optional[Application] = None

    def set_telegram_app(self, app: Application):
        self.telegram_app = app

    async def initialize(self):
        if self._initialized:
            return
        self._initialized = True

    async def analyze_symbol(self, symbol: str, timeframe: str) -> List[TradingSignal]:
        try:
            data = await self.market_data_provider.fetch_ohlcv_data(symbol, timeframe, limit=300)
            if data is None or data.empty or len(data) < 100:
                logger.warning(f"Insufficient data for {symbol} on {timeframe}")
                return []

            all_signals = await self.signal_generator.generate_signals(data, symbol, timeframe)
            min_confidence = self.config.get("min_confidence_score", 0)
            if min_confidence is None:
                min_confidence = 0
            
            qualified_signals = [s for s in all_signals if s.confidence_score >= min_confidence]

            return qualified_signals
        except Exception as e:
            logger.error(f"Analysis failed for {symbol} on {timeframe}: {e}", exc_info=True)
            return []

    async def find_best_signals_for_timeframe(self, timeframe: str) -> List[TradingSignal]:
        symbols = self.config.get("symbols", [])
        if not symbols:
            return []
        tasks = [self.analyze_symbol(symbol, timeframe) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_signals = []
        for res in results:
            if isinstance(res, list):
                all_signals.extend(res)
            elif isinstance(res, Exception):
                logger.error(f"An exception occurred during symbol analysis: {res}")

        ranked_signals = self.signal_ranking.rank_signals(all_signals)
        return ranked_signals

    async def get_comprehensive_analysis(self) -> List[TradingSignal]:
        timeframes = self.config.get("timeframes", TIME_FRAMES)
        if not timeframes:
            return []
        all_signals = []
        
        tasks = [self.find_best_signals_for_timeframe(tf) for tf in timeframes]
        results_per_timeframe = await asyncio.gather(*tasks)

        for result_list in results_per_timeframe:
            all_signals.extend(result_list)
            
        return self.signal_ranking.rank_signals(all_signals)
        
    async def run_comprehensive_analysis_task(self, chat_id: int, message_id: Optional[int] = None):
        if not self.telegram_app:
            logger.error("Telegram application not set in TradingBotService.")
            return

        if message_id:
            try:
                await self.telegram_app.bot.edit_message_text(
                    chat_id=chat_id, message_id=message_id, text="🚀 Starting full analysis... This may take a moment."
                )
            except BadRequest:
                logger.warning("Could not edit initial message, it may have been deleted.")

        all_signals = await self.get_comprehensive_analysis()

        if all_signals:
            best_signal = all_signals[0]
            await self.telegram_app.bot.send_message(
                chat_id, "--- 👑 Top Signal Found Across All Timeframes ---"
            )
            await self.send_signal_to_chat(chat_id, best_signal)
            final_message = "🏁 Full analysis complete. The best signal is shown above."
        else:
            final_message = "🏁 Full analysis complete. No strong signals found in the current market conditions."
        
        if message_id:
            try:
                await self.telegram_app.bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=final_message)
            except BadRequest:
                await self.telegram_app.bot.send_message(chat_id=chat_id, text=final_message)
        else:
            await self.telegram_app.bot.send_message(chat_id=chat_id, text=final_message)


    async def run_find_best_signals_task(self, timeframe: str, chat_id: int, message_id: Optional[int] = None):
        if not self.telegram_app:
            logger.error("Telegram application not set in TradingBotService.")
            return
        
        if message_id:
            try:
                await self.telegram_app.bot.edit_message_text(
                    chat_id=chat_id, message_id=message_id, text=f"⏳ Running quick scan for {timeframe} timeframe..."
                )
            except BadRequest:
                 logger.warning(f"Could not edit message, it might have been deleted.")

        signals = await self.find_best_signals_for_timeframe(timeframe)
        
        if signals:
            best_signal = signals[0]
            await self.telegram_app.bot.send_message(chat_id, f"--- 👑 Top Signal for {timeframe} ---")
            await self.send_signal_to_chat(chat_id, best_signal)
            final_message = f"⚡ Quick scan for {timeframe} complete. The best signal is shown above."
        else:
            final_message = f"No strong signals found on the {timeframe} timeframe right now."
        
        if message_id:
            try:
                await self.telegram_app.bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=final_message)
            except BadRequest:
                await self.telegram_app.bot.send_message(chat_id=chat_id, text=final_message)
        else:
            await self.telegram_app.bot.send_message(chat_id=chat_id, text=final_message)

    async def cleanup(self):
        if hasattr(self, "market_data_provider") and self.market_data_provider and self._initialized:
            await self.market_data_provider.close()
        self._initialized = False
        if self.signal_generator and self.signal_generator.lstm_model_manager:
            self.signal_generator.lstm_model_manager.cleanup()
            
    def format_signal_message(self, signal: TradingSignal) -> str:
        signal_emoji = "🟢" if signal.signal_type == SignalType.BUY else "🔴"
        message = f"{signal_emoji} *{signal.symbol} ({signal.timeframe}) - {signal.signal_type.value.upper()}*\n\n"
        
        message += f"✨ *Confidence:* {signal.confidence_score:.2f}%\n"
        message += f"📈 *Predicted Profit:* {signal.predicted_profit:.2f}%\n"
        message += f"⚖️ *Risk/Reward Ratio:* {signal.risk_reward_ratio:.2f}\n\n"

        if signal.dynamic_levels:
            dl = signal.dynamic_levels
            message += "🎯 *Dynamic Levels:*\n"
            message += f"  - Entry: `{dl.get('primary_entry', 0):.4f}`\n"
            message += f"  - Take Profit: `{dl.get('primary_exit', 0):.4f}`\n"
            message += f"  - Stop Loss: `{dl.get('tight_stop', 0):.4f}`\n\n"

        if signal.market_context:
            mc = signal.market_context
            trend = mc.get('trend', 'N/A').value.replace('_', ' ').title()
            strength = mc.get('trend_strength', 'N/A').value.title()
            volatility = mc.get('volatility', 0)
            message += "📊 *Market Context:*\n"
            message += f"  - Trend: {trend} ({strength})\n"
            message += f"  - Volatility: {volatility:.2f}%\n\n"
        
        if signal.fundamental_analysis:
            fa = signal.fundamental_analysis
            message += "🏢 *Fundamental Analysis:*\n"
            message += f"  - Market Cap: `${fa.market_cap:,.0f}`\n"
            message += f"  - Dev Score: `{fa.developer_score:.1f}`\n"
            if signal.trending_data and signal.symbol.split('/')[0] in signal.trending_data.coingecko_trending:
                message += "  - Trending on CoinGecko 🔥\n"
            message += "\n"

        if signal.derivatives_analysis:
            da = signal.derivatives_analysis
            message += "📈 *Derivatives Analysis:*\n"
            if da.funding_rate is not None: message += f"  - Funding Rate: `{da.funding_rate * 100:.4f}%`\n"
            if da.open_interest is not None: message += f"  - Open Interest: `${da.open_interest:,.0f}`\n"
            if da.taker_long_short_ratio is not None: message += f"  - Taker L/S Ratio: `{da.taker_long_short_ratio:.3f}`\n"
            
            if da.binance_futures_data:
                bfd = da.binance_futures_data
                if bfd.top_trader_long_short_ratio_accounts: message += f"  - Top Trader Acc L/S: `{bfd.top_trader_long_short_ratio_accounts:.3f}`\n"
                if bfd.top_trader_long_short_ratio_positions: message += f"  - Top Trader Pos L/S: `{bfd.top_trader_long_short_ratio_positions:.3f}`\n"
                
                liq_sells = sum(float(o['origQty']) for o in bfd.liquidation_orders if o['side'] == 'SELL')
                liq_buys = sum(float(o['origQty']) for o in bfd.liquidation_orders if o['side'] == 'BUY')
                if liq_sells > 0 or liq_buys > 0: message += f"  - Liquidations (S/B): `{liq_sells:,.0f}` / `{liq_buys:,.0f}`\n"
            message += "\n"

        if signal.order_book:
            ob = signal.order_book
            imbalance = ob.total_bid_volume / ob.total_ask_volume if ob.total_ask_volume > 0 else 1
            message += "📚 *Order Book (Top 100):*\n"
            message += f"  - Bid/Ask Spread: `{ob.bid_ask_spread:.4f}`\n"
            message += f"  - Volume Imbalance (Bid/Ask): `{imbalance:.2f}`\n\n"

        if signal.macro_data:
            md = signal.macro_data
            message += "🌍 *Macro Data:*\n"
            if md.cpi: message += f"  - CPI: `{md.cpi:.2f}`\n"
            if md.fed_rate: message += f"  - Fed Rate: `{md.fed_rate:.2f}%`\n"
            if md.treasury_yield_10y: message += f"  - 10Y Yield: `{md.treasury_yield_10y:.2f}%`\n"
            message += "\n"

        if signal.reasons:
            message += "🧠 *Key Reasons for Score:*\n"
            for reason in signal.reasons[:5]: # Limit to top 5 reasons
                message += f"  • _{reason}_\n"
            message += "\n"

        return message

    async def send_signal_to_chat(self, chat_id: int, signal: TradingSignal):
        if not self.telegram_app:
            logger.error("Telegram application not set, cannot send message.")
            return
            
        message = self.format_signal_message(signal)
        escaped_message = escape_markdown_v2(message)
        
        try:
            await self.telegram_app.bot.send_message(
                chat_id, text=escaped_message, parse_mode=ParseMode.MARKDOWN_V2
            )
        except BadRequest as e:
            logger.error(f"Failed to send signal message for {signal.symbol}: {e}. Retrying without markdown.")
            try:
                unescaped_text = re.sub(r'\\(.)', r'\1', escaped_message)
                await self.telegram_app.bot.send_message(chat_id, text=unescaped_text, parse_mode=None)
            except Exception as final_e:
                logger.error(f"Failed to send signal message even without markdown: {final_e}")


class TelegramBotHandler:
    def __init__(
        self,
        bot_token: str,
        config_manager: ConfigManager,
        trading_service: TradingBotService,
    ):
        self.bot_token = bot_token
        self.config_manager = config_manager
        self.trading_service = trading_service
        self.application: Optional[Application] = None
        self.background_tasks: Set[asyncio.Task] = set()

    def create_application(self, background_tasks: Set[asyncio.Task]) -> Application:
        self.background_tasks = background_tasks
        app_builder = Application.builder().token(self.bot_token)
        app_builder.connect_timeout(30).read_timeout(30).write_timeout(30)
        self.application = app_builder.build()
        self.trading_service.set_telegram_app(self.application)
        self._register_handlers()
        return self.application

    async def initialize(self):
        await self.trading_service.initialize()

    async def cleanup(self):
        await self.trading_service.cleanup()
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.background_tasks, return_exceptions=True)

    def _register_handlers(self):
        if not self.application:
            return
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        self.application.add_error_handler(self.error_handler)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.help_command(update, context)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            [
                InlineKeyboardButton("⚡️ Quick Scan (1h)", callback_data="scan_1h"),
                InlineKeyboardButton("🚀 Full Scan", callback_data="full_scan"),
            ],
            [
                InlineKeyboardButton("⚙️ Configuration", callback_data="config_menu"),
                InlineKeyboardButton("📊 View Symbols", callback_data="config_view_symbols"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        if update.message:
            await update.message.reply_text("Welcome to Yuj Bot! Please choose a command:", reply_markup=reply_markup)
        elif update.callback_query and update.callback_query.message:
             await update.callback_query.message.reply_text("Welcome to Yuj Bot! Please choose a command:", reply_markup=reply_markup)

    def _create_background_task(self, coro):
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        if not query or not query.message:
            return
        await query.answer()
        
        chat_id = query.message.chat_id
        message_id = query.message.message_id
        
        if query.data == "scan_1h":
            await query.edit_message_text(text="Request for quick scan on 1h received. Starting soon...")
            self._create_background_task(
                self.trading_service.run_find_best_signals_task("1h", chat_id, message_id)
            )
        elif query.data == "full_scan":
            await query.edit_message_text(text="Full analysis request received. This might take some time...")
            self._create_background_task(
                self.trading_service.run_comprehensive_analysis_task(chat_id, message_id)
            )
        elif query.data == "config_menu":
            keyboard = [
                [InlineKeyboardButton("View Config", callback_data="config_view")],
                [InlineKeyboardButton("Edit Symbols (Not Implemented)", callback_data="noop")],
                [InlineKeyboardButton("⬅️ Back", callback_data="main_menu")],
            ]
            await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup(keyboard))
        elif query.data == "config_view":
            config_str = "\n".join(f"{k}: {v}" for k, v in self.config_manager.config.items())
            escaped_config = escape_markdown_v2(f"Current Config:\n{config_str}")
            keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data="config_menu")]]
            await query.edit_message_text(text=escaped_config, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=InlineKeyboardMarkup(keyboard))
        elif query.data == "config_view_symbols":
            symbols = self.config_manager.get("symbols", [])
            symbols_text = "Configured Symbols:\n" + ", ".join(symbols or [])
            escaped_symbols = escape_markdown_v2(symbols_text)
            keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data="main_menu")]]
            await query.edit_message_text(text=escaped_symbols, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=InlineKeyboardMarkup(keyboard))
        elif query.data == "main_menu":
            keyboard = [
                [
                    InlineKeyboardButton("⚡️ Quick Scan (1h)", callback_data="scan_1h"),
                    InlineKeyboardButton("🚀 Full Scan", callback_data="full_scan"),
                ],
                [
                    InlineKeyboardButton("⚙️ Configuration", callback_data="config_menu"),
                    InlineKeyboardButton("📊 View Symbols", callback_data="config_view_symbols"),
                ],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text("Welcome to Yuj Bot! Please choose a command:", reply_markup=reply_markup)
        elif query.data == "noop":
            await query.answer("This feature is not implemented yet.", show_alert=True)

    async def run_scheduled_analysis(self):
        chat_id_str = Config.TELEGRAM_CHAT_ID
        if not chat_id_str:
            logger.warning("Scheduled analysis skipped: TELEGRAM_CHAT_ID not set.")
            return
        
        try:
            chat_id = int(chat_id_str)
        except (ValueError, TypeError):
            logger.error(f"Invalid TELEGRAM_CHAT_ID: {chat_id_str}. Must be an integer.")
            return

        if not self.application:
            logger.error("Telegram application not available for scheduled analysis.")
            return

        logger.info("Starting scheduled analysis...")
        await self.application.bot.send_message(chat_id, "Running scheduled analysis...")
        
        signals = await self.trading_service.get_comprehensive_analysis()
        if signals:
            await self.trading_service.send_signal_to_chat(chat_id, signals[0])
        else:
            await self.application.bot.send_message(chat_id, "Scheduled analysis complete. No new strong signals found.")

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)