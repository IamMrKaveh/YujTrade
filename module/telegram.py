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

        self.background_tasks.create_task(analysis_task(), name=f"QuickAnalyze-{query.message.chat_id}")

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

        self.background_tasks.create_task(analysis_task(), name=f"FullAnalyze-{query.message.chat_id}")

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
                messages = self.format_signal_message(signal)
                for message in messages:
                    try:
                        await self.application.bot.send_message(chat_id, message, parse_mode=ParseMode.MARKDOWN_V2)
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        logger.error(f"Failed to send signal message: {e}")
        else:
            await self.application.bot.send_message(
                chat_id,
                escape_markdown_v2(no_signals_text),
                parse_mode=ParseMode.MARKDOWN_V2,
            )

    def format_signal_message(self, signal: TradingSignal) -> list[str]:
        messages = []
        
        signal_type_str = signal.signal_type.value.upper()
        signal_emoji = "📈" if signal_type_str == "BUY" else "📉"
        signal_type = escape_markdown_v2(f"{signal_emoji} {signal_type_str}")
        symbol = escape_markdown_v2(signal.symbol)
        timeframe = escape_markdown_v2(signal.timeframe)
        
        header = f"*═══════════════════════*\n"
        header += f"*{signal_type} SIGNAL*\n"
        header += f"*{symbol} • {timeframe}*\n"
        header += f"*═══════════════════════*"

        confidence_score_str = escape_markdown_v2(f"{signal.confidence_score:.2f}")
        entry_price_str = escape_markdown_v2(f"{signal.entry_price:.8f}")
        exit_price_str = escape_markdown_v2(f"{signal.exit_price:.8f}")
        stop_loss_str = escape_markdown_v2(f"{signal.stop_loss:.8f}")
        risk_reward_ratio_str = escape_markdown_v2(f"{signal.risk_reward_ratio:.2f}")
        predicted_profit_str = escape_markdown_v2(f"{signal.predicted_profit:.2f}")

        main_info = (
            f"\n\n📅 *Date:* `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n"
            f"🎯 *Confidence:* `{confidence_score_str}%`\n"
            f"➡️ *Entry:* `{entry_price_str}`\n"
            f"✅ *Take Profit:* `{exit_price_str}`\n"
            f"🛑 *Stop Loss:* `{stop_loss_str}`\n"
            f"⚖️ *Risk/Reward:* `{risk_reward_ratio_str}`\n"
            f"💰 *Predicted Profit:* `{predicted_profit_str}`"
        )

        messages.append(header + main_info)

        ctx = signal.market_context
        trend = escape_markdown_v2(str(ctx.get("trend", "N/A")))
        trend_strength = escape_markdown_v2(str(ctx.get("trend_strength", "N/A")))
        condition = escape_markdown_v2(str(ctx.get("market_condition", "N/A")))
        volatility_str = escape_markdown_v2(f"{ctx.get('volatility', 0):.2f}")
        support_str = escape_markdown_v2(f"{ctx.get('support_levels', [0])[0]:.8f}" if ctx.get('support_levels') else 'N/A')
        resistance_str = escape_markdown_v2(f"{ctx.get('resistance_levels', [0])[0]:.8f}" if ctx.get('resistance_levels') else 'N/A')

        market_info = (
            f"*📊 MARKET CONTEXT*\n"
            f"├ 📈 Trend: `{trend}` \\({trend_strength}\\)\n"
            f"├ 🔄 Condition: `{condition}`\n"
            f"├ 💨 Volatility: `{volatility_str}%`\n"
            f"├ 🟢 Support: `{support_str}`\n"
            f"└ 🔴 Resistance: `{resistance_str}`"
        )

        if signal.volume_analysis:
            vol_ratio = signal.volume_analysis.get('volume_ratio', 1.0)
            vol_ratio_str = escape_markdown_v2(f"{vol_ratio:.2f}")
            vol_emoji = "📊" if vol_ratio > 1.2 else "📉"
            market_info += f"\n{vol_emoji} *Volume Ratio:* `{vol_ratio_str}x`"

        messages.append(market_info)

        if signal.derivatives_analysis:
            deriv = signal.derivatives_analysis
            deriv_info = "*🔮 DERIVATIVES DATA*\n"
            
            if deriv.funding_rate is not None:
                fr_str = escape_markdown_v2(f"{deriv.funding_rate:.6f}")
                fr_emoji = "🟢" if deriv.funding_rate < 0 else "🔴"
                deriv_info += f"├ {fr_emoji} Funding Rate: `{fr_str}`\n"
            
            if deriv.open_interest is not None:
                oi_str = escape_markdown_v2(f"{deriv.open_interest:,.0f}")
                deriv_info += f"├ 💼 Open Interest: `{oi_str}`\n"
            
            if deriv.taker_long_short_ratio is not None:
                ratio_str = escape_markdown_v2(f"{deriv.taker_long_short_ratio:.2f}")
                ratio_emoji = "🟢" if deriv.taker_long_short_ratio > 1 else "🔴"
                deriv_info += f"├ {ratio_emoji} Taker L/S Ratio: `{ratio_str}`\n"
            
            if deriv.binance_futures_data:
                bfd = deriv.binance_futures_data
                if bfd.top_trader_long_short_ratio_accounts is not None:
                    tt_acc_str = escape_markdown_v2(f"{bfd.top_trader_long_short_ratio_accounts:.2f}")
                    deriv_info += f"├ 👥 Top Trader Accounts: `{tt_acc_str}`\n"
                
                if bfd.top_trader_long_short_ratio_positions is not None:
                    tt_pos_str = escape_markdown_v2(f"{bfd.top_trader_long_short_ratio_positions:.2f}")
                    deriv_info += f"└ 📊 Top Trader Positions: `{tt_pos_str}`\n"
            
            if len(deriv_info) > len("*🔮 DERIVATIVES DATA*\n"):
                messages.append(deriv_info.rstrip('\n'))

        if signal.fundamental_analysis:
            fund = signal.fundamental_analysis
            fund_info = "*💎 FUNDAMENTAL DATA*\n"
            
            if fund.market_cap > 0:
                mcap_str = escape_markdown_v2(f"{fund.market_cap:,.0f}")
                fund_info += f"├ 💰 Market Cap: `${mcap_str}`\n"
            
            if fund.total_volume > 0:
                vol_str = escape_markdown_v2(f"{fund.total_volume:,.0f}")
                fund_info += f"├ 📊 24h Volume: `${vol_str}`\n"
            
            if fund.developer_score > 0:
                dev_str = escape_markdown_v2(f"{fund.developer_score:.1f}")
                fund_info += f"├ 👨‍💻 Developer Score: `{dev_str}`\n"
            
            if fund.community_score > 0:
                comm_str = escape_markdown_v2(f"{fund.community_score:.0f}")
                fund_info += f"└ 👥 Community Score: `{comm_str}`\n"
            
            if len(fund_info) > len("*💎 FUNDAMENTAL DATA*\n"):
                messages.append(fund_info.rstrip('\n'))

        if signal.order_book:
            ob = signal.order_book
            ob_info = "*📖 ORDER BOOK*\n"
            
            if ob.bid_ask_spread is not None:
                spread_str = escape_markdown_v2(f"{ob.bid_ask_spread:.8f}")
                ob_info += f"├ 📏 Spread: `{spread_str}`\n"
            
            if ob.total_bid_volume is not None:
                bid_vol_str = escape_markdown_v2(f"{ob.total_bid_volume:,.2f}")
                ob_info += f"├ 🟢 Total Bids: `{bid_vol_str}`\n"
            
            if ob.total_ask_volume is not None:
                ask_vol_str = escape_markdown_v2(f"{ob.total_ask_volume:,.2f}")
                ob_info += f"└ 🔴 Total Asks: `{ask_vol_str}`\n"
            
            if len(ob_info) > len("*📖 ORDER BOOK*\n"):
                messages.append(ob_info.rstrip('\n'))

        if signal.macro_data:
            macro = signal.macro_data
            macro_info = "*🌍 MACRO ECONOMICS*\n"
            
            if macro.cpi is not None:
                cpi_str = escape_markdown_v2(f"{macro.cpi:.2f}")
                macro_info += f"├ 📊 CPI: `{cpi_str}%`\n"
            
            if macro.fed_rate is not None:
                fed_str = escape_markdown_v2(f"{macro.fed_rate:.2f}")
                macro_info += f"├ 🏦 Fed Rate: `{fed_str}%`\n"
            
            if macro.treasury_yield_10y is not None:
                treasury_str = escape_markdown_v2(f"{macro.treasury_yield_10y:.2f}")
                macro_info += f"├ 💵 10Y Treasury: `{treasury_str}%`\n"
            
            if macro.gdp is not None:
                gdp_str = escape_markdown_v2(f"{macro.gdp:.2f}")
                macro_info += f"├ 📈 GDP: `{gdp_str}%`\n"
            
            if macro.unemployment is not None:
                unemp_str = escape_markdown_v2(f"{macro.unemployment:.2f}")
                macro_info += f"└ 👔 Unemployment: `{unemp_str}%`\n"
            
            if len(macro_info) > len("*🌍 MACRO ECONOMICS*\n"):
                messages.append(macro_info.rstrip('\n'))

        if signal.trending_data and signal.trending_data.coingecko_trending:
            trending_coins = signal.trending_data.coingecko_trending[:5]
            trending_str = ", ".join([escape_markdown_v2(c) for c in trending_coins])
            trending_info = f"*🔥 TRENDING COINS*\n`{trending_str}`"
            messages.append(trending_info)

        if signal.dynamic_levels:
            levels = signal.dynamic_levels
            levels_info = (
                f"*🎯 DYNAMIC LEVELS*\n"
                f"├ 🟢 Primary Entry: `{escape_markdown_v2(f'{levels.get('primary_entry', 0):.8f}')}`\n"
                f"├ 🟡 Secondary Entry: `{escape_markdown_v2(f'{levels.get('secondary_entry', 0):.8f}')}`\n"
                f"├ 🎯 Primary Exit: `{escape_markdown_v2(f'{levels.get('primary_exit', 0):.8f}')}`\n"
                f"├ 🎯 Secondary Exit: `{escape_markdown_v2(f'{levels.get('secondary_exit', 0):.8f}')}`\n"
                f"├ 🛑 Tight Stop: `{escape_markdown_v2(f'{levels.get('tight_stop', 0):.8f}')}`\n"
                f"├ 🛑 Wide Stop: `{escape_markdown_v2(f'{levels.get('wide_stop', 0):.8f}')}`\n"
                f"└ ⚖️ Breakeven: `{escape_markdown_v2(f'{levels.get('breakeven_point', 0):.8f}')}`"
            )
            messages.append(levels_info)

        top_reasons = signal.reasons[:8] if len(signal.reasons) > 8 else signal.reasons
        reasons_list = []
        for r in top_reasons:
            try:
                parts = r.split('(Score:')
                reason_text = parts[0].strip()
                if len(parts) > 1:
                    score_part = parts[1].replace(')', '').strip()
                    score_value = float(score_part)
                    
                    if score_value > 0:
                        emoji = "✅"
                    elif score_value < 0:
                        emoji = "❌"
                    else:
                        emoji = "➖"
                    
                    escaped_reason = escape_markdown_v2(f"{reason_text} (Score: {score_value:.2f})")
                    reasons_list.append(f"{emoji} {escaped_reason}")
                else:
                    reasons_list.append(f"• {escape_markdown_v2(r)}")
            except:
                reasons_list.append(f"• {escape_markdown_v2(r)}")
        
        reasons_text = "\n*🧠 KEY ANALYSIS FACTORS*\n" + "\n".join(reasons_list)
        messages.append(reasons_text)

        footer = (
            f"\n*═══════════════════════*\n"
            f"🤖 *Powered by AI Trading Bot*\n"
            f"⚠️ *Risk Warning:* Trading involves risk\\. Always use proper risk management\\."
        )
        messages.append(footer)

        return messages