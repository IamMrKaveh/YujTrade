from typing import Dict, List
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import asyncio
from exchanges.constants import TIME_FRAMES
from exchanges.manager import ExchangeManager
from logger_config import logger
from models.signal import SignalType, TradingSignal
from signals.manager import SignalGenerator, SignalRanking
from telegrams.config import ConfigManager
from telegrams.constants import BOT_TOKEN


class TradingBotService:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.exchange_manager = ExchangeManager()
        self.signal_generator = SignalGenerator()
        self.signal_ranking = SignalRanking()
    
    async def analyze_symbol(self, symbol: str, timeframe: str) -> List[TradingSignal]:
        logger.info(f"🔍 Starting analysis for {symbol} on {timeframe}")
        
        try:
            data = await self.exchange_manager.fetch_ohlcv_data(symbol, timeframe)
            
            if data.empty:
                logger.warning(f"⚠️ No data available for {symbol} on {timeframe}")
                return []
            
            if len(data) < 50:
                logger.warning(f"⚠️ Insufficient data for {symbol} on {timeframe}: {len(data)} candles")
                return []
            
            all_signals = self.signal_generator.generate_signals(data, symbol, timeframe)
            
            min_confidence = self.config.get('min_confidence_score', 60)
            qualified_signals = [s for s in all_signals if s.confidence_score >= min_confidence]
            
            if qualified_signals:
                logger.info(f"✅ Analysis complete for {symbol} on {timeframe}: {len(qualified_signals)} qualified signals")
            else:
                logger.debug(f"ℹ️ No qualified signals for {symbol} on {timeframe} (min confidence: {min_confidence})")
            
            return qualified_signals
            
        except Exception as e:
            logger.error(f"❌ Analysis failed for {symbol} on {timeframe}: {e}")
            return []
    
    async def find_best_signals_for_timeframe(self, timeframe: str) -> List[TradingSignal]:
        logger.info(f"🚀 Starting comprehensive analysis for {timeframe} timeframe")
        
        symbols = self.config.get('symbols', [])
        logger.info(f"📊 Analyzing {len(symbols)} symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
        
        all_signals = []
        successful_analyses = 0
        failed_analyses = 0
        
        tasks = [self.analyze_symbol(symbol, timeframe) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"❌ Analysis failed for {symbol}: {result}")
                failed_analyses += 1
                continue
            
            if isinstance(result, list):
                all_signals.extend(result)
                successful_analyses += 1
                if result:
                    logger.debug(f"✅ {symbol}: {len(result)} signals found")
        
        logger.info(f"📈 Analysis summary for {timeframe}: {successful_analyses} successful, {failed_analyses} failed")
        
        if not all_signals:
            logger.info(f"ℹ️ No signals found in {timeframe} timeframe")
            return []
        
        ranked_signals = self.signal_ranking.rank_signals(all_signals)
        max_signals = self.config.get('max_signals_per_timeframe', 3)
        top_signals = ranked_signals[:max_signals]
        
        logger.info(f"🏆 Top {len(top_signals)} signals selected for {timeframe}")
        for i, signal in enumerate(top_signals, 1):
            logger.info(f"  #{i}: {signal.symbol} {signal.signal_type.value.upper()} "
                        f"(confidence: {signal.confidence_score:.0f}, profit: {signal.predicted_profit:.2f}%)")
        
        return top_signals
    
    async def get_comprehensive_analysis(self) -> Dict[str, List[TradingSignal]]:
        results = {}
        
        for timeframe in TIME_FRAMES:
            logger.info(f"Analyzing timeframe: {timeframe}")
            signals = await self.find_best_signals_for_timeframe(timeframe)
            results[timeframe] = signals
        
        return results
    
    async def cleanup(self):
        await self.exchange_manager.close_exchange()

class MessageFormatter:
    @staticmethod
    def format_signal_message(signal: TradingSignal) -> str:
        """Format signal to the exact required format"""
        emoji_map = {
            SignalType.BUY: "🟢",
            SignalType.SELL: "🔴",
            SignalType.HOLD: "🟡"
        }
        
        trend_emoji_map = {
            "bullish": "📈",
            "bearish": "📉",
            "sideways": "➡️"
        }
        
        strength_emoji_map = {
            "strong": "💪",
            "moderate": "🔄",
            "weak": "📊"
        }
        
        signal_emoji = emoji_map.get(signal.signal_type, "⚪")
        trend_emoji = trend_emoji_map.get(signal.market_context.get('trend', 'sideways'), "➡️")
        strength_emoji = strength_emoji_map.get(signal.market_context.get('trend_strength', 'weak'), "📊")
        
        reasons_text = "\n• ".join(signal.reasons)
        
        # EXACT FORMAT AS REQUESTED
        message = (
            f"{signal_emoji} **{signal.signal_type.value.upper()} SIGNAL**\n\n"
            f"📊 **Symbol:** `{signal.symbol}`\n"
            f"⏰ **Timeframe:** `{signal.timeframe}`\n\n"
            "🎯 **Dynamic Entry Levels:**\n"
            f"• Primary Entry: `${signal.dynamic_levels['primary_entry']:.4f}`\n"
            f"• Secondary Entry: `${signal.dynamic_levels['secondary_entry']:.4f}`\n\n"
            "💰 **Dynamic Exit Levels:**\n"
            f"• Primary Target: `${signal.dynamic_levels['primary_exit']:.4f}`\n"
            f"• Secondary Target: `${signal.dynamic_levels['secondary_exit']:.4f}`\n\n"
            "🛑 **Dynamic Stop Levels:**\n"
            f"• Tight Stop: `${signal.dynamic_levels['tight_stop']:.4f}`\n"
            f"• Wide Stop: `${signal.dynamic_levels['wide_stop']:.4f}`\n"
            f"• Trailing Stop: `${signal.dynamic_levels['trailing_stop']:.4f}`\n\n"
            "⚡ **Advanced Levels:**\n"
            f"• Breakeven Point: `${signal.dynamic_levels['breakeven_point']:.4f}`\n\n"
            "📈 **Profit Analysis:**\n"
            f"• Expected Profit: `{signal.predicted_profit:.2f}%`\n"
            f"• Risk/Reward Ratio: `{signal.risk_reward_ratio:.2f}`\n"
            f"• Confidence Score: `{signal.confidence_score:.0f}/100`\n\n"
            f"{trend_emoji} **Market Context:**\n"
            f"• Trend: {signal.market_context.get('trend', 'Unknown').title()} {strength_emoji}\n"
            f"• Trend Strength: {signal.market_context.get('trend_strength', 'Unknown').title()}\n"
            f"• Volatility: {signal.market_context.get('volatility', 0):.1%}\n"
            f"• Volume Trend: {signal.market_context.get('volume_trend', 'Unknown').title()}\n"
            f"• Momentum Score: {signal.market_context.get('momentum_score', 0):.2f}%\n"
            f"• Trend Acceleration: {signal.market_context.get('trend_acceleration', 0):.2f}%\n"
            f"• Volume Confirmation: {'✅' if signal.market_context.get('volume_confirmation', False) else '❌'}\n\n"
            f"📋 **Analysis Reasons:**\n• {reasons_text}\n\n"
            f"🕐 **Generated:** {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        return message
    
    @staticmethod
    def format_summary_message(timeframe_results: Dict[str, List[TradingSignal]]) -> str:
        """Simplified summary message"""
        total_signals = sum(len(signals) for signals in timeframe_results.values())
        
        if total_signals == 0:
            return "📊 No signals found in any timeframe."
        
        summary = f"📊 Found {total_signals} signal(s) across all timeframes.\n\n"
        
        for timeframe, signals in timeframe_results.items():
            if signals:
                summary += f"⏰ {timeframe.upper()}: {len(signals)} signal(s)\n"
        
        return summary


class TelegramBotHandler:
    def __init__(self, bot_token: str, config_manager: ConfigManager):
        self.bot_token = bot_token
        self.config = config_manager
        self.trading_service = TradingBotService(config_manager)
        self.formatter = MessageFormatter()
        self.user_sessions = {}
    
    def create_application(self) -> Application:
        application = Application.builder().token(self.bot_token).build()
        
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("config", self.config_command))
        application.add_handler(CommandHandler("quick", self.quick_analysis))
        application.add_handler(CallbackQueryHandler(self.button_callback))
        
        return application
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        logger.info(f"User {user_id} started the bot")
        
        keyboard = [
            [
                InlineKeyboardButton("🚀 Full Analysis", callback_data="full_analysis"),
                InlineKeyboardButton("⚡ Quick Scan", callback_data="quick_scan")
            ],
            [
                InlineKeyboardButton("⚙️ Settings", callback_data="settings")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        welcome_message = (
            "🤖 **Trading Signal Bot**\n\n"
            "Choose an option to get trading signals:"
        )
        
        await update.message.reply_text(
            welcome_message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()
        
        if query.data == "full_analysis":
            await self.run_full_analysis(query)
        elif query.data == "quick_scan":
            await self.run_quick_scan(query)
        elif query.data == "settings":
            await self.show_settings(query)
    
    async def run_full_analysis(self, query) -> None:
        user_id = query.from_user.id
        logger.info(f"User {user_id} started full analysis")
        
        try:
            timeframes = self.config.get('timeframes', TIME_FRAMES)
            
            results = {}
            total_signals = 0
            
            for i, timeframe in enumerate(timeframes, 1):
                await query.edit_message_text(
                    f"🔄 Analyzing {timeframe}... ({i}/{len(timeframes)})",
                    parse_mode='Markdown'
                )
                
                signals = await self.trading_service.find_best_signals_for_timeframe(timeframe)
                results[timeframe] = signals
                total_signals += len(signals)
            
            await query.edit_message_text(
                f"✅ Analysis complete. Found {total_signals} signal(s).",
                parse_mode='Markdown'
            )
            
            # Send only signals in the required format
            signal_count = 0
            for timeframe, signals in results.items():
                for signal in signals:
                    signal_count += 1
                    signal_message = self.formatter.format_signal_message(signal)
                    
                    await query.message.reply_text(
                        signal_message,
                        parse_mode='Markdown'
                    )
                    
                    await asyncio.sleep(1)
            
            if signal_count == 0:
                await query.message.reply_text(
                    "No signals found.",
                    parse_mode='Markdown'
                )
            
        except Exception as e:
            logger.error(f"Full analysis failed for user {user_id}: {e}")
            await query.edit_message_text(
                f"❌ Analysis Error: {str(e)}",
                parse_mode='Markdown'
            )
    
    async def run_quick_scan(self, query) -> None:
        await query.edit_message_text(
            "⚡ Quick scan in progress...",
            parse_mode='Markdown'
        )
        
        try:
            signals = await self.trading_service.find_best_signals_for_timeframe('1m')
            
            if not signals:
                await query.edit_message_text(
                    "❌ No signals found in 1m timeframe",
                    parse_mode='Markdown'
                )
                return
            
            await query.edit_message_text(
                f"✅ Found {len(signals)} signal(s)",
                parse_mode='Markdown'
            )
            
            # Send only signals in the required format
            for signal in signals:
                signal_message = self.formatter.format_signal_message(signal)
                
                await query.message.reply_text(
                    signal_message,
                    parse_mode='Markdown'
                )
                
                await asyncio.sleep(0.5)
        
        except Exception as e:
            logger.error(f"Error in quick scan: {e}")
            await query.edit_message_text(
                f"❌ Quick Scan Error: {str(e)}",
                parse_mode='Markdown'
            )
    
    async def show_settings(self, query) -> None:
        config_info = (
            f"⚙️ **Settings**\n\n"
            f"📊 Symbols: {len(self.config.get('symbols', []))}\n"
            f"⏰ Timeframes: {', '.join(self.config.get('timeframes', []))}\n"
            f"🎯 Min Confidence: {self.config.get('min_confidence_score', 60)}\n"
        )
        
        await query.edit_message_text(config_info, parse_mode='Markdown')
    
    async def config_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        config_text = (
            f"⚙️ **Configuration**\n\n"
            f"• Symbols: {len(self.config.get('symbols', []))}\n"
            f"• Timeframes: {', '.join(self.config.get('timeframes', []))}\n"
            f"• Min Confidence: {self.config.get('min_confidence_score', 60)}\n"
        )
        
        await update.message.reply_text(config_text, parse_mode='Markdown')
    
    async def quick_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        progress_msg = await update.message.reply_text(
            "⚡ Quick analysis starting...",
            parse_mode='Markdown'
        )
        
        try:
            signals = await self.trading_service.find_best_signals_for_timeframe('1h')
            
            if not signals:
                await progress_msg.edit_text(
                    "❌ No signals found in 1H timeframe",
                    parse_mode='Markdown'
                )
                return
            
            await progress_msg.edit_text(
                f"✅ Found {len(signals)} signal(s)",
                parse_mode='Markdown'
            )
            
            # Send only signals in the required format
            for signal in signals:
                signal_message = self.formatter.format_signal_message(signal)
                await update.message.reply_text(signal_message, parse_mode='Markdown')
                await asyncio.sleep(0.5)
        
        except Exception as e:
            logger.error(f"Error in quick analysis command: {e}")
            await progress_msg.edit_text(
                f"❌ Quick Analysis Error: {str(e)}",
                parse_mode='Markdown'
            )
    
    async def cleanup(self):
        await self.trading_service.cleanup()

def main_telegram():
    logger.info("🚀 Starting Trading Signal Bot...")
    
    if not BOT_TOKEN:
        logger.error("❌ BOT_TOKEN environment variable is required")
        return
    
    try:
        config_manager = ConfigManager()
        logger.info("⚙️ Configuration loaded successfully")
        
        bot_handler = TelegramBotHandler(BOT_TOKEN, config_manager)
        application = bot_handler.create_application()
        
        logger.info("🤖 Bot is ready and waiting for commands...")
        
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )
        
    except KeyboardInterrupt:
        logger.info("⏹️ Bot stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"💥 Bot crashed with error: {e}")
        import traceback
        logger.error(f"📋 Traceback:\n{traceback.format_exc()}")
    finally:
        logger.info("🧹 Starting cleanup process...")
        asyncio.run(bot_handler.cleanup())
        logger.info("✅ Bot cleanup completed successfully")