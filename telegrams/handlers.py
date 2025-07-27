import asyncio
from typing import Optional
from telegram import Update
from telegram.ext import ContextTypes

from exchange.exchange_config import SYMBOLS
from logger_config import logger
from market.main import analyze_market
from .background import BackgroundAnalyzer
from .constants import ERROR_MESSAGE, WAIT_MESSAGE
from .message_builder import MessageBuilder
from .system_info import SystemInfo

async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command with improved performance"""
    user_info = self._extract_user_info(update)

    try:
        logger.info(
            f"User {user_info['username']} ({user_info['id']}) started analysis")

        # Send immediate response
        await self._send_safe_message(update, WAIT_MESSAGE)

        # Start background analysis without blocking
        _ = asyncio.create_task(
            self.background_analyzer.run_analysis(update, user_info)
        )

    except Exception as e:
        logger.error(
            f"Error in start command for user {user_info['username']} ({user_info['id']}): {e}", exc_info=True)
        await self._handle_command_error(update, e, "start command")

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command with optimized system checks"""
        user_info = self._extract_user_info(update)

        try:
            logger.info(
                f"Status requested by {user_info['username']} ({user_info['id']})")

            # Run system checks concurrently
            system_data = await self.system_info.get_complete_status()
            message = self.message_builder.build_status_message(system_data)

            await self._send_safe_message(update, message, parse_mode='Markdown')

        except Exception as e:
            logger.error(
                f"Error in status command for user {user_info['username']} ({user_info['id']}): {e}", exc_info=True)
            await self._handle_command_error(update, e, "status command", user_info['username'])

    async def show_symbols(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /symbols command with optimized formatting"""
        try:
            message = self.message_builder.build_symbols_message(SYMBOLS)
            await self._send_safe_message(update, message, parse_mode='Markdown')

        except Exception as e:
            logger.error(f"Error in symbols command: {e}", exc_info=True)
            await self._handle_command_error(update, e, "symbols command")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command"""
        try:
            message = self.message_builder.build_help_message()
            await self._send_safe_message(update, message, parse_mode='Markdown')

        except Exception as e:
            await self._handle_command_error(update, e, "help command")

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
            logger.warning(f"Failed to send formatted message: {e}")
            # Fallback to plain text
            plain_message = message.replace(
                '*', '').replace('`', '').replace('_', '')
            try:
                await update.message.reply_text(plain_message)
                return True
            except Exception as fallback_e:
                logger.error(f"Failed to send fallback message: {fallback_e}")
                return False

    async def _handle_command_error(self, update: Update, error: Exception,
                                    command: str, username: str = "Unknown") -> None:
        """Handle command errors consistently"""
        logger.error(
            f"Error in {command} for user {username}: {error}", exc_info=True)
        error_msg = f"{ERROR_MESSAGE}جزئیات خطا: {str(error)[:100]}"
        await self._send_safe_message(update, error_msg)
