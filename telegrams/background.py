from imports import logger, Update, asyncio
from market.main import analyze_market
from .constants import NO_SIGNAL_FOUND, WAIT_TOO_LONG_MESSAGE, ERROR_MESSAGE
from .message_builder import _build_signal_message


async def _background_analysis(update: Update, user_id: int, username: str) -> None:
    """Run market analysis in background and send results"""
    try:
        logger.info(f"Starting background analysis for user {username} ({user_id})")
        signals = await _get_market_signals()
        
        if signals and len(signals) > 0:
            for sig in signals:
                message = _build_signal_message(sig)
                message += f"\n🕒 تایم‌فریم: `{sig.get('timeframe', 'نامشخص')}`"
                await update.message.reply_text(message, parse_mode='Markdown')
        else:
            await update.message.reply_text(NO_SIGNAL_FOUND, parse_mode='Markdown')
        
        logger.info(f"Background analysis completed for user {username} ({user_id})")
        
    except asyncio.TimeoutError:
        logger.warning(f"Analysis timeout for user {username} ({user_id})")
        await update.message.reply_text(WAIT_TOO_LONG_MESSAGE)
    except Exception as e:
        logger.error(f"Error in background analysis for user {username} ({user_id}): {e}")
        await update.message.reply_text(ERROR_MESSAGE + f"جزئیات خطا: {str(e)}")

async def _get_market_signals():
    """Get market analysis signals with timeout"""
    result = analyze_market()
    if asyncio.iscoroutine(result):
        return await asyncio.wait_for(result, timeout=1800)  # حداکثر 30 دقیقه
    else:
        return result
