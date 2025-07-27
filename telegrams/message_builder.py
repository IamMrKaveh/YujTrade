from logger_config import logger
from telegrams.constants import ASCENDING, BEST_OPPORTUNITY_MESSAGE, DESCENDING, NEAR_FIBONACCI_LEVELS, SIGNAL_POINTS, TECHNICAL_ANALYZE
from telegrams.indicators_status import (_get_cci_status, 
                                        _get_mfi_status, _get_rsi_status,
                                        _get_stoch_status, _get_trend_status,
                                        _get_volume_status, _get_williams_status)

def _build_signal_message(sig):
    """Build complete signal message"""
    message = _build_basic_signal_message(sig)
    message = _add_technical_indicators(message, sig)
    message = _add_additional_info(message, sig)
    return message

def _build_basic_signal_message(sig):
    """Build basic signal information message"""
    emoji = '📈' if sig['type'] == 'Long' else '📉'
    type_color = '🟢' if sig['type'] == 'Long' else '🔴'
    profit_pct, loss_pct = _calculate_profit_loss_percentages(sig)
    
    message = BEST_OPPORTUNITY_MESSAGE
    message += f"{emoji} *{sig['type']} {sig['symbol']}* {type_color}\n"
    message += f"🏆 **امتیاز دقت: {sig['accuracy_score']}/100**\n\n"
    message += "📊 **جزئیات معاملاتی:**\n"
    message += f"💰 قیمت ورودی: `{sig['entry']:.6f}`\n"
    message += f"🎯 هدف قیمت: `{sig['target']:.6f}` (+{profit_pct:.1f}%)\n"
    message += f"🛑 حد ضرر: `{sig['stop_loss']:.6f}` (-{loss_pct:.1f}%)\n\n"
    
    return message

def _add_technical_indicators(message, sig):
    """Add technical indicators to message"""
    message += TECHNICAL_ANALYZE
    
    # RSI
    message += f"• RSI: `{sig['rsi']:.1f}` {_get_rsi_status(sig['rsi'])}\n"
    
    # MACD
    macd_status = ASCENDING if sig['macd'] > 0 else DESCENDING
    message += f"• MACD: `{sig['macd']:.6f}` {macd_status}\n"
    
    # Optional indicators
    if 'stoch_k' in sig:
        message += f"• Stochastic K: `{sig['stoch_k']:.1f}` {_get_stoch_status(sig['stoch_k'])}\n"
    
    if 'mfi' in sig:
        message += f"• MFI: `{sig['mfi']:.1f}` {_get_mfi_status(sig['mfi'])}\n"
    
    if 'cci' in sig:
        message += f"• CCI: `{sig['cci']:.1f}` {_get_cci_status(sig['cci'])}\n"
    
    if 'williams_r' in sig:
        message += f"• Williams %R: `{sig['williams_r']:.1f}` {_get_williams_status(sig['williams_r'])}\n"
    
    if 'volume_ratio' in sig:
        message += f"• نسبت حجم: `{sig['volume_ratio']:.1f}x` {_get_volume_status(sig['volume_ratio'])}\n"
    
    return message

def _add_additional_info(message, sig):
    """Add additional signal information"""
    message += f"• روش تحلیل: `{sig['method']}`\n"
    message += f"• قدرت سیگنال: {'⭐' * sig['strength']} ({sig['strength']}/5)\n"
    
    if 'trend_direction' in sig:
        trend_status = _get_trend_status(sig['trend_direction'])
        message += f"• جهت ترند: {trend_status} (قدرت: {abs(sig['trend_direction']):.1f})\n"
    
    if 'fibonacci_levels' in sig and sig['fibonacci_levels']:
        message += NEAR_FIBONACCI_LEVELS
        for level in sig['fibonacci_levels']:
            message += f"• {level}\n"
    
    if 'buy_score' in sig and 'sell_score' in sig:
        message += SIGNAL_POINTS
        message += f"• امتیاز خرید: `{sig['buy_score']}`\n"
        message += f"• امتیاز فروش: `{sig['sell_score']}`\n"
        score_diff = sig['buy_score'] - sig['sell_score']
        preference = 'خرید' if sig['buy_score'] > sig['sell_score'] else 'فروش'
        message += f"• برتری: {score_diff:+d} به نفع {preference}\n"
    
    message += f"\n⏰ زمان تولید سیگنال: `{sig['timestamp']}`\n"
    return message

def _build_status_message(exchange_status, exchange_error, symbols_count, current_time, python_version):
    """Build the complete status message"""
    message = "🤖 *وضعیت ربات:*\n\n"
    message += "🟢 ربات فعال است\n"
    message += f"📈 تعداد نمادها: `{symbols_count}`\n"
    message += f"🔗 صرافی CoinEx: {exchange_status}\n"
    
    if exchange_error:
        message += f"⚠️ جزئیات خطا: `{exchange_error[:50]}...`\n"
        
    message += f"⏰ آخرین بررسی: `{current_time}`\n"
    message += f"🐍 Python: `{python_version}`\n\n"
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
    
    return message

async def _send_status_message(update, message, username):
    """Send status message with fallback"""
    try:
        await update.message.reply_text(message, parse_mode='Markdown')
        logger.info(f"Status message sent successfully to user {username}")
    except Exception as e:
        logger.error(f"Error sending status message: {e}")
        plain_message = message.replace('*', '').replace('`', '')
        await update.message.reply_text(plain_message)

async def _send_error_message(update, e):
    """Send error message with fallback"""
    try:
        error_message = "❌ خطایی در نمایش وضعیت رخ داد.\n"
        error_message += f"🔧 کد خطا: `{type(e).__name__}`\n"
        error_message += "💡 لطفاً دوباره تلاش کنید یا با پشتیبانی تماس بگیرید."
        await update.message.reply_text(error_message, parse_mode='Markdown')
    except Exception as fallback_error:
        logger.error(f"Failed to send error message: {fallback_error}")
        await update.message.reply_text("خطایی در نمایش وضعیت رخ داد.")
        
def _calculate_profit_loss_percentages(sig):
    """Calculate profit and loss percentages"""
    if sig['type'] == 'Long':
        profit_pct = ((sig['target'] - sig['entry']) / sig['entry']) * 100
        loss_pct = ((sig['entry'] - sig['stop_loss']) / sig['entry']) * 100
    else:
        profit_pct = ((sig['entry'] - sig['target']) / sig['entry']) * 100
        loss_pct = ((sig['stop_loss'] - sig['entry']) / sig['entry']) * 100
    return profit_pct, loss_pct


