import os
import logging
import asyncio
import warnings
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Fix numpy compatibility issue
import sys
if hasattr(np, 'NaN'):
    pass
else:
    np.NaN = np.nan

# Now import pandas_ta after fixing numpy
try:
    import pandas_ta as ta
except ImportError as e:
    print(f"Error importing pandas_ta: {e}")
    print("Please install with: pip install pandas-ta==0.3.14b")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Telegram bot token
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '7873285133:AAHOo3L7DewCgdVZbnx8Gs4xDJRnHs_R2VI')

# Global exchange instance
exchange = None

def load_symbols():
    """Load symbols from file with error handling"""
    try:
        with open('symbols.txt', 'r', encoding='utf-8') as f:
            symbols = [line.strip().upper() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(symbols)} symbols from symbols.txt")
        return symbols
    except FileNotFoundError:
        logger.error("symbols.txt file not found. Using default symbols.")
        # Create default symbols.txt file
        default_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT', 
                          'DOT/USDT', 'LINK/USDT', 'XRP/USDT', 'LTC/USDT', 'MATIC/USDT']
        try:
            with open('symbols.txt', 'w', encoding='utf-8') as f:
                for symbol in default_symbols:
                    f.write(f"{symbol}\n")
            logger.info("Created default symbols.txt file")
        except Exception as e:
            logger.error(f"Could not create symbols.txt: {e}")
        return default_symbols
    except Exception as e:
        logger.error(f"Error loading symbols: {e}")
        return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']

# Load symbols at startup
SYMBOLS = load_symbols()

async def init_exchange():
    """Initialize exchange connection"""
    global exchange
    if exchange is None:
        try:
            exchange = ccxt.coinex({
                'apiKey': os.getenv('COINEX_API_KEY', ''),
                'secret': os.getenv('COINEX_SECRET', ''),
                'sandbox': False,
                'enableRateLimit': True,
                'timeout': 30000,  # 30 seconds timeout
                'options': {'defaultType': 'spot'}
            })
            logger.info("Exchange initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing exchange: {e}")
            return None
    return exchange

async def get_klines(symbol, interval='1h', limit=300):
    """Fetch klines data with improved error handling"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            exchange = await init_exchange()
            if exchange is None:
                return None
            
            # Validate symbol format
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
            
            # Fetch data with timeout
            ohlcv = await asyncio.wait_for(
                exchange.fetch_ohlcv(symbol, interval, limit=limit),
                timeout=15
            )
            
            if not ohlcv or len(ohlcv) < 50:
                logger.warning(f"Insufficient data for {symbol}: {len(ohlcv) if ohlcv else 0} candles")
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert to numeric types to avoid calculation issues
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any NaN values
            df = df.dropna()
            
            if len(df) < 50:
                logger.warning(f"Insufficient clean data for {symbol}: {len(df)} candles")
                return None
                
            logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
            return df
            
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching klines for {symbol}, attempt {attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching klines for {symbol}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching klines for {symbol}: {e}")
            break  # Don't retry exchange errors
        except Exception as e:
            logger.error(f"Unexpected error fetching klines for {symbol}: {e}")
            break
    
    return None

async def get_current_price(symbol):
    """Fetch current price with improved error handling"""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            exchange = await init_exchange()
            if exchange is None:
                return None
            
            # Validate symbol format
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
                
            ticker = await asyncio.wait_for(
                exchange.fetch_ticker(symbol),
                timeout=10
            )
            
            if ticker and 'last' in ticker and ticker['last'] is not None:
                return float(ticker['last'])
            else:
                logger.warning(f"No valid price data for {symbol}")
                return None
                
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching price for {symbol}, attempt {attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
    
    return None

def safe_indicator_calculation(df, func, *args, **kwargs):
    """Safely calculate indicators with error handling"""
    try:
        result = func(*args, **kwargs)
        if result is not None:
            if isinstance(result, pd.DataFrame):
                return result
            elif isinstance(result, pd.Series):
                return result
        return None
    except Exception as e:
        logger.warning(f"Error calculating indicator: {e}")
        return None

def calculate_indicators(df):
    """Calculate technical indicators with enhanced error handling"""
    try:
        if df is None or len(df) < 200:
            logger.warning(f"Insufficient data for indicators: {len(df) if df is not None else 0} candles")
            return None
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Calculate basic moving averages
        df['sma20'] = safe_indicator_calculation(df, ta.sma, df['close'], length=20)
        df['sma50'] = safe_indicator_calculation(df, ta.sma, df['close'], length=50)
        df['sma200'] = safe_indicator_calculation(df, ta.sma, df['close'], length=200)
        
        # Calculate EMAs
        df['ema12'] = safe_indicator_calculation(df, ta.ema, df['close'], length=12)
        df['ema26'] = safe_indicator_calculation(df, ta.ema, df['close'], length=26)
        
        # Calculate RSI
        rsi = safe_indicator_calculation(df, ta.rsi, df['close'], length=14)
        if rsi is not None:
            df['rsi'] = rsi
        
        # Calculate MACD
        try:
            macd_data = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd_data is not None:
                df = df.join(macd_data, how='left')
        except Exception as e:
            logger.warning(f"Error calculating MACD: {e}")
        
        # Calculate Bollinger Bands
        try:
            bbands_data = ta.bbands(df['close'], length=20, std=2)
            if bbands_data is not None:
                df = df.join(bbands_data, how='left')
        except Exception as e:
            logger.warning(f"Error calculating Bollinger Bands: {e}")
        
        # Calculate Stochastic Oscillator
        try:
            stoch_data = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
            if stoch_data is not None:
                df = df.join(stoch_data, how='left')
        except Exception as e:
            logger.warning(f"Error calculating Stochastic: {e}")
        
        # Volume indicators
        volume_sma = safe_indicator_calculation(df, ta.sma, df['volume'], length=20)
        if volume_sma is not None:
            df['volume_sma'] = volume_sma
        
        # Check if we have minimum required indicators
        required_indicators = ['rsi', 'sma50', 'volume_sma']
        missing_indicators = [ind for ind in required_indicators if ind not in df.columns or df[ind].isna().all()]
        
        if missing_indicators:
            logger.warning(f"Missing indicators: {missing_indicators}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

def check_signals(df, symbol):
    """Enhanced signal detection with fallback methods"""
    if df is None or len(df) < 2:
        return None
    
    try:
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # Check if we have RSI (minimum requirement)
        if 'rsi' not in df.columns or pd.isna(last_row['rsi']):
            logger.warning(f"No RSI data for {symbol}")
            return None
        
        # Simple RSI-based signals as fallback
        rsi_value = last_row['rsi']
        
        # Try advanced MACD signals first
        if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
            if (not pd.isna(last_row['MACD_12_26_9']) and 
                not pd.isna(last_row['MACDs_12_26_9']) and 
                not pd.isna(prev_row['MACD_12_26_9']) and 
                not pd.isna(prev_row['MACDs_12_26_9'])):
                
                # MACD crossover signals
                macd_bullish = (prev_row['MACD_12_26_9'] <= prev_row['MACDs_12_26_9'] and 
                               last_row['MACD_12_26_9'] > last_row['MACDs_12_26_9'])
                
                macd_bearish = (prev_row['MACD_12_26_9'] >= prev_row['MACDs_12_26_9'] and 
                               last_row['MACD_12_26_9'] < last_row['MACDs_12_26_9'])
                
                # Volume confirmation
                volume_confirm = True
                if 'volume_sma' in df.columns and not pd.isna(last_row['volume_sma']):
                    volume_confirm = last_row['volume'] > last_row['volume_sma'] * 1.1
                
                # Buy signal
                if macd_bullish and rsi_value < 40 and volume_confirm:
                    return {
                        'type': 'buy',
                        'strength': calculate_signal_strength(df, 'buy'),
                        'rsi': rsi_value,
                        'macd': last_row['MACD_12_26_9'],
                        'method': 'MACD_RSI'
                    }
                
                # Sell signal
                if macd_bearish and rsi_value > 60 and volume_confirm:
                    return {
                        'type': 'sell',
                        'strength': calculate_signal_strength(df, 'sell'),
                        'rsi': rsi_value,
                        'macd': last_row['MACD_12_26_9'],
                        'method': 'MACD_RSI'
                    }
        
        # Fallback to simple RSI signals
        if rsi_value < 25:  # Strong oversold
            return {
                'type': 'buy',
                'strength': 4,
                'rsi': rsi_value,
                'macd': 0,
                'method': 'RSI_Simple'
            }
        elif rsi_value > 75:  # Strong overbought
            return {
                'type': 'sell',
                'strength': 4,
                'rsi': rsi_value,
                'macd': 0,
                'method': 'RSI_Simple'
            }
        elif rsi_value < 30:  # Oversold
            return {
                'type': 'buy',
                'strength': 2,
                'rsi': rsi_value,
                'macd': 0,
                'method': 'RSI_Simple'
            }
        elif rsi_value > 70:  # Overbought
            return {
                'type': 'sell',
                'strength': 2,
                'rsi': rsi_value,
                'macd': 0,
                'method': 'RSI_Simple'
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error checking signals for {symbol}: {e}")
        return None

def calculate_signal_strength(df, signal_type):
    """Calculate signal strength based on multiple factors"""
    try:
        if df is None or len(df) == 0:
            return 2
            
        last_row = df.iloc[-1]
        strength_score = 0
        
        # RSI strength
        if 'rsi' in df.columns and not pd.isna(last_row['rsi']):
            rsi_value = last_row['rsi']
            if signal_type == 'buy':
                if rsi_value < 20:
                    strength_score += 3
                elif rsi_value < 25:
                    strength_score += 2
                elif rsi_value < 30:
                    strength_score += 1
            else:  # sell
                if rsi_value > 80:
                    strength_score += 3
                elif rsi_value > 75:
                    strength_score += 2
                elif rsi_value > 70:
                    strength_score += 1
        
        # Volume strength
        if 'volume_sma' in df.columns and not pd.isna(last_row['volume_sma']):
            try:
                volume_ratio = last_row['volume'] / last_row['volume_sma']
                if volume_ratio > 2:
                    strength_score += 2
                elif volume_ratio > 1.5:
                    strength_score += 1
            except (ZeroDivisionError, TypeError):
                pass
        
        # MACD strength
        if ('MACD_12_26_9' in df.columns and 
            not pd.isna(last_row['MACD_12_26_9'])):
            macd_value = abs(last_row['MACD_12_26_9'])
            if macd_value > 0.001:  # Strong MACD signal
                strength_score += 1
        
        # Normalize to 1-5 scale
        return min(max(strength_score, 1), 5)
        
    except Exception:
        return 2  # Default medium strength

async def analyze_market():
    """Analyze market and return signals with improved error handling"""
    signals = []
    successful_analyses = 0
    failed_analyses = 0
    
    logger.info(f"Starting market analysis for {len(SYMBOLS)} symbols")
    
    # Process symbols in batches to avoid overwhelming the API
    batch_size = 5
    for i in range(0, len(SYMBOLS), batch_size):
        batch_symbols = SYMBOLS[i:i+batch_size]
        
        for symbol in batch_symbols:
            try:
                # Add delay to respect rate limits
                await asyncio.sleep(1)
                
                logger.info(f"Analyzing {symbol}...")
                
                df = await get_klines(symbol)
                if df is None:
                    failed_analyses += 1
                    continue
                
                df = calculate_indicators(df)
                if df is None:
                    failed_analyses += 1
                    continue
                
                signal_data = check_signals(df, symbol)
                if signal_data:
                    current_price = await get_current_price(symbol)
                    if current_price is not None:
                        if signal_data['type'] == 'buy':
                            entry = current_price
                            target = entry * 1.05  # 5% target
                            stop_loss = entry * 0.96  # 4% stop loss
                            signal_type = 'Long'
                        else:  # sell
                            entry = current_price
                            target = entry * 0.95  # 5% target
                            stop_loss = entry * 1.04  # 4% stop loss
                            signal_type = 'Short'
                        
                        signals.append({
                            'symbol': symbol,
                            'type': signal_type,
                            'entry': entry,
                            'target': target,
                            'stop_loss': stop_loss,
                            'strength': signal_data['strength'],
                            'rsi': signal_data['rsi'],
                            'macd': signal_data['macd'],
                            'method': signal_data.get('method', 'Unknown'),
                            'timestamp': datetime.now().strftime('%H:%M:%S')
                        })
                        
                        logger.info(f"Signal found for {symbol}: {signal_type}")
                
                successful_analyses += 1
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                failed_analyses += 1
        
        # Small delay between batches
        if i + batch_size < len(SYMBOLS):
            await asyncio.sleep(2)
    
    logger.info(f"Analysis complete. Success: {successful_analyses}, Failed: {failed_analyses}, Signals: {len(signals)}")
    return signals

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command with improved error handling"""
    try:
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        logger.info(f"User {username} ({user_id}) started analysis")
        
        await update.message.reply_text(
            "🔍 تحلیل بازار در حال انجام...\n"
            "⏳ این کار ممکن است چند دقیقه طول بکشد."
        )
        
        # Set a timeout for the entire analysis
        try:
            signals = await asyncio.wait_for(analyze_market(), timeout=300)  # 5 minutes max
        except asyncio.TimeoutError:
            await update.message.reply_text(
                "⏱️ تحلیل بیش از حد زمان برد. لطفا دوباره تلاش کنید."
            )
            return
        
        if signals:
            # Sort signals by strength
            signals.sort(key=lambda x: x['strength'], reverse=True)
            
            message = f"📊 *فرصت‌های معاملاتی یافت شده:* ({len(signals)} سیگنال)\n\n"
            
            for i, sig in enumerate(signals, 1):
                emoji = '📈' if sig['type'] == 'Long' else '📉'
                strength_stars = '⭐' * sig['strength']
                
                message += f"{emoji} *{sig['type']} {sig['symbol']}* {strength_stars}\n"
                message += f"💰 ورودی: `{sig['entry']:.4f}`\n"
                message += f"🎯 هدف: `{sig['target']:.4f}`\n"
                message += f"🛑 حد ضرر: `{sig['stop_loss']:.4f}`\n"
                message += f"📊 RSI: `{sig['rsi']:.1f}` | روش: `{sig['method']}`\n"
                message += f"⏰ زمان: `{sig['timestamp']}`\n\n"
                
                # Limit message length
                if len(message) > 3500:
                    message += f"... و {len(signals) - i} سیگنال دیگر"
                    break
            
            message += "\n⚠️ *هشدار:* این سیگنال‌ها صرفاً جهت اطلاع‌رسانی هستند."
        else:
            message = (
                "❌ در حال حاضر هیچ سیگنال معاملاتی قوی یافت نشد.\n\n"
                "💡 ممکن است بازار در حال تثبیت باشد یا شرایط مناسب معاملاتی وجود نداشته باشد.\n"
                "🔄 چند دقیقه دیگر مجدداً تلاش کنید."
            )
        
        await update.message.reply_text(message, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in start command: {e}")
        await update.message.reply_text(
            "❌ خطایی در تحلیل بازار رخ داد. لطفا دوباره تلاش کنید.\n"
            f"جزئیات خطا: {str(e)[:100]}..."
        )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command"""
    try:
        # Test exchange connection
        exchange_status = "❌ قطع"
        try:
            test_exchange = await init_exchange()
            if test_exchange:
                await test_exchange.fetch_ticker('BTC/USDT')
                exchange_status = "✅ متصل"
        except:
            pass
        
        message = "🤖 *وضعیت ربات:*\n\n"
        message += "🟢 ربات فعال است\n"
        message += f"📈 تعداد نمادها: `{len(SYMBOLS)}`\n"
        message += f"🔗 صرافی CoinEx: {exchange_status}\n"
        message += f"⏰ آخرین بررسی: `{datetime.now().strftime('%H:%M:%S')}`\n"
        message += f"🐍 Python: `{sys.version.split()[0]}`\n\n"
        message += "💡 *دستورات موجود:*\n"
        message += "`/start` - تحلیل بازار\n"
        message += "`/status` - وضعیت ربات\n"
        message += "`/symbols` - نمایش نمادها\n"
        message += "`/help` - راهنما"
        
        await update.message.reply_text(message, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in status command: {e}")
        await update.message.reply_text("خطایی در نمایش وضعیت رخ داد.")

async def show_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /symbols command"""
    try:
        message = f"📋 *نمادهای تحت نظارت:* ({len(SYMBOLS)} نماد)\n\n"
        
        # Group symbols in rows of 3
        for i in range(0, len(SYMBOLS), 3):
            row_symbols = SYMBOLS[i:i+3]
            message += " | ".join([f"`{symbol}`" for symbol in row_symbols]) + "\n"
        
        message += f"\n💡 برای تغییر نمادها، فایل `symbols.txt` را ویرایش کنید."
        
        await update.message.reply_text(message, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in symbols command: {e}")
        await update.message.reply_text("خطایی در نمایش نمادها رخ داد.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command"""
    try:
        message = (
            "🤖 *راهنمای ربات تحلیل تکنیکال*\n\n"
            "این ربات با استفاده از اندیکاتورهای تکنیکال، فرصت‌های معاملاتی را شناسایی می‌کند.\n\n"
            "📋 *دستورات:*\n"
            "`/start` - شروع تحلیل بازار\n"
            "`/status` - نمایش وضعیت ربات\n"
            "`/symbols` - لیست نمادهای تحت نظارت\n"
            "`/help` - نمایش این راهنما\n\n"
            "📊 *اندیکاتورهای استفاده شده:*\n"
            "• RSI (Relative Strength Index)\n"
            "• MACD (Moving Average Convergence Divergence)\n"
            "• SMA (Simple Moving Average)\n"
            "• Volume Analysis\n\n"
            "⚠️ *هشدار مهم:*\n"
            "این سیگنال‌ها صرفاً جهت اطلاع‌رسانی هستند و توصیه سرمایه‌گذاری محسوب نمی‌شوند. "
            "لطفاً قبل از هر معامله، تحلیل‌های خود را انجام دهید."
        )
        
        await update.message.reply_text(message, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in help command: {e}")
        await update.message.reply_text("خطایی در نمایش راهنما رخ داد.")

async def close_exchange():
    """Close exchange connection"""
    global exchange
    if exchange:
        try:
            await exchange.close()
            logger.info("Exchange connection closed")
        except:
            pass
        exchange = None

def main() -> None:
    """Run the bot with improved error handling"""
    try:
        if not BOT_TOKEN:
            logger.error("BOT_TOKEN not found. Please set TELEGRAM_BOT_TOKEN environment variable.")
            print("Please set the TELEGRAM_BOT_TOKEN environment variable!")
            return
        
        if not SYMBOLS:
            logger.error("No symbols loaded. Please check symbols.txt file.")
            print("Please create a symbols.txt file with trading pairs!")
            return
        
        logger.info("Starting Telegram Trading Bot...")
        logger.info(f"Loaded {len(SYMBOLS)} symbols for analysis")
        
        application = ApplicationBuilder().token(BOT_TOKEN).build()
        
        # Add command handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("status", status))
        application.add_handler(CommandHandler("symbols", show_symbols))
        application.add_handler(CommandHandler("help", help_command))
        
        logger.info("Bot is ready and polling...")
        print("✅ Bot started successfully! Press Ctrl+C to stop.")
        
        # Run with graceful shutdown
        application.run_polling(
            drop_pending_updates=True,
            allowed_updates=['message'],
            stop_signals=[],  # Handle shutdown manually
        )
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        print("Bot stopped.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Fatal error: {e}")
    finally:
        # Cleanup
        try:
            asyncio.run(close_exchange())
        except:
            pass

if __name__ == '__main__':
    main()