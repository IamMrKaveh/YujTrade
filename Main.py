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

def calculate_signal_accuracy_score(df, signal_data, symbol):
    """محاسبه امتیاز دقت سیگنال بر اساس عوامل مختلف"""
    try:
        if df is None or len(df) < 50 or not signal_data:
            return 0
        
        last_row = df.iloc[-1]
        prev_rows = df.iloc[-10:] if len(df) >= 10 else df
        accuracy_score = 0
        
        # 1. امتیاز RSI (وزن: 25%)
        rsi_value = signal_data.get('rsi', 50)
        if signal_data['type'] == 'buy':
            if rsi_value < 20:
                accuracy_score += 25
            elif rsi_value < 25:
                accuracy_score += 20
            elif rsi_value < 30:
                accuracy_score += 15
            elif rsi_value < 35:
                accuracy_score += 10
        else:  # sell
            if rsi_value > 80:
                accuracy_score += 25
            elif rsi_value > 75:
                accuracy_score += 20
            elif rsi_value > 70:
                accuracy_score += 15
            elif rsi_value > 65:
                accuracy_score += 10
        
        # 2. امتیاز MACD (وزن: 20%)
        if ('MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns and
            not pd.isna(last_row.get('MACD_12_26_9')) and not pd.isna(last_row.get('MACDs_12_26_9'))):
            
            macd_line = last_row['MACD_12_26_9']
            signal_line = last_row['MACDs_12_26_9']
            macd_histogram = macd_line - signal_line
            
            if signal_data['type'] == 'buy' and macd_histogram > 0 and macd_line > signal_line:
                accuracy_score += 20
            elif signal_data['type'] == 'sell' and macd_histogram < 0 and macd_line < signal_line:
                accuracy_score += 20
            elif abs(macd_histogram) > 0.001:  # سیگنال قوی MACD
                accuracy_score += 10
        
        # 3. امتیاز حجم معاملات (وزن: 15%)
        if 'volume_sma' in df.columns and not pd.isna(last_row.get('volume_sma')):
            try:
                volume_ratio = last_row['volume'] / last_row['volume_sma']
                if volume_ratio > 2.5:
                    accuracy_score += 15
                elif volume_ratio > 2:
                    accuracy_score += 12
                elif volume_ratio > 1.5:
                    accuracy_score += 8
                elif volume_ratio > 1.2:
                    accuracy_score += 5
            except (ZeroDivisionError, TypeError):
                pass
        
        # 4. امتیاز میانگین متحرک (وزن: 15%)
        if all(col in df.columns for col in ['sma20', 'sma50', 'sma200']):
            current_price = last_row['close']
            sma20 = last_row.get('sma20')
            sma50 = last_row.get('sma50')
            sma200 = last_row.get('sma200')
            
            if not any(pd.isna(val) for val in [sma20, sma50, sma200]):
                if signal_data['type'] == 'buy':
                    # قیمت بالای تمام میانگین‌ها - سیگنال قوی خرید
                    if current_price > sma20 > sma50 > sma200:
                        accuracy_score += 15
                    elif current_price > sma20 > sma50:
                        accuracy_score += 10
                    elif current_price > sma20:
                        accuracy_score += 5
                else:  # sell
                    # قیمت پایین تمام میانگین‌ها - سیگنال قوی فروش
                    if current_price < sma20 < sma50 < sma200:
                        accuracy_score += 15
                    elif current_price < sma20 < sma50:
                        accuracy_score += 10
                    elif current_price < sma20:
                        accuracy_score += 5
        
        # 5. امتیاز نوسان‌گیر استوکاستیک (وزن: 10%)
        if 'STOCHk_14_3_3' in df.columns and 'STOCHd_14_3_3' in df.columns:
            k_value = last_row.get('STOCHk_14_3_3')
            d_value = last_row.get('STOCHd_14_3_3')
            
            if not pd.isna(k_value) and not pd.isna(d_value):
                if signal_data['type'] == 'buy' and k_value < 20 and d_value < 20:
                    accuracy_score += 10
                elif signal_data['type'] == 'sell' and k_value > 80 and d_value > 80:
                    accuracy_score += 10
                elif signal_data['type'] == 'buy' and k_value < 30:
                    accuracy_score += 5
                elif signal_data['type'] == 'sell' and k_value > 70:
                    accuracy_score += 5
        
        # 6. امتیاز روند کلی (وزن: 10%)
        if len(prev_rows) >= 5:
            trend_direction = 0
            close_prices = prev_rows['close'].values
            
            # بررسی روند صعودی یا نزولی
            for i in range(1, len(close_prices)):
                if close_prices[i] > close_prices[i-1]:
                    trend_direction += 1
                elif close_prices[i] < close_prices[i-1]:
                    trend_direction -= 1
            
            trend_strength = abs(trend_direction) / len(close_prices)
            
            if signal_data['type'] == 'buy' and trend_direction > 0:
                accuracy_score += int(10 * trend_strength)
            elif signal_data['type'] == 'sell' and trend_direction < 0:
                accuracy_score += int(10 * trend_strength)
        
        # 7. امتیاز اضافی برای نمادهای پرحجم (وزن: 5%)
        if symbol in ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']:
            accuracy_score += 5
        
        # محدود کردن امتیاز به 100
        accuracy_score = min(accuracy_score, 100)
        
        logger.info(f"Accuracy score for {symbol}: {accuracy_score}")
        return accuracy_score
        
    except Exception as e:
        logger.error(f"Error calculating accuracy score for {symbol}: {e}")
        return 0

async def analyze_market():
    """تحلیل بازار و بازگرداندن بهترین سیگنال"""
    all_signals = []
    successful_analyses = 0
    failed_analyses = 0
    
    logger.info(f"Starting market analysis for {len(SYMBOLS)} symbols")
    
    # پردازش نمادها به صورت دسته‌ای
    batch_size = 5
    for i in range(0, len(SYMBOLS), batch_size):
        batch_symbols = SYMBOLS[i:i+batch_size]
        
        for symbol in batch_symbols:
            try:
                # تاخیر برای رعایت محدودیت نرخ
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
                        # محاسبه امتیاز دقت
                        accuracy_score = calculate_signal_accuracy_score(df, signal_data, symbol)
                        
                        if accuracy_score >= 40:  # حداقل امتیاز قابل قبول
                            if signal_data['type'] == 'buy':
                                entry = current_price
                                target = entry * 1.05  # 5% هدف
                                stop_loss = entry * 0.96  # 4% حد ضرر
                                signal_type = 'Long'
                            else:  # sell
                                entry = current_price
                                target = entry * 0.95  # 5% هدف
                                stop_loss = entry * 1.04  # 4% حد ضرر
                                signal_type = 'Short'
                            
                            all_signals.append({
                                'symbol': symbol,
                                'type': signal_type,
                                'entry': entry,
                                'target': target,
                                'stop_loss': stop_loss,
                                'strength': signal_data['strength'],
                                'accuracy_score': accuracy_score,
                                'rsi': signal_data['rsi'],
                                'macd': signal_data['macd'],
                                'method': signal_data.get('method', 'Unknown'),
                                'timestamp': datetime.now().strftime('%H:%M:%S')
                            })
                            
                            logger.info(f"High accuracy signal found for {symbol}: {signal_type} (Score: {accuracy_score})")
                
                successful_analyses += 1
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                failed_analyses += 1
        
        # تاخیر کوتاه بین دسته‌ها
        if i + batch_size < len(SYMBOLS):
            await asyncio.sleep(2)
    
    # انتخاب بهترین سیگنال
    best_signal = None
    if all_signals:
        # مرتب‌سازی بر اساس امتیاز دقت
        all_signals.sort(key=lambda x: x['accuracy_score'], reverse=True)
        best_signal = all_signals[0]  # بهترین سیگنال
        
        logger.info(f"Best signal selected: {best_signal['symbol']} with accuracy score: {best_signal['accuracy_score']}")
    
    logger.info(f"Analysis complete. Success: {successful_analyses}, Failed: {failed_analyses}, "
                f"Total signals: {len(all_signals)}, Best signal: {best_signal['symbol'] if best_signal else 'None'}")
    
    return [best_signal] if best_signal else []

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """مدیریت دستور /start با ارائه بهترین سیگنال"""
    try:
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        logger.info(f"User {username} ({user_id}) started analysis")
        
        await update.message.reply_text(
            "🔍 در حال تحلیل بازار برای یافتن بهترین فرصت معاملاتی...\n"
            "⏳ این کار ممکن است چند دقیقه طول بکشد."
        )
        
        # تنظیم timeout برای کل تحلیل
        try:
            signals = await asyncio.wait_for(analyze_market(), timeout=300)  # حداکثر 5 دقیقه
        except asyncio.TimeoutError:
            await update.message.reply_text(
                "⏱️ تحلیل بیش از حد زمان برد. لطفا دوباره تلاش کنید."
            )
            return
        
        if signals and len(signals) > 0:
            sig = signals[0]  # بهترین سیگنال
            
            # تعیین emoji و رنگ بر اساس نوع سیگنال
            emoji = '📈' if sig['type'] == 'Long' else '📉'
            type_color = '🟢' if sig['type'] == 'Long' else '🔴'
            
            # محاسبه درصد سود/ضرر
            if sig['type'] == 'Long':
                profit_pct = ((sig['target'] - sig['entry']) / sig['entry']) * 100
                loss_pct = ((sig['entry'] - sig['stop_loss']) / sig['entry']) * 100
            else:
                profit_pct = ((sig['entry'] - sig['target']) / sig['entry']) * 100
                loss_pct = ((sig['stop_loss'] - sig['entry']) / sig['entry']) * 100
            
            # ساختار پیام بهینه‌شده
            message = f"🎯 *بهترین فرصت معاملاتی یافت شده*\n"
            message += f"{'='*30}\n\n"
            
            message += f"{emoji} *{sig['type']} {sig['symbol']}* {type_color}\n"
            message += f"🏆 **امتیاز دقت: {sig['accuracy_score']}/100**\n\n"
            
            message += f"📊 **جزئیات معاملاتی:**\n"
            message += f"💰 قیمت ورودی: `{sig['entry']:.6f}`\n"
            message += f"🎯 هدف قیمت: `{sig['target']:.6f}` (+{profit_pct:.1f}%)\n"
            message += f"🛑 حد ضرر: `{sig['stop_loss']:.6f}` (-{loss_pct:.1f}%)\n\n"
            
            message += f"📈 **تحلیل تکنیکال:**\n"
            message += f"• RSI: `{sig['rsi']:.1f}`\n"
            message += f"• MACD: `{sig['macd']:.6f}`\n"
            message += f"• روش تحلیل: `{sig['method']}`\n"
            message += f"• قدرت سیگنال: {'⭐' * sig['strength']}\n\n"
            
            message += f"⏰ زمان تولید سیگنال: `{sig['timestamp']}`\n\n"
            
            # اضافه کردن توصیه‌های مدیریت ریسک
            message += f"🎖️ **توصیه‌های مدیریت ریسک:**\n"
            message += f"• حداکثر 2-3% از کل سرمایه ریسک کنید\n"
            message += f"• حد ضرر را رعایت کنید\n"
            message += f"• در صورت رسیدن به 50% سود، حد ضرر را به نقطه سربسر منتقل کنید\n\n"
            
            message += f"⚠️ **هشدار:** این تحلیل صرفاً جنبه اطلاع‌رسانی دارد و توصیه سرمایه‌گذاری نیست."
            
        else:
            message = (
                "❌ متأسفانه در حال حاضر هیچ سیگنال معاملاتی با دقت بالا یافت نشد.\n\n"
                "🔍 **دلایل احتمالی:**\n"
                "• بازار در حالت تثبیت قرار دارد\n"
                "• شرایط تکنیکال مناسب معاملاتی وجود ندارد\n"
                "• همه سیگنال‌ها دارای ریسک بالا هستند\n\n"
                "💡 **پیشنهاد:**\n"
                "• 30-60 دقیقه دیگر مجدداً تلاش کنید\n"
                "• در انتظار شکل‌گیری الگوهای تکنیکال باشید\n"
                "• از معاملات پر ریسک خودداری کنید\n\n"
                "🔄 برای تحلیل مجدد /start را ارسال کنید."
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