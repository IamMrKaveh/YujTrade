import os
import logging
import asyncio
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from indicators import *
from exchanges import *
from signals import *
import sys

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Fix numpy compatibility issue
if not hasattr(np, 'NaN'):
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

# Load symbols at startup
SYMBOLS = load_symbols()

async def analyze_market():
    """تحلیل بازار و بازگرداندن بهترین سیگنال با اطلاعات تکمیلی"""
    all_signals = []
    successful_analyses = 0
    failed_analyses = 0
    
    try:
        # Check if SYMBOLS exists and is not empty
        if not SYMBOLS or len(SYMBOLS) == 0:
            logger.warning("No symbols available for analysis")
            return []
        
        logger.info(f"Starting market analysis for {len(SYMBOLS)} symbols")
        
        # بررسی اتصال به صرافی
        global exchange
        if not exchange:
            try:
                exchange = init_exchange()
                if not exchange:
                    logger.error("Exchange initialization returned None")
                    return []
            except Exception as e:
                logger.error(f"Failed to initialize exchange: {e}")
                return []
        
        # پردازش نمادها به صورت دسته‌ای با مدیریت بهتر خطا
        batch_size = min(3, len(SYMBOLS))  # Ensure batch_size doesn't exceed symbol count
        total_batches = (len(SYMBOLS) - 1) // batch_size + 1
        
        for i in range(0, len(SYMBOLS), batch_size):
            try:
                batch_symbols = SYMBOLS[i:i+batch_size]
                current_batch = i // batch_size + 1
                logger.info(f"Processing batch {current_batch}/{total_batches}")
                
                # پردازش موازی در هر دسته
                batch_tasks = []
                for symbol in batch_symbols:
                    if symbol:  # Ensure symbol is not None or empty
                        batch_tasks.append(_analyze_single_symbol(symbol))
                
                if not batch_tasks:
                    logger.warning(f"No valid symbols in batch {current_batch}")
                    failed_analyses += len(batch_symbols)
                    continue
                
                try:
                    # اجرای موازی با timeout
                    batch_results = await asyncio.wait_for(
                        asyncio.gather(*batch_tasks, return_exceptions=True),
                        timeout=300  # 5 دقیقه برای هر دسته
                    )
                    
                    for j, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            failed_analyses += 1
                            symbol_name = batch_symbols[j] if j < len(batch_symbols) else "Unknown"
                            logger.error(f"Analysis error for {symbol_name}: {result}")
                        elif result is not None and isinstance(result, dict):
                            all_signals.append(result)
                            successful_analyses += 1
                        else:
                            failed_analyses += 1
                            
                except asyncio.TimeoutError:
                    logger.warning(f"Batch {current_batch} timed out")
                    failed_analyses += len(batch_symbols)
                except Exception as e:
                    logger.error(f"Error processing batch {current_batch}: {e}")
                    failed_analyses += len(batch_symbols)
                
                # تاخیر بین دسته‌ها برای رعایت محدودیت نرخ
                if i + batch_size < len(SYMBOLS):
                    await asyncio.sleep(3)
                    
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                failed_analyses += len(batch_symbols) if 'batch_symbols' in locals() else batch_size
                continue
        
        # فیلتر و مرتب‌سازی سیگنال‌های معتبر
        try:
            valid_signals = []
            for sig in all_signals:
                if _validate_signal(sig):
                    valid_signals.append(sig)
            
            if not valid_signals:
                logger.info("No valid signals found")
                return []
            
            # مرتب‌سازی بر اساس امتیاز ترکیبی
            valid_signals.sort(key=_calculate_combined_score, reverse=True)
            best_signal = valid_signals[0]
            
            logger.info(f"Analysis complete. Success: {successful_analyses}, Failed: {failed_analyses}, "
                        f"Valid signals: {len(valid_signals)}, Best signal: {best_signal.get('symbol', 'Unknown')} "
                        f"(Score: {best_signal.get('accuracy_score', 0)})")
            
            return [best_signal]
            
        except Exception as e:
            logger.error(f"Error in signal processing: {e}")
            return []
            
    except Exception as e:
        logger.error(f"Critical error in analyze_market: {e}")
        return []

async def _analyze_single_symbol(symbol):
    """تحلیل یک نماد منفرد با مدیریت کامل خطا"""
    try:
        # تاخیر تصادفی برای جلوگیری از همزمانی درخواست‌ها
        await asyncio.sleep(np.random.uniform(0.5, 2.0))
        
        logger.debug(f"Analyzing {symbol}...")
        
        # دریافت داده‌ها با retry mechanism
        df = await _get_data_with_retry(symbol, max_retries=3)
        if df is None or len(df) < 200:
            logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} candles")
            return None
        
        # محاسبه اندیکاتورها
        df = calculate_indicators(df)
        if df is None:
            logger.warning(f"Failed to calculate indicators for {symbol}")
            return None
        
        # بررسی سیگنال‌ها
        signal_data = check_signals(df, symbol)
        if not signal_data:
            return None
        
        # دریافت قیمت فعلی
        current_price = await _get_current_price_with_retry(symbol)
        if current_price is None:
            logger.warning(f"Failed to get current price for {symbol}")
            return None
        
        # محاسبه امتیاز دقت
        accuracy_score = calculate_signal_accuracy_score(df, signal_data, symbol)
        
        # فیلتر امتیاز حداقلی
        if accuracy_score < 45:  # افزایش حداقل امتیاز
            logger.debug(f"Low accuracy score for {symbol}: {accuracy_score}")
            return None
        
        # ساخت سیگنال تکمیلی
        enhanced_signal = _build_enhanced_signal(
            symbol, signal_data, df, current_price, accuracy_score
        )
        
        logger.info(f"Valid signal found for {symbol}: {enhanced_signal['type']} "
                   f"(Score: {accuracy_score})")
        
        return enhanced_signal
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None

async def _get_data_with_retry(symbol, max_retries=3):
    """دریافت داده‌ها با تلاش مجدد"""
    for attempt in range(max_retries):
        try:
            df = await get_klines(symbol)
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    return None

async def _get_current_price_with_retry(symbol, max_retries=3):
    """دریافت قیمت فعلی با تلاش مجدد"""
    for attempt in range(max_retries):
        try:
            price = await get_current_price(symbol)
            if price is not None and price > 0:
                return price
        except Exception as e:
            logger.warning(f"Price fetch attempt {attempt + 1} failed for {symbol}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
    return None

def _build_enhanced_signal(symbol, signal_data, df, current_price, accuracy_score):
    """ساخت سیگنال تکمیلی با تمام اطلاعات"""
    try:
        # تعیین نوع سیگنال و قیمت‌های هدف
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
        
        last_row = df.iloc[-1]
        
        # محاسبه جهت ترند
        trend_direction = _calculate_trend_direction(df)
        
        # ساخت سیگنال پایه
        enhanced_signal = {
            'symbol': symbol,
            'type': signal_type,
            'entry': float(entry),
            'target': float(target),
            'stop_loss': float(stop_loss),
            'strength': signal_data.get('strength', 2),
            'accuracy_score': accuracy_score,
            'rsi': float(signal_data.get('rsi', 50)),
            'macd': float(signal_data.get('macd', 0)),
            'method': signal_data.get('method', 'Unknown'),
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'buy_score': signal_data.get('buy_score', 0),
            'sell_score': signal_data.get('sell_score', 0),
            'trend_direction': trend_direction
        }
        
        # اضافه کردن شاخص‌های اضافی به صورت ایمن
        _add_additional_indicators(enhanced_signal, last_row, df)
        
        # اضافه کردن سطوح فیبوناچی
        fibonacci_levels = _get_nearby_fibonacci_levels(df, current_price)
        if fibonacci_levels:
            enhanced_signal['fibonacci_levels'] = fibonacci_levels
        
        return enhanced_signal
        
    except Exception as e:
        logger.error(f"Error building enhanced signal for {symbol}: {e}")
        return None

def _calculate_trend_direction(df):
    """محاسبه جهت ترند"""
    try:
        if len(df) < 10:
            return 0
        
        prev_rows = df.iloc[-10:]
        close_prices = prev_rows['close'].values
        
        trend_direction = 0
        for j in range(1, len(close_prices)):
            if close_prices[j] > close_prices[j-1]:
                trend_direction += 1
            elif close_prices[j] < close_prices[j-1]:
                trend_direction -= 1
        
        return trend_direction / len(close_prices)
        
    except Exception:
        return 0

def _add_additional_indicators(signal, last_row, df):
    """اضافه کردن شاخص‌های اضافی به صورت ایمن"""
    try:
        # Stochastic
        if 'STOCHk_14_3_3' in df.columns and not pd.isna(last_row.get('STOCHk_14_3_3')):
            signal['stoch_k'] = float(last_row['STOCHk_14_3_3'])
        
        # MFI
        if 'mfi' in df.columns and not pd.isna(last_row.get('mfi')):
            signal['mfi'] = float(last_row['mfi'])
        
        # CCI
        if 'cci' in df.columns and not pd.isna(last_row.get('cci')):
            signal['cci'] = float(last_row['cci'])
        
        # Williams %R
        if 'williams_r' in df.columns and not pd.isna(last_row.get('williams_r')):
            signal['williams_r'] = float(last_row['williams_r'])
        
        # Volume ratio
        if ('volume_sma' in df.columns and 
            not pd.isna(last_row.get('volume_sma')) and 
            last_row.get('volume_sma', 0) > 0):
            try:
                volume_ratio = last_row['volume'] / last_row['volume_sma']
                signal['volume_ratio'] = float(volume_ratio)
            except (ZeroDivisionError, TypeError):
                pass
                
    except Exception as e:
        logger.warning(f"Error adding additional indicators: {e}")

def _get_nearby_fibonacci_levels(df, current_price):
    """دریافت سطوح فیبوناچی نزدیک"""
    try:
        fibonacci_levels = []
        fib_keys = ['fib_236', 'fib_382', 'fib_500', 'fib_618']
        last_row = df.iloc[-1]
        
        for fib_key in fib_keys:
            if fib_key in df.columns and not pd.isna(last_row.get(fib_key)):
                fib_level = last_row[fib_key]
                price_diff_pct = abs(current_price - fib_level) / current_price * 100
                if price_diff_pct < 2:  # کمتر از 2% فاصله
                    fibonacci_levels.append(
                        f"{fib_key.replace('fib_', 'Fib ')}: {fib_level:.6f}"
                    )
        
        return fibonacci_levels if fibonacci_levels else None
        
    except Exception:
        return None

def _validate_signal(signal):
    """اعتبارسنجی سیگنال"""
    if not signal or not isinstance(signal, dict):
        return False
    
    required_fields = ['symbol', 'type', 'entry', 'target', 'stop_loss', 'accuracy_score']
    
    for field in required_fields:
        if field not in signal:
            return False
        
        # بررسی مقادیر عددی
        if field in ['entry', 'target', 'stop_loss', 'accuracy_score']:
            try:
                value = float(signal[field])
                if not (value > 0 and np.isfinite(value)):
                    return False
            except (ValueError, TypeError):
                return False
    
    # بررسی منطقی قیمت‌ها
    entry = signal['entry']
    target = signal['target']
    stop_loss = signal['stop_loss']
    
    if signal['type'] == 'Long':
        if not (target > entry > stop_loss):
            return False
    else:  # Short
        if not (stop_loss > entry > target):
            return False
    
    # بررسی امتیاز دقت
    if signal['accuracy_score'] < 45:
        return False
    
    return True

def _calculate_combined_score(signal):
    """محاسبه امتیاز ترکیبی برای مرتب‌سازی"""
    try:
        base_score = signal.get('accuracy_score', 0)
        strength_bonus = signal.get('strength', 1) * 5
        volume_bonus = min(signal.get('volume_ratio', 1), 3) * 2
        trend_bonus = abs(signal.get('trend_direction', 0)) * 10
        
        # بونوس برای نمادهای اصلی
        major_pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        major_bonus = 5 if signal.get('symbol') in major_pairs else 0
        
        combined_score = base_score + strength_bonus + volume_bonus + trend_bonus + major_bonus
        
        return min(combined_score, 150)  # حداکثر امتیاز
        
    except Exception:
        return signal.get('accuracy_score', 0)
