from datetime import datetime
from exchange.price import get_current_price
from imports import logger, np, asyncio
from signals.core import check_signals
from signals.indicators.indicator_management import calculate_indicators
from exchange.ohlcv import get_klines

async def _analyze_single_symbol(symbol):
    """تحلیل یک نماد منفرد با مدیریت کامل خطا"""
    try:
        logger.debug(f"Starting analysis for {symbol}")
        
        # تاخیر تصادفی برای جلوگیری از همزمانی درخواست‌ها
        await _add_random_delay()
        
        # دریافت و اعتبارسنجی داده‌ها
        df = await _fetch_and_validate_data(symbol)
        if df is None:
            return None
        
        # محاسبه اندیکاتورها
        df = await _calculate_and_validate_indicators(df, symbol)
        if df is None:
            return None
        
        # بررسی سیگنال‌ها
        signal_data = await _check_and_validate_signals(df, symbol)
        if signal_data is None:
            return None
        
        # دریافت قیمت فعلی
        current_price = await _get_current_price_with_retry(symbol)
        if current_price is None:
            logger.warning(f"Failed to get current price for {symbol}")
            return None
        
        # محاسبه و اعتبارسنجی امتیاز دقت
        accuracy_score = await _calculate_and_validate_accuracy_score(df, signal_data, symbol)
        if accuracy_score is None:
            return None
        
        # ساخت سیگنال تکمیلی
        return await _create_final_signal(symbol, signal_data, df, current_price, accuracy_score)
        
    except Exception as e:
        logger.error(f"Unexpected error in _analyze_single_symbol for {symbol}: {e}")
        return None

async def _add_random_delay():
    """اضافه کردن تاخیر تصادفی"""
    rng = np.random.default_rng(999999)
    await asyncio.sleep(rng.uniform(0.5, 2.0))

async def _fetch_and_validate_data(symbol):
    """دریافت و اعتبارسنجی داده‌ها"""
    df = await _get_data_with_retry(symbol, max_retries=3)
    if df is None or len(df) < 200:
        logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} candles")
        return None
    return df

def _calculate_and_validate_indicators(df, symbol):
    """محاسبه و اعتبارسنجی اندیکاتورها"""
    try:
        df = calculate_indicators(df)
        if df is None:
            logger.warning(f"Failed to calculate indicators for {symbol}")
            return None
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators for {symbol}: {e}")
        return None

def _check_and_validate_signals(df, symbol):
    """بررسی و اعتبارسنجی سیگنال‌ها"""
    try:
        signal_data = check_signals(df, symbol)
        if not signal_data:
            logger.debug(f"No signals found for {symbol}")
            return None
        return signal_data
    except Exception as e:
        logger.error(f"Error checking signals for {symbol}: {e}")
        return None

def _calculate_and_validate_accuracy_score(df, signal_data, symbol):
    """محاسبه و اعتبارسنجی امتیاز دقت"""
    try:
        from signals import calculate_signal_strength
        combined_strength = calculate_signal_strength(df, signal_data['type'])
        
        base_strength = signal_data.get('strength', 50)
        accuracy_score = (base_strength + combined_strength) / 2
        
        min_threshold = 50
        if accuracy_score < min_threshold:
            logger.debug(f"Signal for {symbol} below threshold: {accuracy_score:.1f} < {min_threshold}")
            return None
            
        logger.debug(f"Signal strength for {symbol}: base={base_strength}, combined={combined_strength}, final={accuracy_score:.1f}")
        return accuracy_score
        
    except Exception as e:
        logger.error(f"Error calculating accuracy score for {symbol}: {e}")
        accuracy_score = signal_data.get('strength', 50)
        if accuracy_score < 45:
            logger.debug(f"Signal for {symbol} below fallback threshold: {accuracy_score}")
            return None
        return accuracy_score

def _create_final_signal(symbol, signal_data, df, current_price, accuracy_score):
    """ساخت سیگنال نهایی"""
    try:
        enhanced_signal = _build_enhanced_signal(symbol, signal_data, df, current_price, accuracy_score)
        if enhanced_signal is None:
            logger.warning(f"Failed to build enhanced signal for {symbol}")
            return None
        logger.debug(f"Successfully analyzed {symbol} with score {accuracy_score}")
        return enhanced_signal
    except Exception as e:
        logger.error(f"Error building enhanced signal for {symbol}: {e}")
        return None

async def _get_current_price_with_retry(symbol, max_retries=3):
    """دریافت قیمت فعلی با تلاش مجدد"""
    try:
        logger.debug(f"Starting price fetch for {symbol} with max_retries={max_retries}")
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Price fetch attempt {attempt + 1} for {symbol}")
                price = await get_current_price(symbol)
                
                if price is not None and price > 0:
                    logger.debug(f"Current price fetched successfully for {symbol}: {price}")
                    return price
                else:
                    logger.warning(f"Invalid price received for {symbol}: {price}")
                    
            except asyncio.CancelledError:
                logger.warning(f"Price fetch cancelled for {symbol}")
                raise
            except Exception as e:
                logger.warning(f"Price fetch attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
        
        logger.error(f"Failed to fetch current price for {symbol} after {max_retries} attempts")
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error in _get_current_price_with_retry for {symbol}: {e}")
        return None

def _is_valid_data(df, symbol):
    """Check if the fetched data is valid"""
    if df is not None and len(df) > 0:
        logger.debug(f"Data fetched successfully for {symbol}: {len(df)} rows")
        return True
    else:
        logger.warning(f"Invalid data received for {symbol}: {df is not None} rows={len(df) if df is not None else 0}")
        return False

async def _get_data_with_retry(symbol, max_retries=3):
    """دریافت داده‌ها با تلاش مجدد"""
    try:
        logger.debug(f"Starting data fetch for {symbol} with max_retries={max_retries}")
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Data fetch attempt {attempt + 1} for {symbol}")
                df = await get_klines(symbol)
                
                if _is_valid_data(df, symbol):
                    return df
                    
            except asyncio.CancelledError:
                logger.warning(f"Data fetch cancelled for {symbol}")
                raise
            except Exception as e:
                logger.warning(f"Data fetch attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"Failed to fetch data for {symbol} after {max_retries} attempts")
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error in _get_data_with_retry for {symbol}: {e}")
        return None

def _build_enhanced_signal(symbol, signal_data, df, current_price, accuracy_score):
    """ساخت سیگنال تکمیلی با تمام اطلاعات و stop-loss/target داینامیک"""
    try:
        logger.debug(f"Building enhanced signal for {symbol}")
        
        # محاسبه volatility برای تعیین dynamic levels
        volatility_data = _calculate_volatility_metrics(df, symbol)
        if volatility_data is None:
            logger.warning(f"Failed to calculate volatility for {symbol}, using defaults")
            volatility_data = {'atr_percentage': 2.0, 'volatility_factor': 1.0}
        
        # تعیین نوع سیگنال و قیمت‌های هدف داینامیک
        if signal_data['type'] == 'buy':
            entry = current_price
            target, stop_loss = _calculate_dynamic_levels_long(entry, volatility_data, symbol)
            signal_type = 'Long'
        else:  # sell
            entry = current_price
            target, stop_loss = _calculate_dynamic_levels_short(entry, volatility_data, symbol)
            signal_type = 'Short'
        
        try:
            last_row = df.iloc[-1]
        except (IndexError, KeyError) as e:
            logger.error(f"Error accessing last row for {symbol}: {e}")
            return None
        
        # محاسبه جهت ترند
        try:
            trend_direction = _calculate_trend_direction(df)
        except Exception as e:
            logger.warning(f"Error calculating trend direction for {symbol}: {e}")
            trend_direction = 0
        
        # ساخت سیگنال اصلی
        try:
            signal = {
                'symbol': symbol,
                'type': signal_type,
                'entry': float(entry),
                'target': float(target),
                'stop_loss': float(stop_loss),
                'accuracy_score': float(accuracy_score),
                'strength': signal_data.get('strength', 50),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'trend_direction': trend_direction,
                'volatility_factor': volatility_data['volatility_factor'],
                'atr_percentage': volatility_data['atr_percentage']
            }
            
            # اضافه کردن شاخص‌های اضافی
            _add_additional_indicators(signal, last_row, df)
            
            # اضافه کردن سطوح فیبوناچی
            try:
                fibonacci_levels = _get_nearby_fibonacci_levels(df, current_price)
                if fibonacci_levels:
                    signal['fibonacci_levels'] = fibonacci_levels
            except Exception as e:
                logger.debug(f"Error getting Fibonacci levels for {symbol}: {e}")
            
            logger.debug(f"Enhanced signal built successfully for {symbol} with dynamic levels: target={target:.6f}, stop_loss={stop_loss:.6f}")
            return signal
            
        except Exception as e:
            logger.error(f"Error building signal structure for {symbol}: {e}")
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error in _build_enhanced_signal for {symbol}: {e}")
        return None
