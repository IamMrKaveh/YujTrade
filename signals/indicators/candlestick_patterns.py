from logger_config import logger
import pandas as pd
from .cache_utils import _cached_indicator_calculation

def _detect_hammer_doji_patterns_internal(df):
    """Internal function for detecting Hammer and Doji patterns"""
    try:
        if df is None or len(df) < 3:
            return None
        
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # محاسبه اجزای کندل
        body = abs(close - open_price)
        upper_shadow = high - pd.concat([open_price, close], axis=1).max(axis=1)
        lower_shadow = pd.concat([open_price, close], axis=1).min(axis=1) - low
        total_range = high - low
        
        patterns = pd.DataFrame(index=df.index)
        
        # Hammer Pattern
        hammer_condition = (
            (lower_shadow >= 2 * body) &
            (upper_shadow <= 0.1 * total_range) &
            (body <= 0.3 * total_range)
        )
        patterns['hammer'] = hammer_condition
        
        # Doji Pattern
        doji_condition = (body <= 0.1 * total_range)
        patterns['doji'] = doji_condition
        
        # Shooting Star Pattern
        shooting_star_condition = (
            (upper_shadow >= 2 * body) &
            (lower_shadow <= 0.1 * total_range) &
            (body <= 0.3 * total_range)
        )
        patterns['shooting_star'] = shooting_star_condition
        
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting Hammer/Doji patterns: {e}")
        return None

def _detect_hammer_doji_patterns(df):
    """تشخیص الگوهای Hammer و Doji with caching"""
    return _cached_indicator_calculation(df, 'hammer_doji_patterns', _detect_hammer_doji_patterns_internal)

def _detect_engulfing_patterns(df):
    """تشخیص الگوهای Engulfing"""
    try:
        if df is None or len(df) < 2:
            return None
        
        open_price = df['open']
        close = df['close']
        
        patterns = pd.DataFrame(index=df.index)
        patterns['bullish_engulfing'] = False
        patterns['bearish_engulfing'] = False
        
        for i in range(1, len(df)):
            prev_open = open_price.iloc[i-1]
            prev_close = close.iloc[i-1]
            curr_open = open_price.iloc[i]
            curr_close = close.iloc[i]
            
            # Bullish Engulfing
            if (prev_close < prev_open and  # Previous red candle
                curr_close > curr_open and  # Current green candle
                curr_open < prev_close and  # Current opens below previous close
                curr_close > prev_open):    # Current closes above previous open
                patterns.iloc[i, patterns.columns.get_loc('bullish_engulfing')] = True
            
            # Bearish Engulfing
            if (prev_close > prev_open and  # Previous green candle
                curr_close < curr_open and  # Current red candle
                curr_open > prev_close and  # Current opens above previous close
                curr_close < prev_open):    # Current closes below previous open
                patterns.iloc[i, patterns.columns.get_loc('bearish_engulfing')] = True
        
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting Engulfing patterns: {e}")
        return None

def _detect_star_patterns(df):
    """تشخیص الگوهای Morning/Evening Star"""
    try:
        if df is None or len(df) < 3:
            return None
        
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        patterns = pd.DataFrame(index=df.index)
        patterns['morning_star'] = False
        patterns['evening_star'] = False
        
        for i in range(2, len(df)):
            # Morning Star Pattern
            first_red = close.iloc[i-2] < open_price.iloc[i-2]  # First candle is red
            small_body = abs(close.iloc[i-1] - open_price.iloc[i-1]) < abs(close.iloc[i-2] - open_price.iloc[i-2]) * 0.3  # Small middle candle
            gap_down = high.iloc[i-1] < low.iloc[i-2]  # Gap down
            third_green = close.iloc[i] > open_price.iloc[i]  # Third candle is green
            closes_into_first = close.iloc[i] > (open_price.iloc[i-2] + close.iloc[i-2]) / 2  # Closes well into first candle
            
            if first_red and small_body and gap_down and third_green and closes_into_first:
                patterns.iloc[i, patterns.columns.get_loc('morning_star')] = True
            
            # Evening Star Pattern
            first_green = close.iloc[i-2] > open_price.iloc[i-2]  # First candle is green
            gap_up = low.iloc[i-1] > high.iloc[i-2]  # Gap up
            third_red = close.iloc[i] < open_price.iloc[i]  # Third candle is red
            closes_into_first = close.iloc[i] < (open_price.iloc[i-2] + close.iloc[i-2]) / 2  # Closes well into first candle
            
            if first_green and small_body and gap_up and third_red and closes_into_first:
                patterns.iloc[i, patterns.columns.get_loc('evening_star')] = True
        
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting Star patterns: {e}")
        return None

def _detect_morning_evening_star(df):
    """تشخیص الگوهای Morning/Evening Star"""
    try:
        if df is None or len(df) < 3:
            return None
            
        open_price = df['open']
        close = df['close']
        
        # محاسبه بدنه کندل‌ها
        body = abs(close - open_price)
        body_1 = body.shift(1)
        body_2 = body.shift(2)
        
        # Morning Star Pattern
        morning_star = ((close.shift(2) < open_price.shift(2)) &  # کندل نزولی
                        (body_1 < body_2 * 0.3) &  # کندل کوچک میانی
                        (close > open_price) &  # کندل صعودی
                        (close > (close.shift(2) + open_price.shift(2)) / 2))
        
        # Evening Star Pattern
        evening_star = ((close.shift(2) > open_price.shift(2)) &  # کندل صعودی
                        (body_1 < body_2 * 0.3) &  # کندل کوچک میانی
                        (close < open_price) &  # کندل نزولی
                        (close < (close.shift(2) + open_price.shift(2)) / 2))
        
        return {
            'morning_star': morning_star,
            'evening_star': evening_star
        }
    except Exception:
        return None

def _detect_dark_cloud_cover(df):
    """تشخیص الگوی Dark Cloud Cover"""
    try:
        if df is None or len(df) < 2:
            return None
        
        open_price = df['open']
        high = df['high']
        close = df['close']
        
        patterns = pd.Series(False, index=df.index)
        
        for i in range(1, len(df)):
            # کندل اول: صعودی قوی
            first_bullish = close.iloc[i-1] > open_price.iloc[i-1]
            first_body = close.iloc[i-1] - open_price.iloc[i-1]
            
            # کندل دوم: نزولی
            second_bearish = close.iloc[i] < open_price.iloc[i]
            second_body = open_price.iloc[i] - close.iloc[i]
            
            # شرایط Dark Cloud Cover
            opens_above_prev_high = open_price.iloc[i] > high.iloc[i-1]  # باز شدن بالای کندل قبل
            closes_into_first_body = (close.iloc[i] < (open_price.iloc[i-1] + close.iloc[i-1]) / 2)  # بسته شدن در نیمه پایین کندل اول
            significant_penetration = second_body > first_body * 0.5  # نفوذ قابل توجه
            
            if (first_bullish and second_bearish and opens_above_prev_high and 
                closes_into_first_body and significant_penetration):
                patterns.iloc[i] = True
        
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting Dark Cloud Cover: {e}")
        return None

def _detect_piercing_line(df):
    """تشخیص الگوی Piercing Line"""
    try:
        if df is None or len(df) < 2:
            return None
        
        open_price = df['open']
        low = df['low']
        close = df['close']
        
        patterns = pd.Series(False, index=df.index)
        
        for i in range(1, len(df)):
            # کندل اول: نزولی قوی
            first_bearish = close.iloc[i-1] < open_price.iloc[i-1]
            first_body = open_price.iloc[i-1] - close.iloc[i-1]
            
            # کندل دوم: صعودی
            second_bullish = close.iloc[i] > open_price.iloc[i]
            second_body = close.iloc[i] - open_price.iloc[i]
            
            # شرایط Piercing Line
            opens_below_prev_low = open_price.iloc[i] < low.iloc[i-1]  # باز شدن زیر کندل قبل
            closes_into_first_body = (close.iloc[i] > (open_price.iloc[i-1] + close.iloc[i-1]) / 2)  # بسته شدن در نیمه بالای کندل اول
            significant_penetration = second_body > first_body * 0.5  # نفوذ قابل توجه
            
            if (first_bearish and second_bullish and opens_below_prev_low and 
                closes_into_first_body and significant_penetration):
                patterns.iloc[i] = True
        
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting Piercing Line: {e}")
        return None

def _detect_harami_patterns(df):
    """تشخیص الگوهای Harami"""
    try:
        if df is None or len(df) < 2:
            return None
        
        open_price = df['open']
        close = df['close']
        
        patterns = pd.DataFrame(index=df.index)
        patterns['bullish_harami'] = False
        patterns['bearish_harami'] = False
        
        for i in range(1, len(df)):
            # محاسبه اندازه بدنه کندل‌ها
            first_body_size = abs(close.iloc[i-1] - open_price.iloc[i-1])
            second_body_size = abs(close.iloc[i] - open_price.iloc[i])
            
            # کندل دوم باید در داخل کندل اول باشد
            first_max = max(open_price.iloc[i-1], close.iloc[i-1])
            first_min = min(open_price.iloc[i-1], close.iloc[i-1])
            second_max = max(open_price.iloc[i], close.iloc[i])
            second_min = min(open_price.iloc[i], close.iloc[i])
            
            is_inside = (second_max < first_max and second_min > first_min)
            is_smaller = second_body_size < first_body_size * 0.7  # کندل دوم کوچک‌تر
            
            # Bullish Harami
            first_bearish = close.iloc[i-1] < open_price.iloc[i-1]  # کندل اول نزولی
            second_bullish = close.iloc[i] > open_price.iloc[i]     # کندل دوم صعودی
            
            if first_bearish and second_bullish and is_inside and is_smaller:
                patterns.iloc[i, patterns.columns.get_loc('bullish_harami')] = True
            
            # Bearish Harami
            first_bullish = close.iloc[i-1] > open_price.iloc[i-1]  # کندل اول صعودی
            second_bearish = close.iloc[i] < open_price.iloc[i]     # کندل دوم نزولی
            
            if first_bullish and second_bearish and is_inside and is_smaller:
                patterns.iloc[i, patterns.columns.get_loc('bearish_harami')] = True
        
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting Harami patterns: {e}")
        return None

