from logger_config import logger
import pandas as pd
from typing import Dict, Optional, Union

from .cache_utils import cached_calculation


@cached_calculation('hammer_doji_patterns')
def _detect_hammer_doji_patterns(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """تشخیص الگوهای Hammer و Doji with caching"""
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
        logger.error(f"Error detecting Hammer/Doji patterns: {e}")
        return None


@cached_calculation('engulfing_patterns')
def _detect_engulfing_patterns(df: pd.DataFrame) -> Optional[pd.DataFrame]:
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
        logger.error(f"Error detecting Engulfing patterns: {e}")
        return None


@cached_calculation('star_patterns')
def _detect_star_patterns(df: pd.DataFrame) -> Optional[pd.DataFrame]:
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
            first_red = close.iloc[i-2] < open_price.iloc[i-2]
            small_body = abs(close.iloc[i-1] - open_price.iloc[i-1]) < abs(close.iloc[i-2] - open_price.iloc[i-2]) * 0.3
            gap_down = high.iloc[i-1] < low.iloc[i-2]
            third_green = close.iloc[i] > open_price.iloc[i]
            closes_into_first = close.iloc[i] > (open_price.iloc[i-2] + close.iloc[i-2]) / 2
            
            if first_red and small_body and gap_down and third_green and closes_into_first:
                patterns.iloc[i, patterns.columns.get_loc('morning_star')] = True
            
            # Evening Star Pattern
            first_green = close.iloc[i-2] > open_price.iloc[i-2]
            gap_up = low.iloc[i-1] > high.iloc[i-2]
            third_red = close.iloc[i] < open_price.iloc[i]
            closes_into_first = close.iloc[i] < (open_price.iloc[i-2] + close.iloc[i-2]) / 2
            
            if first_green and small_body and gap_up and third_red and closes_into_first:
                patterns.iloc[i, patterns.columns.get_loc('evening_star')] = True
        
        return patterns
    except Exception as e:
        logger.error(f"Error detecting Star patterns: {e}")
        return None


@cached_calculation('dark_cloud_cover')
def _detect_dark_cloud_cover(df: pd.DataFrame) -> Optional[pd.Series]:
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
            opens_above_prev_high = open_price.iloc[i] > high.iloc[i-1]
            closes_into_first_body = (close.iloc[i] < (open_price.iloc[i-1] + close.iloc[i-1]) / 2)
            significant_penetration = second_body > first_body * 0.5
            
            if (first_bullish and second_bearish and opens_above_prev_high and 
                closes_into_first_body and significant_penetration):
                patterns.iloc[i] = True
        
        return patterns
    except Exception as e:
        logger.error(f"Error detecting Dark Cloud Cover: {e}")
        return None


@cached_calculation('piercing_line')
def _detect_piercing_line(df: pd.DataFrame) -> Optional[pd.Series]:
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
            opens_below_prev_low = open_price.iloc[i] < low.iloc[i-1]
            closes_into_first_body = (close.iloc[i] > (open_price.iloc[i-1] + close.iloc[i-1]) / 2)
            significant_penetration = second_body > first_body * 0.5
            
            if (first_bearish and second_bullish and opens_below_prev_low and 
                closes_into_first_body and significant_penetration):
                patterns.iloc[i] = True
        
        return patterns
    except Exception as e:
        logger.error(f"Error detecting Piercing Line: {e}")
        return None


@cached_calculation('harami_patterns')
def _detect_harami_patterns(df: pd.DataFrame) -> Optional[pd.DataFrame]:
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
            is_smaller = second_body_size < first_body_size * 0.7
            
            # Bullish Harami
            first_bearish = close.iloc[i-1] < open_price.iloc[i-1]
            second_bullish = close.iloc[i] > open_price.iloc[i]
            
            if first_bearish and second_bullish and is_inside and is_smaller:
                patterns.iloc[i, patterns.columns.get_loc('bullish_harami')] = True
            
            # Bearish Harami
            first_bullish = close.iloc[i-1] > open_price.iloc[i-1]
            second_bearish = close.iloc[i] < open_price.iloc[i]
            
            if first_bullish and second_bearish and is_inside and is_smaller:
                patterns.iloc[i, patterns.columns.get_loc('bearish_harami')] = True
        
        return patterns
    except Exception as e:
        logger.error(f"Error detecting Harami patterns: {e}")
        return None


def detect_all_candlestick_patterns(df: pd.DataFrame) -> Dict[str, Union[pd.DataFrame, pd.Series, None]]:
    """
    تشخیص تمام الگوهای کندل استیک موجود
    
    Args:
        df: DataFrame حاوی ستون‌های open, high, low, close
        
    Returns:
        Dict حاوی تمام الگوهای تشخیص داده شده
    """
    if df is None or df.empty:
        logger.error("Input DataFrame is empty or None")
        return {}
    
    required_columns = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"DataFrame must contain columns: {required_columns}")
        return {}
    
    logger.info("Starting candlestick pattern detection...")
    
    patterns = {}
    
    # تشخیص الگوهای تک کندلی
    hammer_doji_patterns = _detect_hammer_doji_patterns(df)
    if hammer_doji_patterns is not None:
        patterns.update({
            'hammer': hammer_doji_patterns['hammer'],
            'doji': hammer_doji_patterns['doji'],
            'shooting_star': hammer_doji_patterns['shooting_star']
        })
    
    # تشخیص الگوهای دو کندلی
    engulfing_patterns = _detect_engulfing_patterns(df)
    if engulfing_patterns is not None:
        patterns.update({
            'bullish_engulfing': engulfing_patterns['bullish_engulfing'],
            'bearish_engulfing': engulfing_patterns['bearish_engulfing']
        })
    
    harami_patterns = _detect_harami_patterns(df)
    if harami_patterns is not None:
        patterns.update({
            'bullish_harami': harami_patterns['bullish_harami'],
            'bearish_harami': harami_patterns['bearish_harami']
        })
    
    dark_cloud_cover = _detect_dark_cloud_cover(df)
    if dark_cloud_cover is not None:
        patterns['dark_cloud_cover'] = dark_cloud_cover
    
    piercing_line = _detect_piercing_line(df)
    if piercing_line is not None:
        patterns['piercing_line'] = piercing_line
    
    # تشخیص الگوهای سه کندلی
    star_patterns = _detect_star_patterns(df)
    if star_patterns is not None:
        patterns.update({
            'morning_star': star_patterns['morning_star'],
            'evening_star': star_patterns['evening_star']
        })
    
    detected_count = sum(1 for pattern in patterns.values() if pattern is not None and pattern.any())
    logger.info(f"Pattern detection completed. Found {detected_count} pattern types with signals.")
    
    return patterns


def get_pattern_summary(patterns: Dict[str, Union[pd.DataFrame, pd.Series, None]]) -> Dict[str, int]:
    """
    خلاصه‌ای از تعداد هر الگوی تشخیص داده شده
    
    Args:
        patterns: خروجی تابع detect_all_candlestick_patterns
        
    Returns:
        Dict حاوی نام الگو و تعداد رخدادهای آن
    """
    summary = {}
    
    for pattern_name, pattern_data in patterns.items():
        if pattern_data is not None:
            if isinstance(pattern_data, (pd.Series, pd.DataFrame)):
                count = int(pattern_data.sum()) if hasattr(pattern_data, 'sum') else 0
                summary[pattern_name] = count
            else:
                summary[pattern_name] = 0
        else:
            summary[pattern_name] = 0
    
    return summary


def get_recent_patterns(patterns: Dict[str, Union[pd.DataFrame, pd.Series, None]], 
                       last_n: int = 10) -> Dict[str, Union[pd.DataFrame, pd.Series, None]]:
    """
    الگوهای اخیر (n کندل آخر)
    
    Args:
        patterns: خروجی تابع detect_all_candlestick_patterns
        last_n: تعداد کندل‌های آخر برای بررسی
        
    Returns:
        Dict حاوی الگوهای اخیر
    """
    recent_patterns = {}
    
    for pattern_name, pattern_data in patterns.items():
        if pattern_data is not None and len(pattern_data) > 0:
            recent_data = pattern_data.tail(last_n)
            if recent_data.any():
                recent_patterns[pattern_name] = recent_data
    
    return recent_patterns


def filter_significant_patterns(patterns: Dict[str, Union[pd.DataFrame, pd.Series, None]], 
                               min_occurrences: int = 1) -> Dict[str, Union[pd.DataFrame, pd.Series, None]]:
    """
    فیلتر کردن الگوهای قابل توجه
    
    Args:
        patterns: خروجی تابع detect_all_candlestick_patterns
        min_occurrences: حداقل تعداد رخداد برای در نظر گیری الگو
        
    Returns:
        Dict حاوی الگوهای فیلتر شده
    """
    significant_patterns = {}
    
    for pattern_name, pattern_data in patterns.items():
        if pattern_data is not None:
            count = int(pattern_data.sum()) if hasattr(pattern_data, 'sum') else 0
            if count >= min_occurrences:
                significant_patterns[pattern_name] = pattern_data
    
    return significant_patterns