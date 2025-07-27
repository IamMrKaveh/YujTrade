import sys
from logger_config import logger
import numpy as np
import pandas as pd

# Fix numpy compatibility issue
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

def _calculate_trend_direction(df):
    """محاسبه جهت ترند"""
    try:
        logger.debug(f"Starting trend direction calculation with {len(df)} rows")
        
        close_prices = _extract_valid_close_prices(df)
        if close_prices is None:
            return 0
        
        trend_direction = _calculate_price_movements(close_prices)
        result = trend_direction / len(close_prices)
        
        logger.debug(f"Trend direction calculated successfully: {result} (from {len(close_prices)} prices)")
        return result
        
    except Exception as e:
        logger.error(f"Unexpected error in trend direction calculation: {e}")
        return 0

def _extract_valid_close_prices(df):
    """استخراج قیمت‌های معتبر close"""
    if df is None or len(df) < 10:
        logger.debug(f"Insufficient data for trend calculation: {len(df) if df is not None else 0} rows")
        return None
    
    if 'close' not in df.columns:
        logger.warning("'close' column not found in DataFrame")
        return None
    
    prev_rows = df.iloc[-10:]
    close_prices = prev_rows['close'].values
    
    if len(close_prices) == 0:
        logger.warning("No close prices available")
        return None
    
    # حذف NaN ها
    if np.any(pd.isna(close_prices)):
        logger.warning("NaN values found in close prices")
        close_prices = close_prices[~pd.isna(close_prices)]
        if len(close_prices) < 2:
            logger.warning("Insufficient valid close prices after removing NaN")
            return None
    
    return close_prices

def _calculate_price_movements(close_prices):
    """محاسبه حرکات قیمتی"""
    trend_direction = 0
    for j in range(1, len(close_prices)):
        try:
            if close_prices[j] > close_prices[j-1]:
                trend_direction += 1
            elif close_prices[j] < close_prices[j-1]:
                trend_direction -= 1
        except (IndexError, TypeError) as e:
            logger.warning(f"Error comparing prices at index {j}: {e}")
            continue
    
    return trend_direction

def _calculate_volatility_metrics(df, symbol):
    """محاسبه معیارهای نوسانات برای تعیین سطوح داینامیک"""
    try:
        logger.debug(f"Starting volatility calculation for {symbol}")
        
        if df is None or len(df) < 20:
            logger.warning(f"Insufficient data for volatility calculation for {symbol}")
            return None
        
        # محاسبه ATR (Average True Range)
        atr_value = _calculate_atr(df, symbol)
        if atr_value is None:
            return None
        
        # محاسبه نوسانات قیمت
        price_volatility = _calculate_price_volatility(df, symbol)
        if price_volatility is None:
            return None
        
        # محاسبه درصد ATR نسبت به قیمت فعلی
        current_price = df['close'].iloc[-1]
        atr_percentage = (atr_value / current_price) * 100
        
        # تعیین ضریب نوسانات بر اساس ATR و volatility
        volatility_factor = min(max(atr_percentage / 2, 0.5), 3.0)
        
        logger.debug(f"Volatility metrics for {symbol}: ATR={atr_value:.6f}, ATR%={atr_percentage:.2f}, factor={volatility_factor:.2f}")
        
        return {
            'atr_value': atr_value,
            'atr_percentage': atr_percentage,
            'price_volatility': price_volatility,
            'volatility_factor': volatility_factor
        }
        
    except Exception as e:
        logger.error(f"Error calculating volatility metrics for {symbol}: {e}")
        return None

def _calculate_atr(df, symbol, period=14):
    """محاسبه Average True Range"""
    try:
        if len(df) < period + 1:
            logger.warning(f"Insufficient data for ATR calculation for {symbol}")
            return None
        
        # محاسبه True Range
        high = df['high'].values
        low = df['low'].values
        close_prev = df['close'].shift(1).values
        
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # حذف NaN ها
        valid_tr = true_range[~pd.isna(true_range)]
        
        if len(valid_tr) < period:
            logger.warning(f"Insufficient valid data for ATR calculation for {symbol}")
            return None
        
        # محاسبه ATR
        atr = np.mean(valid_tr[-period:])
        
        if not np.isfinite(atr) or atr <= 0:
            logger.warning(f"Invalid ATR calculated for {symbol}: {atr}")
            return None
        
        return atr
        
    except Exception as e:
        logger.error(f"Error calculating ATR for {symbol}: {e}")
        return None

def _calculate_price_volatility(df, symbol, period=20):
    """محاسبه نوسانات قیمت"""
    try:
        if len(df) < period:
            logger.warning(f"Insufficient data for price volatility calculation for {symbol}")
            return None
        
        close_prices = df['close'].iloc[-period:].values
        
        if len(close_prices) < 2:
            return None
        
        # محاسبه بازده‌های روزانه
        returns = np.diff(close_prices) / close_prices[:-1]
        
        # حذف NaN ها
        valid_returns = returns[~pd.isna(returns)]
        
        if len(valid_returns) < 2:
            return None
        
        # محاسبه انحراف معیار
        volatility = np.std(valid_returns) * 100  # به درصد
        
        if not np.isfinite(volatility):
            return None
        
        return volatility
        
    except Exception as e:
        logger.error(f"Error calculating price volatility for {symbol}: {e}")
        return None
