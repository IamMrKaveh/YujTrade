from logger_config import logger
from .cache_utils import cached_calculation, NUMBA_AVAILABLE, jit
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Union
import talib
import warnings

warnings.filterwarnings('ignore')

if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def _calculate_volatility_score(returns, window=20):
        """محاسبه امتیاز نوسانات با استفاده از Numba"""
        if len(returns) < window:
            return 3.0
        
        vol = np.std(returns[-window:]) * np.sqrt(252)
        if vol > 0.4:
            return 8.0
        elif vol > 0.3:
            return 6.0
        elif vol > 0.2:
            return 4.0
        elif vol > 0.1:
            return 2.0
        else:
            return 1.0
    
    @jit(nopython=True)
    def _calculate_trend_strength_numba(prices, window=14):
        """محاسبه قدرت ترند با استفاده از Numba"""
        if len(prices) < window:
            return 50.0
        
        recent_prices = prices[-window:]
        slope = (recent_prices[-1] - recent_prices[0]) / window
        price_range = np.max(recent_prices) - np.min(recent_prices)
        
        if price_range == 0:
            return 50.0
        
        normalized_slope = slope / price_range * 100
        return min(max(50 + normalized_slope, 0), 100)
else:
    def _calculate_volatility_score(returns, window=20):
        """محاسبه امتیاز نوسانات بدون Numba"""
        if len(returns) < window:
            return 3.0
        
        vol = np.std(returns[-window:]) * np.sqrt(252)
        if vol > 0.4:
            return 8.0
        elif vol > 0.3:
            return 6.0
        elif vol > 0.2:
            return 4.0
        elif vol > 0.1:
            return 2.0
        else:
            return 1.0
    
    def _calculate_trend_strength_numba(prices, window=14):
        """محاسبه قدرت ترند بدون Numba"""
        if len(prices) < window:
            return 50.0
        
        recent_prices = prices[-window:]
        slope = (recent_prices[-1] - recent_prices[0]) / window
        price_range = np.max(recent_prices) - np.min(recent_prices)
        
        if price_range == 0:
            return 50.0
        
        normalized_slope = slope / price_range * 100
        return min(max(50 + normalized_slope, 0), 100)

@cached_calculation("market_conditions_analysis")
def _analyze_market_conditions(df: pd.DataFrame) -> Dict[str, float]:
    """تحلیل جامع شرایط بازار"""
    try:
        if len(df) < 50:
            return {
                'volatility_score': 3.0,
                'trend_strength': 50.0,
                'momentum_score': 50.0,
                'support_resistance_score': 50.0,
                'volume_analysis': 50.0
            }
        
        prices = df['close'].values
        
        # محاسبه بازدهی
        returns = np.diff(np.log(prices))
        
        # 1. تحلیل نوسانات
        volatility_score = _calculate_volatility_score(returns)
        
        # 2. قدرت ترند
        trend_strength = _calculate_trend_strength_numba(prices)
        
        # 3. تحلیل مومنتوم با RSI
        try:
            rsi = talib.RSI(prices, timeperiod=14)
            rsi_current = rsi[-1] if not np.isnan(rsi[-1]) else 50
            if rsi_current > 70:
                momentum_score = 80
            elif rsi_current < 30:
                momentum_score = 20
            else:
                momentum_score = rsi_current
        except:
            momentum_score = 50.0
        
        # 4. تحلیل سطوح حمایت و مقاومت
        try:
            recent_high = np.max(prices[-20:])
            recent_low = np.min(prices[-20:])
            current_price = prices[-1]
            
            # موقعیت قیمت نسبت به کانال
            if recent_high != recent_low:
                position_in_range = (current_price - recent_low) / (recent_high - recent_low)
                if position_in_range > 0.8:
                    support_resistance_score = 80  # نزدیک مقاومت
                elif position_in_range < 0.2:
                    support_resistance_score = 20  # نزدیک حمایت
                else:
                    support_resistance_score = 50
            else:
                support_resistance_score = 50.0
        except:
            support_resistance_score = 50.0
        
        # 5. تحلیل حجم (اگر موجود باشد)
        volume_analysis = 50.0
        if 'volume' in df.columns:
            try:
                recent_volume = df['volume'].tail(10).mean()
                avg_volume = df['volume'].tail(50).mean()
                if recent_volume > avg_volume * 1.5:
                    volume_analysis = 75  # حجم بالا
                elif recent_volume < avg_volume * 0.5:
                    volume_analysis = 25  # حجم پایین
            except:
                pass
        
        return {
            'volatility_score': float(volatility_score),
            'trend_strength': float(trend_strength),
            'momentum_score': float(momentum_score),
            'support_resistance_score': float(support_resistance_score),
            'volume_analysis': float(volume_analysis)
        }
        
    except Exception as e:
        logger.error(f"Error in market conditions analysis: {e}")
        return {
            'volatility_score': 3.0,
            'trend_strength': 50.0,
            'momentum_score': 50.0,
            'support_resistance_score': 50.0,
            'volume_analysis': 50.0
        }

@cached_calculation("kelly_criterion")
def _calculate_kelly_criterion(df: pd.DataFrame, win_rate: float, avg_win: float, avg_loss: float) -> float:
    """محاسبه معیار کلی برای اندازه پوزیشن بهینه"""
    try:
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        # Kelly Criterion: f = (bp - q) / b
        # b = avg_win / avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / abs(avg_loss)
        q = 1 - win_rate
        
        kelly_fraction = (b * win_rate - q) / b
        
        # محدود کردن به حداکثر 25% برای امنیت
        return max(0, min(kelly_fraction, 0.25))
        
    except Exception as e:
        logger.error(f"Error in Kelly Criterion calculation: {e}")
        return 0.1

@cached_calculation("correlation_analysis")
def _analyze_correlation_risk(df: pd.DataFrame, other_positions: Optional[list] = None) -> float:
    """تحلیل ریسک همبستگی با سایر پوزیشن‌ها"""
    try:
        if not other_positions:
            return 1.0
        
        # اگر داده‌های سایر پوزیشن‌ها موجود باشد، همبستگی را محاسبه کن
        correlations = []
        current_returns = df['close'].pct_change().dropna()
        
        for position_data in other_positions:
            if isinstance(position_data, pd.Series):
                other_returns = position_data.pct_change().dropna()
                if len(other_returns) >= 10 and len(current_returns) >= 10:
                    # تطبیق طول سری‌ها
                    min_len = min(len(current_returns), len(other_returns))
                    corr = np.corrcoef(
                        current_returns.tail(min_len),
                        other_returns.tail(min_len)
                    )[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        if correlations:
            max_correlation = max(correlations)
            # کاهش اندازه پوزیشن بر اساس همبستگی
            return max(0.5, 1 - max_correlation * 0.5)
        
        return 1.0
        
    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}")
        return 1.0

@cached_calculation("advanced_position_sizing")
def _advanced_position_sizing(
    df: pd.DataFrame,
    capital: float,
    base_risk_percent: float,
    entry_price: float,
    stop_loss: float,
    win_rate: Optional[float] = None,
    avg_win: Optional[float] = None,
    avg_loss: Optional[float] = None,
    other_positions: Optional[list] = None
) -> Dict[str, Union[float, Dict]]:
    """محاسبه اندازه پوزیشن پیشرفته با در نظر گیری عوامل متعدد"""
    try:
        # تحلیل شرایط بازار
        market_conditions = _analyze_market_conditions(df)
        
        # محاسبه اندازه پوزیشن پایه
        base_risk = capital * (base_risk_percent / 100)
        price_diff = abs(entry_price - stop_loss)
        
        if price_diff == 0:
            return {
                'position_size': 0,
                'risk_amount': 0,
                'analysis': market_conditions
            }
        
        base_position_size = base_risk / price_diff
        
        # ضریب تعدیل نهایی
        adjustment_factor = 1.0
        
        # 1. تعدیل بر اساس نوسانات
        volatility_score = market_conditions['volatility_score']
        if volatility_score > 6:
            adjustment_factor *= 0.6
        elif volatility_score > 4:
            adjustment_factor *= 0.8
        elif volatility_score < 2:
            adjustment_factor *= 1.2
        
        # 2. تعدیل بر اساس قدرت ترند
        trend_strength = market_conditions['trend_strength']
        if trend_strength > 75:
            adjustment_factor *= 1.3
        elif trend_strength < 25:
            adjustment_factor *= 0.7
        
        # 3. تعدیل بر اساس مومنتوم
        momentum_score = market_conditions['momentum_score']
        if momentum_score > 75 or momentum_score < 25:
            adjustment_factor *= 0.9  # در شرایط اشباع خرید/فروش محتاط‌تر باشیم
        
        # 4. استفاده از معیار کلی (اگر داده‌ها موجود باشد)
        kelly_factor = 1.0
        if all([win_rate, avg_win, avg_loss]):
            kelly_fraction = _calculate_kelly_criterion(df, win_rate, avg_win, avg_loss)
            kelly_factor = kelly_fraction / (base_risk_percent / 100) if base_risk_percent > 0 else 1.0
            kelly_factor = min(kelly_factor, 2.0)  # حداکثر دو برابر ریسک پایه
        
        # 5. تعدیل بر اساس همبستگی
        correlation_factor = _analyze_correlation_risk(df, other_positions)
        
        # اعمال تمام ضرایب
        final_adjustment = adjustment_factor * kelly_factor * correlation_factor
        position_size = base_position_size * final_adjustment
        
        # محدودیت‌های نهایی
        max_position_value = capital * 0.15  # حداکثر 15% سرمایه
        max_position_size = max_position_value / entry_price
        
        final_position_size = min(position_size, max_position_size)
        final_position_size = max(final_position_size, 0)
        
        # محاسبه ریسک نهایی
        final_risk = final_position_size * price_diff
        
        return {
            'position_size': final_position_size,
            'risk_amount': final_risk,
            'risk_percent': (final_risk / capital) * 100,
            'adjustment_factor': final_adjustment,
            'kelly_factor': kelly_factor,
            'correlation_factor': correlation_factor,
            'analysis': market_conditions
        }
        
    except Exception as e:
        logger.error(f"Error in advanced position sizing: {e}")
        return {
            'position_size': 0,
            'risk_amount': 0,
            'analysis': {}
        }
@cached_calculation("adaptive_position_sizing")
def _adaptive_position_sizing(df, capital, risk_percent, entry_price, stop_loss, market_conditions=None):
    """محاسبه اندازه پوزیشن تطبیقی"""
    try:
        base_risk = capital * (risk_percent / 100)
        price_diff = abs(entry_price - stop_loss)
        
        if price_diff == 0:
            return 0
        
        base_position_size = base_risk / price_diff
        
        # تنظیم بر اساس شرایط بازار
        if market_conditions:
            volatility = market_conditions.get('volatility_score', 3)
            trend_strength = market_conditions.get('trend_strength', 50)
            
            # نوسانات بالا = اندازه پوزیشن کمتر
            if volatility > 5:
                size_multiplier = 0.7
            elif volatility < 2:
                size_multiplier = 1.2
            else:
                size_multiplier = 1.0
            
            # ترند قوی = اندازه پوزیشن بیشتر
            if trend_strength > 70:
                size_multiplier *= 1.3
            elif trend_strength < 30:
                size_multiplier *= 0.8
            
            position_size = base_position_size * size_multiplier
        else:
            position_size = base_position_size
        
        # محدودیت حداکثر 10% سرمایه
        max_position_value = capital * 0.1
        max_position_size = max_position_value / entry_price
        
        final_position_size = min(position_size, max_position_size)
        
        return max(final_position_size, 0)
        
    except Exception as e:
        logger.error(f"Error in adaptive position sizing: {e}")
        return 0
        
@cached_calculation("position_size_atr")
def _calculate_position_size_atr(df, capital, risk_percent, atr_value, atr_multiplier=2):
    """محاسبه اندازه پوزیشن بر اساس ATR"""
    try:
        risk_amount = capital * (risk_percent / 100)
        stop_distance = atr_value * atr_multiplier
        position_size = risk_amount / stop_distance
        
        return min(position_size, capital * 0.1)  # حداکثر 10% سرمایه
    except Exception:
        return 0

def adaptive_position_sizing(capital, risk_percent, entry_price, stop_loss, market_conditions=None):
    """Wrapper function for backward compatibility"""
    dummy_df = pd.DataFrame({'close': [entry_price]})
    return _adaptive_position_sizing(dummy_df, capital, risk_percent, entry_price, stop_loss, market_conditions)

def calculate_position_size_atr(capital, risk_percent, atr_value, atr_multiplier=2):
    """Wrapper function for backward compatibility"""
    try:
        dummy_df = pd.DataFrame({'close': [1.0]})
        return _calculate_position_size_atr(dummy_df, capital, risk_percent, atr_value, atr_multiplier)
    except Exception as e:
        logger.error(f"Error in calculate_position_size_atr: {e}")
        return 0
