from logger_config import logger
from .cache_utils import cached_calculation, NUMBA_AVAILABLE, jit
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Union, List
import talib
import warnings

warnings.filtererrors('ignore')

# توابع کمکی با Numba (اگر موجود باشد)
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
            
            if recent_high != recent_low:
                position_in_range = (current_price - recent_low) / (recent_high - recent_low)
                if position_in_range > 0.8:
                    support_resistance_score = 80
                elif position_in_range < 0.2:
                    support_resistance_score = 20
                else:
                    support_resistance_score = 50
            else:
                support_resistance_score = 50.0
        except:
            support_resistance_score = 50.0
        
        # 5. تحلیل حجم
        volume_analysis = 50.0
        if 'volume' in df.columns:
            try:
                recent_volume = df['volume'].tail(10).mean()
                avg_volume = df['volume'].tail(50).mean()
                if recent_volume > avg_volume * 1.5:
                    volume_analysis = 75
                elif recent_volume < avg_volume * 0.5:
                    volume_analysis = 25
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
        
        b = avg_win / abs(avg_loss)
        q = 1 - win_rate
        kelly_fraction = (b * win_rate - q) / b
        
        return max(0, min(kelly_fraction, 0.25))
        
    except Exception as e:
        logger.error(f"Error in Kelly Criterion calculation: {e}")
        return 0.1

@cached_calculation("correlation_analysis")
def _analyze_correlation_risk(df: pd.DataFrame, other_positions: Optional[List] = None) -> float:
    """تحلیل ریسک همبستگی با سایر پوزیشن‌ها"""
    try:
        if not other_positions:
            return 1.0
        
        correlations = []
        current_returns = df['close'].pct_change().dropna()
        
        for position_data in other_positions:
            if isinstance(position_data, pd.Series):
                other_returns = position_data.pct_change().dropna()
                if len(other_returns) >= 10 and len(current_returns) >= 10:
                    min_len = min(len(current_returns), len(other_returns))
                    corr = np.corrcoef(
                        current_returns.tail(min_len),
                        other_returns.tail(min_len)
                    )[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        if correlations:
            max_correlation = max(correlations)
            return max(0.5, 1 - max_correlation * 0.5)
        
        return 1.0
        
    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}")
        return 1.0

def _calculate_base_position_size(capital: float, risk_percent: float, entry_price: float, stop_loss: float) -> float:
    """محاسبه اندازه پوزیشن پایه"""
    base_risk = capital * (risk_percent / 100)
    price_diff = abs(entry_price - stop_loss)
    
    if price_diff == 0:
        return 0
    
    return base_risk / price_diff

def _apply_market_adjustments(base_size: float, market_conditions: Dict[str, float]) -> float:
    """اعمال تعدیلات بر اساس شرایط بازار"""
    adjustment_factor = 1.0
    
    # تعدیل بر اساس نوسانات
    volatility_score = market_conditions.get('volatility_score', 3)
    if volatility_score > 6:
        adjustment_factor *= 0.6
    elif volatility_score > 4:
        adjustment_factor *= 0.8
    elif volatility_score < 2:
        adjustment_factor *= 1.2
    
    # تعدیل بر اساس قدرت ترند
    trend_strength = market_conditions.get('trend_strength', 50)
    if trend_strength > 75:
        adjustment_factor *= 1.3
    elif trend_strength < 25:
        adjustment_factor *= 0.7
    
    # تعدیل بر اساس مومنتوم
    momentum_score = market_conditions.get('momentum_score', 50)
    if momentum_score > 75 or momentum_score < 25:
        adjustment_factor *= 0.9
    
    return base_size * adjustment_factor

def _apply_safety_limits(position_size: float, capital: float, entry_price: float) -> float:
    """اعمال محدودیت‌های امنیتی"""
    max_position_value = capital * 0.15  # حداکثر 15% سرمایه
    max_position_size = max_position_value / entry_price
    
    return min(max(position_size, 0), max_position_size)

def calculate_optimal_position_size(
    df: pd.DataFrame,
    capital: float,
    risk_percent: float = 2.0,
    entry_price: float = None,
    stop_loss: float = None,
    win_rate: Optional[float] = None,
    avg_win: Optional[float] = None,
    avg_loss: Optional[float] = None,
    other_positions: Optional[List] = None,
    use_atr: bool = False,
    atr_multiplier: float = 2.0
) -> Dict[str, Union[float, Dict]]:
    """
    تابع اصلی برای محاسبه اندازه پوزیشن بهینه
    
    Args:
        df: دیتافریم حاوی داده‌های قیمت
        capital: سرمایه کل
        risk_percent: درصد ریسک (پیش‌فرض 2%)
        entry_price: قیمت ورود
        stop_loss: قیمت استاپ لاس
        win_rate: نرخ برد (اختیاری)
        avg_win: متوسط سود (اختیاری)
        avg_loss: متوسط ضرر (اختیاری)
        other_positions: لیست سایر پوزیشن‌ها برای تحلیل همبستگی
        use_atr: استفاده از ATR برای محاسبه استاپ لاس
        atr_multiplier: ضریب ATR
    
    Returns:
        دیکشنری حاوی اطلاعات کامل اندازه پوزیشن
    """
    try:
        # بررسی صحت داده‌های ورودی
        if len(df) == 0 or capital <= 0:
            return _get_default_result()
        
        # تنظیم قیمت ورود پیش‌فرض
        if entry_price is None:
            entry_price = df['close'].iloc[-1]
        
        # محاسبه ATR اگر درخواست شده باشد
        if use_atr and stop_loss is None:
            try:
                atr_values = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
                current_atr = atr_values[-1] if not np.isnan(atr_values[-1]) else entry_price * 0.02
                stop_loss = entry_price - (current_atr * atr_multiplier)
            except:
                stop_loss = entry_price * 0.98  # 2% پیش‌فرض
        
        # تنظیم استاپ لاس پیش‌فرض
        if stop_loss is None:
            stop_loss = entry_price * 0.98  # 2% زیر قیمت ورود
        
        # تحلیل شرایط بازار
        market_conditions = _analyze_market_conditions(df)
        
        # محاسبه اندازه پوزیشن پایه
        base_position_size = _calculate_base_position_size(capital, risk_percent, entry_price, stop_loss)
        
        if base_position_size == 0:
            return _get_default_result()
        
        # اعمال تعدیلات بازار
        adjusted_position_size = _apply_market_adjustments(base_position_size, market_conditions)
        
        # محاسبه ضریب کلی
        kelly_factor = 1.0
        if all([win_rate, avg_win, avg_loss]):
            kelly_fraction = _calculate_kelly_criterion(df, win_rate, avg_win, avg_loss)
            kelly_factor = kelly_fraction / (risk_percent / 100) if risk_percent > 0 else 1.0
            kelly_factor = min(kelly_factor, 2.0)
        
        # تحلیل همبستگی
        correlation_factor = _analyze_correlation_risk(df, other_positions)
        
        # اعمال تمام ضرایب
        final_adjustment_factor = kelly_factor * correlation_factor
        final_position_size = adjusted_position_size * final_adjustment_factor
        
        # اعمال محدودیت‌های امنیتی
        safe_position_size = _apply_safety_limits(final_position_size, capital, entry_price)
        
        # محاسبه نتایج نهایی
        price_diff = abs(entry_price - stop_loss)
        final_risk = safe_position_size * price_diff
        position_value = safe_position_size * entry_price
        
        return {
            'position_size': round(safe_position_size, 4),
            'position_value': round(position_value, 2),
            'risk_amount': round(final_risk, 2),
            'risk_percent': round((final_risk / capital) * 100, 3),
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'kelly_factor': round(kelly_factor, 3),
            'correlation_factor': round(correlation_factor, 3),
            'final_adjustment_factor': round(final_adjustment_factor, 3),
            'market_analysis': market_conditions,
            'recommendations': _generate_recommendations(market_conditions, safe_position_size, capital)
        }
        
    except Exception as e:
        logger.error(f"Error in calculate_optimal_position_size: {e}")
        return _get_default_result()

def _get_default_result() -> Dict[str, Union[float, Dict]]:
    """نتیجه پیش‌فرض در صورت بروز خطا"""
    return {
        'position_size': 0,
        'position_value': 0,
        'risk_amount': 0,
        'risk_percent': 0,
        'entry_price': 0,
        'stop_loss': 0,
        'kelly_factor': 1.0,
        'correlation_factor': 1.0,
        'final_adjustment_factor': 1.0,
        'market_analysis': {},
        'recommendations': []
    }

def _generate_recommendations(market_conditions: Dict[str, float], position_size: float, capital: float) -> List[str]:
    """تولید توصیه‌های معاملاتی بر اساس تحلیل"""
    recommendations = []
    
    volatility = market_conditions.get('volatility_score', 3)
    trend_strength = market_conditions.get('trend_strength', 50)
    momentum = market_conditions.get('momentum_score', 50)
    
    if volatility > 6:
        recommendations.append("نوسانات بالا - اندازه پوزیشن کاهش یافته")
    
    if trend_strength > 75:
        recommendations.append("ترند قوی شناسایی شد - فرصت مناسب برای معامله")
    elif trend_strength < 25:
        recommendations.append("ترند ضعیف - احتیاط در معامله")
    
    if momentum > 75:
        recommendations.append("اشباع خرید - احتمال تصحیح قیمت")
    elif momentum < 25:
        recommendations.append("اشباع فروش - احتمال بازگشت قیمت")
    
    position_percent = (position_size * market_conditions.get('entry_price', 1)) / capital * 100
    if position_percent > 10:
        recommendations.append("اندازه پوزیشن بالا - مراقب مدیریت ریسک باشید")
    
    return recommendations

# توابع کمکی برای سازگاری با کد قبلی
def adaptive_position_sizing(capital, risk_percent, entry_price, stop_loss, market_conditions=None):
    """تابع سازگاری برای محاسبه تطبیقی اندازه پوزیشن"""
    dummy_df = pd.DataFrame({'close': [entry_price]})
    result = calculate_optimal_position_size(
        df=dummy_df,
        capital=capital,
        risk_percent=risk_percent,
        entry_price=entry_price,
        stop_loss=stop_loss
    )
    return result['position_size']

def calculate_position_size_atr(capital, risk_percent, atr_value, atr_multiplier=2):
    """تابع سازگاری برای محاسبه اندازه پوزیشن بر اساس ATR"""
    try:
        risk_amount = capital * (risk_percent / 100)
        stop_distance = atr_value * atr_multiplier
        position_size = risk_amount / stop_distance
        return min(position_size, capital * 0.1)
    except Exception as e:
        logger.error(f"Error in calculate_position_size_atr: {e}")
        return 0