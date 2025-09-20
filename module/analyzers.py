from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from module.core import (DynamicLevels, MarketAnalysis, MarketCondition,
                         SignalType, TrendDirection, TrendStrength)
from module.indicators import ATRIndicator, RSIIndicator


class FibonacciLevels:
    @staticmethod
    def calculate_retracement_levels(high: float, low: float) -> Dict[str, float]:
        diff = high - low
        return {'0.236': high - 0.236 * diff, '0.382': high - 0.382 * diff, '0.500': high - 0.500 * diff, '0.618': high - 0.618 * diff, '0.786': high - 0.786 * diff}
    
    @staticmethod
    def calculate_extension_levels(high: float, low: float, entry: float) -> Dict[str, float]:
        range_size = high - low
        return {'1.272': entry + 1.272 * range_size, '1.414': entry + 1.414 * range_size, '1.618': entry + 1.618 * range_size, '2.000': entry + 2.000 * range_size}

class PivotPoints:
    @staticmethod
    def calculate_pivot_levels(high: float, low: float, close: float) -> Dict[str, float]:
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        return {'pivot': pivot, 'r1': r1, 'r2': r2, 'r3': r3, 's1': s1, 's2': s2, 's3': s3}

class PatternAnalyzer:
    @staticmethod
    def detect_patterns(data: pd.DataFrame) -> List[str]:
        patterns = []
        if len(data) < 2:
            return patterns
        prev_open = data['open'].iloc[-2]
        prev_close = data['close'].iloc[-2]
        last_open = data['open'].iloc[-1]
        last_close = data['close'].iloc[-1]
        if prev_close < prev_open and last_close > last_open and last_close > prev_open and last_open <= prev_close:
            patterns.append("bullish_engulfing")
        if prev_close > prev_open and last_close < last_open and last_close < prev_open and last_open >= prev_close:
            patterns.append("bearish_engulfing")
        highs = data['high']
        lows = data['low']
        local_max = []
        local_min = []
        for i in range(1, len(data)-1):
            if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                local_max.append((float(highs.iloc[i]), i))
            if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                local_min.append((float(lows.iloc[i]), i))
        if len(local_max) >= 2:
            sorted_max = sorted(local_max, key=lambda x: x[0], reverse=True)
            top1, top2 = sorted_max[0][0], sorted_max[1][0]
            if abs(top1 - top2) / ((top1+top2)/2) < 0.02:
                patterns.append("double_top")
        if len(local_min) >= 2:
            sorted_min = sorted(local_min, key=lambda x: x[0])
            bot1, bot2 = sorted_min[0][0], sorted_min[1][0]
            if abs(bot1 - bot2) / ((bot1+bot2)/2) < 0.02:
                patterns.append("double_bottom")
        if len(local_max) >= 3:
            sorted_by_index = sorted(local_max, key=lambda x: x[1])
            for i in range(len(sorted_by_index)-2):
                left, center, right = sorted_by_index[i], sorted_by_index[i+1], sorted_by_index[i+2]
                if center[0] > left[0] and center[0] > right[0] and left[0] < right[0]*1.05 and right[0] < left[0]*1.05:
                    patterns.append("head_and_shoulders")
                    break
        if len(local_min) >= 3:
            sorted_by_index = sorted(local_min, key=lambda x: x[1])
            for i in range(len(sorted_by_index)-2):
                left, center, right = sorted_by_index[i], sorted_by_index[i+1], sorted_by_index[i+2]
                if center[0] < left[0] and center[0] < right[0] and left[0] > right[0]*0.95 and right[0] > left[0]*0.95:
                    patterns.append("inverse_head_and_shoulders")
                    break
        return patterns
    
    @staticmethod
    def detect_flag(data: pd.DataFrame) -> bool:
        if len(data) < 20:
            return False
        recent = data.tail(20)
        highs = recent['high']
        lows = recent['low']
        if highs.max() - highs.min() < (data['close'].iloc[-20] * 0.02):
            return True
        return False
    
    @staticmethod
    def detect_wedge(data: pd.DataFrame) -> bool:
        if len(data) < 30:
            return False
        recent = data.tail(30)
        slope_high = np.polyfit(range(len(recent)), recent['high'], 1)[0]
        slope_low = np.polyfit(range(len(recent)), recent['low'], 1)[0]
        return abs(slope_high) < 0.05 and abs(slope_low) < 0.05 and slope_high * slope_low < 0
    
    @staticmethod
    def detect_triangle(data: pd.DataFrame) -> bool:
        if len(data) < 30:
            return False
        recent = data.tail(30)
        highs = recent['high']
        lows = recent['low']
        h_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        l_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        return h_slope < 0 and l_slope > 0

class TrendAnalyzer:
    @staticmethod
    def calculate_trend_strength(data: pd.DataFrame) -> TrendStrength:
        if len(data) < 50:
            return TrendStrength.WEAK
        
        sma_20 = data['close'].rolling(window=20).mean()
        sma_50 = data['close'].rolling(window=50).mean()
        current_sma_20 = sma_20.iloc[-1]
        current_sma_50 = sma_50.iloc[-1]
        
        if pd.isna(current_sma_20) or pd.isna(current_sma_50) or current_sma_50 == 0:
            return TrendStrength.WEAK
            
        sma_separation = abs((current_sma_20 - current_sma_50) / current_sma_50)
        adx = TrendAnalyzer._calculate_adx(data)
        
        if adx > 40 and sma_separation > 0.03:
            return TrendStrength.STRONG
        elif adx > 25 and sma_separation > 0.015:
            return TrendStrength.MODERATE
        else:
            return TrendStrength.WEAK

    @staticmethod
    def _calculate_adx(data: pd.DataFrame, period: int = 14) -> float:
        if len(data) < period + 1:
            return 0
            
        high = data['high']
        low = data['low']
        close = data['close']
        
        high_diff = high.diff()
        low_diff = -low.diff()
        
        plus_dm = high_diff.where(high_diff > low_diff, 0).where(high_diff > 0, 0)
        minus_dm = low_diff.where(low_diff > high_diff, 0).where(low_diff > 0, 0)
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        plus_dm_smooth = plus_dm.rolling(window=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period).mean()
        tr_smooth = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        plus_di = plus_di.fillna(0)
        minus_di = minus_di.fillna(0)
        
        dx_denominator = plus_di + minus_di
        dx = 100 * (plus_di - minus_di).abs() / dx_denominator.where(dx_denominator != 0, np.nan)
        dx = dx.fillna(0)
        
        adx = dx.rolling(window=period).mean()
        
        final_adx = adx.iloc[-1]
        return final_adx if not pd.isna(final_adx) else 0

class SupportResistanceAnalyzer:
    def __init__(self, lookback_period: int = 50):
        self.lookback_period = lookback_period
    
    def find_support_resistance(self, data):
        if data is None:
            return [], []
        try:
            n_total = len(data)
        except Exception:
            return [], []
        if n_total < getattr(self, "lookback_period", 3):
            return [], []
        recent = data.tail(self.lookback_period)
        if 'high' not in recent.columns or 'low' not in recent.columns:
            return [], []
        highs = recent['high'].to_numpy()
        lows = recent['low'].to_numpy()
        resistance = []
        support = []
        n = len(highs)
        for i in range(1, n - 1):
            try:
                if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                    resistance.append(float(highs[i]))
                if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                    support.append(float(lows[i]))
            except Exception:
                continue
        resistance = sorted(list(dict.fromkeys([round(x, 8) for x in resistance])), reverse=True)
        support = sorted(list(dict.fromkeys([round(x, 8) for x in support])))
        return support, resistance

class VolumeAnalyzer:
    def analyze_volume_pattern(self, data: pd.DataFrame) -> Dict[str, float]:
        volume_ma_20 = data['volume'].rolling(window=20).mean()
        current_volume = data['volume'].iloc[-1]
        avg_volume = volume_ma_20.iloc[-1]
        volume_trend = self._calculate_volume_trend(data)
        volume_breakout = current_volume / avg_volume if avg_volume and not np.isnan(avg_volume) else 1
        return {'volume_ratio': volume_breakout, 'volume_trend': volume_trend, 'volume_strength': min(volume_breakout * 50, 100), 'volume_confirmation': volume_breakout > 1.2 and volume_trend > 0}
    
    def _calculate_volume_trend(self, data: pd.DataFrame) -> float:
        recent_volumes = data['volume'].tail(10)
        if len(recent_volumes) < 2:
            return 0
        volume_changes = recent_volumes.pct_change().dropna()
        return volume_changes.mean()
    
    def volume_profile(self, data: pd.DataFrame, bins: int = 30) -> List[Tuple[float, float]]:
        prices = data['close']
        volumes = data['volume']
        min_p, max_p = prices.min(), prices.max()
        if max_p == min_p:
            return [(min_p, volumes.sum())]
        bin_size = (max_p - min_p) / bins
        buckets = {}
        for p, v in zip(prices, volumes):
            idx = int((p - min_p) / bin_size)
            key = min_p + idx * bin_size
            buckets[key] = buckets.get(key, 0) + v
        items = sorted(buckets.items(), key=lambda x: x[0])
        return items
    
    def vwap(self, data: pd.DataFrame) -> float:
        pv = (data['close'] * data['volume']).sum()
        v = data['volume'].sum()
        return pv / v if v else data['close'].iloc[-1]

class MarketConditionAnalyzer:
    def __init__(self):
        self.support_resistance = SupportResistanceAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        
    def analyze_market_condition(self, data: pd.DataFrame) -> MarketAnalysis:
        support_levels, resistance_levels = self.support_resistance.find_support_resistance(data)
        volume_analysis = self.volume_analyzer.analyze_volume_pattern(data)
        trend = self._determine_trend(data)
        trend_strength = self.trend_analyzer.calculate_trend_strength(data)
        volatility = self._calculate_volatility(data)
        momentum_score = self._calculate_momentum(data)
        market_condition = self._determine_market_condition(data)
        trend_acceleration = self._calculate_trend_acceleration(data)
        
        return MarketAnalysis(
            trend=trend, 
            trend_strength=trend_strength, 
            volatility=volatility, 
            volume_trend="increasing" if volume_analysis['volume_trend'] > 0 else "decreasing", 
            support_levels=support_levels, 
            resistance_levels=resistance_levels, 
            momentum_score=momentum_score, 
            market_condition=market_condition, 
            trend_acceleration=trend_acceleration, 
            volume_confirmation=volume_analysis['volume_confirmation']
        )

    def _determine_trend(self, data: pd.DataFrame) -> TrendDirection:
        if len(data) < 50:
            return TrendDirection.SIDEWAYS
            
        sma_20 = data['close'].rolling(window=20).mean()
        sma_50 = data['close'].rolling(window=50).mean()
        
        current_sma_20 = sma_20.iloc[-1]
        current_sma_50 = sma_50.iloc[-1]
        
        if pd.isna(current_sma_20) or pd.isna(current_sma_50):
            return TrendDirection.SIDEWAYS
            
        if current_sma_20 > current_sma_50 * 1.015:
            return TrendDirection.BULLISH
        elif current_sma_20 < current_sma_50 * 0.985:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.SIDEWAYS

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        if len(data) < 2:
            return 0.0
            
        returns = data['close'].pct_change().dropna()
        if returns.empty or len(returns) < 2:
            return 0.0
        
        try:
            periods_per_year = 365
            if len(data.index) > 1 and hasattr(data, 'index'):
                try:
                    if pd.api.types.is_datetime64_any_dtype(data.index):
                        time_diff = data.index[-1] - data.index[0]
                        if hasattr(time_diff, 'total_seconds'):
                            avg_period_seconds = time_diff.total_seconds() / (len(data) - 1)
                            seconds_per_year = 365 * 24 * 3600
                            periods_per_year = seconds_per_year / avg_period_seconds
                        else:
                            periods_per_year = 365
                except:
                    periods_per_year = 365
            
            volatility = returns.std() * np.sqrt(periods_per_year)
            return volatility if not pd.isna(volatility) else 0.0
            
        except Exception:
            return returns.std() if not returns.empty else 0.0

    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        if len(data) < 20:
            return 0.0
            
        current_close = data['close'].iloc[-1]
        previous_close = data['close'].iloc[-20]
        
        if pd.isna(current_close) or pd.isna(previous_close) or previous_close == 0:
            return 0.0
            
        price_change = (current_close - previous_close) / previous_close
        return price_change * 100

    def _calculate_trend_acceleration(self, data: pd.DataFrame) -> float:
        if len(data) < 10:
            return 0.0
            
        current_close = data['close'].iloc[-1]
        mid_close = data['close'].iloc[-5]
        old_close = data['close'].iloc[-10]
        
        if pd.isna(current_close) or pd.isna(mid_close) or pd.isna(old_close):
            return 0.0
            
        if mid_close == 0 or old_close == 0:
            return 0.0
            
        recent_momentum = (current_close - mid_close) / mid_close
        previous_momentum = (mid_close - old_close) / old_close
        
        return (recent_momentum - previous_momentum) * 100

    def _determine_market_condition(self, data: pd.DataFrame) -> MarketCondition:
        try:
            rsi_indicator = RSIIndicator()
            rsi_result = rsi_indicator.calculate(data)
            
            if rsi_result.value > 70:
                return MarketCondition.OVERBOUGHT
            elif rsi_result.value < 30:
                return MarketCondition.OVERSOLD
            else:
                return MarketCondition.NEUTRAL
        except:
            return MarketCondition.NEUTRAL

class DynamicLevelCalculator:
    def __init__(self):
        self.fibonacci = FibonacciLevels()
        self.pivot_points = PivotPoints()
    
    def calculate_dynamic_levels(self, data: pd.DataFrame, signal_type: SignalType, market_analysis: MarketAnalysis) -> DynamicLevels:
        current_price = data['close'].iloc[-1]
        high_20 = data['high'].tail(20).max()
        low_20 = data['low'].tail(20).min()
        atr_indicator = ATRIndicator()
        atr_result = atr_indicator.calculate(data)
        atr_value = atr_result.value
        if signal_type == SignalType.BUY:
            return self._calculate_buy_levels(data, current_price, high_20, low_20, atr_value, market_analysis)
        else:
            return self._calculate_sell_levels(data, current_price, high_20, low_20, atr_value, market_analysis)
    
    def _calculate_buy_levels(self, data: pd.DataFrame, current_price: float, high_20: float, low_20: float, atr_value: float, market_analysis: MarketAnalysis) -> DynamicLevels:
        fib_levels = self.fibonacci.calculate_retracement_levels(high_20, low_20)
        pivot_levels = self.pivot_points.calculate_pivot_levels(data['high'].iloc[-1], data['low'].iloc[-1], data['close'].iloc[-1])
        trend_multiplier = self._get_trend_multiplier(market_analysis)
        volatility_multiplier = self._get_volatility_multiplier(market_analysis)
        primary_entry = current_price
        secondary_entry = min(fib_levels['0.382'], current_price * 0.995)
        if market_analysis.trend_strength == TrendStrength.STRONG:
            primary_exit = current_price * (1 + 0.03 * trend_multiplier)
            secondary_exit = current_price * (1 + 0.05 * trend_multiplier)
        elif market_analysis.trend_strength == TrendStrength.MODERATE:
            primary_exit = current_price * (1 + 0.02 * trend_multiplier)
            secondary_exit = current_price * (1 + 0.035 * trend_multiplier)
        else:
            primary_exit = current_price * (1 + 0.015)
            secondary_exit = current_price * (1 + 0.025)
        if market_analysis.resistance_levels:
            nearest_resistance = min([r for r in market_analysis.resistance_levels if r > current_price], default=primary_exit)
            primary_exit = min(primary_exit, nearest_resistance)
            secondary_exit = min(secondary_exit, nearest_resistance * 1.02)
        tight_stop = max(current_price - (atr_value * volatility_multiplier), fib_levels['0.618'] if fib_levels['0.618'] < current_price else current_price * 0.98)
        wide_stop = max(current_price - (atr_value * 2 * volatility_multiplier), low_20 * 0.995)
        if market_analysis.support_levels:
            nearest_support = max([s for s in market_analysis.support_levels if s < current_price], default=tight_stop)
            tight_stop = max(tight_stop, nearest_support)
            wide_stop = max(wide_stop, nearest_support * 0.995)
        breakeven_point = current_price + (atr_value * 0.5)
        trailing_stop = current_price - (atr_value * 1.5 * volatility_multiplier)
        return DynamicLevels(primary_entry=primary_entry, secondary_entry=secondary_entry, primary_exit=primary_exit, secondary_exit=secondary_exit, tight_stop=tight_stop, wide_stop=wide_stop, breakeven_point=breakeven_point, trailing_stop=trailing_stop)
    
    def _calculate_sell_levels(self, data: pd.DataFrame, current_price: float, high_20: float, low_20: float, atr_value: float, market_analysis: MarketAnalysis) -> DynamicLevels:
        fib_levels = self.fibonacci.calculate_retracement_levels(high_20, low_20)
        pivot_levels = self.pivot_points.calculate_pivot_levels(data['high'].iloc[-1], data['low'].iloc[-1], data['close'].iloc[-1])
        trend_multiplier = self._get_trend_multiplier(market_analysis)
        volatility_multiplier = self._get_volatility_multiplier(market_analysis)
        primary_entry = current_price
        secondary_entry = max(fib_levels['0.382'], current_price * 1.005)
        if market_analysis.trend_strength == TrendStrength.STRONG:
            primary_exit = current_price * (1 - 0.03 * trend_multiplier)
            secondary_exit = current_price * (1 - 0.05 * trend_multiplier)
        elif market_analysis.trend_strength == TrendStrength.MODERATE:
            primary_exit = current_price * (1 - 0.02 * trend_multiplier)
            secondary_exit = current_price * (1 - 0.035 * trend_multiplier)
        else:
            primary_exit = current_price * (1 - 0.015)
            secondary_exit = current_price * (1 - 0.025)
        if market_analysis.support_levels:
            nearest_support = max([s for s in market_analysis.support_levels if s < current_price], default=primary_exit)
            primary_exit = max(primary_exit, nearest_support)
            secondary_exit = max(secondary_exit, nearest_support * 0.98)
        tight_stop = min(current_price + (atr_value * volatility_multiplier), fib_levels['0.618'] if fib_levels['0.618'] > current_price else current_price * 1.02)
        wide_stop = min(current_price + (atr_value * 2 * volatility_multiplier), high_20 * 1.005)
        if market_analysis.resistance_levels:
            nearest_resistance = min([r for r in market_analysis.resistance_levels if r > current_price], default=tight_stop)
            tight_stop = min(tight_stop, nearest_resistance)
            wide_stop = min(wide_stop, nearest_resistance * 1.005)
        breakeven_point = current_price - (atr_value * 0.5)
        trailing_stop = current_price + (atr_value * 1.5 * volatility_multiplier)
        
        return DynamicLevels(
            primary_entry=primary_entry,
            secondary_entry=secondary_entry,
            primary_exit=primary_exit,
            secondary_exit=secondary_exit,
            tight_stop=tight_stop,
            wide_stop=wide_stop,
            breakeven_point=breakeven_point,
            trailing_stop=trailing_stop
            )
    
    def _get_trend_multiplier(self, market_analysis: MarketAnalysis) -> float:
        base_multiplier = 1.0
        if market_analysis.trend == TrendDirection.BULLISH:
            base_multiplier *= 1.2
        elif market_analysis.trend == TrendDirection.BEARISH:
            base_multiplier *= 1.2
        if market_analysis.trend_strength == TrendStrength.STRONG:
            base_multiplier *= 1.5
        elif market_analysis.trend_strength == TrendStrength.MODERATE:
            base_multiplier *= 1.2
        if market_analysis.volume_confirmation:
            base_multiplier *= 1.1
        if abs(market_analysis.trend_acceleration) > 1:
            base_multiplier *= 1.15
        return base_multiplier
    
    def _get_volatility_multiplier(self, market_analysis: MarketAnalysis) -> float:
        if market_analysis.volatility > 0.04:
            return 1.5
        elif market_analysis.volatility > 0.02:
            return 1.2
        else:
            return 1.0