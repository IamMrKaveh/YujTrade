
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from models.indicator import IndicatorResult, TechnicalIndicator

class MovingAverageIndicator(TechnicalIndicator):
    def __init__(self, period: int, ma_type: str = "sma"):
        self.period = period
        self.ma_type = ma_type
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if self.ma_type == "sma":
            ma_values = data['close'].rolling(window=self.period).mean()
        else:
            ma_values = data['close'].ewm(span=self.period).mean()
        
        current_price = data['close'].iloc[-1]
        current_ma = ma_values.iloc[-1]
        
        signal_strength = abs((current_price - current_ma) / current_ma) * 100
        
        if current_price > current_ma:
            interpretation = "bullish_above_ma"
        else:
            interpretation = "bearish_below_ma"
        
        return IndicatorResult(
            name=f"{self.ma_type.upper()}_{self.period}",
            value=current_ma,
            signal_strength=signal_strength,
            interpretation=interpretation
        )

class RSIIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        if current_rsi > 70:
            interpretation = "overbought"
            signal_strength = (current_rsi - 70) / 30 * 100
        elif current_rsi < 30:
            interpretation = "oversold"
            signal_strength = (30 - current_rsi) / 30 * 100
        else:
            interpretation = "neutral"
            signal_strength = 50 - abs(current_rsi - 50)
        
        return IndicatorResult(
            name="RSI",
            value=current_rsi,
            signal_strength=signal_strength,
            interpretation=interpretation
        )

class MACDIndicator(TechnicalIndicator):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        ema_fast = data['close'].ewm(span=self.fast).mean()
        ema_slow = data['close'].ewm(span=self.slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal).mean()
        histogram = macd_line - signal_line
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        
        if current_macd > current_signal and current_histogram > 0:
            interpretation = "bullish_crossover"
            signal_strength = abs(current_histogram) * 100
        elif current_macd < current_signal and current_histogram < 0:
            interpretation = "bearish_crossover"
            signal_strength = abs(current_histogram) * 100
        else:
            interpretation = "neutral"
            signal_strength = 50
        
        return IndicatorResult(
            name="MACD",
            value=current_macd,
            signal_strength=signal_strength,
            interpretation=interpretation
        )

class BollingerBandsIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20, std_dev: float = 2):
        self.period = period
        self.std_dev = std_dev
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        sma = data['close'].rolling(window=self.period).mean()
        std = data['close'].rolling(window=self.period).std()
        upper_band = sma + (std * self.std_dev)
        lower_band = sma - (std * self.std_dev)
        
        current_price = data['close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_middle = sma.iloc[-1]
        
        bb_position = (current_price - current_lower) / (current_upper - current_lower)
        
        if bb_position > 0.8:
            interpretation = "near_upper_band"
            signal_strength = (bb_position - 0.8) / 0.2 * 100
        elif bb_position < 0.2:
            interpretation = "near_lower_band"
            signal_strength = (0.2 - bb_position) / 0.2 * 100
        else:
            interpretation = "middle_range"
            signal_strength = 50
        
        return IndicatorResult(
            name="BB",
            value=bb_position,
            signal_strength=signal_strength,
            interpretation=interpretation
        )

class StochasticIndicator(TechnicalIndicator):
    def __init__(self, k_period: int = 14, d_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        lowest_low = data['low'].rolling(window=self.k_period).min()
        highest_high = data['high'].rolling(window=self.k_period).max()
        k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=self.d_period).mean()
        
        current_k = k_percent.iloc[-1]
        current_d = d_percent.iloc[-1]
        
        if current_k > 80 and current_d > 80:
            interpretation = "overbought"
            signal_strength = ((current_k + current_d) / 2 - 80) / 20 * 100
        elif current_k < 20 and current_d < 20:
            interpretation = "oversold"
            signal_strength = (20 - (current_k + current_d) / 2) / 20 * 100
        else:
            interpretation = "neutral"
            signal_strength = 50
        
        return IndicatorResult(
            name="STOCH",
            value=(current_k + current_d) / 2,
            signal_strength=signal_strength,
            interpretation=interpretation
        )

class VolumeIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        volume_ma = data['volume'].rolling(window=self.period).mean()
        current_volume = data['volume'].iloc[-1]
        average_volume = volume_ma.iloc[-1]
        
        volume_ratio = current_volume / average_volume
        
        if volume_ratio > 1.5:
            interpretation = "high_volume"
            signal_strength = min((volume_ratio - 1) * 50, 100)
        elif volume_ratio < 0.5:
            interpretation = "low_volume"
            signal_strength = (1 - volume_ratio) * 100
        else:
            interpretation = "normal_volume"
            signal_strength = 50
        
        return IndicatorResult(
            name="VOLUME",
            value=volume_ratio,
            signal_strength=signal_strength,
            interpretation=interpretation
        )

class ATRIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=self.period).mean()
        
        current_atr = atr.iloc[-1]
        current_price = data['close'].iloc[-1]
        atr_percentage = (current_atr / current_price) * 100
        
        if atr_percentage > 3:
            interpretation = "high_volatility"
            signal_strength = min(atr_percentage * 20, 100)
        elif atr_percentage < 1:
            interpretation = "low_volatility"
            signal_strength = (1 - atr_percentage) * 100
        else:
            interpretation = "normal_volatility"
            signal_strength = 50
        
        return IndicatorResult(
            name="ATR",
            value=current_atr,
            signal_strength=signal_strength,
            interpretation=interpretation
        )

class FibonacciLevels:
    @staticmethod
    def calculate_retracement_levels(high: float, low: float) -> Dict[str, float]:
        diff = high - low
        return {
            '0.236': high - 0.236 * diff,
            '0.382': high - 0.382 * diff,
            '0.500': high - 0.500 * diff,
            '0.618': high - 0.618 * diff,
            '0.786': high - 0.786 * diff
        }
    
    @staticmethod
    def calculate_extension_levels(high: float, low: float, entry: float) -> Dict[str, float]:
        range_size = high - low
        return {
            '1.272': entry + 1.272 * range_size,
            '1.414': entry + 1.414 * range_size,
            '1.618': entry + 1.618 * range_size,
            '2.000': entry + 2.000 * range_size
        }

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
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }



class SupportResistanceAnalyzer:
    def __init__(self, lookback_period: int = 50):
        self.lookback_period = lookback_period
    
    def find_support_resistance(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        if len(data) < self.lookback_period:
            return [], []
        
        recent_data = data.tail(self.lookback_period)
        
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        
        resistance_levels = []
        support_levels = []
        
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                resistance_levels.append(highs[i])
        
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                support_levels.append(lows[i])
        
        resistance_levels = sorted(set(resistance_levels), reverse=True)[:5]
        support_levels = sorted(set(support_levels), reverse=True)[:5]
        
        return support_levels, resistance_levels

class VolumeAnalyzer:
    def analyze_volume_pattern(self, data: pd.DataFrame) -> Dict[str, float]:
        volume_ma_20 = data['volume'].rolling(window=20).mean()
        current_volume = data['volume'].iloc[-1]
        avg_volume = volume_ma_20.iloc[-1]
        
        volume_trend = self._calculate_volume_trend(data)
        volume_breakout = current_volume / avg_volume
        
        return {
            'volume_ratio': volume_breakout,
            'volume_trend': volume_trend,
            'volume_strength': min(volume_breakout * 50, 100),
            'volume_confirmation': volume_breakout > 1.2 and volume_trend > 0
        }
    
    def _calculate_volume_trend(self, data: pd.DataFrame) -> float:
        recent_volumes = data['volume'].tail(10)
        if len(recent_volumes) < 2:
            return 0
        
        volume_changes = recent_volumes.pct_change().dropna()
        return volume_changes.mean()
