from abc import ABC, abstractmethod
from typing import Dict, List, Protocol, Tuple
import numpy as np
import pandas as pd

from logger_config import logger
from models import DynamicLevels, IndicatorResult, MarketAnalysis, MarketCondition, SignalType, TrendDirection, TrendStrength


class IndicatorInterface(Protocol):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        ...

class TechnicalIndicator(ABC):
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        pass

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
        if 'close' not in data.columns:
            logger.error("Data does not contain 'close' column for RSI calculation")
            raise ValueError("Data must contain 'close' column")
        
        if len(data) < self.period + 1:
            logger.error(f"Insufficient data for RSI calculation. Need at least {self.period + 1} records")
            return IndicatorResult(name="RSI", value=50, signal_strength=0, interpretation="neutral")
        
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.period, min_periods=self.period).mean()
        avg_loss = loss.rolling(window=self.period, min_periods=self.period).mean()
        
        current_gain = avg_gain.iloc[-1]
        current_loss = avg_loss.iloc[-1]

        if pd.isna(current_gain) or pd.isna(current_loss):
            return IndicatorResult(name="RSI", value=50, signal_strength=0, interpretation="neutral")
            
        if current_loss == 0 or current_loss < 1e-10:
            current_rsi = 100.0
        else:
            rs = current_gain / current_loss
            current_rsi = 100 - (100 / (1 + rs))
        
        current_rsi = max(0, min(100, current_rsi))
            
        if current_rsi > 70:
            interpretation = "overbought"
            signal_strength = min((current_rsi - 70) / 30 * 100, 100)
        elif current_rsi < 30:
            interpretation = "oversold"
            signal_strength = min((30 - current_rsi) / 30 * 100, 100)
        else:
            interpretation = "neutral"
            signal_strength = max(0, 50 - abs(current_rsi - 50))
            
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
        if len(data) < max(self.slow, self.fast, self.signal) + 1:
            return IndicatorResult(name="MACD", value=0, signal_strength=0, interpretation="neutral")
            
        ema_fast = data['close'].ewm(span=self.fast, min_periods=self.fast).mean()
        ema_slow = data['close'].ewm(span=self.slow, min_periods=self.slow).mean()
        
        if pd.isna(ema_fast.iloc[-1]) or pd.isna(ema_slow.iloc[-1]):
            return IndicatorResult(name="MACD", value=0, signal_strength=0, interpretation="neutral")
            
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, min_periods=self.signal).mean()
        histogram = macd_line - signal_line
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        
        if pd.isna(current_macd) or pd.isna(current_signal) or pd.isna(current_histogram):
            return IndicatorResult(name="MACD", value=0, signal_strength=0, interpretation="neutral")
        
        if current_macd > current_signal and current_histogram > 0:
            interpretation = "bullish_crossover"
            signal_strength = min(abs(current_histogram) * 1000, 100)
        elif current_macd < current_signal and current_histogram < 0:
            interpretation = "bearish_crossover"
            signal_strength = min(abs(current_histogram) * 1000, 100)
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
        if len(data) < self.period:
            return IndicatorResult(name="BB", value=0.5, signal_strength=0, interpretation="neutral")
            
        sma = data['close'].rolling(window=self.period, min_periods=self.period).mean()
        std = data['close'].rolling(window=self.period, min_periods=self.period).std()
        
        current_sma = sma.iloc[-1]
        current_std = std.iloc[-1]
        
        if pd.isna(current_sma) or pd.isna(current_std):
            return IndicatorResult(name="BB", value=0.5, signal_strength=0, interpretation="neutral")
        
        upper_band = current_sma + (current_std * self.std_dev)
        lower_band = current_sma - (current_std * self.std_dev)
        
        current_price = data['close'].iloc[-1]
        
        band_width = upper_band - lower_band
        if band_width == 0 or band_width < 1e-10:
            return IndicatorResult(name="BB", value=0.5, signal_strength=0, interpretation="neutral")
        
        bb_position = (current_price - lower_band) / band_width
        bb_position = max(0, min(1, bb_position))
        
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
        if len(data) < self.k_period:
            return IndicatorResult(name="STOCH", value=50, signal_strength=0, interpretation="neutral")
            
        lowest_low = data['low'].rolling(window=self.k_period, min_periods=self.k_period).min()
        highest_high = data['high'].rolling(window=self.k_period, min_periods=self.k_period).max()
        
        range_high_low = highest_high - lowest_low
        
        if pd.isna(range_high_low.iloc[-1]) or range_high_low.iloc[-1] == 0 or range_high_low.iloc[-1] < 1e-10:
            return IndicatorResult(name="STOCH", value=50, signal_strength=0, interpretation="neutral")
        
        k_percent = ((data['close'] - lowest_low) / range_high_low) * 100
        
        k_percent = k_percent.dropna()
        if len(k_percent) == 0:
            return IndicatorResult(name="STOCH", value=50, signal_strength=0, interpretation="neutral")
            
        k_percent = k_percent.clip(0, 100)
        
        if len(k_percent) < self.d_period:
            d_percent = k_percent
        else:
            d_percent = k_percent.rolling(window=self.d_period, min_periods=1).mean()
        
        current_k = k_percent.iloc[-1]
        current_d = d_percent.iloc[-1]
        
        if pd.isna(current_k) or pd.isna(current_d):
            return IndicatorResult(name="STOCH", value=50, signal_strength=0, interpretation="neutral")
        
        avg_stoch = (current_k + current_d) / 2
        
        if avg_stoch > 80:
            interpretation = "overbought"
            signal_strength = min((avg_stoch - 80) / 20 * 100, 100)
        elif avg_stoch < 20:
            interpretation = "oversold"
            signal_strength = min((20 - avg_stoch) / 20 * 100, 100)
        else:
            interpretation = "neutral"
            signal_strength = max(0, 50 - abs(avg_stoch - 50))
            
        return IndicatorResult(
            name="STOCH", 
            value=avg_stoch, 
            signal_strength=signal_strength, 
            interpretation=interpretation
        )

class VolumeIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period or 'volume' not in data.columns:
            return IndicatorResult(name="VOLUME", value=1, signal_strength=50, interpretation="normal_volume")
            
        volume_ma = data['volume'].rolling(window=self.period, min_periods=self.period).mean()
        current_volume = data['volume'].iloc[-1]
        average_volume = volume_ma.iloc[-1]
        
        if pd.isna(average_volume) or pd.isna(current_volume):
            return IndicatorResult(name="VOLUME", value=1, signal_strength=50, interpretation="normal_volume")
            
        if average_volume == 0:
            volume_ratio = 1
        else:
            volume_ratio = current_volume / average_volume
            
        volume_ratio = max(0, volume_ratio)
            
        if volume_ratio > 1.5:
            interpretation = "high_volume"
            signal_strength = min((volume_ratio - 1) * 50, 100)
        elif volume_ratio < 0.5:
            interpretation = "low_volume"
            signal_strength = max(0, (1 - volume_ratio) * 100)
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
        if len(data) < self.period + 1:
            return IndicatorResult(name="ATR", value=0, signal_strength=0, interpretation="neutral")
            
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        atr = true_range.rolling(window=self.period, min_periods=self.period).mean()
        current_atr = atr.iloc[-1]
        
        if pd.isna(current_atr) or current_atr < 0:
            return IndicatorResult(name="ATR", value=0, signal_strength=0, interpretation="neutral")
            
        current_price = data['close'].iloc[-1]
        
        if current_price == 0 or pd.isna(current_price):
            atr_percentage = 0
        else:
            atr_percentage = (current_atr / current_price) * 100
            
        if atr_percentage > 3:
            interpretation = "high_volatility"
            signal_strength = min(atr_percentage * 20, 100)
        elif atr_percentage < 1:
            interpretation = "low_volatility"
            signal_strength = max(0, (1 - atr_percentage) * 100)
        else:
            interpretation = "normal_volatility"
            signal_strength = 50
            
        return IndicatorResult(
            name="ATR",
            value=current_atr,
            signal_strength=signal_strength,
            interpretation=interpretation
        )

class IchimokuIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 24:
            return IndicatorResult(name="Ichimoku", value=0, signal_strength=0, interpretation="neutral")
        high_9 = data['high'].rolling(window=6).max().iloc[-1]
        low_9 = data['low'].rolling(window=6).min().iloc[-1]
        tenkan = (high_9 + low_9) / 2
        high_26 = data['high'].rolling(window=12).max().iloc[-1]
        low_26 = data['low'].rolling(window=12).min().iloc[-1]
        kijun = (high_26 + low_26) / 2
        senkou_a = (tenkan + kijun) / 2
        high_52 = data['high'].rolling(window=24).max().iloc[-1]
        low_52 = data['low'].rolling(window=24).min().iloc[-1]
        senkou_b = (high_52 + low_52) / 2
        current_price = data['close'].iloc[-1]
        
        if current_price > senkou_a and current_price > senkou_b:
            interpretation = "price_above_cloud"
            signal_strength = 100
        elif current_price < senkou_a and current_price < senkou_b:
            interpretation = "price_below_cloud"
            signal_strength = 100
        else:
            interpretation = "price_in_cloud"
            signal_strength = 50
        return IndicatorResult(
            name="Ichimoku",
            value=current_price,
            signal_strength=signal_strength,
            interpretation=interpretation
        )

class WilliamsRIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period
        self.wiliamsR = 'Williams %R'
        
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            return IndicatorResult(name=self.wiliamsR, value=-50, signal_strength=0, interpretation="neutral")
            
        highest_high = data['high'].rolling(window=self.period, min_periods=self.period).max().iloc[-1]
        lowest_low = data['low'].rolling(window=self.period, min_periods=self.period).min().iloc[-1]
        current_close = data['close'].iloc[-1]
        
        if pd.isna(highest_high) or pd.isna(lowest_low) or pd.isna(current_close):
            return IndicatorResult(name=self.wiliamsR, value=-50, signal_strength=0, interpretation="neutral")
            
        if highest_high - lowest_low == 0 or highest_high - lowest_low < 1e-10:
            value = -50
        else:
            value = (highest_high - current_close) / (highest_high - lowest_low) * -100
            
        value = max(-100, min(0, value))
        
        if value > -20:
            interpretation = "overbought"
            signal_strength = abs(value) / 20 * 100
        elif value < -80:
            interpretation = "oversold"
            signal_strength = (abs(value) - 80) / 20 * 100
        else:
            interpretation = "neutral"
            signal_strength = max(0, 50 - abs(value + 50) / 30 * 50)
            
        return IndicatorResult(
            name=self.wiliamsR,
            value=value,
            signal_strength=signal_strength,
            interpretation=interpretation
        )

class CCIIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period
        
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            return IndicatorResult(name="CCI", value=0, signal_strength=0, interpretation="neutral")
            
        tp = (data['high'] + data['low'] + data['close']) / 3
        sma = tp.rolling(window=self.period, min_periods=self.period).mean()
        
        current_tp = tp.iloc[-1]
        current_sma = sma.iloc[-1]
        
        if pd.isna(current_sma) or pd.isna(current_tp):
            return IndicatorResult(name="CCI", value=0, signal_strength=0, interpretation="neutral")
        
        recent_tp = tp.tail(self.period)
        if len(recent_tp) < self.period:
            return IndicatorResult(name="CCI", value=0, signal_strength=0, interpretation="neutral")
            
        mean_dev = (recent_tp - current_sma).abs().mean()
        
        if pd.isna(mean_dev) or mean_dev == 0 or mean_dev < 1e-10:
            return IndicatorResult(name="CCI", value=0, signal_strength=0, interpretation="neutral")
        
        cci = (current_tp - current_sma) / (0.015 * mean_dev)
        cci = max(-300, min(300, cci))
        
        if cci > 100:
            interpretation = "overbought"
            signal_strength = min((abs(cci) - 100) / 200 * 100, 100)
        elif cci < -100:
            interpretation = "oversold"
            signal_strength = min((abs(cci) - 100) / 200 * 100, 100)
        else:
            interpretation = "neutral"
            signal_strength = max(0, 50 - abs(cci) / 2)
            
        return IndicatorResult(
            name="CCI",
            value=cci,
            signal_strength=signal_strength,
            interpretation=interpretation
        )

class SuperTrendIndicator(TechnicalIndicator):
    def __init__(self, period=7, multiplier=3):
        self.period = period
        self.multiplier = multiplier
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.period).mean()
        
        hl2 = (high + low) / 2
        upperband = hl2 + (self.multiplier * atr)
        lowerband = hl2 - (self.multiplier * atr)
        
        supertrend = pd.Series(index=data.index, dtype=float)
        direction = pd.Series(index=data.index, dtype=bool)
        
        direction.iloc[0] = True
        supertrend.iloc[0] = lowerband.iloc[0] if direction.iloc[0] else upperband.iloc[0]
        
        for i in range(1, len(data)):
            if close.iloc[i] > upperband.iloc[i-1]:
                direction.iloc[i] = True
            elif close.iloc[i] < lowerband.iloc[i-1]:
                direction.iloc[i] = False
            else:
                direction.iloc[i] = direction.iloc[i-1]
            
            if direction.iloc[i]:
                lowerband.iloc[i] = max(lowerband.iloc[i], lowerband.iloc[i-1])
                supertrend.iloc[i] = lowerband.iloc[i]
            else:
                upperband.iloc[i] = min(upperband.iloc[i], upperband.iloc[i-1])
                supertrend.iloc[i] = upperband.iloc[i]
        
        current_value = supertrend.iloc[-1]
        current_price = close.iloc[-1]
        
        if current_price > current_value:
            interpretation = "bullish"
            strength = 100
        elif current_price < current_value:
            interpretation = "bearish"
            strength = 100
        else:
            interpretation = "neutral"
            strength = 50
            
        return IndicatorResult(
            name="SuperTrend",
            value=current_value,
            signal_strength=strength,
            interpretation=interpretation
            )

class ADXIndicator(TechnicalIndicator):
    def __init__(self, period=14):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
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
        
        plus_dm_smooth = plus_dm.rolling(self.period).mean()
        minus_dm_smooth = minus_dm.rolling(self.period).mean()
        tr_smooth = tr.rolling(self.period).mean()
        
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        plus_di = plus_di.fillna(0)
        minus_di = minus_di.fillna(0)
        
        dx_denominator = plus_di + minus_di
        dx = 100 * (plus_di - minus_di).abs() / dx_denominator.where(dx_denominator != 0, np.nan)
        dx = dx.fillna(0)
        
        adx = dx.rolling(self.period).mean().iloc[-1]
        
        if pd.isna(adx):
            adx = 0
            
        if adx > 25:
            interpretation = "strong_trend"
            strength = min(adx, 100)
        else:
            interpretation = "weak_trend"
            strength = min(adx, 100)

        return IndicatorResult(
            name="ADX",
            value=adx,
            signal_strength=strength,
            interpretation=interpretation
        )

class ChaikinMoneyFlowIndicator(TechnicalIndicator):
    def __init__(self, period=20):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
        mfv = mfm * volume
        
        cmf = mfv.rolling(self.period).sum() / volume.rolling(self.period).sum()
        current_value = cmf.iloc[-1]
        
        if pd.isna(current_value):
            current_value = 0
            
        if current_value > 0:
            interpretation = "buy_pressure"
            strength = min(abs(current_value) * 100, 100)
        elif current_value < 0:
            interpretation = "sell_pressure"
            strength = min(abs(current_value) * 100, 100)
        else:
            interpretation = "neutral"
            strength = 50

        return IndicatorResult(
            name="CMF",
            value=current_value,
            signal_strength=strength,
            interpretation=interpretation
        )

class OBVIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        close = data['close']
        volume = data['volume']
        
        price_change = close.diff()
        volume_direction = np.sign(price_change)
        volume_direction = volume_direction.fillna(0)
        
        obv = (volume * volume_direction).cumsum()
        
        current_value = obv.iloc[-1]
        previous_value = obv.iloc[-2] if len(obv) > 1 else current_value
        
        if current_value > previous_value:
            interpretation = "bullish"
            strength = 100
        elif current_value < previous_value:
            interpretation = "bearish"
            strength = 100
        else:
            interpretation = "neutral"
            strength = 50

        return IndicatorResult(
            name="OBV",
            value=current_value,
            signal_strength=strength,
            interpretation=interpretation
        )

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
    
    def find_support_resistance(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        if len(data) < self.lookback_period:
            return [], []
        recent_data = data.tail(self.lookback_period)
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        resistance_levels = []
        support_levels = []
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                resistance_levels.append(highs[i])
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]):
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
