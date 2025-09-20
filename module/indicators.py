from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
import pandas as pd

from module.core import IndicatorResult
from module.logger_config import logger


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