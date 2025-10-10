from abc import ABC, abstractmethod
from typing import Protocol, Dict, Optional

import numpy as np
import pandas as pd
import talib

import pandas_ta as ta
from pandas_ta.momentum import stoch, roc, stochrsi, trix, uo, squeeze, cmo
from pandas_ta.overlap import ichimoku, supertrend, kama
from pandas_ta.volatility import bbands, atr, kc, massi
from pandas_ta.trend import adx as adx_ta, aroon, psar, dpo
from pandas_ta.volume import cmf, obv, ad, eom, efi, pvt

from ..core import IndicatorResult
from .base import TechnicalIndicator

class PivotPointsIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 2:
            return IndicatorResult(name="PivotPoints", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        prev_high = float(data['high'].iloc[-2])
        prev_low = float(data['low'].iloc[-2])
        prev_close = float(data['close'].iloc[-2])
        current_price = float(data['close'].iloc[-1])

        pivot = (prev_high + prev_low + prev_close) / 3
        r1 = 2 * pivot - prev_low
        s1 = 2 * pivot - prev_high

        if current_price > r1:
            interpretation = "above_resistance"
            strength = 100.0
        elif current_price < s1:
            interpretation = "below_support"
            strength = 100.0
        else:
            interpretation = "in_pivot_range"
            strength = 50.0

        return IndicatorResult(name="PivotPoints", value=pivot, signal_strength=strength, interpretation=interpretation)


class MedianPriceIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        median_price = talib.MEDPRICE(data['high'].to_numpy(dtype=np.float64), data['low'].to_numpy(dtype=np.float64))
        if median_price.size == 0 or pd.isna(median_price[-1]):
            return IndicatorResult(name="MedianPrice", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_median = float(median_price[-1])
        current_close = float(data['close'].iloc[-1])

        if current_close > current_median:
            interpretation = "above_median"
            strength = abs(current_close - current_median) / current_median * 100 if current_median > 0 else 0.0
        elif current_close < current_median:
            interpretation = "below_median"
            strength = abs(current_close - current_median) / current_median * 100 if current_median > 0 else 0.0
        else:
            interpretation = "at_median"
            strength = 50.0

        return IndicatorResult(name="MedianPrice", value=current_median, signal_strength=min(float(strength), 100.0), interpretation=interpretation)


class TypicalPriceIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        typical_price = talib.TYPPRICE(
            data['high'].to_numpy(dtype=np.float64), 
            data['low'].to_numpy(dtype=np.float64), 
            data['close'].to_numpy(dtype=np.float64)
        )
        if typical_price.size == 0 or pd.isna(typical_price[-1]):
            return IndicatorResult(name="TypicalPrice", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_typical = float(typical_price[-1])
        current_close = float(data['close'].iloc[-1])

        if current_close > current_typical:
            interpretation = "above_typical"
            strength = abs(current_close - current_typical) / current_typical * 100 if current_typical > 0 else 0.0
        elif current_close < current_typical:
            interpretation = "below_typical"
            strength = abs(current_close - current_typical) / current_typical * 100 if current_typical > 0 else 0.0
        else:
            interpretation = "at_typical"
            strength = 50.0

        return IndicatorResult(name="TypicalPrice", value=current_typical, signal_strength=min(float(strength), 100.0), interpretation=interpretation)


class WeightedClosePriceIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        weighted_close = talib.WCLPRICE(
            data['high'].to_numpy(dtype=np.float64), 
            data['low'].to_numpy(dtype=np.float64), 
            data['close'].to_numpy(dtype=np.float64)
        )
        if weighted_close.size == 0 or pd.isna(weighted_close[-1]):
            return IndicatorResult(name="WeightedClose", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_weighted = float(weighted_close[-1])
        current_close = float(data['close'].iloc[-1])

        if current_close > current_weighted:
            interpretation = "above_weighted"
            strength = abs(current_close - current_weighted) / current_weighted * 100 if current_weighted > 0 else 0.0
        elif current_close < current_weighted:
            interpretation = "below_weighted"
            strength = abs(current_close - current_weighted) / current_weighted * 100 if current_weighted > 0 else 0.0
        else:
            interpretation = "at_weighted"
            strength = 50.0

        return IndicatorResult(name="WeightedClose", value=current_weighted, signal_strength=min(float(strength), 100.0), interpretation=interpretation)


class FractalIndicator(TechnicalIndicator):
    def __init__(self, period: int = 2):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period * 2 + 1:
            return IndicatorResult(name="Fractal", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        idx = -1 - self.period
        if idx >= -len(data):
            high_center = data['high'].iloc[idx]
            low_center = data['low'].iloc[idx]
            
            is_up_fractal = all(data['high'].iloc[idx] > data['high'].iloc[i] for i in range(idx - self.period, idx)) and \
                           all(data['high'].iloc[idx] > data['high'].iloc[i] for i in range(idx + 1, idx + self.period + 1))
            
            is_down_fractal = all(data['low'].iloc[idx] < data['low'].iloc[i] for i in range(idx - self.period, idx)) and \
                             all(data['low'].iloc[idx] < data['low'].iloc[i] for i in range(idx + 1, idx + self.period + 1))
            
            if is_up_fractal:
                interpretation = "resistance_level"
                strength = 100.0
                value = float(high_center)
            elif is_down_fractal:
                interpretation = "support_level"
                strength = 100.0
                value = float(low_center)
            else:
                interpretation = "no_fractal"
                strength = 0.0
                value = float(data['close'].iloc[-1])
        else:
            interpretation = "insufficient_data"
            strength = 0.0
            value = np.nan

        return IndicatorResult(name="Fractal", value=value, signal_strength=strength, interpretation=interpretation)


class PriceActionPatternIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 5:
            return IndicatorResult(name="PriceActionPattern", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        recent_candles = data.tail(5)
        
        body_sizes = abs(recent_candles['close'] - recent_candles['open'])
        upper_shadows = recent_candles['high'] - recent_candles[['open', 'close']].max(axis=1)
        lower_shadows = recent_candles[['open', 'close']].min(axis=1) - recent_candles['low']
        
        last_candle = recent_candles.iloc[-1]
        prev_candle = recent_candles.iloc[-2]
        
        last_body = abs(last_candle['close'] - last_candle['open'])
        last_upper_shadow = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        last_lower_shadow = min(last_candle['open'], last_candle['close']) - last_candle['low']
        
        pattern_score = 0.0
        interpretation = "no_pattern"
        
        if last_lower_shadow > last_body * 2 and last_upper_shadow < last_body * 0.3:
            pattern_score = 80.0
            interpretation = "hammer_bullish"
        elif last_upper_shadow > last_body * 2 and last_lower_shadow < last_body * 0.3:
            pattern_score = 80.0
            interpretation = "shooting_star_bearish"
        elif (last_candle['close'] > prev_candle['high'] and 
              last_candle['open'] > prev_candle['close']):
            pattern_score = 75.0
            interpretation = "bullish_engulfing"
        elif (last_candle['close'] < prev_candle['low'] and 
              last_candle['open'] < prev_candle['close']):
            pattern_score = 75.0
            interpretation = "bearish_engulfing"
        elif body_sizes.mean() < (upper_shadows.mean() + lower_shadows.mean()):
            pattern_score = 60.0
            interpretation = "doji_indecision"
        else:
            pattern_score = 30.0
            interpretation = "continuation_pattern"

        return IndicatorResult(name="PriceActionPattern", value=pattern_score, signal_strength=float(pattern_score), interpretation=interpretation)


class LiquidityLevelsIndicator(TechnicalIndicator):
    def __init__(self, lookback: int = 50):
        self.lookback = lookback

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.lookback:
            return IndicatorResult(name="LiquidityLevels", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        recent_data = data.tail(self.lookback)
        
        volume_profile = {}
        for i in range(len(recent_data)):
            price = float(recent_data['close'].iloc[i])
            volume = float(recent_data['volume'].iloc[i])
            price_bucket = round(price, -int(np.log10(price)) + 2)
            
            if price_bucket in volume_profile:
                volume_profile[price_bucket] += volume
            else:
                volume_profile[price_bucket] = volume

        sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_levels:
            return IndicatorResult(name="LiquidityLevels", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        highest_volume_price = sorted_levels[0][0]
        current_price = float(data['close'].iloc[-1])
        
        distance_ratio = abs(current_price - highest_volume_price) / highest_volume_price
        
        if current_price > highest_volume_price:
            if distance_ratio < 0.02:
                interpretation = "near_support_zone"
                strength = 85.0
            else:
                interpretation = "above_liquidity"
                strength = 60.0
        else:
            if distance_ratio < 0.02:
                interpretation = "near_resistance_zone"
                strength = 85.0
            else:
                interpretation = "below_liquidity"
                strength = 60.0

        return IndicatorResult(name="LiquidityLevels", value=float(highest_volume_price), signal_strength=float(strength), interpretation=interpretation)


class SmartMoneyConceptIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            return IndicatorResult(name="SmartMoneyConcept", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        recent_data = data.tail(self.period)
        
        bullish_candles = (recent_data['close'] > recent_data['open']).sum()
        bearish_candles = (recent_data['close'] < recent_data['open']).sum()
        
        price_change = (float(recent_data['close'].iloc[-1]) - float(recent_data['close'].iloc[0])) / float(recent_data['close'].iloc[0])
        volume_change = (float(recent_data['volume'].iloc[-1]) - float(recent_data['volume'].iloc[0])) / float(recent_data['volume'].iloc[0])
        
        if price_change > 0 and volume_change < 0:
            interpretation = "distribution_phase"
            strength = min(abs(price_change) * 100 + abs(volume_change) * 50, 100.0)
        elif price_change < 0 and volume_change > 0:
            interpretation = "accumulation_phase"
            strength = min(abs(price_change) * 100 + abs(volume_change) * 50, 100.0)
        elif price_change > 0 and volume_change > 0:
            interpretation = "markup_phase"
            strength = min((price_change + volume_change) * 50, 100.0)
        elif price_change < 0 and volume_change < 0:
            interpretation = "markdown_phase"
            strength = min((abs(price_change) + abs(volume_change)) * 50, 100.0)
        else:
            interpretation = "consolidation"
            strength = 40.0

        smc_score = (bullish_candles - bearish_candles) / self.period * 100

        return IndicatorResult(name="SmartMoneyConcept", value=float(smc_score), signal_strength=float(strength), interpretation=interpretation)


class WyckoffVolumeSpreadIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 3:
            return IndicatorResult(name="WyckoffVSA", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current = data.iloc[-1]
        prev = data.iloc[-2]
        
        spread = float(current['high']) - float(current['low'])
        prev_spread = float(prev['high']) - float(prev['low'])
        
        volume_ratio = float(current['volume']) / float(prev['volume']) if float(prev['volume']) > 0 else 1.0
        spread_ratio = spread / prev_spread if prev_spread > 0 else 1.0
        
        close_position = (float(current['close']) - float(current['low'])) / spread if spread > 0 else 0.5
        
        if volume_ratio > 1.5 and spread_ratio > 1.2 and close_position > 0.7:
            interpretation = "strength_bullish"
            strength = min((volume_ratio + spread_ratio) * 30, 100.0)
        elif volume_ratio > 1.5 and spread_ratio > 1.2 and close_position < 0.3:
            interpretation = "weakness_bearish"
            strength = min((volume_ratio + spread_ratio) * 30, 100.0)
        elif volume_ratio < 0.7 and spread_ratio < 0.8:
            interpretation = "no_demand"
            strength = 70.0
        elif volume_ratio > 1.5 and spread_ratio < 0.8:
            interpretation = "stopping_volume"
            strength = 85.0
        else:
            interpretation = "neutral_vsa"
            strength = 50.0

        vsa_score = (volume_ratio * spread_ratio * close_position) * 33.33

        return IndicatorResult(name="WyckoffVSA", value=float(vsa_score), signal_strength=float(strength), interpretation=interpretation)


class CorrelationCoefficientIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            return IndicatorResult(name="CorrelationCoefficient", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        x = np.arange(self.period)
        y = data['close'].iloc[-self.period:].to_numpy()
        
        correlation = float(np.corrcoef(x, y)[0, 1])

        if np.isnan(correlation):
            return IndicatorResult(name="CorrelationCoefficient", value=0.0, signal_strength=0.0, interpretation="neutral")

        if correlation > 0.7:
            interpretation = "strong_uptrend"
            strength = abs(correlation) * 100
        elif correlation < -0.7:
            interpretation = "strong_downtrend"
            strength = abs(correlation) * 100
        else:
            interpretation = "weak_trend"
            strength = abs(correlation) * 100

        return IndicatorResult(name="CorrelationCoefficient", value=correlation, signal_strength=float(strength), interpretation=interpretation)


class ElderRayIndexIndicator(TechnicalIndicator):
    def __init__(self, period: int = 13):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        ema = talib.EMA(data['close'].to_numpy(dtype=np.float64), timeperiod=self.period)
        if ema.size == 0 or pd.isna(ema[-1]):
            return IndicatorResult(name="ElderRay", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        bull_power = float(data['high'].iloc[-1]) - float(ema[-1])
        bear_power = float(data['low'].iloc[-1]) - float(ema[-1])

        net_power = bull_power + bear_power

        if bull_power > 0 and bear_power > 0:
            interpretation = "strong_bulls"
            strength = min(abs(bull_power / float(data['close'].iloc[-1])) * 1000, 100.0)
        elif bull_power < 0 and bear_power < 0:
            interpretation = "strong_bears"
            strength = min(abs(bear_power / float(data['close'].iloc[-1])) * 1000, 100.0)
        else:
            interpretation = "mixed"
            strength = 50.0

        return IndicatorResult(name="ElderRay", value=net_power, signal_strength=float(strength), interpretation=interpretation)


