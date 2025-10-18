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

from common.core import IndicatorResult
from common.exceptions import InsufficientDataError
from features.indicators.base import TechnicalIndicator


class PivotPointsIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 2:
            raise InsufficientDataError(
                "PivotPointsIndicator requires at least 2 data points."
            )

        prev_high = float(data["high"].iloc[-2])
        prev_low = float(data["low"].iloc[-2])
        prev_close = float(data["close"].iloc[-2])
        current_price = float(data["close"].iloc[-1])

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

        return IndicatorResult(
            name="PivotPoints",
            value=pivot,
            signal_strength=strength,
            interpretation=interpretation,
        )


class MedianPriceIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 1:
            raise InsufficientDataError(
                "MedianPriceIndicator requires at least 1 data point."
            )
        median_price = talib.MEDPRICE(
            data["high"].to_numpy(dtype=np.float64),
            data["low"].to_numpy(dtype=np.float64),
        )
        if median_price.size == 0 or pd.isna(median_price[-1]):
            raise InsufficientDataError("MedianPrice calculation resulted in NaN.")

        current_median = float(median_price[-1])
        current_close = float(data["close"].iloc[-1])

        if current_close > current_median:
            interpretation = "above_median"
            strength = (
                abs(current_close - current_median) / current_median * 100
                if current_median > 0
                else 0.0
            )
        elif current_close < current_median:
            interpretation = "below_median"
            strength = (
                abs(current_close - current_median) / current_median * 100
                if current_median > 0
                else 0.0
            )
        else:
            interpretation = "at_median"
            strength = 50.0

        return IndicatorResult(
            name="MedianPrice",
            value=current_median,
            signal_strength=min(float(strength), 100.0),
            interpretation=interpretation,
        )


class TypicalPriceIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 1:
            raise InsufficientDataError(
                "TypicalPriceIndicator requires at least 1 data point."
            )
        typical_price = talib.TYPPRICE(
            data["high"].to_numpy(dtype=np.float64),
            data["low"].to_numpy(dtype=np.float64),
            data["close"].to_numpy(dtype=np.float64),
        )
        if typical_price.size == 0 or pd.isna(typical_price[-1]):
            raise InsufficientDataError("TypicalPrice calculation resulted in NaN.")

        current_typical = float(typical_price[-1])
        current_close = float(data["close"].iloc[-1])

        strength = (
            abs(current_close - current_typical) / current_typical * 100
            if current_typical > 0
            else 0.0
        )
        interpretation = (
            "above_typical" if current_close > current_typical else "below_typical"
        )

        return IndicatorResult(
            name="TypicalPrice",
            value=current_typical,
            signal_strength=min(float(strength), 100.0),
            interpretation=interpretation,
        )


class WeightedClosePriceIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 1:
            raise InsufficientDataError(
                "WeightedClosePriceIndicator requires at least 1 data point."
            )
        wcl_price = talib.WCLPRICE(
            data["high"].to_numpy(dtype=np.float64),
            data["low"].to_numpy(dtype=np.float64),
            data["close"].to_numpy(dtype=np.float64),
        )
        if wcl_price.size == 0 or pd.isna(wcl_price[-1]):
            raise InsufficientDataError("WCLPRICE calculation resulted in NaN.")

        current_wcl = float(wcl_price[-1])
        current_close = float(data["close"].iloc[-1])

        strength = (
            abs(current_close - current_wcl) / current_wcl * 100
            if current_wcl > 0
            else 0.0
        )
        interpretation = (
            "above_weighted" if current_close > current_wcl else "below_weighted"
        )

        return IndicatorResult(
            name="WeightedClosePrice",
            value=current_wcl,
            signal_strength=min(float(strength), 100.0),
            interpretation=interpretation,
        )


class PriceActionPatternIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 3:
            raise InsufficientDataError("PriceActionPatternIndicator requires at least 3 data points.")

        last = data.iloc[-1]
        prev = data.iloc[-2]

        is_bullish_engulfing = last['close'] > prev['open'] and last['open'] < prev['close'] and last['close'] > prev['high'] and last['open'] < prev['low']
        is_bearish_engulfing = last['open'] > prev['close'] and last['close'] < prev['open'] and last['open'] > prev['high'] and last['close'] < prev['low']

        if is_bullish_engulfing:
            return IndicatorResult("PriceAction", 1.0, 100.0, "bullish_engulfing")
        if is_bearish_engulfing:
            return IndicatorResult("PriceAction", -1.0, 100.0, "bearish_engulfing")

        return IndicatorResult("PriceAction", 0.0, 0.0, "no_pattern")


class MarketStructureIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 20:
            raise InsufficientDataError("MarketStructureIndicator requires at least 20 data points.")

        recent_high = data['high'].rolling(14).max()
        recent_low = data['low'].rolling(14).min()

        is_higher_high = data['high'].iloc[-1] > recent_high.iloc[-2]
        is_higher_low = data['low'].iloc[-1] > recent_low.iloc[-2]
        is_lower_high = data['high'].iloc[-1] < recent_high.iloc[-2]
        is_lower_low = data['low'].iloc[-1] < recent_low.iloc[-2]

        if is_higher_high and is_higher_low:
            return IndicatorResult("MarketStructure", 1.0, 100.0, "higher_high_higher_low")
        if is_lower_low and is_lower_high:
            return IndicatorResult("MarketStructure", -1.0, 100.0, "lower_low_lower_high")

        return IndicatorResult("MarketStructure", 0.0, 50.0, "consolidation")


class LiquidityLevelsIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 50:
            raise InsufficientDataError("LiquidityLevelsIndicator requires at least 50 data points.")

        from scipy.signal import find_peaks
        highs = data['high']
        lows = data['low']

        peak_indices, _ = find_peaks(highs, prominence=data['close'].mean() * 0.05)
        trough_indices, _ = find_peaks(-lows, prominence=data['close'].mean() * 0.05)

        if len(peak_indices) > 0:
            resistance_level = highs.iloc[peak_indices].iloc[-1]
            return IndicatorResult("LiquidityLevels", resistance_level, 80.0, "resistance_identified")

        if len(trough_indices) > 0:
            support_level = lows.iloc[trough_indices].iloc[-1]
            return IndicatorResult("LiquidityLevels", support_level, 80.0, "support_identified")

        return IndicatorResult("LiquidityLevels", 0.0, 0.0, "no_clear_levels")


class WyckoffVolumeSpreadIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 20:
            raise InsufficientDataError("WyckoffVSA requires at least 20 data points.")

        vol_ma = data['volume'].rolling(20).mean()
        last_vol = data['volume'].iloc[-1]
        last_range = data['high'].iloc[-1] - data['low'].iloc[-1]

        is_high_volume = last_vol > vol_ma.iloc[-1] * 1.5
        is_low_volume = last_vol < vol_ma.iloc[-1] * 0.5
        is_wide_spread = last_range > data['close'].rolling(20).std().iloc[-1] * 2

        if is_high_volume and is_wide_spread and data['close'].iloc[-1] > data['open'].iloc[-1]:
            return IndicatorResult("WyckoffVSA", 1.0, 90.0, "demand_wave")
        if is_high_volume and is_wide_spread and data['close'].iloc[-1] < data['open'].iloc[-1]:
            return IndicatorResult("WyckoffVSA", -1.0, 90.0, "supply_wave")

        return IndicatorResult("WyckoffVSA", 0.0, 50.0, "neutral_vsa")