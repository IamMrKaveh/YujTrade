import numpy as np
import pandas as pd
import talib

import pandas_ta as ta

from common.core import IndicatorResult
from common.exceptions import InsufficientDataError
from features.indicators.base import TechnicalIndicator


class VortexIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(
                f"VortexIndicator requires at least {self.period} data points."
            )
        vortex_df = ta.vortex(
            high=data["high"], low=data["low"], close=data["close"], length=self.period
        )
        if vortex_df is None or vortex_df.empty or vortex_df.iloc[-1].isna().any():
            raise InsufficientDataError("Vortex calculation resulted in NaN.")

        vi_plus = float(vortex_df.iloc[-1, 0])
        vi_minus = float(vortex_df.iloc[-1, 1])

        if vi_plus > vi_minus and vi_plus > 1:
            interpretation = "bullish_trend"
            strength = min((vi_plus - 1) * 100, 100.0)
        elif vi_minus > vi_plus and vi_minus > 1:
            interpretation = "bearish_trend"
            strength = min((vi_minus - 1) * 100, 100.0)
        else:
            interpretation = "no_trend"
            strength = 50.0

        return IndicatorResult(
            name="Vortex",
            value=vi_plus - vi_minus,
            signal_strength=float(strength),
            interpretation=interpretation,
        )


class KSTIndicator(TechnicalIndicator):
    def __init__(
        self,
        roc1: int = 10,
        roc2: int = 15,
        roc3: int = 20,
        roc4: int = 30,
        sma1: int = 10,
        sma2: int = 10,
        sma3: int = 10,
        sma4: int = 15,
        signal: int = 9,
    ):
        self.roc1 = roc1
        self.roc2 = roc2
        self.roc3 = roc3
        self.roc4 = roc4
        self.sma1 = sma1
        self.sma2 = sma2
        self.sma3 = sma3
        self.sma4 = sma4
        self.signal = signal
        self.min_period = max(roc1, roc2, roc3, roc4) + max(sma1, sma2, sma3, sma4)

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.min_period:
            raise InsufficientDataError(
                f"KSTIndicator requires at least {self.min_period} data points."
            )
        kst_df = ta.kst(
            close=data["close"],
            roc1=self.roc1,
            roc2=self.roc2,
            roc3=self.roc3,
            roc4=self.roc4,
            sma1=self.sma1,
            sma2=self.sma2,
            sma3=self.sma3,
            sma4=self.sma4,
            signal=self.signal,
        )
        if kst_df is None or kst_df.empty or kst_df.iloc[-1].isna().any():
            raise InsufficientDataError("KST calculation resulted in NaN.")

        kst_value = float(kst_df.iloc[-1, 0])
        kst_signal = float(kst_df.iloc[-1, 1])

        if kst_value > kst_signal and kst_value > 0:
            interpretation = "bullish"
            strength = min(abs(kst_value), 100.0)
        elif kst_value < kst_signal and kst_value < 0:
            interpretation = "bearish"
            strength = min(abs(kst_value), 100.0)
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult(
            name="KST",
            value=kst_value,
            signal_strength=float(strength),
            interpretation=interpretation,
        )


class HilbertDominantCycleIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 32:
            raise InsufficientDataError("HilbertDominantCycleIndicator requires at least 32 data points.")

        ht_dc_period = talib.HT_DCPERIOD(data["close"].to_numpy(dtype=np.float64))
        if ht_dc_period.size == 0 or pd.isna(ht_dc_period[-1]):
            raise InsufficientDataError("HT_DCPERIOD calculation resulted in NaN.")

        value = float(ht_dc_period[-1])
        strength = 100 - min(value, 100)
        interpretation = f"dominant_cycle_{int(value)}_periods"

        return IndicatorResult(
            name="HT_DC", value=value, signal_strength=strength, interpretation=interpretation
        )


class HilbertTrendVsCycleModeIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 63:
            raise InsufficientDataError("HilbertTrendVsCycleModeIndicator requires at least 63 data points.")

        ht_trendmode = talib.HT_TRENDMODE(data["close"].to_numpy(dtype=np.float64))
        if ht_trendmode.size == 0 or pd.isna(ht_trendmode[-1]):
            raise InsufficientDataError("HT_TRENDMODE calculation resulted in NaN.")

        value = int(ht_trendmode[-1])
        interpretation = "trend_mode" if value == 1 else "cycle_mode"
        strength = 100.0 if value == 1 else 50.0

        return IndicatorResult(
            name="HT_TREND_MODE", value=float(value), signal_strength=strength, interpretation=interpretation
        )