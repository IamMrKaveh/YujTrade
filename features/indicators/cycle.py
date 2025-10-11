import numpy as np
import pandas as pd
import talib

import pandas_ta as ta

from ...common.core import IndicatorResult
from .base import TechnicalIndicator

class VortexIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        vortex_df = ta.vortex(high=data['high'], low=data['low'], close=data['close'], length=self.period)
        if vortex_df is None or vortex_df.empty or vortex_df.iloc[-1].isna().any():
            return IndicatorResult(name="Vortex", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

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

        return IndicatorResult(name="Vortex", value=vi_plus - vi_minus, signal_strength=float(strength), interpretation=interpretation)


class KSTIndicator(TechnicalIndicator):
    def __init__(self, roc1: int = 10, roc2: int = 15, roc3: int = 20, roc4: int = 30, 
                sma1: int = 10, sma2: int = 10, sma3: int = 10, sma4: int = 15, signal: int = 9):
        self.roc1 = roc1
        self.roc2 = roc2
        self.roc3 = roc3
        self.roc4 = roc4
        self.sma1 = sma1
        self.sma2 = sma2
        self.sma3 = sma3
        self.sma4 = sma4
        self.signal = signal

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        kst_df = ta.kst(
            close=data['close'], 
            roc1=self.roc1, roc2=self.roc2, roc3=self.roc3, roc4=self.roc4, 
            sma1=self.sma1, sma2=self.sma2, sma3=self.sma3, sma4=self.sma4, 
            signal=self.signal
        )
        if kst_df is None or kst_df.empty or kst_df.iloc[-1].isna().any():
            return IndicatorResult(name="KST", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

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

        return IndicatorResult(name="KST", value=kst_value, signal_strength=float(strength), interpretation=interpretation)


class HilbertDominantCycleIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        dc_period = talib.HT_DCPERIOD(data['close'].to_numpy(dtype=np.float64))
        if dc_period.size == 0 or pd.isna(dc_period[-1]):
            return IndicatorResult(name="HT_DominantCycle", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_period = float(dc_period[-1])

        if current_period < 20:
            interpretation = "short_cycle"
            strength = min((20 - current_period) / 20 * 100, 100.0)
        elif current_period > 40:
            interpretation = "long_cycle"
            strength = min((current_period - 40) / 40 * 100, 100.0)
        else:
            interpretation = "normal_cycle"
            strength = 50.0

        return IndicatorResult(name="HT_DominantCycle", value=current_period, signal_strength=float(strength), interpretation=interpretation)


class HilbertTrendVsCycleModeIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        trend_mode = talib.HT_TRENDMODE(data['close'].to_numpy(dtype=np.float64))
        if trend_mode.size == 0 or pd.isna(trend_mode[-1]):
            return IndicatorResult(name="HT_TrendMode", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_mode = float(trend_mode[-1])

        if current_mode == 1:
            interpretation = "trending_market"
            strength = 100.0
        else:
            interpretation = "cycling_market"
            strength = 100.0

        return IndicatorResult(name="HT_TrendMode", value=current_mode, signal_strength=strength, interpretation=interpretation)


class KaufmanEfficiencyRatioIndicator(TechnicalIndicator):
    def __init__(self, period: int = 10):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            return IndicatorResult(name="ER", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        close_np = data['close'].to_numpy(dtype=np.float64)
        change = abs(close_np[-1] - close_np[-self.period])
        volatility = np.sum(np.abs(np.diff(close_np[-self.period:])))

        if volatility == 0:
            er = 0.0
        else:
            er = change / volatility

        if er > 0.7:
            interpretation = "strong_trend"
            strength = er * 100
        elif er < 0.3:
            interpretation = "weak_trend"
            strength = (1 - er) * 100
        else:
            interpretation = "moderate_trend"
            strength = 50.0

        return IndicatorResult(name="ER", value=float(er), signal_strength=float(strength), interpretation=interpretation)


