import numpy as np
import pandas as pd
import talib

import pandas_ta as ta
from pandas_ta.overlap import ichimoku, supertrend, kama
from pandas_ta.trend import adx, aroon, psar, dpo

from common.core import IndicatorResult
from common.exceptions import InsufficientDataError, IndicatorError
from features.indicators.base import TechnicalIndicator


class MovingAverageIndicator(TechnicalIndicator):
    def __init__(self, period: int, ma_type: str = "sma"):
        self.period = period
        self.ma_type = ma_type.upper()

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(
                f"MovingAverageIndicator requires at least {self.period} data points."
            )
        try:
            close_np = data["close"].to_numpy(dtype=np.float64)
            if self.ma_type == "EMA":
                ma_series = talib.EMA(close_np, timeperiod=self.period)
            elif self.ma_type == "DEMA":
                ma_series = talib.DEMA(close_np, timeperiod=self.period)
            elif self.ma_type == "TEMA":
                ma_series = talib.TEMA(close_np, timeperiod=self.period)
            elif self.ma_type == "HMA":
                ma_series = ta.hma(data["close"], length=self.period)
            elif self.ma_type == "ZLEMA":
                ma_series = ta.zlma(data["close"], length=self.period)
            elif self.ma_type == "KAMA":
                ma_series = talib.KAMA(close_np, timeperiod=self.period)
            elif self.ma_type == "T3":
                ma_series = talib.T3(close_np, timeperiod=self.period)
            else:
                ma_series = talib.SMA(close_np, timeperiod=self.period)
        except Exception as e:
            raise IndicatorError(f"Error calculating {self.ma_type}: {e}") from e

        if ma_series is None or ma_series.size == 0 or pd.isna(ma_series[-1]):
            raise InsufficientDataError(f"{self.ma_type} calculation resulted in NaN.")

        current_price = data["close"].iloc[-1]
        current_ma = ma_series[-1]

        signal_strength = (
            abs((current_price - current_ma) / current_ma) * 100
            if current_ma != 0
            else 0
        )
        interpretation = (
            "bullish_above_ma" if current_price > current_ma else "bearish_below_ma"
        )

        return IndicatorResult(
            name=f"{self.ma_type}_{self.period}",
            value=float(current_ma),
            signal_strength=float(signal_strength),
            interpretation=interpretation,
        )


class MACDIndicator(TechnicalIndicator):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self.min_period = slow + signal

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.min_period:
            raise InsufficientDataError(
                f"MACD requires at least {self.min_period} data points."
            )
        macd, macdsignal, macdhist = talib.MACD(
            data["close"].to_numpy(dtype=np.float64),
            fastperiod=self.fast,
            slowperiod=self.slow,
            signalperiod=self.signal_period,
        )
        if macd.size == 0 or pd.isna(macd[-1]):
            raise InsufficientDataError("MACD calculation resulted in NaN.")

        current_macd = float(macd[-1])
        current_signal = float(macdsignal[-1])
        current_histogram = float(macdhist[-1])

        if current_macd > current_signal and current_histogram > 0:
            interpretation = "bullish_crossover"
            signal_strength = min(abs(current_histogram) * 1000, 100)
        elif current_macd < current_signal and current_histogram < 0:
            interpretation = "bearish_crossover"
            signal_strength = min(abs(current_histogram) * 1000, 100)
        else:
            interpretation = "neutral"
            signal_strength = 50.0

        return IndicatorResult(
            name="MACD",
            value=current_macd,
            signal_strength=float(signal_strength),
            interpretation=interpretation,
        )


class IchimokuIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 52:
            raise InsufficientDataError("IchimokuIndicator requires at least 52 data points.")
        ichimoku_df = ichimoku(high=data["high"], low=data["low"], close=data["close"])
        if ichimoku_df is None or ichimoku_df[0].empty or ichimoku_df[0].iloc[-1].isna().any():
            raise InsufficientDataError("Ichimoku calculation resulted in NaN.")

        span_a = float(ichimoku_df[0].iloc[-1, 0])
        span_b = float(ichimoku_df[0].iloc[-1, 1])
        current_price = data["close"].iloc[-1]
        is_bullish_cloud = span_a > span_b

        if current_price > span_a and current_price > span_b:
            interpretation = "bullish_above_cloud" if is_bullish_cloud else "bullish_breakout"
            strength = 100.0
        elif current_price < span_a and current_price < span_b:
            interpretation = "bearish_below_cloud" if not is_bullish_cloud else "bearish_breakdown"
            strength = 100.0
        else:
            interpretation = "inside_cloud"
            strength = 20.0

        return IndicatorResult("Ichimoku", (span_a + span_b) / 2, strength, interpretation)


class SuperTrendIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 10:
            raise InsufficientDataError("SuperTrendIndicator requires at least 10 data points.")
        st = supertrend(high=data["high"], low=data["low"], close=data["close"])
        if st is None or st.empty or st.iloc[-1].isna().any():
            raise InsufficientDataError("SuperTrend calculation resulted in NaN.")

        value = float(st.iloc[-1, 0])
        direction = float(st.iloc[-1, 1])
        strength = abs(data["close"].iloc[-1] - value) / value * 100
        interpretation = "uptrend" if direction == 1 else "downtrend"

        return IndicatorResult("SuperTrend", value, min(strength, 100.0), interpretation)


class ADXIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period * 2:
            raise InsufficientDataError(f"ADXIndicator requires at least {self.period * 2} data points.")
        adx_val = adx(high=data["high"], low=data["low"], close=data["close"], length=self.period)
        if adx_val is None or adx_val.empty or adx_val.iloc[-1].isna().any():
            raise InsufficientDataError("ADX calculation resulted in NaN.")

        value = float(adx_val.iloc[-1, 0])
        if value > 25:
            interpretation = "strong_trend"
            strength = (value - 25) * 4
        else:
            interpretation = "weak_trend"
            strength = value * 2

        return IndicatorResult("ADX", value, min(strength, 100.0), interpretation)


class ParabolicSARIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 2:
            raise InsufficientDataError("ParabolicSARIndicator requires at least 2 data points.")
        psar_val = psar(high=data["high"], low=data["low"])
        if psar_val is None or psar_val.empty or psar_val.iloc[-1].isna().any():
            raise InsufficientDataError("PSAR calculation resulted in NaN.")

        value = float(psar_val.iloc[-1, 0])
        current_price = data["close"].iloc[-1]
        strength = abs(current_price - value) / value * 100
        interpretation = "bullish" if current_price > value else "bearish"

        return IndicatorResult("PSAR", value, min(strength, 100.0), interpretation)


class AroonIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"AroonIndicator requires at least {self.period} data points.")
        aroon_df = aroon(close=data["close"], length=self.period)
        if aroon_df is None or aroon_df.empty or aroon_df.iloc[-1].isna().any():
            raise InsufficientDataError("Aroon calculation resulted in NaN.")

        up = float(aroon_df.iloc[-1, 0])
        down = float(aroon_df.iloc[-1, 1])
        oscillator = up - down
        strength = abs(oscillator)
        interpretation = "bullish" if oscillator > 0 else "bearish"

        return IndicatorResult("Aroon", oscillator, strength, interpretation)


class DetrendedPriceOscillatorIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period + 10:
            raise InsufficientDataError(f"DPOIndicator requires at least {self.period + 10} data points.")
        dpo_val = dpo(close=data["close"], length=self.period)
        if dpo_val is None or dpo_val.empty or pd.isna(dpo_val.iloc[-1]):
            raise InsufficientDataError("DPO calculation resulted in NaN.")

        value = float(dpo_val.iloc[-1])
        strength = abs(value / data["close"].iloc[-1]) * 1000 if data["close"].iloc[-1] > 0 else 0
        interpretation = "above_zero" if value > 0 else "below_zero"

        return IndicatorResult("DPO", value, min(strength, 100.0), interpretation)


class LinearRegressionIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"LinRegIndicator requires at least {self.period} data points.")
        linreg = talib.LINEARREG(data["close"].to_numpy(dtype=np.float64), timeperiod=self.period)
        if linreg.size == 0 or pd.isna(linreg[-1]):
            raise InsufficientDataError("LinReg calculation resulted in NaN.")

        value = float(linreg[-1])
        current_price = data["close"].iloc[-1]
        strength = abs(current_price - value) / value * 100 if value > 0 else 0
        interpretation = "above_line" if current_price > value else "below_line"

        return IndicatorResult("LinReg", value, min(strength, 100.0), interpretation)


class LinearRegressionSlopeIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"LinRegSlope requires at least {self.period} data points.")
        slope = talib.LINEARREG_SLOPE(data["close"].to_numpy(dtype=np.float64), timeperiod=self.period)
        if slope.size == 0 or pd.isna(slope[-1]):
            raise InsufficientDataError("LinRegSlope calculation resulted in NaN.")

        value = float(slope[-1])
        strength = abs(value) * 100
        interpretation = "positive_slope" if value > 0 else "negative_slope"

        return IndicatorResult("LinRegSlope", value, min(strength, 100.0), interpretation)


class HullMovingAverageIndicator(TechnicalIndicator):
    def __init__(self, period: int = 9):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        return MovingAverageIndicator(period=self.period, ma_type="HMA").calculate(data)


class ZLEMAIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        return MovingAverageIndicator(period=self.period, ma_type="ZLEMA").calculate(data)


class KAMAIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        return MovingAverageIndicator(period=self.period, ma_type="KAMA").calculate(data)


class T3Indicator(TechnicalIndicator):
    def __init__(self, period: int = 5):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        return MovingAverageIndicator(period=self.period, ma_type="T3").calculate(data)


class DEMAIndicator(TechnicalIndicator):
    def __init__(self, period: int = 9):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        return MovingAverageIndicator(period=self.period, ma_type="DEMA").calculate(data)


class TEMAIndicator(TechnicalIndicator):
    def __init__(self, period: int = 9):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        return MovingAverageIndicator(period=self.period, ma_type="TEMA").calculate(data)


class GannHiLoActivatorIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 13:
            raise InsufficientDataError("GannHiLo requires at least 13 data points.")
        gann = ta.hilo(high=data["high"], low=data["low"], close=data["close"])
        if gann is None or gann.empty or gann.iloc[-1].isna().any():
            raise InsufficientDataError("Gann HiLo calculation resulted in NaN.")

        value = float(gann.iloc[-1, 0])
        current_price = data["close"].iloc[-1]
        strength = abs(current_price - value) / value * 100 if value > 0 else 0
        interpretation = "above_activator" if current_price > value else "below_activator"

        return IndicatorResult("GannHiLo", value, min(strength, 100.0), interpretation)


class MovingAverageRibbonIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 60:
            raise InsufficientDataError("MARibbon requires at least 60 data points.")
        emas = [talib.EMA(data["close"], timeperiod=p) for p in [10, 20, 30, 40, 50, 60]]
        if any(e is None or pd.isna(e[-1]) for e in emas):
            raise InsufficientDataError("MA Ribbon EMA calculation failed.")

        is_bullish = all(emas[i][-1] > emas[i + 1][-1] for i in range(len(emas) - 1))
        is_bearish = all(emas[i][-1] < emas[i + 1][-1] for i in range(len(emas) - 1))

        if is_bullish:
            value = 1.0
            interpretation = "bullish_expansion"
            strength = 100.0
        elif is_bearish:
            value = -1.0
            interpretation = "bearish_expansion"
            strength = 100.0
        else:
            value = 0.0
            interpretation = "ribbon_crossed"
            strength = 50.0

        return IndicatorResult("MARibbon", value, strength, interpretation)


class FractalIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 5:
            raise InsufficientDataError("Fractal requires at least 5 data points.")
        highs = data["high"]
        lows = data["low"]
        is_bull_fractal = (lows.iloc[-3] < lows.iloc[-5]) and (lows.iloc[-3] < lows.iloc[-4]) and (lows.iloc[-3] < lows.iloc[-2]) and (lows.iloc[-3] < lows.iloc[-1])
        is_bear_fractal = (highs.iloc[-3] > highs.iloc[-5]) and (highs.iloc[-3] > highs.iloc[-4]) and (highs.iloc[-3] > highs.iloc[-2]) and (highs.iloc[-3] > highs.iloc[-1])

        if is_bear_fractal:
            return IndicatorResult("Fractal", highs.iloc[-3], 100.0, "bearish_fractal")
        if is_bull_fractal:
            return IndicatorResult("Fractal", lows.iloc[-3], 100.0, "bullish_fractal")

        return IndicatorResult("Fractal", 0.0, 0.0, "no_fractal")


class FRAMAIndicator(TechnicalIndicator):
    def __init__(self, period: int = 16):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        return MovingAverageIndicator(period=self.period, ma_type="FRAMA").calculate(data)


class VIDYAIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        return MovingAverageIndicator(period=self.period, ma_type="VIDYA").calculate(data)


class MAMAIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 32:
            raise InsufficientDataError("MAMA requires at least 32 data points.")
        mama, fama = talib.MAMA(data["close"].to_numpy(dtype=np.float64))
        if mama.size == 0 or pd.isna(mama[-1]):
            raise InsufficientDataError("MAMA calculation failed.")

        value = float(mama[-1])
        signal = float(fama[-1])
        strength = abs(value - signal) / value * 100 if value > 0 else 0
        interpretation = "bullish_crossover" if value > signal else "bearish_crossover"

        return IndicatorResult("MAMA", value, min(strength, 100.0), interpretation)


class KaufmanEfficiencyRatioIndicator(TechnicalIndicator):
    def __init__(self, period: int = 10):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period + 1:
            raise InsufficientDataError(f"ER requires at least {self.period + 1} data points.")
        er = ta.er(close=data["close"], length=self.period)
        if er is None or er.empty or pd.isna(er.iloc[-1]):
            raise InsufficientDataError("Kaufman ER calculation failed.")

        value = float(er.iloc[-1])
        strength = value * 100
        interpretation = "trending" if value > 0.7 else "ranging" if value < 0.3 else "neutral"

        return IndicatorResult("ER", value, strength, interpretation)


class TrendIntensityIndexIndicator(TechnicalIndicator):
    def __init__(self, period: int = 30):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period * 2:
            raise InsufficientDataError(f"TII requires at least {self.period * 2} data points.")
        tii = ta.tii(close=data["close"], length=self.period)
        if tii is None or tii.empty or pd.isna(tii.iloc[-1]):
            raise InsufficientDataError("TII calculation failed.")

        value = float(tii.iloc[-1])
        if value > 80:
            interpretation = "strong_uptrend"
            strength = 100.0
        elif value < 20:
            interpretation = "strong_downtrend"
            strength = 100.0
        else:
            interpretation = "no_trend"
            strength = 0.0

        return IndicatorResult("TII", value, strength, interpretation)