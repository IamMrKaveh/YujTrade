import numpy as np
import pandas as pd
import talib

import pandas_ta as ta
from pandas_ta.volatility import bbands, massi, kc, donchian

from common.core import IndicatorResult
from common.exceptions import InsufficientDataError
from features.indicators.base import TechnicalIndicator


class BollingerBandsIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(
                f"BollingerBandsIndicator requires at least {self.period} data points."
            )
        bb_df = bbands(close=data["close"], length=self.period, std=self.std_dev)
        if bb_df is None or bb_df.empty or bb_df.dropna().empty:
            raise InsufficientDataError(
                "Bollinger Bands calculation resulted in empty DataFrame."
            )

        upper_band = float(bb_df.iloc[-1, 2])
        lower_band = float(bb_df.iloc[-1, 0])
        current_price = float(data["close"].iloc[-1])

        band_width = upper_band - lower_band
        if band_width == 0:
            return IndicatorResult(
                name="BB",
                value=0.5,
                signal_strength=0.0,
                interpretation="neutral_flat_bands",
            )

        bb_position = (current_price - lower_band) / band_width

        if bb_position > 0.8:
            interpretation = "near_upper_band"
            strength = (bb_position - 0.8) / 0.2 * 100
        elif bb_position < 0.2:
            interpretation = "near_lower_band"
            strength = (0.2 - bb_position) / 0.2 * 100
        else:
            interpretation = "middle_range"
            strength = 50.0

        return IndicatorResult(
            name="BB",
            value=float(bb_position),
            signal_strength=float(strength),
            interpretation=interpretation,
        )


class ATRIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(
                f"ATRIndicator requires at least {self.period} data points."
            )
        atr_series = talib.ATR(
            data["high"].to_numpy(dtype=np.float64),
            data["low"].to_numpy(dtype=np.float64),
            data["close"].to_numpy(dtype=np.float64),
            timeperiod=self.period,
        )
        if atr_series.size == 0 or pd.isna(atr_series[-1]):
            raise InsufficientDataError("ATR calculation resulted in NaN.")

        current_atr = float(atr_series[-1])
        current_price = float(data["close"].iloc[-1])

        atr_percentage = (
            (current_atr / current_price) * 100 if current_price > 0 else 0.0
        )

        if atr_percentage > 3:
            interpretation = "high_volatility"
            signal_strength = min(atr_percentage * 20, 100)
        elif atr_percentage < 1:
            interpretation = "low_volatility"
            signal_strength = max(0, (1 - atr_percentage) * 100)
        else:
            interpretation = "normal_volatility"
            signal_strength = 50.0

        return IndicatorResult(
            name="ATR",
            value=current_atr,
            signal_strength=float(signal_strength),
            interpretation=interpretation,
        )


class KeltnerChannelsIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"KeltnerChannelsIndicator requires at least {self.period} data points.")
        kc_df = kc(high=data["high"], low=data["low"], close=data["close"], length=self.period)
        if kc_df is None or kc_df.empty or kc_df.iloc[-1].isna().any():
            raise InsufficientDataError("Keltner Channels calculation resulted in NaN.")

        upper = float(kc_df.iloc[-1, 1])
        lower = float(kc_df.iloc[-1, 0])
        current_price = data["close"].iloc[-1]

        if current_price > upper:
            interpretation = "breakout_above"
            strength = 100.0
        elif current_price < lower:
            interpretation = "breakdown_below"
            strength = 100.0
        else:
            interpretation = "in_channel"
            strength = 20.0

        return IndicatorResult("Keltner", (upper + lower) / 2, strength, interpretation)


class DonchianChannelsIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"DonchianChannelsIndicator requires at least {self.period} data points.")
        dc_df = donchian(high=data["high"], low=data["low"], lower_length=self.period, upper_length=self.period)
        if dc_df is None or dc_df.empty or dc_df.iloc[-1].isna().any():
            raise InsufficientDataError("Donchian Channels calculation resulted in NaN.")

        upper = float(dc_df.iloc[-1, 1])
        lower = float(dc_df.iloc[-1, 0])
        current_price = data["close"].iloc[-1]

        if current_price >= upper:
            interpretation = "at_upper_band"
            strength = 100.0
        elif current_price <= lower:
            interpretation = "at_lower_band"
            strength = 100.0
        else:
            interpretation = "in_channel"
            strength = 50.0

        return IndicatorResult("Donchian", (upper + lower) / 2, strength, interpretation)


class StandardDeviationIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"StdDevIndicator requires at least {self.period} data points.")
        std_dev = talib.STDDEV(data["close"].to_numpy(dtype=np.float64), timeperiod=self.period)
        if std_dev.size == 0 or pd.isna(std_dev[-1]):
            raise InsufficientDataError("StdDev calculation resulted in NaN.")

        value = float(std_dev[-1])
        strength = value / data["close"].mean() * 100 if data["close"].mean() > 0 else 0
        interpretation = "high_volatility" if strength > 5 else "low_volatility"

        return IndicatorResult("StdDev", value, min(strength*10, 100.0), interpretation)


class MassIndexIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 25:
            raise InsufficientDataError("MassIndexIndicator requires at least 25 data points.")
        mass = massi(high=data["high"], low=data["low"])
        if mass is None or mass.empty or pd.isna(mass.iloc[-1]):
            raise InsufficientDataError("Mass Index calculation resulted in NaN.")

        value = float(mass.iloc[-1])
        if value > 27:
            interpretation = "reversal_potential"
            strength = (value - 27) * 20
        else:
            interpretation = "no_reversal_signal"
            strength = 30.0

        return IndicatorResult("MassIndex", value, min(strength, 100.0), interpretation)


class CorrelationCoefficientIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 14:
            raise InsufficientDataError("CorrCoefIndicator requires at least 14 data points.")
        corr = talib.CORREL(data["high"].to_numpy(dtype=np.float64), data["low"].to_numpy(dtype=np.float64), timeperiod=14)
        if corr.size == 0 or pd.isna(corr[-1]):
            raise InsufficientDataError("Correlation Coefficient calculation resulted in NaN.")

        value = float(corr[-1])
        strength = abs(value) * 100
        interpretation = "positive_correlation" if value > 0.5 else "negative_correlation" if value < -0.5 else "low_correlation"

        return IndicatorResult("CorrCoef", value, strength, interpretation)


class ElderRayIndexIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 13:
            raise InsufficientDataError("ElderRayIndexIndicator requires at least 13 data points.")
        ema = talib.EMA(data["close"].to_numpy(dtype=np.float64), timeperiod=13)
        bull_power = data["high"] - pd.Series(ema, index=data.index)
        bear_power = data["low"] - pd.Series(ema, index=data.index)
        if bull_power.isna().iloc[-1] or bear_power.isna().iloc[-1]:
            raise InsufficientDataError("Elder Ray calculation resulted in NaN.")

        bull_p = float(bull_power.iloc[-1])
        bear_p = float(bear_power.iloc[-1])

        if bull_p > 0 and bear_p > 0:
            interpretation = "bull_power_dominant"
            strength = 80.0
        elif bull_p < 0 and bear_p < 0:
            interpretation = "bear_power_dominant"
            strength = 80.0
        else:
            interpretation = "conflicted"
            strength = 30.0

        return IndicatorResult("ElderRay", bull_p - bear_p, strength, interpretation)


class ChoppinessIndexIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"ChoppinessIndexIndicator requires at least {self.period} data points.")
        chop = ta.chop(high=data["high"], low=data["low"], close=data["close"], length=self.period)
        if chop is None or chop.empty or pd.isna(chop.iloc[-1]):
            raise InsufficientDataError("Choppiness Index calculation resulted in NaN.")

        value = float(chop.iloc[-1])
        if value > 61.8:
            interpretation = "choppy_market"
            strength = value
        elif value < 38.2:
            interpretation = "trending_market"
            strength = 100 - value
        else:
            interpretation = "neutral_market"
            strength = 50.0

        return IndicatorResult("ChoppinessIndex", value, strength, interpretation)


class ChaikinVolatilityIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 20:
            raise InsufficientDataError("ChaikinVolatilityIndicator requires at least 20 data points.")
        chakin_vol = ta.vhf(high=data["high"], low=data["low"], close=data["close"]) # Using VHF as a proxy
        if chakin_vol is None or chakin_vol.empty or pd.isna(chakin_vol.iloc[-1]):
            raise InsufficientDataError("Chaikin Volatility calculation resulted in NaN.")

        value = float(chakin_vol.iloc[-1])
        strength = value * 100
        interpretation = "high_volatility" if value > 0.5 else "low_volatility"

        return IndicatorResult("ChaikinVolatility", value, strength, interpretation)


class HistoricalVolatilityIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"HistoricalVolatilityIndicator requires at least {self.period} data points.")
        log_returns = np.log(data['close'] / data['close'].shift(1))
        volatility = log_returns.rolling(window=self.period).std() * np.sqrt(self.period)
        if volatility.isna().iloc[-1]:
            raise InsufficientDataError("Historical Volatility calculation resulted in NaN.")

        value = float(volatility.iloc[-1])
        strength = value * 100
        interpretation = "high_volatility" if value > 0.05 else "low_volatility"

        return IndicatorResult("HistoricalVolatility", value, min(strength, 100.0), interpretation)


class UlcerIndexIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"UlcerIndexIndicator requires at least {self.period} data points.")
        ulcer = ta.ui(close=data["close"], length=self.period)
        if ulcer is None or ulcer.empty or pd.isna(ulcer.iloc[-1]):
            raise InsufficientDataError("Ulcer Index calculation resulted in NaN.")

        value = float(ulcer.iloc[-1])
        strength = value * 10
        interpretation = "high_drawdown_risk" if value > 5 else "low_drawdown_risk"

        return IndicatorResult("UlcerIndex", value, min(strength, 100.0), interpretation)


class ATRBandsIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14, multiplier: float = 2.0):
        self.period = period
        self.multiplier = multiplier

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"ATRBandsIndicator requires at least {self.period} data points.")
        atr = talib.ATR(data["high"], data["low"], data["close"], timeperiod=self.period)
        if atr.size == 0 or pd.isna(atr[-1]):
            raise InsufficientDataError("ATR Bands calculation failed on ATR.")

        sma = talib.SMA(data["close"], timeperiod=self.period)
        if sma.size == 0 or pd.isna(sma[-1]):
            raise InsufficientDataError("ATR Bands calculation failed on SMA.")

        upper_band = sma[-1] + (atr[-1] * self.multiplier)
        lower_band = sma[-1] - (atr[-1] * self.multiplier)
        current_price = data["close"].iloc[-1]

        if current_price > upper_band:
            interpretation = "above_upper_band"
            strength = 100.0
        elif current_price < lower_band:
            interpretation = "below_lower_band"
            strength = 100.0
        else:
            interpretation = "within_bands"
            strength = 50.0

        return IndicatorResult("ATRBands", sma[-1], strength, interpretation)


class BollingerBandwidthIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"BollingerBandwidthIndicator requires at least {self.period} data points.")
        bbw = ta.bbands(close=data["close"], length=self.period)
        if bbw is None or bbw.empty or bbw.iloc[-1].isna().any():
            raise InsufficientDataError("BBW calculation resulted in NaN.")

        bandwidth = float(bbw.iloc[-1, 3])
        value = bandwidth
        strength = bandwidth * 100
        interpretation = "expanding_volatility" if bandwidth > bbw.iloc[-2, 3] else "contracting_volatility"

        return IndicatorResult("BBW", value, min(strength, 100.0), interpretation)