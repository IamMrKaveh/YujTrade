import numpy as np
import pandas as pd
import talib

import pandas_ta as ta
from pandas_ta.volume import cmf, obv, ad, eom, efi, pvt, kvo, pvo, nvi, pvi

from common.core import IndicatorResult
from config.logger import logger
from common.exceptions import InsufficientDataError
from features.indicators.base import TechnicalIndicator


class VolumeIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if "volume" not in data.columns or len(data) < self.period:
            raise InsufficientDataError(
                f"VolumeIndicator requires 'volume' column and at least {self.period} data points."
            )

        volume_ma = talib.SMA(
            data["volume"].to_numpy(dtype=np.float64), timeperiod=self.period
        )
        if volume_ma.size == 0 or pd.isna(volume_ma[-1]):
            raise InsufficientDataError("Volume MA calculation resulted in NaN.")

        current_volume = float(data["volume"].iloc[-1])
        average_volume = float(volume_ma[-1])

        volume_ratio = current_volume / average_volume if average_volume > 0 else 1.0

        if volume_ratio > 1.5:
            interpretation = "high_volume"
            signal_strength = min((volume_ratio - 1) * 50, 100)
        elif volume_ratio < 0.5:
            interpretation = "low_volume"
            signal_strength = max(0, (1 - volume_ratio) * 100)
        else:
            interpretation = "normal_volume"
            signal_strength = 50.0

        return IndicatorResult(
            name="VOLUME",
            value=float(volume_ratio),
            signal_strength=float(signal_strength),
            interpretation=interpretation,
        )


class ChaikinMoneyFlowIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(
                f"CMF requires at least {self.period} data points."
            )
        cmf_series = cmf(
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"],
            length=self.period,
        )
        if cmf_series is None or cmf_series.empty or pd.isna(cmf_series.iloc[-1]):
            raise InsufficientDataError("CMF calculation resulted in NaN.")

        current_value = float(cmf_series.iloc[-1])

        if current_value > 0:
            interpretation = "buy_pressure"
            strength = min(abs(current_value) * 200, 100)
        elif current_value < 0:
            interpretation = "sell_pressure"
            strength = min(abs(current_value) * 200, 100)
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult(
            name="CMF",
            value=current_value,
            signal_strength=float(strength),
            interpretation=interpretation,
        )


class OBVIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 2:
            raise InsufficientDataError("OBVIndicator requires at least 2 data points.")
        obv_val = obv(close=data["close"], volume=data["volume"])
        if obv_val is None or obv_val.empty or pd.isna(obv_val.iloc[-1]):
            raise InsufficientDataError("OBV calculation resulted in NaN.")

        value = float(obv_val.iloc[-1])
        strength = 50.0
        interpretation = "bullish" if value > obv_val.iloc[-2] else "bearish"

        return IndicatorResult("OBV", value, strength, interpretation)


class VWAPIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 1:
            raise InsufficientDataError("VWAPIndicator requires at least 1 data point.")
        vwap = ta.vwap(high=data["high"], low=data["low"], close=data["close"], volume=data["volume"])
        if vwap is None or vwap.empty or pd.isna(vwap.iloc[-1]):
            raise InsufficientDataError("VWAP calculation resulted in NaN.")

        value = float(vwap.iloc[-1])
        current_price = data["close"].iloc[-1]
        strength = abs(current_price - value) / value * 100 if value > 0 else 0
        interpretation = "above_vwap" if current_price > value else "below_vwap"

        return IndicatorResult("VWAP", value, min(strength, 100.0), interpretation)


class ADLineIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 1:
            raise InsufficientDataError("ADLineIndicator requires at least 1 data point.")
        ad_val = ad(high=data["high"], low=data["low"], close=data["close"], volume=data["volume"])
        if ad_val is None or ad_val.empty or pd.isna(ad_val.iloc[-1]):
            raise InsufficientDataError("AD Line calculation resulted in NaN.")

        value = float(ad_val.iloc[-1])
        strength = 50.0
        interpretation = "accumulation" if value > ad_val.iloc[-2] else "distribution"

        return IndicatorResult("ADLine", value, strength, interpretation)


class ForceIndexIndicator(TechnicalIndicator):
    def __init__(self, period: int = 13):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"ForceIndexIndicator requires at least {self.period} data points.")
        fi = efi(close=data["close"], volume=data["volume"], length=self.period)
        if fi is None or fi.empty or pd.isna(fi.iloc[-1]):
            raise InsufficientDataError("Force Index calculation resulted in NaN.")

        value = float(fi.iloc[-1])
        strength = abs(value / 1e6)
        interpretation = "bullish" if value > 0 else "bearish"

        return IndicatorResult("ForceIndex", value, min(strength, 100.0), interpretation)


class VWMAIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"VWMAIndicator requires at least {self.period} data points.")
        vwma = ta.vwma(close=data["close"], volume=data["volume"], length=self.period)
        if vwma is None or vwma.empty or pd.isna(vwma.iloc[-1]):
            raise InsufficientDataError("VWMA calculation resulted in NaN.")

        value = float(vwma.iloc[-1])
        current_price = data["close"].iloc[-1]
        strength = abs(current_price - value) / value * 100 if value > 0 else 0
        interpretation = "above_vwma" if current_price > value else "below_vwma"

        return IndicatorResult("VWMA", value, min(strength, 100.0), interpretation)


class EaseOfMovementIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"EOMIndicator requires at least {self.period} data points.")
        eom_val = eom(high=data["high"], low=data["low"], close=data["close"], volume=data["volume"], length=self.period)
        if eom_val is None or eom_val.empty or eom_val.iloc[-1].isna().any():
            raise InsufficientDataError("EOM calculation resulted in NaN.")

        value = float(eom_val.iloc[-1, 0])
        strength = abs(value)
        interpretation = "easy_move_up" if value > 0 else "easy_move_down"

        return IndicatorResult("EOM", value, min(strength, 100.0), interpretation)


class PriceVolumeRankIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 20:
            raise InsufficientDataError("PVRIndicator requires at least 20 data points.")
        price_rank = data['close'].rank(pct=True).iloc[-1]
        volume_rank = data['volume'].rank(pct=True).iloc[-1]
        pvr = price_rank * volume_rank
        strength = pvr * 100
        interpretation = "strong_bullish" if pvr > 0.8 else "strong_bearish" if pvr < 0.2 else "neutral"

        return IndicatorResult("PVR", pvr, strength, interpretation)


class AccumulationDistributionOscillatorIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 10:
            raise InsufficientDataError("ADOIndicator requires at least 10 data points.")
        ad_line = ad(high=data["high"], low=data["low"], close=data["close"], volume=data["volume"])
        fast_ma = ad_line.rolling(3).mean()
        slow_ma = ad_line.rolling(10).mean()
        if fast_ma.isna().iloc[-1] or slow_ma.isna().iloc[-1]:
            raise InsufficientDataError("ADO calculation resulted in NaN.")

        value = fast_ma.iloc[-1] - slow_ma.iloc[-1]
        strength = abs(value / slow_ma.iloc[-1]) * 100 if slow_ma.iloc[-1] != 0 else 0
        interpretation = "accumulation" if value > 0 else "distribution"

        return IndicatorResult("ADO", value, min(strength, 100.0), interpretation)


class PriceVolumeTrendIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 2:
            raise InsufficientDataError("PVTIndicator requires at least 2 data points.")
        pvt_val = pvt(close=data["close"], volume=data["volume"])
        if pvt_val is None or pvt_val.empty or pd.isna(pvt_val.iloc[-1]):
            raise InsufficientDataError("PVT calculation resulted in NaN.")

        value = float(pvt_val.iloc[-1])
        strength = 50.0
        interpretation = "volume_confirming_uptrend" if value > pvt_val.iloc[-2] else "volume_confirming_downtrend"

        return IndicatorResult("PVT", value, strength, interpretation)


class VolumeOscillatorIndicator(TechnicalIndicator):
    def __init__(self, fast_period: int = 5, slow_period: int = 10):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.slow_period:
            raise InsufficientDataError(f"VolumeOscillatorIndicator requires at least {self.slow_period} data points.")
        pvo_val = pvo(volume=data["volume"], fast=self.fast_period, slow=self.slow_period)
        if pvo_val is None or pvo_val.empty or pvo_val.iloc[-1].isna().any():
            raise InsufficientDataError("Volume Oscillator calculation resulted in NaN.")

        value = float(pvo_val.iloc[-1, 0])
        strength = abs(value) * 5
        interpretation = "volume_increasing" if value > 0 else "volume_decreasing"

        return IndicatorResult("VolumeOscillator", value, min(strength, 100.0), interpretation)


class KlingerVolumeOscillatorIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 55:
            raise InsufficientDataError("KVOIndicator requires at least 55 data points.")
        kvo_val = kvo(high=data["high"], low=data["low"], close=data["close"], volume=data["volume"])
        if kvo_val is None or kvo_val.empty or kvo_val.iloc[-1].isna().any():
            raise InsufficientDataError("KVO calculation resulted in NaN.")

        value = float(kvo_val.iloc[-1, 0])
        signal = float(kvo_val.iloc[-1, 1])
        strength = abs(value) / 1e9 * 50 if value != 0 else 0
        interpretation = "bullish" if value > signal else "bearish"

        return IndicatorResult("KVO", value, min(strength, 100.0), interpretation)


class PVOIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 26:
            raise InsufficientDataError("PVO requires at least 26 data points.")
        pvo_val = pvo(volume=data["volume"])
        if pvo_val is None or pvo_val.empty or pvo_val.iloc[-1].isna().any():
            raise InsufficientDataError("PVO calculation resulted in NaN.")

        value = float(pvo_val.iloc[-1, 0])
        signal = float(pvo_val.iloc[-1, 1])
        strength = abs(value) * 5
        interpretation = "bullish" if value > signal else "bearish"

        return IndicatorResult("PVO", value, min(strength, 100.0), interpretation)


class NVIIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 2:
            raise InsufficientDataError("NVI requires at least 2 data points.")
        nvi_val = nvi(close=data["close"], volume=data["volume"])
        if nvi_val is None or nvi_val.empty or pd.isna(nvi_val.iloc[-1]):
            raise InsufficientDataError("NVI calculation resulted in NaN.")

        value = float(nvi_val.iloc[-1])
        strength = 50.0
        interpretation = "smart_money_accumulating" if value > nvi_val.iloc[-2] else "smart_money_distributing"

        return IndicatorResult("NVI", value, strength, interpretation)


class PVIIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 2:
            raise InsufficientDataError("PVI requires at least 2 data points.")
        pvi_val = pvi(close=data["close"], volume=data["volume"])
        if pvi_val is None or pvi_val.empty or pd.isna(pvi_val.iloc[-1]):
            raise InsufficientDataError("PVI calculation resulted in NaN.")

        value = float(pvi_val.iloc[-1])
        strength = 50.0
        interpretation = "crowd_is_bullish" if value > pvi_val.iloc[-2] else "crowd_is_bearish"

        return IndicatorResult("PVI", value, strength, interpretation)


class MFIBillWilliamsIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 2:
            raise InsufficientDataError("MFIBW requires at least 2 data points.")
        mfi = (data['high'] + data['low'] + data['close']) / 3 * data['volume']
        value = mfi.iloc[-1]
        strength = 50.0
        interpretation = "green_bar" if value > mfi.iloc[-2] else "red_bar"

        return IndicatorResult("MFIBW", value, strength, interpretation)


class VolumeWeightedRSIIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"VolumeWeightedRSI requires at least {self.period} data points.")
        vw_rsi = ta.vw_rsi(close=data['close'], volume=data['volume'], length=self.period)
        if vw_rsi is None or vw_rsi.empty or pd.isna(vw_rsi.iloc[-1]):
            raise InsufficientDataError("VW_RSI calculation resulted in NaN.")

        value = float(vw_rsi.iloc[-1])
        if value > 70:
            interpretation = "overbought"
            strength = (value - 70) * 100 / 30
        elif value < 30:
            interpretation = "oversold"
            strength = (30 - value) * 100 / 30
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult("VW_RSI", value, min(strength, 100.0), interpretation)


class AccumulationDistributionIndexIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 2:
            raise InsufficientDataError("ADI requires at least 2 data points.")
        adi = talib.AD(data['high'], data['low'], data['close'], data['volume'])
        if adi.size == 0 or pd.isna(adi[-1]):
            raise InsufficientDataError("ADI calculation failed.")

        value = float(adi[-1])
        strength = 50.0
        interpretation = "accumulation" if value > adi[-2] else "distribution"

        return IndicatorResult("ADI", value, strength, interpretation)


class OrderFlowImbalanceIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 2:
            raise InsufficientDataError("OrderFlowImbalance requires at least 2 data points.")

        price_change = data['close'].diff()
        volume = data['volume']
        flow = (price_change * volume).cumsum()
        if flow.isna().iloc[-1]:
            raise InsufficientDataError("Order Flow calculation failed.")

        value = flow.iloc[-1]
        strength = abs(value / flow.mean()) * 10 if flow.mean() != 0 else 0
        interpretation = "buy_imbalance" if value > 0 else "sell_imbalance"

        return IndicatorResult("OrderFlowImbalance", value, min(strength, 100.0), interpretation)