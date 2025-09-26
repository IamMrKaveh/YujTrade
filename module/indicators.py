from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
import pandas as pd
import talib

import pandas_ta as ta
from pandas_ta.momentum import stoch, roc, stochrsi, trix, uo, squeeze
from pandas_ta.overlap import ichimoku, supertrend
from pandas_ta.volatility import bbands, atr
from pandas_ta.trend import adx as adx_ta, aroon
from pandas_ta.volume import cmf, obv, ad, eom, efi

from .core import IndicatorResult
from .logger_config import logger


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
        self.ma_type = ma_type.upper()

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        try:
            if self.ma_type == "EMA":
                ma_series = talib.EMA(data['close'], timeperiod=self.period)
            else:
                ma_series = talib.SMA(data['close'], timeperiod=self.period)
        except Exception:
            ma_series = pd.Series(np.nan, index=data.index)

        if getattr(ma_series, "empty", False) or pd.isna(ma_series.iloc[-1]):
            return IndicatorResult(name=f"{self.ma_type}_{self.period}", value=0, signal_strength=0, interpretation="neutral")

        current_price = data['close'].iloc[-1]
        current_ma = ma_series.iloc[-1]

        signal_strength = abs((current_price - current_ma) / current_ma) * 100 if current_ma != 0 else 0
        interpretation = "bullish_above_ma" if current_price > current_ma else "bearish_below_ma"

        return IndicatorResult(
            name=f"{self.ma_type}_{self.period}",
            value=current_ma,
            signal_strength=signal_strength,
            interpretation=interpretation
        )


class RSIIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        rsi_series = talib.RSI(data['close'], timeperiod=self.period)
        if getattr(rsi_series, "empty", False) or pd.isna(rsi_series.iloc[-1]):
            return IndicatorResult(name="RSI", value=50, signal_strength=0, interpretation="neutral")

        current_rsi = rsi_series.iloc[-1]

        if current_rsi > 70:
            interpretation = "overbought"
            signal_strength = min((current_rsi - 70) / 30 * 100, 100)
        elif current_rsi < 30:
            interpretation = "oversold"
            signal_strength = min((30 - current_rsi) / 30 * 100, 100)
        else:
            interpretation = "neutral"
            signal_strength = max(0, 50 - abs(current_rsi - 50))

        return IndicatorResult(name="RSI", value=current_rsi, signal_strength=signal_strength, interpretation=interpretation)


class MACDIndicator(TechnicalIndicator):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        macd, macdsignal, macdhist = talib.MACD(data['close'], fastperiod=self.fast, slowperiod=self.slow, signalperiod=self.signal_period)
        if getattr(macd, "empty", False) or pd.isna(macd.iloc[-1]):
            return IndicatorResult(name="MACD", value=0, signal_strength=0, interpretation="neutral")

        current_macd = macd.iloc[-1]
        current_signal = macdsignal.iloc[-1]
        current_histogram = macdhist.iloc[-1]

        if current_macd > current_signal and current_histogram > 0:
            interpretation = "bullish_crossover"
            signal_strength = min(abs(current_histogram) * 1000, 100)
        elif current_macd < current_signal and current_histogram < 0:
            interpretation = "bearish_crossover"
            signal_strength = min(abs(current_histogram) * 1000, 100)
        else:
            interpretation = "neutral"
            signal_strength = 50

        return IndicatorResult(name="MACD", value=current_macd, signal_strength=signal_strength, interpretation=interpretation)


class BollingerBandsIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20, std_dev: float = 2):
        self.period = period
        self.std_dev = std_dev

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        bb_df = bbands(data['close'], length=self.period, std=self.std_dev)
        if bb_df is None or bb_df.empty or bb_df.dropna().empty:
            return IndicatorResult(name="BB", value=0.5, signal_strength=0, interpretation="neutral")

        upper_band = bb_df.iloc[-1, 2]
        lower_band = bb_df.iloc[-1, 0]
        current_price = data['close'].iloc[-1]

        band_width = upper_band - lower_band
        if band_width == 0:
            return IndicatorResult(name="BB", value=0.5, signal_strength=0, interpretation="neutral")

        bb_position = (current_price - lower_band) / band_width

        if bb_position > 0.8:
            interpretation = "near_upper_band"
            signal_strength = (bb_position - 0.8) / 0.2 * 100
        elif bb_position < 0.2:
            interpretation = "near_lower_band"
            signal_strength = (0.2 - bb_position) / 0.2 * 100
        else:
            interpretation = "middle_range"
            signal_strength = 50

        return IndicatorResult(name="BB", value=bb_position, signal_strength=signal_strength, interpretation=interpretation)


class StochasticIndicator(TechnicalIndicator):
    def __init__(self, k_period: int = 14, d_period: int = 3, s_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period
        self.s_period = s_period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        slowk, slowd = talib.STOCH(data['high'], data['low'], data['close'],
                                   fastk_period=self.k_period,
                                   slowk_period=self.s_period,
                                   slowd_period=self.d_period)

        if getattr(slowk, "empty", False) or pd.isna(slowk.iloc[-1]):
            return IndicatorResult(name="STOCH", value=50, signal_strength=0, interpretation="neutral")

        current_k = slowk.iloc[-1]
        current_d = slowd.iloc[-1]
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

        return IndicatorResult(name="STOCH", value=avg_stoch, signal_strength=signal_strength, interpretation=interpretation)


class VolumeIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if 'volume' not in data.columns or len(data) < self.period:
            return IndicatorResult(name="VOLUME", value=1, signal_strength=50, interpretation="normal_volume")

        volume_ma = talib.SMA(data['volume'], timeperiod=self.period)
        if getattr(volume_ma, "empty", False) or pd.isna(volume_ma.iloc[-1]):
            return IndicatorResult(name="VOLUME", value=1, signal_strength=50, interpretation="normal_volume")

        current_volume = data['volume'].iloc[-1]
        average_volume = volume_ma.iloc[-1]

        volume_ratio = current_volume / average_volume if average_volume > 0 else 1

        if volume_ratio > 1.5:
            interpretation = "high_volume"
            signal_strength = min((volume_ratio - 1) * 50, 100)
        elif volume_ratio < 0.5:
            interpretation = "low_volume"
            signal_strength = max(0, (1 - volume_ratio) * 100)
        else:
            interpretation = "normal_volume"
            signal_strength = 50

        return IndicatorResult(name="VOLUME", value=volume_ratio, signal_strength=signal_strength, interpretation=interpretation)


class ATRIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        atr_series = talib.ATR(data['high'], data['low'], data['close'], timeperiod=self.period)
        if getattr(atr_series, "empty", False) or pd.isna(atr_series.iloc[-1]):
            return IndicatorResult(name="ATR", value=0, signal_strength=0, interpretation="neutral")

        current_atr = atr_series.iloc[-1]
        current_price = data['close'].iloc[-1]

        atr_percentage = (current_atr / current_price) * 100 if current_price > 0 else 0

        if atr_percentage > 3:
            interpretation = "high_volatility"
            signal_strength = min(atr_percentage * 20, 100)
        elif atr_percentage < 1:
            interpretation = "low_volatility"
            signal_strength = max(0, (1 - atr_percentage) * 100)
        else:
            interpretation = "normal_volatility"
            signal_strength = 50

        return IndicatorResult(name="ATR", value=current_atr, signal_strength=signal_strength, interpretation=interpretation)


class IchimokuIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        ichimoku_df, _ = ichimoku(data['high'], data['low'], data['close'])
        if ichimoku_df is None or ichimoku_df.empty or ichimoku_df.iloc[-1].isna().any():
            return IndicatorResult(name="Ichimoku", value=0, signal_strength=0, interpretation="neutral")

        senkou_a = ichimoku_df.iloc[-1, 0]
        senkou_b = ichimoku_df.iloc[-1, 1]
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
        return IndicatorResult(name="Ichimoku", value=current_price, signal_strength=signal_strength, interpretation=interpretation)


class WilliamsRIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        willr_series = talib.WILLR(data['high'], data['low'], data['close'], timeperiod=self.period)
        if getattr(willr_series, "empty", False) or pd.isna(willr_series.iloc[-1]):
            return IndicatorResult(name="Williams %R", value=-50, signal_strength=0, interpretation="neutral")

        value = willr_series.iloc[-1]

        if value > -20:
            interpretation = "overbought"
            signal_strength = abs(value) / 20 * 100
        elif value < -80:
            interpretation = "oversold"
            signal_strength = (abs(value) - 80) / 20 * 100
        else:
            interpretation = "neutral"
            signal_strength = max(0, 50 - abs(value + 50) / 30 * 50)

        return IndicatorResult(name="Williams %R", value=value, signal_strength=signal_strength, interpretation=interpretation)


class CCIIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        cci_series = talib.CCI(data['high'], data['low'], data['close'], timeperiod=self.period)
        if getattr(cci_series, "empty", False) or pd.isna(cci_series.iloc[-1]):
            return IndicatorResult(name="CCI", value=0, signal_strength=0, interpretation="neutral")

        cci = cci_series.iloc[-1]

        if cci > 100:
            interpretation = "overbought"
            signal_strength = min((abs(cci) - 100) / 200 * 100, 100)
        elif cci < -100:
            interpretation = "oversold"
            signal_strength = min((abs(cci) - 100) / 200 * 100, 100)
        else:
            interpretation = "neutral"
            signal_strength = max(0, 50 - abs(cci) / 2)

        return IndicatorResult(name="CCI", value=cci, signal_strength=signal_strength, interpretation=interpretation)


class SuperTrendIndicator(TechnicalIndicator):
    def __init__(self, period=7, multiplier=3):
        self.period = period
        self.multiplier = multiplier

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        st_df = supertrend(data['high'], data['low'], data['close'], length=self.period, multiplier=self.multiplier)
        if st_df is None or st_df.empty or st_df.iloc[-1].isna().any():
            return IndicatorResult(name="SuperTrend", value=0, signal_strength=50, interpretation="neutral")

        current_value = st_df.iloc[-1, 0]
        current_direction = st_df.iloc[-1, 1]

        if current_direction == 1:
            interpretation = "bullish"
            strength = 100
        elif current_direction == -1:
            interpretation = "bearish"
            strength = 100
        else:
            interpretation = "neutral"
            strength = 50

        return IndicatorResult(name="SuperTrend", value=current_value, signal_strength=strength, interpretation=interpretation)


class ADXIndicator(TechnicalIndicator):
    def __init__(self, period=14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        adx_series = talib.ADX(data['high'], data['low'], data['close'], timeperiod=self.period)
        if getattr(adx_series, "empty", False) or pd.isna(adx_series.iloc[-1]):
            return IndicatorResult(name="ADX", value=0, signal_strength=0, interpretation="weak_trend")

        adx = adx_series.iloc[-1]

        if adx > 25:
            interpretation = "strong_trend"
        else:
            interpretation = "weak_trend"
        strength = min(adx, 100)

        return IndicatorResult(name="ADX", value=adx, signal_strength=strength, interpretation=interpretation)


class ChaikinMoneyFlowIndicator(TechnicalIndicator):
    def __init__(self, period=20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        cmf_series = cmf(data['high'], data['low'], data['close'], data['volume'], length=self.period)
        if cmf_series is None or cmf_series.empty or pd.isna(cmf_series.iloc[-1]):
            return IndicatorResult(name="CMF", value=0, signal_strength=50, interpretation="neutral")

        current_value = cmf_series.iloc[-1]

        if current_value > 0:
            interpretation = "buy_pressure"
            strength = min(abs(current_value) * 200, 100)
        elif current_value < 0:
            interpretation = "sell_pressure"
            strength = min(abs(current_value) * 200, 100)
        else:
            interpretation = "neutral"
            strength = 50

        return IndicatorResult(name="CMF", value=current_value, signal_strength=strength, interpretation=interpretation)


class OBVIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        obv_series = obv(data['close'], data['volume'])
        if obv_series is None or obv_series.empty or len(obv_series) < 2:
            return IndicatorResult(name="OBV", value=0, signal_strength=50, interpretation="neutral")

        current_value = obv_series.iloc[-1]
        previous_value = obv_series.iloc[-2]

        if current_value > previous_value:
            interpretation = "bullish"
            strength = 100
        elif current_value < previous_value:
            interpretation = "bearish"
            strength = 100
        else:
            interpretation = "neutral"
            strength = 50

        return IndicatorResult(name="OBV", value=current_value, signal_strength=strength, interpretation=interpretation)


class ParabolicSARIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        psar_series = talib.SAR(data['high'], data['low'])
        if getattr(psar_series, "empty", False) or psar_series.iloc[-1] is None or np.isnan(psar_series.iloc[-1]):
            return IndicatorResult("ParabolicSAR", 0, 0, "neutral")

        current_psar = psar_series.iloc[-1]
        current_close = data['close'].iloc[-1]

        interpretation = "neutral"
        strength = 50

        if current_close > current_psar:
            interpretation = "bullish_trend"
            strength = 100
        elif current_close < current_psar:
            interpretation = "bearish_trend"
            strength = 100

        return IndicatorResult(
            name="ParabolicSAR",
            value=current_psar,
            signal_strength=strength,
            interpretation=interpretation
        )


class SqueezeMomentumIndicator(TechnicalIndicator):
    def __init__(self, length=20, mult=2, length_kc=20, mult_kc=1.5):
        self.length = length
        self.mult = mult
        self.length_kc = length_kc
        self.mult_kc = mult_kc

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        squeeze_df = squeeze(data['high'], data['low'], data['close'],
                                bb_length=self.length, bb_std=self.mult,
                                kc_length=self.length_kc, kc_scalar=self.mult_kc)
        if squeeze_df is None or squeeze_df.empty or squeeze_df.iloc[-1].isna().any():
            return IndicatorResult("SqueezeMomentum", 0, 0, "neutral")

        momentum_val = squeeze_df.iloc[-1, 3]
        current_squeeze = squeeze_df.iloc[-1, 0]

        interpretation = "neutral"
        strength = 0

        if current_squeeze == 1:
            interpretation = "squeeze_on"
            strength = 50
        else:
            if momentum_val > 0:
                interpretation = "bullish_momentum"
                strength = min(abs(momentum_val) * 100, 100)
            else:
                interpretation = "bearish_momentum"
                strength = min(abs(momentum_val) * 100, 100)

        return IndicatorResult(
            name="SqueezeMomentum",
            value=momentum_val,
            signal_strength=strength,
            interpretation=interpretation
        )


class VWAPIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        df = data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            else:
                 return IndicatorResult(name="VWAP", value=0, signal_strength=0, interpretation="neutral_no_datetime")
        
        current_vwap = ta.vwap(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
        if current_vwap is None or (hasattr(current_vwap, "empty") and current_vwap.empty) or pd.isna(current_vwap.iloc[-1]):
            return IndicatorResult(name="VWAP", value=0, signal_strength=0, interpretation="neutral")

        current_vwap_val = current_vwap.iloc[-1]
        current_price = df['close'].iloc[-1]

        interpretation = "price_above_vwap" if current_price > current_vwap_val else "price_below_vwap"
        strength = min(abs(current_price - current_vwap_val) / current_vwap_val * 100, 100) if current_vwap_val > 0 else 0

        return IndicatorResult(name="VWAP", value=current_vwap_val, signal_strength=strength, interpretation=interpretation)


class MoneyFlowIndexIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        mfi_series = ta.mfi(high=data['high'], low=data['low'], close=data['close'], volume=data['volume'], length=self.period)
        if mfi_series is None or mfi_series.empty or pd.isna(mfi_series.iloc[-1]):
            return IndicatorResult(name="MFI", value=50, signal_strength=0, interpretation="neutral")

        current_mfi = mfi_series.iloc[-1]

        if current_mfi > 80:
            interpretation = "overbought"
            signal_strength = min((current_mfi - 80) / 20 * 100, 100)
        elif current_mfi < 20:
            interpretation = "oversold"
            signal_strength = min((20 - current_mfi) / 20 * 100, 100)
        else:
            interpretation = "neutral"
            signal_strength = 50

        return IndicatorResult(name="MFI", value=current_mfi, signal_strength=signal_strength, interpretation=interpretation)

class AroonIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        aroon_df = aroon(data['high'], data['low'], length=self.period)
        if aroon_df is None or aroon_df.empty or aroon_df.iloc[-1].isna().any():
            return IndicatorResult(name="Aroon", value=0, signal_strength=0, interpretation="neutral")

        aroon_up = aroon_df.iloc[-1, 0]
        aroon_down = aroon_df.iloc[-1, 1]

        if aroon_up > 70 and aroon_down < 30:
            interpretation = "strong_uptrend"
            strength = aroon_up
        elif aroon_down > 70 and aroon_up < 30:
            interpretation = "strong_downtrend"
            strength = aroon_down
        else:
            interpretation = "ranging"
            strength = 50

        return IndicatorResult(name="Aroon", value=aroon_up - aroon_down, signal_strength=strength, interpretation=interpretation)


class UltimateOscillatorIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        uo_series = uo(data['high'], data['low'], data['close'])
        if uo_series is None or uo_series.empty or pd.isna(uo_series.iloc[-1]):
            return IndicatorResult(name="UltimateOscillator", value=50, signal_strength=0, interpretation="neutral")

        current_uo = uo_series.iloc[-1]

        if current_uo > 70:
            interpretation = "overbought"
            strength = min((current_uo - 70) / 30 * 100, 100)
        elif current_uo < 30:
            interpretation = "oversold"
            strength = min((30 - current_uo) / 30 * 100, 100)
        else:
            interpretation = "neutral"
            strength = 50

        return IndicatorResult(name="UltimateOscillator", value=current_uo, signal_strength=strength, interpretation=interpretation)


class ROCIndicator(TechnicalIndicator):
    def __init__(self, period: int = 10):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        roc_series = roc(data['close'], length=self.period)
        if roc_series is None or roc_series.empty or pd.isna(roc_series.iloc[-1]):
            return IndicatorResult(name="ROC", value=0, signal_strength=0, interpretation="neutral")

        current_roc = roc_series.iloc[-1]

        if current_roc > 0:
            interpretation = "bullish_momentum"
        else:
            interpretation = "bearish_momentum"

        strength = min(abs(current_roc) * 10, 100)

        return IndicatorResult(name="ROC", value=current_roc, signal_strength=strength, interpretation=interpretation)


class ADLineIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        ad_series = ad(data['high'], data['low'], data['close'], data['volume'])
        if ad_series is None or ad_series.empty or len(ad_series) < 2:
            return IndicatorResult(name="AD_Line", value=0, signal_strength=50, interpretation="neutral")

        if ad_series.iloc[-1] > ad_series.iloc[-2]:
            interpretation = "accumulation"
            strength = 100
        else:
            interpretation = "distribution"
            strength = 100

        return IndicatorResult(name="AD_Line", value=ad_series.iloc[-1], signal_strength=strength, interpretation=interpretation)


class ForceIndexIndicator(TechnicalIndicator):
    def __init__(self, period: int = 13):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        fi_series = efi(data['close'], data['volume'], length=self.period)
        if fi_series is None or fi_series.empty or pd.isna(fi_series.iloc[-1]):
            return IndicatorResult(name="ForceIndex", value=0, signal_strength=0, interpretation="neutral")

        current_fi = fi_series.iloc[-1]

        if current_fi > 0:
            interpretation = "bull_power"
        else:
            interpretation = "bear_power"

        strength = min(abs(current_fi) / 1_000_000, 100)

        return IndicatorResult(name="ForceIndex", value=current_fi, signal_strength=strength, interpretation=interpretation)


class VWMAIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        vwma_series = ta.vwma(data['close'], data['volume'], length=self.period)
        if vwma_series is None or vwma_series.empty or pd.isna(vwma_series.iloc[-1]):
            return IndicatorResult(name="VWMA", value=0, signal_strength=0, interpretation="neutral")

        current_vwma = vwma_series.iloc[-1]
        current_price = data['close'].iloc[-1]

        interpretation = "price_above_vwma" if current_price > current_vwma else "price_below_vwma"
        strength = min(abs(current_price - current_vwma) / current_vwma * 100, 100) if current_vwma > 0 else 0

        return IndicatorResult(name="VWMA", value=current_vwma, signal_strength=strength, interpretation=interpretation)


class KeltnerChannelsIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20, atr_period: int = 10):
        self.period = period
        self.atr_period = atr_period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        kc_df = ta.kc(data['high'], data['low'], data['close'], length=self.period, atr_length=self.atr_period)
        if kc_df is None or kc_df.empty or kc_df.iloc[-1].isna().any():
            return IndicatorResult(name="KeltnerChannels", value=0.5, signal_strength=0, interpretation="neutral")

        upper = kc_df.iloc[-1, 1]
        lower = kc_df.iloc[-1, 0]
        current_price = data['close'].iloc[-1]

        if current_price > upper:
            interpretation = "breakout_above"
            strength = 100
        elif current_price < lower:
            interpretation = "breakdown_below"
            strength = 100
        else:
            interpretation = "in_channel"
            strength = 50

        return IndicatorResult(name="KeltnerChannels", value=current_price, signal_strength=strength, interpretation=interpretation)


class DonchianChannelsIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        dc_df = ta.donchian(data['high'], data['low'], lower_length=self.period, upper_length=self.period)
        if dc_df is None or dc_df.empty or dc_df.iloc[-1].isna().any():
            return IndicatorResult(name="DonchianChannels", value=0.5, signal_strength=0, interpretation="neutral")

        upper = dc_df.iloc[-1, 1]
        lower = dc_df.iloc[-1, 0]
        current_price = data['close'].iloc[-1]

        if current_price >= upper:
            interpretation = "at_upper_channel"
            strength = 100
        elif current_price <= lower:
            interpretation = "at_lower_channel"
            strength = 100
        else:
            interpretation = "in_channel"
            strength = 50

        return IndicatorResult(name="DonchianChannels", value=current_price, signal_strength=strength, interpretation=interpretation)


class TRIXIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        trix_series = trix(data['close'], length=self.period)
        if trix_series is None or trix_series.empty:
            return IndicatorResult(name="TRIX", value=0, signal_strength=0, interpretation="neutral")

        if isinstance(trix_series, pd.DataFrame):
            if trix_series.iloc[-1].isna().any():
                return IndicatorResult(name="TRIX", value=0, signal_strength=0, interpretation="neutral")
            current_trix = trix_series.iloc[-1, 0]
        else:
            if pd.isna(trix_series.iloc[-1]):
                return IndicatorResult(name="TRIX", value=0, signal_strength=0, interpretation="neutral")
            current_trix = trix_series.iloc[-1]

        if current_trix > 0:
            interpretation = "bullish"
        else:
            interpretation = "bearish"

        strength = min(abs(current_trix) * 100, 100)

        return IndicatorResult(name="TRIX", value=current_trix, signal_strength=strength, interpretation=interpretation)


class EaseOfMovementIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        eom_series = eom(data['high'], data['low'], data['close'], data['volume'], length=self.period)
        if eom_series is None or eom_series.empty or pd.isna(eom_series.iloc[-1]):
            return IndicatorResult(name="EOM", value=0, signal_strength=0, interpretation="neutral")

        current_eom = eom_series.iloc[-1]

        if current_eom > 0:
            interpretation = "easy_upward_move"
            strength = min(current_eom / 1000, 100)
        else:
            interpretation = "easy_downward_move"
            strength = min(abs(current_eom) / 1000, 100)

        return IndicatorResult(name="EOM", value=current_eom, signal_strength=strength, interpretation=interpretation)


class StandardDeviationIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        std_dev_series = talib.STDDEV(data['close'], timeperiod=self.period)
        if getattr(std_dev_series, "empty", False) or pd.isna(std_dev_series.iloc[-1]):
            return IndicatorResult(name="StdDev", value=0, signal_strength=0, interpretation="neutral")

        current_std_dev = std_dev_series.iloc[-1]
        avg_std_dev = std_dev_series.mean()

        if current_std_dev > avg_std_dev * 1.5:
            interpretation = "high_volatility"
        elif current_std_dev < avg_std_dev * 0.5:
            interpretation = "low_volatility"
        else:
            interpretation = "normal_volatility"

        strength = min((current_std_dev / data['close'].iloc[-1]) * 100, 100) if data['close'].iloc[-1] > 0 else 0

        return IndicatorResult(name="StdDev", value=current_std_dev, signal_strength=strength, interpretation=interpretation)