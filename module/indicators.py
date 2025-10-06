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
            close_np = data['close'].to_numpy(dtype=np.float64)
            if self.ma_type == "EMA":
                ma_series = talib.EMA(close_np, timeperiod=self.period)
            else:
                ma_series = talib.SMA(close_np, timeperiod=self.period)
        except Exception:
            ma_series = np.array([])

        if ma_series.size == 0 or pd.isna(ma_series[-1]):
            return IndicatorResult(name=f"{self.ma_type}_{self.period}", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_price = data['close'].iloc[-1]
        current_ma = ma_series[-1]

        signal_strength = abs((current_price - current_ma) / current_ma) * 100 if current_ma != 0 else 0
        interpretation = "bullish_above_ma" if current_price > current_ma else "bearish_below_ma"

        return IndicatorResult(
            name=f"{self.ma_type}_{self.period}",
            value=float(current_ma),
            signal_strength=float(signal_strength),
            interpretation=interpretation
        )


class RSIIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        rsi_series = talib.RSI(data['close'].to_numpy(dtype=np.float64), timeperiod=self.period)
        if rsi_series.size == 0 or pd.isna(rsi_series[-1]):
            return IndicatorResult(name="RSI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_rsi = float(rsi_series[-1])

        if current_rsi > 70:
            interpretation = "overbought"
            signal_strength = min((current_rsi - 70) / 30 * 100, 100)
        elif current_rsi < 30:
            interpretation = "oversold"
            signal_strength = min((30 - current_rsi) / 30 * 100, 100)
        else:
            interpretation = "neutral"
            signal_strength = max(0, 50 - abs(current_rsi - 50))

        return IndicatorResult(name="RSI", value=current_rsi, signal_strength=float(signal_strength), interpretation=interpretation)


class MACDIndicator(TechnicalIndicator):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        macd, macdsignal, macdhist = talib.MACD(data['close'].to_numpy(dtype=np.float64), fastperiod=self.fast, slowperiod=self.slow, signalperiod=self.signal_period)
        if macd.size == 0 or pd.isna(macd[-1]):
            return IndicatorResult(name="MACD", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

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

        return IndicatorResult(name="MACD", value=current_macd, signal_strength=float(signal_strength), interpretation=interpretation)


class BollingerBandsIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        bb_df = bbands(close=data['close'], length=self.period, std=self.std_dev)
        if bb_df is None or bb_df.empty or bb_df.dropna().empty:
            return IndicatorResult(name="BB", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        upper_band = float(bb_df.iloc[-1, 2])
        lower_band = float(bb_df.iloc[-1, 0])
        current_price = float(data['close'].iloc[-1])

        band_width = upper_band - lower_band
        if band_width == 0:
            return IndicatorResult(name="BB", value=0.5, signal_strength=0.0, interpretation="neutral")

        bb_position = (current_price - lower_band) / band_width

        if bb_position > 0.8:
            interpretation = "near_upper_band"
            signal_strength = (bb_position - 0.8) / 0.2 * 100
        elif bb_position < 0.2:
            interpretation = "near_lower_band"
            signal_strength = (0.2 - bb_position) / 0.2 * 100
        else:
            interpretation = "middle_range"
            signal_strength = 50.0

        return IndicatorResult(name="BB", value=float(bb_position), signal_strength=float(signal_strength), interpretation=interpretation)


class StochasticIndicator(TechnicalIndicator):
    def __init__(self, k_period: int = 14, d_period: int = 3, s_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period
        self.s_period = s_period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        slowk, slowd = talib.STOCH(
            data['high'].to_numpy(dtype=np.float64), 
            data['low'].to_numpy(dtype=np.float64), 
            data['close'].to_numpy(dtype=np.float64),
            fastk_period=self.k_period,
            slowk_period=self.s_period,
            slowd_period=self.d_period
        )

        if slowk.size == 0 or pd.isna(slowk[-1]):
            return IndicatorResult(name="STOCH", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_k = float(slowk[-1])
        current_d = float(slowd[-1])
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

        return IndicatorResult(name="STOCH", value=float(avg_stoch), signal_strength=float(signal_strength), interpretation=interpretation)


class VolumeIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if 'volume' not in data.columns or len(data) < self.period:
            return IndicatorResult(name="VOLUME", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        volume_ma = talib.SMA(data['volume'].to_numpy(dtype=np.float64), timeperiod=self.period)
        if volume_ma.size == 0 or pd.isna(volume_ma[-1]):
            return IndicatorResult(name="VOLUME", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_volume = float(data['volume'].iloc[-1])
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

        return IndicatorResult(name="VOLUME", value=float(volume_ratio), signal_strength=float(signal_strength), interpretation=interpretation)


class ATRIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        atr_series = talib.ATR(
            data['high'].to_numpy(dtype=np.float64), 
            data['low'].to_numpy(dtype=np.float64), 
            data['close'].to_numpy(dtype=np.float64), 
            timeperiod=self.period
        )
        if atr_series.size == 0 or pd.isna(atr_series[-1]):
            return IndicatorResult(name="ATR", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_atr = float(atr_series[-1])
        current_price = float(data['close'].iloc[-1])

        atr_percentage = (current_atr / current_price) * 100 if current_price > 0 else 0.0

        if atr_percentage > 3:
            interpretation = "high_volatility"
            signal_strength = min(atr_percentage * 20, 100)
        elif atr_percentage < 1:
            interpretation = "low_volatility"
            signal_strength = max(0, (1 - atr_percentage) * 100)
        else:
            interpretation = "normal_volatility"
            signal_strength = 50.0

        return IndicatorResult(name="ATR", value=current_atr, signal_strength=float(signal_strength), interpretation=interpretation)


class IchimokuIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        ichimoku_result = ichimoku(high=data['high'], low=data['low'], close=data['close'])
        
        if ichimoku_result is None:
            return IndicatorResult(name="Ichimoku", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")
        
        if isinstance(ichimoku_result, tuple) and len(ichimoku_result) > 0:
            ichimoku_df = ichimoku_result[0]
        else:
            ichimoku_df = ichimoku_result
            
        if ichimoku_df is None or ichimoku_df.empty or ichimoku_df.iloc[-1].isna().any():
            return IndicatorResult(name="Ichimoku", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        senkou_a = float(ichimoku_df.iloc[-1, 0])
        senkou_b = float(ichimoku_df.iloc[-1, 1])
        current_price = float(data['close'].iloc[-1])

        if current_price > senkou_a and current_price > senkou_b:
            interpretation = "price_above_cloud"
            signal_strength = 100.0
        elif current_price < senkou_a and current_price < senkou_b:
            interpretation = "price_below_cloud"
            signal_strength = 100.0
        else:
            interpretation = "price_in_cloud"
            signal_strength = 50.0
        
        return IndicatorResult(name="Ichimoku", value=current_price, signal_strength=signal_strength, interpretation=interpretation)


class WilliamsRIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        willr_series = talib.WILLR(
            data['high'].to_numpy(dtype=np.float64), 
            data['low'].to_numpy(dtype=np.float64), 
            data['close'].to_numpy(dtype=np.float64), 
            timeperiod=self.period
        )
        if willr_series.size == 0 or pd.isna(willr_series[-1]):
            return IndicatorResult(name="Williams %R", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        value = float(willr_series[-1])

        if value > -20:
            interpretation = "overbought"
            signal_strength = abs(value) / 20 * 100
        elif value < -80:
            interpretation = "oversold"
            signal_strength = (abs(value) - 80) / 20 * 100
        else:
            interpretation = "neutral"
            signal_strength = max(0, 50 - abs(value + 50) / 30 * 50)

        return IndicatorResult(name="Williams %R", value=value, signal_strength=float(signal_strength), interpretation=interpretation)


class CCIIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        cci_series = talib.CCI(
            data['high'].to_numpy(dtype=np.float64), 
            data['low'].to_numpy(dtype=np.float64), 
            data['close'].to_numpy(dtype=np.float64), 
            timeperiod=self.period
        )
        if cci_series.size == 0 or pd.isna(cci_series[-1]):
            return IndicatorResult(name="CCI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        cci = float(cci_series[-1])

        if cci > 100:
            interpretation = "overbought"
            signal_strength = min((abs(cci) - 100) / 200 * 100, 100)
        elif cci < -100:
            interpretation = "oversold"
            signal_strength = min((abs(cci) - 100) / 200 * 100, 100)
        else:
            interpretation = "neutral"
            signal_strength = max(0, 50 - abs(cci) / 2)

        return IndicatorResult(name="CCI", value=cci, signal_strength=float(signal_strength), interpretation=interpretation)


class SuperTrendIndicator(TechnicalIndicator):
    def __init__(self, period: int = 7, multiplier: float = 3.0):
        self.period = period
        self.multiplier = multiplier

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        st_df = supertrend(high=data['high'], low=data['low'], close=data['close'], length=self.period, multiplier=self.multiplier)
        if st_df is None or st_df.empty or st_df.iloc[-1].isna().any():
            return IndicatorResult(name="SuperTrend", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_value = float(st_df.iloc[-1, 0])
        current_direction = float(st_df.iloc[-1, 1])

        if current_direction == 1:
            interpretation = "bullish"
            strength = 100.0
        elif current_direction == -1:
            interpretation = "bearish"
            strength = 100.0
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult(name="SuperTrend", value=current_value, signal_strength=strength, interpretation=interpretation)


class ADXIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        adx_series = talib.ADX(
            data['high'].to_numpy(dtype=np.float64), 
            data['low'].to_numpy(dtype=np.float64), 
            data['close'].to_numpy(dtype=np.float64), 
            timeperiod=self.period
        )
        if adx_series.size == 0 or pd.isna(adx_series[-1]):
            return IndicatorResult(name="ADX", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        adx = float(adx_series[-1])

        if adx > 25:
            interpretation = "strong_trend"
        else:
            interpretation = "weak_trend"
        strength = min(adx, 100.0)

        return IndicatorResult(name="ADX", value=adx, signal_strength=strength, interpretation=interpretation)


class ChaikinMoneyFlowIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        cmf_series = cmf(high=data['high'], low=data['low'], close=data['close'], volume=data['volume'], length=self.period)
        if cmf_series is None or cmf_series.empty or pd.isna(cmf_series.iloc[-1]):
            return IndicatorResult(name="CMF", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

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

        return IndicatorResult(name="CMF", value=current_value, signal_strength=float(strength), interpretation=interpretation)


class OBVIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        obv_series = obv(close=data['close'], volume=data['volume'])
        if obv_series is None or obv_series.empty or len(obv_series) < 2:
            return IndicatorResult(name="OBV", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_value = float(obv_series.iloc[-1])
        previous_value = float(obv_series.iloc[-2])

        if current_value > previous_value:
            interpretation = "bullish"
            strength = 100.0
        elif current_value < previous_value:
            interpretation = "bearish"
            strength = 100.0
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult(name="OBV", value=current_value, signal_strength=strength, interpretation=interpretation)


class ParabolicSARIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        psar_series = talib.SAR(data['high'].to_numpy(dtype=np.float64), data['low'].to_numpy(dtype=np.float64))
        if psar_series.size == 0 or psar_series[-1] is None or np.isnan(psar_series[-1]):
            return IndicatorResult("ParabolicSAR", np.nan, np.nan, "insufficient_data")

        current_psar = float(psar_series[-1])
        current_close = float(data['close'].iloc[-1])

        interpretation = "neutral"
        strength = 50.0

        if current_close > current_psar:
            interpretation = "bullish_trend"
            strength = 100.0
        elif current_close < current_psar:
            interpretation = "bearish_trend"
            strength = 100.0

        return IndicatorResult(
            name="ParabolicSAR",
            value=current_psar,
            signal_strength=strength,
            interpretation=interpretation
        )


class SqueezeMomentumIndicator(TechnicalIndicator):
    def __init__(self, length: int = 20, mult: float = 2.0, length_kc: int = 20, mult_kc: float = 1.5):
        self.length = length
        self.mult = mult
        self.length_kc = length_kc
        self.mult_kc = mult_kc

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        squeeze_df = squeeze(
            high=data['high'], low=data['low'], close=data['close'],
            bb_length=self.length, bb_std=self.mult,
            kc_length=self.length_kc, kc_scalar=self.mult_kc
        )
        if squeeze_df is None or squeeze_df.empty or squeeze_df.iloc[-1].isna().any():
            return IndicatorResult("SqueezeMomentum", np.nan, np.nan, "insufficient_data")

        momentum_val = float(squeeze_df.iloc[-1, 3])
        current_squeeze = float(squeeze_df.iloc[-1, 0])

        interpretation = "neutral"
        strength = 0.0

        if current_squeeze == 1:
            interpretation = "squeeze_on"
            strength = 50.0
        else:
            if momentum_val > 0:
                interpretation = "bullish_momentum"
                strength = min(abs(momentum_val) * 100, 100.0)
            else:
                interpretation = "bearish_momentum"
                strength = min(abs(momentum_val) * 100, 100.0)

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
                return IndicatorResult(name="VWAP", value=np.nan, signal_strength=np.nan, interpretation="neutral_no_datetime")
        
        current_vwap = ta.vwap(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
        if current_vwap is None or (hasattr(current_vwap, "empty") and current_vwap.empty):
            return IndicatorResult(name="VWAP", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")
        
        if isinstance(current_vwap, pd.Series):
            if pd.isna(current_vwap.iloc[-1]):
                return IndicatorResult(name="VWAP", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")
            current_vwap_val = float(current_vwap.iloc[-1])
        elif isinstance(current_vwap, pd.DataFrame):
            if current_vwap.iloc[-1].isna().any():
                return IndicatorResult(name="VWAP", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")
            current_vwap_val = float(current_vwap.iloc[-1, 0])
        else:
            return IndicatorResult(name="VWAP", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_price = float(df['close'].iloc[-1])

        interpretation = "price_above_vwap" if current_price > current_vwap_val else "price_below_vwap"
        strength = min(abs(current_price - current_vwap_val) / current_vwap_val * 100, 100.0) if current_vwap_val > 0 else 0.0

        return IndicatorResult(name="VWAP", value=current_vwap_val, signal_strength=float(strength), interpretation=interpretation)


class MoneyFlowIndexIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        mfi_series = ta.mfi(high=data['high'], low=data['low'], close=data['close'], volume=data['volume'], length=self.period)
        if mfi_series is None or mfi_series.empty or pd.isna(mfi_series.iloc[-1]):
            return IndicatorResult(name="MFI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_mfi = float(mfi_series.iloc[-1])

        if current_mfi > 80:
            interpretation = "overbought"
            signal_strength = min((current_mfi - 80) / 20 * 100, 100)
        elif current_mfi < 20:
            interpretation = "oversold"
            signal_strength = min((20 - current_mfi) / 20 * 100, 100)
        else:
            interpretation = "neutral"
            signal_strength = 50.0

        return IndicatorResult(name="MFI", value=current_mfi, signal_strength=float(signal_strength), interpretation=interpretation)


class AroonIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        aroon_df = aroon(high=data['high'], low=data['low'], length=self.period)
        if aroon_df is None or aroon_df.empty or aroon_df.iloc[-1].isna().any():
            return IndicatorResult(name="Aroon", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        aroon_up = float(aroon_df.iloc[-1, 0])
        aroon_down = float(aroon_df.iloc[-1, 1])

        if aroon_up > 70 and aroon_down < 30:
            interpretation = "strong_uptrend"
            strength = aroon_up
        elif aroon_down > 70 and aroon_up < 30:
            interpretation = "strong_downtrend"
            strength = aroon_down
        else:
            interpretation = "ranging"
            strength = 50.0

        return IndicatorResult(name="Aroon", value=aroon_up - aroon_down, signal_strength=strength, interpretation=interpretation)


class UltimateOscillatorIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        uo_series = uo(high=data['high'], low=data['low'], close=data['close'])
        if uo_series is None or uo_series.empty or pd.isna(uo_series.iloc[-1]):
            return IndicatorResult(name="UltimateOscillator", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_uo = float(uo_series.iloc[-1])

        if current_uo > 70:
            interpretation = "overbought"
            strength = min((current_uo - 70) / 30 * 100, 100)
        elif current_uo < 30:
            interpretation = "oversold"
            strength = min((30 - current_uo) / 30 * 100, 100)
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult(name="UltimateOscillator", value=current_uo, signal_strength=float(strength), interpretation=interpretation)


class ROCIndicator(TechnicalIndicator):
    def __init__(self, period: int = 10):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        roc_series = roc(close=data['close'], length=self.period)
        if roc_series is None or roc_series.empty or pd.isna(roc_series.iloc[-1]):
            return IndicatorResult(name="ROC", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_roc = float(roc_series.iloc[-1])

        if current_roc > 0:
            interpretation = "bullish_momentum"
        else:
            interpretation = "bearish_momentum"

        strength = min(abs(current_roc) * 10, 100.0)

        return IndicatorResult(name="ROC", value=current_roc, signal_strength=float(strength), interpretation=interpretation)


class ADLineIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        ad_series = ad(high=data['high'], low=data['low'], close=data['close'], volume=data['volume'])
        if ad_series is None or ad_series.empty or len(ad_series) < 2:
            return IndicatorResult(name="AD_Line", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_val = float(ad_series.iloc[-1])
        prev_val = float(ad_series.iloc[-2])

        if current_val > prev_val:
            interpretation = "accumulation"
            strength = 100.0
        else:
            interpretation = "distribution"
            strength = 100.0

        return IndicatorResult(name="AD_Line", value=current_val, signal_strength=strength, interpretation=interpretation)


class ForceIndexIndicator(TechnicalIndicator):
    def __init__(self, period: int = 13):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        fi_series = efi(close=data['close'], volume=data['volume'], length=self.period)
        if fi_series is None or fi_series.empty or pd.isna(fi_series.iloc[-1]):
            return IndicatorResult(name="ForceIndex", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_fi = float(fi_series.iloc[-1])

        if current_fi > 0:
            interpretation = "bull_power"
        else:
            interpretation = "bear_power"

        strength = min(abs(current_fi) / 1_000_000, 100.0)

        return IndicatorResult(name="ForceIndex", value=current_fi, signal_strength=float(strength), interpretation=interpretation)


class VWMAIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        vwma_series = ta.vwma(close=data['close'], volume=data['volume'], length=self.period)
        if vwma_series is None or vwma_series.empty or pd.isna(vwma_series.iloc[-1]):
            return IndicatorResult(name="VWMA", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_vwma = float(vwma_series.iloc[-1])
        current_price = float(data['close'].iloc[-1])

        interpretation = "price_above_vwma" if current_price > current_vwma else "price_below_vwma"
        strength = min(abs(current_price - current_vwma) / current_vwma * 100, 100.0) if current_vwma > 0 else 0.0

        return IndicatorResult(name="VWMA", value=current_vwma, signal_strength=float(strength), interpretation=interpretation)


class KeltnerChannelsIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20, atr_multiplier: float = 2.0):
        self.period = period
        self.atr_multiplier = atr_multiplier

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        kc_df = ta.kc(high=data['high'], low=data['low'], close=data['close'], length=self.period, scalar=self.atr_multiplier)
        if kc_df is None or kc_df.empty or kc_df.iloc[-1].isna().any():
            return IndicatorResult(name="KeltnerChannels", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        upper = float(kc_df.iloc[-1, 2])
        lower = float(kc_df.iloc[-1, 0])
        current_price = float(data['close'].iloc[-1])

        if current_price > upper:
            interpretation = "breakout_above"
            strength = 100.0
        elif current_price < lower:
            interpretation = "breakdown_below"
            strength = 100.0
        else:
            interpretation = "in_channel"
            strength = 50.0

        return IndicatorResult(name="KeltnerChannels", value=current_price, signal_strength=strength, interpretation=interpretation)


class DonchianChannelsIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        dc_df = ta.donchian(high=data['high'], low=data['low'], lower_length=self.period, upper_length=self.period)
        if dc_df is None or dc_df.empty or dc_df.iloc[-1].isna().any():
            return IndicatorResult(name="DonchianChannels", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        upper = float(dc_df.iloc[-1, 1])
        lower = float(dc_df.iloc[-1, 0])
        current_price = float(data['close'].iloc[-1])

        if current_price >= upper:
            interpretation = "at_upper_channel"
            strength = 100.0
        elif current_price <= lower:
            interpretation = "at_lower_channel"
            strength = 100.0
        else:
            interpretation = "in_channel"
            strength = 50.0

        return IndicatorResult(name="DonchianChannels", value=current_price, signal_strength=strength, interpretation=interpretation)


class TRIXIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        trix_series = trix(close=data['close'], length=self.period)
        if trix_series is None or trix_series.empty:
            return IndicatorResult(name="TRIX", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")
        
        if isinstance(trix_series, pd.DataFrame):
            if trix_series.iloc[-1].isna().any():
                return IndicatorResult(name="TRIX", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")
            current_trix = float(trix_series.iloc[-1, 0])
        else:
            if pd.isna(trix_series.iloc[-1]):
                return IndicatorResult(name="TRIX", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")
            current_trix = float(trix_series.iloc[-1])

        if current_trix > 0:
            interpretation = "bullish"
        else:
            interpretation = "bearish"

        strength = min(abs(current_trix) * 100, 100.0)

        return IndicatorResult(name="TRIX", value=current_trix, signal_strength=float(strength), interpretation=interpretation)


class EaseOfMovementIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        eom_series = eom(high=data['high'], low=data['low'], close=data['close'], volume=data['volume'], length=self.period)
        if eom_series is None or eom_series.empty or pd.isna(eom_series.iloc[-1]):
            return IndicatorResult(name="EOM", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_eom = float(eom_series.iloc[-1])

        if current_eom > 0:
            interpretation = "easy_upward_move"
            strength = min(current_eom / 1000, 100.0)
        else:
            interpretation = "easy_downward_move"
            strength = min(abs(current_eom) / 1000, 100.0)

        return IndicatorResult(name="EOM", value=current_eom, signal_strength=float(strength), interpretation=interpretation)


class StandardDeviationIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        std_dev_series = talib.STDDEV(data['close'].to_numpy(dtype=np.float64), timeperiod=self.period)
        if std_dev_series.size == 0 or pd.isna(std_dev_series[-1]):
            return IndicatorResult(name="StdDev", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_std_dev = float(std_dev_series[-1])
        avg_std_dev = float(np.nanmean(std_dev_series))

        if current_std_dev > avg_std_dev * 1.5:
            interpretation = "high_volatility"
        elif current_std_dev < avg_std_dev * 0.5:
            interpretation = "low_volatility"
        else:
            interpretation = "normal_volatility"

        strength = min((current_std_dev / float(data['close'].iloc[-1])) * 100, 100.0) if float(data['close'].iloc[-1]) > 0 else 0.0

        return IndicatorResult(name="StdDev", value=current_std_dev, signal_strength=float(strength), interpretation=interpretation)


class StochRSIIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14, rsi_period: int = 14, k: int = 3, d: int = 3):
        self.period = period
        self.rsi_period = rsi_period
        self.k = k
        self.d = d

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        stochrsi_df = stochrsi(close=data['close'], length=self.rsi_period, rsi_length=self.period, k=self.k, d=self.d)
        if stochrsi_df is None or stochrsi_df.empty or stochrsi_df.iloc[-1].isna().any():
            return IndicatorResult(name="StochRSI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_k = float(stochrsi_df.iloc[-1, 0])
        current_d = float(stochrsi_df.iloc[-1, 1])
        avg_stochrsi = (current_k + current_d) / 2

        if avg_stochrsi > 0.8:
            interpretation = "overbought"
            signal_strength = min((avg_stochrsi - 0.8) / 0.2 * 100, 100)
        elif avg_stochrsi < 0.2:
            interpretation = "oversold"
            signal_strength = min((0.2 - avg_stochrsi) / 0.2 * 100, 100)
        else:
            interpretation = "neutral"
            signal_strength = 50.0

        return IndicatorResult(name="StochRSI", value=avg_stochrsi, signal_strength=float(signal_strength), interpretation=interpretation)


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


class MassIndexIndicator(TechnicalIndicator):
    def __init__(self, fast: int = 9, slow: int = 25):
        self.fast = fast
        self.slow = slow

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        mass_idx = massi(high=data['high'], low=data['low'], fast=self.fast, slow=self.slow)
        if mass_idx is None or mass_idx.empty or pd.isna(mass_idx.iloc[-1]):
            return IndicatorResult(name="MassIndex", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_mass = float(mass_idx.iloc[-1])

        if current_mass > 27:
            interpretation = "reversal_warning"
            strength = min((current_mass - 27) * 10, 100.0)
        elif current_mass < 26.5:
            interpretation = "no_reversal"
            strength = 0.0
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult(name="MassIndex", value=current_mass, signal_strength=float(strength), interpretation=interpretation)


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


class MomentumIndicator(TechnicalIndicator):
    def __init__(self, period: int = 10):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        momentum_series = talib.MOM(data['close'].to_numpy(dtype=np.float64), timeperiod=self.period)
        if momentum_series.size == 0 or pd.isna(momentum_series[-1]):
            return IndicatorResult(name="Momentum", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_momentum = float(momentum_series[-1])

        if current_momentum > 0:
            interpretation = "positive_momentum"
            strength = min(abs(current_momentum / float(data['close'].iloc[-1])) * 100, 100.0)
        elif current_momentum < 0:
            interpretation = "negative_momentum"
            strength = min(abs(current_momentum / float(data['close'].iloc[-1])) * 100, 100.0)
        else:
            interpretation = "no_momentum"
            strength = 0.0

        return IndicatorResult(name="Momentum", value=current_momentum, signal_strength=float(strength), interpretation=interpretation)


class DetrendedPriceOscillatorIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        dpo_series = dpo(close=data['close'], length=self.period)
        if dpo_series is None or dpo_series.empty or pd.isna(dpo_series.iloc[-1]):
            return IndicatorResult(name="DPO", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_dpo = float(dpo_series.iloc[-1])

        if current_dpo > 0:
            interpretation = "overbought"
            strength = min(abs(current_dpo / float(data['close'].iloc[-1])) * 100, 100.0)
        elif current_dpo < 0:
            interpretation = "oversold"
            strength = min(abs(current_dpo / float(data['close'].iloc[-1])) * 100, 100.0)
        else:
            interpretation = "neutral"
            strength = 0.0

        return IndicatorResult(name="DPO", value=current_dpo, signal_strength=float(strength), interpretation=interpretation)
    

class ChoppinessIndexIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            return IndicatorResult(name="ChoppinessIndex", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        atr_sum = talib.SUM(
            talib.ATR(
                data['high'].to_numpy(dtype=np.float64), 
                data['low'].to_numpy(dtype=np.float64), 
                data['close'].to_numpy(dtype=np.float64), 
                timeperiod=1
            ), 
            timeperiod=self.period
        )
        
        if atr_sum.size == 0 or pd.isna(atr_sum[-1]) or atr_sum[-1] == 0:
            return IndicatorResult(name="ChoppinessIndex", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        high_low_range = float(data['high'].iloc[-self.period:].max()) - float(data['low'].iloc[-self.period:].min())
        
        if high_low_range == 0:
            return IndicatorResult(name="ChoppinessIndex", value=50.0, signal_strength=0.0, interpretation="neutral")

        choppiness = 100 * np.log10(float(atr_sum[-1]) / high_low_range) / np.log10(self.period)

        if choppiness > 61.8:
            interpretation = "choppy_market"
            strength = min((choppiness - 61.8) / 38.2 * 100, 100.0)
        elif choppiness < 38.2:
            interpretation = "trending_market"
            strength = min((38.2 - choppiness) / 38.2 * 100, 100.0)
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult(name="ChoppinessIndex", value=float(choppiness), signal_strength=float(strength), interpretation=interpretation)


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


class AwesomeOscillatorIndicator(TechnicalIndicator):
    def __init__(self, fast: int = 5, slow: int = 34):
        self.fast = fast
        self.slow = slow

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        ao_series = ta.ao(high=data['high'], low=data['low'], fast=self.fast, slow=self.slow)
        if ao_series is None or ao_series.empty or pd.isna(ao_series.iloc[-1]):
            return IndicatorResult(name="AwesomeOscillator", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_ao = float(ao_series.iloc[-1])

        if len(ao_series) > 1:
            prev_ao = float(ao_series.iloc[-2])
            if current_ao > 0 and current_ao > prev_ao:
                interpretation = "bullish_momentum"
                strength = min(abs(current_ao) / float(data['close'].iloc[-1]) * 10000, 100.0)
            elif current_ao < 0 and current_ao < prev_ao:
                interpretation = "bearish_momentum"
                strength = min(abs(current_ao) / float(data['close'].iloc[-1]) * 10000, 100.0)
            else:
                interpretation = "weakening"
                strength = 50.0
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult(name="AwesomeOscillator", value=current_ao, signal_strength=float(strength), interpretation=interpretation)


class ChandeMomentumOscillatorIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        cmo_series = cmo(close=data['close'], length=self.period)
        if cmo_series is None or cmo_series.empty or pd.isna(cmo_series.iloc[-1]):
            return IndicatorResult(name="CMO", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_cmo = float(cmo_series.iloc[-1])

        if current_cmo > 50:
            interpretation = "overbought"
            signal_strength = min((current_cmo - 50) / 50 * 100, 100)
        elif current_cmo < -50:
            interpretation = "oversold"
            signal_strength = min((abs(current_cmo) - 50) / 50 * 100, 100)
        else:
            interpretation = "neutral"
            signal_strength = 50.0

        return IndicatorResult(name="CMO", value=current_cmo, signal_strength=float(signal_strength), interpretation=interpretation)


class RelativeVigorIndexIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        try:
            if 'open' not in data.columns:
                return IndicatorResult(name="RVI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")
            
            rvi_df = ta.rvi(open_=data['open'], high=data['high'], low=data['low'], close=data['close'], length=self.period)
            
            if rvi_df is None or rvi_df.empty:
                return IndicatorResult(name="RVI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")
            
            if isinstance(rvi_df, pd.Series):
                if pd.isna(rvi_df.iloc[-1]):
                    return IndicatorResult(name="RVI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")
                rvi_value = float(rvi_df.iloc[-1])
                rvi_signal = rvi_value
            elif isinstance(rvi_df, pd.DataFrame):
                if len(rvi_df.columns) < 1 or rvi_df.iloc[-1].isna().all():
                    return IndicatorResult(name="RVI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")
                
                last_row = rvi_df.iloc[-1]
                rvi_value = float(last_row.iloc[0]) if not pd.isna(last_row.iloc[0]) else 0.0
                
                if len(rvi_df.columns) >= 2 and not pd.isna(last_row.iloc[1]):
                    rvi_signal = float(last_row.iloc[1])
                else:
                    rvi_signal = rvi_value
            else:
                return IndicatorResult(name="RVI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

            if rvi_value > rvi_signal and rvi_value > 0:
                interpretation = "bullish"
                strength = min(abs(rvi_value) * 100, 100.0)
            elif rvi_value < rvi_signal and rvi_value < 0:
                interpretation = "bearish"
                strength = min(abs(rvi_value) * 100, 100.0)
            else:
                interpretation = "neutral"
                strength = 50.0

            return IndicatorResult(name="RVI", value=rvi_value, signal_strength=float(strength), interpretation=interpretation)
        except Exception as e:
            logger.warning(f"Error calculating RVI: {e}")
            return IndicatorResult(name="RVI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")


class PriceVolumeRankIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            return IndicatorResult(name="PVR", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        price_rank = float(data['close'].iloc[-self.period:].rank(pct=True).iloc[-1])
        volume_rank = float(data['volume'].iloc[-self.period:].rank(pct=True).iloc[-1])

        pvr = (price_rank + volume_rank) / 2 * 100

        if pvr > 70:
            interpretation = "strong_uptrend"
            strength = min((pvr - 70) / 30 * 100, 100.0)
        elif pvr < 30:
            interpretation = "strong_downtrend"
            strength = min((30 - pvr) / 30 * 100, 100.0)
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult(name="PVR", value=pvr, signal_strength=float(strength), interpretation=interpretation)


class AccumulationDistributionOscillatorIndicator(TechnicalIndicator):
    def __init__(self, fast: int = 3, slow: int = 10):
        self.fast = fast
        self.slow = slow

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        ad_line = ad(high=data['high'], low=data['low'], close=data['close'], volume=data['volume'])
        if ad_line is None or ad_line.empty or len(ad_line) < self.slow:
            return IndicatorResult(name="ADO", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        fast_ema = ad_line.ewm(span=self.fast, adjust=False).mean()
        slow_ema = ad_line.ewm(span=self.slow, adjust=False).mean()
        
        ado = float(fast_ema.iloc[-1]) - float(slow_ema.iloc[-1])

        if ado > 0:
            interpretation = "accumulation"
            strength = min(abs(ado) / 1_000_000, 100.0)
        elif ado < 0:
            interpretation = "distribution"
            strength = min(abs(ado) / 1_000_000, 100.0)
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult(name="ADO", value=ado, signal_strength=float(strength), interpretation=interpretation)


class PriceVolumeTrendIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        pvt_series = pvt(close=data['close'], volume=data['volume'])
        if pvt_series is None or pvt_series.empty or len(pvt_series) < 2:
            return IndicatorResult(name="PVT", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_value = float(pvt_series.iloc[-1])
        previous_value = float(pvt_series.iloc[-2])

        if current_value > previous_value:
            interpretation = "bullish"
            strength = 100.0
        elif current_value < previous_value:
            interpretation = "bearish"
            strength = 100.0
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult(name="PVT", value=current_value, signal_strength=strength, interpretation=interpretation)


class BalanceOfPowerIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 1:
            return IndicatorResult(name="BOP", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        if 'open' not in data.columns:
            return IndicatorResult(name="BOP", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        close = float(data['close'].iloc[-1])
        open_price = float(data['open'].iloc[-1])
        high = float(data['high'].iloc[-1])
        low = float(data['low'].iloc[-1])

        if high == low:
            bop = 0.0
        else:
            bop = (close - open_price) / (high - low)

        if bop > 0.5:
            interpretation = "buyers_control"
            strength = bop * 100
        elif bop < -0.5:
            interpretation = "sellers_control"
            strength = abs(bop) * 100
        else:
            interpretation = "balanced"
            strength = 50.0

        return IndicatorResult(name="BOP", value=bop, signal_strength=float(strength), interpretation=interpretation)


class LinearRegressionIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        linreg_series = talib.LINEARREG(data['close'].to_numpy(dtype=np.float64), timeperiod=self.period)
        if linreg_series.size == 0 or pd.isna(linreg_series[-1]):
            return IndicatorResult(name="LinearRegression", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_linreg = float(linreg_series[-1])
        current_price = float(data['close'].iloc[-1])

        deviation = (current_price - current_linreg) / current_linreg * 100 if current_linreg != 0 else 0.0

        if deviation > 2:
            interpretation = "above_regression"
            strength = min(abs(deviation) * 10, 100.0)
        elif deviation < -2:
            interpretation = "below_regression"
            strength = min(abs(deviation) * 10, 100.0)
        else:
            interpretation = "on_regression"
            strength = 50.0

        return IndicatorResult(name="LinearRegression", value=current_linreg, signal_strength=float(strength), interpretation=interpretation)


class LinearRegressionSlopeIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        slope_series = talib.LINEARREG_SLOPE(data['close'].to_numpy(dtype=np.float64), timeperiod=self.period)
        if slope_series.size == 0 or pd.isna(slope_series[-1]):
            return IndicatorResult(name="LinearRegressionSlope", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_slope = float(slope_series[-1])

        if current_slope > 0:
            interpretation = "uptrend"
            strength = min(abs(current_slope) * 100, 100.0)
        elif current_slope < 0:
            interpretation = "downtrend"
            strength = min(abs(current_slope) * 100, 100.0)
        else:
            interpretation = "flat"
            strength = 0.0

        return IndicatorResult(name="LinearRegressionSlope", value=current_slope, signal_strength=float(strength), interpretation=interpretation)


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


class HullMovingAverageIndicator(TechnicalIndicator):
    def __init__(self, period: int = 9):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        hma_series = ta.hma(close=data['close'], length=self.period)
        if hma_series is None or hma_series.empty or pd.isna(hma_series.iloc[-1]):
            return IndicatorResult(name="HMA", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_hma = float(hma_series.iloc[-1])
        current_price = float(data['close'].iloc[-1])

        interpretation = "price_above_hma" if current_price > current_hma else "price_below_hma"
        strength = min(abs(current_price - current_hma) / current_hma * 100, 100.0) if current_hma > 0 else 0.0

        return IndicatorResult(name="HMA", value=current_hma, signal_strength=float(strength), interpretation=interpretation)


class ZLEMAIndicator(TechnicalIndicator):
    def __init__(self, period: int = 26):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        zlema_series = ta.zlma(close=data['close'], length=self.period)
        if zlema_series is None or zlema_series.empty or pd.isna(zlema_series.iloc[-1]):
            return IndicatorResult(name="ZLEMA", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_zlema = float(zlema_series.iloc[-1])
        current_price = float(data['close'].iloc[-1])

        interpretation = "price_above_zlema" if current_price > current_zlema else "price_below_zlema"
        strength = min(abs(current_price - current_zlema) / current_zlema * 100, 100.0) if current_zlema > 0 else 0.0

        return IndicatorResult(name="ZLEMA", value=current_zlema, signal_strength=float(strength), interpretation=interpretation)


class KAMAIndicator(TechnicalIndicator):
    def __init__(self, period: int = 10):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        kama_series = kama(close=data['close'], length=self.period)
        if kama_series is None or kama_series.empty or pd.isna(kama_series.iloc[-1]):
            return IndicatorResult(name="KAMA", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_kama = float(kama_series.iloc[-1])
        current_price = float(data['close'].iloc[-1])

        interpretation = "price_above_kama" if current_price > current_kama else "price_below_kama"
        strength = min(abs(current_price - current_kama) / current_kama * 100, 100.0) if current_kama > 0 else 0.0

        return IndicatorResult(name="KAMA", value=current_kama, signal_strength=float(strength), interpretation=interpretation)


class T3Indicator(TechnicalIndicator):
    def __init__(self, period: int = 5):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        t3_series = talib.T3(data['close'].to_numpy(dtype=np.float64), timeperiod=self.period)
        if t3_series.size == 0 or pd.isna(t3_series[-1]):
            return IndicatorResult(name="T3", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_t3 = float(t3_series[-1])
        current_price = float(data['close'].iloc[-1])

        interpretation = "price_above_t3" if current_price > current_t3 else "price_below_t3"
        strength = min(abs(current_price - current_t3) / current_t3 * 100, 100.0) if current_t3 > 0 else 0.0

        return IndicatorResult(name="T3", value=current_t3, signal_strength=float(strength), interpretation=interpretation)


class DEMAIndicator(TechnicalIndicator):
    def __init__(self, period: int = 30):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        dema_series = talib.DEMA(data['close'].to_numpy(dtype=np.float64), timeperiod=self.period)
        if dema_series.size == 0 or pd.isna(dema_series[-1]):
            return IndicatorResult(name="DEMA", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_dema = float(dema_series[-1])
        current_price = float(data['close'].iloc[-1])

        interpretation = "price_above_dema" if current_price > current_dema else "price_below_dema"
        strength = min(abs(current_price - current_dema) / current_dema * 100, 100.0) if current_dema > 0 else 0.0

        return IndicatorResult(name="DEMA", value=current_dema, signal_strength=float(strength), interpretation=interpretation)


class TEMAIndicator(TechnicalIndicator):
    def __init__(self, period: int = 30):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        tema_series = talib.TEMA(data['close'].to_numpy(dtype=np.float64), timeperiod=self.period)
        if tema_series.size == 0 or pd.isna(tema_series[-1]):
            return IndicatorResult(name="TEMA", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_tema = float(tema_series[-1])
        current_price = float(data['close'].iloc[-1])

        interpretation = "price_above_tema" if current_price > current_tema else "price_below_tema"
        strength = min(abs(current_price - current_tema) / current_tema * 100, 100.0) if current_tema > 0 else 0.0

        return IndicatorResult(name="TEMA", value=current_tema, signal_strength=float(strength), interpretation=interpretation)


class FisherTransformIndicator(TechnicalIndicator):
    def __init__(self, period: int = 9):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        fisher_df = ta.fisher(high=data['high'], low=data['low'], length=self.period)
        if fisher_df is None or fisher_df.empty or fisher_df.iloc[-1].isna().any():
            return IndicatorResult(name="FisherTransform", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        fisher_value = float(fisher_df.iloc[-1, 0])
        fisher_signal = float(fisher_df.iloc[-1, 1])

        if fisher_value > fisher_signal and fisher_value > 0:
            interpretation = "bullish"
            strength = min(abs(fisher_value) * 50, 100.0)
        elif fisher_value < fisher_signal and fisher_value < 0:
            interpretation = "bearish"
            strength = min(abs(fisher_value) * 50, 100.0)
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult(name="FisherTransform", value=fisher_value, signal_strength=float(strength), interpretation=interpretation)


class SchaffTrendCycleIndicator(TechnicalIndicator):
    def __init__(self, fast: int = 23, slow: int = 50, cycle: int = 10):
        self.fast = fast
        self.slow = slow
        self.cycle = cycle

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        try:
            if len(data) < max(self.fast, self.slow, self.cycle):
                return IndicatorResult(name="STC", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")
            
            stc_series = ta.stc(close=data['close'], fast=self.fast, slow=self.slow, cycle=self.cycle)
            
            if stc_series is None or stc_series.empty:
                return IndicatorResult(name="STC", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")
            
            stc_series_values = stc_series.iloc[:, 0]
            if stc_series_values.isna().all():
                 return IndicatorResult(name="STC", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")
            
            last_val = stc_series_values.dropna().iloc[-1]
            current_stc = float(last_val)

            if current_stc > 75:
                interpretation = "overbought"
                signal_strength = min((current_stc - 75) / 25 * 100, 100.0)
            elif current_stc < 25:
                interpretation = "oversold"
                signal_strength = min((25 - current_stc) / 25 * 100, 100.0)
            else:
                interpretation = "neutral"
                signal_strength = 50.0

            return IndicatorResult(name="STC", value=current_stc, signal_strength=float(signal_strength), interpretation=interpretation)
        except Exception as e:
            logger.warning(f"Error calculating STC: {e}")
            return IndicatorResult(name="STC", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")


class QQEIndicator(TechnicalIndicator):
    def __init__(self, rsi_period: int = 14, sf: int = 5, wilders_period: int = 27):
        self.rsi_period = rsi_period
        self.sf = sf
        self.wilders_period = wilders_period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < max(self.rsi_period, self.wilders_period):
            return IndicatorResult(name="QQE", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")
        
        rsi_series = talib.RSI(data['close'].to_numpy(dtype=np.float64), timeperiod=self.rsi_period)
        if rsi_series.size == 0 or pd.isna(rsi_series[-1]):
            return IndicatorResult(name="QQE", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        rsi_ma = talib.EMA(rsi_series, timeperiod=self.sf)
        
        if rsi_ma.size < 2:
            return IndicatorResult(name="QQE", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")
        
        atr_rsi = np.abs(np.diff(rsi_series))
        ma_atr_rsi = talib.EMA(np.concatenate([[0], atr_rsi]), timeperiod=self.wilders_period)
        
        if ma_atr_rsi.size == 0 or pd.isna(ma_atr_rsi[-1]):
            return IndicatorResult(name="QQE", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")
        
        dar = ma_atr_rsi * 4.236
        
        current_rsi = float(rsi_series[-1])
        current_rsi_ma = float(rsi_ma[-1])
        current_dar = float(dar[-1])
        
        upper_band = current_rsi_ma + current_dar
        lower_band = current_rsi_ma - current_dar
        
        if current_rsi > upper_band:
            interpretation = "overbought"
            strength = min((current_rsi - 50) * 2, 100.0)
        elif current_rsi < lower_band:
            interpretation = "oversold"
            strength = min((50 - current_rsi) * 2, 100.0)
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult(name="QQE", value=current_rsi, signal_strength=float(strength), interpretation=interpretation)


class ConnorsRSIIndicator(TechnicalIndicator):
    def __init__(self, rsi_period: int = 3, streak_period: int = 2, pct_rank_period: int = 100):
        self.rsi_period = rsi_period
        self.streak_period = streak_period
        self.pct_rank_period = pct_rank_period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.pct_rank_period:
            return IndicatorResult(name="ConnorsRSI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        rsi = talib.RSI(data['close'].to_numpy(dtype=np.float64), timeperiod=self.rsi_period)
        if rsi.size == 0 or pd.isna(rsi[-1]):
            return IndicatorResult(name="ConnorsRSI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        close_array = data['close'].to_numpy()
        close_changes = np.diff(close_array, prepend=close_array[0])
        streak = np.zeros(len(close_changes))
        current_streak = 0
        
        for i in range(1, len(close_changes)):
            if close_changes[i] > 0:
                current_streak = current_streak + 1 if current_streak > 0 else 1
            elif close_changes[i] < 0:
                current_streak = current_streak - 1 if current_streak < 0 else -1
            else:
                current_streak = 0
            streak[i] = current_streak

        streak_rsi = talib.RSI(streak.astype(np.float64), timeperiod=self.streak_period)
        
        if streak_rsi.size == 0 or pd.isna(streak_rsi[-1]):
            return IndicatorResult(name="ConnorsRSI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")
        
        pct_rank = pd.Series(close_changes).rolling(self.pct_rank_period).apply(
            lambda x: (x < x.iloc[-1]).sum() / len(x) * 100 if len(x) > 0 else 50.0, raw=False
        )

        if pd.isna(pct_rank.iloc[-1]):
            return IndicatorResult(name="ConnorsRSI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        crsi = (float(rsi[-1]) + float(streak_rsi[-1]) + float(pct_rank.iloc[-1])) / 3

        if crsi > 70:
            interpretation = "overbought"
            strength = min((crsi - 70) / 30 * 100, 100.0)
        elif crsi < 30:
            interpretation = "oversold"
            strength = min((30 - crsi) / 30 * 100, 100.0)
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult(name="ConnorsRSI", value=crsi, signal_strength=float(strength), interpretation=interpretation)


class StochasticMomentumIndexIndicator(TechnicalIndicator):
    def __init__(self, k_period: int = 10, d_period: int = 3, ema_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period
        self.ema_period = ema_period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.k_period + self.ema_period:
            return IndicatorResult(name="SMI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        high_low = data['high'].rolling(self.k_period).max() - data['low'].rolling(self.k_period).min()
        high_close = data['high'].rolling(self.k_period).max() - data['close']
        close_low = data['close'] - data['low'].rolling(self.k_period).min()
        
        hl_ema1 = high_low.ewm(span=self.ema_period, adjust=False).mean()
        hl_ema2 = hl_ema1.ewm(span=self.ema_period, adjust=False).mean()
        
        cl_diff = close_low - high_close
        cl_ema1 = cl_diff.ewm(span=self.ema_period, adjust=False).mean()
        cl_ema2 = cl_ema1.ewm(span=self.ema_period, adjust=False).mean()
        
        smi = pd.Series(np.where(hl_ema2 != 0, (cl_ema2 / (hl_ema2 / 2)) * 100, 0), index=data.index)
        smi_signal = smi.ewm(span=self.d_period, adjust=False).mean()
        
        if pd.isna(smi.iloc[-1]) or pd.isna(smi_signal.iloc[-1]):
            return IndicatorResult(name="SMI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_smi = float(smi.iloc[-1])
        current_signal = float(smi_signal.iloc[-1])

        if current_smi > 40:
            interpretation = "overbought"
            strength = min(abs(current_smi), 100.0)
        elif current_smi < -40:
            interpretation = "oversold"
            strength = min(abs(current_smi), 100.0)
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult(name="SMI", value=current_smi, signal_strength=float(strength), interpretation=interpretation)


class TSIIndicator(TechnicalIndicator):
    def __init__(self, long_period: int = 25, short_period: int = 13, signal_period: int = 13):
        self.long_period = long_period
        self.short_period = short_period
        self.signal_period = signal_period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        tsi_df = ta.tsi(close=data['close'], fast=self.short_period, slow=self.long_period, signal=self.signal_period)
        if tsi_df is None or tsi_df.empty or tsi_df.iloc[-1].isna().any():
            return IndicatorResult(name="TSI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        tsi_value = float(tsi_df.iloc[-1, 0])
        tsi_signal = float(tsi_df.iloc[-1, 1])

        if tsi_value > tsi_signal and tsi_value > 0:
            interpretation = "bullish"
            strength = min(abs(tsi_value) * 5, 100.0)
        elif tsi_value < tsi_signal and tsi_value < 0:
            interpretation = "bearish"
            strength = min(abs(tsi_value) * 5, 100.0)
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult(name="TSI", value=tsi_value, signal_strength=float(strength), interpretation=interpretation)


class GannHiLoActivatorIndicator(TechnicalIndicator):
    def __init__(self, period: int = 10):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            return IndicatorResult(name="GannHiLo", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        high_ma = data['high'].rolling(self.period).max()
        low_ma = data['low'].rolling(self.period).min()
        
        if pd.isna(high_ma.iloc[-1]) or pd.isna(low_ma.iloc[-1]):
            return IndicatorResult(name="GannHiLo", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_close = float(data['close'].iloc[-1])
        current_high = float(high_ma.iloc[-1])
        current_low = float(low_ma.iloc[-1])

        if current_close > current_high:
            interpretation = "bullish_trend"
            strength = 100.0
            value = current_high
        elif current_close < current_low:
            interpretation = "bearish_trend"
            strength = 100.0
            value = current_low
        else:
            interpretation = "neutral"
            strength = 50.0
            value = (current_high + current_low) / 2

        return IndicatorResult(name="GannHiLo", value=value, signal_strength=strength, interpretation=interpretation)


class MovingAverageRibbonIndicator(TechnicalIndicator):
    def __init__(self, periods: list = None):
        self.periods = periods if periods else [5, 8, 13, 21, 34, 55, 89, 144, 233]

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        close_np = data['close'].to_numpy(dtype=np.float64)
        mas = []
        
        for period in self.periods:
            ma = talib.EMA(close_np, timeperiod=period)
            if ma.size > 0 and not pd.isna(ma[-1]):
                mas.append(float(ma[-1]))
        
        if len(mas) < len(self.periods):
            return IndicatorResult(name="MA_Ribbon", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_price = float(data['close'].iloc[-1])
        
        above_count = sum(1 for ma in mas if current_price > ma)
        total_count = len(mas)
        
        above_ratio = above_count / total_count

        if above_ratio >= 0.7:
            interpretation = "strong_uptrend"
            strength = above_ratio * 100
        elif above_ratio <= 0.3:
            interpretation = "strong_downtrend"
            strength = (1 - above_ratio) * 100
        else:
            interpretation = "ranging"
            strength = 50.0

        return IndicatorResult(name="MA_Ribbon", value=float(np.mean(mas)), signal_strength=float(strength), interpretation=interpretation)


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


class ChaikinVolatilityIndicator(TechnicalIndicator):
    def __init__(self, period: int = 10, roc_period: int = 10):
        self.period = period
        self.roc_period = roc_period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period + self.roc_period:
            return IndicatorResult(name="ChaikinVolatility", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        high_low = data['high'] - data['low']
        ema_hl = high_low.ewm(span=self.period, adjust=False).mean()
        
        chaikin_vol = ((ema_hl - ema_hl.shift(self.roc_period)) / ema_hl.shift(self.roc_period)) * 100
        
        if pd.isna(chaikin_vol.iloc[-1]):
            return IndicatorResult(name="ChaikinVolatility", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_vol = float(chaikin_vol.iloc[-1])

        if current_vol > 10:
            interpretation = "high_volatility"
            strength = min(current_vol * 5, 100.0)
        elif current_vol < -10:
            interpretation = "low_volatility"
            strength = min(abs(current_vol) * 5, 100.0)
        else:
            interpretation = "normal_volatility"
            strength = 50.0

        return IndicatorResult(name="ChaikinVolatility", value=current_vol, signal_strength=float(strength), interpretation=interpretation)


class HistoricalVolatilityIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period + 1:
            return IndicatorResult(name="HistoricalVolatility", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        log_returns = np.log(data['close'] / data['close'].shift(1))
        hv = log_returns.rolling(self.period).std() * np.sqrt(252) * 100
        
        if pd.isna(hv.iloc[-1]):
            return IndicatorResult(name="HistoricalVolatility", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_hv = float(hv.iloc[-1])
        avg_hv = float(hv.mean())

        if current_hv > avg_hv * 1.5:
            interpretation = "high_volatility"
            strength = min((current_hv / avg_hv - 1) * 100, 100.0)
        elif current_hv < avg_hv * 0.5:
            interpretation = "low_volatility"
            strength = min((1 - current_hv / avg_hv) * 100, 100.0)
        else:
            interpretation = "normal_volatility"
            strength = 50.0

        return IndicatorResult(name="HistoricalVolatility", value=current_hv, signal_strength=float(strength), interpretation=interpretation)


class UlcerIndexIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        ui_series = ta.ui(close=data['close'], length=self.period)
        if ui_series is None or ui_series.empty or pd.isna(ui_series.iloc[-1]):
            return IndicatorResult(name="UlcerIndex", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_ui = float(ui_series.iloc[-1])
        avg_ui = float(ui_series.mean())

        if current_ui > avg_ui * 1.5:
            interpretation = "high_risk"
            strength = min((current_ui / avg_ui - 1) * 100, 100.0)
        elif current_ui < avg_ui * 0.5:
            interpretation = "low_risk"
            strength = min((1 - current_ui / avg_ui) * 100, 100.0)
        else:
            interpretation = "moderate_risk"
            strength = 50.0

        return IndicatorResult(name="UlcerIndex", value=current_ui, signal_strength=float(strength), interpretation=interpretation)


class ATRBandsIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14, multiplier: float = 2.0):
        self.period = period
        self.multiplier = multiplier

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        atr_series = talib.ATR(
            data['high'].to_numpy(dtype=np.float64),
            data['low'].to_numpy(dtype=np.float64),
            data['close'].to_numpy(dtype=np.float64),
            timeperiod=self.period
        )
        
        if atr_series.size == 0 or pd.isna(atr_series[-1]):
            return IndicatorResult(name="ATRBands", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        close_np = data['close'].to_numpy(dtype=np.float64)
        ma = talib.SMA(close_np, timeperiod=self.period)
        
        if ma.size == 0 or pd.isna(ma[-1]):
            return IndicatorResult(name="ATRBands", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_atr = float(atr_series[-1])
        current_ma = float(ma[-1])
        current_price = float(data['close'].iloc[-1])
        
        upper_band = current_ma + (self.multiplier * current_atr)
        lower_band = current_ma - (self.multiplier * current_atr)

        if current_price > upper_band:
            interpretation = "above_upper_band"
            strength = min((current_price - upper_band) / upper_band * 100, 100.0)
        elif current_price < lower_band:
            interpretation = "below_lower_band"
            strength = min((lower_band - current_price) / lower_band * 100, 100.0)
        else:
            interpretation = "within_bands"
            strength = 50.0

        return IndicatorResult(name="ATRBands", value=current_ma, signal_strength=float(strength), interpretation=interpretation)


class BollingerBandwidthIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        bb_df = bbands(close=data['close'], length=self.period, std=self.std_dev)
        if bb_df is None or bb_df.empty or bb_df.dropna().empty:
            return IndicatorResult(name="BBW", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        upper = bb_df.iloc[-1, 2]
        lower = bb_df.iloc[-1, 0]
        middle = bb_df.iloc[-1, 1]
        
        if pd.isna(upper) or pd.isna(lower) or pd.isna(middle) or middle == 0:
            return IndicatorResult(name="BBW", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        bandwidth = ((float(upper) - float(lower)) / float(middle)) * 100
        
        bbw_series = ((bb_df.iloc[:, 2] - bb_df.iloc[:, 0]) / bb_df.iloc[:, 1]) * 100
        avg_bbw = float(bbw_series.mean())

        if bandwidth < avg_bbw * 0.5:
            interpretation = "squeeze"
            strength = min((1 - bandwidth / avg_bbw) * 100, 100.0)
        elif bandwidth > avg_bbw * 1.5:
            interpretation = "expansion"
            strength = min((bandwidth / avg_bbw - 1) * 100, 100.0)
        else:
            interpretation = "normal"
            strength = 50.0

        return IndicatorResult(name="BBW", value=bandwidth, signal_strength=float(strength), interpretation=interpretation)


class VolumeOscillatorIndicator(TechnicalIndicator):
    def __init__(self, fast: int = 12, slow: int = 26):
        self.fast = fast
        self.slow = slow

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if 'volume' not in data.columns or len(data) < self.slow:
            return IndicatorResult(name="VolumeOscillator", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        volume_np = data['volume'].to_numpy(dtype=np.float64)
        fast_ema = talib.EMA(volume_np, timeperiod=self.fast)
        slow_ema = talib.EMA(volume_np, timeperiod=self.slow)
        
        if fast_ema.size == 0 or slow_ema.size == 0 or pd.isna(fast_ema[-1]) or pd.isna(slow_ema[-1]) or slow_ema[-1] == 0:
            return IndicatorResult(name="VolumeOscillator", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        vo = ((float(fast_ema[-1]) - float(slow_ema[-1])) / float(slow_ema[-1])) * 100

        if vo > 5:
            interpretation = "increasing_volume"
            strength = min(vo * 10, 100.0)
        elif vo < -5:
            interpretation = "decreasing_volume"
            strength = min(abs(vo) * 10, 100.0)
        else:
            interpretation = "stable_volume"
            strength = 50.0

        return IndicatorResult(name="VolumeOscillator", value=vo, signal_strength=float(strength), interpretation=interpretation)


class KlingerVolumeOscillatorIndicator(TechnicalIndicator):
    def __init__(self, fast: int = 34, slow: int = 55, signal: int = 13):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        kvo_df = ta.kvo(high=data['high'], low=data['low'], close=data['close'], volume=data['volume'], 
                        fast=self.fast, slow=self.slow, signal=self.signal)
        if kvo_df is None or kvo_df.empty or kvo_df.iloc[-1].isna().any():
            return IndicatorResult(name="KVO", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        kvo_value = float(kvo_df.iloc[-1, 0])
        kvo_signal = float(kvo_df.iloc[-1, 1])

        if kvo_value > kvo_signal and kvo_value > 0:
            interpretation = "bullish"
            strength = min(abs(kvo_value) / 1000, 100.0)
        elif kvo_value < kvo_signal and kvo_value < 0:
            interpretation = "bearish"
            strength = min(abs(kvo_value) / 1000, 100.0)
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult(name="KVO", value=kvo_value, signal_strength=float(strength), interpretation=interpretation)


class FRAMAIndicator(TechnicalIndicator):
    def __init__(self, period: int = 16):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period * 2:
            return IndicatorResult(name="FRAMA", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        close_np = data['close'].to_numpy(dtype=np.float64)
        frama = np.zeros(len(close_np))
        frama[:] = np.nan
        
        for i in range(self.period, len(close_np)):
            window = close_np[i - self.period:i]
            half_period = self.period // 2
            n1 = window[:half_period]
            n2 = window[half_period:]
            
            hh1, ll1 = n1.max(), n1.min()
            hh2, ll2 = n2.max(), n2.min()
            hh, ll = window.max(), window.min()
            
            n1_range = (hh1 - ll1) / half_period if (hh1 - ll1) > 0 else 0
            n2_range = (hh2 - ll2) / half_period if (hh2 - ll2) > 0 else 0
            n_range = (hh - ll) / self.period if (hh - ll) > 0 else 0
            
            if n_range > 0 and (n1_range + n2_range) > 0:
                dimension = (np.log(n1_range + n2_range) - np.log(n_range)) / np.log(2)
                alpha = np.exp(-4.6 * (dimension - 1))
                alpha = np.clip(alpha, 0.01, 1.0)
            else:
                alpha = 0.5
            
            if i == self.period:
                frama[i] = close_np[i]
            else:
                if not np.isnan(frama[i - 1]):
                    frama[i] = alpha * close_np[i] + (1 - alpha) * frama[i - 1]
                else:
                    frama[i] = close_np[i]

        if pd.isna(frama[-1]):
            return IndicatorResult(name="FRAMA", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_frama = float(frama[-1])
        current_price = float(data['close'].iloc[-1])

        interpretation = "price_above_frama" if current_price > current_frama else "price_below_frama"
        strength = min(abs(current_price - current_frama) / current_frama * 100, 100.0) if current_frama > 0 else 0.0

        return IndicatorResult(name="FRAMA", value=current_frama, signal_strength=float(strength), interpretation=interpretation)


class VIDYAIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14, cmo_period: int = 9):
        self.period = period
        self.cmo_period = cmo_period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        cmo_series = cmo(close=data['close'], length=self.cmo_period)
        if cmo_series is None or cmo_series.empty or pd.isna(cmo_series.iloc[-1]):
            return IndicatorResult(name="VIDYA", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        close_np = data['close'].to_numpy(dtype=np.float64)
        cmo_np = cmo_series.to_numpy(dtype=np.float64)
        
        vidya = np.zeros(len(close_np))
        vidya[:] = np.nan
        vidya[self.period - 1] = close_np[:self.period].mean()
        
        alpha_base = 2.0 / (self.period + 1)
        
        for i in range(self.period, len(close_np)):
            if not np.isnan(cmo_np[i]) and not np.isnan(vidya[i - 1]):
                alpha = alpha_base * abs(cmo_np[i]) / 100.0
                vidya[i] = alpha * close_np[i] + (1 - alpha) * vidya[i - 1]
            elif not np.isnan(vidya[i - 1]):
                vidya[i] = vidya[i - 1]
            else:
                vidya[i] = close_np[i]

        if pd.isna(vidya[-1]):
            return IndicatorResult(name="VIDYA", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_vidya = float(vidya[-1])
        current_price = float(data['close'].iloc[-1])

        interpretation = "price_above_vidya" if current_price > current_vidya else "price_below_vidya"
        strength = min(abs(current_price - current_vidya) / current_vidya * 100, 100.0) if current_vidya > 0 else 0.0

        return IndicatorResult(name="VIDYA", value=current_vidya, signal_strength=float(strength), interpretation=interpretation)


class MAMAIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        mama, fama = talib.MAMA(data['close'].to_numpy(dtype=np.float64))
        if mama.size == 0 or pd.isna(mama[-1]) or pd.isna(fama[-1]):
            return IndicatorResult(name="MAMA", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_mama = float(mama[-1])
        current_fama = float(fama[-1])
        current_price = float(data['close'].iloc[-1])

        if current_mama > current_fama:
            interpretation = "bullish_trend"
            strength = 100.0
        else:
            interpretation = "bearish_trend"
            strength = 100.0

        return IndicatorResult(name="MAMA", value=current_mama, signal_strength=strength, interpretation=interpretation)


class RMIIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14, momentum: int = 5):
        self.period = period
        self.momentum = momentum

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period + self.momentum:
            return IndicatorResult(name="RMI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        close_np = data['close'].to_numpy(dtype=np.float64)
        momentum_changes = close_np[self.momentum:] - close_np[:-self.momentum]
        
        gains = np.where(momentum_changes > 0, momentum_changes, 0)
        losses = np.where(momentum_changes < 0, abs(momentum_changes), 0)
        
        pad_length = len(close_np) - len(gains)
        gains_padded = np.concatenate([np.zeros(pad_length), gains])
        losses_padded = np.concatenate([np.zeros(pad_length), losses])
        
        avg_gain = talib.EMA(gains_padded, timeperiod=self.period)
        avg_loss = talib.EMA(losses_padded, timeperiod=self.period)
        
        if avg_gain.size == 0 or avg_loss.size == 0 or pd.isna(avg_gain[-1]) or pd.isna(avg_loss[-1]):
            return IndicatorResult(name="RMI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")
        
        if avg_loss[-1] == 0:
            rmi = 100.0
        else:
            rs = avg_gain[-1] / avg_loss[-1]
            rmi = 100 - (100 / (1 + rs))

        if np.isnan(rmi):
            return IndicatorResult(name="RMI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        if rmi > 70:
            interpretation = "overbought"
            signal_strength = min((rmi - 70) / 30 * 100, 100)
        elif rmi < 30:
            interpretation = "oversold"
            signal_strength = min((30 - rmi) / 30 * 100, 100)
        else:
            interpretation = "neutral"
            signal_strength = 50.0

        return IndicatorResult(name="RMI", value=float(rmi), signal_strength=float(signal_strength), interpretation=interpretation)


class PPOIndicator(TechnicalIndicator):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        ppo_df = ta.ppo(close=data['close'], fast=self.fast, slow=self.slow, signal=self.signal_period)
        if ppo_df is None or ppo_df.empty or ppo_df.iloc[-1].isna().any():
            return IndicatorResult(name="PPO", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        ppo_value = float(ppo_df.iloc[-1, 0])
        ppo_signal = float(ppo_df.iloc[-1, 1])
        ppo_hist = float(ppo_df.iloc[-1, 2])

        if ppo_value > ppo_signal and ppo_hist > 0:
            interpretation = "bullish_crossover"
            signal_strength = min(abs(ppo_hist) * 100, 100)
        elif ppo_value < ppo_signal and ppo_hist < 0:
            interpretation = "bearish_crossover"
            signal_strength = min(abs(ppo_hist) * 100, 100)
        else:
            interpretation = "neutral"
            signal_strength = 50.0

        return IndicatorResult(name="PPO", value=ppo_value, signal_strength=float(signal_strength), interpretation=interpretation)


class PVOIndicator(TechnicalIndicator):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        pvo_df = ta.pvo(volume=data['volume'], fast=self.fast, slow=self.slow, signal=self.signal_period)
        if pvo_df is None or pvo_df.empty or pvo_df.iloc[-1].isna().any():
            return IndicatorResult(name="PVO", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        pvo_value = float(pvo_df.iloc[-1, 0])
        pvo_signal = float(pvo_df.iloc[-1, 1])
        pvo_hist = float(pvo_df.iloc[-1, 2])

        if pvo_value > pvo_signal and pvo_hist > 0:
            interpretation = "bullish_volume"
            signal_strength = min(abs(pvo_hist) * 10, 100)
        elif pvo_value < pvo_signal and pvo_hist < 0:
            interpretation = "bearish_volume"
            signal_strength = min(abs(pvo_hist) * 10, 100)
        else:
            interpretation = "neutral"
            signal_strength = 50.0

        return IndicatorResult(name="PVO", value=pvo_value, signal_strength=float(signal_strength), interpretation=interpretation)


class NVIIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        nvi_series = ta.nvi(close=data['close'], volume=data['volume'])
        if nvi_series is None or nvi_series.empty or len(nvi_series) < 2:
            return IndicatorResult(name="NVI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_value = float(nvi_series.iloc[-1])
        previous_value = float(nvi_series.iloc[-2])

        if current_value > previous_value:
            interpretation = "smart_money_buying"
            strength = 100.0
        elif current_value < previous_value:
            interpretation = "smart_money_selling"
            strength = 100.0
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult(name="NVI", value=current_value, signal_strength=strength, interpretation=interpretation)


class PVIIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        try:
            pvi_series = ta.pvi(close=data['close'], volume=data['volume'])
            if pvi_series is None or (isinstance(pvi_series, pd.Series) and pvi_series.empty):
                return IndicatorResult(name="PVI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")
            
            if isinstance(pvi_series, pd.DataFrame):
                if pvi_series.empty:
                    return IndicatorResult(name="PVI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")
                pvi_series = pvi_series.iloc[:, 0]
            
            valid_pvi = pvi_series.dropna()
            if len(valid_pvi) < 2:
                return IndicatorResult(name="PVI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

            current_value = float(valid_pvi.iloc[-1])
            previous_value = float(valid_pvi.iloc[-2])

            if current_value > previous_value:
                interpretation = "crowd_buying"
                strength = 100.0
            elif current_value < previous_value:
                interpretation = "crowd_selling"
                strength = 100.0
            else:
                interpretation = "neutral"
                strength = 50.0

            return IndicatorResult(name="PVI", value=current_value, signal_strength=strength, interpretation=interpretation)
        except Exception as e:
            logger.warning(f"Error calculating PVI: {e}")
            return IndicatorResult(name="PVI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")


class MFIBillWilliamsIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 2:
            return IndicatorResult(name="MFI_BW", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_high = float(data['high'].iloc[-1])
        current_low = float(data['low'].iloc[-1])
        current_volume = float(data['volume'].iloc[-1])
        prev_high = float(data['high'].iloc[-2])
        prev_low = float(data['low'].iloc[-2])
        prev_volume = float(data['volume'].iloc[-2])

        current_range = current_high - current_low
        prev_range = prev_high - prev_low

        if current_volume == 0:
            mfi = 0.0
        else:
            mfi = current_range / current_volume * 1000000

        if current_volume > prev_volume and current_range > prev_range:
            interpretation = "green_bar_increasing"
            strength = 100.0
        elif current_volume < prev_volume and current_range < prev_range:
            interpretation = "red_bar_decreasing"
            strength = 100.0
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult(name="MFI_BW", value=mfi, signal_strength=strength, interpretation=interpretation)


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


class CoppockCurveIndicator(TechnicalIndicator):
    def __init__(self, roc1: int = 14, roc2: int = 11, wma: int = 10):
        self.roc1 = roc1
        self.roc2 = roc2
        self.wma = wma

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < max(self.roc1, self.roc2) + self.wma:
            return IndicatorResult(name="Coppock", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        close_np = data['close'].to_numpy(dtype=np.float64)
        roc1_series = talib.ROC(close_np, timeperiod=self.roc1)
        roc2_series = talib.ROC(close_np, timeperiod=self.roc2)
        
        roc_sum = roc1_series + roc2_series
        coppock = talib.WMA(roc_sum, timeperiod=self.wma)

        if coppock.size == 0 or pd.isna(coppock[-1]):
            return IndicatorResult(name="Coppock", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_coppock = float(coppock[-1])

        if len(coppock) > 1:
            prev_coppock = float(coppock[-2])
            if current_coppock > 0 and prev_coppock < 0:
                interpretation = "bullish_signal"
                strength = 100.0
            elif current_coppock < 0 and prev_coppock > 0:
                interpretation = "bearish_signal"
                strength = 100.0
            elif current_coppock > 0:
                interpretation = "bullish"
                strength = min(abs(current_coppock) * 10, 100.0)
            else:
                interpretation = "bearish"
                strength = min(abs(current_coppock) * 10, 100.0)
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult(name="Coppock", value=current_coppock, signal_strength=float(strength), interpretation=interpretation)


class RSI2Indicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        rsi_series = talib.RSI(data['close'].to_numpy(dtype=np.float64), timeperiod=2)
        if rsi_series.size == 0 or pd.isna(rsi_series[-1]):
            return IndicatorResult(name="RSI2", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_rsi = float(rsi_series[-1])

        if current_rsi > 95:
            interpretation = "extreme_overbought"
            signal_strength = 100.0
        elif current_rsi < 5:
            interpretation = "extreme_oversold"
            signal_strength = 100.0
        elif current_rsi > 70:
            interpretation = "overbought"
            signal_strength = min((current_rsi - 70) / 30 * 100, 100)
        elif current_rsi < 30:
            interpretation = "oversold"
            signal_strength = min((30 - current_rsi) / 30 * 100, 100)
        else:
            interpretation = "neutral"
            signal_strength = 50.0

        return IndicatorResult(name="RSI2", value=current_rsi, signal_strength=float(signal_strength), interpretation=interpretation)


def get_all_indicators() -> Dict[str, TechnicalIndicator]:
    return {
        'sma_20': MovingAverageIndicator(20, "sma"), 'sma_50': MovingAverageIndicator(50, "sma"),
        'ema_12': MovingAverageIndicator(12, "ema"), 'ema_26': MovingAverageIndicator(26, "ema"),
        'rsi': RSIIndicator(), 'macd': MACDIndicator(), 'bb': BollingerBandsIndicator(),
        'stoch': StochasticIndicator(), 'volume': VolumeIndicator(), 'atr': ATRIndicator(),
        'ichimoku': IchimokuIndicator(), 'williams_r': WilliamsRIndicator(), 'cci': CCIIndicator(),
        'supertrend': SuperTrendIndicator(), 'adx': ADXIndicator(), 'cmf': ChaikinMoneyFlowIndicator(),
        'obv': OBVIndicator(), 'squeeze': SqueezeMomentumIndicator(), 'psar': ParabolicSARIndicator(),
        'vwap': VWAPIndicator(), 'mfi': MoneyFlowIndexIndicator(), 'aroon': AroonIndicator(),
        'uo': UltimateOscillatorIndicator(), 'roc': ROCIndicator(), 'ad_line': ADLineIndicator(),
        'force_index': ForceIndexIndicator(), 'vwma': VWMAIndicator(), 'keltner': KeltnerChannelsIndicator(),
        'donchian': DonchianChannelsIndicator(), 'trix': TRIXIndicator(), 'eom': EaseOfMovementIndicator(),
        'std_dev': StandardDeviationIndicator(), 'stochrsi': StochRSIIndicator(), 
        'kst': KSTIndicator(), 'mass': MassIndexIndicator(), 'corr_coef': CorrelationCoefficientIndicator(),
        'elder_ray': ElderRayIndexIndicator(), 'pivot': PivotPointsIndicator(), 'momentum': MomentumIndicator(),
        'dpo': DetrendedPriceOscillatorIndicator(), 'choppiness': ChoppinessIndexIndicator(),
        'vortex': VortexIndicator(), 'awesome': AwesomeOscillatorIndicator(), 'cmo': ChandeMomentumOscillatorIndicator(),
        'rvi': RelativeVigorIndexIndicator(), 'pvr': PriceVolumeRankIndicator(), 
        'ado': AccumulationDistributionOscillatorIndicator(), 'pvt': PriceVolumeTrendIndicator(),
        'bop': BalanceOfPowerIndicator(), 'linreg': LinearRegressionIndicator(), 
        'linreg_slope': LinearRegressionSlopeIndicator(), 'median_price': MedianPriceIndicator(),
        'typical_price': TypicalPriceIndicator(), 'weighted_close': WeightedClosePriceIndicator(),
        'hma': HullMovingAverageIndicator(), 'zlema': ZLEMAIndicator(), 'kama': KAMAIndicator(),
        't3': T3Indicator(), 'dema': DEMAIndicator(), 'tema': TEMAIndicator(),
        'fisher': FisherTransformIndicator(), 'stc': SchaffTrendCycleIndicator(),
        'qqe': QQEIndicator(), 'connors_rsi': ConnorsRSIIndicator(), 'smi': StochasticMomentumIndexIndicator(),
        'tsi': TSIIndicator(), 'gann_hilo': GannHiLoActivatorIndicator(), 'ma_ribbon': MovingAverageRibbonIndicator(),
        'fractal': FractalIndicator(), 'chaikin_vol': ChaikinVolatilityIndicator(), 
        'historical_vol': HistoricalVolatilityIndicator(), 'ulcer_index': UlcerIndexIndicator(),
        'atr_bands': ATRBandsIndicator(), 'bbw': BollingerBandwidthIndicator(),
        'volume_osc': VolumeOscillatorIndicator(), 'kvo': KlingerVolumeOscillatorIndicator(),
        'frama': FRAMAIndicator(), 'vidya': VIDYAIndicator(), 'mama': MAMAIndicator(),
        'rmi': RMIIndicator(), 'rsi2': RSI2Indicator(), 'ppo': PPOIndicator(), 'pvo': PVOIndicator(),
        'nvi': NVIIndicator(), 'pvi': PVIIndicator(), 'mfi_bw': MFIBillWilliamsIndicator(),
        'ht_dc': HilbertDominantCycleIndicator(), 'ht_trend_mode': HilbertTrendVsCycleModeIndicator(),
        'er': KaufmanEfficiencyRatioIndicator(), 'coppock': CoppockCurveIndicator()
    }

