import numpy as np
import pandas as pd
import talib

import pandas_ta as ta
from pandas_ta.volume import cmf, obv, ad, eom, efi, pvt

from ...common.core import IndicatorResult
from .base import TechnicalIndicator
from ...config.logger import logger

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


class VolumeWeightedRSIIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            return IndicatorResult(name="VW_RSI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        price_changes = data['close'].diff()
        volume = data['volume']
        
        gains = price_changes.apply(lambda x: x if x > 0 else 0) * volume
        losses = price_changes.apply(lambda x: abs(x) if x < 0 else 0) * volume
        
        avg_gain = gains.rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()
        avg_loss = losses.rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()
        
        if pd.isna(avg_gain.iloc[-1]) or pd.isna(avg_loss.iloc[-1]) or avg_loss.iloc[-1] == 0:
            return IndicatorResult(name="VW_RSI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
        vw_rsi = 100 - (100 / (1 + rs))

        if vw_rsi > 70:
            interpretation = "volume_overbought"
            signal_strength = min((vw_rsi - 70) / 30 * 100, 100)
        elif vw_rsi < 30:
            interpretation = "volume_oversold"
            signal_strength = min((30 - vw_rsi) / 30 * 100, 100)
        else:
            interpretation = "volume_neutral"
            signal_strength = 50.0

        return IndicatorResult(name="VW_RSI", value=float(vw_rsi), signal_strength=float(signal_strength), interpretation=interpretation)


class AccumulationDistributionIndexIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            return IndicatorResult(name="ADI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        ad_values = []
        cumulative_ad = 0.0
        
        for i in range(len(data)):
            high = float(data['high'].iloc[i])
            low = float(data['low'].iloc[i])
            close = float(data['close'].iloc[i])
            volume = float(data['volume'].iloc[i])
            
            if high != low:
                mfm = ((close - low) - (high - close)) / (high - low)
                mfv = mfm * volume
                cumulative_ad += mfv
            
            ad_values.append(cumulative_ad)

        ad_series = pd.Series(ad_values, index=data.index)
        ad_ma = ad_series.rolling(window=self.period).mean()
        
        if pd.isna(ad_series.iloc[-1]) or pd.isna(ad_ma.iloc[-1]):
            return IndicatorResult(name="ADI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current_ad = float(ad_series.iloc[-1])
        current_ma = float(ad_ma.iloc[-1])
        
        if current_ad > current_ma:
            interpretation = "accumulation_trend"
            strength = min(abs(current_ad - current_ma) / abs(current_ma) * 100, 100.0) if current_ma != 0 else 50.0
        else:
            interpretation = "distribution_trend"
            strength = min(abs(current_ma - current_ad) / abs(current_ma) * 100, 100.0) if current_ma != 0 else 50.0

        return IndicatorResult(name="ADI", value=current_ad, signal_strength=float(strength), interpretation=interpretation)


class OrderFlowImbalanceIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 2:
            return IndicatorResult(name="OrderFlowImbalance", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        current = data.iloc[-1]
        
        current_delta = float(current['close']) - float(current['open'])
        current_volume = float(current['volume'])
        
        if (float(current['high']) - float(current['low'])) == 0:
             return IndicatorResult(name="OrderFlowImbalance", value=0, signal_strength=0.0, interpretation="balanced_flow")

        if current_delta > 0:
            buy_volume = current_volume * (current_delta / (float(current['high']) - float(current['low'])))
            sell_volume = current_volume - buy_volume
        else:
            sell_volume = current_volume * (abs(current_delta) / (float(current['high']) - float(current['low'])))
            buy_volume = current_volume - sell_volume

        imbalance = (buy_volume - sell_volume) / current_volume if current_volume > 0 else 0.0

        if imbalance > 0.3:
            interpretation = "strong_buying"
            strength = min(abs(imbalance) * 200, 100.0)
        elif imbalance < -0.3:
            interpretation = "strong_selling"
            strength = min(abs(imbalance) * 200, 100.0)
        else:
            interpretation = "balanced_flow"
            strength = 50.0

        return IndicatorResult(name="OrderFlowImbalance", value=float(imbalance * 100), signal_strength=float(strength), interpretation=interpretation)


class BalanceOfPowerRSIIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if 'open' not in data.columns or len(data) < self.period:
            return IndicatorResult(name="BOP_RSI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        bop_values = []
        for i in range(len(data)):
            close = float(data['close'].iloc[i])
            open_price = float(data['open'].iloc[i])
            high = float(data['high'].iloc[i])
            low = float(data['low'].iloc[i])
            
            if high == low:
                bop_values.append(0.0)
            else:
                bop_values.append((close - open_price) / (high - low))

        bop_series = pd.Series(bop_values, index=data.index)
        
        gains = bop_series.diff().apply(lambda x: x if x > 0 else 0)
        losses = bop_series.diff().apply(lambda x: abs(x) if x < 0 else 0)
        
        avg_gain = gains.rolling(window=self.period).mean()
        avg_loss = losses.rolling(window=self.period).mean()
        
        if pd.isna(avg_gain.iloc[-1]) or pd.isna(avg_loss.iloc[-1]) or avg_loss.iloc[-1] == 0:
            return IndicatorResult(name="BOP_RSI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
        bop_rsi = 100 - (100 / (1 + rs))

        if bop_rsi > 70:
            interpretation = "strong_buying_pressure"
            strength = min((bop_rsi - 70) / 30 * 100, 100.0)
        elif bop_rsi < 30:
            interpretation = "strong_selling_pressure"
            strength = min((30 - bop_rsi) / 30 * 100, 100.0)
        else:
            interpretation = "balanced_pressure"
            strength = 50.0

        return IndicatorResult(name="BOP_RSI", value=float(bop_rsi), signal_strength=float(strength), interpretation=interpretation)