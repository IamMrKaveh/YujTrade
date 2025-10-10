import numpy as np
import pandas as pd
import talib

import pandas_ta as ta
from pandas_ta.volatility import bbands, massi

from ..core import IndicatorResult
from .base import TechnicalIndicator

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


