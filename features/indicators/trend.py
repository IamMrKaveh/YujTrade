import numpy as np
import pandas as pd
import talib

import pandas_ta as ta
from pandas_ta.overlap import ichimoku, supertrend, kama
from pandas_ta.trend import adx, aroon

from ...common.core import IndicatorResult
from .base import TechnicalIndicator

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

        adx_val = float(adx_series[-1])

        if adx_val > 25:
            interpretation = "strong_trend"
        else:
            interpretation = "weak_trend"
        strength = min(adx_val, 100.0)

        return IndicatorResult(name="ADX", value=adx_val, signal_strength=strength, interpretation=interpretation)


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
        from pandas_ta.momentum import cmo
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

        if current_mama > current_fama:
            interpretation = "bullish_trend"
            strength = 100.0
        else:
            interpretation = "bearish_trend"
            strength = 100.0

        return IndicatorResult(name="MAMA", value=current_mama, signal_strength=strength, interpretation=interpretation)


class MarketStructureIndicator(TechnicalIndicator):
    def __init__(self, swing_period: int = 10):
        self.swing_period = swing_period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.swing_period * 3:
            return IndicatorResult(name="MarketStructure", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        highs = data['high'].rolling(window=self.swing_period, center=True).max()
        lows = data['low'].rolling(window=self.swing_period, center=True).min()
        
        swing_highs = (data['high'] == highs).astype(int)
        swing_lows = (data['low'] == lows).astype(int)
        
        recent_swing_highs = swing_highs.tail(self.swing_period * 2)
        recent_swing_lows = swing_lows.tail(self.swing_period * 2)
        
        high_values = data.loc[recent_swing_highs[recent_swing_highs == 1].index, 'high']
        low_values = data.loc[recent_swing_lows[recent_swing_lows == 1].index, 'low']
        
        if len(high_values) < 2 or len(low_values) < 2:
            return IndicatorResult(name="MarketStructure", value=50.0, signal_strength=50.0, interpretation="insufficient_swings")

        higher_highs = sum(high_values.diff() > 0)
        lower_lows = sum(low_values.diff() < 0)
        
        total_swings = len(high_values) + len(low_values) - 2
        
        if total_swings == 0:
            return IndicatorResult(name="MarketStructure", value=50.0, signal_strength=50.0, interpretation="neutral_structure")

        bullish_ratio = (higher_highs + (len(low_values) - 1 - lower_lows)) / total_swings
        
        if bullish_ratio > 0.7:
            interpretation = "uptrend_structure"
            strength = min(bullish_ratio * 100, 100.0)
        elif bullish_ratio < 0.3:
            interpretation = "downtrend_structure"
            strength = min((1 - bullish_ratio) * 100, 100.0)
        else:
            interpretation = "ranging_structure"
            strength = 50.0

        return IndicatorResult(name="MarketStructure", value=float(bullish_ratio * 100), signal_strength=float(strength), interpretation=interpretation)


class TrendIntensityIndexIndicator(TechnicalIndicator):
    def __init__(self, period: int = 30):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            return IndicatorResult(name="TII", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        price_changes = data['close'].diff()
        
        up_moves = price_changes.apply(lambda x: abs(x) if x > 0 else 0)
        down_moves = price_changes.apply(lambda x: abs(x) if x < 0 else 0)
        
        sum_up = up_moves.rolling(window=self.period).sum()
        sum_down = down_moves.rolling(window=self.period).sum()
        
        if pd.isna(sum_up.iloc[-1]) or pd.isna(sum_down.iloc[-1]):
            return IndicatorResult(name="TII", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

        total_movement = sum_up.iloc[-1] + sum_down.iloc[-1]
        
        if total_movement == 0:
            tii = 50.0
        else:
            tii = (sum_up.iloc[-1] / total_movement) * 100

        if tii > 65:
            interpretation = "strong_uptrend"
            strength = min((tii - 50) * 2, 100.0)
        elif tii < 35:
            interpretation = "strong_downtrend"
            strength = min((50 - tii) * 2, 100.0)
        else:
            interpretation = "weak_trend"
            strength = 40.0

        return IndicatorResult(name="TII", value=float(tii), signal_strength=float(strength), interpretation=interpretation)