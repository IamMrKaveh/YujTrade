import numpy as np
import pandas as pd
import talib

import pandas_ta as ta
from pandas_ta.momentum import roc, stochrsi, trix, uo, squeeze, cmo

from ...common.core import IndicatorResult
from .base import TechnicalIndicator

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

        if avg_stochrsi > 80:
            interpretation = "overbought"
            signal_strength = min((avg_stochrsi - 80) / 20 * 100, 100)
        elif avg_stochrsi < 20:
            interpretation = "oversold"
            signal_strength = min((20 - avg_stochrsi) / 20 * 100, 100)
        else:
            interpretation = "neutral"
            signal_strength = 50.0

        return IndicatorResult(name="StochRSI", value=avg_stochrsi, signal_strength=float(signal_strength), interpretation=interpretation)


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
        dpo_series = ta.dpo(close=data['close'], length=self.period)
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
            from ...config.logger import logger
            logger.warning(f"Error calculating RVI: {e}")
            return IndicatorResult(name="RVI", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")


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
            from ...config.logger import logger
            logger.warning(f"Error calculating STC: {e}")
            return IndicatorResult(name="STC", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")


class QQEIndicator(TechnicalIndicator):
    def __init__(self, rsi_period: int = 14, sf: int = 5, wilders_period: int = 27):
        self.rsi_period = rsi_period
        self.sf = sf
        self.wilders_period = wilders_period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        try:
            qqe_df = ta.qqe(close=data['close'], length=self.rsi_period, smooth=self.sf, factor=self.wilders_period)
            if qqe_df is None or qqe_df.empty or qqe_df.iloc[-1].isna().any():
                return IndicatorResult(name="QQE", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")

            qqe_line = float(qqe_df.iloc[-1, 0])
            rsi_ma = float(qqe_df.iloc[-1, 1])
            
            if qqe_line > rsi_ma:
                interpretation = "bullish"
                strength = min((qqe_line - 50) * 2, 100.0) if qqe_line > 50 else 50.0
            elif qqe_line < rsi_ma:
                interpretation = "bearish"
                strength = min((50 - qqe_line) * 2, 100.0) if qqe_line < 50 else 50.0
            else:
                interpretation = "neutral"
                strength = 50.0

            return IndicatorResult(name="QQE", value=qqe_line, signal_strength=strength, interpretation=interpretation)
        except Exception as e:
            from ...config.logger import logger
            logger.warning(f"Error calculating QQE: {e}")
            return IndicatorResult(name="QQE", value=np.nan, signal_strength=np.nan, interpretation="insufficient_data")


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