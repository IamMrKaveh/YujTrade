import numpy as np
import pandas as pd
import talib

import pandas_ta as ta
from pandas_ta.momentum import roc, stochrsi, trix, uo, squeeze, cmo

from common.core import IndicatorResult
from common.exceptions import InsufficientDataError
from features.indicators.base import TechnicalIndicator


class RSIIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(
                f"RSIIndicator requires at least {self.period} data points."
            )
        rsi_series = talib.RSI(
            data["close"].to_numpy(dtype=np.float64), timeperiod=self.period
        )
        if rsi_series.size == 0 or pd.isna(rsi_series[-1]):
            raise InsufficientDataError("RSI calculation resulted in NaN.")

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

        return IndicatorResult(
            name="RSI",
            value=current_rsi,
            signal_strength=float(signal_strength),
            interpretation=interpretation,
        )


class StochasticIndicator(TechnicalIndicator):
    def __init__(self, k_period: int = 14, d_period: int = 3, s_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period
        self.s_period = s_period
        self.min_period = k_period + s_period + d_period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.min_period:
            raise InsufficientDataError(
                f"StochasticIndicator requires at least {self.min_period} data points."
            )
        slowk, slowd = talib.STOCH(
            data["high"].to_numpy(dtype=np.float64),
            data["low"].to_numpy(dtype=np.float64),
            data["close"].to_numpy(dtype=np.float64),
            fastk_period=self.k_period,
            slowk_period=self.s_period,
            slowd_period=self.d_period,
        )

        if slowk.size == 0 or pd.isna(slowk[-1]):
            raise InsufficientDataError("Stochastic calculation resulted in NaN.")

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

        return IndicatorResult(
            name="STOCH",
            value=float(avg_stoch),
            signal_strength=float(signal_strength),
            interpretation=interpretation,
        )


class WilliamsRIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"WilliamsRIndicator requires at least {self.period} data points.")
        willr = talib.WILLR(
            data["high"].to_numpy(dtype=np.float64),
            data["low"].to_numpy(dtype=np.float64),
            data["close"].to_numpy(dtype=np.float64),
            timeperiod=self.period,
        )
        if willr.size == 0 or pd.isna(willr[-1]):
            raise InsufficientDataError("WILLR calculation resulted in NaN.")

        value = float(willr[-1])
        if value > -20:
            interpretation = "overbought"
            strength = (value + 20) / 20 * 100
        elif value < -80:
            interpretation = "oversold"
            strength = (-80 - value) / 20 * 100
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult("WilliamsR", value, min(strength, 100.0), interpretation)


class CCIIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"CCIIndicator requires at least {self.period} data points.")
        cci = talib.CCI(
            data["high"].to_numpy(dtype=np.float64),
            data["low"].to_numpy(dtype=np.float64),
            data["close"].to_numpy(dtype=np.float64),
            timeperiod=self.period,
        )
        if cci.size == 0 or pd.isna(cci[-1]):
            raise InsufficientDataError("CCI calculation resulted in NaN.")

        value = float(cci[-1])
        if value > 100:
            interpretation = "overbought"
            strength = (value - 100) / 100 * 50
        elif value < -100:
            interpretation = "oversold"
            strength = abs(value + 100) / 100 * 50
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult("CCI", value, min(strength, 100.0), interpretation)


class SqueezeMomentumIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 20:
            raise InsufficientDataError("SqueezeMomentumIndicator requires at least 20 data points.")
        sqz = squeeze(high=data["high"], low=data["low"], close=data["close"])
        if sqz is None or sqz.empty or sqz.iloc[-1].isna().any():
            raise InsufficientDataError("Squeeze calculation resulted in NaN.")

        value = float(sqz.iloc[-1, 0])
        if value > 0 and sqz.iloc[-2, 0] < 0:
            interpretation = "bullish_squeeze_release"
            strength = 100.0
        elif value < 0 and sqz.iloc[-2, 0] > 0:
            interpretation = "bearish_squeeze_release"
            strength = 100.0
        else:
            interpretation = "in_squeeze" if float(sqz.iloc[-1, 1]) == 0 else "momentum"
            strength = 50.0

        return IndicatorResult("Squeeze", value, strength, interpretation)


class MoneyFlowIndexIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"MFIIndicator requires at least {self.period} data points.")
        mfi = talib.MFI(
            data["high"].to_numpy(dtype=np.float64),
            data["low"].to_numpy(dtype=np.float64),
            data["close"].to_numpy(dtype=np.float64),
            data["volume"].to_numpy(dtype=np.float64),
            timeperiod=self.period,
        )
        if mfi.size == 0 or pd.isna(mfi[-1]):
            raise InsufficientDataError("MFI calculation resulted in NaN.")

        value = float(mfi[-1])
        if value > 80:
            interpretation = "overbought"
            strength = (value - 80) / 20 * 100
        elif value < 20:
            interpretation = "oversold"
            strength = (20 - value) / 20 * 100
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult("MFI", value, min(strength, 100.0), interpretation)


class UltimateOscillatorIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 28:
            raise InsufficientDataError("UOIndicator requires at least 28 data points.")
        uo_val = talib.ULTOSC(
            data["high"].to_numpy(dtype=np.float64),
            data["low"].to_numpy(dtype=np.float64),
            data["close"].to_numpy(dtype=np.float64),
        )
        if uo_val.size == 0 or pd.isna(uo_val[-1]):
            raise InsufficientDataError("UO calculation resulted in NaN.")

        value = float(uo_val[-1])
        if value > 70:
            interpretation = "overbought"
            strength = (value - 70) / 30 * 100
        elif value < 30:
            interpretation = "oversold"
            strength = (30 - value) / 30 * 100
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult("UO", value, min(strength, 100.0), interpretation)


class ROCIndicator(TechnicalIndicator):
    def __init__(self, period: int = 10):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"ROCIndicator requires at least {self.period} data points.")
        roc_val = talib.ROC(data["close"].to_numpy(dtype=np.float64), timeperiod=self.period)
        if roc_val.size == 0 or pd.isna(roc_val[-1]):
            raise InsufficientDataError("ROC calculation resulted in NaN.")

        value = float(roc_val[-1])
        strength = abs(value) * 5
        interpretation = "bullish" if value > 0 else "bearish"

        return IndicatorResult("ROC", value, min(strength, 100.0), interpretation)


class TRIXIndicator(TechnicalIndicator):
    def __init__(self, period: int = 30):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period * 3:
            raise InsufficientDataError(f"TRIXIndicator requires at least {self.period * 3} data points.")
        trix_val = trix(close=data["close"], length=self.period)
        if trix_val is None or trix_val.empty or trix_val.iloc[-1, 0] is None or pd.isna(trix_val.iloc[-1, 0]):
            raise InsufficientDataError("TRIX calculation resulted in NaN.")

        value = float(trix_val.iloc[-1, 0])
        strength = abs(value) * 100
        interpretation = "bullish" if value > 0 else "bearish"

        return IndicatorResult("TRIX", value, min(strength, 100.0), interpretation)


class StochRSIIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 30:
            raise InsufficientDataError("StochRSIIndicator requires at least 30 data points.")
        stoch_rsi_df = stochrsi(close=data["close"])
        if stoch_rsi_df is None or stoch_rsi_df.empty or stoch_rsi_df.iloc[-1].isna().any():
            raise InsufficientDataError("StochRSI calculation resulted in NaN.")

        k = float(stoch_rsi_df.iloc[-1, 0])
        if k > 80:
            interpretation = "overbought"
            strength = (k - 80) / 20 * 100
        elif k < 20:
            interpretation = "oversold"
            strength = (20 - k) / 20 * 100
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult("StochRSI", k, min(strength, 100.0), interpretation)


class MomentumIndicator(TechnicalIndicator):
    def __init__(self, period: int = 10):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"MomentumIndicator requires at least {self.period} data points.")
        mom_val = talib.MOM(data["close"].to_numpy(dtype=np.float64), timeperiod=self.period)
        if mom_val.size == 0 or pd.isna(mom_val[-1]):
            raise InsufficientDataError("Momentum calculation resulted in NaN.")

        value = float(mom_val[-1])
        strength = abs(value / data['close'].iloc[-1]) * 1000 if data['close'].iloc[-1] > 0 else 0
        interpretation = "positive" if value > 0 else "negative"

        return IndicatorResult("Momentum", value, min(strength, 100.0), interpretation)


class AwesomeOscillatorIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 34:
            raise InsufficientDataError("AwesomeOscillatorIndicator requires at least 34 data points.")
        ao = ta.ao(high=data["high"], low=data["low"])
        if ao is None or ao.empty or pd.isna(ao.iloc[-1]):
            raise InsufficientDataError("AO calculation resulted in NaN.")

        value = float(ao.iloc[-1])
        strength = abs(value / data['close'].iloc[-1]) * 1000 if data['close'].iloc[-1] > 0 else 0
        interpretation = "bullish" if value > 0 else "bearish"

        return IndicatorResult("AwesomeOscillator", value, min(strength, 100.0), interpretation)


class ChandeMomentumOscillatorIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"CMOIndicator requires at least {self.period} data points.")
        cmo_val = cmo(close=data["close"], length=self.period)
        if cmo_val is None or cmo_val.empty or pd.isna(cmo_val.iloc[-1]):
            raise InsufficientDataError("CMO calculation resulted in NaN.")

        value = float(cmo_val.iloc[-1])
        if value > 50:
            interpretation = "overbought"
            strength = (value - 50) * 2
        elif value < -50:
            interpretation = "oversold"
            strength = (-50 - value) * -2
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult("CMO", value, min(strength, 100.0), interpretation)


class RelativeVigorIndexIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 14:
            raise InsufficientDataError("RVIIndicator requires at least 14 data points.")
        rvi = ta.rvi(close=data["close"], high=data["high"], low=data["low"], open=data["open"])
        if rvi is None or rvi.empty or rvi.iloc[-1].isna().any():
            raise InsufficientDataError("RVI calculation resulted in NaN.")

        value = float(rvi.iloc[-1, 0])
        signal = float(rvi.iloc[-1, 1])
        strength = abs(value - signal) * 100
        interpretation = "bullish" if value > signal else "bearish"

        return IndicatorResult("RVI", value, min(strength, 100.0), interpretation)


class BalanceOfPowerIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 1:
            raise InsufficientDataError("BOPIndicator requires at least 1 data point.")
        bop = talib.BOP(
            data["open"].to_numpy(dtype=np.float64),
            data["high"].to_numpy(dtype=np.float64),
            data["low"].to_numpy(dtype=np.float64),
            data["close"].to_numpy(dtype=np.float64),
        )
        if bop.size == 0 or pd.isna(bop[-1]):
            raise InsufficientDataError("BOP calculation resulted in NaN.")

        value = float(bop[-1])
        strength = abs(value) * 100
        interpretation = "buy_power" if value > 0 else "sell_power"

        return IndicatorResult("BOP", value, strength, interpretation)


class FisherTransformIndicator(TechnicalIndicator):
    def __init__(self, period: int = 9):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            raise InsufficientDataError(f"FisherTransform requires at least {self.period} data points.")
        fisher = ta.fisher(high=data["high"], low=data["low"], length=self.period)
        if fisher is None or fisher.empty or fisher.iloc[-1].isna().any():
            raise InsufficientDataError("Fisher Transform calculation resulted in NaN.")

        value = float(fisher.iloc[-1, 0])
        signal = float(fisher.iloc[-1, 1])
        strength = abs(value) * 25
        interpretation = "bullish_crossover" if value > signal else "bearish_crossover"

        return IndicatorResult("Fisher", value, min(strength, 100.0), interpretation)


class SchaffTrendCycleIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 50:
            raise InsufficientDataError("SchaffTrendCycle requires at least 50 data points.")
        stc = ta.stc(close=data["close"])
        if stc is None or stc.empty or stc.iloc[-1].isna().any():
            raise InsufficientDataError("STC calculation resulted in NaN.")

        value = float(stc.iloc[-1, 0])
        if value > 75:
            interpretation = "overbought"
            strength = (value - 75) * 4
        elif value < 25:
            interpretation = "oversold"
            strength = (25 - value) * 4
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult("STC", value, min(strength, 100.0), interpretation)


class QQEIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 20:
            raise InsufficientDataError("QQE requires at least 20 data points.")
        qqe = ta.qqe(close=data["close"])
        if qqe is None or qqe.empty or qqe.iloc[-1].isna().any():
            raise InsufficientDataError("QQE calculation resulted in NaN.")

        value = float(qqe.iloc[-1, 0])
        signal = float(qqe.iloc[-1, 1])
        strength = abs(value - 50) * 2
        interpretation = "bullish" if value > signal else "bearish"

        return IndicatorResult("QQE", value, min(strength, 100.0), interpretation)


class ConnorsRSIIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 100:
            raise InsufficientDataError("ConnorsRSI requires at least 100 data points.")
        crsi = ta.crsi(close=data["close"])
        if crsi is None or crsi.empty or pd.isna(crsi.iloc[-1]):
            raise InsufficientDataError("ConnorsRSI calculation resulted in NaN.")

        value = float(crsi.iloc[-1])
        if value > 90:
            interpretation = "overbought"
            strength = (value - 90) * 10
        elif value < 10:
            interpretation = "oversold"
            strength = (10 - value) * 10
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult("ConnorsRSI", value, min(strength, 100.0), interpretation)


class StochasticMomentumIndexIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 20:
            raise InsufficientDataError("SMI requires at least 20 data points.")
        smi = ta.smi(close=data["close"])
        if smi is None or smi.empty or smi.iloc[-1].isna().any():
            raise InsufficientDataError("SMI calculation resulted in NaN.")

        value = float(smi.iloc[-1, 0])
        signal = float(smi.iloc[-1, 1])
        strength = abs(value)
        interpretation = "bullish" if value > signal else "bearish"

        return IndicatorResult("SMI", value, min(strength, 100.0), interpretation)


class TSIIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 50:
            raise InsufficientDataError("TSI requires at least 50 data points.")
        tsi = ta.tsi(close=data["close"])
        if tsi is None or tsi.empty or tsi.iloc[-1].isna().any():
            raise InsufficientDataError("TSI calculation resulted in NaN.")

        value = float(tsi.iloc[-1, 0])
        signal = float(tsi.iloc[-1, 1])
        strength = abs(value) * 2
        interpretation = "bullish" if value > signal else "bearish"

        return IndicatorResult("TSI", value, min(strength, 100.0), interpretation)


class RMIIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period + 5:
            raise InsufficientDataError(f"RMI requires at least {self.period + 5} data points.")
        rmi = ta.rmi(close=data["close"], length=self.period)
        if rmi is None or rmi.empty or pd.isna(rmi.iloc[-1]):
            raise InsufficientDataError("RMI calculation resulted in NaN.")

        value = float(rmi.iloc[-1])
        if value > 70:
            interpretation = "overbought"
            strength = (value - 70) * 100 / 30
        elif value < 30:
            interpretation = "oversold"
            strength = (30 - value) * 100 / 30
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult("RMI", value, min(strength, 100.0), interpretation)


class RSI2Indicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 5:
            raise InsufficientDataError("RSI2 requires at least 5 data points.")
        rsi2 = ta.rsi(close=data["close"], length=2)
        if rsi2 is None or rsi2.empty or pd.isna(rsi2.iloc[-1]):
            raise InsufficientDataError("RSI2 calculation resulted in NaN.")

        value = float(rsi2.iloc[-1])
        if value > 90:
            interpretation = "extreme_overbought"
            strength = 100.0
        elif value < 10:
            interpretation = "extreme_oversold"
            strength = 100.0
        else:
            interpretation = "neutral"
            strength = 50.0

        return IndicatorResult("RSI2", value, strength, interpretation)


class PPOIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 26:
            raise InsufficientDataError("PPO requires at least 26 data points.")
        ppo = ta.ppo(close=data["close"])
        if ppo is None or ppo.empty or ppo.iloc[-1].isna().any():
            raise InsufficientDataError("PPO calculation resulted in NaN.")

        value = float(ppo.iloc[-1, 0])
        signal = float(ppo.iloc[-1, 1])
        strength = abs(value) * 10
        interpretation = "bullish" if value > signal else "bearish"

        return IndicatorResult("PPO", value, min(strength, 100.0), interpretation)


class CoppockCurveIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 30:
            raise InsufficientDataError("CoppockCurve requires at least 30 data points.")
        coppock = ta.coppock(close=data["close"])
        if coppock is None or coppock.empty or pd.isna(coppock.iloc[-1]):
            raise InsufficientDataError("Coppock Curve calculation resulted in NaN.")

        value = float(coppock.iloc[-1])
        strength = abs(value)
        interpretation = "uptrend" if value > 0 else "downtrend"

        return IndicatorResult("Coppock", value, min(strength, 100.0), interpretation)


class BalanceOfPowerRSIIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 15:
            raise InsufficientDataError("BOPRSI requires at least 15 data points.")

        bop = talib.BOP(data['open'], data['high'], data['low'], data['close'])
        bop_series = pd.Series(bop, index=data.index).dropna()

        if len(bop_series) < 14:
            raise InsufficientDataError("Not enough BOP values to calculate RSI.")

        bop_rsi = talib.RSI(bop_series.to_numpy(dtype=np.float64), timeperiod=14)
        if bop_rsi.size == 0 or pd.isna(bop_rsi[-1]):
            raise InsufficientDataError("BOP RSI calculation resulted in NaN.")

        value = float(bop_rsi[-1])
        if value > 70:
            interpretation = "strong_buying_pressure"
            strength = (value - 70) * 100 / 30
        elif value < 30:
            interpretation = "strong_selling_pressure"
            strength = (30 - value) * 100 / 30
        else:
            interpretation = "neutral_pressure"
            strength = 50.0

        return IndicatorResult("BOPRSI", value, min(strength, 100.0), interpretation)


class SmartMoneyConceptIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 20:
            raise InsufficientDataError("SMC requires at least 20 data points.")

        last_swing_high = data['high'].rolling(9).max().iloc[-2]
        last_swing_low = data['low'].rolling(9).min().iloc[-2]

        break_of_structure = data['close'].iloc[-1] > last_swing_high
        if break_of_structure:
            return IndicatorResult("SMC", 1.0, 100.0, "break_of_structure_up")

        change_of_character = data['close'].iloc[-1] < last_swing_low
        if change_of_character:
            return IndicatorResult("SMC", -1.0, 100.0, "change_of_character_down")

        return IndicatorResult("SMC", 0.0, 0.0, "no_smc_event")