
import os
import warnings
import sys
import asyncio
import json
import sqlite3
import aiosqlite
import requests
import time
import warnings

from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Protocol, Any, Union
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from contextlib import asynccontextmanager
from pathlib import Path
from contextlib import asynccontextmanager
from sklearn.preprocessing import MinMaxScaler
from web3 import Web3

import pandas as pd
import ccxt.async_support as ccxt
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from logger_config import logger
from exchanges.constants import COINEX_API_KEY, COINEX_SECRET, TIME_FRAMES, SYMBOLS
from telegrams.constants import BOT_TOKEN

tf.get_logger().setLevel('ERROR')

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.runtime_version')
warnings.filterwarnings('ignore', message='.*Protobuf gencode version.*')

class RateLimiter:
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window  #in seconds
        self.requests: Dict[str, list] = {}
    
    def wait_if_needed(self, endpoint: str):
        now = time.time()
        if endpoint not in self.requests:
            self.requests[endpoint] = []
        
        self.requests[endpoint] = [req_time for req_time in self.requests[endpoint] 
                                    if now - req_time < self.time_window]
        
        if len(self.requests[endpoint]) >= self.max_requests:
            sleep_time = self.time_window - (now - self.requests[endpoint][0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached for {endpoint}. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                self.requests[endpoint] = []
        
        self.requests[endpoint].append(now)
        
rate_limiter = RateLimiter(max_requests=10, time_window=60)

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"

class MarketCondition(Enum):
    OVERSOLD = "oversold"
    OVERBOUGHT = "overbought"
    NEUTRAL = "neutral"

class TrendStrength(Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"

@dataclass
class TradingSignal:
    symbol: str
    signal_type: SignalType
    entry_price: float
    exit_price: float
    stop_loss: float
    timestamp: datetime
    timeframe: str
    confidence_score: float
    reasons: List[str]
    risk_reward_ratio: float
    predicted_profit: float
    volume_analysis: Dict[str, float]
    market_context: Dict[str, Any]
    dynamic_levels: Dict[str, float]

@dataclass
class MarketAnalysis:
    trend: TrendDirection
    trend_strength: TrendStrength
    volatility: float
    volume_trend: str
    support_levels: List[float]
    resistance_levels: List[float]
    momentum_score: float
    market_condition: MarketCondition
    trend_acceleration: float
    volume_confirmation: bool

@dataclass
class IndicatorResult:
    name: str
    value: float
    signal_strength: float
    interpretation: str

@dataclass
class DynamicLevels:
    primary_entry: float
    secondary_entry: float
    primary_exit: float
    secondary_exit: float
    tight_stop: float
    wide_stop: float
    breakeven_point: float
    trailing_stop: float

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
        self.ma_type = ma_type
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if self.ma_type == "sma":
            ma_values = data['close'].rolling(window=self.period).mean()
        else:
            ma_values = data['close'].ewm(span=self.period).mean()
        current_price = data['close'].iloc[-1]
        current_ma = ma_values.iloc[-1]
        signal_strength = abs((current_price - current_ma) / current_ma) * 100
        if current_price > current_ma:
            interpretation = "bullish_above_ma"
        else:
            interpretation = "bearish_below_ma"
        return IndicatorResult(name=f"{self.ma_type.upper()}_{self.period}", value=current_ma, signal_strength=signal_strength, interpretation=interpretation)

class RSIIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if 'close' not in data.columns:
            logger.error("Data does not contain 'close' column for RSI calculation")
            raise ValueError("Data must contain 'close' column")
        
        if len(data) < self.period + 1:
            logger.error(f"Insufficient data for RSI calculation. Need at least {self.period + 1} records")
            raise ValueError(f"Insufficient data for RSI calculation. Need at least {self.period + 1} records")
        
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        
        current_gain = gain.iloc[-1]
        current_loss = loss.iloc[-1]

        if pd.isna(current_gain) or pd.isna(current_loss):
            current_rsi = 50
        elif current_loss == 0 or current_loss < 1e-10:
            current_rsi = 100 if current_gain > 0 else 50
        else:
            rs = current_gain / current_loss
            current_rsi = 100 - (100 / (1 + rs))
            
        current_rsi = max(0, min(100, current_rsi))
            
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
            signal_strength=signal_strength, 
            interpretation=interpretation
        )

class MACDIndicator(TechnicalIndicator):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        ema_fast = data['close'].ewm(span=self.fast).mean()
        ema_slow = data['close'].ewm(span=self.slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal).mean()
        histogram = macd_line - signal_line
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        if current_macd > current_signal and current_histogram > 0:
            interpretation = "bullish_crossover"
            signal_strength = abs(current_histogram) * 100
        elif current_macd < current_signal and current_histogram < 0:
            interpretation = "bearish_crossover"
            signal_strength = abs(current_histogram) * 100
        else:
            interpretation = "neutral"
            signal_strength = 50
        return IndicatorResult(name="MACD", value=current_macd, signal_strength=signal_strength, interpretation=interpretation)

class BollingerBandsIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20, std_dev: float = 2):
        self.period = period
        self.std_dev = std_dev
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        sma = data['close'].rolling(window=self.period).mean()
        std = data['close'].rolling(window=self.period).std()
        upper_band = sma + (std * self.std_dev)
        lower_band = sma - (std * self.std_dev)
        current_price = data['close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_middle = sma.iloc[-1]
        bb_position = (current_price - current_lower) / (current_upper - current_lower) if (current_upper - current_lower) != 0 else 0.5
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
    def __init__(self, k_period: int = 14, d_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.k_period:
            return IndicatorResult(name="STOCH", value=50, signal_strength=0, interpretation="neutral")
            
        lowest_low = data['low'].rolling(window=self.k_period).min()
        highest_high = data['high'].rolling(window=self.k_period).max()
        
        range_diff = highest_high - lowest_low
        range_diff = range_diff.replace(0, np.nan)
        
        k_percent = ((data['close'] - lowest_low) / range_diff) * 100
        k_percent = k_percent.fillna(50)
        k_percent = k_percent.replace([np.inf, -np.inf], 50)
        k_percent = k_percent.clip(0, 100)
        
        d_percent = k_percent.rolling(window=self.d_period).mean()
        
        current_k = k_percent.iloc[-1]
        current_d = d_percent.iloc[-1]
        
        if pd.isna(current_k) or pd.isna(current_d):
            return IndicatorResult(name="STOCH", value=50, signal_strength=0, interpretation="neutral")
        
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
            value=avg_stoch, 
            signal_strength=signal_strength, 
            interpretation=interpretation
        )

class VolumeIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            return IndicatorResult(name="VOLUME", value=1, signal_strength=50, interpretation="normal_volume")
            
        volume_ma = data['volume'].rolling(window=self.period).mean()
        current_volume = data['volume'].iloc[-1]
        average_volume = volume_ma.iloc[-1]
        
        if pd.isna(average_volume) or average_volume == 0:
            volume_ratio = 1
        else:
            volume_ratio = current_volume / average_volume
            
        if volume_ratio > 1.5:
            interpretation = "high_volume"
            signal_strength = min((volume_ratio - 1) * 50, 100)
        elif volume_ratio < 0.5:
            interpretation = "low_volume"
            signal_strength = (1 - volume_ratio) * 100
        else:
            interpretation = "normal_volume"
            signal_strength = 50
        return IndicatorResult(name="VOLUME", value=volume_ratio, signal_strength=signal_strength, interpretation=interpretation)

class ATRIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=self.period).mean()
        current_atr = atr.iloc[-1]
        current_price = data['close'].iloc[-1]
        atr_percentage = (current_atr / current_price) * 100 if current_price else 0
        if atr_percentage > 3:
            interpretation = "high_volatility"
            signal_strength = min(atr_percentage * 20, 100)
        elif atr_percentage < 1:
            interpretation = "low_volatility"
            signal_strength = (1 - atr_percentage) * 100
        else:
            interpretation = "normal_volatility"
            signal_strength = 50
        return IndicatorResult(name="ATR", value=current_atr, signal_strength=signal_strength, interpretation=interpretation)

class IchimokuIndicator(TechnicalIndicator):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < 52:
            return IndicatorResult(name="Ichimoku", value=0, signal_strength=0, interpretation="neutral")
        high_9 = data['high'].rolling(window=9).max().iloc[-1]
        low_9 = data['low'].rolling(window=9).min().iloc[-1]
        tenkan = (high_9 + low_9) / 2
        high_26 = data['high'].rolling(window=26).max().iloc[-1]
        low_26 = data['low'].rolling(window=26).min().iloc[-1]
        kijun = (high_26 + low_26) / 2
        senkou_a = (tenkan + kijun) / 2
        high_52 = data['high'].rolling(window=52).max().iloc[-1]
        low_52 = data['low'].rolling(window=52).min().iloc[-1]
        senkou_b = (high_52 + low_52) / 2
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
        highest_high = data['high'].rolling(window=self.period).max().iloc[-1]
        lowest_low = data['low'].rolling(window=self.period).min().iloc[-1]
        current_close = data['close'].iloc[-1]
        if highest_high - lowest_low == 0:
            value = 0
        else:
            value = (highest_high - current_close) / (highest_high - lowest_low) * -100
        if value > -20:
            interpretation = "overbought"
            signal_strength = (value + 100) / 80 * 100
        elif value < -80:
            interpretation = "oversold"
            signal_strength = (abs(value) - 80) / 80 * 100
        else:
            interpretation = "neutral"
            signal_strength = 50
        return IndicatorResult(name="Williams %R", value=value, signal_strength=signal_strength, interpretation=interpretation)

class CCIIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period
        
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            return IndicatorResult(name="CCI", value=0, signal_strength=0, interpretation="neutral")
            
        tp = (data['high'] + data['low'] + data['close']) / 3
        sma = tp.rolling(window=self.period).mean()
        
        current_tp = tp.iloc[-1]
        current_sma = sma.iloc[-1]
        
        if pd.isna(current_sma):
            return IndicatorResult(name="CCI", value=0, signal_strength=0, interpretation="neutral")
        
        recent_tp = tp.tail(self.period)
        mean_dev = (recent_tp - current_sma).abs().mean()
        
        if pd.isna(mean_dev) or mean_dev == 0 or mean_dev < 1e-10:
            cci = 0
        else:
            cci = (current_tp - current_sma) / (0.015 * mean_dev)
        
        cci = max(-300, min(300, cci))
        
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

class FibonacciLevels:
    @staticmethod
    def calculate_retracement_levels(high: float, low: float) -> Dict[str, float]:
        diff = high - low
        return {'0.236': high - 0.236 * diff, '0.382': high - 0.382 * diff, '0.500': high - 0.500 * diff, '0.618': high - 0.618 * diff, '0.786': high - 0.786 * diff}
    @staticmethod
    def calculate_extension_levels(high: float, low: float, entry: float) -> Dict[str, float]:
        range_size = high - low
        return {'1.272': entry + 1.272 * range_size, '1.414': entry + 1.414 * range_size, '1.618': entry + 1.618 * range_size, '2.000': entry + 2.000 * range_size}

class PivotPoints:
    @staticmethod
    def calculate_pivot_levels(high: float, low: float, close: float) -> Dict[str, float]:
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        return {'pivot': pivot, 'r1': r1, 'r2': r2, 'r3': r3, 's1': s1, 's2': s2, 's3': s3}

class PatternAnalyzer:
    @staticmethod
    def detect_patterns(data: pd.DataFrame) -> List[str]:
        patterns = []
        if len(data) < 2:
            return patterns
        prev_open = data['open'].iloc[-2]
        prev_close = data['close'].iloc[-2]
        last_open = data['open'].iloc[-1]
        last_close = data['close'].iloc[-1]
        if prev_close < prev_open and last_close > last_open and last_close > prev_open and last_open <= prev_close:
            patterns.append("bullish_engulfing")
        if prev_close > prev_open and last_close < last_open and last_close < prev_open and last_open >= prev_close:
            patterns.append("bearish_engulfing")
        highs = data['high']
        lows = data['low']
        local_max = []
        local_min = []
        for i in range(1, len(data)-1):
            if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                local_max.append((float(highs.iloc[i]), i))
            if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                local_min.append((float(lows.iloc[i]), i))
        if len(local_max) >= 2:
            sorted_max = sorted(local_max, key=lambda x: x[0], reverse=True)
            top1, top2 = sorted_max[0][0], sorted_max[1][0]
            if abs(top1 - top2) / ((top1+top2)/2) < 0.02:
                patterns.append("double_top")
        if len(local_min) >= 2:
            sorted_min = sorted(local_min, key=lambda x: x[0])
            bot1, bot2 = sorted_min[0][0], sorted_min[1][0]
            if abs(bot1 - bot2) / ((bot1+bot2)/2) < 0.02:
                patterns.append("double_bottom")
        if len(local_max) >= 3:
            sorted_by_index = sorted(local_max, key=lambda x: x[1])
            for i in range(len(sorted_by_index)-2):
                left, center, right = sorted_by_index[i], sorted_by_index[i+1], sorted_by_index[i+2]
                if center[0] > left[0] and center[0] > right[0] and left[0] < right[0]*1.05 and right[0] < left[0]*1.05:
                    patterns.append("head_and_shoulders")
                    break
        if len(local_min) >= 3:
            sorted_by_index = sorted(local_min, key=lambda x: x[1])
            for i in range(len(sorted_by_index)-2):
                left, center, right = sorted_by_index[i], sorted_by_index[i+1], sorted_by_index[i+2]
                if center[0] < left[0] and center[0] < right[0] and left[0] > right[0]*0.95 and right[0] > left[0]*0.95:
                    patterns.append("inverse_head_and_shoulders")
                    break
        return patterns
    @staticmethod
    def detect_flag(data: pd.DataFrame) -> bool:
        if len(data) < 20:
            return False
        recent = data.tail(20)
        highs = recent['high']
        lows = recent['low']
        if highs.max() - highs.min() < (data['close'].iloc[-20] * 0.02):
            return True
        return False
    @staticmethod
    def detect_wedge(data: pd.DataFrame) -> bool:
        if len(data) < 30:
            return False
        recent = data.tail(30)
        slope_high = np.polyfit(range(len(recent)), recent['high'], 1)[0]
        slope_low = np.polyfit(range(len(recent)), recent['low'], 1)[0]
        return abs(slope_high) < 0.05 and abs(slope_low) < 0.05 and slope_high * slope_low < 0
    @staticmethod
    def detect_triangle(data: pd.DataFrame) -> bool:
        if len(data) < 30:
            return False
        recent = data.tail(30)
        highs = recent['high']
        lows = recent['low']
        h_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        l_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        return h_slope < 0 and l_slope > 0

class TrendAnalyzer:
    @staticmethod
    def calculate_trend_strength(data: pd.DataFrame) -> TrendStrength:
        if len(data) < 50:
            return TrendStrength.WEAK
        sma_20 = data['close'].rolling(window=20).mean()
        sma_50 = data['close'].rolling(window=50).mean()
        current_sma_20 = sma_20.iloc[-1]
        current_sma_50 = sma_50.iloc[-1]
        
        if pd.isna(current_sma_20) or pd.isna(current_sma_50) or current_sma_50 == 0:
            return TrendStrength.WEAK
            
        sma_separation = abs((current_sma_20 - current_sma_50) / current_sma_50)
        adx = TrendAnalyzer._calculate_adx(data)
        
        if adx > 40 and sma_separation > 0.03:
            return TrendStrength.STRONG
        elif adx > 25 and sma_separation > 0.015:
            return TrendStrength.MODERATE
        else:
            return TrendStrength.WEAK

    @staticmethod
    def _calculate_adx(data: pd.DataFrame, period: int = 14) -> float:
        if len(data) < period + 1:
            return 0
            
        high = data['high']
        low = data['low']
        close = data['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr_list = []
        for i in range(len(data)):
            if i == 0:
                tr_list.append(high.iloc[i] - low.iloc[i])
            else:
                tr_list.append(max(
                    high.iloc[i] - low.iloc[i], 
                    abs(high.iloc[i] - close.iloc[i-1]), 
                    abs(low.iloc[i] - close.iloc[i-1])
                ))
        
        tr = pd.Series(tr_list)
        
        plus_dm_smooth = plus_dm.rolling(window=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period).mean()
        tr_smooth = tr.rolling(window=period).mean()
        
        tr_smooth = tr_smooth.replace(0, np.nan)
        
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        plus_di = plus_di.fillna(0)
        minus_di = minus_di.fillna(0)
        
        dx_denominator = plus_di + minus_di
        dx_denominator = dx_denominator.replace(0, np.nan)
        
        dx = 100 * abs(plus_di - minus_di) / dx_denominator
        dx = dx.fillna(0)
        
        adx = dx.rolling(window=period).mean()
        
        final_adx = adx.iloc[-1]
        return final_adx if not pd.isna(final_adx) else 0

class SupportResistanceAnalyzer:
    def __init__(self, lookback_period: int = 50):
        self.lookback_period = lookback_period
    def find_support_resistance(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        if len(data) < self.lookback_period:
            return [], []
        recent_data = data.tail(self.lookback_period)
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        resistance_levels = []
        support_levels = []
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                resistance_levels.append(highs[i])
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                support_levels.append(lows[i])
        resistance_levels = sorted(set(resistance_levels), reverse=True)[:5]
        support_levels = sorted(set(support_levels), reverse=True)[:5]
        return support_levels, resistance_levels

class VolumeAnalyzer:
    def analyze_volume_pattern(self, data: pd.DataFrame) -> Dict[str, float]:
        volume_ma_20 = data['volume'].rolling(window=20).mean()
        current_volume = data['volume'].iloc[-1]
        avg_volume = volume_ma_20.iloc[-1]
        volume_trend = self._calculate_volume_trend(data)
        volume_breakout = current_volume / avg_volume if avg_volume and not np.isnan(avg_volume) else 1
        return {'volume_ratio': volume_breakout, 'volume_trend': volume_trend, 'volume_strength': min(volume_breakout * 50, 100), 'volume_confirmation': volume_breakout > 1.2 and volume_trend > 0}
    def _calculate_volume_trend(self, data: pd.DataFrame) -> float:
        recent_volumes = data['volume'].tail(10)
        if len(recent_volumes) < 2:
            return 0
        volume_changes = recent_volumes.pct_change().dropna()
        return volume_changes.mean()
    def volume_profile(self, data: pd.DataFrame, bins: int = 30) -> List[Tuple[float, float]]:
        prices = data['close']
        volumes = data['volume']
        min_p, max_p = prices.min(), prices.max()
        if max_p == min_p:
            return [(min_p, volumes.sum())]
        bin_size = (max_p - min_p) / bins
        buckets = {}
        for p, v in zip(prices, volumes):
            idx = int((p - min_p) / bin_size)
            key = min_p + idx * bin_size
            buckets[key] = buckets.get(key, 0) + v
        items = sorted(buckets.items(), key=lambda x: x[0])
        return items
    def vwap(self, data: pd.DataFrame) -> float:
        pv = (data['close'] * data['volume']).sum()
        v = data['volume'].sum()
        return pv / v if v else data['close'].iloc[-1]

class MarketConditionAnalyzer:
    def __init__(self):
        self.support_resistance = SupportResistanceAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        
    def analyze_market_condition(self, data: pd.DataFrame) -> MarketAnalysis:
        support_levels, resistance_levels = self.support_resistance.find_support_resistance(data)
        volume_analysis = self.volume_analyzer.analyze_volume_pattern(data)
        trend = self._determine_trend(data)
        trend_strength = self.trend_analyzer.calculate_trend_strength(data)
        volatility = self._calculate_volatility(data)
        momentum_score = self._calculate_momentum(data)
        market_condition = self._determine_market_condition(data)
        trend_acceleration = self._calculate_trend_acceleration(data)
        
        return MarketAnalysis(
            trend=trend, 
            trend_strength=trend_strength, 
            volatility=volatility, 
            volume_trend="increasing" if volume_analysis['volume_trend'] > 0 else "decreasing", 
            support_levels=support_levels, 
            resistance_levels=resistance_levels, 
            momentum_score=momentum_score, 
            market_condition=market_condition, 
            trend_acceleration=trend_acceleration, 
            volume_confirmation=volume_analysis['volume_confirmation']
        )

    def _determine_trend(self, data: pd.DataFrame) -> TrendDirection:
        if len(data) < 50:
            return TrendDirection.SIDEWAYS
            
        sma_20 = data['close'].rolling(window=20).mean()
        sma_50 = data['close'].rolling(window=50).mean()
        
        current_sma_20 = sma_20.iloc[-1]
        current_sma_50 = sma_50.iloc[-1]
        
        if pd.isna(current_sma_20) or pd.isna(current_sma_50):
            return TrendDirection.SIDEWAYS
            
        if current_sma_20 > current_sma_50 * 1.015:
            return TrendDirection.BULLISH
        elif current_sma_20 < current_sma_50 * 0.985:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.SIDEWAYS

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        if len(data) < 2:
            return 0.0
            
        returns = data['close'].pct_change().dropna()
        if returns.empty or len(returns) < 2:
            return 0.0
        
        try:
            periods_per_year = 365
            if len(data.index) > 1 and hasattr(data, 'index'):
                try:
                    if pd.api.types.is_datetime64_any_dtype(data.index):
                        time_diff = data.index[-1] - data.index[0]
                        if hasattr(time_diff, 'total_seconds'):
                            avg_period_seconds = time_diff.total_seconds() / (len(data) - 1)
                            seconds_per_year = 365 * 24 * 3600
                            periods_per_year = seconds_per_year / avg_period_seconds
                        else:
                            periods_per_year = 365
                except:
                    periods_per_year = 365
            
            volatility = returns.std() * np.sqrt(periods_per_year)
            return volatility if not pd.isna(volatility) else 0.0
            
        except Exception:
            return returns.std() if not returns.empty else 0.0

    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        if len(data) < 20:
            return 0.0
            
        current_close = data['close'].iloc[-1]
        previous_close = data['close'].iloc[-20]
        
        if pd.isna(current_close) or pd.isna(previous_close) or previous_close == 0:
            return 0.0
            
        price_change = (current_close - previous_close) / previous_close
        return price_change * 100

    def _calculate_trend_acceleration(self, data: pd.DataFrame) -> float:
        if len(data) < 10:
            return 0.0
            
        current_close = data['close'].iloc[-1]
        mid_close = data['close'].iloc[-5]
        old_close = data['close'].iloc[-10]
        
        if pd.isna(current_close) or pd.isna(mid_close) or pd.isna(old_close):
            return 0.0
            
        if mid_close == 0 or old_close == 0:
            return 0.0
            
        recent_momentum = (current_close - mid_close) / mid_close
        previous_momentum = (mid_close - old_close) / old_close
        
        return (recent_momentum - previous_momentum) * 100

    def _determine_market_condition(self, data: pd.DataFrame) -> MarketCondition:
        try:
            rsi_indicator = RSIIndicator()
            rsi_result = rsi_indicator.calculate(data)
            
            if rsi_result.value > 70:
                return MarketCondition.OVERBOUGHT
            elif rsi_result.value < 30:
                return MarketCondition.OVERSOLD
            else:
                return MarketCondition.NEUTRAL
        except:
            return MarketCondition.NEUTRAL

class DynamicLevelCalculator:
    def __init__(self):
        self.fibonacci = FibonacciLevels()
        self.pivot_points = PivotPoints()
    def calculate_dynamic_levels(self, data: pd.DataFrame, signal_type: SignalType, market_analysis: MarketAnalysis) -> DynamicLevels:
        current_price = data['close'].iloc[-1]
        high_20 = data['high'].tail(20).max()
        low_20 = data['low'].tail(20).min()
        atr_indicator = ATRIndicator()
        atr_result = atr_indicator.calculate(data)
        atr_value = atr_result.value
        if signal_type == SignalType.BUY:
            return self._calculate_buy_levels(data, current_price, high_20, low_20, atr_value, market_analysis)
        else:
            return self._calculate_sell_levels(data, current_price, high_20, low_20, atr_value, market_analysis)
    def _calculate_buy_levels(self, data: pd.DataFrame, current_price: float, high_20: float, low_20: float, atr_value: float, market_analysis: MarketAnalysis) -> DynamicLevels:
        fib_levels = self.fibonacci.calculate_retracement_levels(high_20, low_20)
        pivot_levels = self.pivot_points.calculate_pivot_levels(data['high'].iloc[-1], data['low'].iloc[-1], data['close'].iloc[-1])
        trend_multiplier = self._get_trend_multiplier(market_analysis)
        volatility_multiplier = self._get_volatility_multiplier(market_analysis)
        primary_entry = current_price
        secondary_entry = min(fib_levels['0.382'], current_price * 0.995)
        if market_analysis.trend_strength == TrendStrength.STRONG:
            primary_exit = current_price * (1 + 0.03 * trend_multiplier)
            secondary_exit = current_price * (1 + 0.05 * trend_multiplier)
        elif market_analysis.trend_strength == TrendStrength.MODERATE:
            primary_exit = current_price * (1 + 0.02 * trend_multiplier)
            secondary_exit = current_price * (1 + 0.035 * trend_multiplier)
        else:
            primary_exit = current_price * (1 + 0.015)
            secondary_exit = current_price * (1 + 0.025)
        if market_analysis.resistance_levels:
            nearest_resistance = min([r for r in market_analysis.resistance_levels if r > current_price], default=primary_exit)
            primary_exit = min(primary_exit, nearest_resistance)
            secondary_exit = min(secondary_exit, nearest_resistance * 1.02)
        tight_stop = max(current_price - (atr_value * volatility_multiplier), fib_levels['0.618'] if fib_levels['0.618'] < current_price else current_price * 0.98)
        wide_stop = max(current_price - (atr_value * 2 * volatility_multiplier), low_20 * 0.995)
        if market_analysis.support_levels:
            nearest_support = max([s for s in market_analysis.support_levels if s < current_price], default=tight_stop)
            tight_stop = max(tight_stop, nearest_support)
            wide_stop = max(wide_stop, nearest_support * 0.995)
        breakeven_point = current_price + (atr_value * 0.5)
        trailing_stop = current_price - (atr_value * 1.5 * volatility_multiplier)
        return DynamicLevels(primary_entry=primary_entry, secondary_entry=secondary_entry, primary_exit=primary_exit, secondary_exit=secondary_exit, tight_stop=tight_stop, wide_stop=wide_stop, breakeven_point=breakeven_point, trailing_stop=trailing_stop)
    def _calculate_sell_levels(self, data: pd.DataFrame, current_price: float, high_20: float, low_20: float, atr_value: float, market_analysis: MarketAnalysis) -> DynamicLevels:
        fib_levels = self.fibonacci.calculate_retracement_levels(high_20, low_20)
        pivot_levels = self.pivot_points.calculate_pivot_levels(data['high'].iloc[-1], data['low'].iloc[-1], data['close'].iloc[-1])
        trend_multiplier = self._get_trend_multiplier(market_analysis)
        volatility_multiplier = self._get_volatility_multiplier(market_analysis)
        primary_entry = current_price
        secondary_entry = max(fib_levels['0.382'], current_price * 1.005)
        if market_analysis.trend_strength == TrendStrength.STRONG:
            primary_exit = current_price * (1 - 0.03 * trend_multiplier)
            secondary_exit = current_price * (1 - 0.05 * trend_multiplier)
        elif market_analysis.trend_strength == TrendStrength.MODERATE:
            primary_exit = current_price * (1 - 0.02 * trend_multiplier)
            secondary_exit = current_price * (1 - 0.035 * trend_multiplier)
        else:
            primary_exit = current_price * (1 - 0.015)
            secondary_exit = current_price * (1 - 0.025)
        if market_analysis.support_levels:
            nearest_support = max([s for s in market_analysis.support_levels if s < current_price], default=primary_exit)
            primary_exit = max(primary_exit, nearest_support)
            secondary_exit = max(secondary_exit, nearest_support * 0.98)
        tight_stop = min(current_price + (atr_value * volatility_multiplier), fib_levels['0.618'] if fib_levels['0.618'] > current_price else current_price * 1.02)
        wide_stop = min(current_price + (atr_value * 2 * volatility_multiplier), high_20 * 1.005)
        if market_analysis.resistance_levels:
            nearest_resistance = min([r for r in market_analysis.resistance_levels if r > current_price], default=tight_stop)
            tight_stop = min(tight_stop, nearest_resistance)
            wide_stop = min(wide_stop, nearest_resistance * 1.005)
        breakeven_point = current_price - (atr_value * 0.5)
        trailing_stop = current_price + (atr_value * 1.5 * volatility_multiplier)
        return DynamicLevels(primary_entry=primary_entry, secondary_entry=secondary_entry, primary_exit=primary_exit, secondary_exit=secondary_exit, tight_stop=tight_stop, wide_stop=wide_stop, breakeven_point=breakeven_point, trailing_stop=trailing_stop)
    def _get_trend_multiplier(self, market_analysis: MarketAnalysis) -> float:
        base_multiplier = 1.0
        if market_analysis.trend == TrendDirection.BULLISH:
            base_multiplier *= 1.2
        elif market_analysis.trend == TrendDirection.BEARISH:
            base_multiplier *= 1.2
        if market_analysis.trend_strength == TrendStrength.STRONG:
            base_multiplier *= 1.5
        elif market_analysis.trend_strength == TrendStrength.MODERATE:
            base_multiplier *= 1.2
        if market_analysis.volume_confirmation:
            base_multiplier *= 1.1
        if abs(market_analysis.trend_acceleration) > 1:
            base_multiplier *= 1.15
        return base_multiplier
    def _get_volatility_multiplier(self, market_analysis: MarketAnalysis) -> float:
        if market_analysis.volatility > 0.04:
            return 1.5
        elif market_analysis.volatility > 0.02:
            return 1.2
        else:
            return 1.0

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class LSTMModel:
    def __init__(self, input_shape=(60, 1), units=64, lr=0.001):
        self.model = None
        self.input_shape = input_shape
        self.trained = False
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        self._setup_gpu()
        self._create_model(units, lr)
        
    def _setup_gpu(self):
        """تنظیم GPU با مدیریت خطا"""
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"GPU setup successful: {len(gpus)} GPU(s) found")
                except RuntimeError as e:
                    logger.warning(f"GPU configuration error: {e}")
        except Exception as e:
            logger.warning(f"GPU setup error: {e}")
    
    def _create_model(self, units, lr):
        """ایجاد مدل LSTM"""
        try:            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                self.model = Sequential([
                    LSTM(units, 
                        input_shape=self.input_shape, 
                        return_sequences=True,
                        dropout=0.1,
                        recurrent_dropout=0.1),
                    LSTM(units // 2, 
                        return_sequences=False,
                        dropout=0.1,
                        recurrent_dropout=0.1),
                    Dropout(0.2),
                    Dense(25, activation='relu'),
                    Dense(1, activation='linear')
                ])
                
                optimizer = Adam(learning_rate=lr, clipnorm=1.0)
                self.model.compile(
                    optimizer=optimizer, 
                    loss='huber',
                    metrics=['mae']
                )
                
            logger.info("LSTM model created successfully")
            
        except Exception as e:
            logger.error(f"LSTM model creation failed: {e}")
            self.model = None
    
    def prepare_sequences(self, series: pd.Series, window: int = None, for_training: bool = True):
        """آماده‌سازی توالی‌ها با بهبود"""
        try:
            if window is None:
                window = self.input_shape[0]
                
            values = pd.to_numeric(series.values, errors='coerce')
            values = values[~np.isnan(values)]
            
            if len(values) < window + 10:
                logger.warning(f"Insufficient data: {len(values)} < {window + 10}")
                return np.array([]), np.array([])
            
            if for_training:
                values_scaled = self.scaler.fit_transform(values.reshape(-1, 1)).flatten()
                self.is_fitted = True
            else:
                if not self.is_fitted:
                    logger.error("Scaler not fitted yet")
                    return np.array([]), np.array([])
                values_scaled = self.scaler.transform(values.reshape(-1, 1)).flatten()
            
            X, y = [], []
            for i in range(window, len(values_scaled)):
                X.append(values_scaled[i-window:i])
                y.append(values_scaled[i])
            
            X = np.array(X)
            y = np.array(y)
            
            if X.ndim == 2:
                X = X.reshape((X.shape[0], X.shape[1], 1))
                
            logger.info(f"Sequences prepared: X.shape={X.shape}, y.shape={y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing sequences: {e}")
            return np.array([]), np.array([])
    
    def fit(self, X, y, epochs=10, batch_size=32, verbose=0, validation_split=0.2):
        """تمرین مدل با بهبود"""
        if self.model is None:
            logger.error("Model not initialized")
            return False
            
        try:
            if X.size == 0 or y.size == 0:
                logger.error("Empty training data")
                return False
                
            if len(X.shape) != 3:
                logger.error(f"Invalid X shape: {X.shape}. Expected 3D array")
                return False
                
            if X.shape[1] != self.input_shape[0] or X.shape[2] != self.input_shape[1]:
                logger.error(f"Input shape mismatch: {X.shape} vs expected {self.input_shape}")
                return False
            
            if np.isnan(X).any() or np.isnan(y).any() or np.isinf(X).any() or np.isinf(y).any():
                logger.warning("Invalid values found, cleaning...")
                X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
                y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
            
            epochs = max(5, min(epochs, 100))
            batch_size = max(8, min(batch_size, len(X) // 4))
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=0
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7,
                    verbose=0
                )
            ]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                history = self.model.fit(
                    X, y,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    verbose=verbose,
                    shuffle=True
                )
            
            final_loss = history.history['loss'][-1]
            if final_loss < float('inf') and not np.isnan(final_loss):
                self.trained = True
                logger.info(f"Model trained successfully. Final loss: {final_loss:.6f}")
                return True
            else:
                logger.error("Training failed - invalid final loss")
                return False
                
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            self.trained = False
            return False
    
    def predict(self, X):
        """پیش‌بینی با بهبود"""
        if self.model is None:
            logger.error("Model not initialized")
            return None
            
        if not self.trained:
            logger.error("Model not trained yet")
            return None
        
        try:
            if X.size == 0:
                logger.warning("Empty input data for prediction")
                return None
                
            if np.isnan(X).any() or np.isinf(X).any():
                logger.warning("Invalid input data, cleaning...")
                X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if len(X.shape) == 2:
                X = X.reshape(1, X.shape[0], X.shape[1])
            elif len(X.shape) == 1:
                X = X.reshape(1, len(X), 1)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prediction = self.model.predict(X, verbose=0)
            
            if self.is_fitted:
                prediction_scaled = self.scaler.inverse_transform(prediction.reshape(-1, 1))
                return prediction_scaled.flatten()
            else:
                return prediction.flatten()
                
        except Exception as e:
            logger.error(f"Error predicting with LSTM model: {e}")
            return None
    
    def is_ready(self):
        """بررسی آمادگی مدل"""
        return self.model is not None and self.trained and self.is_fitted

class SentimentFetcher:
    def __init__(self, cryptopanic_key: str):
        self.cryptopanic_key = cryptopanic_key
        if not cryptopanic_key or not cryptopanic_key.strip():
            logger.warning("CryptoPanic API key not provided")
        
    def fetch_fear_greed(self, max_retries=3, retry_delay=5):
        for attempt in range(max_retries):
            try:
                r = requests.get("https://api.alternative.me/fng/", timeout=10)
                if r.status_code == 200:
                    j = r.json()
                    data = j.get("data", [])
                    if data:
                        value = data[0].get("value")
                        if value is not None:
                            return int(value)
                
                logger.warning(f"Attempt {attempt+1}: Invalid response from Fear & Greed API")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt+1}: Error connecting to Fear & Greed API: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        logger.error("Failed to fetch Fear & Greed index after multiple attempts")
        return None
    
    def fetch_news(self, currencies: List[str] = ["BTC","ETH"]):
        try:
            if not self.cryptopanic_key or not self.cryptopanic_key.strip():
                logger.warning("No CryptoPanic API key available")
                return []
                
            url = "https://cryptopanic.com/api/v1/posts/"
            params = {"auth_token": self.cryptopanic_key, "currencies": ",".join(currencies)}
            r = requests.get(url, params=params, timeout=10)
            if r.ok:
                return r.json().get("results", [])
            else:
                logger.warning(f"CryptoPanic API returned status {r.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    def score_news(self, news_items: List[Dict[str,Any]]):
        score = 0
        for it in news_items:
            title = (it.get("title") or "").lower()
            if any(k in title for k in ["bull", "rally", "surge", "gain", "pump"]):
                score += 1
            if any(k in title for k in ["crash", "dump", "fall", "drop", "hack"]):
                score -= 1
        return score

class OnChainFetcher:
    def __init__(self, alchemy_url: str):
        self.alchemy_url = alchemy_url
        self.web3 = None
        if alchemy_url and alchemy_url.strip():
            self._connect()
    
    def _connect(self):
        try:
            if not self.alchemy_url or not self.alchemy_url.strip():
                logger.warning("No Alchemy URL provided, skipping Web3 connection")
                return
                
            self.web3 = Web3(Web3.HTTPProvider(self.alchemy_url))
            if not self.web3.is_connected():
                raise ConnectionError("Failed to connect to Web3 provider")
            logger.info("Successfully connected to Web3 provider")
        except Exception as e:
            logger.error(f"Error connecting to Web3 provider: {e}")
            self.web3 = None
    
    def _ensure_connected(self):
        if not self.web3 or not self.web3.is_connected():
            self._connect()
        if not self.web3 or not self.web3.is_connected():
            raise ConnectionError("Web3 provider not available")
    
    def active_addresses(self, lookback_blocks: int = 100):
        try:
            if not self.web3:
                logger.warning("Web3 not initialized")
                return None
                
            self._ensure_connected()
            block = self.web3.eth.block_number
            start = max(0, block - lookback_blocks)
            addresses = set()
            
            for i in range(start, min(block, start + 50)):
                try:
                    blk = self.web3.eth.get_block(i, full_transactions=True)
                    for tx in blk.transactions:
                        if hasattr(tx, 'from') and tx['from']:
                            addresses.add(tx['from'])
                        if hasattr(tx, 'to') and tx['to']:
                            addresses.add(tx['to'])
                except Exception as e:
                    logger.warning(f"Error processing block {i}: {e}")
                    continue
                    
            return len(addresses)
        except Exception as e:
            logger.error(f"Error fetching active addresses: {e}")
            return None
    def transaction_volume(self, lookback_blocks: int = 100):
        try:
            block = self.web3.eth.blockNumber
            start = max(0, block - lookback_blocks)
            total = 0
            for i in range(start, block):
                blk = self.web3.eth.getBlock(i, full_transactions=True)
                for tx in blk.transactions:
                    total += getattr(tx, 'value', 0) or tx.get('value', 0)
            return total
        except Exception:
            return 0
    def exchange_flow(self, exchange_addresses: List[str], lookback_blocks: int = 100):
        try:
            block = self.web3.eth.blockNumber
            start = max(0, block - lookback_blocks)
            inflow = 0
            outflow = 0
            for i in range(start, block):
                blk = self.web3.eth.getBlock(i, full_transactions=True)
                for tx in blk.transactions:
                    if tx['to'] in exchange_addresses:
                        outflow += getattr(tx, 'value', 0) or tx.get('value', 0)
                    if tx['from'] in exchange_addresses:
                        inflow += getattr(tx, 'value', 0) or tx.get('value', 0)
            return inflow, outflow
        except Exception:
            return 0, 0

class MultiTimeframeAnalyzer:
    def __init__(self, exchange_manager):
        self.exchange_manager = exchange_manager
    async def is_direction_aligned(self, symbol: str, exec_tf: str, confirm_tfs: List[str]) -> bool:
        try:
            base = await self.exchange_manager.fetch_ohlcv_data(symbol, exec_tf)
            if base.empty:
                return False
            base_trend = self._simple_trend(base)
            for tf in confirm_tfs:
                confirm = await self.exchange_manager.fetch_ohlcv_data(symbol, tf)
                if confirm.empty or self._simple_trend(confirm) != base_trend:
                    return False
            return True
        except Exception:
            return False
    def _simple_trend(self, df: pd.DataFrame) -> str:
        sma_short = df['close'].rolling(window=5).mean().iloc[-1]
        sma_long = df['close'].rolling(window=20).mean().iloc[-1]
        if sma_short > sma_long:
            return "up"
        if sma_short < sma_long:
            return "down"
        return "side"

class WalkForwardOptimizer:
    def __init__(self, trading_service):
        self.trading_service = trading_service
    async def run(self, symbol: str, timeframe: str, lookback_days: int, test_days: int):
        end = datetime.utcnow()
        start = end - timedelta(days=lookback_days + test_days)
        df = await self.trading_service.exchange_manager.fetch_ohlcv_data(symbol, timeframe, limit=2000)
        if df.empty:
            return {}
        df = df.set_index('timestamp').sort_index()
        results = []
        window = timedelta(days=lookback_days)
        step = timedelta(days=test_days)
        s = df.index[0]
        while s + window + step <= df.index[-1]:
            train = df[s:s+window]
            test = df[s+window:s+window+step]
            params = self.trading_service.signal_generator.optimize_params(train)
            perf = self.trading_service.signal_generator.test_params(test, params)
            results.append(perf)
            s = s + step
        return results

class SignalGenerator:
    def __init__(self, sentiment_fetcher: Optional[SentimentFetcher] = None, onchain_fetcher: Optional[OnChainFetcher] = None, lstm_model: Optional[LSTMModel] = None):
        self.indicators = {
            'sma_20': MovingAverageIndicator(20, "sma"),
            'sma_50': MovingAverageIndicator(50, "sma"),
            'ema_12': MovingAverageIndicator(12, "ema"),
            'ema_26': MovingAverageIndicator(26, "ema"),
            'rsi': RSIIndicator(),
            'macd': MACDIndicator(),
            'bb': BollingerBandsIndicator(),
            'stoch': StochasticIndicator(),
            'volume': VolumeIndicator(),
            'atr': ATRIndicator(),
            'ichimoku': IchimokuIndicator(),
            'williams_r': WilliamsRIndicator(),
            'cci': CCIIndicator()
        }
        self.market_analyzer = MarketConditionAnalyzer()
        self.level_calculator = DynamicLevelCalculator()
        self.sentiment_fetcher = sentiment_fetcher
        self.onchain_fetcher = onchain_fetcher
        self.lstm_model = lstm_model
        self.lstm_training_attempted = False
        
    def _train_lstm_if_needed(self, data: pd.DataFrame) -> bool:
        """تمرین LSTM در صورت نیاز"""
        if not self.lstm_model:
            return False
            
        if self.lstm_model.is_ready():
            return True
            
        if self.lstm_training_attempted:
            return False
        
        try:
            logger.info("Training LSTM model...")
            self.lstm_training_attempted = True
            
            series = data['close'].astype(float)
            X, y = self.lstm_model.prepare_sequences(series, for_training=True)
            
            if X.size == 0 or y.size == 0:
                logger.warning("Insufficient data for LSTM training")
                return False
            
            success = self.lstm_model.fit(X, y, epochs=20, batch_size=16, verbose=0)
            
            if success:
                logger.info("LSTM model trained successfully")
                return True
            else:
                logger.error("LSTM training failed")
                return False
                
        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            return False
        
    def generate_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[TradingSignal]:
        """تولید سیگنال با LSTM بهبود یافته"""
        logger.info(f"🔄 Generating signals for {symbol} on {timeframe} with {len(data)} candles")
        
        if len(data) < 60:
            logger.warning(f"⚠️ Insufficient data for {symbol} on {timeframe}: {len(data)} candles (need 60+)")
            return []
        
        if data.isnull().any().any():
            logger.warning(f"⚠️ Data contains null values for {symbol}")
            data = data.dropna()
            if len(data) < 60:
                return []
        
        indicator_results = {}
        failed_indicators = []
        
        for name, indicator in self.indicators.items():
            try:
                result = indicator.calculate(data)
                
                if (hasattr(result, 'value') and 
                    not pd.isna(result.value) and 
                    not np.isinf(result.value)):
                    indicator_results[name] = result
                    logger.debug(f"✅ {name} calculated successfully for {symbol}")
                else:
                    logger.warning(f"⚠️ Invalid result from {name} for {symbol}")
                    failed_indicators.append(name)
                    
            except Exception as e:
                logger.warning(f"⚠️ Error calculating {name} for {symbol}: {e}")
                failed_indicators.append(name)
                continue
        
        critical_indicators = ['rsi', 'macd', 'volume']
        missing_critical = [ind for ind in critical_indicators if ind not in indicator_results]

        if missing_critical:
            logger.warning(f"⚠️ Critical indicators failed for {symbol}: {', '.join(missing_critical)}")
            return []

        if len(indicator_results) < len(self.indicators) * 0.7:
            logger.warning(f"⚠️ Too many indicators failed for {symbol}: {', '.join(failed_indicators)}")
            return []

        if failed_indicators:
            logger.warning(f"❌ Failed indicators for {symbol}: {', '.join(failed_indicators)}")
        try:
            market_analysis = self.market_analyzer.analyze_market_condition(data)
            logger.debug(f"📊 Market analysis completed for {symbol}: trend={market_analysis.trend.value}")
        except Exception as e:
            logger.error(f"❌ Market analysis failed for {symbol}: {e}")
            return []
        patterns = PatternAnalyzer.detect_patterns(data)
        advanced_patterns = []
        if PatternAnalyzer.detect_flag(data):
            advanced_patterns.append("flag")
        if PatternAnalyzer.detect_wedge(data):
            advanced_patterns.append("wedge")
        if PatternAnalyzer.detect_triangle(data):
            advanced_patterns.append("triangle")
        vp = self.market_analyzer.volume_analyzer.volume_profile(data)
        vwap = self.market_analyzer.volume_analyzer.vwap(data)
        sentiment_fg = None
        sentiment_news_score = 0
        if self.sentiment_fetcher:
            try:
                sentiment_fg = self.sentiment_fetcher.fetch_fear_greed()
                if self.sentiment_fetcher.cryptopanic_key:
                    news = self.sentiment_fetcher.fetch_news()
                    sentiment_news_score = self.sentiment_fetcher.score_news(news)
            except Exception as e:
                logger.warning(f"Error fetching sentiment data: {e}")
                
        onchain_active = None
        onchain_volume = None
        if self.onchain_fetcher and self.onchain_fetcher.web3:
            try:
                onchain_active = self.onchain_fetcher.active_addresses()
                onchain_volume = self.onchain_fetcher.transaction_volume()
            except Exception as e:
                logger.warning(f"Error fetching on-chain data: {e}")
        signals = []
        buy_signal = self._evaluate_buy_signal(indicator_results, data, symbol, timeframe, market_analysis, patterns, advanced_patterns, vp, vwap, sentiment_fg, sentiment_news_score, onchain_active, onchain_volume)
        if buy_signal:
            logger.info(f"🟢 BUY signal generated for {symbol} on {timeframe} - Confidence: {buy_signal.confidence_score:.0f}")
            signals.append(buy_signal)
        sell_signal = self._evaluate_sell_signal(indicator_results, data, symbol, timeframe, market_analysis, patterns, advanced_patterns, vp, vwap, sentiment_fg, sentiment_news_score, onchain_active, onchain_volume)
        if sell_signal:
            logger.info(f"🔴 SELL signal generated for {symbol} on {timeframe} - Confidence: {sell_signal.confidence_score:.0f}")
            signals.append(sell_signal)
            
        lstm_prediction = None
        if self.lstm_model:
            try:
                if self._train_lstm_if_needed(data):
                    series = data['close'].astype(float)
                    X, _ = self.lstm_model.prepare_sequences(series, for_training=False)
                    
                    if X.size > 0:
                        last_sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
                        prediction = self.lstm_model.predict(last_sequence)
                        
                        if prediction is not None and len(prediction) > 0:
                            lstm_prediction = prediction[0]
                            logger.debug(f"LSTM prediction: {lstm_prediction}")
                        else:
                            logger.warning("LSTM prediction returned None")
                    else:
                        logger.warning("No data for LSTM prediction")
                else:
                    logger.warning("LSTM model not ready for prediction")
                    
            except Exception as e:
                logger.error(f"LSTM prediction error: {e}")
        
        if lstm_prediction is not None and signals:
            try:
                current_price = data['close'].iloc[-1]
                price_change_percent = ((lstm_prediction - current_price) / current_price) * 100
                
                for signal in signals:
                    if signal.signal_type == SignalType.BUY and price_change_percent > 0:
                        signal.confidence_score += min(abs(price_change_percent) * 2, 10)
                        signal.reasons.append(f"LSTM predicts {price_change_percent:.2f}% price increase")
                    elif signal.signal_type == SignalType.SELL and price_change_percent < 0:
                        signal.confidence_score += min(abs(price_change_percent) * 2, 10)
                        signal.reasons.append(f"LSTM predicts {price_change_percent:.2f}% price decrease")
                    
            except Exception as e:
                logger.error(f"Error applying LSTM prediction to signals: {e}")
        
        return signals
    
    def _evaluate_buy_signal(self,
                            indicators: Dict[str, IndicatorResult],
                            data: pd.DataFrame,
                            symbol: str,
                            timeframe: str,
                            market_analysis: MarketAnalysis,
                            patterns: List[str],
                            advanced_patterns: List[str],
                            vp: List[Tuple[float,float]],
                            vwap: float,
                            sentiment_fg: Optional[int],
                            sentiment_news: int,
                            onchain_active: Optional[int],
                            onchain_volume: Optional[int]
                            ) -> Optional[TradingSignal]:
        try:
            if data.empty or 'close' not in data.columns:
                logger.warning("Invalid or empty data for buy signal evaluation")
                return None
            
            score = 0
            reasons = []
            current_price = data['close'].iloc[-1]
            if 'rsi' in indicators and indicators['rsi'].interpretation == "oversold":
                score += 25
                reasons.append("RSI oversold condition")
            if 'macd' in indicators and indicators['macd'].interpretation == "bullish_crossover":
                score += 20
                reasons.append("MACD bullish crossover")
            if ('sma_20' in indicators and 'sma_50' in indicators and indicators['sma_20'].value > indicators['sma_50'].value):
                score += 15
                reasons.append("Price above SMA trend")
            if 'bb' in indicators and indicators['bb'].interpretation == "near_lower_band":
                score += 15
                reasons.append("Price near Bollinger lower band")
            if 'stoch' in indicators and indicators['stoch'].interpretation == "oversold":
                score += 10
                reasons.append("Stochastic oversold")
            if 'volume' in indicators and indicators['volume'].interpretation == "high_volume":
                score += 10
                reasons.append("High volume confirmation")
            if market_analysis.trend == TrendDirection.BULLISH:
                score += 15
                reasons.append("Overall bullish trend")
            if market_analysis.trend_strength == TrendStrength.STRONG:
                score += 10
                reasons.append("Strong trend momentum")
            if market_analysis.volume_confirmation:
                score += 8
                reasons.append("Volume trend confirmation")
            if market_analysis.trend_acceleration > 0.5:
                score += 7
                reasons.append("Positive trend acceleration")
            if 'bullish_engulfing' in patterns:
                score += 15
                reasons.append("Bullish Engulfing pattern")
            if 'double_bottom' in patterns:
                score += 15
                reasons.append("Double Bottom pattern")
            if 'inverse_head_and_shoulders' in patterns:
                score += 15
                reasons.append("Inverse Head & Shoulders pattern")
            if 'flag' in advanced_patterns:
                score += 8
                reasons.append("Flag pattern")
            if 'triangle' in advanced_patterns or 'wedge' in advanced_patterns:
                score += 7
                reasons.append("Triangle/Wedge consolidation")
            if vwap and current_price < vwap:
                score += 5
                reasons.append("Price below VWAP - potential mean reversion")
            if sentiment_fg is not None:
                if sentiment_fg < 40:
                    score += 5
                    reasons.append("Fear & Greed indicates fear - favorable for buys")
                elif sentiment_fg > 70:
                    score -= 5
                    reasons.append("High greed - caution")
            if sentiment_news > 0:
                score += 3
                reasons.append("Positive news sentiment")
            if onchain_active and onchain_active > 1000:
                score += 2
                reasons.append("Healthy on-chain activity")
            if score >= 60:
                try:
                    dynamic_levels = self.level_calculator.calculate_dynamic_levels(data, SignalType.BUY, market_analysis)
                    return TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        entry_price=dynamic_levels.primary_entry,
                        exit_price=dynamic_levels.primary_exit,
                        stop_loss=dynamic_levels.tight_stop,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        confidence_score=score,
                        reasons=reasons,
                        risk_reward_ratio=self._calculate_risk_reward(dynamic_levels.primary_entry, dynamic_levels.primary_exit, dynamic_levels.tight_stop),
                        predicted_profit=((dynamic_levels.primary_exit - dynamic_levels.primary_entry) / dynamic_levels.primary_entry) * 100,
                        volume_analysis=self.market_analyzer.volume_analyzer.analyze_volume_pattern(data),
                        market_context=self._create_market_context(market_analysis),
                        dynamic_levels={
                            'primary_entry': dynamic_levels.primary_entry,
                            'secondary_entry': dynamic_levels.secondary_entry,
                            'primary_exit': dynamic_levels.primary_exit,
                            'secondary_exit': dynamic_levels.secondary_exit,
                            'tight_stop': dynamic_levels.tight_stop,
                            'wide_stop': dynamic_levels.wide_stop,
                            'breakeven_point': dynamic_levels.breakeven_point,
                            'trailing_stop': dynamic_levels.trailing_stop
                        }
                    )
                except Exception as e:
                    logger.error(f"Error calculating dynamic levels: {e}")
                    return None

            return None
        except Exception as e:
            logger.error(f"Error evaluating buy signal: {e}")
            return None
        
    def _evaluate_sell_signal(self,
                            indicators: Dict[str, IndicatorResult],
                            data: pd.DataFrame,
                            symbol: str,
                            timeframe: str,
                            market_analysis: MarketAnalysis,
                            patterns: List[str],
                            advanced_patterns: List[str],
                            vp: List[Tuple[float,float]],
                            vwap: float,
                            sentiment_fg: Optional[int],
                            sentiment_news: int,
                            onchain_active: Optional[int],
                            onchain_volume: Optional[int]
                            ) -> Optional[TradingSignal]:
        try:
            if data.empty or 'close' not in data.columns:
                logger.warning("Invalid or empty data for sell signal evaluation")
                return None
            
            score = 0
            reasons = []
            current_price = data['close'].iloc[-1]
            if 'rsi' in indicators and indicators['rsi'].interpretation == "overbought":
                score += 25
                reasons.append("RSI overbought condition")
            if 'macd' in indicators and indicators['macd'].interpretation == "bearish_crossover":
                score += 20
                reasons.append("MACD bearish crossover")
            if ('sma_20' in indicators and 'sma_50' in indicators and indicators['sma_20'].value < indicators['sma_50'].value):
                score += 15
                reasons.append("Price below SMA trend")
            if 'bb' in indicators and indicators['bb'].interpretation == "near_upper_band":
                score += 15
                reasons.append("Price near Bollinger upper band")
            if 'stoch' in indicators and indicators['stoch'].interpretation == "overbought":
                score += 10
                reasons.append("Stochastic overbought")
            if 'volume' in indicators and indicators['volume'].interpretation == "high_volume":
                score += 10
                reasons.append("High volume confirmation")
            if market_analysis.trend == TrendDirection.BEARISH:
                score += 15
                reasons.append("Overall bearish trend")
            if market_analysis.trend_strength == TrendStrength.STRONG:
                score += 10
                reasons.append("Strong trend momentum")
            if market_analysis.volume_confirmation:
                score += 8
                reasons.append("Volume trend confirmation")
            if market_analysis.trend_acceleration < -0.5:
                score += 7
                reasons.append("Negative trend acceleration")
            if 'bearish_engulfing' in patterns:
                score += 15
                reasons.append("Bearish Engulfing pattern")
            if 'double_top' in patterns:
                score += 15
                reasons.append("Double Top pattern")
            if 'head_and_shoulders' in patterns:
                score += 15
                reasons.append("Head & Shoulders pattern")
            if 'flag' in advanced_patterns:
                score += 6
                reasons.append("Flag breakdown")
            if vwap and current_price > vwap:
                score += 5
                reasons.append("Price above VWAP - potential mean reversion")
            if sentiment_fg is not None:
                if sentiment_fg > 70:
                    score += 5
                    reasons.append("Greed indicated - favorable for sells")
                elif sentiment_fg < 30:
                    score -= 5
                    reasons.append("Extreme fear - caution on sells")
            if sentiment_news < 0:
                score += 3
                reasons.append("Negative news sentiment")
            if onchain_active and onchain_active < 500:
                score += 2
                reasons.append("Low on-chain activity - weakness")
            if score >= 60:
                try:
                    dynamic_levels = self.level_calculator.calculate_dynamic_levels(data, SignalType.SELL, market_analysis)
                    return TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        entry_price=dynamic_levels.primary_entry,
                        exit_price=dynamic_levels.primary_exit,
                        stop_loss=dynamic_levels.tight_stop,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        confidence_score=score,
                        reasons=reasons,
                        risk_reward_ratio=self._calculate_risk_reward(dynamic_levels.primary_entry, dynamic_levels.primary_exit, dynamic_levels.tight_stop),
                        predicted_profit=((dynamic_levels.primary_entry - dynamic_levels.primary_exit) / dynamic_levels.primary_entry) * 100,
                        volume_analysis=self.market_analyzer.volume_analyzer.analyze_volume_pattern(data),
                        market_context=self._create_market_context(market_analysis),
                        dynamic_levels={
                            'primary_entry': dynamic_levels.primary_entry,
                            'secondary_entry': dynamic_levels.secondary_entry,
                            'primary_exit': dynamic_levels.primary_exit,
                            'secondary_exit': dynamic_levels.secondary_exit,
                            'tight_stop': dynamic_levels.tight_stop,
                            'wide_stop': dynamic_levels.wide_stop,
                            'breakeven_point': dynamic_levels.breakeven_point,
                            'trailing_stop': dynamic_levels.trailing_stop
                        }
                    )
                except Exception as e:
                    logger.error(f"Error calculating dynamic levels: {e}")
                    return None
            return None
        except Exception as e:
            logger.error(f"Error evaluating sell signal: {e}")
            return None
    def _calculate_risk_reward(self, entry: float, exit: float, stop_loss: float) -> float:
        if entry == stop_loss:
            return 0
        potential_profit = abs(exit - entry)
        potential_loss = abs(entry - stop_loss)
        return potential_profit / potential_loss if potential_loss > 0 else 0
    def _create_market_context(self, market_analysis: MarketAnalysis) -> Dict[str, Any]:
        return {'trend': market_analysis.trend.value, 'trend_strength': market_analysis.trend_strength.value, 'volatility': market_analysis.volatility, 'momentum_score': market_analysis.momentum_score, 'market_condition': market_analysis.market_condition.value, 'volume_trend': market_analysis.volume_trend, 'trend_acceleration': market_analysis.trend_acceleration, 'volume_confirmation': market_analysis.volume_confirmation}
    def optimize_params(self, train: pd.DataFrame) -> Dict[str, Any]:
        return {}
    def test_params(self, test: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}

class ExchangeManager:
    def __init__(self):
        self.exchange = None
        self._lock = asyncio.Lock()
        self.db_path = 'trading_bot.db'
        self.ohlcv_cache = {}
        
    async def init_database(self):
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS ohlcv (
                        symbol TEXT, 
                        timeframe TEXT, 
                        timestamp INTEGER, 
                        open REAL, 
                        high REAL, 
                        low REAL, 
                        close REAL, 
                        volume REAL, 
                        PRIMARY KEY(symbol, timeframe, timestamp)
                    )
                """)
                await db.commit()
                logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")

    @asynccontextmanager
    async def get_exchange(self):
        async with self._lock:
            if self.exchange is None:
                self.exchange = ccxt.coinex({
                    'apiKey': os.getenv('COINEX_API_KEY', COINEX_API_KEY), 
                    'secret': os.getenv('COINEX_SECRET', COINEX_SECRET), 
                    'sandbox': False, 
                    'enableRateLimit': True, 
                    'timeout': 30000, 
                    'options': {'defaultType': 'spot'}
                })
            try:
                yield self.exchange
            except Exception:
                pass

    async def close_exchange(self):
        async with self._lock:
            if self.exchange:
                try:
                    await self.exchange.close()
                    logger.info("Exchange connection closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing exchange: {e}")
                finally:
                    self.exchange = None

    async def _save_ohlcv_to_db(self, df: pd.DataFrame, symbol: str, timeframe: str):
        try:
            async with aiosqlite.connect(self.db_path) as db:
                for idx, row in df.iterrows():
                    try:
                        if hasattr(row['timestamp'], 'timestamp'):
                            ts = int(row['timestamp'].timestamp() * 1000)
                        else:
                            ts = int(row['timestamp'])
                        await db.execute(
                            "INSERT OR IGNORE INTO ohlcv VALUES (?,?,?,?,?,?,?,?)", 
                            (symbol, timeframe, ts, float(row['open']), 
                            float(row['high']), float(row['low']), 
                            float(row['close']), float(row['volume']))
                        )
                    except (ValueError, TypeError):
                        continue
                await db.commit()
        except Exception as e:
            logger.error(f"Database save error: {e}")
                
    async def fetch_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        logger.info(f"Fetching OHLCV data for {symbol} on {timeframe} (limit: {limit})")
        rate_limiter.wait_if_needed(f"ohlcv_{symbol}")
        
        try:
            async with self.get_exchange() as exchange:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                        break
                    except ccxt.NetworkError as ne:
                        logger.warning(f"Network error for {symbol} (attempt {attempt+1}): {ne}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(5)
                        else:
                            raise
                    except ccxt.ExchangeError as ee:
                        logger.error(f"Exchange error for {symbol}: {ee}")
                        return pd.DataFrame()
            
            if not ohlcv:
                logger.warning(f"No OHLCV data received for {symbol} on {timeframe}")
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            
            if df.empty:
                logger.warning(f"Empty OHLCV data for {symbol}")
                return pd.DataFrame()
            
            df = df.dropna()
            if df.empty:
                logger.warning(f"All data invalid after cleaning for {symbol}")
                return pd.DataFrame()
                
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            except Exception as e:
                logger.error(f"Timestamp conversion error: {e}")
                return pd.DataFrame()
            
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=numeric_columns)
            
            if df.empty:
                logger.warning(f"No valid numeric data for {symbol}")
                return pd.DataFrame()
            
            await self._save_ohlcv_to_db(df, symbol, timeframe)
            
            return df.sort_values('timestamp').reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Unexpected error fetching {symbol} on {timeframe}: {e}")
            return pd.DataFrame()

class SignalRanking:
    @staticmethod
    def rank_signals(signals: List[TradingSignal]) -> List[TradingSignal]:
        def signal_score(signal: TradingSignal) -> float:
            base_score = signal.confidence_score
            rr_bonus = min(signal.risk_reward_ratio * 10, 20)
            profit_bonus = min(abs(signal.predicted_profit) * 2, 15)
            volume_bonus = 0
            if signal.volume_analysis.get('volume_ratio', 1) > 1.5:
                volume_bonus = 10
            trend_bonus = 0
            trend_strength = signal.market_context.get('trend_strength', 'weak')
            if trend_strength == 'strong':
                trend_bonus = 15
            elif trend_strength == 'moderate':
                trend_bonus = 10
            acceleration_bonus = 0
            trend_acceleration = abs(signal.market_context.get('trend_acceleration', 0))
            if trend_acceleration > 1:
                acceleration_bonus = 8
            volume_confirmation_bonus = 0
            if signal.market_context.get('volume_confirmation', False):
                volume_confirmation_bonus = 5
            return base_score + rr_bonus + profit_bonus + volume_bonus + trend_bonus + acceleration_bonus + volume_confirmation_bonus
        return sorted(signals, key=signal_score, reverse=True)

class ConfigManager:
    DEFAULT_CONFIG = {'symbols': SYMBOLS, 'timeframes': TIME_FRAMES, 'min_confidence_score': 50, 'max_signals_per_timeframe': 3, 'risk_reward_threshold': 1.5}
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    def _load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return {**self.DEFAULT_CONFIG, **json.load(f)}
            except Exception as e:
                logger.warning(f"Error loading config: {e}. Using defaults.")
        return self.DEFAULT_CONFIG.copy()
    def save_config(self):
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    def get(self, key: str, default=None):
        return self.config.get(key, default)
    def set(self, key: str, value):
        self.config[key] = value
        self.save_config()

class TradingBotService:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.exchange_manager = ExchangeManager()
        
        cryptopanic_key = self.config.get('cryptopanic_key', os.getenv('CRYPTOPANIC_KEY', ''))
        alchemy_url = self.config.get('alchemy_url', os.getenv('ALCHEMY_URL', ''))
        
        self.sentiment_fetcher = None
        self.onchain_fetcher = None
        
        if cryptopanic_key and cryptopanic_key.strip():
            self.sentiment_fetcher = SentimentFetcher(cryptopanic_key)
        else:
            logger.warning("Sentiment analysis disabled - no CryptoPanic API key")
            
        if alchemy_url and alchemy_url.strip():
            self.onchain_fetcher = OnChainFetcher(alchemy_url)
        else:
            logger.warning("On-chain analysis disabled - no Alchemy URL")
        
        try:
            self.lstm_model = LSTMModel(
                input_shape=(60, 1),
                units=50,
                lr=0.001
            )
            logger.info("LSTM model initialized")
        except Exception as e:
            logger.error(f"LSTM initialization failed: {e}")
            self.lstm_model = None
        
        self.signal_generator = SignalGenerator(
            sentiment_fetcher=self.sentiment_fetcher,
            onchain_fetcher=self.onchain_fetcher,
            lstm_model=self.lstm_model
        )
        
        self.signal_ranking = SignalRanking()
        
    async def initialize(self):
        await self.exchange_manager.init_database()
        
    async def analyze_symbol(self, symbol: str, timeframe: str) -> List[TradingSignal]:
        logger.info(f"🔍 Starting analysis for {symbol} on {timeframe}")
        try:
            data = await self.exchange_manager.fetch_ohlcv_data(symbol, timeframe)
            if data.empty:
                logger.warning(f"⚠️ No data available for {symbol} on {timeframe}")
                return []
            if len(data) < 50:
                logger.warning(f"⚠️ Insufficient data for {symbol} on {timeframe}: {len(data)} candles")
                return []
            data = data.rename(columns={'timestamp':'timestamp','open':'open','high':'high','low':'low','close':'close','volume':'volume'})
            all_signals = self.signal_generator.generate_signals(data, symbol, timeframe)
            min_confidence = self.config.get('min_confidence_score', 60)
            qualified_signals = [s for s in all_signals if s.confidence_score >= min_confidence]
            if qualified_signals:
                logger.info(f"✅ Analysis complete for {symbol} on {timeframe}: {len(qualified_signals)} qualified signals")
            else:
                logger.debug(f"ℹ️ No qualified signals for {symbol} on {timeframe} (min confidence: {min_confidence})")
            return qualified_signals
        except Exception as e:
            logger.error(f"❌ Analysis failed for {symbol} on {timeframe}: {e}")
            return []
    
    async def find_best_signals_for_timeframe(self, timeframe: str) -> List[TradingSignal]:
        logger.info(f"🚀 Starting comprehensive analysis for {timeframe} timeframe")
        symbols = self.config.get('symbols', [])
        logger.info(f"📊 Analyzing {len(symbols)} symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
        all_signals = []
        successful_analyses = 0
        failed_analyses = 0
        tasks = [self.analyze_symbol(symbol, timeframe) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"❌ Analysis failed for {symbol}: {result}")
                failed_analyses += 1
                continue
            if isinstance(result, list):
                all_signals.extend(result)
                successful_analyses += 1
                if result:
                    logger.debug(f"✅ {symbol}: {len(result)} signals found")
        logger.info(f"📈 Analysis summary for {timeframe}: {successful_analyses} successful, {failed_analyses} failed")
        if not all_signals:
            logger.info(f"ℹ️ No signals found in {timeframe} timeframe")
            return []
        ranked_signals = self.signal_ranking.rank_signals(all_signals)
        max_signals = self.config.get('max_signals_per_timeframe', 3)
        top_signals = ranked_signals[:max_signals]
        logger.info(f"🏆 Top {len(top_signals)} signals selected for {timeframe}")
        for i, signal in enumerate(top_signals, 1):
            logger.info(f"  #{i}: {signal.symbol} {signal.signal_type.value.upper()} (confidence: {signal.confidence_score:.0f}, profit: {signal.predicted_profit:.2f}%)")
        return top_signals
    async def get_comprehensive_analysis(self) -> Dict[str, List[TradingSignal]]:
        results = {}
        for timeframe in TIME_FRAMES:
            logger.info(f"Analyzing timeframe: {timeframe}")
            signals = await self.find_best_signals_for_timeframe(timeframe)
            results[timeframe] = signals
        return results
    
    async def cleanup(self):
        await self.exchange_manager.close_exchange()

class BacktestingEngine:
    def __init__(self, trading_service: TradingBotService):
        self.trading_service = trading_service
    async def run_backtest(self, symbol: str, timeframe: str, start: str, end: str, initial_capital: float) -> Dict[str, float]:
        data = await self.trading_service.exchange_manager.fetch_ohlcv_data(symbol, timeframe, limit=1000)
        if data.empty:
            return {}
        data = data[(data['timestamp'] >= pd.to_datetime(start)) & (data['timestamp'] <= pd.to_datetime(end))]
        if data.empty:
            return {}
        returns = data['close'].pct_change().dropna()
        if returns.empty:
            return {}
        sharpe = (returns.mean() / returns.std()) * (len(returns) ** 0.5) if returns.std() else 0
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() if not negative_returns.empty else 0
        sortino = (returns.mean() / downside_std) * (len(returns) ** 0.5) if downside_std != 0 else 0
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min() * 100
        return {"sharpe_ratio": sharpe, "sortino_ratio": sortino, "max_drawdown": max_drawdown}

class PaperTradingSimulator:
    def __init__(self, initial_balance: float):
        self.balance = initial_balance
        self.positions = {}
        self.trade_history = []
    def simulate_trade(self, signal: TradingSignal):
        if signal.signal_type == SignalType.BUY:
            self.positions[signal.symbol] = self.balance / signal.entry_price if signal.entry_price else 0
            self.balance = 0
        elif signal.signal_type == SignalType.SELL and signal.symbol in self.positions:
            self.balance = self.positions[signal.symbol] * signal.exit_price
            del self.positions[signal.symbol]
        self.trade_history.append(signal)
    def get_performance(self) -> Dict[str, float]:
        total_trades = len(self.trade_history)
        profit_loss = self.balance - 0
        return {"total_trades": total_trades, "final_balance": self.balance, "profit_loss": profit_loss}

class MessageFormatter:
    @staticmethod
    def format_signal_message(signal: TradingSignal) -> str:
        emoji_map = {SignalType.BUY: "🟢", SignalType.SELL: "🔴", SignalType.HOLD: "🟡"}
        trend_emoji_map = {"bullish": "📈", "bearish": "📉", "sideways": "➡️"}
        strength_emoji_map = {"strong": "💪", "moderate": "🔄", "weak": "📊"}
        signal_emoji = emoji_map.get(signal.signal_type, "⚪")
        trend_emoji = trend_emoji_map.get(signal.market_context.get('trend', 'sideways'), "➡️")
        strength_emoji = strength_emoji_map.get(signal.market_context.get('trend_strength', 'weak'), "📊")
        reasons_text = "\n• ".join(signal.reasons)
        message = (
            f"{signal_emoji} **{signal.signal_type.value.upper()} SIGNAL**\n\n"
            f"📊 **Symbol:** `{signal.symbol}`\n"
            f"⏰ **Timeframe:** `{signal.timeframe}`\n\n"
            "🎯 **Dynamic Entry Levels:**\n"
            f"• Primary Entry: `${signal.dynamic_levels['primary_entry']:.4f}`\n"
            f"• Secondary Entry: `${signal.dynamic_levels['secondary_entry']:.4f}`\n\n"
            "💰 **Dynamic Exit Levels:**\n"
            f"• Primary Target: `${signal.dynamic_levels['primary_exit']:.4f}`\n"
            f"• Secondary Target: `${signal.dynamic_levels['secondary_exit']:.4f}`\n\n"
            "🛑 **Dynamic Stop Levels:**\n"
            f"• Tight Stop: `${signal.dynamic_levels['tight_stop']:.4f}`\n"
            f"• Wide Stop: `${signal.dynamic_levels['wide_stop']:.4f}`\n"
            f"• Trailing Stop: `${signal.dynamic_levels['trailing_stop']:.4f}`\n\n"
            "⚡ **Advanced Levels:**\n"
            f"• Breakeven Point: `${signal.dynamic_levels['breakeven_point']:.4f}`\n\n"
            "📈 **Profit Analysis:**\n"
            f"• Expected Profit: `{signal.predicted_profit:.2f}%`\n"
            f"• Risk/Reward Ratio: `{signal.risk_reward_ratio:.2f}`\n"
            f"• Confidence Score: `{signal.confidence_score:.0f}/100`\n\n"
            f"{trend_emoji} **Market Context:**\n"
            f"• Trend: {signal.market_context.get('trend', 'Unknown').title()} {strength_emoji}\n"
            f"• Trend Strength: {signal.market_context.get('trend_strength', 'Unknown').title()}\n"
            f"• Volatility: {signal.market_context.get('volatility', 0):.1%}\n"
            f"• Volume Trend: {signal.market_context.get('volume_trend', 'Unknown').title()}\n"
            f"• Momentum Score: {signal.market_context.get('momentum_score', 0):.2f}%\n"
            f"• Trend Acceleration: {signal.market_context.get('trend_acceleration', 0):.2f}%\n"
            f"• Volume Confirmation: {'✅' if signal.market_context.get('volume_confirmation', False) else '❌'}\n\n"
            f"📋 **Analysis Reasons:**\n• {reasons_text}\n\n"
            f"🕐 **Generated:** {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return message
    @staticmethod
    def format_summary_message(timeframe_results: Dict[str, List[TradingSignal]]) -> str:
        total_signals = sum(len(signals) for signals in timeframe_results.values())
        if total_signals == 0:
            return "📊 No signals found in any timeframe."
        summary = f"📊 Found {total_signals} signal(s) across all timeframes.\n\n"
        for timeframe, signals in timeframe_results.items():
            if signals:
                summary += f"⏰ {timeframe.upper()}: {len(signals)} signal(s)\n"
        return summary

class TelegramBotHandler:
    def __init__(self, bot_token: str, config_manager: ConfigManager):
        self.bot_token = bot_token
        self.config = config_manager
        self.trading_service = TradingBotService(config_manager)
        self.formatter = MessageFormatter()
        self.user_sessions = {}

    async def initialize(self):
        await self.trading_service.initialize()
        
    def create_application(self) -> Application:
        application = Application.builder().token(self.bot_token).build()
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("config", self.config_command))
        application.add_handler(CommandHandler("quick", self.quick_analysis))
        application.add_handler(CallbackQueryHandler(self.button_callback))
        return application
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        logger.info(f"User {user_id} started the bot")
        keyboard = [[InlineKeyboardButton("🚀 Full Analysis", callback_data="full_analysis"), InlineKeyboardButton("⚡ Quick Scan", callback_data="quick_scan")],[InlineKeyboardButton("⚙️ Settings", callback_data="settings")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        welcome_message = ("🤖 **Trading Signal Bot**\n\n" "Choose an option to get trading signals:")
        await update.message.reply_text(welcome_message, reply_markup=reply_markup, parse_mode='Markdown')
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()
        if query.data == "full_analysis":
            await self.run_full_analysis(query)
        elif query.data == "quick_scan":
            await self.run_quick_scan(query)
        elif query.data == "settings":
            await self.show_settings(query)
    async def run_full_analysis(self, query) -> None:
        user_id = query.from_user.id
        logger.info(f"User {user_id} started full analysis")
        try:
            timeframes = self.config.get('timeframes', TIME_FRAMES)
            results = {}
            total_signals = 0
            for i, timeframe in enumerate(timeframes, 1):
                await query.edit_message_text(f"🔄 Analyzing {timeframe}... ({i}/{len(timeframes)})", parse_mode='Markdown')
                signals = await self.trading_service.find_best_signals_for_timeframe(timeframe)
                results[timeframe] = signals
                total_signals += len(signals)
            await query.edit_message_text(f"✅ Analysis complete. Found {total_signals} signal(s).", parse_mode='Markdown')
            signal_count = 0
            for timeframe, signals in results.items():
                for signal in signals:
                    signal_count += 1
                    signal_message = self.formatter.format_signal_message(signal)
                    await query.message.reply_text(signal_message, parse_mode='Markdown')
                    await asyncio.sleep(2)
            if signal_count == 0:
                await query.message.reply_text("No signals found.", parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Full analysis failed for user {user_id}: {e}")
            await query.edit_message_text(f"❌ Analysis Error: {str(e)}", parse_mode='Markdown')
    async def run_quick_scan(self, query) -> None:
        await query.edit_message_text("⚡ Quick scan in progress...", parse_mode='Markdown')
        try:
            signals = await self.trading_service.find_best_signals_for_timeframe('1m')
            if not signals:
                await query.edit_message_text("❌ No signals found in 1m timeframe", parse_mode='Markdown')
                return
            await query.edit_message_text(f"✅ Found {len(signals)} signal(s)", parse_mode='Markdown')
            for signal in signals:
                signal_message = self.formatter.format_signal_message(signal)
                await query.message.reply_text(signal_message, parse_mode='Markdown')
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Error in quick scan: {e}")
            await query.edit_message_text(f"❌ Quick Scan Error: {str(e)}", parse_mode='Markdown')
    async def show_settings(self, query) -> None:
        config_info = (f"⚙️ **Settings**\n\n" f"📊 Symbols: {len(self.config.get('symbols', []))}\n" f"⏰ Timeframes: {', '.join(self.config.get('timeframes', []))}\n" f"🎯 Min Confidence: {self.config.get('min_confidence_score', 60)}\n")
        await query.edit_message_text(config_info, parse_mode='Markdown')
    async def config_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        config_text = (f"⚙️ **Configuration**\n\n" f"• Symbols: {len(self.config.get('symbols', []))}\n" f"• Timeframes: {', '.join(self.config.get('timeframes', []))}\n" f"• Min Confidence: {self.config.get('min_confidence_score', 60)}\n")
        await update.message.reply_text(config_text, parse_mode='Markdown')
    async def quick_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        progress_msg = await update.message.reply_text("⚡ Quick analysis starting...", parse_mode='Markdown')
        try:
            signals = await self.trading_service.find_best_signals_for_timeframe('1m')
            if not signals:
                await progress_msg.edit_text("❌ No signals found in 1m timeframe", parse_mode='Markdown')
                return
            await progress_msg.edit_text(f"✅ Found {len(signals)} signal(s)", parse_mode='Markdown')
            for signal in signals:
                signal_message = self.formatter.format_signal_message(signal)
                await update.message.reply_text(signal_message, parse_mode='Markdown')
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Error in quick analysis command: {e}")
            await progress_msg.edit_text(f"❌ Quick Analysis Error: {str(e)}", parse_mode='Markdown')
    async def cleanup(self):
        await self.trading_service.cleanup()
        

@asynccontextmanager
async def create_bot_application(bot_token: str, config_manager: ConfigManager):
    bot_handler = None
    application = None
    
    try:
        bot_handler = TelegramBotHandler(bot_token, config_manager)
        await bot_handler.initialize()
        
        application = bot_handler.create_application()
        
        logger.info("🤖 Bot application created successfully")
        yield application, bot_handler
        
    except Exception as e:
        logger.error(f"Error creating bot application: {e}")
        raise
    finally:
        logger.info("🧹 Starting application cleanup...")
        
        if bot_handler:
            try:
                await bot_handler.cleanup()
                logger.info("✅ Bot handler cleanup completed")
            except Exception as e:
                logger.error(f"Error cleaning up bot handler: {e}")

async def main_telegram():
    logger.info("🚀 Starting Trading Signal Bot...")
    
    bot_token = os.getenv('BOT_TOKEN') or BOT_TOKEN
    if not bot_token or not bot_token.strip():
        logger.error("❌ BOT_TOKEN environment variable is required")
        return
    
    config_manager = ConfigManager()
    logger.info("⚙️ Configuration loaded successfully")
    
    application = None
    
    try:
        async with create_bot_application(bot_token, config_manager) as (app, bot_handler):
            application = app
            
            logger.info("🤖 Bot is ready and waiting for commands...")
            
            await application.initialize()
            await application.start()
            
            try:
                await application.updater.start_polling(
                    allowed_updates=Update.ALL_TYPES,
                    drop_pending_updates=True
                )
                
                stop_event = asyncio.Event()
                await stop_event.wait()
                
            except (KeyboardInterrupt, SystemExit):
                logger.info("⏹️ Received shutdown signal")
            finally:
                if application.updater.running:
                    await application.updater.stop()
                    
                if application.running:
                    await application.stop()
                    
                await application.shutdown()
                    
    except KeyboardInterrupt:
        logger.info("⏹️ Bot stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"💥 Bot crashed with error: {e}")
        import traceback
        logger.error(f"📋 Traceback:\n{traceback.format_exc()}")
    
    logger.info("✅ Bot shutdown completed successfully")

def main():
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main_telegram())
        
    except KeyboardInterrupt:
        logger.info("⏹️ Bot stopped by user")
    except Exception as e:
        logger.error(f"💥 Main function error: {e}")
    finally:
        try:
            loop = asyncio.get_event_loop()
            if loop and not loop.is_closed():
                pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
                if pending_tasks:
                    logger.info(f"Cancelling {len(pending_tasks)} pending tasks...")
                    for task in pending_tasks:
                        task.cancel()
                    
                loop.close()
        except RuntimeError:
            pass
        
if __name__ == "__main__":
    main()