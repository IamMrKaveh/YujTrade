from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import ccxt.async_support as ccxt
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import numpy as np
from datetime import datetime, timedelta
import os
from logger_config import logger
from config.constants import BOT_TOKEN, SYMBOLS, TIME_FRAMES

from typing import Dict, List, Tuple, Optional, Protocol
import warnings

warnings.filterwarnings('ignore')

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class MarketTrend(Enum):
    BULLISH = "صعودی"
    BEARISH = "نزولی" 
    SIDEWAYS = "خنثی"


@dataclass
class Signal:
    signal_type: SignalType
    entry_price: float
    exit_price: float
    timestamp: pd.Timestamp
    confidence: float
    reasons: List[str]
    risk_reward_ratio: float
    stop_loss: Optional[float] = None


@dataclass
class MarketAnalysis:
    trend: MarketTrend
    volatility: float
    momentum: float
    strength: float
    support_levels: List[float]
    resistance_levels: List[float]
    risk_level: str


class IndicatorCalculator(Protocol):
    def calculate(self, data: pd.Series, **kwargs) -> pd.Series:
        ...


class TechnicalIndicators:
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period, min_periods=period//2).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def wma(data: pd.Series, period: int) -> pd.Series:
        weights = np.arange(1, period + 1)
        return data.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                  k_period: int = 14, d_period: int = 3, smooth_k: int = 3) -> Tuple[pd.Series, pd.Series]:
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_raw = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k_percent = k_raw.rolling(window=smooth_k).mean()
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent.fillna(50), d_percent.fillna(50)
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return wr.fillna(-50)
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci.fillna(0)
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.fillna(tr.mean())
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = TechnicalIndicators.atr(high, low, close, 1)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx.fillna(25)
    
    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi.fillna(50)


class PatternDetector:
    @staticmethod
    def detect_hammer(open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        body = abs(close - open_price)
        lower_shadow = open_price.combine(close, min) - low
        upper_shadow = high - open_price.combine(close, max)
        
        hammer = (
            (lower_shadow > 2 * body) &
            (upper_shadow < 0.3 * body) &
            (body > 0)
        )
        return hammer
    
    @staticmethod
    def detect_doji(open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        body = abs(close - open_price)
        total_range = high - low
        
        doji = body < (0.1 * total_range)
        return doji
    
    @staticmethod
    def detect_engulfing(open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        prev_open = open_price.shift(1)
        prev_close = close.shift(1)
        
        bullish_engulfing = (
            (prev_close < prev_open) &
            (close > open_price) &
            (open_price < prev_close) &
            (close > prev_open)
        )
        
        bearish_engulfing = (
            (prev_close > prev_open) &
            (close < open_price) &
            (open_price > prev_close) &
            (close < prev_open)
        )
        
        return bullish_engulfing | bearish_engulfing


class VolumeAnalyzer:
    @staticmethod
    def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def volume_weighted_average_price(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    @staticmethod
    def accumulation_distribution(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        ad = (clv * volume).cumsum()
        return ad


class SupportResistanceFinder:
    @staticmethod
    def find_pivot_points(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 5) -> Dict[str, List[float]]:
        highs = high.rolling(window=window*2+1, center=True).max()
        lows = low.rolling(window=window*2+1, center=True).min()
        
        resistance_points = []
        support_points = []
        
        for i in range(window, len(high) - window):
            if high.iloc[i] == highs.iloc[i] and high.iloc[i] > high.iloc[i-window:i].max() and high.iloc[i] > high.iloc[i+1:i+window+1].max():
                resistance_points.append(float(high.iloc[i]))
            
            if low.iloc[i] == lows.iloc[i] and low.iloc[i] < low.iloc[i-window:i].min() and low.iloc[i] < low.iloc[i+1:i+window+1].min():
                support_points.append(float(low.iloc[i]))
        
        return {
            'resistance': sorted(set(resistance_points), reverse=True)[:5],
            'support': sorted(set(support_points))[-5:]
        }
    
    @staticmethod
    def find_fibonacci_levels(high: pd.Series, low: pd.Series, period: int = 50) -> Dict[str, float]:
        recent_high = high.tail(period).max()
        recent_low = low.tail(period).min()
        diff = recent_high - recent_low
        
        levels = {
            'level_0': recent_low,
            'level_236': recent_low + 0.236 * diff,
            'level_382': recent_low + 0.382 * diff,
            'level_50': recent_low + 0.5 * diff,
            'level_618': recent_low + 0.618 * diff,
            'level_786': recent_low + 0.786 * diff,
            'level_100': recent_high
        }
        
        return levels


class MarketRegimeDetector:
    @staticmethod
    def detect_volatility_regime(returns: pd.Series, window: int = 20) -> pd.Series:
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        vol_median = rolling_vol.median()
        
        regime = pd.Series(index=returns.index, dtype=str)
        regime[rolling_vol < vol_median * 0.7] = 'low_vol'
        regime[rolling_vol > vol_median * 1.3] = 'high_vol'
        regime[(rolling_vol >= vol_median * 0.7) & (rolling_vol <= vol_median * 1.3)] = 'normal_vol'
        
        return regime.fillna('normal_vol')
    
    @staticmethod
    def detect_trend_strength(close: pd.Series, period: int = 20) -> pd.Series:
        sma = close.rolling(window=period).mean()
        distance = abs(close - sma) / sma
        
        strength = pd.Series(index=close.index, dtype=str)
        strength[distance < 0.02] = 'weak'
        strength[distance > 0.05] = 'strong'
        strength[(distance >= 0.02) & (distance <= 0.05)] = 'moderate'
        
        return strength.fillna('moderate')


class SignalGenerator(ABC):
    @abstractmethod
    def generate(self, df: pd.DataFrame) -> List[Signal]:
        pass


class TrendFollowingSignals(SignalGenerator):
    def generate(self, df: pd.DataFrame) -> List[Signal]:
        signals = []
        
        for i in range(max(50, len(df) - 10), len(df)):
            if i < 1:
                continue
                
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            confidence = 0
            reasons = []
            
            if (current['sma_20'] > current['sma_50'] and 
                previous['sma_20'] <= previous['sma_50']):
                confidence += 25
                reasons.append("تقاطع طلایی SMA")
            
            if (current['ema_12'] > current['ema_26'] and
                current['close'] > current['ema_12']):
                confidence += 20
                reasons.append("ترند EMA صعودی")
            
            if current['adx'] > 25 and current['sma_20'] > current['sma_50']:
                confidence += 15
                reasons.append("ترند قوی صعودی")
            
            if confidence >= 40:
                entry_price = current['close']
                atr_value = current.get('atr', entry_price * 0.02)
                
                signals.append(Signal(
                    signal_type=SignalType.BUY,
                    entry_price=entry_price,
                    exit_price=entry_price + (2 * atr_value),
                    timestamp=current['timestamp'],
                    confidence=confidence / 60,
                    reasons=reasons,
                    risk_reward_ratio=2.0,
                    stop_loss=entry_price - atr_value
                ))
        
        return signals


class MeanReversionSignals(SignalGenerator):
    def generate(self, df: pd.DataFrame) -> List[Signal]:
        signals = []
        
        for i in range(max(50, len(df) - 10), len(df)):
            if i < 1:
                continue
                
            current = df.iloc[i]
            confidence = 0
            reasons = []
            
            if current['rsi'] < 30:
                confidence += 30
                reasons.append("RSI oversold")
            
            if current['close'] <= current['bb_lower']:
                confidence += 25
                reasons.append("نزدیک باند پایین بولینگر")
            
            if current['stoch_k'] < 20 and current['stoch_d'] < 20:
                confidence += 20
                reasons.append("استوکاستیک oversold")
            
            if current['williams_r'] < -80:
                confidence += 15
                reasons.append("Williams %R oversold")
            
            if confidence >= 50:
                entry_price = current['close']
                target_price = current.get('bb_middle', entry_price * 1.03)
                
                signals.append(Signal(
                    signal_type=SignalType.BUY,
                    entry_price=entry_price,
                    exit_price=target_price,
                    timestamp=current['timestamp'],
                    confidence=confidence / 90,
                    reasons=reasons,
                    risk_reward_ratio=(target_price - entry_price) / (entry_price * 0.02),
                    stop_loss=entry_price * 0.98
                ))
        
        return signals


class MomentumSignals(SignalGenerator):
    def generate(self, df: pd.DataFrame) -> List[Signal]:
        signals = []
        
        for i in range(max(50, len(df) - 10), len(df)):
            if i < 1:
                continue
                
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            confidence = 0
            reasons = []
            
            if (current['macd'] > current['macd_signal'] and 
                previous['macd'] <= previous['macd_signal']):
                confidence += 30
                reasons.append("MACD تقاطع صعودی")
            
            if current['macd_histogram'] > previous['macd_histogram']:
                confidence += 15
                reasons.append("MACD momentum افزایشی")
            
            if current['mfi'] > 20 and current['mfi'] < 80:
                confidence += 20
                reasons.append("Money Flow متعادل")
            
            if current['cci'] > -100 and previous['cci'] <= -100:
                confidence += 25
                reasons.append("CCI بازگشت از oversold")
            
            if confidence >= 45:
                entry_price = current['close']
                atr_value = current.get('atr', entry_price * 0.02)
                
                signals.append(Signal(
                    signal_type=SignalType.BUY,
                    entry_price=entry_price,
                    exit_price=entry_price + (1.5 * atr_value),
                    timestamp=current['timestamp'],
                    confidence=confidence / 90,
                    reasons=reasons,
                    risk_reward_ratio=1.5,
                    stop_loss=entry_price - atr_value
                ))
        
        return signals


class AdvancedSignalAnalyzer:
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.pattern_detector = PatternDetector()
        self.volume_analyzer = VolumeAnalyzer()
        self.sr_finder = SupportResistanceFinder()
        self.regime_detector = MarketRegimeDetector()
        
        self.signal_generators = [
            TrendFollowingSignals(),
            MeanReversionSignals(),
            MomentumSignals()
        ]
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict:
        if df.empty or len(df) < 100:
            return {'signals': [], 'analysis': {}, 'dataframe': df}
        
        enriched_df = self._enrich_dataframe(df)
        signals = self._generate_comprehensive_signals(enriched_df)
        market_analysis = self._comprehensive_market_analysis(enriched_df)
        
        return {
            'signals': signals,
            'analysis': market_analysis,
            'dataframe': enriched_df
        }
    
    def _enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['returns'] = df['close'].pct_change()
        
        df['sma_20'] = self.indicators.sma(df['close'], 20)
        df['sma_50'] = self.indicators.sma(df['close'], 50)
        df['sma_200'] = self.indicators.sma(df['close'], 200)
        df['ema_12'] = self.indicators.ema(df['close'], 12)
        df['ema_26'] = self.indicators.ema(df['close'], 26)
        df['wma_20'] = self.indicators.wma(df['close'], 20)
        
        df['rsi'] = self.indicators.rsi(df['close'])
        df['rsi_14'] = self.indicators.rsi(df['close'], 14)
        df['rsi_21'] = self.indicators.rsi(df['close'], 21)
        
        macd_line, signal_line, histogram = self.indicators.macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        upper_bb, middle_bb, lower_bb = self.indicators.bollinger_bands(df['close'])
        df['bb_upper'] = upper_bb
        df['bb_middle'] = middle_bb
        df['bb_lower'] = lower_bb
        df['bb_width'] = (upper_bb - lower_bb) / middle_bb
        df['bb_position'] = (df['close'] - lower_bb) / (upper_bb - lower_bb)
        
        k_percent, d_percent = self.indicators.stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = k_percent
        df['stoch_d'] = d_percent
        
        df['williams_r'] = self.indicators.williams_r(df['high'], df['low'], df['close'])
        df['cci'] = self.indicators.cci(df['high'], df['low'], df['close'])
        df['atr'] = self.indicators.atr(df['high'], df['low'], df['close'])
        df['adx'] = self.indicators.adx(df['high'], df['low'], df['close'])
        
        if 'volume' in df.columns:
            df['mfi'] = self.indicators.mfi(df['high'], df['low'], df['close'], df['volume'])
            df['obv'] = self.volume_analyzer.on_balance_volume(df['close'], df['volume'])
            df['vwap'] = self.volume_analyzer.volume_weighted_average_price(
                df['high'], df['low'], df['close'], df['volume']
            )
            df['ad_line'] = self.volume_analyzer.accumulation_distribution(
                df['high'], df['low'], df['close'], df['volume']
            )
            df['volume_sma'] = self.indicators.sma(df['volume'], 20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        if 'open' in df.columns:
            df['hammer'] = self.pattern_detector.detect_hammer(df['open'], df['high'], df['low'], df['close'])
            df['doji'] = self.pattern_detector.detect_doji(df['open'], df['high'], df['low'], df['close'])
            df['engulfing'] = self.pattern_detector.detect_engulfing(df['open'], df['high'], df['low'], df['close'])
        
        df['volatility_regime'] = self.regime_detector.detect_volatility_regime(df['returns'])
        df['trend_strength'] = self.regime_detector.detect_trend_strength(df['close'])
        
        return df.fillna(method='ffill').fillna(method='bfill')
    
    def _generate_comprehensive_signals(self, df: pd.DataFrame) -> List[Signal]:
        all_signals = []
        
        for generator in self.signal_generators:
            signals = generator.generate(df)
            all_signals.extend(signals)
        
        all_signals.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered_signals = self._filter_conflicting_signals(all_signals)
        enhanced_signals = self._enhance_signals_with_confluence(filtered_signals, df)
        
        return enhanced_signals[:10]
    
    def _filter_conflicting_signals(self, signals: List[Signal]) -> List[Signal]:
        filtered = []
        used_timestamps = set()
        
        for signal in signals:
            timestamp_key = signal.timestamp.floor('H')
            
            if timestamp_key not in used_timestamps:
                filtered.append(signal)
                used_timestamps.add(timestamp_key)
        
        return filtered
    
    def _enhance_signals_with_confluence(self, signals: List[Signal], df: pd.DataFrame) -> List[Signal]:
        enhanced_signals = []
        
        for signal in signals:
            timestamp_idx = df[df['timestamp'] == signal.timestamp].index
            if len(timestamp_idx) == 0:
                continue
                
            idx = timestamp_idx[0]
            current = df.iloc[idx]
            
            confluence_score = 0
            additional_reasons = []
            
            if 'volume_ratio' in current and current['volume_ratio'] > 1.5:
                confluence_score += 10
                additional_reasons.append("حجم بالا")
            
            if current.get('bb_width', 0) < 0.1:
                confluence_score += 5
                additional_reasons.append("کاهش نوسان")
            
            if current.get('adx', 0) > 25:
                confluence_score += 15
                additional_reasons.append("ترند قوی")
                
            if current.get('trend_strength') == 'strong':
                confluence_score += 10
                additional_reasons.append("قدرت ترند بالا")
            
            enhanced_confidence = min(1.0, signal.confidence + (confluence_score / 100))
            enhanced_reasons = signal.reasons + additional_reasons
            
            enhanced_signal = Signal(
                signal_type=signal.signal_type,
                entry_price=signal.entry_price,
                exit_price=signal.exit_price,
                timestamp=signal.timestamp,
                confidence=enhanced_confidence,
                reasons=enhanced_reasons,
                risk_reward_ratio=signal.risk_reward_ratio,
                stop_loss=signal.stop_loss
            )
            
            enhanced_signals.append(enhanced_signal)
        
        return enhanced_signals
    
    def _comprehensive_market_analysis(self, df: pd.DataFrame) -> MarketAnalysis:
        if df.empty:
            return MarketAnalysis(
                trend=MarketTrend.SIDEWAYS,
                volatility=0,
                momentum=0,
                strength=0,
                support_levels=[],
                resistance_levels=[],
                risk_level="نامعلوم"
            )
        
        current = df.iloc[-1]
        
        trend = self._determine_market_trend(df)
        volatility = self._calculate_market_volatility(df)
        momentum = self._calculate_momentum_score(df)
        strength = self._calculate_market_strength(df)
        
        pivot_points = self.sr_finder.find_pivot_points(df['high'], df['low'], df['close'])
        fib_levels = self.sr_finder.find_fibonacci_levels(df['high'], df['low'])
        
        support_levels = pivot_points['support'] + [fib_levels['level_382'], fib_levels['level_618']]
        resistance_levels = pivot_points['resistance'] + [fib_levels['level_618'], fib_levels['level_786']]
        
        risk_level = self._assess_risk_level(volatility, strength, current)
        
        return MarketAnalysis(
            trend=trend,
            volatility=volatility,
            momentum=momentum,
            strength=strength,
            support_levels=sorted(set(support_levels))[:5],
            resistance_levels=sorted(set(resistance_levels), reverse=True)[:5],
            risk_level=risk_level
        )
    
    def _determine_market_trend(self, df: pd.DataFrame) -> MarketTrend:
        current = df.iloc[-1]
        
        sma_score = 0
        if current['sma_20'] > current['sma_50']:
            sma_score += 1
        if current['sma_50'] > current['sma_200']:
            sma_score += 1
        if current['close'] > current['sma_20']:
            sma_score += 1
        
        if sma_score >= 2:
            return MarketTrend.BULLISH
        elif sma_score <= 1:
            return MarketTrend.BEARISH
        else:
            return MarketTrend.SIDEWAYS
    
    def _calculate_market_volatility(self, df: pd.DataFrame) -> float:
        returns = df['returns'].dropna()
        if len(returns) < 20:
            return 0.0
        
        volatility = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)
        return float(volatility) if not pd.isna(volatility) else 0.0
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> float:
        current = df.iloc[-1]
        
        momentum_indicators = [
            (current.get('rsi', 50) - 50) / 50,
            current.get('macd_histogram', 0) / abs(current.get('macd_histogram', 1)),
            (current.get('stoch_k', 50) - 50) / 50,
            (current.get('cci', 0)) / 100
        ]
        
        valid_indicators = [x for x in momentum_indicators if not pd.isna(x) and abs(x) < 10]
        
        if not valid_indicators:
            return 0.0
        
        return float(np.mean(valid_indicators))
    
    def _calculate_market_strength(self, df: pd.DataFrame) -> float:
        current = df.iloc[-1]
        
        adx_strength = min(current.get('adx', 0) / 50, 1.0)
        volume_strength = min(current.get('volume_ratio', 1) / 2, 1.0) if 'volume_ratio' in current else 0.5
        
        return float((adx_strength + volume_strength) / 2)
    
    def _assess_risk_level(self, volatility: float, strength: float, current: pd.Series) -> str:
        risk_score = 0
        
        if volatility > 0.3:
            risk_score += 2
        elif volatility > 0.2:
            risk_score += 1
        
        if strength < 0.3:
            risk_score += 1
        
        rsi_val = current.get('rsi', 50)
        if rsi_val > 80 or rsi_val < 20:
            risk_score += 2
        elif rsi_val > 70 or rsi_val < 30:
            risk_score += 1
        
        bb_position = current.get('bb_position', 0.5)
        if bb_position > 0.9 or bb_position < 0.1:
            risk_score += 1
        
        if current.get('volatility_regime') == 'high_vol':
            risk_score += 1
        
        if risk_score >= 4:
            return "بالا"
        elif risk_score >= 2:
            return "متوسط"
        else:
            return "پایین"


class RiskManager:
    @staticmethod
    def calculate_position_size(account_balance: float, risk_per_trade: float, 
                              entry_price: float, stop_loss: float) -> float:
        if stop_loss is None or stop_loss == entry_price:
            return account_balance * 0.01
        
        risk_amount = account_balance * risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        position_size = risk_amount / price_risk
        
        max_position = account_balance * 0.1
        return min(position_size, max_position)
    
    @staticmethod
    def adjust_for_volatility(base_size: float, volatility: float) -> float:
        if volatility > 0.3:
            return base_size * 0.5
        elif volatility > 0.2:
            return base_size * 0.7
        else:
            return base_size
    
    @staticmethod
    def calculate_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        if avg_loss == 0:
            return 0.0
        
        win_loss_ratio = avg_win / avg_loss
        kelly_percentage = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        return max(0.0, min(0.25, kelly_percentage))


class PortfolioOptimizer:
    def __init__(self):
        self.risk_manager = RiskManager()
    
    def optimize_signals(self, signals: List[Signal], account_balance: float = 100000, 
                        max_concurrent_trades: int = 5) -> List[Dict]:
        if not signals:
            return []
        
        optimized_trades = []
        correlation_matrix = self._calculate_signal_correlation(signals)
        
        selected_signals = self._select_uncorrelated_signals(
            signals, correlation_matrix, max_concurrent_trades
        )
        
        for signal in selected_signals:
            position_size = self.risk_manager.calculate_position_size(
                account_balance, 0.02, signal.entry_price, signal.stop_loss
            )
            
            trade_info = {
                'signal': signal,
                'position_size': position_size,
                'risk_amount': account_balance * 0.02,
                'potential_profit': (signal.exit_price - signal.entry_price) * position_size,
                'potential_loss': (signal.entry_price - (signal.stop_loss or signal.entry_price * 0.98)) * position_size,
                'expected_value': self._calculate_expected_value(signal, position_size)
            }
            
            optimized_trades.append(trade_info)
        
        return sorted(optimized_trades, key=lambda x: x['expected_value'], reverse=True)
    
    def _calculate_signal_correlation(self, signals: List[Signal]) -> np.ndarray:
        n_signals = len(signals)
        correlation_matrix = np.eye(n_signals)
        
        for i in range(n_signals):
            for j in range(i + 1, n_signals):
                correlation = self._calculate_pairwise_correlation(signals[i], signals[j])
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        return correlation_matrix
    
    def _calculate_pairwise_correlation(self, signal1: Signal, signal2: Signal) -> float:
        time_diff = abs((signal1.timestamp - signal2.timestamp).total_seconds())
        time_correlation = max(0, 1 - time_diff / 86400)
        
        price_diff = abs(signal1.entry_price - signal2.entry_price) / max(signal1.entry_price, signal2.entry_price)
        price_correlation = max(0, 1 - price_diff * 10)
        
        same_type = 1.0 if signal1.signal_type == signal2.signal_type else 0.5
        
        return (time_correlation + price_correlation + same_type) / 3
    
    def _select_uncorrelated_signals(self, signals: List[Signal], 
                                   correlation_matrix: np.ndarray, 
                                   max_signals: int) -> List[Signal]:
        selected_indices = []
        remaining_indices = list(range(len(signals)))
        
        remaining_indices.sort(key=lambda i: signals[i].confidence, reverse=True)
        
        while len(selected_indices) < max_signals and remaining_indices:
            best_idx = remaining_indices[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            
            remaining_indices = [
                idx for idx in remaining_indices
                if correlation_matrix[best_idx, idx] < 0.7
            ]
        
        return [signals[i] for i in selected_indices]
    
    def _calculate_expected_value(self, signal: Signal, position_size: float) -> float:
        estimated_win_rate = min(0.8, signal.confidence)
        
        potential_profit = (signal.exit_price - signal.entry_price) * position_size
        potential_loss = (signal.entry_price - (signal.stop_loss or signal.entry_price * 0.98)) * position_size
        
        expected_value = (estimated_win_rate * potential_profit) - ((1 - estimated_win_rate) * potential_loss)
        
        return expected_value


class PerformanceTracker:
    def __init__(self):
        self.trades = []
        self.equity_curve = []
    
    def add_trade(self, entry_price: float, exit_price: float, position_size: float, 
                 signal_type: SignalType, timestamp: pd.Timestamp):
        if signal_type == SignalType.BUY:
            pnl = (exit_price - entry_price) * position_size
        else:
            pnl = (entry_price - exit_price) * position_size
        
        trade = {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'pnl': pnl,
            'timestamp': timestamp,
            'signal_type': signal_type
        }
        
        self.trades.append(trade)
        
        current_equity = self.equity_curve[-1] if self.equity_curve else 100000
        new_equity = current_equity + pnl
        self.equity_curve.append(new_equity)
    
    def calculate_metrics(self) -> Dict:
        if not self.trades:
            return {}
        
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        avg_win = np.mean([t['pnl'] for t in self.trades if t['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([abs(t['pnl']) for t in self.trades if t['pnl'] < 0]) if losing_trades > 0 else 0
        
        profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades) if losing_trades > 0 else float('inf')
        
        equity_series = pd.Series(self.equity_curve)
        max_drawdown = self._calculate_max_drawdown(equity_series)
        sharpe_ratio = self._calculate_sharpe_ratio(equity_series)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_equity': self.equity_curve[-1] if self.equity_curve else 100000
        }
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        if len(equity_series) < 2:
            return 0.0
        
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        return abs(max_drawdown)
    
    def _calculate_sharpe_ratio(self, equity_series: pd.Series) -> float:
        if len(equity_series) < 2:
            return 0.0
        
        returns = equity_series.pct_change().dropna()
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        return sharpe


class TradingSystem:
    def __init__(self):
        self.analyzer = AdvancedSignalAnalyzer()
        self.optimizer = PortfolioOptimizer()
        self.performance_tracker = PerformanceTracker()
    
    def run_analysis(self, df: pd.DataFrame, account_balance: float = 100000) -> Dict:
        analysis_result = self.analyzer.analyze_dataframe(df)
        
        if not analysis_result['signals']:
            return {
                'signals': [],
                'optimized_trades': [],
                'market_analysis': analysis_result['analysis'],
                'performance_metrics': {},
                'dataframe': analysis_result['dataframe']
            }
        
        optimized_trades = self.optimizer.optimize_signals(
            analysis_result['signals'], account_balance
        )
        
        performance_metrics = self.performance_tracker.calculate_metrics()
        
        return {
            'signals': [self._signal_to_dict(s) for s in analysis_result['signals']],
            'optimized_trades': optimized_trades,
            'market_analysis': self._analysis_to_dict(analysis_result['analysis']),
            'performance_metrics': performance_metrics,
            'dataframe': analysis_result['dataframe']
        }
    
    def _signal_to_dict(self, signal: Signal) -> Dict:
        return {
            'signal_type': signal.signal_type.value,
            'entry_price': signal.entry_price,
            'exit_price': signal.exit_price,
            'timestamp': signal.timestamp.isoformat() if hasattr(signal.timestamp, 'isoformat') else str(signal.timestamp),
            'confidence': signal.confidence,
            'reasons': signal.reasons,
            'risk_reward_ratio': signal.risk_reward_ratio,
            'stop_loss': signal.stop_loss
        }
    
    def _analysis_to_dict(self, analysis: MarketAnalysis) -> Dict:
        return {
            'trend': analysis.trend.value,
            'volatility': analysis.volatility,
            'momentum': analysis.momentum,
            'strength': analysis.strength,
            'support_levels': analysis.support_levels,
            'resistance_levels': analysis.resistance_levels,
            'risk_level': analysis.risk_level
        }
        
class SignalAnalyzer:
    """کلاس تحلیل سیگنال‌های معاملاتی"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict:
        """تحلیل کامل داده‌ها و تولید سیگنال‌ها"""
        if df.empty or len(df) < 50:
            return {'signals': {'buy': [], 'sell': []}, 'analysis': {}}
        
        # محاسبه اندیکاتورها
        df['sma_20'] = self.indicators.sma(df['close'], 20)
        df['sma_50'] = self.indicators.sma(df['close'], 50)
        df['ema_12'] = self.indicators.ema(df['close'], 12)
        df['ema_26'] = self.indicators.ema(df['close'], 26)
        df['rsi'] = self.indicators.rsi(df['close'])
        
        macd_line, signal_line, histogram = self.indicators.macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        upper_bb, middle_bb, lower_bb = self.indicators.bollinger_bands(df['close'])
        df['bb_upper'] = upper_bb
        df['bb_middle'] = middle_bb
        df['bb_lower'] = lower_bb
        
        k_percent, d_percent = self.indicators.stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = k_percent
        df['stoch_d'] = d_percent
        
        # تولید سیگنال‌ها
        signals = self._generate_signals(df)
        
        # تحلیل وضعیت فعلی
        current_analysis = self._current_market_analysis(df)
        
        return {
            'signals': signals,
            'analysis': current_analysis,
            'dataframe': df
        }
    
    def _generate_signals(self, df: pd.DataFrame) -> Dict:
        """تولید سیگنال‌های خرید و فروش"""
        buy_signals = []
        sell_signals = []
        
        # بررسی آخرین 10 کندل برای سیگنال‌ها
        for i in range(len(df) - 10, len(df)):
            if i < 1:
                continue
                
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # سیگنال‌های خرید
            buy_score = 0
            buy_reasons = []
            
            # 1. تقاطع SMA صعودی
            if (current['sma_20'] > current['sma_50'] and 
                previous['sma_20'] <= previous['sma_50']):
                buy_score += 2
                buy_reasons.append("تقاطع طلایی SMA")
            
            # 2. RSI در ناحیه oversold
            if current['rsi'] < 30 and previous['rsi'] >= 30:
                buy_score += 2
                buy_reasons.append("RSI oversold")
            
            # 3. MACD تقاطع صعودی
            if (current['macd'] > current['macd_signal'] and 
                previous['macd'] <= previous['macd_signal']):
                buy_score += 2
                buy_reasons.append("MACD تقاطع صعودی")
            
            # 4. قیمت در نزدیکی باند پایین بولینگر
            if current['close'] <= current['bb_lower'] * 1.02:
                buy_score += 1
                buy_reasons.append("نزدیک باند پایین بولینگر")
            
            # 5. استوکاستیک در ناحیه oversold
            if current['stoch_k'] < 20 and current['stoch_d'] < 20:
                buy_score += 1
                buy_reasons.append("استوکاستیک oversold")
            
            # سیگنال‌های فروش
            sell_score = 0
            sell_reasons = []
            
            # 1. تقاطع SMA نزولی
            if (current['sma_20'] < current['sma_50'] and 
                previous['sma_20'] >= previous['sma_50']):
                sell_score += 2
                sell_reasons.append("تقاطع مرگ SMA")
            
            # 2. RSI در ناحیه overbought
            if current['rsi'] > 70 and previous['rsi'] <= 70:
                sell_score += 2
                sell_reasons.append("RSI overbought")
            
            # 3. MACD تقاطع نزولی
            if (current['macd'] < current['macd_signal'] and 
                previous['macd'] >= previous['macd_signal']):
                sell_score += 2
                sell_reasons.append("MACD تقاطع نزولی")
            
            # 4. قیمت در نزدیکی باند بالای بولینگر
            if current['close'] >= current['bb_upper'] * 0.98:
                sell_score += 1
                sell_reasons.append("نزدیک باند بالای بولینگر")
            
            # 5. استوکاستیک در ناحیه overbought
            if current['stoch_k'] > 80 and current['stoch_d'] > 80:
                sell_score += 1
                sell_reasons.append("استوکاستیک overbought")
            
            # اگر امتیاز سیگنال بالای 3 باشد، سیگنال را ثبت کن
            if buy_score >= 3:
                entry_price = current['close']
                # محاسبه نقطه خروج بر اساس مقاومت یا 2% سود
                resistance = self._find_resistance(df, i)
                exit_price = max(entry_price * 1.02, resistance) if resistance else entry_price * 1.025
                
                buy_signals.append({
                    'entry': float(entry_price),
                    'exit': float(exit_price),
                    'timestamp': current['timestamp'],
                    'score': buy_score,
                    'reasons': buy_reasons,
                    'indicator': 'ترکیبی'
                })
            
            if sell_score >= 3:
                entry_price = current['close']
                # محاسبه نقطه خروج بر اساس حمایت یا 2% سود
                support = self._find_support(df, i)
                exit_price = min(entry_price * 0.98, support) if support else entry_price * 0.975
                
                sell_signals.append({
                    'entry': float(entry_price),
                    'exit': float(exit_price),
                    'timestamp': current['timestamp'],
                    'score': sell_score,
                    'reasons': sell_reasons,
                    'indicator': 'ترکیبی'
                })
        
        return {'buy': buy_signals, 'sell': sell_signals}
    
    def _find_resistance(self, df: pd.DataFrame, current_index: int) -> Optional[float]:
        """پیدا کردن نزدیکترین سطح مقاومت"""
        current_price = df.iloc[current_index]['close']
        lookback = min(50, current_index)
        
        highs = df.iloc[current_index-lookback:current_index]['high']
        resistance_levels = highs[highs > current_price * 1.01]
        
        return float(resistance_levels.min()) if not resistance_levels.empty else None
    
    def _find_support(self, df: pd.DataFrame, current_index: int) -> Optional[float]:
        """پیدا کردن نزدیکترین سطح حمایت"""
        current_price = df.iloc[current_index]['close']
        lookback = min(50, current_index)
        
        lows = df.iloc[current_index-lookback:current_index]['low']
        support_levels = lows[lows < current_price * 0.99]
        
        return float(support_levels.max()) if not support_levels.empty else None
    
    def _current_market_analysis(self, df: pd.DataFrame) -> Dict:
        """تحلیل وضعیت فعلی بازار"""
        if df.empty:
            return {}
        
        current = df.iloc[-1]
        
        # تعیین ترند
        if current['sma_20'] > current['sma_50']:
            trend = "صعودی"
        elif current['sma_20'] < current['sma_50']:
            trend = "نزولی"
        else:
            trend = "خنثی"
        
        # وضعیت RSI
        if current['rsi'] > 70:
            rsi_status = "خریداری بیش از حد"
        elif current['rsi'] < 30:
            rsi_status = "فروخته بیش از حد"
        else:
            rsi_status = "طبیعی"
        
        return {
            'trend': trend,
            'rsi_status': rsi_status,
            'rsi_value': float(current['rsi']) if not pd.isna(current['rsi']) else 0,
            'current_price': float(current['close']),
            'volume': float(current['volume'])
        }

class TradingBot:
    """کلاس اصلی ربات معاملاتی"""
    
    def __init__(self):
        self.analyzer = SignalAnalyzer()
        self.exchange = None
    
    async def get_exchange(self):
        """ایجاد اتصال به صرافی"""
        if self.exchange is None:
            self.exchange = ccxt.coinex({
                'apiKey': os.getenv('COINEX_API_KEY', ''),
                'secret': os.getenv('COINEX_SECRET', ''),
                'sandbox': False,
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {'defaultType': 'spot'}
            })
        return self.exchange
    
    async def close_exchange(self):
        """بستن اتصال صرافی"""
        if self.exchange:
            await self.exchange.close()
            self.exchange = None
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """دریافت داده‌های OHLCV"""
        try:
            exchange = await self.get_exchange()
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                logger.warning(f"No data returned for {symbol} on {timeframe}")
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
            
        except Exception as e:
            logger.error(f"Error fetching ohlcv for {symbol} on {timeframe}: {e}")
            return pd.DataFrame()
    
    async def find_best_signal_for_timeframe(self, timeframe: str) -> Optional[Dict]:
        """پیدا کردن بهترین سیگنال برای تایم فریم مشخص"""
        best_signal = None
        max_profit = 0.0
        
        for symbol in SYMBOLS:
            try:
                logger.info(f"Processing {symbol} for timeframe {timeframe}...")
                df = await self.fetch_ohlcv(symbol, timeframe)
                
                if df.empty or len(df) < 50:
                    logger.warning(f"Not enough data for {symbol} on {timeframe}. Skipping.")
                    continue
                
                analysis_result = self.analyzer.analyze_dataframe(df)
                
                if 'signals' in analysis_result and analysis_result['signals']:
                    signals = analysis_result['signals']
                    
                    # بررسی سیگنال‌های خرید
                    for signal in signals.get('buy', []):
                        profit = (signal['exit'] / signal['entry']) - 1
                        if profit > max_profit:
                            max_profit = profit
                            best_signal = {
                                'symbol': symbol,
                                'type': 'خرید (Buy)',
                                'profit': profit * 100,
                                'entry': signal['entry'],
                                'exit': signal['exit'],
                                'indicator': signal.get('indicator', 'N/A'),
                                'score': signal.get('score', 0),
                                'reasons': signal.get('reasons', []),
                                'analysis': analysis_result.get('analysis', {})
                            }
                    
                    # بررسی سیگنال‌های فروش
                    for signal in signals.get('sell', []):
                        profit = (signal['entry'] / signal['exit']) - 1
                        if profit > max_profit:
                            max_profit = profit
                            best_signal = {
                                'symbol': symbol,
                                'type': 'فروش (Sell)',
                                'profit': profit * 100,
                                'entry': signal['entry'],
                                'exit': signal['exit'],
                                'indicator': signal.get('indicator', 'N/A'),
                                'score': signal.get('score', 0),
                                'reasons': signal.get('reasons', []),
                                'analysis': analysis_result.get('analysis', {})
                            }
                            
            except Exception as e:
                logger.error(f"Error processing {symbol} for {timeframe}: {e}")
                continue
        
        return best_signal

# ایجاد نمونه از ربات
trading_bot = TradingBot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """دستور /start را مدیریت می‌کند"""
    user_id = update.effective_user.id
    logger.info(f"User {user_id} started the bot")
    
    await update.message.reply_text(
        '🤖 **ربات سیگنال‌های معاملاتی**\n\n'
        'در حال بررسی سیگنال‌ها در تمام تایم فریم‌ها... \n'
        'لطفاً شکیبا باشید، این فرآیند ممکن است چند دقیقه طول بکشد. 🧐\n\n'
        '📊 در حال تحلیل 10 نماد برتر...',
        parse_mode='Markdown'
    )
    
    try:
        # حلقه برای بررسی هر تایم فریم
        for i, timeframe in enumerate(TIME_FRAMES):
            progress = f"🔍 ({i+1}/{len(TIME_FRAMES)}) در حال جستجوی بهترین سیگنال برای تایم فریم **{timeframe}**..."
            await update.message.reply_text(progress, parse_mode='Markdown')
            
            best_signal = await trading_bot.find_best_signal_for_timeframe(timeframe)
            
            if best_signal:
                # ساخت پیام دقیق‌تر
                reasons_text = "، ".join(best_signal.get('reasons', []))
                analysis = best_signal.get('analysis', {})
                
                message = (
                    f"🚀 **بهترین سیگنال برای تایم فریم {timeframe}**\n\n"
                    f"📈 **نماد:** `{best_signal['symbol']}`\n"
                    f"📊 **نوع سیگنال:** {best_signal['type']}\n"
                    f"💰 **سود احتمالی:** `{best_signal['profit']:.2f}%`\n"
                    f"🟢 **نقطه ورود:** `${best_signal['entry']:.4f}`\n"
                    f"🔴 **نقطه خروج:** `${best_signal['exit']:.4f}`\n"
                    f"⭐ **امتیاز سیگنال:** `{best_signal.get('score', 0)}/8`\n"
                    f"📋 **دلایل:** {reasons_text}\n"
                )
                
                if analysis:
                    message += (
                        f"\n📊 **تحلیل بازار:**\n"
                        f"📈 **ترند:** {analysis.get('trend', 'نامشخص')}\n"
                        f"📊 **RSI:** {analysis.get('rsi_value', 0):.1f} ({analysis.get('rsi_status', 'طبیعی')})\n"
                        f"💵 **قیمت فعلی:** `${analysis.get('current_price', 0):.4f}` \n"
                        f"📅 **تاریخ:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                
                await update.message.reply_text(message, parse_mode='Markdown')
            else:
                message = f"⚠️ برای تایم فریم `{timeframe}` هیچ سیگنال قوی یافت نشد."
                await update.message.reply_text(message, parse_mode='Markdown')
        
        # پیام پایانی
        await update.message.reply_text(
            "✅ **تحلیل کامل شد!**\n\n"
            "⚠️ **هشدار:** این سیگنال‌ها صرفاً جهت اطلاع‌رسانی هستند و نباید به عنوان مشاوره مالی تلقی شوند.\n"
            "همیشه تحقیقات خود را انجام دهید و مدیریت ریسک را رعایت کنید.\n\n"
            "🔄 برای به‌روزرسانی سیگنال‌ها، دوباره /start را بزنید.",
            parse_mode='Markdown'
        )
        
    except Exception as e:
        logger.error(f"An error occurred during the start command: {e}")
        await update.message.reply_text(
            f"❌ **خطایی رخ داد:**\n`{str(e)}`\n\n"
            "لطفاً دوباره تلاش کنید یا با توسعه‌دهنده تماس بگیرید.",
            parse_mode='Markdown'
        )
    finally:
        # بستن اتصال صرافی
        await trading_bot.close_exchange()

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """دستور /help را مدیریت می‌کند"""
    help_text = (
        "🤖 **راهنمای ربات سیگنال‌های معاملاتی**\n\n"
        "**دستورات موجود:**\n"
        "• `/start` - شروع تحلیل و دریافت سیگنال‌ها\n"
        "• `/help` - نمایش این راهنما\n\n"
        "**ویژگی‌ها:**\n"
        "• تحلیل تکنیکال پیشرفته\n"
        "• پشتیبانی از 10 نماد برتر\n"
        "• بررسی 8 تایم فریم مختلف\n"
        "• محاسبه امتیاز سیگنال\n"
        "• تحلیل وضعیت بازار\n\n"
        "**اندیکاتورهای استفاده شده:**\n"
        "• SMA (میانگین متحرک ساده)\n"
        "• EMA (میانگین متحرک نمایی)\n"
        "• RSI (شاخص قدرت نسبی)\n"
        "• MACD\n"
        "• Bollinger Bands\n"
        "• Stochastic\n\n"
        "⚠️ **هشدار:** این سیگنال‌ها جهت اطلاع‌رسانی هستند و مشاوره مالی نیستند."
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')

def main() -> None:
    """اجرای ربات"""
    if not BOT_TOKEN:
        logger.error("لطفاً توکن ربات تلگرام را در متغیر BOT_TOKEN وارد کنید")
        return
    
    logger.info("Starting Trading Signal Bot...")
    
    application = Application.builder().token(BOT_TOKEN).build()
    
    # اضافه کردن دستورات
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    
    # اجرای ربات
    logger.info("Bot is running. Press Ctrl+C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()