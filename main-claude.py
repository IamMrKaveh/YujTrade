from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import ccxt.async_support as ccxt
import numpy as np
from datetime import datetime
import os
import warnings
from typing import Dict, List, Tuple, Optional, Protocol, Any, Union
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
from config.constants import BOT_TOKEN, SYMBOLS, TIME_FRAMES
from logger_config import logger


warnings.filterwarnings('ignore')

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

@dataclass
class MarketAnalysis:
    trend: TrendDirection
    volatility: float
    volume_trend: str
    support_levels: List[float]
    resistance_levels: List[float]
    momentum_score: float
    market_condition: MarketCondition

@dataclass
class IndicatorResult:
    name: str
    value: float
    signal_strength: float
    interpretation: str

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
        
        return IndicatorResult(
            name=f"{self.ma_type.upper()}_{self.period}",
            value=current_ma,
            signal_strength=signal_strength,
            interpretation=interpretation
        )

class RSIIndicator(TechnicalIndicator):
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        if current_rsi > 70:
            interpretation = "overbought"
            signal_strength = (current_rsi - 70) / 30 * 100
        elif current_rsi < 30:
            interpretation = "oversold"
            signal_strength = (30 - current_rsi) / 30 * 100
        else:
            interpretation = "neutral"
            signal_strength = 50 - abs(current_rsi - 50)
        
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
        
        return IndicatorResult(
            name="MACD",
            value=current_macd,
            signal_strength=signal_strength,
            interpretation=interpretation
        )

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
        
        bb_position = (current_price - current_lower) / (current_upper - current_lower)
        
        if bb_position > 0.8:
            interpretation = "near_upper_band"
            signal_strength = (bb_position - 0.8) / 0.2 * 100
        elif bb_position < 0.2:
            interpretation = "near_lower_band"
            signal_strength = (0.2 - bb_position) / 0.2 * 100
        else:
            interpretation = "middle_range"
            signal_strength = 50
        
        return IndicatorResult(
            name="BB",
            value=bb_position,
            signal_strength=signal_strength,
            interpretation=interpretation
        )

class StochasticIndicator(TechnicalIndicator):
    def __init__(self, k_period: int = 14, d_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        lowest_low = data['low'].rolling(window=self.k_period).min()
        highest_high = data['high'].rolling(window=self.k_period).max()
        k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=self.d_period).mean()
        
        current_k = k_percent.iloc[-1]
        current_d = d_percent.iloc[-1]
        
        if current_k > 80 and current_d > 80:
            interpretation = "overbought"
            signal_strength = ((current_k + current_d) / 2 - 80) / 20 * 100
        elif current_k < 20 and current_d < 20:
            interpretation = "oversold"
            signal_strength = (20 - (current_k + current_d) / 2) / 20 * 100
        else:
            interpretation = "neutral"
            signal_strength = 50
        
        return IndicatorResult(
            name="STOCH",
            value=(current_k + current_d) / 2,
            signal_strength=signal_strength,
            interpretation=interpretation
        )

class VolumeIndicator(TechnicalIndicator):
    def __init__(self, period: int = 20):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        volume_ma = data['volume'].rolling(window=self.period).mean()
        current_volume = data['volume'].iloc[-1]
        average_volume = volume_ma.iloc[-1]
        
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
        
        return IndicatorResult(
            name="VOLUME",
            value=volume_ratio,
            signal_strength=signal_strength,
            interpretation=interpretation
        )

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
        atr_percentage = (current_atr / current_price) * 100
        
        if atr_percentage > 3:
            interpretation = "high_volatility"
            signal_strength = min(atr_percentage * 20, 100)
        elif atr_percentage < 1:
            interpretation = "low_volatility"
            signal_strength = (1 - atr_percentage) * 100
        else:
            interpretation = "normal_volatility"
            signal_strength = 50
        
        return IndicatorResult(
            name="ATR",
            value=current_atr,
            signal_strength=signal_strength,
            interpretation=interpretation
        )

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
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                resistance_levels.append(highs[i])
        
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
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
        volume_breakout = current_volume / avg_volume
        
        return {
            'volume_ratio': volume_breakout,
            'volume_trend': volume_trend,
            'volume_strength': min(volume_breakout * 50, 100)
        }
    
    def _calculate_volume_trend(self, data: pd.DataFrame) -> float:
        recent_volumes = data['volume'].tail(10)
        if len(recent_volumes) < 2:
            return 0
        
        volume_changes = recent_volumes.pct_change().dropna()
        return volume_changes.mean()

class MarketConditionAnalyzer:
    def __init__(self):
        self.support_resistance = SupportResistanceAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
    
    def analyze_market_condition(self, data: pd.DataFrame) -> MarketAnalysis:
        support_levels, resistance_levels = self.support_resistance.find_support_resistance(data)
        volume_analysis = self.volume_analyzer.analyze_volume_pattern(data)
        
        trend = self._determine_trend(data)
        volatility = self._calculate_volatility(data)
        momentum_score = self._calculate_momentum(data)
        market_condition = self._determine_market_condition(data)
        
        return MarketAnalysis(
            trend=trend,
            volatility=volatility,
            volume_trend="increasing" if volume_analysis['volume_trend'] > 0 else "decreasing",
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            momentum_score=momentum_score,
            market_condition=market_condition
        )
    
    def _determine_trend(self, data: pd.DataFrame) -> TrendDirection:
        sma_20 = data['close'].rolling(window=20).mean()
        sma_50 = data['close'].rolling(window=50).mean()
        
        if len(sma_20) < 50 or len(sma_50) < 50:
            return TrendDirection.SIDEWAYS
        
        current_sma_20 = sma_20.iloc[-1]
        current_sma_50 = sma_50.iloc[-1]
        
        if current_sma_20 > current_sma_50 * 1.01:
            return TrendDirection.BULLISH
        elif current_sma_20 < current_sma_50 * 0.99:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.SIDEWAYS
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        returns = data['close'].pct_change().dropna()
        return returns.std() * np.sqrt(len(returns))
    
    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        if len(data) < 20:
            return 0.0
        
        price_change = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
        return price_change * 100
    
    def _determine_market_condition(self, data: pd.DataFrame) -> MarketCondition:
        rsi_indicator = RSIIndicator()
        rsi_result = rsi_indicator.calculate(data)
        
        if rsi_result.value > 70:
            return MarketCondition.OVERBOUGHT
        elif rsi_result.value < 30:
            return MarketCondition.OVERSOLD
        else:
            return MarketCondition.NEUTRAL

class SignalGenerator:
    def __init__(self):
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
            'atr': ATRIndicator()
        }
        self.market_analyzer = MarketConditionAnalyzer()
    
    def generate_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[TradingSignal]:
        logger.info(f"🔄 Generating signals for {symbol} on {timeframe} with {len(data)} candles")
        
        if len(data) < 50:
            logger.warning(f"⚠️ Insufficient data for {symbol} on {timeframe}: {len(data)} candles (need 50+)")
            return []
        
        indicator_results = {}
        failed_indicators = []
        
        for name, indicator in self.indicators.items():
            try:
                indicator_results[name] = indicator.calculate(data)
                logger.debug(f"✅ {name} calculated successfully for {symbol}")
            except Exception as e:
                logger.warning(f"⚠️ Error calculating {name} for {symbol}: {e}")
                failed_indicators.append(name)
                continue
        
        if failed_indicators:
            logger.warning(f"❌ Failed indicators for {symbol}: {', '.join(failed_indicators)}")
        
        try:
            market_analysis = self.market_analyzer.analyze_market_condition(data)
            logger.debug(f"📊 Market analysis completed for {symbol}: trend={market_analysis.trend.value}")
        except Exception as e:
            logger.error(f"❌ Market analysis failed for {symbol}: {e}")
            return []
        
        signals = []
        
        # بررسی سیگنال خرید
        buy_signal = self._evaluate_buy_signal(indicator_results, data, symbol, timeframe, market_analysis)
        if buy_signal:
            logger.info(f"🟢 BUY signal generated for {symbol} on {timeframe} - Confidence: {buy_signal.confidence_score:.0f}")
            signals.append(buy_signal)
        
        # بررسی سیگنال فروش
        sell_signal = self._evaluate_sell_signal(indicator_results, data, symbol, timeframe, market_analysis)
        if sell_signal:
            logger.info(f"🔴 SELL signal generated for {symbol} on {timeframe} - Confidence: {sell_signal.confidence_score:.0f}")
            signals.append(sell_signal)
        
        if not signals:
            logger.debug(f"ℹ️ No qualifying signals for {symbol} on {timeframe}")
        
        return signals
    
    def _evaluate_buy_signal(self, indicators: Dict[str, IndicatorResult], data: pd.DataFrame, 
                            symbol: str, timeframe: str, market_analysis: MarketAnalysis) -> Optional[TradingSignal]:
        score = 0
        reasons = []
        
        current_price = data['close'].iloc[-1]
        
        if 'rsi' in indicators and indicators['rsi'].interpretation == "oversold":
            score += 25
            reasons.append("RSI oversold condition")
        
        if 'macd' in indicators and indicators['macd'].interpretation == "bullish_crossover":
            score += 20
            reasons.append("MACD bullish crossover")
        
        if ('sma_20' in indicators and 'sma_50' in indicators and 
            indicators['sma_20'].value > indicators['sma_50'].value):
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
            score += 5
            reasons.append("Overall bullish trend")
        
        if score >= 60:
            stop_loss = self._calculate_stop_loss(current_price, market_analysis, "buy")
            exit_price = self._calculate_exit_price(current_price, market_analysis, "buy")
            
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                entry_price=current_price,
                exit_price=exit_price,
                stop_loss=stop_loss,
                timestamp=datetime.now(),
                timeframe=timeframe,
                confidence_score=score,
                reasons=reasons,
                risk_reward_ratio=self._calculate_risk_reward(current_price, exit_price, stop_loss),
                predicted_profit=((exit_price - current_price) / current_price) * 100,
                volume_analysis=self.market_analyzer.volume_analyzer.analyze_volume_pattern(data),
                market_context=self._create_market_context(market_analysis)
            )
        
        return None
    
    def _evaluate_sell_signal(self, indicators: Dict[str, IndicatorResult], data: pd.DataFrame,
                            symbol: str, timeframe: str, market_analysis: MarketAnalysis) -> Optional[TradingSignal]:
        score = 0
        reasons = []
        
        current_price = data['close'].iloc[-1]
        
        if 'rsi' in indicators and indicators['rsi'].interpretation == "overbought":
            score += 25
            reasons.append("RSI overbought condition")
        
        if 'macd' in indicators and indicators['macd'].interpretation == "bearish_crossover":
            score += 20
            reasons.append("MACD bearish crossover")
        
        if ('sma_20' in indicators and 'sma_50' in indicators and 
            indicators['sma_20'].value < indicators['sma_50'].value):
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
            score += 5
            reasons.append("Overall bearish trend")
        
        if score >= 60:
            stop_loss = self._calculate_stop_loss(current_price, market_analysis, "sell")
            exit_price = self._calculate_exit_price(current_price, market_analysis, "sell")
            
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                entry_price=current_price,
                exit_price=exit_price,
                stop_loss=stop_loss,
                timestamp=datetime.now(),
                timeframe=timeframe,
                confidence_score=score,
                reasons=reasons,
                risk_reward_ratio=self._calculate_risk_reward(current_price, exit_price, stop_loss),
                predicted_profit=((current_price - exit_price) / current_price) * 100,
                volume_analysis=self.market_analyzer.volume_analyzer.analyze_volume_pattern(data),
                market_context=self._create_market_context(market_analysis)
            )
        
        return None
    
    def _calculate_stop_loss(self, entry_price: float, market_analysis: MarketAnalysis, signal_type: str) -> float:
        atr_based_stop = entry_price * 0.02
        
        if signal_type == "buy":
            if market_analysis.support_levels:
                nearest_support = max([s for s in market_analysis.support_levels if s < entry_price], default=entry_price * 0.95)
                return max(nearest_support, entry_price - atr_based_stop)
            return entry_price - atr_based_stop
        else:
            if market_analysis.resistance_levels:
                nearest_resistance = min([r for r in market_analysis.resistance_levels if r > entry_price], default=entry_price * 1.05)
                return min(nearest_resistance, entry_price + atr_based_stop)
            return entry_price + atr_based_stop
    
    def _calculate_exit_price(self, entry_price: float, market_analysis: MarketAnalysis, signal_type: str) -> float:
        base_target = 0.025
        
        if market_analysis.volatility > 0.03:
            base_target *= 1.5
        
        if signal_type == "buy":
            if market_analysis.resistance_levels:
                nearest_resistance = min([r for r in market_analysis.resistance_levels if r > entry_price], default=entry_price * (1 + base_target))
                return min(nearest_resistance, entry_price * (1 + base_target))
            return entry_price * (1 + base_target)
        else:
            if market_analysis.support_levels:
                nearest_support = max([s for s in market_analysis.support_levels if s < entry_price], default=entry_price * (1 - base_target))
                return max(nearest_support, entry_price * (1 - base_target))
            return entry_price * (1 - base_target)
    
    def _calculate_risk_reward(self, entry: float, exit: float, stop_loss: float) -> float:
        if entry == stop_loss:
            return 0
        
        potential_profit = abs(exit - entry)
        potential_loss = abs(entry - stop_loss)
        
        return potential_profit / potential_loss if potential_loss > 0 else 0
    
    def _create_market_context(self, market_analysis: MarketAnalysis) -> Dict[str, Any]:
        return {
            'trend': market_analysis.trend.value,
            'volatility': market_analysis.volatility,
            'momentum_score': market_analysis.momentum_score,
            'market_condition': market_analysis.market_condition.value,
            'volume_trend': market_analysis.volume_trend
        }

class ExchangeManager:
    def __init__(self):
        self.exchange = None
        self._lock = asyncio.Lock()
    
    @asynccontextmanager
    async def get_exchange(self):
        async with self._lock:
            if self.exchange is None:
                self.exchange = ccxt.coinex({
                'apiKey': os.getenv('COINEX_API_KEY', ''),
                'secret': os.getenv('COINEX_SECRET', ''),
                'sandbox': False,
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {'defaultType': 'spot'}
            })
            
            try:
                yield self.exchange
            except Exception as e:
                logger.error(f"Error accessing exchange: {e}")
                await self.close_exchange()
                raise e
    
    async def close_exchange(self):
        async with self._lock:
            if self.exchange:
                await self.exchange.close()
                self.exchange = None
    
    async def fetch_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        logger.info(f"🔍 Fetching OHLCV data for {symbol} on {timeframe} (limit: {limit})")
        
        try:
            async with self.get_exchange() as exchange:
                start_time = asyncio.get_event_loop().time()
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                fetch_time = asyncio.get_event_loop().time() - start_time
                
                if not ohlcv:
                    logger.warning(f"⚠️ No OHLCV data received for {symbol} on {timeframe}")
                    return pd.DataFrame()
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                logger.info(f"✅ Successfully fetched {len(df)} candles for {symbol} on {timeframe} in {fetch_time:.2f}s")
                return df
                
        except ccxt.NetworkError as e:
            logger.error(f"🌐 Network error fetching {symbol} on {timeframe}: {e}")
            return pd.DataFrame()
        except ccxt.ExchangeError as e:
            logger.error(f"🏪 Exchange error fetching {symbol} on {timeframe}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ Unexpected error fetching {symbol} on {timeframe}: {e}")
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
            if signal.market_context.get('trend') in ['bullish', 'bearish']:
                trend_bonus = 5
            
            return base_score + rr_bonus + profit_bonus + volume_bonus + trend_bonus
        
        return sorted(signals, key=signal_score, reverse=True)

class ConfigManager:
    DEFAULT_CONFIG = {
        'symbols': SYMBOLS,
        'timeframes': TIME_FRAMES,
        'min_confidence_score': 50,
        'max_signals_per_timeframe': 3,
        'risk_reward_threshold': 1.5
    }
    
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
        self.signal_generator = SignalGenerator()
        self.signal_ranking = SignalRanking()
    
    async def analyze_symbol(self, symbol: str, timeframe: str) -> List[TradingSignal]:
        logger.info(f"🔍 Starting analysis for {symbol} on {timeframe}")
        
        try:
            # دریافت داده‌ها
            data = await self.exchange_manager.fetch_ohlcv_data(symbol, timeframe)
            
            if data.empty:
                logger.warning(f"⚠️ No data available for {symbol} on {timeframe}")
                return []
            
            if len(data) < 50:
                logger.warning(f"⚠️ Insufficient data for {symbol} on {timeframe}: {len(data)} candles")
                return []
            
            # تولید سیگنال‌ها
            all_signals = self.signal_generator.generate_signals(data, symbol, timeframe)
            
            # فیلتر کردن بر اساس حداقل امتیاز اعتماد
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
        
        # تحلیل موازی همه نمادها
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
        
        # رتبه‌بندی سیگنال‌ها
        ranked_signals = self.signal_ranking.rank_signals(all_signals)
        max_signals = self.config.get('max_signals_per_timeframe', 3)
        top_signals = ranked_signals[:max_signals]
        
        logger.info(f"🏆 Top {len(top_signals)} signals selected for {timeframe}")
        for i, signal in enumerate(top_signals, 1):
            logger.info(f"  #{i}: {signal.symbol} {signal.signal_type.value.upper()} "
                        f"(confidence: {signal.confidence_score:.0f}, profit: {signal.predicted_profit:.2f}%)")
        
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

class MessageFormatter:
    @staticmethod
    def format_signal_message(signal: TradingSignal) -> str:
        emoji_map = {
            SignalType.BUY: "🟢",
            SignalType.SELL: "🔴",
            SignalType.HOLD: "🟡"
        }
        
        trend_emoji_map = {
            "bullish": "📈",
            "bearish": "📉",
            "sideways": "➡️"
        }
        
        signal_emoji = emoji_map.get(signal.signal_type, "⚪")
        trend_emoji = trend_emoji_map.get(signal.market_context.get('trend', 'sideways'), "➡️")
        
        reasons_text = "\n• ".join(signal.reasons)
        
        message = (
            f"{signal_emoji} **{signal.signal_type.value.upper()} SIGNAL**\n\n"
            f"📊 **Symbol:** `{signal.symbol}`\n"
            f"⏰ **Timeframe:** `{signal.timeframe}`\n"
            f"💰 **Entry Price:** `${signal.entry_price:.4f}`\n"
            f"🎯 **Target Price:** `${signal.exit_price:.4f}`\n"
            f"🛑 **Stop Loss:** `${signal.stop_loss:.4f}`\n"
            f"📈 **Predicted Profit:** `{signal.predicted_profit:.2f}%`\n"
            f"⚖️ **Risk/Reward:** `{signal.risk_reward_ratio:.2f}`\n"
            f"⭐ **Confidence:** `{signal.confidence_score:.0f}/100`\n\n"
            f"{trend_emoji} **Market Context:**\n"
            f"• Trend: {signal.market_context.get('trend', 'Unknown').title()}\n"
            f"• Volatility: {signal.market_context.get('volatility', 0):.1%}\n"
            f"• Volume Trend: {signal.market_context.get('volume_trend', 'Unknown').title()}\n\n"
            f"📋 **Analysis Reasons:**\n• {reasons_text}\n\n"
            f"🕐 **Generated:** {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        return message
    
    @staticmethod
    def format_summary_message(timeframe_results: Dict[str, List[TradingSignal]]) -> str:
        total_signals = sum(len(signals) for signals in timeframe_results.values())
        
        if total_signals == 0:
            return (
                "📊 **Trading Analysis Complete**\n\n"
                "⚠️ No strong signals found in any timeframe.\n"
                "Market conditions may not be favorable for trading at this time.\n\n"
                "🔄 Try again later or adjust your analysis parameters."
            )
        
        summary = (
            f"📊 **Trading Analysis Summary**\n\n"
            f"🎯 **Total Signals Found:** {total_signals}\n\n"
        )
        
        for timeframe, signals in timeframe_results.items():
            if signals:
                best_signal = signals[0]
                summary += (
                    f"⏰ **{timeframe.upper()}:** {len(signals)} signal(s)\n"
                    f"└ Best: {best_signal.symbol} {best_signal.signal_type.value.upper()} "
                    f"({best_signal.predicted_profit:.1f}% profit potential)\n\n"
                )
        
        summary += (
            "💡 **Next Steps:**\n"
            "• Review each signal carefully\n"
            "• Consider your risk tolerance\n"
            "• Use proper position sizing\n"
            "• Set stop losses as recommended\n\n"
            "⚠️ **Disclaimer:** These are automated signals for educational purposes only."
        )
        
        return summary

class TelegramBotHandler:
    def __init__(self, bot_token: str, config_manager: ConfigManager):
        self.bot_token = bot_token
        self.config = config_manager
        self.trading_service = TradingBotService(config_manager)
        self.formatter = MessageFormatter()
        self.user_sessions = {}
    
    def create_application(self) -> Application:
        application = Application.builder().token(self.bot_token).build()
        
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("config", self.config_command))
        application.add_handler(CommandHandler("quick", self.quick_analysis))
        application.add_handler(CallbackQueryHandler(self.button_callback))
        
        return application
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        logger.info(f"User {user_id} started the bot")
        
        keyboard = [
            [
                InlineKeyboardButton("🚀 Full Analysis", callback_data="full_analysis"),
                InlineKeyboardButton("⚡ Quick Scan", callback_data="quick_scan")
            ],
            [
                InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
                InlineKeyboardButton("ℹ️ Help", callback_data="help")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        welcome_message = (
            "🤖 **Advanced Trading Signal Bot**\n\n"
            "Welcome to the most comprehensive trading analysis bot!\n\n"
            "🎯 **Features:**\n"
            "• Multi-timeframe analysis\n"
            "• 10+ technical indicators\n"
            "• Risk/reward calculations\n"
            "• Volume & market context analysis\n"
            "• Real-time signal ranking\n\n"
            "Choose an option below to get started:"
        )
        
        await update.message.reply_text(
            welcome_message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()
        
        if query.data == "full_analysis":
            await self.run_full_analysis(query)
        elif query.data == "quick_scan":
            await self.run_quick_scan(query)
        elif query.data == "settings":
            await self.show_settings(query)
        elif query.data == "help":
            await self.show_help(query)
    
    async def run_full_analysis(self, query) -> None:
        user_id = query.from_user.id
        logger.info(f"👤 User {user_id} started full analysis")
        
        await query.edit_message_text(
            "🔄 **Starting Comprehensive Analysis...**\n\n"
            "📊 Analyzing multiple timeframes\n"
            "🔍 Processing technical indicators\n"
            "📈 Evaluating market conditions\n\n"
            "⏳ This may take 2-3 minutes...",
            parse_mode='Markdown'
        )
        
        try:
            timeframes = self.config.get('timeframes', TIME_FRAMES)
            logger.info(f"📅 Analyzing {len(timeframes)} timeframes: {', '.join(timeframes)}")
            
            results = {}
            total_signals = 0
            analysis_start_time = asyncio.get_event_loop().time()
            
            for i, timeframe in enumerate(timeframes, 1):
                logger.info(f"🔄 Processing timeframe {i}/{len(timeframes)}: {timeframe}")
                
                # به‌روزرسانی پیام وضعیت
                await query.edit_message_text(
                    f"🔄 **Analysis Progress ({i}/{len(timeframes)})**\n\n"
                    f"📊 Currently analyzing: **{timeframe.upper()}**\n"
                    f"🔍 Processing technical indicators\n"
                    f"📈 Evaluating market conditions\n\n"
                    f"⏳ Please wait...",
                    parse_mode='Markdown'
                )
                
                timeframe_start = asyncio.get_event_loop().time()
                signals = await self.trading_service.find_best_signals_for_timeframe(timeframe)
                timeframe_duration = asyncio.get_event_loop().time() - timeframe_start
                
                results[timeframe] = signals
                total_signals += len(signals)
                
                logger.info(f"✅ {timeframe} analysis completed in {timeframe_duration:.2f}s - {len(signals)} signals found")
                
                # ارسال نتیجه فوری این تایم فریم
                if signals:
                    best_signal = signals[0]
                    timeframe_result = (
                        f"✅ **{timeframe.upper()} Analysis Complete**\n\n"
                        f"🎯 **Found {len(signals)} signal(s)**\n\n"
                        f"🏆 **Best Signal:**\n"
                        f"• Symbol: `{best_signal.symbol}`\n"
                        f"• Type: {best_signal.signal_type.value.upper()}\n"
                        f"• Confidence: {best_signal.confidence_score:.0f}/100\n"
                        f"• Profit Potential: {best_signal.predicted_profit:.2f}%\n"
                        f"• Risk/Reward: {best_signal.risk_reward_ratio:.2f}\n\n"
                        f"📋 **Main Reasons:**\n"
                    )
                    
                    # نمایش 3 دلیل اول
                    for reason in best_signal.reasons[:3]:
                        timeframe_result += f"• {reason}\n"
                    
                    timeframe_result += "\n🔍 **Detailed signals will follow after complete analysis**"
                else:
                    timeframe_result = (
                        f"❌ **{timeframe.upper()} Analysis Complete**\n\n"
                        f"No strong signals found in this timeframe.\n"
                        f"Market conditions may not be favorable.\n\n"
                    )
                
                await query.message.reply_text(timeframe_result, parse_mode='Markdown')
                await asyncio.sleep(1)
            
            total_duration = asyncio.get_event_loop().time() - analysis_start_time
            logger.info(f"🎉 Full analysis completed in {total_duration:.2f}s - Total signals: {total_signals}")
            
            # ادامه کد برای نمایش نتایج...
            
        except Exception as e:
            logger.error(f"❌ Full analysis failed for user {user_id}: {e}")
            await query.edit_message_text(
                f"❌ **Analysis Error**\n\n"
                f"An error occurred during analysis:\n`{str(e)}`\n\n"
                "Please try again later or contact support.",
                parse_mode='Markdown'
            )
    
    async def run_quick_scan(self, query) -> None:
        await query.edit_message_text(
            "⚡ **Quick Scan in Progress...**\n\n"
            "🔍 Scanning 1m timeframe only\n"
            "⏳ This will take about 30 seconds...",
            parse_mode='Markdown'
        )
        
        try:
            signals = await self.trading_service.find_best_signals_for_timeframe('1m')
            
            if not signals:
                await query.edit_message_text(
                    "⚡ **Quick Scan Results**\n\n"
                    "❌ No strong signals found in 1m timeframe\n\n"
                    "💡 Try full analysis for other timeframes or wait for better market conditions.",
                    parse_mode='Markdown'
                )
                return
            
            await query.edit_message_text(
                f"⚡ **Quick Scan Results**\n\n"
                f"✅ Found {len(signals)} signal(s) in 1m timeframe",
                parse_mode='Markdown'
            )
            
            for i, signal in enumerate(signals, 1):
                signal_message = self.formatter.format_signal_message(signal)
                signal_header = f"**Quick Signal #{i}**\n\n"
                
                await query.message.reply_text(
                    signal_header + signal_message,
                    parse_mode='Markdown'
                )
                
                await asyncio.sleep(0.3)
        
        except Exception as e:
            logger.error(f"Error in quick scan: {e}")
            await query.edit_message_text(
                f"❌ **Quick Scan Error**\n\n"
                f"Error: `{str(e)}`\n\n"
                "Please try again.",
                parse_mode='Markdown'
            )
    
    async def show_settings(self, query) -> None:
        config_info = (
            "⚙️ **Current Settings**\n\n"
            f"📊 **Symbols:** {len(self.config.get('symbols', []))}\n"
            f"⏰ **Timeframes:** {', '.join(self.config.get('timeframes', []))}\n"
            f"🎯 **Min Confidence:** {self.config.get('min_confidence_score', 60)}\n"
            f"📈 **Max Signals/TF:** {self.config.get('max_signals_per_timeframe', 3)}\n"
            f"⚖️ **Min Risk/Reward:** {self.config.get('risk_reward_threshold', 1.5)}\n\n"
            "Use /config to modify settings"
        )
        
        await query.edit_message_text(config_info, parse_mode='Markdown')
    
    async def show_help(self, query) -> None:
        help_text = (
            "ℹ️ **Help & Commands**\n\n"
            "**Commands:**\n"
            "• /start - Main menu\n"
            "• /quick - Quick 1H analysis\n"
            "• /config - Settings management\n"
            "• /help - This help message\n\n"
            "**Signal Interpretation:**\n"
            "🟢 BUY - Long position recommended\n"
            "🔴 SELL - Short position recommended\n"
            "⭐ Confidence - Signal strength (0-100)\n"
            "⚖️ Risk/Reward - Profit vs loss ratio\n\n"
            "**Indicators Used:**\n"
            "• Moving Averages (SMA/EMA)\n"
            "• RSI (Relative Strength Index)\n"
            "• MACD (Moving Average Convergence Divergence)\n"
            "• Bollinger Bands\n"
            "• Stochastic Oscillator\n"
            "• Volume Analysis\n"
            "• ATR (Average True Range)\n\n"
            "⚠️ **Disclaimer:** This bot provides educational signals only. "
            "Always do your own research and never risk more than you can afford to lose."
        )
        
        await query.edit_message_text(help_text, parse_mode='Markdown')
    
    async def config_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        config_text = (
            "⚙️ **Configuration Management**\n\n"
            f"**Current Settings:**\n"
            f"• Symbols: {len(self.config.get('symbols', []))}\n"
            f"• Timeframes: {', '.join(self.config.get('timeframes', []))}\n"
            f"• Min Confidence: {self.config.get('min_confidence_score', 60)}\n"
            f"• Max Signals per TF: {self.config.get('max_signals_per_timeframe', 3)}\n"
            f"• Risk/Reward Threshold: {self.config.get('risk_reward_threshold', 1.5)}\n\n"
            "**Available Symbols:**\n"
        )
        
        symbols = self.config.get('symbols', [])
        config_text += f"```\n{', '.join(symbols)}\n```\n\n"
        config_text += "Contact admin to modify configuration."
        
        await update.message.reply_text(config_text, parse_mode='Markdown')
    
    async def quick_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        progress_msg = await update.message.reply_text(
            "⚡ **Quick Analysis Starting...**\n\n"
            "🔍 Scanning 1H timeframe for immediate opportunities...",
            parse_mode='Markdown'
        )
        
        try:
            signals = await self.trading_service.find_best_signals_for_timeframe('1h')
            
            if not signals:
                await progress_msg.edit_text(
                    "⚡ **Quick Analysis Complete**\n\n"
                    "❌ No strong signals found in 1H timeframe\n\n"
                    "💡 Use /start for comprehensive multi-timeframe analysis.",
                    parse_mode='Markdown'
                )
                return
            
            await progress_msg.edit_text(
                f"⚡ **Quick Analysis Complete**\n\n"
                f"✅ Found {len(signals)} high-confidence signal(s)",
                parse_mode='Markdown'
            )
            
            for signal in signals:
                signal_message = self.formatter.format_signal_message(signal)
                await update.message.reply_text(signal_message, parse_mode='Markdown')
                await asyncio.sleep(0.3)
        
        except Exception as e:
            logger.error(f"Error in quick analysis command: {e}")
            await progress_msg.edit_text(
                f"❌ **Quick Analysis Error**\n\n"
                f"Error: `{str(e)}`",
                parse_mode='Markdown'
            )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        help_text = (
            "🤖 **Advanced Trading Signal Bot - Help**\n\n"
            "**🚀 Commands:**\n"
            "• `/start` - Main menu with analysis options\n"
            "• `/quick` - Fast 1H timeframe scan\n"
            "• `/config` - View current configuration\n"
            "• `/help` - Display this help message\n\n"
            "**📊 Analysis Types:**\n"
            "• **Full Analysis** - Multi-timeframe comprehensive scan\n"
            "• **Quick Scan** - Rapid 1H timeframe analysis\n\n"
            "**🎯 Signal Quality:**\n"
            "• Confidence Score: 60-100 (higher is better)\n"
            "• Risk/Reward Ratio: >1.5 recommended\n"
            "• Multiple indicator confirmation required\n\n"
            "**⚠️ Risk Warning:**\n"
            "Trading involves substantial risk. These signals are for "
            "educational purposes only and should not be considered as "
            "financial advice. Always conduct your own research and "
            "never risk more than you can afford to lose.\n\n"
            "**🔧 Technical Support:**\n"
            "If you encounter any issues, please report them to the bot administrator."
        )
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def cleanup(self):
        await self.trading_service.cleanup()

def main():    
    logger.info("🚀 Starting Advanced Trading Signal Bot...")
    
    if not BOT_TOKEN:
        logger.error("❌ BOT_TOKEN environment variable is required")
        return
    
    try:
        config_manager = ConfigManager()
        logger.info("⚙️ Configuration loaded successfully")
        
        bot_handler = TelegramBotHandler(BOT_TOKEN, config_manager)
        application = bot_handler.create_application()
        
        symbols_count = len(config_manager.get('symbols', []))
        timeframes_count = len(config_manager.get('timeframes', []))
        
        logger.info("📊 Bot configured with:")
        logger.info(f"   • {symbols_count} symbols")
        logger.info(f"   • {timeframes_count} timeframes: {', '.join(config_manager.get('timeframes', []))}")
        logger.info(f"   • Min confidence: {config_manager.get('min_confidence_score', 60)}")
        logger.info("🤖 Bot is ready and waiting for commands...")
        
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )
        
    except KeyboardInterrupt:
        logger.info("⏹️ Bot stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"💥 Bot crashed with error: {e}")
        import traceback
        logger.error(f"📋 Traceback:\n{traceback.format_exc()}")
    finally:
        logger.info("🧹 Starting cleanup process...")
        asyncio.run(bot_handler.cleanup())
        logger.info("✅ Bot cleanup completed successfully")

if __name__ == "__main__":
    main()