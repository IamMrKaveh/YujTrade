from indicators.manager import ATRIndicator, FibonacciLevels, PivotPoints, RSIIndicator, SupportResistanceAnalyzer, VolumeAnalyzer
from models.market import DynamicLevels, MarketAnalysis, MarketCondition, TrendDirection, TrendStrength
import pandas as pd
import numpy as np

from models.signal import SignalType

class TrendAnalyzer:
    @staticmethod
    def calculate_trend_strength(data: pd.DataFrame) -> TrendStrength:
        if len(data) < 50:
            return TrendStrength.WEAK
        
        sma_20 = data['close'].rolling(window=20).mean()
        sma_50 = data['close'].rolling(window=50).mean()
        
        current_sma_20 = sma_20.iloc[-1]
        current_sma_50 = sma_50.iloc[-1]
        
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
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 0
    
    
    
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
        sma_20 = data['close'].rolling(window=20).mean()
        sma_50 = data['close'].rolling(window=50).mean()
        
        if len(sma_20) < 50 or len(sma_50) < 50:
            return TrendDirection.SIDEWAYS
        
        current_sma_20 = sma_20.iloc[-1]
        current_sma_50 = sma_50.iloc[-1]
        
        if current_sma_20 > current_sma_50 * 1.015:
            return TrendDirection.BULLISH
        elif current_sma_20 < current_sma_50 * 0.985:
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
    
    def _calculate_trend_acceleration(self, data: pd.DataFrame) -> float:
        if len(data) < 10:
            return 0.0
        
        recent_momentum = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]
        previous_momentum = (data['close'].iloc[-6] - data['close'].iloc[-10]) / data['close'].iloc[-10]
        
        return (recent_momentum - previous_momentum) * 100
    
    def _determine_market_condition(self, data: pd.DataFrame) -> MarketCondition:
        rsi_indicator = RSIIndicator()
        rsi_result = rsi_indicator.calculate(data)
        
        if rsi_result.value > 70:
            return MarketCondition.OVERBOUGHT
        elif rsi_result.value < 30:
            return MarketCondition.OVERSOLD
        else:
            return MarketCondition.NEUTRAL

class DynamicLevelCalculator:
    def __init__(self):
        self.fibonacci = FibonacciLevels()
        self.pivot_points = PivotPoints()
    
    def calculate_dynamic_levels(self, data: pd.DataFrame, signal_type: SignalType, 
                                market_analysis: MarketAnalysis) -> DynamicLevels:
        current_price = data['close'].iloc[-1]
        high_20 = data['high'].tail(20).max()
        low_20 = data['low'].tail(20).min()
        
        atr_indicator = ATRIndicator()
        atr_result = atr_indicator.calculate(data)
        atr_value = atr_result.value
        
        if signal_type == SignalType.BUY:
            return self._calculate_buy_levels(data, current_price, high_20, low_20, 
                                            atr_value, market_analysis)
        else:
            return self._calculate_sell_levels(data, current_price, high_20, low_20, 
                                                atr_value, market_analysis)
    
    def _calculate_buy_levels(self, data: pd.DataFrame, current_price: float, 
                            high_20: float, low_20: float, atr_value: float,
                            market_analysis: MarketAnalysis) -> DynamicLevels:
        
        fib_levels = self.fibonacci.calculate_retracement_levels(high_20, low_20)
        pivot_levels = self.pivot_points.calculate_pivot_levels(
            data['high'].iloc[-1], data['low'].iloc[-1], data['close'].iloc[-1]
        )
        
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
            nearest_resistance = min([r for r in market_analysis.resistance_levels 
                                    if r > current_price], default=primary_exit)
            primary_exit = min(primary_exit, nearest_resistance)
            secondary_exit = min(secondary_exit, nearest_resistance * 1.02)
        
        tight_stop = max(
            current_price - (atr_value * volatility_multiplier),
            fib_levels['0.618'] if fib_levels['0.618'] < current_price else current_price * 0.98
        )
        
        wide_stop = max(
            current_price - (atr_value * 2 * volatility_multiplier),
            low_20 * 0.995
        )
        
        if market_analysis.support_levels:
            nearest_support = max([s for s in market_analysis.support_levels 
                                    if s < current_price], default=tight_stop)
            tight_stop = max(tight_stop, nearest_support)
            wide_stop = max(wide_stop, nearest_support * 0.995)
        
        breakeven_point = current_price + (atr_value * 0.5)
        trailing_stop = current_price - (atr_value * 1.5 * volatility_multiplier)
        
        return DynamicLevels(
            primary_entry=primary_entry,
            secondary_entry=secondary_entry,
            primary_exit=primary_exit,
            secondary_exit=secondary_exit,
            tight_stop=tight_stop,
            wide_stop=wide_stop,
            breakeven_point=breakeven_point,
            trailing_stop=trailing_stop
        )
    
    def _calculate_sell_levels(self, data: pd.DataFrame, current_price: float, 
                                high_20: float, low_20: float, atr_value: float,
                                market_analysis: MarketAnalysis) -> DynamicLevels:
        
        fib_levels = self.fibonacci.calculate_retracement_levels(high_20, low_20)
        pivot_levels = self.pivot_points.calculate_pivot_levels(
            data['high'].iloc[-1], data['low'].iloc[-1], data['close'].iloc[-1]
        )
        
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
            nearest_support = max([s for s in market_analysis.support_levels 
                                    if s < current_price], default=primary_exit)
            primary_exit = max(primary_exit, nearest_support)
            secondary_exit = max(secondary_exit, nearest_support * 0.98)
        
        tight_stop = min(
            current_price + (atr_value * volatility_multiplier),
            fib_levels['0.618'] if fib_levels['0.618'] > current_price else current_price * 1.02
        )
        
        wide_stop = min(
            current_price + (atr_value * 2 * volatility_multiplier),
            high_20 * 1.005
        )
        
        if market_analysis.resistance_levels:
            nearest_resistance = min([r for r in market_analysis.resistance_levels 
                                    if r > current_price], default=tight_stop)
            tight_stop = min(tight_stop, nearest_resistance)
            wide_stop = min(wide_stop, nearest_resistance * 1.005)
        
        breakeven_point = current_price - (atr_value * 0.5)
        trailing_stop = current_price + (atr_value * 1.5 * volatility_multiplier)
        
        return DynamicLevels(
            primary_entry=primary_entry,
            secondary_entry=secondary_entry,
            primary_exit=primary_exit,
            secondary_exit=secondary_exit,
            tight_stop=tight_stop,
            wide_stop=wide_stop,
            breakeven_point=breakeven_point,
            trailing_stop=trailing_stop
        )
    
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