from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from indicators.manager import ATRIndicator, BollingerBandsIndicator, MACDIndicator, MovingAverageIndicator, RSIIndicator, StochasticIndicator, VolumeIndicator
from markets.manager import DynamicLevelCalculator, MarketConditionAnalyzer
from models.indicator import IndicatorResult
from models.market import MarketAnalysis, TrendDirection, TrendStrength
from models.signal import SignalType, TradingSignal
from logger_config import logger


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
        self.level_calculator = DynamicLevelCalculator()
    
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
        
        buy_signal = self._evaluate_buy_signal(indicator_results, data, symbol, timeframe, market_analysis)
        if buy_signal:
            logger.info(f"🟢 BUY signal generated for {symbol} on {timeframe} - Confidence: {buy_signal.confidence_score:.0f}")
            signals.append(buy_signal)
        
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
        
        if score >= 60:
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
                risk_reward_ratio=self._calculate_risk_reward(
                    dynamic_levels.primary_entry, 
                    dynamic_levels.primary_exit, 
                    dynamic_levels.tight_stop
                ),
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
        
        if score >= 60:
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
                risk_reward_ratio=self._calculate_risk_reward(
                    dynamic_levels.primary_entry, 
                    dynamic_levels.primary_exit, 
                    dynamic_levels.tight_stop
                ),
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
        
        return None
    
    def _calculate_risk_reward(self, entry: float, exit: float, stop_loss: float) -> float:
        if entry == stop_loss:
            return 0
        
        potential_profit = abs(exit - entry)
        potential_loss = abs(entry - stop_loss)
        
        return potential_profit / potential_loss if potential_loss > 0 else 0
    
    def _create_market_context(self, market_analysis: MarketAnalysis) -> Dict[str, Any]:
        return {
            'trend': market_analysis.trend.value,
            'trend_strength': market_analysis.trend_strength.value,
            'volatility': market_analysis.volatility,
            'momentum_score': market_analysis.momentum_score,
            'market_condition': market_analysis.market_condition.value,
            'volume_trend': market_analysis.volume_trend,
            'trend_acceleration': market_analysis.trend_acceleration,
            'volume_confirmation': market_analysis.volume_confirmation
        }



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