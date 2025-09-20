from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

import numpy as np


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

class Trade:
    def __init__(self, entry_time: datetime, entry_price: float, quantity: float, 
                    trade_type: str, stop_loss: float = 0.0, take_profit: float = 0.0):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.quantity = quantity
        self.trade_type = trade_type
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.exit_time = None
        self.exit_price = None
        self.pnl = 0.0
        self.commission = 0.0
        self.status = "open"

    def close_trade(self, exit_time: datetime, exit_price: float, commission_rate: float = 0.001):
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.status = "closed"
        
        if self.trade_type == "long":
            gross_pnl = (exit_price - self.entry_price) * self.quantity
        else:
            gross_pnl = (self.entry_price - exit_price) * self.quantity
            
        self.commission = (self.entry_price + exit_price) * self.quantity * commission_rate
        self.pnl = gross_pnl - self.commission

class Portfolio:
    def __init__(self, initial_capital: float, commission_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_rate = commission_rate
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve = []
        self.trade_history = []

    def open_trade(self, timestamp: datetime, price: float, signal: Dict[str, Any]) -> bool:
        if len(self.open_trades) > 0:
            return False
            
        trade_type = "long" if signal['action'] == 'buy' else "short"
        risk_amount = self.capital * signal.get('risk_percent', 0.02)
        
        if signal.get('stop_loss'):
            if trade_type == "long":
                risk_per_share = abs(price - signal['stop_loss'])
            else:
                risk_per_share = abs(signal['stop_loss'] - price)
            
            if risk_per_share > 0:
                quantity = risk_amount / risk_per_share
            else:
                quantity = self.capital * 0.1 / price
        else:
            quantity = self.capital * 0.1 / price

        if quantity * price > self.capital * 0.95:
            return False

        trade = Trade(
            entry_time=timestamp,
            entry_price=price,
            quantity=quantity,
            trade_type=trade_type,
            stop_loss=signal.get('stop_loss'),
            take_profit=signal.get('take_profit')
        )
        
        self.open_trades.append(trade)
        return True

    def close_trade(self, timestamp: datetime, price: float, trade_idx: int = 0) -> bool:
        if not self.open_trades:
            return False
            
        trade = self.open_trades.pop(trade_idx)
        trade.close_trade(timestamp, price, self.commission_rate)
        
        self.capital += trade.pnl
        self.closed_trades.append(trade)
        
        self.trade_history.append({
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'quantity': trade.quantity,
            'type': trade.trade_type,
            'pnl': trade.pnl,
            'commission': trade.commission,
            'return_pct': (trade.pnl / (trade.entry_price * trade.quantity)) * 100
        })
        
        return True

    def update_equity(self, timestamp: datetime, current_prices: Dict[str, float]):
        unrealized_pnl = 0
        for trade in self.open_trades:
            current_price = current_prices.get('close', trade.entry_price)
            if trade.trade_type == "long":
                unrealized_pnl += (current_price - trade.entry_price) * trade.quantity
            else:
                unrealized_pnl += (trade.entry_price - current_price) * trade.quantity
        
        total_equity = self.capital + unrealized_pnl
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'cash': self.capital,
            'unrealized_pnl': unrealized_pnl
        })

    def check_stop_loss_take_profit(self, timestamp: datetime, current_price: float) -> List[int]:
        trades_to_close = []
        
        for i, trade in enumerate(self.open_trades):
            should_close = False
            
            if trade.trade_type == "long":
                if trade.stop_loss and current_price <= trade.stop_loss:
                    should_close = True
                elif trade.take_profit and current_price >= trade.take_profit:
                    should_close = True
            else:
                if trade.stop_loss and current_price >= trade.stop_loss:
                    should_close = True
                elif trade.take_profit and current_price <= trade.take_profit:
                    should_close = True
            
            if should_close:
                trades_to_close.append(i)
        
        return trades_to_close