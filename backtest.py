

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

class BacktestingEngine:
    def __init__(self, trading_service):
        self.trading_service = trading_service
        self.portfolio = None
        
    async def run_backtest(self, symbol: str, timeframe: str, start: str, end: str, 
                            initial_capital: float = 10000, commission_rate: float = 0.001) -> Dict[str, Any]:
        
        data = await self.trading_service.exchange_manager.fetch_ohlcv_data(symbol, timeframe, limit=2000)
        if data.empty:
            return {}
        
        data = data[(data['timestamp'] >= pd.to_datetime(start)) & (data['timestamp'] <= pd.to_datetime(end))]
        if data.empty:
            return {}
        
        data = data.set_index('timestamp').sort_index()
        
        self.portfolio = Portfolio(initial_capital, commission_rate)
        
        signals = await self._generate_signals(data)
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_prices = {
                'open': row['open'],
                'high': row['high'], 
                'low': row['low'],
                'close': row['close']
            }
            
            trades_to_close = self.portfolio.check_stop_loss_take_profit(timestamp, row['close'])
            for trade_idx in reversed(trades_to_close):
                self.portfolio.close_trade(timestamp, row['close'], trade_idx)
            
            signal = signals.get(timestamp)
            if signal:
                if signal['action'] in ['buy', 'short'] and not self.portfolio.open_trades:
                    self.portfolio.open_trade(timestamp, row['close'], signal)
                elif signal['action'] in ['sell', 'cover'] and self.portfolio.open_trades:
                    self.portfolio.close_trade(timestamp, row['close'])
            
            self.portfolio.update_equity(timestamp, current_prices)
        
        for trade_idx in range(len(self.portfolio.open_trades)):
            self.portfolio.close_trade(data.index[-1], data.iloc[-1]['close'], 0)
        
        return self._calculate_performance_metrics()
    
    async def _generate_signals(self, data: pd.DataFrame) -> Dict[datetime, Dict[str, Any]]:
        signals = {}
        
        if hasattr(self.trading_service, 'signal_generator'):
            try:
                raw_signals = await self.trading_service.signal_generator.generate_signals(data)
                
                for signal in raw_signals:
                    if isinstance(signal, dict) and 'timestamp' in signal:
                        timestamp = pd.to_datetime(signal['timestamp'])
                        signals[timestamp] = signal
                        
            except Exception as e:
                print(f"Error generating signals: {e}")
        
        if not signals and hasattr(self.trading_service, 'walk_forward_optimizer'):
            optimizer = self.trading_service.walk_forward_optimizer
            if hasattr(optimizer, 'best_params') and optimizer.best_params:
                raw_signals = optimizer.generate_signals_with_params(data, optimizer.best_params)
                
                for signal in raw_signals:
                    if isinstance(signal, dict):
                        timestamp = signal.get('timestamp')
                        if timestamp:
                            if isinstance(timestamp, str):
                                timestamp = pd.to_datetime(timestamp)
                            signals[timestamp] = signal
        
        return signals
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        if not self.portfolio.closed_trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
        
        trades = self.portfolio.trade_history
        total_trades = len(trades)
        
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        total_wins = sum(t['pnl'] for t in winning_trades)
        total_losses = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        if self.portfolio.equity_curve:
            equity_values = [e['equity'] for e in self.portfolio.equity_curve]
            returns = np.diff(equity_values) / equity_values[:-1]
            
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
            
            negative_returns = returns[returns < 0]
            downside_std = np.std(negative_returns) if len(negative_returns) > 0 else np.std(returns)
            sortino_ratio = np.mean(returns) / downside_std * np.sqrt(252) if downside_std != 0 else 0
            
            peak = np.maximum.accumulate(equity_values)
            drawdown = (np.array(equity_values) - peak) / peak
            max_drawdown = np.min(drawdown) * 100
            
            total_return = (equity_values[-1] - self.portfolio.initial_capital) / self.portfolio.initial_capital * 100
        else:
            sharpe_ratio = sortino_ratio = max_drawdown = total_return = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'total_return': round(total_return, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'total_pnl': round(sum(t['pnl'] for t in trades), 2),
            'total_commission': round(sum(t['commission'] for t in trades), 2),
            'final_capital': round(self.portfolio.capital, 2),
            'equity_curve': self.portfolio.equity_curve,
            'trade_history': trades
        }

    async def run_backtest_with_params(self, data: pd.DataFrame, params: Dict[str, Any], 
                                        initial_capital: float = 10000) -> Dict[str, Any]:
        if data.empty or params is None:
            return {}
        
        self.portfolio = Portfolio(initial_capital)
        
        optimizer = self.trading_service.walk_forward_optimizer
        signals_list = optimizer.generate_signals_with_params(data, params)
        
        signals = {}
        for signal in signals_list:
            if isinstance(signal, dict) and 'timestamp' in signal:
                timestamp = pd.to_datetime(signal['timestamp']) if isinstance(signal['timestamp'], str) else signal['timestamp']
                signals[timestamp] = signal
        
        for timestamp, row in data.iterrows():
            current_prices = {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'], 
                'close': row['close']
            }
            
            trades_to_close = self.portfolio.check_stop_loss_take_profit(timestamp, row['close'])
            for trade_idx in reversed(trades_to_close):
                self.portfolio.close_trade(timestamp, row['close'], trade_idx)
            
            signal = signals.get(timestamp)
            if signal:
                if signal['action'] in ['buy', 'short'] and not self.portfolio.open_trades:
                    self.portfolio.open_trade(timestamp, row['close'], signal)
                elif signal['action'] in ['sell', 'cover'] and self.portfolio.open_trades:
                    self.portfolio.close_trade(timestamp, row['close'])
            
            self.portfolio.update_equity(timestamp, current_prices)
        
        for trade_idx in range(len(self.portfolio.open_trades)):
            self.portfolio.close_trade(data.index[-1], data.iloc[-1]['close'], 0)
        
        return self._calculate_performance_metrics()

class Trade:
    def __init__(self, entry_time: datetime, entry_price: float, quantity: float, 
                    trade_type: str, stop_loss: float = None, take_profit: float = None):
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