from datetime import datetime, timedelta
from typing import Any, Dict

import numpy as np
import pandas as pd
import ta

from module.core import Portfolio, TradingSignal


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
        
        signals = await self._generate_signals(data, symbol, timeframe)
        
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
    
    async def _generate_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[datetime, Dict[str, Any]]:
        signals = {}
        
        if hasattr(self.trading_service, 'signal_generator'):
            try:
                raw_signals = await self.trading_service.signal_generator.generate_signals(data, symbol, timeframe)
                
                for signal in raw_signals:
                    if hasattr(signal, 'timestamp'):
                        timestamp = pd.to_datetime(signal.timestamp)
                        signals[timestamp] = {
                            'action': signal.signal_type.value,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.exit_price
                        }
                        
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
            returns = pd.Series(equity_values).pct_change().dropna()
            
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
            
            negative_returns = returns[returns < 0]
            downside_std = np.std(negative_returns) if len(negative_returns) > 0 else np.std(returns)
            sortino_ratio = np.mean(returns) / downside_std * np.sqrt(252) if downside_std != 0 else 0
            
            peak = pd.Series(equity_values).cummax()
            drawdown = (pd.Series(equity_values) - peak) / peak
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

class WalkForwardOptimizer:
    def __init__(self, trading_service):
        self.trading_service = trading_service
        self.best_params = None
        self.performance_history = []
    
    async def optimize_parameters(self, train_data):
        param_combinations = [
            {'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70},
            {'rsi_period': 21, 'rsi_oversold': 25, 'rsi_overbought': 75},
            {'rsi_period': 14, 'rsi_oversold': 35, 'rsi_overbought': 65},
            {'rsi_period': 28, 'rsi_oversold': 20, 'rsi_overbought': 80}
        ]
        
        best_params = None
        best_score = -float('inf')
        
        for params in param_combinations:
            try:
                signals = self.generate_signals_with_params(train_data, params)
                if len(signals) > 0:
                    score = self.calculate_parameter_score(train_data, signals)
                    if score > best_score:
                        best_score = score
                        best_params = params
            except Exception:
                continue
        
        return best_params or param_combinations[0]
    
    def generate_signals_with_params(self, data, params):
        rsi = ta.momentum.RSIIndicator(data['close'], window=params['rsi_period']).rsi()
        signals = []
        
        for i in range(1, len(data)):
            if rsi.iloc[i-1] <= params['rsi_oversold'] and rsi.iloc[i] > params['rsi_oversold']:
                signals.append({'timestamp': data.index[i], 'action': 'buy', 'price': data['close'].iloc[i]})
            elif rsi.iloc[i-1] >= params['rsi_overbought'] and rsi.iloc[i] < params['rsi_overbought']:
                signals.append({'timestamp': data.index[i], 'action': 'sell', 'price': data['close'].iloc[i]})
        
        return signals
    
    def calculate_parameter_score(self, data, signals):
        if len(signals) < 2:
            return -1
        
        returns = []
        position = 0
        entry_price = 0
        
        for signal in signals:
            if signal['action'] == 'buy' and position == 0:
                position = 1
                entry_price = signal['price']
            elif signal['action'] == 'sell' and position == 1:
                ret = (signal['price'] - entry_price) / entry_price
                returns.append(ret)
                position = 0
        
        if not returns:
            return -1
        
        avg_return = sum(returns) / len(returns)
        return avg_return * len(returns)
    
    async def run(self, symbol: str, timeframe: str, lookback_days: int, test_days: int):
        end = datetime.now()
        start = end - timedelta(days=lookback_days + test_days * 5)
        df = await self.trading_service.exchange_manager.fetch_ohlcv_data(symbol, timeframe, limit=2000)
        
        if df.empty:
            return {}
        
        df = df.set_index('timestamp').sort_index()
        results = []
        window = timedelta(days=lookback_days)
        step = timedelta(days=test_days)
        current_start = df.index[0]
        
        while current_start + window + step <= df.index[-1]:
            train_data = df[current_start:current_start + window]
            test_data = df[current_start + window:current_start + window + step]
            
            if len(train_data) < 50 or len(test_data) < 10:
                current_start = current_start + step
                continue
            
            optimized_params = await self.optimize_parameters(train_data)
            
            backtest_result = await self.trading_service.backtesting_engine.run_backtest_with_params(
                test_data, optimized_params
            )
            
            period_result = {
                'train_start': current_start,
                'train_end': current_start + window,
                'test_start': current_start + window,
                'test_end': current_start + window + step,
                'params': optimized_params,
                'performance': backtest_result
            }
            
            results.append(period_result)
            self.performance_history.append(backtest_result)
            current_start = current_start + step
        
        self.best_params = self.select_best_params(results)
        return {
            'periods': results,
            'best_params': self.best_params,
            'avg_performance': self.calculate_average_performance()
        }
    
    def select_best_params(self, results):
        if not results:
            return None
        
        param_scores = {}
        for result in results:
            param_key = str(sorted(result['params'].items()))
            if param_key not in param_scores:
                param_scores[param_key] = {'params': result['params'], 'scores': []}
            
            sharpe = result['performance'].get('sharpe_ratio', 0)
            param_scores[param_key]['scores'].append(sharpe)
        
        best_param_key = None
        best_avg_score = -float('inf')
        
        for param_key, data in param_scores.items():
            avg_score = sum(data['scores']) / len(data['scores'])
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_param_key = param_key
        
        return param_scores[best_param_key]['params'] if best_param_key else None
    
    def calculate_average_performance(self):
        if not self.performance_history:
            return {}
        
        avg_metrics = {}
        for metric in ['sharpe_ratio', 'sortino_ratio', 'max_drawdown']:
            values = [p.get(metric, 0) for p in self.performance_history if p.get(metric) is not None]
            avg_metrics[metric] = sum(values) / len(values) if values else 0
        
        return avg_metrics

class PaperTradingSimulator:
    def __init__(self, initial_balance: float):
        self.balance = initial_balance
        self.positions = {}
        self.trade_history = []
    
    def simulate_trade(self, signal: TradingSignal):
        if signal.signal_type == "BUY":
            self.positions[signal.symbol] = self.balance / signal.entry_price if signal.entry_price else 0
            self.balance = 0
        elif signal.signal_type == "SELL" and signal.symbol in self.positions:
            self.balance = self.positions[signal.symbol] * signal.exit_price
            del self.positions[signal.symbol]
        self.trade_history.append(signal)
    
    def get_performance(self) -> Dict[str, float]:
        total_trades = len(self.trade_history)
        profit_loss = self.balance - 0
        return {"total_trades": total_trades, "final_balance": self.balance, "profit_loss": profit_loss}