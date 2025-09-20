import asyncio
import backtrader as bt
import pandas as pd

from module.logger_config import logger


class BacktraderStrategy(bt.Strategy):
    params = (
        ("signal_generator", None),
        ("symbol", None),
        ("timeframe", None),
        ("owner_engine", None),
    )

    def __init__(self):
        self.signal_generator = self.p.signal_generator
        self.symbol = self.p.symbol
        self.timeframe = self.p.timeframe
        self.owner_engine = self.p.owner_engine
        self.order = None
        self.loop = self.owner_engine.loop

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        logger.debug(f"{dt.isoformat()} - {txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}")
            elif order.issell():
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}")
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f"OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}")

    def next(self):
        if self.order:
            return

        historical_data = self.owner_engine.get_historical_data(len(self))
        if historical_data.empty:
            return

        signals = self.loop.run_until_complete(
            self.signal_generator.generate_signals(historical_data, self.symbol, self.timeframe)
        )

        if not signals:
            return

        signal = signals[0]

        if not self.position:
            if signal.signal_type.value == "buy":
                self.log(f"BUY CREATE, {self.datas[0].close[0]:.2f}")
                self.order = self.buy()
            elif signal.signal_type.value == "sell":
                self.log(f"SELL CREATE, {self.datas[0].close[0]:.2f}")
                self.order = self.sell()
        else:
            if self.position.size > 0 and signal.signal_type.value == "sell":
                self.log(f"CLOSE (SELL) CREATE, {self.datas[0].close[0]:.2f}")
                self.order = self.close()
            elif self.position.size < 0 and signal.signal_type.value == "buy":
                self.log(f"CLOSE (BUY) CREATE, {self.datas[0].close[0]:.2f}")
                self.order = self.close()


class BacktestingEngine:
    def __init__(self, trading_service):
        self.trading_service = trading_service
        self.full_data = None
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    def get_historical_data(self, current_index):
        if self.full_data is not None and not self.full_data.empty:
            end_pos = self.full_data.index.get_loc(self.full_data.index[current_index -1]) + 1
            return self.full_data.iloc[:end_pos]
        return pd.DataFrame()

    def run_backtest(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        initial_capital: float = 10000,
        commission_rate: float = 0.001,
    ):
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.broker.setcash(initial_capital)
        cerebro.broker.setcommission(commission=commission_rate)

        self.full_data = self.loop.run_until_complete(
            self.trading_service.exchange_manager.fetch_ohlcv_data(symbol, timeframe, limit=5000)
        )
        if self.full_data.empty:
            logger.error(f"No data for {symbol} on {timeframe}")
            return {}
        
        self.full_data = self.full_data.set_index("timestamp")

        data_feed = bt.feeds.PandasData(
            dataname=self.full_data[start:end],
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            openinterest=-1,
        )
        cerebro.adddata(data_feed)

        cerebro.addstrategy(
            BacktraderStrategy,
            signal_generator=self.trading_service.signal_generator,
            symbol=symbol,
            timeframe=timeframe,
            owner_engine=self,
        )
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

        results = cerebro.run()
        strat = results[0]

        trade_analysis = strat.analyzers.trades.get_analysis()

        return {
            "initial_capital": initial_capital,
            "final_capital": cerebro.broker.getvalue(),
            "total_return_pct": (cerebro.broker.getvalue() / initial_capital - 1) * 100,
            "sharpe_ratio": strat.analyzers.sharpe.get_analysis().get("sharperatio", 0),
            "max_drawdown_pct": strat.analyzers.drawdown.get_analysis().max.drawdown,
            "total_trades": trade_analysis.total.total,
            "winning_trades": trade_analysis.won.total,
            "losing_trades": trade_analysis.lost.total,
            "win_rate_pct": (trade_analysis.won.total / trade_analysis.total.total * 100)
            if trade_analysis.total.total > 0
            else 0,
        }