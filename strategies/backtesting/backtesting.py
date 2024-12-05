# backtesting/backtesting.py
from datetime import datetime
import backtrader as bt
from strategies.sma_cross_strategy import SmaCross


def run_backtest(data_path: str):
    """
    Run backtest using Backtrader framework.

    Parameters:
    - data_path (str): Path to the historical data CSV file.

    Returns:
    - None
    """
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCross)

    data = bt.feeds.YahooFinanceCSVData(
        dataname=data_path,
        fromdate=datetime(2017, 1, 6),
        todate=datetime(2022, 5, 4),
        reverse=False,
    )
    cerebro.adddata(data)
    cerebro.broker.set_cash(1000000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.Transactions, _name="tx")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    back = cerebro.run()
    sharpe = back[0].analyzers.sharpe.get_analysis()
    txs = back[0].analyzers.tx.get_analysis()
    trades = back[0].analyzers.trades.get_analysis()
    endvalue = cerebro.broker.getvalue()

    print(f"Sharpe Ratio: {sharpe}")
    print(f"Transactions: {len(txs)}")
    print(f"Final Portfolio Value: {endvalue}")
    cerebro.plot()
