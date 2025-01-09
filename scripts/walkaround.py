# scripts/walk_forward.py

import backtrader as bt
from datetime import datetime
import pandas as pd

# Define your strategy
from strategies.sma_cross import SmaCross

def run_walk_forward(train_start, train_end, test_start, test_end, params, data_path):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCross, **params)
    
    # Load training data
    train_data = bt.feeds.YahooFinanceCSVData(
        dataname=data_path,
        fromdate=train_start,
        todate=train_end,
        reverse=False
    )
    cerebro.adddata(train_data)
    
    # Broker settings
    cerebro.broker.set_cash(1000000)
    cerebro.broker.setcommission(commission=0.001)
    
    # Add analyzers as needed
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    
    # Run training backtest
    results = cerebro.run()
    strat = results[0]
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 'N/A')
    print(f"Training Sharpe Ratio: {sharpe}")
    
    # Setup for testing
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCross, **params)
    
    # Load testing data
    test_data = bt.feeds.YahooFinanceCSVData(
        dataname=data_path,
        fromdate=test_start,
        todate=test_end,
        reverse=False
    )
    cerebro.adddata(test_data)
    
    # Broker settings
    cerebro.broker.set_cash(1000000)
    cerebro.broker.setcommission(commission=0.001)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    
    # Run testing backtest
    results = cerebro.run()
    strat = results[0]
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 'N/A')
    print(f"Testing Sharpe Ratio: {sharpe}")

# Example usage
train_start = datetime(2017, 1, 6)
train_end = datetime(2020, 12, 31)
test_start = datetime(2021, 1, 1)
test_end = datetime(2022, 5, 4)
params = {'sma_period': 20}
data_path = '/Users/colekimball/ztech/finova/data/historical/BTC-USD_clean.csv'

run_walk_forward(train_start, train_end, test_start, test_end, params, data_path)

