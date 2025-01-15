# backtesting_enhanced.py

import backtrader as bt
from datetime import datetime
import logging
import csv
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define multiple strategies
class SmaCross(bt.SignalStrategy):
    params = (('sma_period', 20),)

    def __init__(self):
        sma = bt.ind.SMA(period=self.p.sma_period)
        crossover = bt.ind.CrossOver(self.data.close, sma)
        self.signal_add(bt.SIGNAL_LONG, crossover)

class SmaRsiStrategy(bt.SignalStrategy):
    params = (
        ('sma_period', 20),
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30)
    )

    def __init__(self):
        sma = bt.ind.SMA(period=self.p.sma_period)
        rsi = bt.ind.RSI_SMA(period=self.p.rsi_period)
        crossover = bt.ind.CrossOver(self.data.close, sma)

        # Long signal: Price crosses above SMA and RSI is not overbought
        long_signal = (crossover > 0) & (rsi < self.p.rsi_overbought)
        self.signal_add(bt.SIGNAL_LONG, long_signal)

        # Short signal: Price crosses below SMA and RSI is not oversold
        short_signal = (crossover < 0) & (rsi > self.p.rsi_oversold)
        self.signal_add(bt.SIGNAL_SHORT, short_signal)

# Function to load data
def load_data(filepath, from_date, to_date):
    if not os.path.exists(filepath):
        logger.error(f"Data file {filepath} not found.")
        raise FileNotFoundError(f"Data file {filepath} not found.")
    
    return bt.feeds.YahooFinanceCSVData(
        dataname=filepath,
        fromdate=from_date,
        todate=to_date, 
        reverse=False
    )

# Function to setup cerebro
def setup_cerebro(strategy, params, data):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy, **params)
    cerebro.adddata(data)
    cerebro.broker.set_cash(1000000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addsizer(bt.sizers.AllInSizer, percents=95)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Transactions, _name='tx')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    return cerebro

# Function to run backtest and save results
def run_and_save_backtest(cerebro, strategy_name, params, results_file='backtest_results.csv'):
    try:
        results = cerebro.run()
        strat = results[0]
        
        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 'N/A')
        txs = strat.analyzers.tx.get_analysis()
        trades = strat.analyzers.trades.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        endvalue = cerebro.broker.getvalue()
        
        total_transactions = len(txs)
        total_trades = trades['total']['total'] if 'total' in trades else 0
        profitable_trades = trades['won']['total'] if 'won' in trades else 0
        losing_trades = trades['lost']['total'] if 'lost' in trades else 0
        max_drawdown = drawdown['max']['drawdown'] if 'max' in drawdown else 'N/A'
        total_return = returns.get('rtot', 'N/A')
        
        # Log results
        logger.info(f"Strategy: {strategy_name} | Params: {params}")
        logger.info(f"Sharpe Ratio: {sharpe}")
        logger.info(f"Total Transactions: {total_transactions}")
        logger.info(f"Total Trades: {total_trades} | Won: {profitable_trades} | Lost: {losing_trades}")
        logger.info(f"Max Drawdown: {max_drawdown}%")
        logger.info(f"Total Return: {total_return}%")
        logger.info(f"Final Portfolio Value: ${endvalue:,.2f}")
        logger.info("-" * 50)
        
        # Save to CSV
        file_exists = os.path.isfile(results_file)
        with open(results_file, 'a', newline='') as csvfile:
            fieldnames = ['Strategy', 'Parameters', 'Sharpe Ratio', 'Total Transactions', 
                          'Total Trades', 'Won Trades', 'Lost Trades', 'Max Drawdown', 
                          'Total Return', 'Final Value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file does not exist
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'Strategy': strategy_name,
                'Parameters': str(params),
                'Sharpe Ratio': sharpe,
                'Total Transactions': total_transactions,
                'Total Trades': total_trades,
                'Won Trades': profitable_trades,
                'Lost Trades': losing_trades,
                'Max Drawdown': max_drawdown,
                'Total Return': total_return,
                'Final Value': endvalue
            })
        
        # Plot results
        cerebro.plot(style='candlestick')
    
    except Exception as e:
        logger.error(f"Error during backtest for {strategy_name}: {e}")

def main():
    # Path to your CSV data
    data_path = '/Users/colekimball/ztech/finova/data/historical/BTC-USD.csv'
    
    # Load data
    data = load_data(data_path, datetime(2017, 1, 6), datetime(2022, 5, 4))
    
    # Define strategies and their parameters
    strategies = [
        ('SmaCross', SmaCross, {'sma_period': 20}),
        ('SmaCrossShort', SmaCross, {'sma_period': 10}),
        ('SmaCrossLong', SmaCross, {'sma_period': 50}),
        ('SmaRsiStrategy', SmaRsiStrategy, {
            'sma_period': 20, 
            'rsi_period': 14, 
            'rsi_overbought': 70, 
            'rsi_oversold': 30
        }),
    ]
    
    # Iterate through each strategy
    for strategy_name, strategy, params in strategies:
        logger.info(f"Starting backtest for strategy: {strategy_name} with params: {params}")
        cerebro = setup_cerebro(strategy, params, data)
        run_and_save_backtest(cerebro, strategy_name, params)
        logger.info(f"Completed backtest for strategy: {strategy_name}\n")

if __name__ == "__main__":
    main()
