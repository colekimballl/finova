# backtesting_comprehensive.py

import backtrader as bt
from datetime import datetime
import logging
import csv
import os
import pandas as pd
import importlib.util
from multiprocessing import Pool
from itertools import product

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtests.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom Analyzer for Win Rate
class WinRateAnalyzer(bt.Analyzer):
    def __init__(self):
        self.won = 0
        self.lost = 0

    def notify_trade(self, trade):
        if trade.isclosed:
            if trade.pnl > 0:
                self.won += 1
            elif trade.pnl < 0:
                self.lost += 1

    def get_analysis(self):
        total = self.won + self.lost
        win_rate = (self.won / total) * 100 if total > 0 else 0
        return {'win_rate': win_rate, 'total_won': self.won, 'total_lost': self.lost}

# Function to dynamically load strategy classes
def load_strategy(module_path, class_name):
    spec = importlib.util.spec_from_file_location(class_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

# Function to generate parameter combinations for optimization
def generate_param_combinations(params):
    keys, values = zip(*params.items())
    for v in product(*values):
        yield dict(zip(keys, v))

# Function to run a single backtest
def run_backtest(strategy_info):
    strategy_name, module_path, class_name, params, data_path, results_file = strategy_info
    try:
        # Load strategy class
        StrategyClass = load_strategy(module_path, class_name)
        
        # Setup Cerebro
        cerebro = bt.Cerebro()
        cerebro.addstrategy(StrategyClass, **params)
        
        # Load data
        data = bt.feeds.YahooFinanceCSVData(
            dataname=data_path,
            fromdate=datetime(2017, 1, 6),
            todate=datetime(2022, 5, 4),
            reverse=False
        )
        cerebro.adddata(data)
        
        # Broker settings
        cerebro.broker.set_cash(1000000)
        cerebro.broker.setcommission(commission=0.001)
        cerebro.addsizer(bt.sizers.AllInSizer, percents=95)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.Transactions, _name='tx')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(WinRateAnalyzer, _name='winrate')
        
        # Run backtest
        results = cerebro.run()
        strat = results[0]
        
        # Extract metrics
        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 'N/A')
        txs = strat.analyzers.tx.get_analysis()
        trades = strat.analyzers.trades.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        winrate = strat.analyzers.winrate.get_analysis()
        endvalue = cerebro.broker.getvalue()
        
        # Calculate metrics
        total_transactions = len(txs) if txs else 0
        total_trades = trades['total']['total'] if 'total' in trades else 0
        profitable_trades = trades['won']['total'] if 'won' in trades else 0
        losing_trades = trades['lost']['total'] if 'lost' in trades else 0
        max_drawdown = drawdown['max']['drawdown'] if 'max' in drawdown else 'N/A'
        total_return = returns.get('rtot', 'N/A')
        win_rate = winrate.get('win_rate', 'N/A')
        
        # Log results
        logger.info(f"Strategy: {strategy_name} | Params: {params}")
        logger.info(f"Sharpe Ratio: {sharpe}")
        logger.info(f"Win Rate: {win_rate}% | Won: {profitable_trades} | Lost: {losing_trades}")
        logger.info(f"Total Transactions: {total_transactions}")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Max Drawdown: {max_drawdown}%")
        logger.info(f"Total Return: {total_return}%")
        logger.info(f"Final Portfolio Value: ${endvalue:,.2f}")
        logger.info("-" * 50)
        
        # Save to CSV using Pandas
        result_data = {
            'Strategy': strategy_name,
            'Parameters': str(params),
            'Sharpe Ratio': sharpe,
            'Win Rate (%)': win_rate,
            'Total Transactions': total_transactions,
            'Total Trades': total_trades,
            'Won Trades': profitable_trades,
            'Lost Trades': losing_trades,
            'Max Drawdown (%)': max_drawdown,
            'Total Return (%)': total_return,
            'Final Value ($)': endvalue
        }
        
        df = pd.DataFrame([result_data])
        
        # Append to CSV
        if not os.path.isfile(results_file):
            df.to_csv(results_file, index=False)
        else:
            df.to_csv(results_file, mode='a', header=False, index=False)
        
        # Plot results
        cerebro.plot(style='candlestick')
    
    except Exception as e:
        logger.error(f"Error running backtest for {strategy_name}: {e}")

def main():
    # Define strategies with parameter grids for optimization
    strategies = [
        {
            'strategy_name': 'SmaCross',
            'module_path': '/Users/colekimball/ztech/finova/scripts/backtesting/strategies/sma_cross.py',
            'class_name': 'SmaCross',
            'params': {'sma_period': [20, 25, 30]}
        },
        {
            'strategy_name': 'SmaRsiStrategy',
            'module_path': '/Users/colekimball/ztech/finova/scripts/backtesting/strategies/sma_rsi_strategy.py',
            'class_name': 'SmaRsiStrategy',
            'params': {
                'sma_period': [20, 25],
                'rsi_period': [14, 21],
                'rsi_overbought': [70, 75],
                'rsi_oversold': [30, 25]
            }
        },
        # Add more strategies and parameter grids as needed
    ]
    
    # Prepare list of backtest jobs with all parameter combinations
    backtest_jobs = []
    data_path = '/Users/colekimball/ztech/finova/data/historical/BTC-USD_clean.csv'
    results_file = 'backtest_results.csv'
    
    for strat in strategies:
        strategy_name = strat['strategy_name']
        module_path = strat['module_path']
        class_name = strat['class_name']
        param_grid = strat['params']
        
        param_combinations = generate_param_combinations(param_grid)
        
        for params in param_combinations:
            backtest_jobs.append((
                strategy_name,
                module_path,
                class_name,
                params,
                data_path,
                results_file
            ))
    
    # Ensure results file exists or create it
    if not os.path.isfile(results_file):
        with open(results_file, 'w', newline='') as csvfile:
            fieldnames = ['Strategy', 'Parameters', 'Sharpe Ratio', 'Win Rate (%)', 'Total Transactions', 
                          'Total Trades', 'Won Trades', 'Lost Trades', 'Max Drawdown (%)', 
                          'Total Return (%)', 'Final Value ($)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    # Run backtests in parallel
    pool = Pool(processes=4)  # Adjust based on CPU cores
    pool.map(run_backtest, backtest_jobs)
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
