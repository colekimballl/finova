#!/usr/bin/env python
'''
Project Solaris AI - Backtest Runner
-----------------------------------
Example script showing how to use the backtesting framework.
This script runs multiple predefined strategies and compares their performance.
'''

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Make sure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the backtest framework and strategies
from simple_backtest import (
    SimpleBacktest, 
    simple_ma_crossover, 
    rsi_strategy, 
    bollinger_band_strategy
)

# Create output directories
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

def run_single_strategy(strategy_name, strategy_func, strategy_params, timeframe, initial_capital):
    """
    Run a single backtest strategy.
    
    Parameters:
    -----------
    strategy_name : str
        Name of the strategy for display purposes
    strategy_func : function
        The strategy function to run
    strategy_params : dict
        Parameters to pass to the strategy function
    timeframe : str
        Timeframe to use for the backtest
    initial_capital : float
        Initial capital to use for the backtest
    
    Returns:
    --------
    tuple
        (backtest instance, metrics dictionary)
    """
    print(f"\nRunning {strategy_name} on {timeframe} timeframe...")
    
    # Create backtest instance
    backtest = SimpleBacktest(
        symbol='ADA-USD',
        timeframe=timeframe,
        initial_capital=initial_capital
    )
    
    # Run the strategy
    backtest.run_strategy(strategy_func, **strategy_params)
    
    # Print metrics
    backtest.print_metrics()
    
    # Create a clean filename
    filename = strategy_name.lower().replace(' ', '_').replace('-', '_')
    
    # Plot and save results
    backtest.plot_results(
        save_path=f'plots/{filename}_{timeframe}.png'
    )
    
    # Save detailed results to CSV
    backtest.results.to_csv(f'results/{filename}_{timeframe}.csv')
    
    print(f"Results saved to plots/{filename}_{timeframe}.png and results/{filename}_{timeframe}.csv")
    
    return backtest, backtest.metrics

def compare_strategies(timeframe='1h', initial_capital=10000):
    """
    Compare different trading strategies on the same timeframe.
    
    Parameters:
    -----------
    timeframe : str
        Timeframe to use (default: '1h')
    initial_capital : float
        Initial capital (default: 10000)
    """
    print(f"Comparing strategies on {timeframe} timeframe with ${initial_capital} initial capital...")
    
    # Dictionary to store results
    strategy_results = {
        'Moving Average Crossover': {
            'function': simple_ma_crossover,
            'params': {'short_period': 20, 'long_period': 50}
        },
        'RSI Strategy': {
            'function': rsi_strategy,
            'params': {'rsi_period': 14, 'overbought': 70, 'oversold': 30}
        },
        'Bollinger Bands Strategy': {
            'function': bollinger_band_strategy,
            'params': {'period': 20, 'std_dev': 2}
        }
    }
    
    # Run each strategy and collect metrics
    metrics_data = {}
    
    for strategy_name, strategy_info in strategy_results.items():
        _, metrics = run_single_strategy(
            strategy_name,
            strategy_info['function'],
            strategy_info['params'],
            timeframe,
            initial_capital
        )
        metrics_data[strategy_name] = metrics
    
    # Create a comparison DataFrame
    comparison_metrics = ['total_return', 'annual_return', 'sharpe_ratio', 
                         'max_drawdown', 'win_rate', 'num_trades']
    
    comparison_data = {}
    for strategy_name, metrics in metrics_data.items():
        comparison_data[strategy_name] = {
            metric: metrics.get(metric, 0) for metric in comparison_metrics
        }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create a more readable version of the comparison
    readable_df = pd.DataFrame({
        'Metric': [
            'Total Return (%)',
            'Annual Return (%)',
            'Sharpe Ratio',
            'Max Drawdown (%)',
            'Win Rate (%)',
            'Number of Trades'
        ]
    })
    
    for strategy_name in comparison_data.keys():
        readable_df[strategy_name] = [
            f"{comparison_data[strategy_name]['total_return']:.2%}",
            f"{comparison_data[strategy_name]['annual_return']:.2%}",
            f"{comparison_data[strategy_name]['sharpe_ratio']:.2f}",
            f"{comparison_data[strategy_name]['max_drawdown']:.2%}",
            f"{comparison_data[strategy_name]['win_rate']:.2%}",
            f"{comparison_data[strategy_name]['num_trades']}"
        ]
    
    # Save comparison to CSV
    comparison_df.to_csv(f'results/strategy_comparison_{timeframe}.csv')
    
    # Print comparison
    print("\n" + "="*80)
    print(f" STRATEGY COMPARISON - {timeframe} Timeframe")
    print("="*80)
    print(readable_df.to_string(index=False))
    print("="*80)
    
    # Plot comparison chart for total returns
    plt.figure(figsize=(10, 6))
    returns = [metrics_data[s]['total_return'] for s in comparison_data.keys()]
    plt.bar(comparison_data.keys(), returns)
    plt.title(f'Strategy Comparison - Total Return ({timeframe} Timeframe)')
    plt.ylabel('Total Return')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels to the bars
    for i, v in enumerate(returns):
        plt.text(i, v + 0.01, f"{v:.2%}", ha='center')
    
    plt.tight_layout()
    plt.savefig(f'plots/strategy_comparison_{timeframe}.png')
    
    print(f"Comparison chart saved to plots/strategy_comparison_{timeframe}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run backtest comparisons for Cardano.')
    parser.add_argument('--timeframe', type=str, default='1h', 
                      choices=['1m', '5m', '15m', '1h', '4h', '1d'],
                      help='Timeframe to use (default: 1h)')
    parser.add_argument('--capital', type=float, default=10000,
                      help='Initial capital (default: 10000)')
    parser.add_argument('--strategy', type=str, default='all',
                      choices=['all', 'ma', 'rsi', 'bb'],
                      help='Strategy to run (default: all)')
    
    args = parser.parse_args()
    
    if args.strategy == 'all':
        # Run comparison of all strategies
        compare_strategies(args.timeframe, args.capital)
    else:
        # Run a single strategy
        if args.strategy == 'ma':
            strategy_name = 'Moving Average Crossover'
            strategy_func = simple_ma_crossover
            strategy_params = {'short_period': 20, 'long_period': 50}
        elif args.strategy == 'rsi':
            strategy_name = 'RSI Strategy'
            strategy_func = rsi_strategy
            strategy_params = {'rsi_period': 14, 'overbought': 70, 'oversold': 30}
        elif args.strategy == 'bb':
            strategy_name = 'Bollinger Bands Strategy'
            strategy_func = bollinger_band_strategy
            strategy_params = {'period': 20, 'std_dev': 2}
        
        # Run the selected strategy
        run_single_strategy(
            strategy_name,
            strategy_func,
            strategy_params,
            args.timeframe,
            args.capital
        )
