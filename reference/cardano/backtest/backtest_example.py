#!/usr/bin/env python
'''
Project Solaris AI - Backtest Example
------------------------------------
Example script showing how to use the backtesting framework.
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
from backtest.simple_backtest import SimpleBacktest, simple_ma_crossover, rsi_strategy, bollinger_band_strategy

# Create output directories if they don't exist
os.makedirs('backtest/plots', exist_ok=True)
os.makedirs('backtest/results', exist_ok=True)

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
            'params': {'short_period': 20, 'long_period': 50},
            'metrics': {}
        },
        'RSI Strategy': {
            'function': rsi_strategy,
            'params': {'rsi_period': 14, 'overbought': 70, 'oversold': 30},
            'metrics': {}
        },
        'Bollinger Bands Strategy': {
            'function': bollinger_band_strategy,
            'params': {'period': 20, 'std_dev': 2},
            'metrics': {}
        }
    }
    
    # Run each strategy
    for strategy_name, strategy_info in strategy_results.items():
        print(f"\nRunning {strategy_name}...")
        
        # Create backtest instance
        backtest = SimpleBacktest(timeframe=timeframe, initial_capital=initial_capital)
        
        # Run the strategy
        backtest.run_strategy(strategy_info['function'], **strategy_info['params'])
        
        # Store the metrics
        strategy_info['metrics'] = backtest.metrics.copy()
        
        # Print metrics
        backtest.print_metrics()
        
        # Plot and save results
        filename = strategy_name.lower().replace(' ', '_')
        backtest.plot_results(save_path=f'backtest/plots/{filename}_{timeframe}.png')
        
        # Save the results dataframe
        backtest.results.to_csv(f'backtest/results/{filename}_{timeframe}.csv')
    
    # Compare the strategies
    print("\n" + "="*50)
    print(" STRATEGY COMPARISON")
    print("="*50)
    
    metrics_df = pd.DataFrame({
        name: {
            'Total Return': info['metrics']['total_return'],
            '
