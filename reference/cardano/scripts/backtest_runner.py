#!/usr/bin/env python
'''
Cardano Backtest Runner
---------------------
Script to run multiple backtests with different parameters
and find optimal settings.
'''

import os
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import argparse
from backtest_system import BacktestEngine, TradingStrategy, RiskManager, ImprovedTradingStrategy, load_processed_data

def run_parameter_optimization(data, base_params, param_ranges, timeframe='1h', initial_capital=10000):
    """
    Run backtests with different parameter combinations to find optimal settings.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing processed price data
    base_params : dict
        Base parameters to use (only specified parameters will be varied)
    param_ranges : dict
        Dictionary of parameter names and lists of values to test
    timeframe : str
        Timeframe being tested (for display purposes)
    initial_capital : float
        Initial capital for backtests
        
    Returns:
    --------
    list : Results of all parameter combinations
    """
    # Create all combinations of parameters
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    combinations = list(itertools.product(*param_values))
    
    results = []
    
    # Run backtest for each parameter combination
    for i, combo in enumerate(combinations):
        # Create parameter dictionary for this run
        run_params = base_params.copy()
        for j, param_name in enumerate(param_names):
            run_params[param_name] = combo[j]
        
        # Create strategy with these parameters
        strategy = ImprovedTradingStrategy(params=run_params)
        
        # Create risk manager
        risk_manager = RiskManager(initial_capital=initial_capital)
        
        # Create and run backtest
        backtest = BacktestEngine(
            data=data,
            strategy=strategy,
            risk_manager=risk_manager
        )
        
        print(f"Running parameter combination {i+1}/{len(combinations)}...")
        backtest.run()
        
        # Get performance stats
        stats = backtest.get_performance_stats()
        
        # Add parameters to results
        result = {
            'params': run_params.copy(),
            'stats': stats
        }
        
        # Print brief summary
        print(f"  Return: {stats['total_return_pct']:.2f}%, Win Rate: {stats['win_rate']:.2f}%")
        
        results.append(result)
    
    # Sort results by total return
    results.sort(key=lambda x: x['stats']['total_return_pct'], reverse=True)
    
    return results

def plot_optimization_results(results, param_name, metric='total_return_pct'):
    """
    Plot the relationship between a parameter and a performance metric.
    
    Parameters:
    -----------
    results : list
        Results from parameter optimization
    param_name : str
        Name of parameter to plot
    metric : str
        Name of metric to plot
    """
    # Extract parameter values and metric values
    param_values = []
    metric_values = []
    
    for result in results:
        if param_name in result['params']:
            param_values.append(result['params'][param_name])
            metric_values.append(result['stats'][metric])
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'param_value': param_values,
        'metric_value': metric_values
    })
    
    # Group by parameter value and calculate mean
    grouped = df.groupby('param_value').mean().reset_index()
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['param_value'], df['metric_value'], alpha=0.3)
    plt.plot(grouped['param_value'], grouped['metric_value'], 'r-', linewidth=2)
    
    # Set labels and title
    plt.xlabel(param_name)
    plt.ylabel(metric)
    plt.title(f'Effect of {param_name} on {metric}')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def run_optimization_grid(data, timeframe, initial_capital=10000, output_dir='../backtest_results'):
    """
    Run a comprehensive grid optimization for strategy parameters.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Processed price data
    timeframe : str
        Timeframe being tested
    initial_capital : float
        Initial capital for backtests
    output_dir : str
        Directory to save results
    """
    # Define base parameters and ranges to test
    base_params = {
        'rsi_oversold_threshold': 30,
        'rsi_overbought_threshold': 70,
        'stop_atr_multiplier': 2.0,
        'take_profit_atr_multiplier': 3.0,
        'max_bb_width': 0.2,
        'min_rel_volume': 1.2,
        'trend_filter_enabled': True,
        'macd_filter_enabled': True,
        'support_resistance_filter': True
    }
    
    # Parameter ranges for initial optimization
    param_ranges = {
        'rsi_oversold_threshold': [25, 30, 35],
        'rsi_overbought_threshold': [65, 70, 75],
        'stop_atr_multiplier': [1.5, 2.0, 2.5, 3.0],
        'take_profit_atr_multiplier': [2.0, 3.0, 4.0, 5.0]
    }
    
    # Run parameter optimization
    results = run_parameter_optimization(
        data=data,
        base_params=base_params,
        param_ranges=param_ranges,
        timeframe=timeframe,
        initial_capital=initial_capital
    )
    
    # Print best parameter combination
    best_result = results[0]
    print("\n" + "="*50)
    print("BEST PARAMETER COMBINATION")
    print("="*50)
    print("Parameters:")
    for param_name, param_value in best_result['params'].items():
        print(f"  {param_name}: {param_value}")
    
    print("\nPerformance:")
    stats = best_result['stats']
    print(f"  Total Return: {stats['total_return_pct']:.2f}%")
    print(f"  Win Rate: {stats['win_rate']:.2f}%")
    print(f"  Profit Factor: {stats['profit_factor']:.2f}")
    print(f"  Maximum Drawdown: {stats['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    print("="*50)
    
    # Save results to CSV
    results_df = pd.DataFrame([
        {
            **{f'param_{k}': v for k, v in r['params'].items()},
            **{f'stat_{k}': v for k, v in r['stats'].items()}
        } for r in results
    ])
    
    results_path = os.path.join(output_dir, f"optimization_results_{timeframe}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Optimization results saved to {results_path}")
    
    # Plot relationship between parameters and performance
    for param_name in param_ranges.keys():
        plot_optimization_results(results, param_name)
    
    # Run a backtest with the best parameters
    run_backtest_with_params(data, best_result['params'], timeframe, initial_capital, output_dir)
    
    return best_result

def run_backtest_with_params(data, params, timeframe, initial_capital=10000, output_dir='../backtest_results'):
    """
    Run a backtest with specific parameters.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Processed price data
    params : dict
        Strategy parameters
    timeframe : str
        Timeframe being tested
    initial_capital : float
        Initial capital for backtest
    output_dir : str
        Directory to save results
    """
    # Create strategy with the specified parameters
    strategy = ImprovedTradingStrategy(params=params)
    
    # Create risk manager
    risk_manager = RiskManager(initial_capital=initial_capital)
    
    # Create and run backtest
    backtest = BacktestEngine(
        data=data,
        strategy=strategy,
        risk_manager=risk_manager
    )
    
    print("\nRunning backtest with optimal parameters...")
    backtest.run()
    
    # Get performance stats
    stats = backtest.get_performance_stats()
    
    # Print results
    print("\n" + "="*50)
    print("OPTIMAL PARAMETER BACKTEST RESULTS")
    print("="*50)
    print(f"Timeframe: {timeframe}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Capital: ${backtest.risk_manager.current_capital:,.2f}")
    print(f"Total Return: ${stats['total_return']:,.2f} ({stats['total_return_pct']:.2f}%)")
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Win Rate: {stats['win_rate']:.2f}%")
    print(f"Profit Factor: {stats['profit_factor']:.2f}")
    print(f"Average Trade: ${stats['avg_trade']:,.2f}")
    print(f"Maximum Drawdown: {stats['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    print("="*50)
    
    # Plot results
    plot_path = os.path.join(output_dir, f"optimal_backtest_{timeframe}.png")
    try:
        backtest.plot_results(save_path=plot_path)
        print(f"Results plot saved to {plot_path}")
        
        # Also show the plot
        backtest.plot_results()
    except Exception as e:
        print(f"Error plotting results: {str(e)}")

def run_timeframe_comparison(timeframes=['1h', '4h', '1d'], initial_capital=10000, output_dir='../backtest_results'):
    """
    Run backtests on multiple timeframes and compare performance.
    
    Parameters:
    -----------
    timeframes : list
        List of timeframes to test
    initial_capital : float
        Initial capital for backtests
    output_dir : str
        Directory to save results
    """
    results = []
    
    for timeframe in timeframes:
        print(f"\n{'='*50}")
        print(f"RUNNING BACKTEST ON {timeframe} TIMEFRAME")
        print(f"{'='*50}")
        
        # Load data
        data = load_processed_data(timeframe)
        if data is None:
            print(f"Failed to load data for {timeframe} timeframe. Skipping.")
            continue
        
        # Create strategy
        strategy = ImprovedTradingStrategy()
        
        # Create risk manager
        risk_manager = RiskManager(initial_capital=initial_capital)
        
        # Create and run backtest
        backtest = BacktestEngine(
            data=data,
            strategy=strategy,
            risk_manager=risk_manager
        )
        
        print(f"Running backtest on {timeframe} timeframe...")
        backtest.run()
        
        # Get performance stats
        stats = backtest.get_performance_stats()
        
        # Add timeframe to results
        result = {
            'timeframe': timeframe,
            'stats': stats
        }
        
        results.append(result)
        
        # Print results
        print(f"\nBacktest results for {timeframe} timeframe:")
        print(f"  Total Return: {stats['total_return_pct']:.2f}%")
        print(f"  Win Rate: {stats['win_rate']:.2f}%")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        print(f"  Maximum Drawdown: {stats['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        
        # Plot results
        plot_path = os.path.join(output_dir, f"backtest_{timeframe}.png")
        try:
            backtest.plot_results(save_path=plot_path)
            print(f"Results plot saved to {plot_path}")
        except Exception as e:
            print(f"Error plotting results: {str(e)}")
    
    # Create comparison chart
    if results:
        # Create DataFrame for plotting
        comparison_df = pd.DataFrame([
            {
                'Timeframe': r['timeframe'],
                'Return (%)': r['stats']['total_return_pct'],
                'Win Rate (%)': r['stats']['win_rate'],
                'Profit Factor': r['stats']['profit_factor'],
                'Max Drawdown (%)': r['stats']['max_drawdown_pct'],
                'Sharpe Ratio': r['stats']['sharpe_ratio']
            } for r in results
        ])
        
        # Save comparison to CSV
        comparison_path = os.path.join(output_dir, "timeframe_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nTimeframe comparison saved to {comparison_path}")
        
        # Plot comparison
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Return
        axs[0, 0].bar(comparison_df['Timeframe'], comparison_df['Return (%)'])
        axs[0, 0].set_title('Total Return (%)')
        axs[0, 0].grid(True, alpha=0.3)
        
        # Win Rate
        axs[0, 1].bar(comparison_df['Timeframe'], comparison_df['Win Rate (%)'])
        axs[0, 1].set_title('Win Rate (%)')
        axs[0, 1].grid(True, alpha=0.3)
        
        # Profit Factor
        axs[1, 0].bar(comparison_df['Timeframe'], comparison_df['Profit Factor'])
        axs[1, 0].set_title('Profit Factor')
        axs[1, 0].grid(True, alpha=0.3)
        
        # Max Drawdown
        axs[1, 1].bar(comparison_df['Timeframe'], comparison_df['Max Drawdown (%)'])
        axs[1, 1].set_title('Maximum Drawdown (%)')
        axs[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_plot_path = os.path.join(output_dir, "timeframe_comparison.png")
        plt.savefig(comparison_plot_path)
        print(f"Timeframe comparison plot saved to {comparison_plot_path}")
        
        # Show comparison plot
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cardano Backtest Runner')
    parser.add_argument('--timeframe', type=str, default='1h', 
                      choices=['1h', '4h', '1d'],
                      help='Timeframe to use (default: 1h)')
    parser.add_argument('--capital', type=float, default=10000,
                      help='Initial capital (default: $10,000)')
    parser.add_argument('--optimize', action='store_true',
                      help='Run parameter optimization')
    parser.add_argument('--improved', action='store_true',
                      help='Use improved strategy')
    parser.add_argument('--compare-timeframes', action='store_true',
                      help='Compare performance across timeframes')
    parser.add_argument('--output', type=str, default='../backtest_results',
                      help='Output directory for backtest results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    if args.compare_timeframes:
        # Run timeframe comparison
        run_timeframe_comparison(
            timeframes=['1h', '4h', '1d'],
            initial_capital=args.capital,
            output_dir=args.output
        )
    elif args.optimize:
        # Load data
        data = load_processed_data(args.timeframe)
        if data is None:
            print("Failed to load data. Exiting.")
            exit(1)
        
        # Run optimization
        run_optimization_grid(
            data=data,
            timeframe=args.timeframe,
            initial_capital=args.capital,
            output_dir=args.output
        )
    else:
        # Load data
        data = load_processed_data(args.timeframe)
        if data is None:
            print("Failed to load data. Exiting.")
            exit(1)
        
        # Run a single backtest
        if args.improved:
            # Use improved strategy
            strategy = ImprovedTradingStrategy()
            print("Using improved strategy...")
        else:
            # Use basic strategy
            strategy = TradingStrategy()
            print("Using basic strategy...")
        
        # Create risk manager
        risk_manager = RiskManager(
            initial_capital=args.capital
        )
        
        # Create and run backtest
        backtest = BacktestEngine(
            data=data,
            strategy=strategy,
            risk_manager=risk_manager,
            trading_fee_pct=0.1
        )
        
        print("Running backtest...")
        backtest.run()
        
        # Get performance stats
        stats = backtest.get_performance_stats()
        
        # Print results
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Timeframe: {args.timeframe}")
        print(f"Initial Capital: ${args.capital:,.2f}")
        print(f"Final Capital: ${backtest.risk_manager.current_capital:,.2f}")
        print(f"Total Return: ${stats['total_return']:,.2f} ({stats['total_return_pct']:.2f}%)")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']:.2f}%")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        print(f"Average Trade: ${stats['avg_trade']:,.2f}")
        print(f"Maximum Drawdown: {stats['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        print("="*50)
        
        # Plot results
        plot_path = os.path.join(args.output, f"backtest_{args.timeframe}.png")
        try:
            backtest.plot_results(save_path=plot_path)
            print(f"Results plot saved to {plot_path}")
            
            # Also show the plot
            backtest.plot_results()
        except Exception as e:
            print(f"Error plotting results: {str(e)}")
