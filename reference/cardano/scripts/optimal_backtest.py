#!/usr/bin/env python
'''
Cardano Optimal Backtest
-------------------------
Run a backtest using the optimal parameters found during optimization
across the entire history of Cardano data.
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from backtest_system import BacktestEngine, RiskManager, ImprovedTradingStrategy, load_processed_data

# Optimal parameters found from optimization
OPTIMAL_PARAMS = {
    'rsi_oversold_threshold': 25,
    'rsi_overbought_threshold': 65,
    'stop_atr_multiplier': 1.5,
    'take_profit_atr_multiplier': 5.0,
    'max_bb_width': 0.2,
    'min_rel_volume': 1.2,
    'trend_filter_enabled': True,
    'macd_filter_enabled': True,
    'support_resistance_filter': True
}

def run_optimal_backtest(timeframe='1h', initial_capital=10000, trading_fee=0.1, output_dir='../backtest_results'):
    """
    Run a backtest with optimal parameters.
    
    Parameters:
    -----------
    timeframe : str
        Timeframe to use ('1h', '4h', '1d')
    initial_capital : float
        Initial capital for backtest
    trading_fee : float
        Trading fee percentage
    output_dir : str
        Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = load_processed_data(timeframe)
    if data is None:
        print(f"Failed to load {timeframe} data. Exiting.")
        return
    
    print(f"Running optimal parameter backtest on {len(data)} rows of {timeframe} data...")
    print("Using the following optimal parameters:")
    for param, value in OPTIMAL_PARAMS.items():
        print(f"  {param}: {value}")
    
    # Create strategy with optimal parameters
    strategy = ImprovedTradingStrategy(params=OPTIMAL_PARAMS)
    
    # Create risk manager
    risk_manager = RiskManager(
        initial_capital=initial_capital,
        max_drawdown_pct=15,  # Increased to allow testing through drawdown periods
        risk_per_trade_pct=1
    )
    
    # Create and run backtest
    backtest = BacktestEngine(
        data=data,
        strategy=strategy,
        risk_manager=risk_manager,
        trading_fee_pct=trading_fee
    )
    
    print("Running backtest...")
    backtest.run()
    
    # Get performance stats
    stats = backtest.get_performance_stats()
    
    # Print results
    print("\n" + "="*50)
    print("OPTIMAL PARAMETER BACKTEST RESULTS")
    print("="*50)
    print(f"Timeframe: {timeframe}")
    print(f"Data Range: {data['datetime'].min().date()} to {data['datetime'].max().date()}")
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
    
    # Calculate annual returns
    if len(backtest.trades) > 0:
        first_trade_date = min(trade.entry_time for trade in backtest.trades)
        last_trade_date = max(trade.exit_time if trade.exit_time else trade.entry_time for trade in backtest.trades)
        years_diff = (last_trade_date - first_trade_date).days / 365.25
        if years_diff > 0:
            annual_return = (stats['total_return_pct'] / years_diff)
            print(f"Annualized Return: {annual_return:.2f}%")
    
    # Plot results
    plot_path = os.path.join(output_dir, f"optimal_backtest_{timeframe}_full_history.png")
    try:
        backtest.plot_results(save_path=plot_path)
        print(f"Results plot saved to {plot_path}")
        
        # Also show the plot
        backtest.plot_results()
    except Exception as e:
        print(f"Error plotting results: {str(e)}")
    
    # Save detailed trade statistics
    trade_stats = []
    for i, trade in enumerate(backtest.trades):
        trade_stats.append({
            'trade_number': i + 1,
            'direction': trade.direction,
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'position_size': trade.position_size,
            'profit_loss': trade.profit_loss,
            'profit_loss_pct': trade.profit_loss_pct,
            'exit_reason': trade.exit_reason
        })
    
    if trade_stats:
        trade_df = pd.DataFrame(trade_stats)
        trade_stats_path = os.path.join(output_dir, f"trade_stats_{timeframe}_full_history.csv")
        trade_df.to_csv(trade_stats_path, index=False)
        print(f"Detailed trade statistics saved to {trade_stats_path}")
    
    return backtest, stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cardano Optimal Backtest')
    parser.add_argument('--timeframe', type=str, default='1h', 
                      choices=['1h', '4h', '1d'],
                      help='Timeframe to use (default: 1h)')
    parser.add_argument('--capital', type=float, default=10000,
                      help='Initial capital (default: $10,000)')
    parser.add_argument('--fee', type=float, default=0.1,
                      help='Trading fee percentage (default: 0.1%)')
    parser.add_argument('--output', type=str, default='../backtest_results',
                      help='Output directory for backtest results')
    
    args = parser.parse_args()
    
    run_optimal_backtest(
        timeframe=args.timeframe,
        initial_capital=args.capital,
        trading_fee=args.fee,
        output_dir=args.output
    )
