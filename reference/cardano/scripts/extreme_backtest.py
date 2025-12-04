#!/usr/bin/env python
'''
Extreme Crypto Trading Strategy (Moon Shot Version) - Part 2
-----------------------------------------------------------
Backtesting engine for ultra-aggressive trading
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtest_system import BacktestEngine, Trade, load_processed_data
from extreme_strategy import ExtremeRiskManager, ExtremeStrategy
import argparse

class ExtremeBacktestEngine(BacktestEngine):
    """
    Enhanced backtesting engine with advanced features.
    """
    def __init__(self, data, strategy, risk_manager, trading_fee_pct=0.1):
        super().__init__(data, strategy, risk_manager, trading_fee_pct)
        self.trailing_stops = {}  # Track trailing stops for open trades
    
    def _manage_open_trades(self, index, time, price):
        """
        Manage open trades with trailing stops.
        """
        # Check each open trade
        remaining_trades = []
        for trade in self.open_trades:
            # Update trailing stop if enabled
            if hasattr(self.strategy, 'use_trailing_stop') and self.strategy.use_trailing_stop:
                self._update_trailing_stop(trade, price)
            
            # Get current stop level (trailing or original)
            current_stop = (self.trailing_stops.get(id(trade), {}).get('current_stop', trade.stop_loss))
            
            # Check if stop is hit
            if ((trade.direction == 'long' and price <= current_stop) or
                (trade.direction == 'short' and price >= current_stop)):
                self._close_trade(trade, time, current_stop, 'stop_loss')
                continue
            
            # Check if take profit is hit
            if trade.is_take_profit_hit(price):
                self._close_trade(trade, time, trade.take_profit, 'take_profit')
                continue
            
            # Keep the trade open
            remaining_trades.append(trade)
        
        # Update open trades list
        self.open_trades = remaining_trades
    
    def _update_trailing_stop(self, trade, current_price):
        """
        Update trailing stop if conditions are met.
        """
        # Initialize trailing stop info if needed
        if id(trade) not in self.trailing_stops:
            self.trailing_stops[id(trade)] = {
                'activated': False,
                'current_stop': trade.stop_loss
            }
        
        trail_info = self.trailing_stops[id(trade)]
        trail_activation_pct = self.strategy.trail_activation_pct / 100
        
        if trade.direction == 'long':
            # Activation threshold
            activation_price = trade.entry_price * (1 + trail_activation_pct)
            
            # Check if price reached activation threshold
            if not trail_info['activated'] and current_price >= activation_price:
                trail_info['activated'] = True
            
            # If activated, move stop up as price rises
            if trail_info['activated']:
                # New stop is X ATR below current price
                atr = self.data.iloc[-1]['atr_14'] if 'atr_14' in self.data.columns else 0
                new_stop = current_price - (atr * 1.5)
                
                # Only move stop up, never down
                if new_stop > trail_info['current_stop']:
                    trail_info['current_stop'] = new_stop
        
        else:  # short
            # Activation threshold for shorts
            activation_price = trade.entry_price * (1 - trail_activation_pct)
            
            # Check if price reached activation threshold
            if not trail_info['activated'] and current_price <= activation_price:
                trail_info['activated'] = True
            
            # If activated, move stop down as price falls
            if trail_info['activated']:
                # New stop is X ATR above current price
                atr = self.data.iloc[-1]['atr_14'] if 'atr_14' in self.data.columns else 0
                new_stop = current_price + (atr * 1.5)
                
                # Only move stop down, never up
                if new_stop < trail_info['current_stop']:
                    trail_info['current_stop'] = new_stop


def run_extreme_backtest(timeframe='1h', initial_capital=10000, max_leverage=20, risk_per_trade=10, output_dir='../backtest_results'):
    """
    Run the extreme strategy backtest.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = load_processed_data(timeframe)
    if data is None:
        print(f"Failed to load {timeframe} data. Exiting.")
        return
    
    print(f"Running extreme strategy backtest on {len(data)} rows of {timeframe} data...")
    
    # Create strategy and risk manager
    strategy = ExtremeStrategy()
    risk_manager = ExtremeRiskManager(
        initial_capital=initial_capital,
        max_drawdown_pct=50,
        risk_per_trade_pct=risk_per_trade,
        max_leverage=max_leverage
    )
    
    # Create and run backtest
    backtest = ExtremeBacktestEngine(
        data=data,
        strategy=strategy,
        risk_manager=risk_manager,
        trading_fee_pct=0.1
    )
    
    print("Running extreme backtest...")
    backtest.run()
    
    # Get performance stats
    stats = backtest.get_performance_stats()
    
    # Print results
    print("\n" + "="*50)
    print("EXTREME STRATEGY BACKTEST RESULTS")
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
    print("="*50)
    
    # Save trade statistics
    if backtest.trades:
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
        
        trade_df = pd.DataFrame(trade_stats)
        trade_stats_path = os.path.join(output_dir, f"extreme_trade_stats_{timeframe}.csv")
        trade_df.to_csv(trade_stats_path, index=False)
        print(f"Detailed trade statistics saved to {trade_stats_path}")
    
    # Plot results
    try:
        backtest.plot_results()
    except Exception as e:
        print(f"Error plotting results: {str(e)}")
    
    return backtest, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extreme Crypto Trading Strategy')
    parser.add_argument('--timeframe', type=str, default='1h', 
                      choices=['1h', '4h', '1d'],
                      help='Timeframe to use (default: 1h)')
    parser.add_argument('--capital', type=float, default=10000,
                      help='Initial capital (default: $10,000)')
    parser.add_argument('--leverage', type=float, default=20,
                      help='Maximum leverage (default: 20x)')
    parser.add_argument('--risk', type=float, default=10,
                      help='Risk per trade percentage (default: 10%)')
    
    args = parser.parse_args()
    
    run_extreme_backtest(
        timeframe=args.timeframe,
        initial_capital=args.capital,
        max_leverage=args.leverage,
        risk_per_trade=args.risk
    )
