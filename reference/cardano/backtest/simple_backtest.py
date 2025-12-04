#!/usr/bin/env python
'''
Project Solaris AI - Simple Backtesting Framework
------------------------------------------------
This module provides a simple framework for backtesting trading strategies
on the Cardano historical data.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Ensure logs directory exists
os.makedirs('../logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='../logs/backtest.log'
)
logger = logging.getLogger("Backtest")

class SimpleBacktest:
    """
    A simple backtesting framework for testing trading strategies.
    """
    
    def __init__(self, symbol='ADA-USD', timeframe='1h', initial_capital=10000):
        """
        Initialize the backtesting framework.
        
        Parameters:
        -----------
        symbol : str
            The trading symbol (default: 'ADA-USD')
        timeframe : str
            The timeframe to use (default: '1h', options: '1m', '5m', '15m', '1h', '4h', '1d')
        initial_capital : float
            Initial capital in USD (default: 10000)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.data = None
        self.results = None
        self.metrics = {}
        
        # Load the data
        self._load_data()
        
    def _load_data(self):
        """Load historical data for the specified symbol and timeframe."""
        try:
            # Construct the filepath - look in the parent directory's data folder
            base_dir = Path('../data/cardano')
            
            if self.timeframe == '1m':
                filepath = base_dir / f'{self.symbol.replace("-", "")}-{self.timeframe}-data.csv'
            else:
                filepath = base_dir / f'{self.symbol.replace("-", "")}-{self.timeframe}.csv'
            
            if not filepath.exists():
                logger.error(f"Data file not found: {filepath}")
                raise FileNotFoundError(f"Data file not found: {filepath}")
            
            # Load the data
            self.data = pd.read_csv(filepath)
            
            # Ensure datetime is properly formatted
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
            
            # Set datetime as index
            self.data.set_index('datetime', inplace=True)
            
            logger.info(f"Loaded {len(self.data)} rows of {self.timeframe} data for {self.symbol}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def run_strategy(self, strategy_function, **strategy_params):
        """
        Run a trading strategy on the historical data.
        
        Parameters:
        -----------
        strategy_function : function
            A function that takes a dataframe and parameters and returns signals
        **strategy_params : dict
            Parameters to pass to the strategy function
        
        Returns:
        --------
        pd.DataFrame
            The results dataframe with positions and performance metrics
        """
        if self.data is None:
            logger.error("Data not loaded. Cannot run strategy.")
            return None
        
        try:
            # Create a copy of the data for the strategy
            df = self.data.copy()
            
            # Run the strategy to generate signals
            df = strategy_function(df, **strategy_params)
            
            # Ensure 'signal' column exists
            if 'signal' not in df.columns:
                logger.error("Strategy did not generate a 'signal' column")
                return None
            
            # Calculate positions (1 for long, -1 for short, 0 for no position)
            df['position'] = df['signal'].shift(1)  # Signal becomes position in the next period
            df['position'].fillna(0, inplace=True)
            
            # Calculate returns
            df['market_return'] = df['close'].pct_change()
            df['strategy_return'] = df['market_return'] * df['position']
            
            # Calculate cumulative returns
            df['market_cumulative_return'] = (1 + df['market_return']).cumprod() - 1
            df['strategy_cumulative_return'] = (1 + df['strategy_return']).cumprod() - 1
            
            # Calculate equity curve
            df['equity'] = self.initial_capital * (1 + df['strategy_cumulative_return'])
            
            # Store the results
            self.results = df
            
            # Calculate performance metrics
            self._calculate_metrics()
            
            logger.info(f"Strategy backtest completed with {len(df)} data points")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error running strategy: {str(e)}")
            raise
    
    def _calculate_metrics(self):
        """Calculate performance metrics for the strategy."""
        if self.results is None:
            logger.warning("No results to calculate metrics from")
            return
        
        df = self.results
        
        # Filter out rows with NaN returns
        df_valid = df.dropna(subset=['strategy_return'])
        
        try:
            # Calculate basic metrics
            total_return = df['strategy_cumulative_return'].iloc[-1]
            market_return = df['market_cumulative_return'].iloc[-1]
            
            # Calculate annualized return
            days = (df.index[-1] - df.index[0]).days
            if days > 0:
                annual_return = ((1 + total_return) ** (365 / days)) - 1
            else:
                annual_return = 0
            
            # Calculate risk metrics
            volatility = df_valid['strategy_return'].std() * (252 ** 0.5)  # Annualized
            market_volatility = df_valid['market_return'].std() * (252 ** 0.5)
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0)
            if volatility > 0:
                sharpe = annual_return / volatility
            else:
                sharpe = 0
            
            # Calculate drawdown
            df['drawdown'] = 1 - df['equity'] / df['equity'].cummax()
            max_drawdown = df['drawdown'].max()
            
            # Calculate win rate
            df['trade'] = df['position'].diff()
            trades = df[df['trade'] != 0].copy()
            
            if len(trades) > 0:
                trades['trade_return'] = 0.0
                current_position = 0
                entry_price = 0
                
                for idx, row in trades.iterrows():
                    if row['trade'] != 0:
                        if current_position == 0:  # Opening a position
                            current_position = row['trade']
                            entry_price = row['close']
                        else:  # Closing a position
                            if current_position > 0:  # Long position
                                trade_return = (row['close'] / entry_price) - 1
                            else:  # Short position
                                trade_return = 1 - (row['close'] / entry_price)
                            
                            trades.loc[idx, 'trade_return'] = trade_return
                            current_position = 0
                
                winning_trades = trades[trades['trade_return'] > 0]
                win_rate = len(winning_trades) / len(trades[~trades['trade_return'].isna()])
                avg_win = winning_trades['trade_return'].mean() if len(winning_trades) > 0 else 0
                
                losing_trades = trades[trades['trade_return'] < 0]
                avg_loss = losing_trades['trade_return'].mean() if len(losing_trades) > 0 else 0
                
                profit_factor = abs(sum(winning_trades['trade_return']) / sum(losing_trades['trade_return'])) \
                                if len(losing_trades) > 0 and sum(losing_trades['trade_return']) != 0 else float('inf')
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
            
            # Store the metrics
            self.metrics = {
                'total_return': total_return,
                'market_return': market_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'market_volatility': market_volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'num_trades': len(trades[~trades['trade_return'].isna()])
            }
            
            logger.info(f"Calculated performance metrics: Sharpe={sharpe:.2f}, Return={total_return:.2%}")
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            self.metrics = {}
    
    def plot_results(self, figsize=(14, 10), save_path=None):
        """
        Plot the backtest results.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height) in inches
        save_path : str
            Path to save the figure (optional)
        """
        if self.results is None:
            logger.warning("No results to plot")
            return
        
        try:
            # Create the figure
            fig, axes = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 1, 1]})
            
            # Plot price and positions
            self.results['close'].plot(ax=axes[0], label='Price', color='blue', alpha=0.5)
            
            # Add buy/sell markers
            buy_signals = self.results[self.results['signal'] > 0].index
            sell_signals = self.results[self.results['signal'] < 0].index
            
            if len(buy_signals) > 0:
                axes[0].scatter(buy_signals, self.results.loc[buy_signals, 'close'], 
                             marker='^', color='green', s=100, label='Buy')
            
            if len(sell_signals) > 0:
                axes[0].scatter(sell_signals, self.results.loc[sell_signals, 'close'], 
                             marker='v', color='red', s=100, label='Sell')
            
            axes[0].set_title(f'{self.symbol} - {self.timeframe} Backtest Results')
            axes[0].set_ylabel('Price')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot equity curve
            self.results['equity'].plot(ax=axes[1], label='Strategy Equity', color='green')
            initial_capital_line = pd.Series(self.initial_capital, index=self.results.index)
            initial_capital_line.plot(ax=axes[1], label='Initial Capital', linestyle='--', color='gray')
            axes[1].set_ylabel('Equity (USD)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Plot drawdown
            self.results['drawdown'].plot(ax=axes[2], label='Drawdown', color='red', alpha=0.5)
            axes[2].set_ylabel('Drawdown')
            axes[2].set_ylim(0, max(0.01, self.results['drawdown'].max() * 1.1))  # Ensure y-axis is visible
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            # Add performance metrics as text
            if self.metrics:
                metrics_text = (
                    f"Total Return: {self.metrics['total_return']:.2%}\n"
                    f"Market Return: {self.metrics['market_return']:.2%}\n"
                    f"Annual Return: {self.metrics['annual_return']:.2%}\n"
                    f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}\n"
                    f"Max Drawdown: {self.metrics['max_drawdown']:.2%}\n"
                    f"Win Rate: {self.metrics['win_rate']:.2%}\n"
                    f"Trades: {self.metrics['num_trades']}"
                )
                
                # Place text box in the upper left corner of the price chart
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                axes[0].text(0.02, 0.98, metrics_text, transform=axes[0].transAxes, fontsize=10,
                         verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            
            # Save the figure if a path is provided
            if save_path:
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"Saved backtest plot to {save_path}")
            
            return fig, axes
            
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            raise
    
    def print_metrics(self):
        """Print the performance metrics in a formatted way."""
        if not self.metrics:
            logger.warning("No metrics to print")
            print("No metrics available. Run a strategy first.")
            return
        
        print("\n" + "="*50)
        print(f" BACKTEST RESULTS: {self.symbol} - {self.timeframe}")
        print("="*50)
        
        print(f"\nPERIOD: {self.results.index[0].date()} to {self.results.index[-1].date()}")
        print(f"INITIAL CAPITAL: ${self.initial_capital:,.2f}")
        print(f"FINAL CAPITAL: ${self.results['equity'].iloc[-1]:,.2f}")
        
        print("\nPERFORMANCE METRICS:")
        print(f"Total Return: {self.metrics['total_return']:.2%}")
        print(f"Market Return: {self.metrics['market_return']:.2%}")
        print(f"Annual Return: {self.metrics['annual_return']:.2%}")
        print(f"Volatility: {self.metrics['volatility']:.2%}")
        print(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {self.metrics['max_drawdown']:.2%}")
        
        print("\nTRADE STATISTICS:")
        print(f"Number of Trades: {self.metrics['num_trades']}")
        print(f"Win Rate: {self.metrics['win_rate']:.2%}")
        print(f"Average Win: {self.metrics['avg_win']:.2%}")
        print(f"Average Loss: {self.metrics['avg_loss']:.2%}")
        print(f"Profit Factor: {self.metrics['profit_factor']:.2f}")
        
        print("\nCOMPARISON:")
        if self.metrics['total_return'] > self.metrics['market_return']:
            outperf = (1 + self.metrics['total_return']) / (1 + self.metrics['market_return']) - 1
            print(f"Strategy OUTPERFORMED market by {outperf:.2%}")
        else:
            underperf = (1 + self.metrics['market_return']) / (1 + self.metrics['total_return']) - 1
            print(f"Strategy UNDERPERFORMED market by {underperf:.2%}")
        
        print("="*50 + "\n")

# Example strategy functions
def simple_ma_crossover(df, short_period=20, long_period=50):
    """
    Simple moving average crossover strategy.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The price data
    short_period : int
        The short moving average period
    long_period : int
        The long moving average period
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with signals added
    """
    # Calculate moving averages
    df['ma_short'] = df['close'].rolling(window=short_period).mean()
    df['ma_long'] = df['close'].rolling(window=long_period).mean()
    
    # Generate signals
    df['signal'] = 0  # Default is no position
    df.loc[df['ma_short'] > df['ma_long'], 'signal'] = 1  # Long signal
    df.loc[df['ma_short'] < df['ma_long'], 'signal'] = -1  # Short signal
    
    return df

def rsi_strategy(df, rsi_period=14, overbought=70, oversold=30):
    """
    RSI-based strategy, buy when oversold and sell when overbought.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The price data
    rsi_period : int
        The RSI calculation period
    overbought : float
        The overbought threshold (0-100)
    oversold : float
        The oversold threshold (0-100)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with signals added
    """
    # Calculate RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Generate signals
    df['signal'] = 0  # Default is no position
    df.loc[df['rsi'] < oversold, 'signal'] = 1  # Buy when oversold
    df.loc[df['rsi'] > overbought, 'signal'] = -1  # Sell when overbought
    
    return df

def bollinger_band_strategy(df, period=20, std_dev=2):
    """
    Bollinger Band strategy, buy at lower band and sell at upper band.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The price data
    period : int
        The period for calculating moving average and standard deviation
    std_dev : float
        Number of standard deviations for the bands
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with signals added
    """
    # Calculate Bollinger Bands
    df['middle_band'] = df['close'].rolling(window=period).mean()
    df['std'] = df['close'].rolling(window=period).std()
    df['upper_band'] = df['middle_band'] + (df['std'] * std_dev)
    df['lower_band'] = df['middle_band'] - (df['std'] * std_dev)
    
    # Generate signals
    df['signal'] = 0  # Default is no position
    df.loc[df['close'] < df['lower_band'], 'signal'] = 1  # Buy at lower band
    df.loc[df['close'] > df['upper_band'], 'signal'] = -1  # Sell at upper band
    
    return df

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run a backtest on Cardano data.')
    parser.add_argument('--strategy', type=str, default='ma', choices=['ma', 'rsi', 'bb'],
                        help='Strategy to use (default: ma)')
    parser.add_argument('--timeframe', type=str, default='1h', 
                        choices=['1m', '5m', '15m', '1h', '4h', '1d'],
                        help='Timeframe to use (default: 1h)')
    parser.add_argument('--capital', type=float, default=10000,
                        help='Initial capital (default: 10000)')
    
    args = parser.parse_args()
    
    # Create backtest instance
    backtest = SimpleBacktest(symbol='ADA-USD', timeframe=args.timeframe, 
                             initial_capital=args.capital)
    
    # Run the specified strategy
    if args.strategy == 'ma':
        backtest.run_strategy(simple_ma_crossover, short_period=20, long_period=50)
    elif args.strategy == 'rsi':
        backtest.run_strategy(rsi_strategy, rsi_period=14, overbought=70, oversold=30)
    elif args.strategy == 'bb':
        backtest.run_strategy(bollinger_band_strategy, period=20, std_dev=2)
    
    # Print results
    backtest.print_metrics()
    
    # Plot results
    os.makedirs('plots', exist_ok=True)
    backtest.plot_results(save_path=f'plots/{args.strategy}_{args.timeframe}.png')
    
    print(f"Backtest plot saved to plots/{args.strategy}_{args.timeframe}.png")
