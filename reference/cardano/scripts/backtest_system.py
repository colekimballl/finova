#!/usr/bin/env python
'''
Cardano Trading System
---------------------
Backtesting framework for Cardano trading strategies with risk management
and performance analytics.
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
from pathlib import Path

class RiskManager:
    """
    Risk management system to control position sizing and enforce drawdown limits.
    """
    def __init__(self, initial_capital=10000, max_drawdown_pct=10, risk_per_trade_pct=1):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_capital = initial_capital
        self.peak_capital = initial_capital
        self.max_drawdown_pct = max_drawdown_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        self.drawdown_history = []
        
    def calculate_position_size(self, entry_price, stop_loss_price):
        """
        Calculate the position size based on risk parameters.
        
        Parameters:
        -----------
        entry_price : float
            Entry price for the trade
        stop_loss_price : float
            Stop loss price for the trade
            
        Returns:
        --------
        float : Number of units to trade
        """
        # Risk amount in dollars
        risk_amount = self.current_capital * (self.risk_per_trade_pct / 100)
        
        # Risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit == 0:  # Avoid division by zero
            return 0
        
        # Units to trade
        units = risk_amount / risk_per_unit
        
        # Check if we've hit max drawdown
        drawdown_pct = ((self.peak_capital - self.current_capital) / self.peak_capital) * 100
        self.drawdown_history.append(drawdown_pct)
        
        if drawdown_pct >= self.max_drawdown_pct:
            # Stop trading when max drawdown reached
            return 0
            
        return units
    
    def update_capital(self, profit_loss):
        """
        Update the capital after a trade.
        
        Parameters:
        -----------
        profit_loss : float
            Profit or loss from the trade
        """
        self.current_capital += profit_loss
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
    
    def get_current_drawdown(self):
        """
        Get the current drawdown percentage.
        
        Returns:
        --------
        float : Current drawdown percentage
        """
        return ((self.peak_capital - self.current_capital) / self.peak_capital) * 100
    
    def get_max_drawdown(self):
        """
        Get the maximum drawdown percentage experienced.
        
        Returns:
        --------
        float : Maximum drawdown percentage
        """
        if not self.drawdown_history:
            return 0
        return max(self.drawdown_history)


class Trade:
    """
    Represents a single trade with entry, exit, and performance details.
    """
    def __init__(self, entry_time, entry_price, position_size, direction, stop_loss, take_profit=None):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.position_size = position_size
        self.direction = direction  # 'long' or 'short'
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None
        self.profit_loss = None
        self.profit_loss_pct = None
        self.status = "open"
        
    def close_trade(self, exit_time, exit_price, exit_reason):
        """
        Close the trade and calculate profit/loss.
        
        Parameters:
        -----------
        exit_time : datetime
            Time of trade exit
        exit_price : float
            Exit price
        exit_reason : str
            Reason for exit (e.g., 'stop_loss', 'take_profit', 'signal')
        """
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        
        # Calculate profit/loss
        if self.direction == 'long':
            self.profit_loss = (exit_price - self.entry_price) * self.position_size
            self.profit_loss_pct = ((exit_price / self.entry_price) - 1) * 100
        else:  # short
            self.profit_loss = (self.entry_price - exit_price) * self.position_size
            self.profit_loss_pct = ((self.entry_price / exit_price) - 1) * 100
        
        self.status = "closed"
        
    def is_stop_loss_hit(self, current_price):
        """
        Check if the stop loss has been hit.
        
        Parameters:
        -----------
        current_price : float
            Current price to check against stop loss
            
        Returns:
        --------
        bool : True if stop loss is hit, False otherwise
        """
        if self.direction == 'long':
            return current_price <= self.stop_loss
        else:  # short
            return current_price >= self.stop_loss
    
    def is_take_profit_hit(self, current_price):
        """
        Check if the take profit has been hit.
        
        Parameters:
        -----------
        current_price : float
            Current price to check against take profit
            
        Returns:
        --------
        bool : True if take profit is hit, False otherwise
        """
        if self.take_profit is None:
            return False
            
        if self.direction == 'long':
            return current_price >= self.take_profit
        else:  # short
            return current_price <= self.take_profit
    
    def get_current_profit_loss(self, current_price):
        """
        Calculate current profit/loss based on current market price.
        
        Parameters:
        -----------
        current_price : float
            Current market price
            
        Returns:
        --------
        float : Current profit/loss
        """
        if self.direction == 'long':
            return (current_price - self.entry_price) * self.position_size
        else:  # short
            return (self.entry_price - current_price) * self.position_size


class TradingStrategy:
    """
    Base trading strategy class that implements rules for entries and exits.
    """
    def __init__(self, params=None):
        self.params = params or self.get_default_params()
    
    def get_default_params(self):
        """
        Get default strategy parameters.
        
        Returns:
        --------
        dict : Default parameters
        """
        return {
            'rsi_oversold_threshold': 30,
            'rsi_overbought_threshold': 70,
            'stop_atr_multiplier': 2.0,
            'take_profit_atr_multiplier': 3.0,
            'max_bb_width': 0.2,
            'min_rel_volume': 1.2,
            'trend_filter_enabled': True
        }
    
    def generate_signals(self, data, index):
        """
        Generate trading signals based on current data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing price and indicator data
        index : int
            Current index in the DataFrame
            
        Returns:
        --------
        tuple : (buy_signal, sell_signal)
        """
        # Ensure we have enough data for lookback
        if index < 20:
            return False, False
        
        # Extract current row
        row = data.iloc[index]
        
        # Trend filter (only trade in the direction of the larger trend)
        trend_bullish = True
        if self.params['trend_filter_enabled'] and 'sma_50' in data.columns and 'sma_20' in data.columns:
            trend_bullish = row['sma_20'] > row['sma_50']
        
        # Momentum setup (RSI conditions)
        rsi_oversold = False
        rsi_overbought = False
        if 'rsi_14' in data.columns:
            rsi_oversold = row['rsi_14'] < self.params['rsi_oversold_threshold']
            rsi_overbought = row['rsi_14'] > self.params['rsi_overbought_threshold']
        
        # Volatility filter (reduce trading during extreme volatility)
        normal_volatility = True
        if 'bb_width' in data.columns:
            normal_volatility = row['bb_width'] < self.params['max_bb_width']
        
        # Volume confirmation
        high_volume = True
        if 'relative_volume' in data.columns:
            high_volume = row['relative_volume'] > self.params['min_rel_volume']
        
        # Signal generation
        buy_signal = trend_bullish and rsi_oversold and normal_volatility and high_volume
        sell_signal = (not trend_bullish) and rsi_overbought and high_volume
        
        return buy_signal, sell_signal
    
    def calculate_stop_loss(self, data, index, direction):
        """
        Calculate stop loss based on ATR.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing price and indicator data
        index : int
            Current index in the DataFrame
        direction : str
            Trade direction ('long' or 'short')
            
        Returns:
        --------
        float : Stop loss price
        """
        row = data.iloc[index]
        price = row['close']
        
        # If ATR is available, use it for dynamic stop loss
        if 'atr_14' in data.columns:
            atr = row['atr_14']
            if direction == 'long':
                return price - (self.params['stop_atr_multiplier'] * atr)
            else:  # short
                return price + (self.params['stop_atr_multiplier'] * atr)
        else:
            # Fallback to percentage-based stop loss
            if direction == 'long':
                return price * 0.95  # 5% stop loss
            else:  # short
                return price * 1.05  # 5% stop loss
    
    def calculate_take_profit(self, data, index, direction):
        """
        Calculate take profit level based on ATR.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing price and indicator data
        index : int
            Current index in the DataFrame
        direction : str
            Trade direction ('long' or 'short')
            
        Returns:
        --------
        float : Take profit price
        """
        row = data.iloc[index]
        price = row['close']
        
        # If ATR is available, use it for dynamic take profit
        if 'atr_14' in data.columns:
            atr = row['atr_14']
            if direction == 'long':
                return price + (self.params['take_profit_atr_multiplier'] * atr)
            else:  # short
                return price - (self.params['take_profit_atr_multiplier'] * atr)
        else:
            # Fallback to percentage-based take profit
            if direction == 'long':
                return price * 1.1  # 10% take profit
            else:  # short
                return price * 0.9  # 10% take profit


class ImprovedTradingStrategy(TradingStrategy):
    """
    Improved trading strategy with better indicator combinations.
    """
    def get_default_params(self):
        """
        Get default strategy parameters.
        
        Returns:
        --------
        dict : Default parameters
        """
        return {
            'rsi_oversold_threshold': 25,         # More conservative RSI thresholds
            'rsi_overbought_threshold': 75,
            'stop_atr_multiplier': 2.5,           # Wider stops to avoid noise
            'take_profit_atr_multiplier': 4.0,    # Higher reward potential
            'max_bb_width': 0.15,                 # Stricter volatility filter
            'min_rel_volume': 1.5,                # Higher volume requirement
            'trend_filter_enabled': True,
            'macd_filter_enabled': True,          # Add MACD filter
            'support_resistance_filter': True     # Add S/R filter
        }
    
    def generate_signals(self, data, index):
        """
        Generate trading signals based on current data with improved logic.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing price and indicator data
        index : int
            Current index in the DataFrame
            
        Returns:
        --------
        tuple : (buy_signal, sell_signal)
        """
        # Ensure we have enough data for lookback
        if index < 20:
            return False, False
        
        # Extract current row
        row = data.iloc[index]
        
        # Trend filter (only trade in the direction of the larger trend)
        trend_bullish = True
        if self.params['trend_filter_enabled'] and 'sma_50' in data.columns and 'sma_20' in data.columns:
            trend_bullish = row['sma_20'] > row['sma_50']
        
        # Momentum setup (RSI conditions)
        rsi_signal = 'neutral'
        if 'rsi_14' in data.columns:
            if row['rsi_14'] < self.params['rsi_oversold_threshold']:
                rsi_signal = 'bullish'
            elif row['rsi_14'] > self.params['rsi_overbought_threshold']:
                rsi_signal = 'bearish'
        
        # MACD Filter
        macd_bullish = True
        if self.params['macd_filter_enabled'] and 'macd' in data.columns and 'macd_signal' in data.columns:
            # Bullish MACD crossover (MACD line crosses above signal line)
            if index > 0:
                prev_row = data.iloc[index-1]
                macd_bullish = (row['macd'] > row['macd_signal']) and (prev_row['macd'] <= prev_row['macd_signal'])
        
        # Volatility filter (reduce trading during extreme volatility)
        normal_volatility = True
        if 'bb_width' in data.columns:
            normal_volatility = row['bb_width'] < self.params['max_bb_width']
        
        # Volume confirmation
        high_volume = True
        if 'relative_volume' in data.columns:
            high_volume = row['relative_volume'] > self.params['min_rel_volume']
        
        # Support/Resistance filter using Bollinger Bands
        sr_filter_pass = True
        if self.params['support_resistance_filter'] and all(x in data.columns for x in ['bb_upper', 'bb_lower', 'close']):
            # Buy near lower band, sell near upper band
            near_support = row['close'] < (row['bb_lower'] * 1.02)  # Within 2% of lower band
            near_resistance = row['close'] > (row['bb_upper'] * 0.98)  # Within 2% of upper band
            
            # Only buy if near support, only sell if near resistance
            if rsi_signal == 'bullish' and not near_support:
                sr_filter_pass = False
            if rsi_signal == 'bearish' and not near_resistance:
                sr_filter_pass = False
        
        # Signal generation with enhanced logic
        buy_signal = (rsi_signal == 'bullish' and 
                     trend_bullish and 
                     normal_volatility and 
                     high_volume and
                     sr_filter_pass and
                     macd_bullish)
        
        sell_signal = (rsi_signal == 'bearish' and 
                      not trend_bullish and 
                      high_volume and
                      sr_filter_pass)
        
        return buy_signal, sell_signal


class BacktestEngine:
    """
    Engine to run backtests of trading strategies.
    """
    def __init__(self, data, strategy=None, risk_manager=None, trading_fee_pct=0.1):
        self.data = data.copy()
        self.strategy = strategy or TradingStrategy()
        self.risk_manager = risk_manager or RiskManager()
        self.trading_fee_pct = trading_fee_pct / 100  # Convert to decimal
        
        # Performance tracking
        self.trades = []
        self.open_trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        
    def run(self):
        """
        Run the backtest through the entire dataset.
        """
        # Ensure datetime is the index
        if 'datetime' in self.data.columns:
            self.data.set_index('datetime', inplace=True)
        
        # Reset performance tracking
        self.trades = []
        self.open_trades = []
        self.equity_curve = [self.risk_manager.initial_capital]
        self.drawdown_curve = [0]
        
        # Iterate through each bar (starting after lookback period)
        for i in range(20, len(self.data)):
            # Get current row
            current_bar = self.data.iloc[i]
            current_time = self.data.index[i]
            current_price = current_bar['close']
            
            # Update open positions first
            self._manage_open_trades(i, current_time, current_price)
            
            # Check for new signals
            buy_signal, sell_signal = self.strategy.generate_signals(self.data, i)
            
            # Open new positions if signaled and no open position
            if buy_signal and not self._has_open_position():
                self._open_trade(i, current_time, current_price, 'long')
            elif sell_signal and not self._has_open_position():
                self._open_trade(i, current_time, current_price, 'short')
                
            # Update equity curve
            self.equity_curve.append(self.risk_manager.current_capital)
            self.drawdown_curve.append(self.risk_manager.get_current_drawdown())
    
    def _open_trade(self, index, time, price, direction):
        """
        Open a new trade.
        
        Parameters:
        -----------
        index : int
            Current index in the DataFrame
        time : datetime
            Current time
        price : float
            Current price
        direction : str
            Trade direction ('long' or 'short')
        """
        # Calculate stop loss
        stop_loss = self.strategy.calculate_stop_loss(self.data, index, direction)
        
        # Calculate take profit
        take_profit = self.strategy.calculate_take_profit(self.data, index, direction)
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(price, stop_loss)
        
        # If position size is 0, don't open trade (risk limit hit)
        if position_size <= 0:
            return
        
        # Create trade object
        trade = Trade(
            entry_time=time,
            entry_price=price,
            position_size=position_size,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Add to open trades
        self.open_trades.append(trade)
        
        # Account for fees
        fee = price * position_size * self.trading_fee_pct
        self.risk_manager.update_capital(-fee)
    
    def _manage_open_trades(self, index, time, price):
        """
        Manage open trades - check for stop loss, take profit, or exit signals.
        
        Parameters:
        -----------
        index : int
            Current index in the DataFrame
        time : datetime
            Current time
        price : float
            Current price
        """
        # Check each open trade
        remaining_trades = []
        for trade in self.open_trades:
            # Check if stop loss is hit
            if trade.is_stop_loss_hit(price):
                self._close_trade(trade, time, trade.stop_loss, 'stop_loss')
            
            # Check if take profit is hit
            elif trade.is_take_profit_hit(price):
                self._close_trade(trade, time, trade.take_profit, 'take_profit')
            
            # Check for exit signal
            else:
                # Logic for exit signals goes here
                # For now, keep the trade open
                remaining_trades.append(trade)
        
        # Update open trades list
        self.open_trades = remaining_trades
    
    def _close_trade(self, trade, time, price, reason):
        """
        Close a trade and update capital.
        
        Parameters:
        -----------
        trade : Trade
            Trade object to close
        time : datetime
            Exit time
        price : float
            Exit price
        reason : str
            Reason for exit
        """
        # Close the trade
        trade.close_trade(time, price, reason)
        
        # Calculate fees
        fee = price * trade.position_size * self.trading_fee_pct
        
        # Update capital (profit/loss minus fees)
        self.risk_manager.update_capital(trade.profit_loss - fee)
        
        # Add to completed trades list
        self.trades.append(trade)
    
    def _has_open_position(self):
        """
        Check if there is an open position.
        
        Returns:
        --------
        bool : True if there is an open position, False otherwise
        """
        return len(self.open_trades) > 0
    
    def get_performance_stats(self):
        """
        Calculate performance statistics.
        
        Returns:
        --------
        dict : Performance statistics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return_pct': 0,
                'max_drawdown_pct': 0,
                'sharpe_ratio': 0
            }
        
        # Basic stats
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade.profit_loss > 0)
        
        # Win rate
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(trade.profit_loss for trade in self.trades if trade.profit_loss > 0)
        gross_loss = abs(sum(trade.profit_loss for trade in self.trades if trade.profit_loss < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Total return
        total_return = self.risk_manager.current_capital - self.risk_manager.initial_capital
        total_return_pct = (total_return / self.risk_manager.initial_capital) * 100
        
        # Max drawdown
        max_drawdown_pct = self.risk_manager.get_max_drawdown()
        
        # Average trade
        avg_trade = sum(trade.profit_loss for trade in self.trades) / total_trades if total_trades > 0 else 0
        
        # Sharpe ratio (simplified, assuming risk-free rate = 0)
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'max_drawdown_pct': max_drawdown_pct,
            'avg_trade': avg_trade,
            'sharpe_ratio': sharpe_ratio
        }
    
    def plot_results(self, save_path=None):
        """
        Plot equity curve, drawdown, and trade markers.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot, if None, the plot is displayed
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Create index for the equity curve
        if len(self.data) > 19:
            # Use actual data index from the processed data
            date_index = self.data.index[19:19+len(self.equity_curve)]
        else:
            # Fallback in case of insufficient data
            date_index = pd.date_range(start=self.data.index[0], periods=len(self.equity_curve), freq='H')
        
        # Convert equity curve to DataFrame with datetime index
        equity_df = pd.DataFrame({
            'equity': self.equity_curve,
            'drawdown': self.drawdown_curve
        }, index=date_index)
        
        # Plot equity curve
        ax1.plot(equity_df.index, equity_df['equity'], label='Equity Curve')
        
        # Plot trades
        for trade in self.trades:
            if trade.profit_loss > 0:
                marker = '^'  # Up triangle for winning trades
                color = 'green'
            else:
                marker = 'v'  # Down triangle for losing trades
                color = 'red'
            
            # Plot entry and exit
            ax1.scatter(trade.entry_time, self.risk_manager.initial_capital, marker=marker, 
                        s=50, color=color, alpha=0.7)
            
            # Only plot exit for closed trades
            if trade.exit_time:
                ax1.scatter(trade.exit_time, self.risk_manager.initial_capital, marker='o', 
                            s=30, color=color, alpha=0.5)
        
        # Plot drawdown
        ax2.fill_between(equity_df.index, 0, equity_df['drawdown'], color='red', alpha=0.3, label='Drawdown')
        ax2.set_ylim(max(equity_df['drawdown']) * 1.5, 0)  # Invert y-axis for drawdown
        
        # Set labels and title
        ax1.set_title('Backtest Results')
        ax1.set_ylabel('Equity ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # Format dates on x-axis
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


def load_processed_data(timeframe, data_dir='../data/processed'):
    """
    Load processed data for a specific timeframe.
    
    Parameters:
    -----------
    timeframe : str
        Timeframe to load ('1h', '4h', '1d')
    data_dir : str
        Directory containing the processed data files
    
    Returns:
    --------
    pd.DataFrame : Loaded data or None if file not found
    """
    try:
        filepath = os.path.join(data_dir, f'ADAUSD-{timeframe}-processed.csv')
        if not os.path.exists(filepath):
            print(f"Error: Processed data file for {timeframe} timeframe not found at {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"Loaded {len(df)} rows of processed {timeframe} data")
        return df
    except Exception as e:
        print(f"Error loading processed {timeframe} data: {str(e)}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cardano Trading System Backtester')
    parser.add_argument('--timeframe', type=str, default='1h', 
                      choices=['1h', '4h', '1d'],
                      help='Timeframe to use (default: 1h)')
    parser.add_argument('--capital', type=float, default=10000,
                      help='Initial capital (default: $10,000)')
    parser.add_argument('--max-drawdown', type=float, default=10,
                      help='Maximum allowed drawdown percentage (default: 10%%)')
    parser.add_argument('--risk-per-trade', type=float, default=1,
                      help='Risk per trade as percentage of capital (default: 1%%)')
    parser.add_argument('--trading-fee', type=float, default=0.1,
                      help='Trading fee percentage (default: 0.1%%)')
    parser.add_argument('--improved', action='store_true',
                      help='Use improved strategy')
    parser.add_argument('--output', type=str, default='../backtest_results',
                      help='Output directory for backtest results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    data = load_processed_data(args.timeframe)
    if data is None:
        print("Failed to load data. Exiting.")
        exit(1)
    
    # Create strategy
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
        initial_capital=args.capital,
        max_drawdown_pct=args.max_drawdown,
        risk_per_trade_pct=args.risk_per_trade
    )
    
    # Create and run backtest
    backtest = BacktestEngine(
        data=data,
        strategy=strategy,
        risk_manager=risk_manager,
        trading_fee_pct=args.trading_fee
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
