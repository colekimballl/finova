'''
Project Solaris AI - Trading Strategy Base
-----------------------------------------
Base class for implementing trading strategies
'''

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from pathlib import Path
import os
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/strategy.log'
)

class StrategyBase(ABC):
    """
    Abstract base class for all trading strategies
    """
    def __init__(self, name, config=None):
        """
        Initialize the strategy
        
        Parameters:
        -----------
        name: str
            Name of the strategy
        config: dict
            Configuration parameters for the strategy
        """
        self.name = name
        self.logger = logging.getLogger(f"Strategy-{name}")
        
        # Initialize default configuration
        self.config = {
            'timeframe': '1h',
            'profit_target': 0.05,
            'stop_loss': 0.03,
            'max_risk_per_trade': 0.02,  # 2% of account
            'position_size': 10,  # In USD
            'trailing_stop': False,
            'trailing_stop_distance': 0.02
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
            
        self.logger.info(f"Initialized strategy: {name}")
        self.logger.info(f"Configuration: {self.config}")
        
        # State tracking
        self.position = None  # 'long', 'short', or None
        self.entry_price = None
        self.entry_time = None
        self.position_size = None
        self.highest_price = None  # For trailing stop
        self.lowest_price = None   # For trailing stop
        
    @abstractmethod
    def generate_signals(self, df):
        """
        Generate trading signals from the data
        
        Parameters:
        -----------
        df: pandas.DataFrame
            DataFrame with price and indicator data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added signal columns
        """
        pass
        
    def should_enter_long(self, row):
        """
        Check if we should enter a long position
        
        Parameters:
        -----------
        row: pandas.Series
            Current data row
            
        Returns:
        --------
        bool
            True if we should enter a long position
        """
        return False  # Override in subclass
        
    def should_exit_long(self, row, entry_price):
        """
        Check if we should exit a long position
        
        Parameters:
        -----------
        row: pandas.Series
            Current data row
        entry_price: float
            Entry price of the position
            
        Returns:
        --------
        bool
            True if we should exit the position
        """
        # Default implementation with profit target and stop loss
        if row['close'] >= entry_price * (1 + self.config['profit_target']):
            return True, "profit_target"
            
        if row['close'] <= entry_price * (1 - self.config['stop_loss']):
            return True, "stop_loss"
            
        return False, None
        
    def should_enter_short(self, row):
        """
        Check if we should enter a short position
        
        Parameters:
        -----------
        row: pandas.Series
            Current data row
            
        Returns:
        --------
        bool
            True if we should enter a short position
        """
        return False  # Override in subclass
        
    def should_exit_short(self, row, entry_price):
        """
        Check if we should exit a short position
        
        Parameters:
        -----------
        row: pandas.Series
            Current data row
        entry_price: float
            Entry price of the position
            
        Returns:
        --------
        bool
            True if we should exit the position
        """
        # Default implementation with profit target and stop loss
        if row['close'] <= entry_price * (1 - self.config['profit_target']):
            return True, "profit_target"
            
        if row['close'] >= entry_price * (1 + self.config['stop_loss']):
            return True, "stop_loss"
            
        return False, None
        
    def calculate_position_size(self, price, risk_per_trade=None):
        """
        Calculate position size based on risk parameters
        
        Parameters:
        -----------
        price: float
            Current price
        risk_per_trade: float
            Risk per trade as a percentage of account (overrides config)
            
        Returns:
        --------
        float
            Position size in USD
        """
        risk = risk_per_trade if risk_per_trade else self.config['max_risk_per_trade']
        
        # Fixed position size from config
        position_size = self.config['position_size']
        
        # Log the calculation
        self.logger.info(f"Calculated position size: {position_size} USD")
        
        return position_size
        
    def backtest(self, df):
        """
        Backtest the strategy on historical data
        
        Parameters:
        -----------
        df: pandas.DataFrame
            DataFrame with price and indicator data
            
        Returns:
        --------
        tuple
            (results_df, performance_metrics)
        """
        # Generate signals
        signals_df = self.generate_signals(df)
        
        # Initialize results
        results = signals_df.copy()
        results['position'] = None
        results['entry_price'] = None
        results['exit_price'] = None
        results['pnl'] = 0.0
        results['equity'] = 100.0  # Start with $100
        
        # State tracking
        position = None
        entry_price = None
        entry_idx = None
        position_size = 0
        trades = []
        
        # Run backtest
        for i, row in signals_df.iterrows():
            # Skip if NaN values in key columns
            if pd.isna(row['close']):
                continue
                
            # Check for exit if in position
            if position == 'long':
                # Update trailing stops if enabled
                if self.config.get('trailing_stop', False) and self.highest_price:
                    self.highest_price = max(self.highest_price, row['close'])
                    trailing_stop = self.highest_price * (1 - self.config['trailing_stop_distance'])
                    if row['close'] <= trailing_stop:
                        # Exit on trailing stop
                        exit_price = row['close']
                        pnl = (exit_price - entry_price) / entry_price
                        trades.append({
                            'entry_time': signals_df.index[entry_idx],
                            'exit_time': i,
                            'position': position,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'pnl_pct': pnl * 100,
                            'reason': 'trailing_stop'
                        })
                        
                        results.loc[i, 'exit_price'] = exit_price
                        results.loc[i, 'pnl'] = pnl * position_size
                        
                        position = None
                        entry_price = None
                        entry_idx = None
                        continue
                
                # Regular exit check
                should_exit, reason = self.should_exit_long(row, entry_price)
                if should_exit:
                    exit_price = row['close']
                    pnl = (exit_price - entry_price) / entry_price
                    trades.append({
                        'entry_time': signals_df.index[entry_idx],
                        'exit_time': i,
                        'position': position,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_pct': pnl * 100,
                        'reason': reason
                    })
                    
                    results.loc[i, 'exit_price'] = exit_price
                    results.loc[i, 'pnl'] = pnl * position_size
                    
                    position = None
                    entry_price = None
                    entry_idx = None
                    continue
            
            elif position == 'short':
                # Update trailing stops if enabled
                if self.config.get('trailing_stop', False) and self.lowest_price:
                    self.lowest_price = min(self.lowest_price, row['close'])
                    trailing_stop = self.lowest_price * (1 + self.config['trailing_stop_distance'])
                    if row['close'] >= trailing_stop:
                        # Exit on trailing stop
                        exit_price = row['close']
                        pnl = (entry_price - exit_price) / entry_price
                        trades.append({
                            'entry_time': signals_df.index[entry_idx],
                            'exit_time': i,
                            'position': position,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'pnl_pct': pnl * 100,
                            'reason': 'trailing_stop'
                        })
