#!/usr/bin/env python
'''
Extreme Crypto Trading Strategy (Moon Shot Version) - Part 1
-----------------------------------------------------------
Ultra-aggressive trading strategy designed for maximum returns
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtest_system import RiskManager, load_processed_data
import argparse

class ExtremeRiskManager(RiskManager):
    """
    Ultra-aggressive risk management with extreme leverage and position sizing.
    """
    def __init__(self, initial_capital=10000, max_drawdown_pct=50, risk_per_trade_pct=10, max_leverage=20):
        super().__init__(initial_capital, max_drawdown_pct, risk_per_trade_pct)
        self.max_leverage = max_leverage
        self.current_leverage = 5.0  # Start with 5x leverage
        self.profitable_trades_streak = 0
        self.losing_trades_streak = 0
        self.equity_history = []
        
    def calculate_position_size(self, entry_price, stop_loss_price, market_regime='neutral'):
        """
        Calculate position size with extreme leverage and dynamic sizing.
        """
        # Base position size calculation
        price_delta = abs(entry_price - stop_loss_price)
        if price_delta == 0:  # Avoid division by zero
            return 0
            
        # Increase risk on winning streaks
        effective_risk_pct = self.risk_per_trade_pct
        if self.profitable_trades_streak >= 2:
            streak_multiplier = min(3.0, 1.0 + (self.profitable_trades_streak * 0.2))
            effective_risk_pct *= streak_multiplier
        
        # Adjust risk based on market regime
        if market_regime == 'bull':
            effective_risk_pct *= 2.0  # Double risk in bull markets
        elif market_regime == 'bear':
            effective_risk_pct *= 0.8  # Reduce risk in bear markets
        
        # Cap at 25% risk per trade absolute maximum
        effective_risk_pct = min(25.0, effective_risk_pct)
        
        # Calculate dollar risk amount
        risk_amount = self.current_capital * (effective_risk_pct / 100)
        
        # Calculate base position size
        base_position_size = risk_amount / price_delta
        
        # Apply dynamic leverage based on conditions
        if self.profitable_trades_streak >= 3:
            # Increase leverage on winning streak
            self.current_leverage = min(self.max_leverage, self.current_leverage + 1.0)
        elif self.losing_trades_streak >= 2:
            # Decrease leverage on losing streak
            self.current_leverage = max(1.0, self.current_leverage - 1.0)
        
        # Apply leverage to position size
        leveraged_size = base_position_size * self.current_leverage
        
        # Check if we've hit max drawdown - if so, reduce position to 0
        drawdown_pct = self.get_current_drawdown()
        if drawdown_pct >= self.max_drawdown_pct:
            print(f"WARNING: Max drawdown reached ({drawdown_pct:.2f}%). Trading halted.")
            return 0
        
        return leveraged_size
    
    def update_capital(self, profit_loss):
        """
        Update capital, track streak, and manage equity tracking.
        """
        # Update winning/losing streaks
        if profit_loss > 0:
            self.profitable_trades_streak += 1
            self.losing_trades_streak = 0
        elif profit_loss < 0:
            self.losing_trades_streak += 1
            self.profitable_trades_streak = 0
        
        # Update capital using parent method
        super().update_capital(profit_loss)
        
        # Track equity history
        self.equity_history.append(self.current_capital)


class ExtremeStrategy:
    """
    Ultra-aggressive trading strategy designed for exceptional returns.
    """
    def __init__(self):
        # Define ultra-aggressive parameters
        self.rsi_oversold = 35        # Less strict for more entries
        self.rsi_overbought = 65      # Less strict for more entries
        self.stop_atr_multiplier = 1.0  # Tight stops
        self.take_profit_atr_multiplier = 10.0  # Very ambitious targets
        self.use_trailing_stop = True
        self.trail_activation_pct = 1.0  # Start trailing after 1% profit
        
    def identify_market_regime(self, data, index):
        """
        Identify market regime (bull, bear, or neutral)
        """
        if index < 50:
            return 'neutral'
            
        row = data.iloc[index]
        
        # Basic regime detection using moving averages
        if 'sma_20' in data.columns and 'sma_50' in data.columns and 'sma_200' in data.columns:
            if row['sma_20'] > row['sma_50'] > row['sma_200']:
                return 'bull'
            elif row['sma_20'] < row['sma_50'] < row['sma_200']:
                return 'bear'
        
        return 'neutral'
    
    def generate_signals(self, data, index):
        """
        Generate trading signals based on multiple indicators.
        """
        if index < 50:
            return False, False
            
        row = data.iloc[index]
        regime = self.identify_market_regime(data, index)
        
        # RSI conditions
        rsi_bullish = False
        rsi_bearish = False
        if 'rsi_14' in data.columns:
            rsi_bullish = row['rsi_14'] < self.rsi_oversold
            rsi_bearish = row['rsi_14'] > self.rsi_overbought
        
        # Moving average trends
        ma_bullish = False
        ma_bearish = False
        if 'sma_20' in data.columns and 'sma_50' in data.columns:
            ma_bullish = row['sma_20'] > row['sma_50'] and row['close'] > row['sma_20']
            ma_bearish = row['sma_20'] < row['sma_50'] and row['close'] < row['sma_20']
        
        # MACD signals if available
        macd_bullish = False
        macd_bearish = False
        if all(col in data.columns for col in ['macd', 'macd_signal']) and index > 0:
            prev = data.iloc[index-1]
            macd_bullish = prev['macd'] < prev['macd_signal'] and row['macd'] > row['macd_signal']
            macd_bearish = prev['macd'] > prev['macd_signal'] and row['macd'] < row['macd_signal']
        
        # Combine signals based on market regime
        buy_signal = False
        sell_signal = False
        
        if regime == 'bull':
            # More aggressive in bull markets
            buy_signal = (rsi_bullish or (ma_bullish and macd_bullish))
            sell_signal = (rsi_bearish and ma_bearish and macd_bearish)
        elif regime == 'bear':
            # More conservative in bear markets
            buy_signal = (rsi_bullish and ma_bullish and macd_bullish)
            sell_signal = (rsi_bearish or (ma_bearish and macd_bearish))
        else:  # neutral
            buy_signal = (rsi_bullish and (ma_bullish or macd_bullish))
            sell_signal = (rsi_bearish and (ma_bearish or macd_bearish))
        
        return buy_signal, sell_signal
    
    def calculate_stop_loss(self, data, index, direction):
        """
        Calculate aggressive stop loss levels.
        """
        row = data.iloc[index]
        atr = row['atr_14'] if 'atr_14' in data.columns else 0
        
        if direction == 'long':
            return row['close'] - (atr * self.stop_atr_multiplier)
        else:  # short
            return row['close'] + (atr * self.stop_atr_multiplier)
    
    def calculate_take_profit(self, data, index, direction):
        """
        Calculate ambitious take profit levels.
        """
        row = data.iloc[index]
        atr = row['atr_14'] if 'atr_14' in data.columns else 0
        
        if direction == 'long':
            return row['close'] + (atr * self.take_profit_atr_multiplier)
        else:  # short
            return row['close'] - (atr * self.take_profit_atr_multiplier)
