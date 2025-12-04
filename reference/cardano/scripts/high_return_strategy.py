#!/usr/bin/env python
'''
High-Risk, High-Return Crypto Trading Strategy
----------------------------------------------
Aggressive trading strategy designed to maximize returns 
with higher risk tolerance, leveraged positions, and 
multiple entry/exit signals.
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtest_system import RiskManager, Trade, BacktestEngine, load_processed_data
import argparse

class AggressiveRiskManager(RiskManager):
    """
    Aggressive risk management with higher position sizing and leverage.
    """
    def __init__(self, initial_capital=10000, max_drawdown_pct=30, risk_per_trade_pct=5, max_leverage=5):
        super().__init__(initial_capital, max_drawdown_pct, risk_per_trade_pct)
        self.max_leverage = max_leverage
        self.current_leverage = 1.0  # Start with no leverage
        self.profitable_trades_streak = 0
        self.losing_trades_streak = 0
        
    def calculate_position_size(self, entry_price, stop_loss_price):
        """
        Calculate position size with dynamic leverage based on market conditions and past performance.
        """
        # Base position size from parent class
        base_size = super().calculate_position_size(entry_price, stop_loss_price)
        
        # Apply dynamic leverage based on winning/losing streak
        if self.profitable_trades_streak >= 2:
            # Increase leverage after consecutive wins (confidence boost)
            self.current_leverage = min(self.max_leverage, self.current_leverage + 0.5)
        elif self.losing_trades_streak >= 2:
            # Decrease leverage after consecutive losses (risk reduction)
            self.current_leverage = max(1.0, self.current_leverage - 0.5)
        
        # Apply leverage to position size
        leveraged_size = base_size * self.current_leverage
        
        return leveraged_size
    
    def update_capital(self, profit_loss):
        """
        Update capital and track winning/losing streaks.
        """
        # Track winning/losing streaks
        if profit_loss > 0:
            self.profitable_trades_streak += 1
            self.losing_trades_streak = 0
        elif profit_loss < 0:
            self.losing_trades_streak += 1
            self.profitable_trades_streak = 0
        
        # Update capital using parent method
        super().update_capital(profit_loss)


class HighReturnStrategy:
    """
    Aggressive trading strategy designed for high returns.
    """
    def __init__(self, params=None):
        self.params = params or self.get_default_params()
    
    def get_default_params(self):
        """
        Get default strategy parameters optimized for high returns.
        """
        return {
            # RSI Parameters - More aggressive thresholds
            'rsi_oversold_threshold': 30,
            'rsi_overbought_threshold': 70,
            'rsi_lookback': 14,
            
            # Moving Average parameters
            'fast_ma': 8,    # Faster MAs for more signals
            'slow_ma': 21,
            'trend_ma': 55,
            
            # Volatility parameters - Accept higher volatility
            'max_bb_width': 0.3,
            'min_atr': 0.02,  # Require minimum volatility for meaningful moves
            
            # Stop-loss and Take-profit - Tighter stops, wider targets
            'stop_atr_multiplier': 1.2,
            'take_profit_atr_multiplier': 6.0,
            'trailing_stop_enabled': True,
            'trailing_stop_activation_pct': 2.0,  # Start trailing after 2% profit
            'trailing_stop_distance_atr': 2.0,    # Trail by 2 ATR
            
            # Volume parameters
            'min_rel_volume': 1.0,   # Accept average volume
            
            # Pattern detection
            'use_engulfing_patterns': True,
            'use_reversal_patterns': True,
            
            # Multiple timeframe analysis
            'use_higher_timeframe_trend': True,
            
            # Market regime parameters
            'bull_market_bias': 1.5,  # Take larger positions in bull markets
            'bear_market_bias': 0.5,  # Smaller positions in bear markets
            
            # Momentum indicators
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            
            # Breakout parameters
            'breakout_lookback': 20,  # Look for breakouts from 20-period ranges
            'breakout_threshold': 2.0,  # ATR multiplier for breakout significance
            
            # Entry/exit refinement
            'use_fractals': True,
            'fractal_lookback': 5,
            
            # Position management
            'scale_in_enabled': True,
            'scale_in_levels': 3,
            'scale_in_interval_pct': 1.0,  # Add to position every 1% against
            
            # Advanced filters
            'use_price_action': True,
            'use_zigzag': True,
            'zigzag_threshold': 5
        }
    
    def identify_market_regime(self, data, index):
        """
        Identify current market regime (bull, bear, or sideways).
        """
        if index < 55:  # Ensure enough data
            return 'unknown'
        
        row = data.iloc[index]
        
        # Use multiple timeframe analysis for regime identification
        if 'sma_20' in data.columns and 'sma_50' in data.columns and 'sma_200' in data.columns:
            # Bull market conditions
            if (row['sma_20'] > row['sma_50'] > row['sma_200'] and
                row['close'] > row['sma_20']):
                return 'bull'
            
            # Bear market conditions
            elif (row['sma_20'] < row['sma_50'] < row['sma_200'] and
                  row['close'] < row['sma_20']):
                return 'bear'
            
            # Transition/sideways market
            else:
                return 'sideways'
        
        return 'unknown'
    
    def detect_breakout(self, data, index):
        """
        Detect price breakouts from consolidation patterns.
        """
        if index < self.params['breakout_lookback'] + 5:
            return False, False
        
        # Calculate recent price range
        lookback = self.params['breakout_lookback']
        recent_high = max(data.iloc[index-lookback:index]['high'])
        recent_low = min(data.iloc[index-lookback:index]['low'])
        
        # Get ATR for volatility context
        atr = data.iloc[index]['atr_14'] if 'atr_14' in data.columns else (recent_high - recent_low) / lookback
        
        # Check if current bar breaks the range
        current_close = data.iloc[index]['close']
        prev_close = data.iloc[index-1]['close']
        
        # Breakout thresholds
        upper_threshold = recent_high + (atr * 0.5)
        lower_threshold = recent_low - (atr * 0.5)
        
        # Check for breakouts
        upside_breakout = prev_close < recent_high and current_close > upper_threshold
        downside_breakout = prev_close > recent_low and current_close < lower_threshold
        
        return upside_breakout, downside_breakout
    
    def detect_momentum_shift(self, data, index):
        """
        Detect shifts in momentum using MACD and RSI.
        """
        if index < 30:
            return False, False
        
        bullish_shift = False
        bearish_shift = False
        
        # MACD crossover
        if all(col in data.columns for col in ['macd', 'macd_signal']):
            current = data.iloc[index]
            prev = data.iloc[index-1]
            
            # Bullish crossover
            if prev['macd'] < prev['macd_signal'] and current['macd'] > current['macd_signal']:
                bullish_shift = True
            
            # Bearish crossover
            if prev['macd'] > prev['macd_signal'] and current['macd'] < current['macd_signal']:
                bearish_shift = True
        
        # RSI divergence
        if 'rsi_14' in data.columns and index >= 10:
            rsi_vals = data.iloc[index-10:index+1]['rsi_14']
            price_vals = data.iloc[index-10:index+1]['close']
            
            # Simplified divergence check (not complete but indicative)
            if (price_vals.iloc[-1] < price_vals.min() and 
                rsi_vals.iloc[-1] > rsi_vals.min()):
                bullish_shift = True
            
            if (price_vals.iloc[-1] > price_vals.max() and 
                rsi_vals.iloc[-1] < rsi_vals.max()):
                bearish_shift = True
        
        return bullish_shift, bearish_shift
    
    def detect_reversal_pattern(self, data, index):
        """
        Detect candlestick reversal patterns.
        """
        if not self.params['use_reversal_patterns'] or index < 3:
            return False, False
        
        current = data.iloc[index]
        prev = data.iloc[index-1]
        prev2 = data.iloc[index-2]
        
        # Bullish engulfing
        bullish_engulfing = (
            prev['close'] < prev['open'] and  # Previous candle is bearish
            current['open'] < prev['close'] and  # Open below previous close
            current['close'] > prev['open'] and  # Close above previous open
            current['close'] > current['open']  # Current candle is bullish
        )
        
        # Bearish engulfing
        bearish_engulfing = (
            prev['close'] > prev['open'] and  # Previous candle is bullish
            current['open'] > prev['close'] and  # Open above previous close
            current['close'] < prev['open'] and  # Close below previous open
            current['close'] < current['open']  # Current candle is bearish
        )
        
        # Morning star (bullish)
        morning_star = (
            prev2['close'] < prev2['open'] and  # First candle is bearish
            abs(prev['close'] - prev['open']) < abs(prev2['close'] - prev2['open']) * 0.5 and  # Second candle is small
            current['close'] > current['open'] and  # Third candle is bullish
            current['close'] > (prev2['open'] + prev2['close']) / 2  # Closes above midpoint of first candle
        )
        
        # Evening star (bearish)
        evening_star = (
            prev2['close'] > prev2['open'] and  # First candle is bullish
            abs(prev['close'] - prev['open']) < abs(prev2['close'] - prev2['open']) * 0.5 and  # Second candle is small
            current['close'] < current['open'] and  # Third candle is bearish
            current['close'] < (prev2['open'] + prev2['close']) / 2  # Closes below midpoint of first candle
        )
        
        bullish_reversal = bullish_engulfing or morning_star
        bearish_reversal = bearish_engulfing or evening_star
        
        return bullish_reversal, bearish_reversal
    
    def generate_signals(self, data, index):
        """
        Generate trading signals based on multiple indicators and patterns.
        """
        if index < 55:  # Need sufficient data for all indicators
            return False, False
        
        # Get current data
        row = data.iloc[index]
        
        # --- 1. Determine market regime ---
        regime = self.identify_market_regime(data, index)
        
        # --- 2. Check RSI conditions ---
        rsi_bullish = False
        rsi_bearish = False
        if 'rsi_14' in data.columns:
            rsi_bullish = row['rsi_14'] < self.params['rsi_oversold_threshold']
            rsi_bearish = row['rsi_14'] > self.params['rsi_overbought_threshold']
        
        # --- 3. Check moving average trends ---
        ma_bullish = False
        ma_bearish = False
        if 'sma_20' in data.columns and 'sma_50' in data.columns:
            ma_bullish = row['sma_20'] > row['sma_50'] and row['close'] > row['sma_20']
            ma_bearish = row['sma_20'] < row['sma_50'] and row['close'] < row['sma_20']
        
        # --- 4. Check for breakouts ---
        breakout_up, breakout_down = self.detect_breakout(data, index)
        
        # --- 5. Check for momentum shifts ---
        momentum_up, momentum_down = self.detect_momentum_shift(data, index)
        
        # --- 6. Check for reversal patterns ---
        reversal_up, reversal_down = self.detect_reversal_pattern(data, index)
        
        # --- 7. Check volume confirmation ---
        volume_high = False
        if 'relative_volume' in data.columns:
            volume_high = row['relative_volume'] > self.params['min_rel_volume']
        
        # --- 8. Combine signals based on market regime ---
        buy_signal = False
        sell_signal = False
        
        if regime == 'bull':
            # In bull markets, be more aggressive with buys
            buy_signal = (
                # Strong trend signals in bull market
                (ma_bullish and (rsi_bullish or breakout_up or momentum_up)) or
                # Reversal signals need to be stronger to trigger
                (reversal_up and rsi_bullish and volume_high)
            )
            
            # More conservative with sells in bull markets
            sell_signal = (
                (rsi_bearish and ma_bearish and (breakout_down or momentum_down)) or
                (reversal_down and rsi_bearish and volume_high and ma_bearish)
            )
            
        elif regime == 'bear':
            # In bear markets, be more aggressive with sells
            sell_signal = (
                # Strong trend signals in bear market
                (ma_bearish and (rsi_bearish or breakout_down or momentum_down)) or
                # Reversal signals need to be stronger to trigger
                (reversal_down and rsi_bearish and volume_high)
            )
            
            # More conservative with buys in bear markets
            buy_signal = (
                (rsi_bullish and ma_bullish and (breakout_up or momentum_up)) or
                (reversal_up and rsi_bullish and volume_high and ma_bullish)
            )
            
        else:  # Sideways or unknown regime
            # In ranging markets, focus on oversold/overbought conditions with pattern confirmation
            buy_signal = (
                (rsi_bullish and (reversal_up or breakout_up) and volume_high) or
                (rsi_bullish and momentum_up and volume_high)
            )
            
            sell_signal = (
                (rsi_bearish and (reversal_down or breakout_down) and volume_high) or
                (rsi_bearish and momentum_down and volume_high)
            )
        
        return buy_signal, sell_signal
    
    def calculate_stop_loss(self, data, index, direction):
        """
        Calculate stop loss with dynamic ATR multiplier based on market regime.
        """
        row = data.iloc[index]
        atr = row['atr_14'] if 'atr_14' in data.columns else 0
        
        # Base stop loss calculation
        if direction == 'long':
            stop_price = row['close'] - (atr * self.params['stop_atr_multiplier'])
        else:  # short
            stop_price = row['close'] + (atr * self.params['stop_atr_multiplier'])
        
        # Adjust for recent swing points for better placement
        if self.params['use_fractals'] and index >= self.params['fractal_lookback'] * 2:
            if direction == 'long':
                # For longs, find recent low
                recent_low = min(data.iloc[index-self.params['fractal_lookback']:index]['low'])
                stop_price = min(stop_price, recent_low * 0.99)  # Place just below recent low
            else:
                # For shorts, find recent high
                recent_high = max(data.iloc[index-self.params['fractal_lookback']:index]['high'])
                stop_price = max(stop_price, recent_high * 1.01)  # Place just above recent high
        
        return stop_price
    
    def calculate_take_profit(self, data, index, direction):
        """
        Calculate take profit level based on ATR and price structure.
        """
        row = data.iloc[index]
        atr = row['atr_14'] if 'atr_14' in data.columns else 0
        
        # Base take profit calculation
        if direction == 'long':
            take_profit = row['close'] + (atr * self.params['take_profit_atr_multiplier'])
        else:  # short
            take_profit = row['close'] - (atr * self.params['take_profit_atr_multiplier'])
        
        # Adjust for key levels if available
        if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
            if direction == 'long':
                # For longs, consider upper Bollinger Band as a target
                take_profit = max(take_profit, row['bb_upper'] * 1.05)
            else:
                # For shorts, consider lower Bollinger Band as a target
                take_profit = min(take_profit, row['bb_lower'] * 0.95)
        
        return take_profit


class AggressiveBacktestEngine(BacktestEngine):
    """
    Enhanced backtesting engine with advanced features.
    """
    def __init__(self, data, strategy=None, risk_manager=None, trading_fee_pct=0.1):
        super().__init__(data, strategy, risk_manager, trading_fee_pct)
        self.trailing_stops = {}  # Track trailing stops for open trades
    
    def _open_trade(self, index, time, price, direction):
        """
        Open a new trade with enhanced position management.
        """
        # Calculate stop loss and take profit
        stop_loss = self.strategy.calculate_stop_loss(self.data, index, direction)
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
        
        # Initialize trailing stop if enabled
        if hasattr(self.strategy.params, 'trailing_stop_enabled') and self.strategy.params['trailing_stop_enabled']:
            self.trailing_stops[id(trade)] = {
                'activated': False,
                'current_stop': stop_loss
            }
        
        # Account for fees
        fee = price * position_size * self.trading_fee_pct
        self.risk_manager.update_capital(-fee)
    
    def _manage_open_trades(self, index, time, price):
        """
        Manage open trades with trailing stops and advanced exit criteria.
        """
        # Check each open trade
        remaining_trades = []
        for trade in self.open_trades:
            # Update trailing stop if activated and enabled
            if hasattr(self.strategy.params, 'trailing_stop_enabled') and self.strategy.params['trailing_stop_enabled']:
                self._update_trailing_stop(trade, price)
            
            # Get current stop level (trailing or original)
            current_stop = (self.trailing_stops[id(trade)]['current_stop'] 
                           if id(trade) in self.trailing_stops 
                           else trade.stop_loss)
            
            # Check if trailing stop is hit
            if ((trade.direction == 'long' and price <= current_stop) or
                (trade.direction == 'short' and price >= current_stop)):
                self._close_trade(trade, time, current_stop, 'trailing_stop')
                if id(trade) in self.trailing_stops:
                    del self.trailing_stops[id(trade)]
                continue
            
            # Check if take profit is hit
            if trade.is_take_profit_hit(price):
                self._close_trade(trade, time, trade.take_profit, 'take_profit')
                if id(trade) in self.trailing_stops:
                    del self.trailing_stops[id(trade)]
                continue
            
            # Check for reversal signals as exit criteria
            buy_signal, sell_signal = self.strategy.generate_signals(self.data, index)
            
            # Exit long if sell signal appears
            if trade.direction == 'long' and sell_signal:
                self._close_trade(trade, time, price, 'reversal_signal')
                if id(trade) in self.trailing_stops:
                    del self.trailing_stops[id(trade)]
                continue
            
            # Exit short if buy signal appears
            if trade.direction == 'short' and buy_signal:
                self._close_trade(trade, time, price, 'reversal_signal')
                if id(trade) in self.trailing_stops:
                    del self.trailing_stops[id(trade)]
                continue
            
            # Keep the trade open
            remaining_trades.append(trade)
        
        # Update open trades list
        self.open_trades = remaining_trades
    
    def _update_trailing_stop(self, trade, current_price):
        """
        Update trailing stop if conditions are met.
        """
        if id(trade) not in self.trailing_stops:
            return
        
        trail_info = self.trailing_stops[id(trade)]
        activation_threshold = trade.entry_price * (1 + self.strategy.params['trailing_stop_activation_pct']/100)
        
        if trade.direction == 'long':
            # Check if price has moved enough to activate trailing stop
            if not trail_info['activated'] and current_price >= activation_threshold:
                trail_info['activated'] = True
            
            # Update trailing stop if activated
            if trail_info['activated']:
                # Calculate new stop level
                atr = self.data.iloc[-1]['atr_14'] if 'atr_14' in self.data.columns else 0
                new_stop = current_price - (atr * self.strategy.params['trailing_stop_distance_atr'])
                
                # Only move stop up, never down
                if new_stop > trail_info['current_stop']:
                    trail_info['current_stop'] = new_stop
        
        else:  # short
            # For shorts, trailing stop activates when price moves down enough
            activation_threshold = trade.entry_price * (1 - self.strategy.params['trailing_stop_activation_pct']/100)
            
            if not trail_info['activated'] and current_price <= activation_threshold:
                trail_info['activated'] = True
            
            # Update trailing stop if activated
            if trail_info['activated']:
                atr = self.data.iloc[-1]['atr_14'] if 'atr_14' in self.data.columns else 0
                new_stop = current_price + (atr * self.strategy.params['trailing_stop_distance_atr'])
                
                # Only move stop down, never up
                if new_stop < trail_info['current_stop']:
                    trail_info['current_stop'] = new_stop


def run_high_return_backtest(
    timeframe='1h', 
    initial_capital=10000, 
    max_drawdown=30, 
    risk_per_trade=5, 
    max_leverage=5,
    trading_fee=0.1, 
    output_dir='../backtest_results'
):
    """
    Run a high-return strategy backtest.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = load_processed_data(timeframe)
    if data is None:
        print(f"Failed to load {timeframe} data. Exiting.")
        return
    
    print(f"Running high-return strategy backtest on {len(data)} rows of {timeframe} data...")
    
    # Create strategy
    strategy = HighReturnStrategy()
    
    print("High-Return Strategy Parameters:")
    for param, value in strategy.params.items():
        print(f"  {param}: {value}")
    
    # Create aggressive risk manager
    risk_manager = AggressiveRiskManager(
        initial_capital=initial_capital,
        max_drawdown_pct=max_drawdown,
        risk_per_trade_pct=risk_per_trade,
        max_leverage=max_leverage
    )
    
    # Create and run backtest
    backtest = AggressiveBacktestEngine(
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
    print("HIGH-RETURN STRATEGY BACKTEST RESULTS")
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
            # Calculate compound effect over 4 years
            compound_4yr = ((1 + annual_return/100) ** 4 - 1) * 100
            print(f"4-Year Compound Return: {compound_4yr:.2f}%")
    
    # Plot results
    plot_path = os.path.join(output_dir, f"high_return_backtest_{timeframe}_full_history.png")
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
        trade_stats_path = os.path.join(output_dir, f"high_return_trade_stats_{timeframe}.csv")
        trade_df.to_csv(trade_stats_path, index=False)
        print(f"Detailed trade statistics saved to {trade_stats_path}")
    
    return backtest, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='High-Return Crypto Trading Strategy')
    parser.add_argument('--timeframe', type=str, default='1h', 
                      choices=['1h', '4h', '1d'],
                      help='Timeframe to use (default: 1h)')
    parser.add_argument('--capital', type=float, default=10000,
                      help='Initial capital (default: $10,000)')
    parser.add_argument('--max-drawdown', type=float, default=30,
                      help='Maximum allowed drawdown percentage (default: 30%%)')
    parser.add_argument('--risk-per-trade', type=float, default=5,
                      help='Risk per trade as percentage of capital (default: 5%%)')
    parser.add_argument('--max-leverage', type=float, default=5,
                      help='Maximum leverage to use (default: 5x)')
    parser.add_argument('--fee', type=float, default=0.1,
                      help='Trading fee percentage (default: 0.1%%)')
    parser.add_argument('--output', type=str, default='../backtest_results',
                      help='Output directory for backtest results')
    
    args = parser.parse_args()
    
    run_high_return_backtest(
        timeframe=args.timeframe,
        initial_capital=args.capital,
        max_drawdown=args.max_drawdown,
        risk_per_trade=args.risk_per_trade,
        max_leverage=args.max_leverage,
        trading_fee=args.fee,
        output_dir=args.output
    )
