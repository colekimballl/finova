# scripts/backtesting/sma_backtest.py

from scripts.common.gui import prompt_user_input

# Prompt for SMA period
sma_period_input = prompt_user_input("Enter SMA period for backtesting:")
sma_period = int(sma_period_input) if sma_period_input else 20

# Run backtest with the specified SMA period
run_backtest(sma_period=sma_period)
