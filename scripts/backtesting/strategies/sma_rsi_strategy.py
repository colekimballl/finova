# strategies/sma_rsi_strategy.py

import backtrader as bt

class SmaRsiStrategy(bt.SignalStrategy):
    params = (
        ('sma_period', 20),
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30)
    )

    def __init__(self):
        sma = bt.ind.SMA(period=self.p.sma_period)
        rsi = bt.ind.RSI_SMA(period=self.p.rsi_period)
        crossover = bt.ind.CrossOver(self.data.close, sma)

        # Long signal: Price crosses above SMA and RSI is not overbought
        long_signal = (crossover > 0) & (rsi < self.p.rsi_overbought)
        self.signal_add(bt.SIGNAL_LONG, long_signal)

        # Short signal: Price crosses below SMA and RSI is not oversold
        short_signal = (crossover < 0) & (rsi > self.p.rsi_oversold)
        self.signal_add(bt.SIGNAL_SHORT, short_signal)

