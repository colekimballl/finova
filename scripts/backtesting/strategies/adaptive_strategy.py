# strategies/adaptive_strategy.py

import backtrader as bt
from strategies.sma_cross import SmaCross
from strategies.sma_rsi_strategy import SmaRsiStrategy

class AdaptiveStrategy(bt.Strategy):
    params = (
        ('sma_period', 20),
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
    )

    def __init__(self):
        self.sma = bt.ind.SMA(period=self.p.sma_period)
        self.rsi = bt.ind.RSI_SMA(period=self.p.rsi_period)
        self.crossover = bt.ind.CrossOver(self.data.close, self.sma)

        self.strategy_a = SmaCross(self.p.sma_period)
        self.strategy_b = SmaRsiStrategy(
            self.p.sma_period, 
            self.p.rsi_period, 
            self.p.rsi_overbought, 
            self.p.rsi_oversold
        )

    def next(self):
        if self.rsi < self.p.rsi_oversold:
            # Activate Strategy A
            self.strategy_a.next()
        elif self.rsi > self.p.rsi_overbought:
            # Activate Strategy B
            self.strategy_b.next()
        else:
            # Hold or implement a neutral strategy
            pass

