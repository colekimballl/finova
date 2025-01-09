# strategies/sma_cross.py

import backtrader as bt

class SmaCross(bt.SignalStrategy):
    params = (('sma_period', 20),)

    def __init__(self):
        sma = bt.ind.SMA(period=self.p.sma_period)
        crossover = bt.ind.CrossOver(self.data.close, sma)
        self.signal_add(bt.SIGNAL_LONG, crossover)

