# strategies/ensemble_strategy.py

import backtrader as bt
from strategies.sma_cross import SmaCross
from strategies.sma_rsi_strategy import SmaRsiStrategy

class EnsembleStrategy(bt.Strategy):
    def __init__(self):
        self.sma_cross = SmaCross()
        self.sma_rsi = SmaRsiStrategy()

