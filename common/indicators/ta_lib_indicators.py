# common/indicators/ta_lib_indicators.py

import pandas as pd
import talib as ta
import logging
from .base_indicator import BaseIndicator


class TA_LibSMA(BaseIndicator):
    def __init__(self, period: int):
        self.period = period
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Calculating TA-Lib SMA for period {self.period}")
        df[f"SMA_{self.period}"] = ta.SMA(df["Close"], timeperiod=self.period)
        return df


class TA_LibEMA(BaseIndicator):
    def __init__(self, period: int):
        self.period = period
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Calculating TA-Lib EMA for period {self.period}")
        df[f"EMA_{self.period}"] = ta.EMA(df["Close"], timeperiod=self.period)
        return df


# Add more TA-Lib indicators following the same pattern

