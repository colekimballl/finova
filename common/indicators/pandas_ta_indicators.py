# common/indicators/pandas_ta_indicators.py

import pandas as pd
import pandas_ta as ta_pt
import logging
from .base_indicator import BaseIndicator


class PandasTA_SMA(BaseIndicator):
    def __init__(self, length: int):
        self.length = length
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Calculating Pandas_TA SMA for length {self.length}")
        df[f"pandas_ta_SMA_{self.length}"] = ta_pt.sma(df["Close"], length=self.length)
        return df


class PandasTA_EMA(BaseIndicator):
    def __init__(self, length: int):
        self.length = length
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Calculating Pandas_TA EMA for length {self.length}")
        df[f"pandas_ta_EMA_{self.length}"] = ta_pt.ema(df["Close"], length=self.length)
        return df


class PandasTA_RSI(BaseIndicator):
    def __init__(self, length: int):
        self.length = length
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Calculating Pandas_TA RSI for length {self.length}")
        df[f"pandas_ta_RSI_{self.length}"] = ta_pt.rsi(df["Close"], length=self.length)
        return df


# Add more Pandas_TA indicators following the same pattern

