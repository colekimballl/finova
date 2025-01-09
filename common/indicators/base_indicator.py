# common/indicators/base_indicator.py

from abc import ABC, abstractmethod
import pandas as pd


class BaseIndicator(ABC):
    """
    Abstract base class for all technical indicators.
    """

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the indicator and adds it to the DataFrame.

        Parameters:
        - df (pd.DataFrame): DataFrame containing OHLCV data.

        Returns:
        - pd.DataFrame: DataFrame with the new indicator.
        """
        pass

