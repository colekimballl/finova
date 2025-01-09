import pandas as pd
from typing import List
from .ta_lib_indicators import TA_LibSMA, TA_LibEMA  # Import required indicators
from .pandas_ta_indicators import PandasTA_SMA, PandasTA_EMA, PandasTA_RSI
import logging


class IndicatorManager:
    """
    Manages the calculation of various technical indicators.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.indicators = []

    def add_indicator(self, indicator: 'BaseIndicator'):
        """
        Adds an indicator to the manager.

        Parameters:
        - indicator (BaseIndicator): An instance of an indicator.
        """
        self.indicators.append(indicator)
        self.logger.info(f"Added indicator: {indicator.__class__.__name__}")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all added indicators and appends them to the DataFrame.

        Parameters:
        - df (pd.DataFrame): DataFrame containing OHLCV data.

        Returns:
        - pd.DataFrame: DataFrame with new indicators.
        """
        for indicator in self.indicators:
            df = indicator.calculate(df)
        return df

    def list_indicators(self) -> List[str]:
        """
        Lists all added indicators.

        Returns:
        - List[str]: List of indicator names.
        """
        return [indicator.__class__.__name__ for indicator in self.indicators]


def get_ohlcv_df(exchange, symbol, timeframe, limit):
    """
    Fetches OHLCV data from the exchange and returns a DataFrame.

    Parameters:
    - exchange: The exchange instance (e.g., phemex).
    - symbol (str): Trading symbol (e.g., 'BTC/USD').
    - timeframe (str): Timeframe for the data (e.g., '1h').
    - limit (int): Number of data points to fetch.

    Returns:
    - pd.DataFrame: DataFrame containing OHLCV data.
    """
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        print(f"Error fetching OHLCV data: {e}")
        return pd.DataFrame()


def calculate_vwap(df):
    """
    Calculates the VWAP for the given DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'high', 'low', 'close', and 'volume' columns.

    Returns:
    - pd.DataFrame: DataFrame with an additional 'VWAP' column.
    """
    if df.empty:
        print("DataFrame is empty. Cannot calculate VWAP.")
        return df

    # Calculate Typical Price
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3

    # Calculate Volume * Typical Price
    df["vol_x_typ_price"] = df["volume"] * df["typical_price"]

    # Calculate Cumulative Volume and Cumulative Volume * Typical Price
    df["cumulative_volume"] = df["volume"].cumsum()
    df["cumulative_vol_x_typ_price"] = df["vol_x_typ_price"].cumsum()

    # Calculate VWAP
    df["VWAP"] = df["cumulative_vol_x_typ_price"] / df["cumulative_volume"]

    # Clean up intermediate columns if not needed
    df.drop(
        ["typical_price", "vol_x_typ_price", "cumulative_vol_x_typ_price"],
        axis=1,
        inplace=True,
    )

    # Handle any NaN values
    df.fillna(method="ffill", inplace=True)
    df.fillna(0, inplace=True)

    return df
