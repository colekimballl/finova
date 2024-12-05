import pandas as pd


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
