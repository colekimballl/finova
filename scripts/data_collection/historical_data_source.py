# utils/historical_data_source.py
import yfinance as yf
import pandas as pd
import os


def download_historical_data(symbol: str, period: str, interval: str, file_path: str):
    """
    Download historical data and save it as a CSV file.

    Parameters:
    - symbol (str): Ticker symbol.
    - period (str): Data period (e.g., '5y').
    - interval (str): Data interval (e.g., '1d').
    - file_path (str): Destination CSV file path.

    Returns:
    - None
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        data = yf.download(tickers=symbol, period=period, interval=interval)
        if data.empty:
            print(
                f"No data fetched for {symbol}. Please check the symbol and parameters."
            )
            return
        data.to_csv(file_path)
        print(f"Downloaded data for {symbol} to {file_path}")
    except Exception as e:
        print(f"Error downloading data: {e}")


if __name__ == "__main__":
    # Example usage with daily interval
    download_historical_data("BTC-USD", "5y", "1d", "data/historical/BTC-USD.csv")
