# indicators.py

import ccxt
import pandas as pd
import pandas_ta as ta


class IndicatorCalculator:
    def __init__(self, exchange, symbol, timeframe="1d", num_bars=100):
        """
        Initializes the IndicatorCalculator with exchange, symbol, and data parameters.
        :param exchange: ccxt exchange instance
        :param symbol: Trading symbol (e.g., 'BTC/USD')
        :param timeframe: Timeframe for OHLCV data (e.g., '1d', '1h')
        :param num_bars: Number of bars to fetch
        """
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.num_bars = num_bars
        self.df = None

    def fetch_data(self):
        """
        Fetches OHLCV data from the exchange and stores it in a DataFrame.
        """
        try:
            bars = self.exchange.fetch_ohlcv(
                self.symbol, timeframe=self.timeframe, limit=self.num_bars
            )
            self.df = pd.DataFrame(
                bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], unit="ms")
            self.df.set_index("timestamp", inplace=True)
        except Exception as e:
            print(f"Error fetching data: {e}")

    def calculate_vwma(self, periods=[20, 41, 75]):
        """
        Calculates the VWMA indicators for the specified periods.
        :param periods: List of periods for which to calculate VWMA
        """
        if self.df is None:
            self.fetch_data()

        for period in periods:
            # Calculate VWMA using pandas_ta
            self.df[f"VWMA_{period}"] = ta.vwma(
                self.df["close"], self.df["volume"], length=period
            )

    def calculate_sma(self, periods=[20, 41, 75]):
        """
        Calculates the SMA indicators for the specified periods.
        :param periods: List of periods for which to calculate SMA
        """
        if self.df is None:
            self.fetch_data()

        for period in periods:
            self.df[f"SMA_{period}"] = self.df["close"].rolling(window=period).mean()

    def calculate_additional_indicators(self):
        """
        Calculates additional indicators like RSI, MACD, etc.
        """
        if self.df is None:
            self.fetch_data()

        # RSI
        self.df["RSI_14"] = ta.rsi(self.df["close"], length=14)

        # MACD
        macd = ta.macd(self.df["close"])
        self.df = pd.concat([self.df, macd], axis=1)

        # Stochastic Oscillator
        stoch = ta.stoch(self.df["high"], self.df["low"], self.df["close"])
        self.df = pd.concat([self.df, stoch], axis=1)

    def generate_signals(self):
        """
        Generates trading signals based on the indicators.
        """
        if self.df is None:
            self.fetch_data()

        # Example signal: VWMA crosses above SMA
        self.df["Signal"] = 0
        self.df.loc[self.df["VWMA_20"] > self.df["SMA_20"], "Signal"] = 1  # Buy
        self.df.loc[self.df["VWMA_20"] < self.df["SMA_20"], "Signal"] = -1  # Sell

    def get_dataframe(self):
        """
        Returns the DataFrame containing all the data and indicators.
        """
        if self.df is None:
            self.fetch_data()
        return self.df

    def update_data(self):
        """
        Updates the data by fetching the latest OHLCV data.
        """
        self.fetch_data()
