# main.py

import ccxt
from scripts.common.indicators import IndicatorCalculator
from scripts.phemex.exchange_interface_phemex import PhemexClient
import os
import time
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize the exchange client
api_key = os.getenv("PHEMEX_API_KEY")
api_secret = os.getenv("PHEMEX_API_SECRET")

phemex = ccxt.phemex(
    {
        "enableRateLimit": True,
        "apiKey": api_key,
        "secret": api_secret,
    }
)

symbol = "BTC/USD"

# Initialize the indicator calculator
indicator_calculator = IndicatorCalculator(
    exchange=phemex, symbol=symbol, timeframe="1h", num_bars=500
)

# Initialize the exchange interface
client = PhemexClient(api_key=api_key, api_secret=api_secret)

# Main loop
while True:
    try:
        # Fetch data and calculate indicators
        indicator_calculator.update_data()
        indicator_calculator.calculate_vwma()
        indicator_calculator.calculate_sma()
        indicator_calculator.calculate_additional_indicators()
        indicator_calculator.generate_signals()

        # Get the latest signal
        df = indicator_calculator.get_dataframe()
        latest_signal = df["Signal"].iloc[-1]

        if latest_signal == 1:
            print("Signal to BUY")
            # Place a buy order
            order = client.place_order(symbol=symbol, side="buy", amount=0.001)
            print(f"Order placed: {order}")
        elif latest_signal == -1:
            print("Signal to SELL")
            # Place a sell order
            order = client.place_order(symbol=symbol, side="sell", amount=0.001)
            print(f"Order placed: {order}")
        else:
            print("No clear signal")

        # Wait for the next iteration
        time.sleep(3600)  # Sleep for one hour (adjust based on your timeframe)
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
        time.sleep(60)  # Wait before retrying
