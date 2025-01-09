# In your strategy or script
from scripts.common.technical_indicators import get_ohlcv_df, calculate_vwap
import ccxt

# Exchange setup (ensure you have your API keys set up properly)
exchange = ccxt.phemex(
    {
        "apiKey": YOUR_API_KEY,
        "secret": YOUR_SECRET_KEY,
    }
)

symbol = "BTC/USD"
timeframe = "1h"
limit = 100

# Get OHLCV data
df = get_ohlcv_df(exchange, symbol, timeframe, limit)

# Calculate VWAP
df_with_vwap = calculate_vwap(df)

print(df_with_vwap.tail())
