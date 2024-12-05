# RSI


def df_rsi(symbol=symbol, timeframe=timeframe, limit=limit):

    print("starting indis...")

    bars = phemex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    # print(bars)

    # pandas & TA, talib
    df_rsi = pd.DataFrame(
        bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df_rsi["timestamp"] = pd.to_datetime(df_rsi["timestamp"], unit="ms")
    # if bid < the 20 day sma then = BEARISH, if bid > 20 day sma = BULLISH
    bid = ask_bid(symbol)[1]

    # RSI
    rsi = RSIIndicator(df_rsi["close"])
    df_rsi["rsi"] = rsi.rsi()

    print(df_rsi)

    return df_rsi
