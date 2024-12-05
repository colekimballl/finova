#

import asyncio
import json
import os
from datetime import datetime
import pytz
from websockets import connect
from termcolor import cprint
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("binance_liquidation.log"), logging.StreamHandler()],
)

# Correct WebSocket URL for Futures forced liquidation orders
websocket_url = "wss://fstream.binance.com/ws/!forceOrder@arr"
filename = "binance.csv"

# Initialize CSV file with headers if it doesn't exist
if not os.path.isfile(filename):
    with open(filename, "w") as f:
        f.write(
            ",".join(
                [
                    "symbol",
                    "side",
                    "order_type",
                    "time_in_force",
                    "original_quantity",
                    "price",
                    "average_price",
                    "order_status",
                    "usd_size",
                ]
            )
            + "\n"
        )  # Added newline


async def binance_liquidation(uri, filename):
    while True:
        try:
            async with connect(uri) as websocket:
                logging.info("Connected to WebSocket.")
                while True:
                    msg = await websocket.recv()
                    order_data = json.loads(msg).get("o", {})

                    # Extract and process data
                    symbol = order_data.get("s", "").replace("USDT", "")
                    side = order_data.get("S", "")
                    order_type = order_data.get("o", "")
                    time_in_force = order_data.get("f", "")
                    original_quantity = float(order_data.get("q", 0))
                    price = float(order_data.get("p", 0))
                    average_price = float(order_data.get("ap", 0))
                    order_status = order_data.get("X", "")
                    timestamp = int(order_data.get("T", 0))
                    filled_quantity = float(order_data.get("z", 0))
                    usd_size = filled_quantity * price

                    # Convert timestamp to US/Eastern timezone
                    est = pytz.timezone("US/Eastern")
                    time_est = datetime.fromtimestamp(timestamp / 1000, est).strftime(
                        "%H:%M:%S"
                    )

                    if usd_size > 3000:
                        liquidation_type = (
                            "L LIQ" if side.upper() == "SELL" else "S LIQ"
                        )
                        symbol_short = symbol[:4]
                        output = f"{liquidation_type} {symbol_short} {time_est} ${usd_size:,.0f}"
                        color = "green" if side.upper() == "SELL" else "red"
                        attrs = ["bold"] if usd_size > 10000 else []

                        if usd_size > 250000:
                            stars = "*" * 3
                            attrs.append("blink")
                            output = f"{stars} {output}"
                            for _ in range(4):
                                cprint(output, "white", f"on_{color}", attrs=attrs)
                        elif usd_size > 100000:
                            stars = "*" * 1
                            attrs.append("blink")
                            output = f"{stars} {output}"
                            for _ in range(2):
                                cprint(output, "white", f"on_{color}", attrs=attrs)
                        elif usd_size > 25000:
                            cprint(output, color, attrs=attrs)
                        else:
                            cprint(output, color, attrs=attrs)

                        logging.info("")  # Placeholder for spacing

                    # Prepare CSV data
                    msg_values = [
                        symbol,
                        side,
                        order_type,
                        time_in_force,
                        str(original_quantity),
                        str(price),
                        str(average_price),
                        order_status,
                        f"{usd_size:.2f}",
                    ]

                    with open(filename, "a") as f:
                        trade_info = ",".join(msg_values) + "\n"
                        f.write(trade_info)

        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)
            logging.info("Reconnecting in 5 seconds...")
            await asyncio.sleep(5)


# Run the asynchronous function
if __name__ == "__main__":
    try:
        asyncio.run(binance_liquidation(websocket_url, filename))
    except KeyboardInterrupt:
        logging.info("Program terminated by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
