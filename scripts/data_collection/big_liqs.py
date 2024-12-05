import asyncio
import json
import os
from datetime import datetime
import pytz
from websockets import connect
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from termcolor import cprint
import logging
import aiofiles
from logging.handlers import RotatingFileHandler

# Configure logging with RotatingFileHandler to prevent log files from growing indefinitely
logger = logging.getLogger("BinanceBigLiquidation")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# Rotating File Handler: 5 MB per file, keep up to 5 backups
file_handler = RotatingFileHandler(
    "binance_big_liquidation.log", maxBytes=5 * 1024 * 1024, backupCount=5
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Stream Handler for console output
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# WebSocket URL for Binance Futures forced liquidation orders
websocket_url = "wss://fstream.binance.com/ws/!forceOrder@arr"
filename = "binance_bigliqs.csv"


# Initialize CSV file with headers asynchronously
async def initialize_csv(filename):
    if not os.path.isfile(filename):
        async with aiofiles.open(filename, "w") as f:
            header = (
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
            )
            await f.write(header)
            logger.info(f"Created CSV file with headers: {filename}")


# Exponential backoff for reconnection attempts
async def exponential_backoff(retries):
    backoff_time = min(60, (2**retries))
    logger.info(f"Reconnecting in {backoff_time} seconds...")
    await asyncio.sleep(backoff_time)


async def binance_big_liquidation(uri, filename):
    await initialize_csv(filename)
    retries = 0
    while True:
        try:
            # Establish WebSocket connection with ping_interval and ping_timeout
            async with connect(uri, ping_interval=30, ping_timeout=10) as websocket:
                logger.info("Connected to WebSocket.")
                retries = 0  # Reset retries after successful connection
                while True:
                    try:
                        # Receive message with a timeout to handle inactivity
                        msg = await asyncio.wait_for(websocket.recv(), timeout=60)
                        data = json.loads(msg)
                        order_data = data.get("o", {})

                        # Extract and process data
                        symbol = order_data.get("s", "").replace("USDT", "")
                        side = order_data.get("S", "").upper()
                        order_type = order_data.get("o", "")
                        time_in_force = order_data.get("f", "")
                        original_quantity = float(order_data.get("q", 0))
                        price = float(order_data.get("p", 0))
                        average_price = float(order_data.get("ap", 0))
                        order_status = order_data.get("X", "")
                        timestamp = int(order_data.get("T", 0))
                        filled_quantity = float(order_data.get("z", 0))
                        usd_size = filled_quantity * price

                        # Only process liquidations over $100,000
                        if usd_size < 100000:
                            continue  # Skip smaller liquidations

                        # Convert timestamp to US/Eastern timezone
                        est = pytz.timezone("US/Eastern")
                        time_est = datetime.fromtimestamp(
                            timestamp / 1000, est
                        ).strftime("%Y-%m-%d %H:%M:%S")

                        # Determine liquidation type and formatting
                        liquidation_type = "L LIQ" if side == "SELL" else "S LIQ"
                        symbol_short = symbol[:4]
                        output = f"{liquidation_type} {symbol_short} {time_est} ${usd_size:,.0f}"

                        # Determine color and attributes based on usd_size
                        if usd_size >= 1000000:
                            color = "magenta"
                            attrs = ["bold", "blink"]
                            stars = "*" * 5
                            output = f"{stars} {output}"
                            repeat = 5
                        elif usd_size >= 250000:
                            color = "yellow"
                            attrs = ["bold", "blink"]
                            stars = "*" * 3
                            output = f"{stars} {output}"
                            repeat = 3
                        elif usd_size >= 100000:
                            color = "cyan"
                            attrs = ["bold"]
                            stars = "*" * 1
                            output = f"{stars} {output}"
                            repeat = 1
                        else:
                            # This else block won't be reached due to the usd_size filter above
                            color = "white"
                            attrs = []
                            repeat = 1

                        # Print the liquidation with the determined color and attributes
                        for _ in range(repeat):
                            cprint(output, color, attrs=attrs)

                        # Log an empty line for readability in logs
                        logger.info("")

                        # Prepare CSV data
                        msg_values = [
                            symbol,
                            side,
                            order_type,
                            time_in_force,
                            f"{original_quantity:.8f}",  # Adjust decimal places as needed
                            f"{price:.2f}",
                            f"{average_price:.2f}",
                            order_status,
                            f"{usd_size:.2f}",
                        ]

                        # Asynchronously write to CSV
                        async with aiofiles.open(filename, "a") as f:
                            trade_info = ",".join(msg_values) + "\n"
                            await f.write(trade_info)

                    except asyncio.TimeoutError:
                        # If no message is received within timeout, send a ping to keep the connection alive
                        try:
                            await websocket.ping()
                            logger.debug("Sent ping to keep connection alive.")
                        except Exception as e:
                            logger.error(f"Failed to send ping: {e}", exc_info=True)

        except (ConnectionClosedError, ConnectionClosedOK) as e:
            logger.error(f"WebSocket connection closed: {e}", exc_info=True)
            retries += 1
            await exponential_backoff(retries)
        except asyncio.TimeoutError as e:
            logger.error(f"Timeout during receiving: {e}", exc_info=True)
            retries += 1
            await exponential_backoff(retries)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            retries += 1
            await exponential_backoff(retries)


# Run the asynchronous function
if __name__ == "__main__":
    try:
        asyncio.run(binance_big_liquidation(websocket_url, filename))
    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
