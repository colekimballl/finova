# orders/algo_orders.py
import ccxt
import time
import schedule
from utils.config import phemex, symbol

params = {"timeInForce": "PostOnly"}
size = 1
bid = 29000  # Example bid price, replace with dynamic values as needed


def place_order():
    try:
        order = phemex.create_limit_buy_order(symbol, size, bid, params)
        print(f"Order placed: {order}")
        time.sleep(10)
        phemex.cancel_all_orders(symbol)
        print("All orders canceled.")
    except Exception as e:
        print(f"Error placing order: {e}")


# Schedule the order to be placed every 2 seconds
schedule.every(2).seconds.do(place_order)


def run_scheduler():
    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            print(f"Scheduler error: {e}")
            time.sleep(30)
