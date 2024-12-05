import ccxt
import logging

class ExchangeClient:
    def __init__(self, exchange_name, api_key, secret):
        self.exchange = getattr(ccxt, exchange_name)({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
        })
        self.logger = logging.getLogger(exchange_name)

    def fetch_order_book(self, symbol):
        try:
            return self.exchange.fetch_order_book(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching order book for {symbol}: {e}")
            return None

    # Add more methods as needed (e.g., create_order, cancel_orders)

