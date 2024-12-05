# exchange_interface_phemex.py

import ccxt


class PhemexClient:
    def __init__(self, api_key, api_secret):
        self.exchange = ccxt.phemex(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            }
        )

    def place_order(self, symbol, side, amount, price=None, params={}):
        """
        Places an order on Phemex.
        :param symbol: Symbol to trade (e.g., 'BTC/USD')
        :param side: 'buy' or 'sell'
        :param amount: Amount to trade
        :param price: Limit price. If None, a market order is placed.
        :param params: Additional parameters
        :return: Order object
        """
        try:
            if price is None:
                if side == "buy":
                    order = self.exchange.create_market_buy_order(
                        symbol, amount, params
                    )
                else:
                    order = self.exchange.create_market_sell_order(
                        symbol, amount, params
                    )
            else:
                if side == "buy":
                    order = self.exchange.create_limit_buy_order(
                        symbol, amount, price, params
                    )
                else:
                    order = self.exchange.create_limit_sell_order(
                        symbol, amount, price, params
                    )
            return order
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def cancel_order(self, symbol, order_id):
        """
        Cancels an order.
        :param symbol: Symbol of the order
        :param order_id: ID of the order to cancel
        :return: Response from the exchange
        """
        try:
            return self.exchange.cancel_order(order_id, symbol)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def fetch_open_positions(self):
        """
        Fetches open positions.
        :return: List of open positions
        """
        try:
            return self.exchange.fetch_positions()
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
