# common/api_client.py

import ccxt
import logging
from typing import Optional, Dict, Any


class ExchangeClient:
    """
    Base API Client for interacting with exchanges using CCXT.
    """

    def __init__(self, exchange_name: str, api_key: str, secret: str):
        """
        Initializes the ExchangeClient with exchange credentials.

        Parameters:
        - exchange_name (str): Name of the exchange (e.g., 'phemex').
        - api_key (str): API key for the exchange.
        - secret (str): API secret for the exchange.
        """
        self.exchange = getattr(ccxt, exchange_name)(
            {
                "apiKey": api_key,
                "secret": secret,
                "enableRateLimit": True,
            }
        )
        self.logger = logging.getLogger(exchange_name)
        self.connected = False

    def connect(self) -> bool:
        """
        Connects to the exchange by loading markets.

        Returns:
        - bool: True if connected successfully, False otherwise.
        """
        try:
            self.exchange.load_markets()
            self.connected = True
            self.logger.info(f"Connected to {self.exchange.name} successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to {self.exchange.name}: {e}")
            return False

    def disconnect(self):
        """
        Disconnects from the exchange. (CCXT does not require explicit disconnection.)
        """
        self.connected = False
        self.logger.info(f"Disconnected from {self.exchange.name}.")

    def fetch_order_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetches the order book for a given symbol.

        Parameters:
        - symbol (str): Trading pair symbol (e.g., 'BTC/USD').

        Returns:
        - Optional[Dict[str, Any]]: Order book data if successful, None otherwise.
        """
        try:
            order_book = self.exchange.fetch_order_book(symbol)
            self.logger.debug(f"Fetched order book for {symbol}.")
            return order_book
        except Exception as e:
            self.logger.error(f"Error fetching order book for {symbol}: {e}")
            return None

    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        params: Dict[str, Any] = {},
    ) -> Optional[Dict[str, Any]]:
        """
        Creates an order on the exchange.

        Parameters:
        - symbol (str): Trading pair symbol.
        - side (str): 'buy' or 'sell'.
        - order_type (str): 'limit' or 'market'.
        - amount (float): Amount to trade.
        - price (Optional[float]): Price for limit orders.
        - params (Dict[str, Any]): Additional parameters.

        Returns:
        - Optional[Dict[str, Any]]: Order details if successful, None otherwise.
        """
        try:
            if order_type.lower() == "limit":
                if price is None:
                    raise ValueError("Price must be specified for limit orders.")
                order = self.exchange.create_limit_order(symbol, side, amount, price, params)
            elif order_type.lower() == "market":
                order = self.exchange.create_market_order(symbol, side, amount, params)
            else:
                self.logger.error(f"Unsupported order type: {order_type}")
                return None
            self.logger.info(f"Created {side} {order_type} order: {order}")
            return order
        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            return None

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancels a specific order.

        Parameters:
        - symbol (str): Trading pair symbol.
        - order_id (str): ID of the order to cancel.

        Returns:
        - bool: True if canceled successfully, False otherwise.
        """
        try:
            self.exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Canceled order {order_id} for {symbol}.")
            return True
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {e}")
            return False

    def fetch_balance(self) -> Optional[Dict[str, Any]]:
        """
        Fetches the account balance.

        Returns:
        - Optional[Dict[str, Any]]: Balance information if successful, None otherwise.
        """
        try:
            balance = self.exchange.fetch_balance()
            self.logger.debug("Fetched account balance.")
            return balance
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return None

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Sets leverage for a given symbol.

        Parameters:
        - symbol (str): Trading pair symbol.
        - leverage (int): Desired leverage.

        Returns:
        - bool: True if leverage set successfully, False otherwise.
        """
        try:
            self.exchange.set_leverage(leverage, symbol)
            self.logger.info(f"Set leverage to {leverage}x for {symbol}.")
            return True
        except Exception as e:
            self.logger.error(f"Error setting leverage for {symbol}: {e}")
            return False

    def fetch_positions(self) -> Optional[Dict[str, Any]]:
        """
        Fetches all open positions.

        Returns:
        - Optional[Dict[str, Any]]: Positions data if successful, None otherwise.
        """
        try:
            positions = self.exchange.fetch_positions()
            self.logger.debug("Fetched open positions.")
            return positions
        except Exception as e:
            self.logger.error(f"Error fetching positions: {e}")
            return None

    # Add more methods as needed for extended functionality

