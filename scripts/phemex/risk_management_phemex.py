# risk_management_phemex.py

from dataclasses import dataclass
from typing import Optional
from logger import setup_logger
from phemex_functions import pnl_close, kill_switch, acct_bal
from exchange_interface_phemex import PhemexClient

logger = setup_logger("risk_management_phemex")

@dataclass
class RiskParameters:
    max_position_size: float
    max_drawdown: float
    leverage: float
    # Add other risk parameters as needed

class RiskManager:
    """
    Handles risk management for Phemex trading operations.
    """

    def __init__(self, params: RiskParameters, exchange: PhemexClient, symbol: str):
        """
        Initializes the RiskManager with risk parameters.

        :param params: RiskParameters instance containing risk settings.
        :param exchange: Instance of PhemexClient.
        :param symbol: Trading pair.
        """
        self.params = params
        self.exchange = exchange
        self.symbol = symbol
        self.logger = logger

    def validate_order(self, current_position: float, order_size: float) -> bool:
        """
        Validates if an order adheres to risk parameters.

        :param current_position: Current position size.
        :param order_size: Size of the new order.
        :return: True if order is valid, False otherwise.
        """
        if (current_position + order_size) > self.params.max_position_size:
            self.logger.warning("Order size exceeds maximum position size.")
            return False
        # Add other validation checks as needed
        return True

    def monitor_pnl(self):
        """
        Monitors PnL and triggers closure if targets or limits are hit.
        """
        pnl_triggered, in_pos, size, is_long = pnl_close(
            self.exchange, self.symbol, self.params.max_drawdown, self.params.max_drawdown
        )
        if pnl_triggered:
            self.logger.info("PnL target or max loss hit. Initiating kill switch.")
            kill_switch(self.exchange, self.symbol, {"timeInForce": "PostOnly"})

    def check_account_balance(self):
        """
        Checks if the account balance is above the minimum threshold.
        """
        current_balance = acct_bal(self.exchange, self.symbol)
        if current_balance < self.params.max_drawdown:
            self.logger.warning(f"Account balance {current_balance} below minimum threshold.")
            kill_switch(self.exchange, self.symbol, {"timeInForce": "PostOnly"})

