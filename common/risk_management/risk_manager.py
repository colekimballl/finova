# common/risk_management/risk_manager.py

from typing import Optional
import logging
from dataclasses import dataclass
from ..interfaces.base_interface import PositionInfo
from ..interfaces.phemex import PhemexInterface  # Import specific interface


@dataclass
class RiskParameters:
    max_position_size: float  # Maximum allowable position size in USD
    max_drawdown: float       # Maximum allowable loss in percentage
    leverage: float           # Trading leverage


class RiskManager:
    """
    Handles risk management for trading operations.
    """

    def __init__(self, exchange: PhemexInterface, params: RiskParameters):
        """
        Initializes the RiskManager with exchange interface and risk parameters.

        Parameters:
        - exchange (PhemexInterface): Instance of the exchange interface.
        - params (RiskParameters): Risk management parameters.
        """
        self.exchange = exchange
        self.params = params
        self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate_position(self, position: PositionInfo) -> bool:
        """
        Evaluates the current position against risk parameters.

        Parameters:
        - position (PositionInfo): Current open position.

        Returns:
        - bool: True if position is within risk parameters, False otherwise.
        """
        position_value = self.exchange.calculate_position_value(position)
        liquidation_risk = self.exchange.calculate_liquidation_risk(position)

        self.logger.info(f"Evaluating Position - Value: {position_value}, Liquidation Risk: {liquidation_risk}%")

        if position_value > self.params.max_position_size:
            self.logger.warning("Position size exceeds maximum allowed. Initiating risk mitigation.")
            self._mitigate_risk(position)
            return False

        if liquidation_risk < self.params.max_drawdown:
            self.logger.warning("Position is too close to liquidation. Initiating risk mitigation.")
            self._mitigate_risk(position)
            return False

        self.logger.info("Position is within risk parameters.")
        return True

    def _mitigate_risk(self, position: PositionInfo):
        """
        Mitigates risk by closing the position.

        Parameters:
        - position (PositionInfo): Current open position.
        """
        self.logger.info(f"Mitigating risk for position: {position}")
        success = self.exchange.cancel_all_orders(position.symbol)
        if success:
            self.logger.info("All open orders canceled successfully.")
        else:
            self.logger.error("Failed to cancel open orders.")

        # Close the position
        side = "sell" if position.side.lower() == "buy" else "buy"
        self.logger.info(f"Closing position by placing a {side} order.")
        close_order = self.exchange.create_order(
            symbol=position.symbol,
            side=side,
            order_type="market",
            size=position.size
        )
        if close_order:
            self.logger.info(f"Position closed successfully: {close_order}")
        else:
            self.logger.error("Failed to close the position.")

