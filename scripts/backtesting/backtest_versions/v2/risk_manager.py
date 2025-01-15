# backtest_versions/v2/risk_manager.py

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class RiskConfig:
    max_position_size: float = 1.0
    stop_loss_pct: float = 0.02  # 2% stop loss
    trailing_stop_pct: float = 0.03  # 3% trailing stop
    vol_lookback: int = 20
    vol_target: float = 0.15  # 15% annualized volatility target
    max_drawdown_limit: float = 0.20  # 20% max drawdown limit
    position_timeout: int = 5  # days to hold position
    min_trades_per_month: int = 5

class RiskManager:
    def __init__(self, config: RiskConfig = None):
        """Initialize risk manager with configuration"""
        self.config = config if config else RiskConfig()
        self.positions = {}
        self.high_watermarks = {}
        
    def calculate_position_size(self, 
                              symbol: str,
                              price_data: pd.DataFrame,
                              prediction: float,
                              confidence: float) -> float:
        """Calculate position size based on volatility and prediction confidence"""
        # Calculate volatility-based scaling
        returns = price_data['close'].pct_change()
        vol = returns.rolling(self.config.vol_lookback).std() * np.sqrt(252)
        vol_scale = self.config.vol_target / (vol.iloc[-1] + 1e-6)
        
        # Scale by prediction confidence
        confidence_scale = abs(prediction - 0.5) * 2
        
        # Calculate base position size
        base_size = np.sign(prediction - 0.5) * min(
            vol_scale * confidence_scale,
            self.config.max_position_size
        )
        
        # Apply drawdown control
        equity_curve = (1 + returns).cumprod()
        current_drawdown = 1 - equity_curve.iloc[-1] / equity_curve.cummax().iloc[-1]
        
        if current_drawdown > self.config.max_drawdown_limit:
            base_size *= (1 - current_drawdown/self.config.max_drawdown_limit)
            
        return base_size
        
    def check_stops(self, 
                   symbol: str, 
                   current_price: float, 
                   position: float) -> Tuple[float, bool]:
        """Check and apply stop loss and trailing stop rules"""
        if symbol not in self.positions:
            self.positions[symbol] = {
                'entry_price': current_price,
                'high_watermark': current_price
            }
            return position, False
            
        pos_data = self.positions[symbol]
        
        # Update high watermark for trailing stop
        if current_price > pos_data['high_watermark']:
            pos_data['high_watermark'] = current_price
            
        # Check stop loss
        loss_pct = (current_price - pos_data['entry_price']) / pos_data['entry_price']
        if abs(loss_pct) > self.config.stop_loss_pct:
            return 0, True
            
        # Check trailing stop
        trail_pct = (current_price - pos_data['high_watermark']) / pos_data['high_watermark']
        if abs(trail_pct) > self.config.trailing_stop_pct:
            return 0, True
            
        return position, False
        
    def apply_filters(self,
                     symbol: str,
                     prediction: float,
                     volume: float,
                     avg_volume: float) -> bool:
        """Apply additional trading filters"""
        # Volume filter
        if volume < avg_volume * 0.5:  # Require at least 50% of average volume
            return False
            
        # Prediction confidence filter
        if abs(prediction - 0.5) < 0.2:  # Require strong signals
            return False
            
        return True

    def reset(self):
        """Reset risk manager state"""
        self.positions = {}
        self.high_watermarks = {}
