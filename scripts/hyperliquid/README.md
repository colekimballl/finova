# Hyperliquid Trading System

Advanced algorithmic trading implementation for the Hyperliquid exchange, featuring modular components for exchange interaction, strategy execution, and risk management.

## Directory Structure

hyperliquid/
├── exchange_interface_hyperliquid.py  # Main exchange interface
├── risk_management_hyperliquid.py     # Risk management system
├── hyperliquid_functions.py           # Core trading functions
├── bollinger_bot.py                   # Bollinger Bands strategy
└── logger.py                          # Logging configuration

## Core Components

### Exchange Interface
- Standardized interaction with Hyperliquid API
- Order management (creation, cancellation)
- Position tracking
- Market data retrieval

### Risk Management
- Position size calculation
- Drawdown monitoring
- Leverage management
- Emergency position closure
- Account balance monitoring

### Trading Functions
- Order book management
- Price data processing
- Technical indicators
- Position management
- PnL tracking

### Strategy Implementation
The Bollinger Bands strategy bot features:
- Real-time market analysis
- Dynamic position sizing
- Automated entry/exit signals
- Risk-adjusted order placement

## Configuration

1. Environment Setup:
    export HYPERLIQUID_PRIVATE_KEY="your_private_key"

2. Risk Parameters:
- Maximum position size
- Target profit levels
- Stop loss limits
- Leverage settings

## Usage

1. Start the Bollinger Bands bot:
    python bollinger_bot.py

2. Monitor positions:
    from hyperliquid_functions import get_position
    position_info = get_position(symbol, account)

3. Execute manual trades:
    from hyperliquid_functions import limit_order
    limit_order(symbol, is_buy, size, price, reduce_only, account)

## Safety Features

- Automatic position closure on predefined loss limits
- Account balance monitoring
- Emergency kill switch functionality
- Rate limiting and error handling
- Comprehensive logging

## Requirements
- Python 3.9+
- eth-account
- pandas
- pandas-ta
- requests
- hyperliquid-py

## Best Practices

1. Always test with small positions first
2. Monitor logs regularly
3. Keep private keys secure
4. Regularly check risk parameters
5. Maintain backup of configuration

## Contributing

When contributing:
1. Follow the existing code structure
2. Add comprehensive error handling
3. Include logging for important events
4. Update tests where appropriate
5. Document any new features

## License

Private repository - All rights reserved
