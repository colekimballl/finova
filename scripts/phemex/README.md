# Phemex Trading System

Professional algorithmic trading implementation for the Phemex exchange with modular components for exchange interaction, strategy execution, risk management, and automated trading.

## Directory Structure

phemex/
├── exchange_interface_phemex.py    # Main exchange interface implementation
├── risk_management_phemex.py       # Risk management system
├── phemex_functions.py            # Core trading utilities
├── bot.py                         # Primary trading bot
├── logger.py                      # Logging configuration
└── requirements.txt               # Project dependencies

## Core Components

### Exchange Interface
- CCXT-based Phemex API integration
- Comprehensive order management
- Real-time position tracking
- Market data retrieval
- Robust error handling

### Risk Management
- Dynamic position sizing
- PnL monitoring
- Leverage control
- Emergency position closure
- Account balance monitoring
- Drawdown protection

### Trading Functions
- Order book analysis
- Technical indicators (SMA, RSI)
- Position management
- Real-time PnL tracking
- Market data processing

### Trading Bot Features
- Automated strategy execution
- Real-time market analysis
- Configurable trading parameters
- Risk-adjusted position sizing
- Scheduled execution

## Configuration

1. Environment Setup:
    export PHEMEX_API_KEY="your_api_key"
    export PHEMEX_API_SECRET="your_api_secret"

2. Risk Parameters:
- Maximum position size
- Target profit levels
- Stop loss thresholds
- Leverage settings
- Account balance minimums

## Usage

1. Start the trading bot:
    python bot.py

2. Monitor positions:
    from phemex_functions import open_positions
    positions = open_positions(exchange, symbol)

3. Execute manual trades:
    from exchange_interface_phemex import PhemexClient
    client = PhemexClient(api_key, api_secret)
    client.place_order(symbol, side, order_type, amount, price)

## Safety Features

- Automatic position closure on loss limits
- Account balance monitoring
- Emergency kill switch
- Rate limiting
- Comprehensive error handling
- Detailed logging

## Requirements
- Python 3.9+
- ccxt
- pandas
- pandas-ta
- python-dotenv
- schedule

## Best Practices

1. Test with small positions initially
2. Monitor logs continuously
3. Secure API credentials
4. Regular risk parameter review
5. Keep configuration backups

## Development Standards

1. Code Structure:
- Type hints for all functions
- Comprehensive error handling
- Detailed logging
- Clean function documentation

2. Testing Requirements:
- Unit tests for core functions
- Integration tests for exchange operations
- Strategy backtesting
- Risk management validation

3. Deployment Guidelines:
- Environment variable configuration
- Logging setup verification
- Initial position size limits
- Connection testing

## Contributing

When contributing:
1. Follow existing code structure
2. Maintain comprehensive error handling
3. Include detailed logging
4. Update documentation
5. Add tests for new features

## License

Private repository - All rights reserved
