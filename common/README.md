# Common Trading System Components

Core trading system infrastructure providing standardized interfaces, utilities, and shared functionality for algorithmic trading operations.

## Directory Structure

```
common/
├── interfaces/               # Exchange interface implementations
│   ├── base.py             # Base exchange interface
│   ├── phemex.py           # Phemex implementation
│   └── hyperliquid.py      # Hyperliquid implementation
├── indicators/              # Technical analysis indicators
│   ├── base_indicator.py   # Base indicator class
│   ├── ta_lib_indicators.py    # TA-Lib implementations
│   └── pandas_ta_indicators.py # Pandas-TA implementations
├── risk_management/         # Risk management system
│   └── risk_manager.py     # Risk control implementation
├── config/                  # Configuration management
│   └── config_loader.py    # Configuration loading utilities
├── logger/                  # Logging system
│   └── logger.py           # Centralized logging
├── utils/                   # Utility functions
│   └── utils.py            # Common utilities
└── technical_analysis/      # Advanced technical analysis
    └── ta_lib_indicators.py # TA-Lib indicator suite
```

## Core Components

### Exchange Interfaces
- Standardized exchange interaction
- Order management
- Position tracking
- Market data access
- Common error handling

### Technical Indicators
- Base indicator framework
- TA-Lib integration
- Pandas-TA implementation
- Real-time calculation support
- Historical data analysis

### Risk Management
- Position sizing
- Drawdown monitoring
- Leverage control
- Emergency position closure
- Account balance monitoring

### Configuration System
- YAML-based configuration
- Environment variable integration
- Secure credential management
- Interactive configuration support

### Logging System
- Centralized logging
- Rotating file handlers
- Console output
- Comprehensive error tracking

## Usage

### Exchange Interface
```python
from common.interfaces import PhemexInterface

interface = PhemexInterface(api_key="your_key", api_secret="your_secret")
position = interface.get_position("BTC/USD")
```

### Risk Management
```python
from common.risk_management import RiskManager, RiskParameters

risk_params = RiskParameters(
    max_position_size=1000.0,
    max_drawdown=15.0,
    leverage=3.0
)
risk_manager = RiskManager(exchange=interface, params=risk_params)
```

### Technical Indicators
```python
from common.indicators import IndicatorManager
from common.indicators.ta_lib_indicators import TA_LibSMA

indicator_manager = IndicatorManager()
indicator_manager.add_indicator(TA_LibSMA(period=20))
```

## Best Practices

1. Error Handling
- Always use try-except blocks
- Log errors appropriately
- Implement graceful fallbacks

2. Configuration
- Use environment variables for credentials
- Keep sensitive data out of code
- Validate all configurations

3. Logging
- Use appropriate log levels
- Include relevant context
- Rotate logs regularly

4. Testing
- Unit test core functionality
- Test exchange integrations
- Validate risk parameters

## Dependencies

Core Requirements:
- Python 3.9+
- ccxt
- pandas
- pandas-ta
- ta-lib
- pyyaml
- python-dotenv

## Contributing

When adding new components:
1. Follow existing patterns
2. Add comprehensive error handling
3. Include detailed logging
4. Update documentation
5. Add unit tests

## Safety Notes

1. Always test with small positions
2. Verify risk parameters
3. Monitor logs regularly
4. Keep credentials secure
5. Test in testnet first

## License

Private repository - All rights reserved
