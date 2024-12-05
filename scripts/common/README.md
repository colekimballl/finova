# Common Utilities

This directory contains utility scripts and modules that are shared across different exchanges and components.

## Modules

- `utils.py`: General utility functions, such as fetching ask/bid prices, handling time conversions, etc.
- `risk_management.py`: Risk management functions that are exchange-agnostic.
- `technical_indicators.py`: Functions for calculating technical indicators like VWAP, SMA, RSI, etc.

## Usage

- Import these modules into your scripts as needed.
- Example:
  ```python
  from common.utils import ask_bid

**Importing Utility Functions:**

```python
from scripts.common.utils import ask_bid

from scripts.common.technical_indicators import calculate_vwap


---

### 9. `/scripts/hyperliquid/README.md`

```markdown
# Hyperliquid Exchange Scripts

This directory contains scripts and modules specific to interacting with the Hyperliquid exchange.

## Modules

- `exchange_interface_hyperliquid.py`: Interface for interacting with Hyperliquid's API, including placing orders and fetching data.
- `risk_management_hyperliquid.py`: Risk management functions tailored to Hyperliquid's API and data structures.

## Usage

- Import these modules when working with Hyperliquid.
- Ensure that your API keys for Hyperliquid are set in your configuration file.


