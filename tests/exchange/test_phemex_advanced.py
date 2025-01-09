# tests/exchange/test_phemex_advanced.py

import pytest
import logging
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from common.config.config_loader import load_config
from common.interfaces.phemex import PhemexInterface

class PerformanceMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {}
        
    def record_metric(self, name: str, duration: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(duration)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for name, durations in self.metrics.items():
            summary[name] = {
                'avg': np.mean(durations),
                'min': np.min(durations),
                'max': np.max(durations),
                'count': len(durations)
            }
        return summary

class TestDataGenerator:
    @staticmethod
    def generate_test_orders(symbol: str, count: int = 5) -> List[Dict[str, Any]]:
        """Generate test orders for validation"""
        orders = []
        sides = ['buy', 'sell']
        types = ['limit', 'market']
        
        for _ in range(count):
            orders.append({
                'symbol': symbol,
                'side': np.random.choice(sides),
                'type': np.random.choice(types),
                'amount': round(np.random.uniform(0.001, 0.01), 6),
                'price': round(np.random.uniform(35000, 45000), 2) if types == 'limit' else None
            })
        return orders

@pytest.fixture
def performance_metrics():
    return PerformanceMetrics()

@pytest.fixture
def phemex_client():
    config = load_config("tests/config/test_config.yaml")
    if not config['phemex']['api_key'] or not config['phemex']['api_secret']:
        pytest.skip("Phemex API credentials not configured")
    
    return PhemexInterface(
        api_key=config['phemex']['api_key'],
        api_secret=config['phemex']['api_secret']
    )

def test_personal_trading_data(phemex_client, performance_metrics):
    """Test fetching personal trading data"""
    symbol = "BTC/USD"
    logger = logging.getLogger(__name__)
    
    # Test balance with performance tracking
    start = time.time()
    balance = phemex_client.get_balance()
    performance_metrics.record_metric('balance_fetch', time.time() - start)
    logger.info(f"Current balance: {balance}")
    
    # Fetch open positions
    start = time.time()
    try:
        positions = phemex_client.get_positions(symbol)
        performance_metrics.record_metric('positions_fetch', time.time() - start)
        logger.info(f"Current positions: {positions}")
        
        for position in positions:
            assert 'size' in position, "Position size not found"
            assert 'entryPrice' in position, "Entry price not found"
            logger.info(f"Position Details - Size: {position['size']}, Entry: {position['entryPrice']}")
    except Exception as e:
        logger.error(f"Position fetch error: {str(e)}")
    
    # Fetch recent orders
    start = time.time()
    try:
        recent_orders = phemex_client.fetch_orders(symbol, limit=10)
        performance_metrics.record_metric('orders_fetch', time.time() - start)
        logger.info(f"Recent orders: {recent_orders}")
        
        # Analyze order history
        if recent_orders:
            df_orders = pd.DataFrame(recent_orders)
            logger.info(f"Order statistics:\n{df_orders['status'].value_counts()}")
    except Exception as e:
        logger.error(f"Orders fetch error: {str(e)}")

def test_pre_trade_validation(phemex_client, performance_metrics):
    """Validate potential trades before execution"""
    symbol = "BTC/USD"
    logger = logging.getLogger(__name__)
    
    # Generate test orders
    test_orders = TestDataGenerator.generate_test_orders(symbol)
    
    # Validate each test order
    for order in test_orders:
        start = time.time()
        try:
            # Get current market price
            ticker = phemex_client.get_ticker(symbol)
            current_price = float(ticker['last'])
            
            # Calculate potential profit/loss
            if order['type'] == 'limit':
                price_diff = abs(order['price'] - current_price)
                price_diff_percent = (price_diff / current_price) * 100
                logger.info(f"Order price differs from market by {price_diff_percent:.2f}%")
            
            # Validate order parameters
            validation = phemex_client.validate_order(
                symbol=order['symbol'],
                side=order['side'],
                order_type=order['type'],
                amount=order['amount'],
                price=order['price']
            )
            
            performance_metrics.record_metric('order_validation', time.time() - start)
            logger.info(f"Order validation result: {validation}")
            
            # Check account balance sufficiency
            balance = phemex_client.get_balance()
            required_margin = order['amount'] * current_price * 0.01  # Example margin calculation
            assert float(balance['total']['USD']) >= required_margin, "Insufficient margin"
            
        except Exception as e:
            logger.error(f"Order validation error: {str(e)}")

def test_latency_benchmarks(phemex_client, performance_metrics):
    """Test API endpoint latencies"""
    symbol = "BTC/USD"
    logger = logging.getLogger(__name__)
    
    endpoints = [
        ('ticker', lambda: phemex_client.get_ticker(symbol)),
        ('orderbook', lambda: phemex_client.get_order_book(symbol)),
        ('balance', lambda: phemex_client.get_balance()),
        ('positions', lambda: phemex_client.get_positions(symbol))
    ]
    
    for name, func in endpoints:
        for _ in range(5):  # Test each endpoint 5 times
            start = time.time()
            try:
                func()
                performance_metrics.record_metric(f'{name}_latency', time.time() - start)
            except Exception as e:
                logger.error(f"Benchmark error for {name}: {str(e)}")
            time.sleep(0.2)  # Rate limiting
    
    # Print performance summary
    summary = performance_metrics.get_summary()
    logger.info("\nPerformance Summary:")
    for endpoint, metrics in summary.items():
        logger.info(f"\n{endpoint}:")
        logger.info(f"  Average: {metrics['avg']*1000:.2f}ms")
        logger.info(f"  Min: {metrics['min']*1000:.2f}ms")
        logger.info(f"  Max: {metrics['max']*1000:.2f}ms")
        logger.info(f"  Requests: {metrics['count']}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v"])
