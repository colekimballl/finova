# tests/exchange/hyperliquid/conftest.py
import pytest
import os
from eth_account import Account
from eth_account.signers.local import LocalAccount
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from dotenv import load_dotenv
from termcolor import cprint

load_dotenv()  # Load environment variables from .env

# Global test results tracker
test_results = {"passed": 0, "failed": 0}

@pytest.fixture(scope="session")
def private_key():
    """Fixture for private key"""
    key = os.getenv("HYPERLIQUID_API_SECRET")
    if not key:
        pytest.skip("HYPERLIQUID_API_SECRET not set in environment")
    return key

@pytest.fixture(scope="session")
def account(private_key) -> LocalAccount:
    """Fixture for Hyperliquid account"""
    return Account.from_key(private_key)

@pytest.fixture(scope="session")
def exchange(account) -> Exchange:
    """Fixture for Hyperliquid exchange"""
    return Exchange(account, constants.MAINNET_API_URL)

@pytest.fixture(scope="session")
def info() -> Info:
    """Fixture for Hyperliquid info"""
    return Info(constants.MAINNET_API_URL, skip_ws=True)

def print_test_summary():
    """Print colorful test summary"""
    cprint("\n============================", "blue", attrs=["bold"])
    cprint("🔍 Hyperliquid Test Summary", "blue", attrs=["bold"])
    cprint("============================", "blue", attrs=["bold"])
    
    cprint(f"\n✅ Passed Tests: {test_results['passed']}", "green")
    if test_results['failed'] == 0:
        cprint("❌ Failed Tests: 0", "green")
        cprint("\n🎉 All tests completed successfully!", "green", attrs=["bold"])
    else:
        cprint(f"❌ Failed Tests: {test_results['failed']}", "red")
        cprint("\n⚠️  Some tests failed. Please check the logs.", "yellow", attrs=["bold"])

def pytest_runtest_logreport(report):
    """Collect test results"""
    if report.when == "call":
        if report.passed:
            test_results["passed"] += 1
        elif report.failed:
            test_results["failed"] += 1

def pytest_sessionfinish(session):
    """Print final summary at the end of test session"""
    print_test_summary()
