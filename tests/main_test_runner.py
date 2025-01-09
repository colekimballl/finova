# tests/main_test_runner.py

import sys
import time
import pytest
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

@dataclass
class TestResult:
    name: str
    success: bool
    duration: float
    error_message: str = ""

class TestSuite:
    def __init__(self, name: str, test_files: List[str]):
        self.name = name
        self.test_files = test_files
        self.results: List[TestResult] = []

class TradingTestRunner:
    def __init__(self):
        self.console = Console()
        self.test_suites = {
            "connectivity": TestSuite("Connectivity", [
                "tests/exchange/test_connectivity.py",
                "tests/exchange/test_environment.py"
            ]),
            "account": TestSuite("Account", [
                "tests/exchange/test_fetch_balance.py"
            ]),
            "trading": TestSuite("Trading", [
                "tests/exchange/test_order_book.py",
                "tests/exchange/test_trade.py"
            ]),
            "performance": TestSuite("Performance", [
                "tests/exchange/test_phemex_advanced.py"
            ])
        }
        self.functionality_map = {
            "Basic Connectivity": {
                "Phemex API Connection": ["test_connectivity"],
                "Market Data Access": ["test_connectivity", "test_order_book"],
                "Authentication": ["test_connectivity"]
            },
            "Account Operations": {
                "Balance Fetching": ["test_fetch_balance"],
                "Position Information": ["test_phemex_advanced"],
                "Order History": ["test_phemex_advanced"]
            },
            "Trading Operations": {
                "Order Validation": ["test_phemex_advanced"],
                "Order Book Access": ["test_order_book"],
                "Trade Execution": ["test_trade"]
            },
            "Performance": {
                "API Latency": ["test_phemex_advanced"],
                "Response Times": ["test_phemex_advanced"],
                "Error Handling": ["test_phemex_advanced"]
            }
        }

    def run_tests(self) -> None:
        """Execute all test suites"""
        start_time = time.time()
        self.console.print("\n[bold blue]Starting Trading System Tests...[/bold blue]")

        for suite_name, suite in self.test_suites.items():
            self._run_test_suite(suite)

        self._print_summary(time.time() - start_time)

    def _run_test_suite(self, suite: TestSuite) -> None:
        """Run a single test suite"""
        self.console.print(f"\n[bold cyan]Running {suite.name} Tests[/bold cyan]")
        
        for test_file in suite.test_files:
            if not Path(test_file).exists():
                suite.results.append(
                    TestResult(
                        Path(test_file).stem,
                        False,
                        0.0,
                        "Test file not found"
                    )
                )
                continue

            start_time = time.time()
            try:
                result = pytest.main(["-v", test_file])
                duration = time.time() - start_time
                suite.results.append(
                    TestResult(
                        Path(test_file).stem,
                        result == 0,
                        duration
                    )
                )
            except Exception as e:
                suite.results.append(
                    TestResult(
                        Path(test_file).stem,
                        False,
                        time.time() - start_time,
                        str(e)
                    )
                )

    def _print_summary(self, total_duration: float) -> None:
        """Print detailed test results summary"""
        self._print_results_table()
        self._print_functionality_checklist()
        self._print_statistics(total_duration)

    def _print_results_table(self) -> None:
        """Print test results in a formatted table"""
        table = Table(title="Test Results")
        table.add_column("Category", style="cyan")
        table.add_column("Test", style="magenta")
        table.add_column("Status", justify="center")
        table.add_column("Duration", justify="right")
        table.add_column("Error", style="red")

        for suite_name, suite in self.test_suites.items():
            for result in suite.results:
                table.add_row(
                    suite.name,
                    result.name,
                    "✅" if result.success else "❌",
                    f"{result.duration:.2f}s",
                    result.error_message
                )

        self.console.print("\n", table)

    def _check_functionality(self, category: str, item: str) -> bool:
        """Check if specific functionality tests passed"""
        if category not in self.functionality_map or item not in self.functionality_map[category]:
            return False

        required_tests = self.functionality_map[category][item]
        for suite in self.test_suites.values():
            for result in suite.results:
                if result.name in required_tests and not result.success:
                    return False
        return True

    def _print_functionality_checklist(self) -> None:
        """Print functionality checklist"""
        table = Table(title="Functionality Checklist")
        table.add_column("Category", style="cyan")
        table.add_column("Feature", style="magenta")
        table.add_column("Status", justify="center")

        for category, items in self.functionality_map.items():
            for item in items:
                status = "✅" if self._check_functionality(category, item) else "❌"
                table.add_row(category, item, status)

        self.console.print("\n", table)

    def _print_statistics(self, total_duration: float) -> None:
        """Print overall test statistics"""
        total_tests = sum(len(suite.results) for suite in self.test_suites.values())
        passed_tests = sum(
            sum(1 for result in suite.results if result.success)
            for suite in self.test_suites.values()
        )

        stats_table = Table(title="Test Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="magenta")

        stats_table.add_row("Total Tests", str(total_tests))
        stats_table.add_row("Passed Tests", str(passed_tests))
        stats_table.add_row("Failed Tests", str(total_tests - passed_tests))
        stats_table.add_row("Success Rate", f"{(passed_tests/total_tests)*100:.1f}%")
        stats_table.add_row("Total Duration", f"{total_duration:.2f}s")

        self.console.print("\n", stats_table)

def main():
    runner = TradingTestRunner()
    runner.run_tests()

if __name__ == "__main__":
    main()
