"""
Live Balance Monitoring Test
Safely test the balance monitoring system with your DraftKings account
"""

import asyncio
import os
import sys
from decimal import Decimal

import structlog

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from balance_monitor import BalanceMonitor, FundManager
from browserbase_executor import BrowserBaseConfig

logger = structlog.get_logger()


class LiveBalanceTester:
    """Test balance monitoring with live credentials."""

    def __init__(self):
        self.config = None
        self.monitor = None
        self.fund_manager = None

    async def setup(self):
        """Setup the test environment."""
        try:
            # Check required environment variables
            required_vars = [
                "BROWSERBASE_API_KEY",
                "BROWSERBASE_PROJECT_ID",
                "DRAFTKINGS_USERNAME",
                "DRAFTKINGS_PASSWORD",
            ]

            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                print(f"❌ Missing environment variables: {missing_vars}")
                print("\nPlease set these environment variables:")
                for var in missing_vars:
                    print(f"export {var}='your_value'")
                return False

            print("✅ All required environment variables found")

            # Create BrowserBase config
            self.config = BrowserBaseConfig(
                api_key=os.getenv("BROWSERBASE_API_KEY"),
                project_id=os.getenv("BROWSERBASE_PROJECT_ID"),
                stealth_mode=True,
            )

            # Create balance monitor
            self.monitor = BalanceMonitor(self.config)
            await self.monitor.__aenter__()

            # Create fund manager
            self.fund_manager = FundManager(self.monitor)

            print("✅ Test environment setup completed")
            return True

        except Exception as e:
            print(f"❌ Error setting up test environment: {e}")
            return False

    async def teardown(self):
        """Cleanup test environment."""
        try:
            if self.monitor:
                await self.monitor.__aexit__(None, None, None)
            print("✅ Test environment cleanup completed")
        except Exception as e:
            print(f"❌ Error during teardown: {e}")

    async def test_draftkings_balance(self):
        """Test DraftKings balance checking."""
        try:
            print("\n🔍 Testing DraftKings balance checking...")

            # Get credentials
            credentials = {
                "username": os.getenv("DRAFTKINGS_USERNAME"),
                "password": os.getenv("DRAFTKINGS_PASSWORD"),
            }

            # Check balance
            balance_info = await self.monitor.check_balance("draftkings", credentials)

            if balance_info:
                print("✅ Balance check successful!")
                print(f"   Platform: {balance_info.platform}")
                print(f"   Current Balance: ${balance_info.current_balance}")
                print(f"   Status: {balance_info.status.value}")
                print(f"   Last Updated: {balance_info.last_updated}")

                if balance_info.threshold_warnings:
                    print(f"   Warnings: {balance_info.threshold_warnings}")

                return balance_info
            else:
                print("❌ Balance check failed")
                return None

        except Exception as e:
            print(f"❌ Error testing DraftKings balance: {e}")
            return None

    async def test_fund_request(self):
        """Test fund request functionality."""
        try:
            print("\n💰 Testing fund request functionality...")

            # Create a test fund request
            request_id = await self.fund_manager.interface.request_funds(
                platform="draftkings",
                amount=Decimal("100.00"),
                reason="Test fund request",
            )

            print(f"✅ Fund request created: {request_id}")

            # Get pending requests
            requests = await self.fund_manager.interface.get_fund_requests()

            print(f"   Pending requests: {len(requests)}")
            for request in requests:
                print(
                    f"   - {request['request_id']}: ${request['amount']} ({request['status']})"
                )

            return request_id

        except Exception as e:
            print(f"❌ Error testing fund request: {e}")
            return None

    async def test_balance_summary(self):
        """Test balance summary functionality."""
        try:
            print("\n📊 Testing balance summary...")

            summary = await self.fund_manager.interface.get_balance_summary(
                "draftkings"
            )

            if "error" in summary:
                print(f"❌ Error getting balance summary: {summary['error']}")
                return None

            print("✅ Balance summary retrieved:")
            print(f"   Platform: {summary['platform']}")
            print(f"   Balance: ${summary['balance']['current_balance']}")
            print(f"   Status: {summary['balance']['status']}")
            print(f"   Pending Requests: {len(summary['pending_requests'])}")

            return summary

        except Exception as e:
            print(f"❌ Error testing balance summary: {e}")
            return None


async def main():
    """Main test function."""
    print("🧪 Live Balance Monitoring Test")
    print("=" * 50)
    print("This will test the balance monitoring system with your DraftKings account.")
    print("Make sure you have set up your environment variables first.")
    print()

    # Confirm before proceeding
    confirm = (
        input("Do you want to proceed with live testing? (yes/no): ").strip().lower()
    )
    if confirm != "yes":
        print("Test cancelled.")
        return

    tester = LiveBalanceTester()

    try:
        # Setup
        if not await tester.setup():
            print("❌ Setup failed. Please check your configuration.")
            return

        # Run tests
        print("\n" + "=" * 50)
        print("RUNNING TESTS")
        print("=" * 50)

        # Test 1: Balance checking
        balance_info = await tester.test_draftkings_balance()

        # Test 2: Fund request
        request_id = await tester.test_fund_request()

        # Test 3: Balance summary
        summary = await tester.test_balance_summary()

        # Summary
        print("\n" + "=" * 50)
        print("TEST RESULTS")
        print("=" * 50)

        if balance_info:
            print("✅ Balance checking: PASSED")
        else:
            print("❌ Balance checking: FAILED")

        if request_id:
            print("✅ Fund request: PASSED")
        else:
            print("❌ Fund request: FAILED")

        if summary:
            print("✅ Balance summary: PASSED")
        else:
            print("❌ Balance summary: FAILED")

        print("\n🎉 Live testing completed!")

        if balance_info:
            print(f"\n📈 Your DraftKings balance: ${balance_info.current_balance}")
            if balance_info.status.value in ["low", "critical", "depleted"]:
                print("⚠️  Consider adding funds to your account.")

    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")

    finally:
        # Cleanup
        await tester.teardown()


if __name__ == "__main__":
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Run test
    asyncio.run(main())
