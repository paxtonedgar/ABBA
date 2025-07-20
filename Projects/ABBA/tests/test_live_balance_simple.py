"""
Simple Live Balance Monitoring Test
Test basic BrowserBase connectivity and DraftKings balance checking
"""

import asyncio
import os
import sys

import structlog

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from balance_monitor import BalanceMonitor
from browserbase_executor import BrowserBaseConfig

logger = structlog.get_logger()


class SimpleLiveBalanceTester:
    """Simple test balance monitoring with live credentials."""

    def __init__(self):
        self.config = None
        self.monitor = None

    async def setup(self):
        """Setup the test environment."""
        try:
            # Check required environment variables
            required_vars = [
                "BROWSERBASE_API_KEY",
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

            # Create BrowserBase config (with optional project ID)
            project_id = os.getenv("BROWSERBASE_PROJECT_ID", "default")
            self.config = BrowserBaseConfig(
                api_key=os.getenv("BROWSERBASE_API_KEY"),
                project_id=project_id,
                stealth_mode=True,
            )

            print(f"✅ BrowserBase config created (Project ID: {project_id})")

            # Create balance monitor
            self.monitor = BalanceMonitor(self.config)
            await self.monitor.__aenter__()

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

    async def test_browserbase_connectivity(self):
        """Test basic BrowserBase connectivity."""
        try:
            print("\n🔌 Testing BrowserBase connectivity...")

            # Test basic session creation
            session = await self.monitor.executor.create_session()

            if session and session.get("sessionId"):
                print("✅ BrowserBase connection successful!")
                print(f"   Session ID: {session['sessionId'][:20]}...")
                return True
            else:
                print("❌ BrowserBase connection failed")
                return False

        except Exception as e:
            print(f"❌ Error testing BrowserBase connectivity: {e}")
            return False

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

    async def test_simple_navigation(self):
        """Test simple navigation to DraftKings."""
        try:
            print("\n🧭 Testing navigation to DraftKings...")

            # Navigate to DraftKings homepage
            result = await self.monitor.executor.navigate("https://www.draftkings.com")

            if result and result.get("success"):
                print("✅ Navigation to DraftKings successful!")
                return True
            else:
                print("❌ Navigation to DraftKings failed")
                return False

        except Exception as e:
            print(f"❌ Error testing navigation: {e}")
            return False


async def main():
    """Main test function."""
    print("🧪 Simple Live Balance Monitoring Test")
    print("=" * 50)
    print("This will test basic BrowserBase connectivity and DraftKings balance.")
    print("Make sure you have set up your environment variables first.")
    print()

    # Confirm before proceeding
    confirm = (
        input("Do you want to proceed with live testing? (yes/no): ").strip().lower()
    )
    if confirm != "yes":
        print("Test cancelled.")
        return

    tester = SimpleLiveBalanceTester()

    try:
        # Setup
        if not await tester.setup():
            print("❌ Setup failed. Please check your configuration.")
            return

        # Run tests
        print("\n" + "=" * 50)
        print("RUNNING TESTS")
        print("=" * 50)

        # Test 1: BrowserBase connectivity
        connectivity_ok = await tester.test_browserbase_connectivity()

        # Test 2: Simple navigation
        navigation_ok = await tester.test_simple_navigation()

        # Test 3: Balance checking (only if previous tests pass)
        balance_info = None
        if connectivity_ok and navigation_ok:
            balance_info = await tester.test_draftkings_balance()

        # Summary
        print("\n" + "=" * 50)
        print("TEST RESULTS")
        print("=" * 50)

        if connectivity_ok:
            print("✅ BrowserBase connectivity: PASSED")
        else:
            print("❌ BrowserBase connectivity: FAILED")

        if navigation_ok:
            print("✅ Navigation: PASSED")
        else:
            print("❌ Navigation: FAILED")

        if balance_info:
            print("✅ Balance checking: PASSED")
        else:
            print("❌ Balance checking: FAILED")

        print("\n🎉 Simple live testing completed!")

        if balance_info:
            print(f"\n📈 Your DraftKings balance: ${balance_info.current_balance}")
            if balance_info.status.value in ["low", "critical", "depleted"]:
                print("⚠️  Consider adding funds to your account.")

        # Recommendations
        print("\n📋 Recommendations:")
        if not connectivity_ok:
            print("   - Check your BrowserBase API key")
            print("   - Verify your BrowserBase account is active")
        if not navigation_ok:
            print("   - Check your internet connection")
            print("   - Verify DraftKings website is accessible")
        if not balance_info:
            print("   - Check your DraftKings credentials")
            print("   - Verify your account is not restricted")

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
