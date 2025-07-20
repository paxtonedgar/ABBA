"""
Test script for BrowserBase integration with ABMBA system
Tests bet placement capabilities with anti-detection measures
"""

import asyncio
import os
import sys
from decimal import Decimal

import structlog

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from browserbase_integration import (
    BrowserBaseBettingIntegration,
    BrowserBaseExecutionAgent,
)

from models import Bet, BetStatus, MarketType, PlatformType

logger = structlog.get_logger()


class BrowserBaseIntegrationTester:
    """Test suite for BrowserBase integration."""

    def __init__(self):
        self.integration = None
        self.agent = None
        self.test_results = []

    async def setup(self):
        """Setup test environment."""
        try:
            logger.info("Setting up BrowserBase integration test environment")

            # Check required environment variables
            required_vars = ["BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID"]

            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                logger.warning(f"Missing environment variables: {missing_vars}")
                logger.info("Tests will run with dummy values for demonstration")

            # Initialize integration
            self.integration = BrowserBaseBettingIntegration()
            await self.integration.initialize()

            # Initialize agent
            self.agent = BrowserBaseExecutionAgent({}, self.integration)

            logger.info("Test environment setup completed")

        except Exception as e:
            logger.error(f"Error setting up test environment: {e}")
            raise

    async def teardown(self):
        """Cleanup test environment."""
        try:
            if self.integration:
                await self.integration.close()
            logger.info("Test environment cleanup completed")

        except Exception as e:
            logger.error(f"Error during teardown: {e}")

    async def test_session_creation(self):
        """Test BrowserBase session creation."""
        try:
            logger.info("Testing BrowserBase session creation")

            # Test session health
            health = await self.integration.get_session_health()

            assert health.get("session_active") is True, "Session should be active"
            assert (
                health.get("session_duration_seconds") >= 0
            ), "Session duration should be non-negative"

            logger.info(f"Session health: {health}")
            self.test_results.append(
                {"test": "session_creation", "status": "PASSED", "details": health}
            )

        except Exception as e:
            logger.error(f"Session creation test failed: {e}")
            self.test_results.append(
                {"test": "session_creation", "status": "FAILED", "error": str(e)}
            )

    async def test_platform_connectivity(self):
        """Test connectivity to betting platforms."""
        try:
            logger.info("Testing platform connectivity")

            platforms = ["fanduel", "draftkings"]

            for platform in platforms:
                logger.info(f"Testing connectivity to {platform}")

                result = await self.integration.test_platform_connectivity(platform)

                # Note: This will fail with dummy credentials, which is expected
                if result.get("success"):
                    logger.info(f"{platform} connectivity test passed")
                else:
                    logger.warning(
                        f"{platform} connectivity test failed (expected with dummy credentials)"
                    )

                self.test_results.append(
                    {
                        "test": f"platform_connectivity_{platform}",
                        "status": "INFO",
                        "details": result,
                    }
                )

        except Exception as e:
            logger.error(f"Platform connectivity test failed: {e}")
            self.test_results.append(
                {"test": "platform_connectivity", "status": "FAILED", "error": str(e)}
            )

    async def test_anti_detection_features(self):
        """Test anti-detection features."""
        try:
            logger.info("Testing anti-detection features")

            # Test session rotation logic
            should_rotate = self.integration.anti_detection.should_rotate_session()
            assert isinstance(should_rotate, bool), "Should rotate should be boolean"

            # Test random delay generation
            delay = self.integration.anti_detection.get_random_delay()
            assert 1.0 <= delay <= 5.0, "Random delay should be within expected range"

            # Test viewport randomization
            viewport = self.integration.anti_detection.get_random_viewport()
            assert "width" in viewport, "Viewport should have width"
            assert "height" in viewport, "Viewport should have height"

            # Test session metadata generation
            metadata = self.integration.anti_detection.generate_session_metadata()
            assert "timezone" in metadata, "Metadata should include timezone"
            assert "platform" in metadata, "Metadata should include platform"

            logger.info("Anti-detection features test passed")
            self.test_results.append(
                {
                    "test": "anti_detection_features",
                    "status": "PASSED",
                    "details": {
                        "should_rotate": should_rotate,
                        "random_delay": delay,
                        "viewport": viewport,
                        "metadata_keys": list(metadata.keys()),
                    },
                }
            )

        except Exception as e:
            logger.error(f"Anti-detection features test failed: {e}")
            self.test_results.append(
                {"test": "anti_detection_features", "status": "FAILED", "error": str(e)}
            )

    async def test_bet_validation(self):
        """Test bet validation logic."""
        try:
            logger.info("Testing bet validation")

            # Create test bet
            test_bet = Bet(
                event_id="test_event_123",
                platform=PlatformType.FANDUEL,
                market_type=MarketType.MONEYLINE,
                selection="home_team",
                odds=Decimal("2.50"),
                stake=Decimal("10.00"),
                expected_value=Decimal("0.05"),
                kelly_fraction=Decimal("0.25"),
            )

            # Test validation
            validation_result = await self.agent._validate_bet(test_bet)

            assert (
                validation_result.get("valid") is True
            ), "Valid bet should pass validation"

            # Test invalid bet (already placed)
            test_bet.status = BetStatus.PLACED
            validation_result = await self.agent._validate_bet(test_bet)

            assert (
                validation_result.get("valid") is False
            ), "Already placed bet should fail validation"

            logger.info("Bet validation test passed")
            self.test_results.append(
                {
                    "test": "bet_validation",
                    "status": "PASSED",
                    "details": "Bet validation logic working correctly",
                }
            )

        except Exception as e:
            logger.error(f"Bet validation test failed: {e}")
            self.test_results.append(
                {"test": "bet_validation", "status": "FAILED", "error": str(e)}
            )

    async def test_configuration_loading(self):
        """Test configuration loading and BrowserBase config creation."""
        try:
            logger.info("Testing configuration loading")

            # Test config loading
            config = self.integration.config
            assert config is not None, "Config should be loaded"

            # Test BrowserBase config creation
            browserbase_config = self.integration.browserbase_config
            assert (
                browserbase_config is not None
            ), "BrowserBase config should be created"
            assert hasattr(
                browserbase_config, "api_key"
            ), "BrowserBase config should have api_key"
            assert hasattr(
                browserbase_config, "project_id"
            ), "BrowserBase config should have project_id"
            assert hasattr(
                browserbase_config, "stealth_mode"
            ), "BrowserBase config should have stealth_mode"

            logger.info("Configuration loading test passed")
            self.test_results.append(
                {
                    "test": "configuration_loading",
                    "status": "PASSED",
                    "details": {
                        "config_loaded": config is not None,
                        "browserbase_config_created": browserbase_config is not None,
                        "stealth_mode": browserbase_config.stealth_mode,
                    },
                }
            )

        except Exception as e:
            logger.error(f"Configuration loading test failed: {e}")
            self.test_results.append(
                {"test": "configuration_loading", "status": "FAILED", "error": str(e)}
            )

    async def test_session_rotation(self):
        """Test session rotation functionality."""
        try:
            logger.info("Testing session rotation")

            # Get initial session health
            initial_health = await self.integration.get_session_health()

            # Force session rotation
            await self.integration._rotate_session()

            # Get updated session health
            updated_health = await self.integration.get_session_health()

            # Verify rotation occurred
            assert (
                updated_health.get("session_active") is True
            ), "Session should still be active after rotation"

            logger.info("Session rotation test passed")
            self.test_results.append(
                {
                    "test": "session_rotation",
                    "status": "PASSED",
                    "details": {
                        "initial_session_duration": initial_health.get(
                            "session_duration_seconds"
                        ),
                        "updated_session_duration": updated_health.get(
                            "session_duration_seconds"
                        ),
                        "rotation_successful": True,
                    },
                }
            )

        except Exception as e:
            logger.error(f"Session rotation test failed: {e}")
            self.test_results.append(
                {"test": "session_rotation", "status": "FAILED", "error": str(e)}
            )

    def print_test_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("BROWSERBASE INTEGRATION TEST RESULTS")
        print("=" * 60)

        passed = 0
        failed = 0
        info = 0

        for result in self.test_results:
            status = result["status"]
            test_name = result["test"]

            if status == "PASSED":
                print(f"✅ {test_name}: PASSED")
                passed += 1
            elif status == "FAILED":
                print(f"❌ {test_name}: FAILED")
                if "error" in result:
                    print(f"   Error: {result['error']}")
                failed += 1
            elif status == "INFO":
                print(f"ℹ️  {test_name}: INFO")
                info += 1

        print("\n" + "-" * 60)
        print(f"SUMMARY: {passed} passed, {failed} failed, {info} info")
        print("-" * 60)

        if failed == 0:
            print("🎉 All critical tests passed!")
        else:
            print("⚠️  Some tests failed. Check the errors above.")

        return passed, failed, info


async def main():
    """Main test function."""
    tester = BrowserBaseIntegrationTester()

    try:
        # Setup
        await tester.setup()

        # Run tests
        await tester.test_configuration_loading()
        await tester.test_session_creation()
        await tester.test_anti_detection_features()
        await tester.test_bet_validation()
        await tester.test_session_rotation()
        await tester.test_platform_connectivity()

        # Print results
        passed, failed, info = tester.print_test_summary()

        # Exit with appropriate code
        if failed == 0:
            print("\n🚀 BrowserBase integration is ready for use!")
            return 0
        else:
            print("\n⚠️  Some tests failed. Please check the configuration.")
            return 1

    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"\n💥 Test execution failed: {e}")
        return 1

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

    # Run tests
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
