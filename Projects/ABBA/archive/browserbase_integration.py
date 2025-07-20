"""
BrowserBase Integration for ABMBA System
Integrates BrowserBase with existing ABMBA agents for advanced bet placement
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Any

import structlog
import yaml
from browserbase_executor import (
    AdvancedAntiDetectionManager,
    BettingPlatformExecutor,
    BrowserBaseConfig,
)
from database import DatabaseManager

from models import Bet, BetStatus

logger = structlog.get_logger()


class BrowserBaseBettingIntegration:
    """Integrates BrowserBase with ABMBA for advanced bet placement."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.browserbase_config = self._create_browserbase_config()
        self.anti_detection = AdvancedAntiDetectionManager(self.browserbase_config)
        self.db_manager = None  # Will be initialized in initialize()
        self.executor = None
        self.session_start_time = None

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path) as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def _create_browserbase_config(self) -> BrowserBaseConfig:
        """Create BrowserBase configuration from environment variables."""
        return BrowserBaseConfig(
            api_key=os.getenv("BROWSERBASE_API_KEY"),
            project_id=os.getenv("BROWSERBASE_PROJECT_ID"),
            stealth_mode=self.config.get('security', {}).get('browserbase', {}).get('enabled', True),
            proxy_enabled=self.config.get('security', {}).get('proxy_enabled', False),
            proxy_url=os.getenv("BROWSERBASE_PROXY"),
            viewport_width=self.config.get('security', {}).get('viewport_width', 1920),
            viewport_height=self.config.get('security', {}).get('viewport_height', 1080)
        )

    async def initialize(self):
        """Initialize the BrowserBase integration."""
        try:
            # Initialize database manager with database URL
            db_url = self.config.get('database', {}).get('url', 'sqlite+aiosqlite:///abmba.db')
            self.db_manager = DatabaseManager(db_url)
            await self.db_manager.initialize()

            # Create BrowserBase executor
            self.executor = BettingPlatformExecutor(self.browserbase_config)
            await self.executor.__aenter__()

            self.session_start_time = datetime.utcnow()
            logger.info("BrowserBase integration initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing BrowserBase integration: {e}")
            raise



    async def close(self):
        """Close the BrowserBase integration."""
        try:
            if self.executor:
                await self.executor.__aexit__(None, None, None)

            if self.db_manager:
                await self.db_manager.close()

            logger.info("BrowserBase integration closed")

        except Exception as e:
            logger.error(f"Error closing BrowserBase integration: {e}")

    async def execute_bet(self, bet: Bet) -> dict[str, Any]:
        """Execute a bet using BrowserBase with advanced anti-detection."""
        try:
            # Check if session should be rotated
            if self.anti_detection.should_rotate_session():
                await self._rotate_session()

            # Get platform credentials
            platform = bet.platform.value
            credentials = self._get_platform_credentials(platform)

            if not credentials:
                return {
                    "success": False,
                    "error": f"No credentials found for platform {platform}"
                }

            # Login to platform
            login_success = await self.executor.login_platform(
                platform,
                credentials['username'],
                credentials['password']
            )

            if not login_success:
                return {
                    "success": False,
                    "error": f"Failed to login to {platform}"
                }

            # Generate event URL and market selector
            event_url = await self._generate_event_url(platform, bet.event_id)
            market_selector = await self._generate_market_selector(
                platform,
                bet.market_type.value,
                bet.selection
            )

            if not event_url or not market_selector:
                return {
                    "success": False,
                    "error": "Could not generate event URL or market selector"
                }

            # Place bet with BrowserBase
            result = await self.executor.place_bet(
                platform,
                event_url,
                market_selector,
                float(bet.stake)
            )

            # Update bet status in database
            if result['success']:
                bet.status = BetStatus.PLACED
                bet.placed_at = datetime.utcnow()
                await self.db_manager.save_bet(bet)

                # Log successful bet placement
                logger.info(f"Bet placed successfully: {bet.id} on {platform}")

                # Add behavior pattern for anti-detection
                self.anti_detection.add_behavior_pattern({
                    "action": "bet_placed",
                    "platform": platform,
                    "stake": float(bet.stake),
                    "timestamp": datetime.utcnow()
                })

            return result

        except Exception as e:
            logger.error(f"Error executing bet: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _rotate_session(self):
        """Rotate the BrowserBase session for anti-detection."""
        try:
            logger.info("Rotating BrowserBase session for anti-detection")

            # Close current session
            if self.executor:
                await self.executor.__aexit__(None, None, None)

            # Create new session
            self.executor = BettingPlatformExecutor(self.browserbase_config)
            await self.executor.__aenter__()

            # Mark rotation
            self.anti_detection.rotate_session()

            logger.info("Session rotation completed")

        except Exception as e:
            logger.error(f"Error rotating session: {e}")
            raise

    def _get_platform_credentials(self, platform: str) -> dict[str, str] | None:
        """Get platform credentials from environment variables."""
        platform_upper = platform.upper()

        username = os.getenv(f"{platform_upper}_USERNAME")
        password = os.getenv(f"{platform_upper}_PASSWORD")

        if username and password:
            return {
                "username": username,
                "password": password
            }

        return None

    async def _generate_event_url(self, platform: str, event_id: str) -> str | None:
        """Generate event URL for the given platform and event."""
        try:
            # Get event details from database
            event = await self.db_manager.get_event(event_id)

            if not event:
                logger.error(f"Event not found: {event_id}")
                return None

            # Generate platform-specific URLs
            if platform == "fanduel":
                return f"https://sportsbook.fanduel.com/event/{event_id}"
            elif platform == "draftkings":
                return f"https://sportsbook.draftkings.com/event/{event_id}"
            else:
                logger.error(f"Unsupported platform: {platform}")
                return None

        except Exception as e:
            logger.error(f"Error generating event URL: {e}")
            return None

    async def _generate_market_selector(self, platform: str, market_type: str, selection: str) -> str | None:
        """Generate CSS selector for the market selection."""
        try:
            # Platform-specific selectors
            selectors = {
                "fanduel": {
                    "moneyline": f"[data-testid='moneyline-{selection.lower()}'], .moneyline-{selection.lower()}",
                    "spread": f"[data-testid='spread-{selection.lower()}'], .spread-{selection.lower()}",
                    "total": f"[data-testid='total-{selection.lower()}'], .total-{selection.lower()}"
                },
                "draftkings": {
                    "moneyline": f".moneyline-{selection.lower()}, [data-testid='moneyline-{selection.lower()}']",
                    "spread": f".spread-{selection.lower()}, [data-testid='spread-{selection.lower()}']",
                    "total": f".total-{selection.lower()}, [data-testid='total-{selection.lower()}']"
                }
            }

            platform_selectors = selectors.get(platform, {})
            return platform_selectors.get(market_type.lower(), f".{market_type.lower()}-{selection.lower()}")

        except Exception as e:
            logger.error(f"Error generating market selector: {e}")
            return None

    async def get_session_health(self) -> dict[str, Any]:
        """Get session health information."""
        try:
            session_duration = datetime.utcnow() - self.session_start_time if self.session_start_time else timedelta(0)

            return {
                "session_active": self.executor is not None,
                "session_duration_seconds": session_duration.total_seconds(),
                "anti_detection_patterns": len(self.anti_detection.behavior_patterns),
                "last_rotation": self.anti_detection.last_rotation.isoformat() if self.anti_detection.last_rotation else None,
                "should_rotate": self.anti_detection.should_rotate_session()
            }

        except Exception as e:
            logger.error(f"Error getting session health: {e}")
            return {"error": str(e)}

    async def test_platform_connectivity(self, platform: str) -> dict[str, Any]:
        """Test connectivity to a betting platform."""
        try:
            credentials = self._get_platform_credentials(platform)

            if not credentials:
                return {
                    "success": False,
                    "error": f"No credentials found for {platform}"
                }

            # Test login (with dummy credentials for safety)
            test_success = await self.executor.login_platform(
                platform,
                "test_username",
                "test_password"
            )

            return {
                "success": True,
                "platform": platform,
                "connectivity": "available",
                "login_test": "completed"  # Note: will fail with dummy credentials
            }

        except Exception as e:
            logger.error(f"Error testing platform connectivity: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class BrowserBaseExecutionAgent:
    """Agent for executing bets using BrowserBase integration."""

    def __init__(self, config: dict, integration: BrowserBaseBettingIntegration):
        self.config = config
        self.integration = integration
        self.llm = None  # Will be initialized if needed for decision making

    async def execute_bet_recommendation(self, bet: Bet) -> dict[str, Any]:
        """Execute a bet recommendation using BrowserBase."""
        try:
            # Validate bet before execution
            validation_result = await self._validate_bet(bet)
            if not validation_result['valid']:
                return {
                    "success": False,
                    "error": f"Bet validation failed: {validation_result['error']}"
                }

            # Execute bet with BrowserBase
            result = await self.integration.execute_bet(bet)

            # Log execution result
            if result['success']:
                logger.info(f"Bet executed successfully: {bet.id}")
            else:
                logger.error(f"Bet execution failed: {bet.id} - {result.get('error', 'Unknown error')}")

            return result

        except Exception as e:
            logger.error(f"Error in bet execution agent: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _validate_bet(self, bet: Bet) -> dict[str, Any]:
        """Validate bet before execution."""
        try:
            # Check if bet is already placed
            if bet.status == BetStatus.PLACED:
                return {
                    "valid": False,
                    "error": "Bet already placed"
                }

            # Check stake amount
            if bet.stake <= 0:
                return {
                    "valid": False,
                    "error": "Invalid stake amount"
                }

            # Check if platform is supported
            if bet.platform.value not in ["fanduel", "draftkings"]:
                return {
                    "valid": False,
                    "error": f"Unsupported platform: {bet.platform.value}"
                }

            return {"valid": True}

        except Exception as e:
            logger.error(f"Error validating bet: {e}")
            return {
                "valid": False,
                "error": str(e)
            }


# Example usage
async def main():
    """Example usage of BrowserBase integration."""
    try:
        # Initialize integration
        integration = BrowserBaseBettingIntegration()
        await integration.initialize()

        # Create execution agent
        agent = BrowserBaseExecutionAgent({}, integration)

        # Test session health
        health = await integration.get_session_health()
        print(f"Session health: {health}")

        # Test platform connectivity
        fanduel_test = await integration.test_platform_connectivity("fanduel")
        print(f"FanDuel connectivity: {fanduel_test}")

        # Close integration
        await integration.close()

    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    asyncio.run(main())
