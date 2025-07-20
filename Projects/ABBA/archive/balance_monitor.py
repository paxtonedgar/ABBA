"""
Balance Monitor for ABMBA System
Monitors betting platform balances and manages fund synchronization
"""

import asyncio
import os
import random
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import structlog
from browserbase_executor import BrowserBaseConfig, BrowserBaseSession

logger = structlog.get_logger()


class BalanceStatus(Enum):
    """Balance status enumeration."""
    SUFFICIENT = "sufficient"
    LOW = "low"
    CRITICAL = "critical"
    DEPLETED = "depleted"


@dataclass
class BalanceInfo:
    """Balance information structure."""
    platform: str
    current_balance: Decimal
    available_balance: Decimal
    pending_bets: Decimal
    last_updated: datetime
    status: BalanceStatus
    threshold_warnings: list[str] = None


class BalanceMonitor:
    """Monitors betting platform balances with anti-detection measures."""

    def __init__(self, config: BrowserBaseConfig):
        self.config = config
        self.session = None
        self.balance_cache = {}
        self.last_check = {}
        self.check_intervals = {
            "fanduel": 300,  # 5 minutes
            "draftkings": 300  # 5 minutes
        }
        self.low_balance_threshold = Decimal("50.00")  # $50
        self.critical_balance_threshold = Decimal("20.00")  # $20

        # Platform-specific balance selectors
        self.balance_selectors = {
            "fanduel": [
                ".balance",
                "[data-testid='balance']",
                ".account-balance",
                ".user-balance",
                ".wallet-balance",
                "span[class*='balance']",
                "div[class*='balance']"
            ],
            "draftkings": [
                ".balance",
                "[data-testid='balance']",
                ".account-balance",
                ".user-balance",
                ".wallet-balance",
                "span[class*='balance']",
                "div[class*='balance']"
            ]
        }

        # Balance extraction patterns
        self.balance_patterns = [
            r'\$?([\d,]+\.?\d*)',  # $1,234.56 or 1234.56
            r'([\d,]+\.?\d*)\s*USD',  # 1,234.56 USD
            r'Balance:\s*\$?([\d,]+\.?\d*)',  # Balance: $1,234.56
            r'Available:\s*\$?([\d,]+\.?\d*)',  # Available: $1,234.56
        ]

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = BrowserBaseSession(self.config)
        await self.session.create_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close_session()

    async def check_balance(self, platform: str, credentials: dict[str, str]) -> BalanceInfo | None:
        """Check balance on a betting platform with anti-detection measures."""
        try:
            # Check if we need to update (respect rate limits)
            if self._should_skip_check(platform):
                cached_balance = self.balance_cache.get(platform)
                if cached_balance:
                    logger.info(f"Using cached balance for {platform}: ${cached_balance.current_balance}")
                    return cached_balance

            logger.info(f"Checking balance for {platform}")

            # Login to platform if not already logged in
            login_success = await self._ensure_login(platform, credentials)
            if not login_success:
                logger.error(f"Failed to login to {platform} for balance check")
                return None

            # Navigate to account/balance page with stealth
            balance_url = self._get_balance_page_url(platform)
            await self.session.navigate(balance_url)

            # Add random delay to avoid detection
            await asyncio.sleep(random.uniform(2.0, 4.0))

            # Extract balance using multiple methods
            balance = await self._extract_balance(platform)

            if balance is None:
                logger.error(f"Could not extract balance from {platform}")
                return None

            # Calculate balance status
            status = self._calculate_balance_status(balance)

            # Create balance info
            balance_info = BalanceInfo(
                platform=platform,
                current_balance=balance,
                available_balance=balance,  # Simplified for now
                pending_bets=Decimal("0.00"),  # Would need additional logic
                last_updated=datetime.utcnow(),
                status=status,
                threshold_warnings=self._generate_warnings(balance)
            )

            # Cache the result
            self.balance_cache[platform] = balance_info
            self.last_check[platform] = datetime.utcnow()

            logger.info(f"Balance check completed for {platform}: ${balance} ({status.value})")
            return balance_info

        except Exception as e:
            logger.error(f"Error checking balance for {platform}: {e}")
            return None

    async def _ensure_login(self, platform: str, credentials: dict[str, str]) -> bool:
        """Ensure we're logged into the platform."""
        try:
            # Check if already logged in by looking for balance elements
            for selector in self.balance_selectors[platform]:
                balance_text = await self.session.get_text(selector)
                if balance_text and any(char.isdigit() for char in balance_text):
                    logger.info(f"Already logged into {platform}")
                    return True

            # Need to login
            logger.info(f"Logging into {platform} for balance check")

            # Navigate to login page
            login_url = self._get_login_url(platform)
            await self.session.navigate(login_url)
            await asyncio.sleep(random.uniform(2.0, 4.0))

            # Fill login form with human-like behavior
            username_selector = self._get_username_selector(platform)
            password_selector = self._get_password_selector(platform)
            login_button_selector = self._get_login_button_selector(platform)

            await self.session.type_text(username_selector, credentials['username'])
            await asyncio.sleep(random.uniform(1.0, 2.0))

            await self.session.type_text(password_selector, credentials['password'])
            await asyncio.sleep(random.uniform(1.0, 2.0))

            await self.session.click(login_button_selector)
            await asyncio.sleep(random.uniform(3.0, 6.0))

            # Verify login success
            for selector in self.balance_selectors[platform]:
                balance_text = await self.session.get_text(selector)
                if balance_text and any(char.isdigit() for char in balance_text):
                    logger.info(f"Successfully logged into {platform}")
                    return True

            logger.error(f"Login verification failed for {platform}")
            return False

        except Exception as e:
            logger.error(f"Error during login for {platform}: {e}")
            return False

    async def _extract_balance(self, platform: str) -> Decimal | None:
        """Extract balance using multiple methods to avoid detection."""
        try:
            # Try multiple selectors
            for selector in self.balance_selectors[platform]:
                balance_text = await self.session.get_text(selector)
                if balance_text:
                    balance = self._parse_balance_text(balance_text)
                    if balance:
                        return balance

            # Try JavaScript evaluation as fallback
            js_expressions = [
                "document.querySelector('.balance')?.textContent",
                "document.querySelector('[data-testid=\"balance\"]')?.textContent",
                "document.querySelector('.account-balance')?.textContent",
                "Array.from(document.querySelectorAll('*')).find(el => el.textContent.includes('$') && /\\$?[\\d,]+\\.?\\d*/.test(el.textContent))?.textContent"
            ]

            for expression in js_expressions:
                try:
                    response = await self.session.client.post(
                        f"/sessions/{self.session.session_id}/evaluate",
                        json={"expression": expression}
                    )
                    response.raise_for_status()
                    result = response.json()
                    balance_text = result.get("result", "")

                    if balance_text:
                        balance = self._parse_balance_text(balance_text)
                        if balance:
                            return balance
                except Exception as e:
                    logger.debug(f"JavaScript evaluation failed: {e}")
                    continue

            return None

        except Exception as e:
            logger.error(f"Error extracting balance: {e}")
            return None

    def _parse_balance_text(self, text: str) -> Decimal | None:
        """Parse balance text using multiple patterns."""
        import re

        try:
            # Clean the text
            text = text.strip()

            # Try multiple patterns
            for pattern in self.balance_patterns:
                match = re.search(pattern, text)
                if match:
                    balance_str = match.group(1).replace(',', '')
                    try:
                        return Decimal(balance_str)
                    except (ValueError, TypeError):
                        continue

            return None

        except Exception as e:
            logger.error(f"Error parsing balance text '{text}': {e}")
            return None

    def _calculate_balance_status(self, balance: Decimal) -> BalanceStatus:
        """Calculate balance status based on thresholds."""
        if balance <= Decimal("0"):
            return BalanceStatus.DEPLETED
        elif balance <= self.critical_balance_threshold:
            return BalanceStatus.CRITICAL
        elif balance <= self.low_balance_threshold:
            return BalanceStatus.LOW
        else:
            return BalanceStatus.SUFFICIENT

    def _generate_warnings(self, balance: Decimal) -> list[str]:
        """Generate warnings based on balance level."""
        warnings = []

        if balance <= self.critical_balance_threshold:
            warnings.append(f"CRITICAL: Balance ${balance} is below critical threshold ${self.critical_balance_threshold}")
        elif balance <= self.low_balance_threshold:
            warnings.append(f"LOW: Balance ${balance} is below low threshold ${self.low_balance_threshold}")

        return warnings

    def _should_skip_check(self, platform: str) -> bool:
        """Check if we should skip balance check due to rate limiting."""
        last_check_time = self.last_check.get(platform)
        if not last_check_time:
            return False

        interval = self.check_intervals[platform]
        time_since_check = (datetime.utcnow() - last_check_time).total_seconds()

        return time_since_check < interval

    def _get_balance_page_url(self, platform: str) -> str:
        """Get the balance/account page URL for a platform."""
        urls = {
            "fanduel": "https://sportsbook.fanduel.com/account",
            "draftkings": "https://sportsbook.draftkings.com/account"
        }
        return urls.get(platform, "")

    def _get_login_url(self, platform: str) -> str:
        """Get the login URL for a platform."""
        urls = {
            "fanduel": "https://sportsbook.fanduel.com/login",
            "draftkings": "https://sportsbook.draftkings.com/login"
        }
        return urls.get(platform, "")

    def _get_username_selector(self, platform: str) -> str:
        """Get username input selector for a platform."""
        selectors = {
            "fanduel": "input[name='username'], input[name='email'], input[type='email']",
            "draftkings": "input[name='username'], input[name='email'], input[type='email']"
        }
        return selectors.get(platform, "")

    def _get_password_selector(self, platform: str) -> str:
        """Get password input selector for a platform."""
        selectors = {
            "fanduel": "input[name='password'], input[type='password']",
            "draftkings": "input[name='password'], input[type='password']"
        }
        return selectors.get(platform, "")

    def _get_login_button_selector(self, platform: str) -> str:
        """Get login button selector for a platform."""
        selectors = {
            "fanduel": "button[type='submit'], input[type='submit']",
            "draftkings": "button[type='submit'], input[type='submit']"
        }
        return selectors.get(platform, "")


class FundManager:
    """Manages fund synchronization between human and agent."""

    def __init__(self, balance_monitor: BalanceMonitor):
        self.balance_monitor = balance_monitor
        self.expected_balances = {}
        self.deposit_confirmations = {}
        self.notification_callbacks = []

    async def monitor_all_platforms(self, credentials: dict[str, dict[str, str]]) -> dict[str, BalanceInfo]:
        """Monitor balances on all platforms."""
        results = {}

        for platform, creds in credentials.items():
            try:
                balance_info = await self.balance_monitor.check_balance(platform, creds)
                if balance_info:
                    results[platform] = balance_info

                    # Check for fund needs
                    await self._check_fund_needs(platform, balance_info)

            except Exception as e:
                logger.error(f"Error monitoring {platform}: {e}")

        return results

    async def _check_fund_needs(self, platform: str, balance_info: BalanceInfo):
        """Check if platform needs additional funds."""
        if balance_info.status in [BalanceStatus.LOW, BalanceStatus.CRITICAL, BalanceStatus.DEPLETED]:
            await self._notify_fund_needed(platform, balance_info)

    async def _notify_fund_needed(self, platform: str, balance_info: BalanceInfo):
        """Notify about fund needs."""
        message = f"FUND NEEDED: {platform.upper()} balance is ${balance_info.current_balance} ({balance_info.status.value})"

        for callback in self.notification_callbacks:
            try:
                await callback(platform, balance_info)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")

        logger.warning(message)

    def register_notification_callback(self, callback):
        """Register a callback for fund notifications."""
        self.notification_callbacks.append(callback)

    async def confirm_deposit(self, platform: str, amount: Decimal, confirmation_id: str = None):
        """Confirm a deposit made by the human."""
        try:
            # Wait a bit for the deposit to process
            await asyncio.sleep(random.uniform(5.0, 10.0))

            # Check the new balance
            credentials = self._get_platform_credentials(platform)
            if not credentials:
                logger.error(f"No credentials found for {platform}")
                return False

            balance_info = await self.balance_monitor.check_balance(platform, credentials)
            if not balance_info:
                logger.error(f"Could not verify balance after deposit for {platform}")
                return False

            # Store confirmation
            self.deposit_confirmations[platform] = {
                "amount": amount,
                "new_balance": balance_info.current_balance,
                "timestamp": datetime.utcnow(),
                "confirmation_id": confirmation_id
            }

            logger.info(f"Deposit confirmed for {platform}: ${amount}, new balance: ${balance_info.current_balance}")
            return True

        except Exception as e:
            logger.error(f"Error confirming deposit for {platform}: {e}")
            return False

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

    def get_deposit_history(self, platform: str) -> list[dict[str, Any]]:
        """Get deposit history for a platform."""
        return [self.deposit_confirmations.get(platform, {})]

    def get_expected_balance(self, platform: str) -> Decimal | None:
        """Get expected balance for a platform."""
        return self.expected_balances.get(platform)


# Example notification callbacks
async def email_notification(platform: str, balance_info: BalanceInfo):
    """Send email notification about fund needs."""
    # Implementation would use your email service
    logger.info(f"EMAIL: {platform} needs funds - balance: ${balance_info.current_balance}")

async def slack_notification(platform: str, balance_info: BalanceInfo):
    """Send Slack notification about fund needs."""
    # Implementation would use your Slack webhook
    logger.info(f"SLACK: {platform} needs funds - balance: ${balance_info.current_balance}")

async def console_notification(platform: str, balance_info: BalanceInfo):
    """Console notification about fund needs."""
    print(f"\n🚨 FUND ALERT: {platform.upper()}")
    print(f"   Current Balance: ${balance_info.current_balance}")
    print(f"   Status: {balance_info.status.value}")
    print(f"   Warnings: {balance_info.threshold_warnings}")
    print(f"   Time: {balance_info.last_updated.strftime('%Y-%m-%d %H:%M:%S')}\n")


# Example usage
async def main():
    """Example usage of balance monitoring."""
    config = BrowserBaseConfig(
        api_key=os.getenv("BROWSERBASE_API_KEY"),
        project_id=os.getenv("BROWSERBASE_PROJECT_ID"),
        stealth_mode=True
    )

    async with BalanceMonitor(config) as monitor:
        fund_manager = FundManager(monitor)

        # Register notification callbacks
        fund_manager.register_notification_callback(console_notification)
        fund_manager.register_notification_callback(email_notification)
        fund_manager.register_notification_callback(slack_notification)

        # Monitor all platforms
        credentials = {
            "fanduel": {
                "username": os.getenv("FANDUEL_USERNAME"),
                "password": os.getenv("FANDUEL_PASSWORD")
            },
            "draftkings": {
                "username": os.getenv("DRAFTKINGS_USERNAME"),
                "password": os.getenv("DRAFTKINGS_PASSWORD")
            }
        }

        results = await fund_manager.monitor_all_platforms(credentials)

        for platform, balance_info in results.items():
            print(f"{platform}: ${balance_info.current_balance} ({balance_info.status.value})")


if __name__ == "__main__":
    asyncio.run(main())
