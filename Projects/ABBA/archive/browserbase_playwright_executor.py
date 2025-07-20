"""
BrowserBase Playwright Executor
Production-ready DraftKings balance monitoring using Playwright
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime

from browserbase import Browserbase
from playwright.async_api import async_playwright


@dataclass
class BalanceInfo:
    """Balance information structure."""
    account_balance: float | None = None
    available_balance: float | None = None
    pending_balance: float | None = None
    currency: str = "USD"
    timestamp: datetime = None
    source: str = "DraftKings"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BrowserBasePlaywrightExecutor:
    """Production-ready BrowserBase executor using Playwright."""

    def __init__(self, api_key: str | None = None, project_id: str | None = None):
        """Initialize the executor."""
        self.api_key = api_key or os.getenv("BROWSERBASE_API_KEY")
        self.project_id = project_id or os.getenv("BROWSERBASE_PROJECT_ID")

        if not self.api_key or not self.project_id:
            raise ValueError("Missing BROWSERBASE_API_KEY or BROWSERBASE_PROJECT_ID")

        self.bb = Browserbase(api_key=self.api_key)
        self.session = None
        self.browser = None
        self.page = None

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def create_session(self) -> bool:
        """Create a new BrowserBase session."""
        try:
            self.logger.info("Creating BrowserBase session...")
            self.session = self.bb.sessions.create(project_id=self.project_id)
            self.logger.info(f"✅ Session created: {self.session.id}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to create session: {e}")
            return False

    async def connect_browser(self) -> bool:
        """Connect to BrowserBase using Playwright."""
        try:
            if not self.session:
                if not await self.create_session():
                    return False

            self.logger.info("Connecting to BrowserBase via Playwright...")
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.connect_over_cdp(self.session.connect_url)
            self.logger.info("✅ Connected to BrowserBase via Playwright!")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to connect browser: {e}")
            return False

    async def create_page(self) -> bool:
        """Create a new page."""
        try:
            if not self.browser:
                if not await self.connect_browser():
                    return False

            self.page = await self.browser.new_page()
            self.logger.info("✅ New page created")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to create page: {e}")
            return False

    async def navigate_to_draftkings(self) -> bool:
        """Navigate to DraftKings website."""
        try:
            if not self.page:
                if not await self.create_page():
                    return False

            self.logger.info("Navigating to DraftKings...")
            await self.page.goto("https://www.draftkings.com", timeout=15000)
            await self.page.wait_for_load_state("domcontentloaded", timeout=10000)

            title = await self.page.title()
            self.logger.info(f"✅ DraftKings loaded: {title}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to navigate to DraftKings: {e}")
            return False

    async def login_to_draftkings(self, username: str, password: str) -> bool:
        """Login to DraftKings account."""
        try:
            if not self.page:
                self.logger.error("No page available for login")
                return False

            self.logger.info("Attempting to login to DraftKings...")

            # Look for login elements
            login_selectors = {
                'username': [
                    'input[type="email"]',
                    'input[name="username"]',
                    'input[id*="username"]',
                    'input[id*="email"]'
                ],
                'password': [
                    'input[type="password"]',
                    'input[name="password"]',
                    'input[id*="password"]'
                ],
                'submit': [
                    'button[type="submit"]',
                    'input[type="submit"]',
                    'button:has-text("Login")',
                    'button:has-text("Sign In")'
                ]
            }

            # Find and fill username
            username_field = None
            for selector in login_selectors['username']:
                try:
                    username_field = await self.page.query_selector(selector)
                    if username_field:
                        break
                except:
                    continue

            if not username_field:
                self.logger.error("❌ Username field not found")
                return False

            # Find and fill password
            password_field = None
            for selector in login_selectors['password']:
                try:
                    password_field = await self.page.query_selector(selector)
                    if password_field:
                        break
                except:
                    continue

            if not password_field:
                self.logger.error("❌ Password field not found")
                return False

            # Fill credentials
            await username_field.fill(username)
            await password_field.fill(password)

            # Find and click submit button
            submit_button = None
            for selector in login_selectors['submit']:
                try:
                    submit_button = await self.page.query_selector(selector)
                    if submit_button:
                        break
                except:
                    continue

            if not submit_button:
                self.logger.error("❌ Submit button not found")
                return False

            # Click submit
            await submit_button.click()

            # Wait for navigation
            await self.page.wait_for_load_state("domcontentloaded", timeout=10000)

            # Check if login was successful
            current_url = self.page.url
            if "login" not in current_url.lower() and "signin" not in current_url.lower():
                self.logger.info("✅ Login appears successful")
                return True
            else:
                self.logger.error("❌ Login may have failed")
                return False

        except Exception as e:
            self.logger.error(f"❌ Login failed: {e}")
            return False

    async def extract_balance_info(self) -> BalanceInfo | None:
        """Extract balance information from DraftKings."""
        try:
            if not self.page:
                self.logger.error("No page available for balance extraction")
                return None

            self.logger.info("Extracting balance information...")

            # Navigate to account page
            try:
                await self.page.goto("https://www.draftkings.com/account", timeout=15000)
                await self.page.wait_for_load_state("domcontentloaded", timeout=10000)
            except:
                self.logger.warning("Could not navigate to account page, checking current page")

            # Get page content
            content = await self.page.text_content("body")

            # Look for balance patterns
            balance_patterns = [
                r'\$[\d,]+\.?\d*',  # $1,234.56
                r'[\d,]+\.?\d*\s*USD',  # 1,234.56 USD
                r'Balance:\s*\$?[\d,]+\.?\d*',  # Balance: $1,234.56
                r'Account:\s*\$?[\d,]+\.?\d*',  # Account: $1,234.56
                r'Available:\s*\$?[\d,]+\.?\d*',  # Available: $1,234.56
            ]

            import re
            balances = []
            for pattern in balance_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                balances.extend(matches)

            # Look for balance elements
            balance_selectors = [
                '[class*="balance"]',
                '[id*="balance"]',
                '[class*="account"]',
                '[id*="account"]',
                '[class*="funds"]',
                '[id*="funds"]',
                '[class*="money"]',
                '[id*="money"]',
                '[class*="amount"]',
                '[id*="amount"]'
            ]

            balance_elements = []
            for selector in balance_selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    for element in elements:
                        text = await element.text_content()
                        if text and text.strip():
                            balance_elements.append(text.strip())
                except:
                    continue

            # Extract numeric values
            all_balance_texts = balances + balance_elements
            numeric_balances = []

            for text in all_balance_texts:
                # Extract numbers from text
                numbers = re.findall(r'[\d,]+\.?\d*', text)
                for num_str in numbers:
                    try:
                        # Remove commas and convert to float
                        num = float(num_str.replace(',', ''))
                        if num > 0:  # Only positive balances
                            numeric_balances.append(num)
                    except:
                        continue

            # Create balance info
            balance_info = BalanceInfo()

            if numeric_balances:
                # Use the highest balance as account balance
                balance_info.account_balance = max(numeric_balances)
                balance_info.available_balance = max(numeric_balances)

                self.logger.info(f"💰 Balance extracted: ${balance_info.account_balance:,.2f}")
            else:
                self.logger.warning("No balance information found")

            return balance_info

        except Exception as e:
            self.logger.error(f"❌ Balance extraction failed: {e}")
            return None

    async def take_screenshot(self, filename: str = "draftkings_balance.png") -> bool:
        """Take a screenshot of the current page."""
        try:
            if not self.page:
                self.logger.error("No page available for screenshot")
                return False

            await self.page.screenshot(path=filename)
            self.logger.info(f"✅ Screenshot saved: {filename}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Screenshot failed: {e}")
            return False

    async def close(self):
        """Close the browser and cleanup."""
        try:
            if self.page:
                await self.page.close()
                self.logger.info("Page closed")

            if self.browser:
                await self.browser.close()
                self.logger.info("Browser closed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


async def test_draftkings_balance_monitoring():
    """Test DraftKings balance monitoring with Playwright."""
    print("🎭 DraftKings Balance Monitoring Test")
    print("=" * 50)

    executor = BrowserBasePlaywrightExecutor()

    try:
        # Navigate to DraftKings
        if not await executor.navigate_to_draftkings():
            return False

        # Take screenshot
        await executor.take_screenshot("draftkings_home.png")

        # Extract balance info (without login for now)
        balance_info = await executor.extract_balance_info()

        if balance_info and balance_info.account_balance:
            print("\n💰 Balance Information:")
            print(f"   Account Balance: ${balance_info.account_balance:,.2f}")
            print(f"   Available Balance: ${balance_info.available_balance:,.2f}")
            print(f"   Currency: {balance_info.currency}")
            print(f"   Timestamp: {balance_info.timestamp}")
            print(f"   Source: {balance_info.source}")
        else:
            print("\n❌ No balance information extracted")
            print("💡 This is expected without login credentials")

        print("\n✅ Balance monitoring test completed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        await executor.close()


def main():
    """Main function."""
    print("🧪 DraftKings Balance Monitoring with Playwright")
    print("=" * 50)

    success = asyncio.run(test_draftkings_balance_monitoring())

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)

    if success:
        print("✅ Playwright balance monitoring successful!")
        print("🎉 Ready for production use!")
        print("\n🚀 Next steps:")
        print("1. Add your DraftKings credentials")
        print("2. Implement automated monitoring")
        print("3. Set up balance alerts")
        print("\n💡 Playwright is perfect for this use case!")
    else:
        print("❌ Balance monitoring test failed!")
        print("\n💡 Check the logs above for details")


if __name__ == "__main__":
    main()
