"""
DraftKings Manual Login + Automated Balance Monitor
Hybrid approach: Manual login with automated balance extraction
"""

import asyncio
import logging
import os
import random
import re

from playwright.async_api import async_playwright


class DraftKingsManualLoginMonitor:
    """DraftKings balance monitor using manual login + automated extraction."""

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.session_file = "draftkings_session.json"

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    async def create_browser_context(self) -> tuple:
        """Create browser context for manual login."""
        try:
            self.logger.info("Creating browser context for manual login...")

            # Start Playwright
            self.playwright = await async_playwright().start()

            # Launch browser
            self.browser = await self.playwright.chromium.launch(
                headless=False,  # Show browser for manual login
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-blink-features=AutomationControlled'
                ]
            )

            # Create context
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )

            self.logger.info("✅ Browser context created!")
            return self.context, self.browser

        except Exception as e:
            self.logger.error(f"❌ Failed to create browser context: {e}")
            return None, None

    async def manual_login_setup(self) -> bool:
        """Setup for manual login and session capture."""
        try:
            if not self.context:
                context, browser = await self.create_browser_context()
                if not context:
                    return False
                self.context = context
                self.browser = browser

            self.logger.info("Setting up manual login...")

            # Create page
            self.page = await self.context.new_page()
            self.logger.info("✅ Page created for manual login")

            # Navigate to DraftKings login
            await self.page.goto("https://www.draftkings.com/login", timeout=60000)

            title = await self.page.title()
            current_url = self.page.url

            self.logger.info(f"✅ Navigated to: {title}")
            self.logger.info(f"Current URL: {current_url}")

            # Take screenshot
            await self.page.screenshot(path="manual_login_setup.png")
            self.logger.info("✅ Screenshot saved: manual_login_setup.png")

            # Instructions for user
            print("\n" + "=" * 60)
            print("MANUAL LOGIN INSTRUCTIONS")
            print("=" * 60)
            print("1. The browser window should now be open")
            print("2. Manually log in to your DraftKings account")
            print("3. Wait for the login to complete")
            print("4. Navigate to your account/balance page")
            print("5. Press Enter in this terminal when ready")
            print("=" * 60)

            # Wait for user input
            input("Press Enter when you've completed the manual login...")

            # Capture session state
            await self.context.storage_state(path=self.session_file)
            self.logger.info(f"✅ Session saved to: {self.session_file}")

            # Take screenshot after login
            await self.page.screenshot(path="manual_login_complete.png")
            self.logger.info("✅ Screenshot saved: manual_login_complete.png")

            return True

        except Exception as e:
            self.logger.error(f"❌ Manual login setup failed: {e}")
            return False

    async def load_session_and_extract_balance(self) -> str:
        """Load saved session and extract balance automatically."""
        try:
            if not os.path.exists(self.session_file):
                self.logger.error(f"❌ Session file not found: {self.session_file}")
                return None

            self.logger.info("Loading saved session...")

            if not self.context:
                context, browser = await self.create_browser_context()
                if not context:
                    return None
                self.context = context
                self.browser = browser

            # Load session state
            await self.context.storage_state(path=self.session_file)
            self.logger.info("✅ Session loaded")

            # Create page with session
            self.page = await self.context.new_page()
            self.logger.info("✅ Page created with session")

            # Navigate to balance page
            balance_urls = [
                "https://www.draftkings.com/account",
                "https://sportsbook.draftkings.com/account",
                "https://www.draftkings.com/account/balance",
                "https://sportsbook.draftkings.com/account/balance",
                "https://www.draftkings.com/wallet",
                "https://sportsbook.draftkings.com/wallet"
            ]

            for url in balance_urls:
                try:
                    self.logger.info(f"Trying balance URL: {url}")

                    await self.page.goto(url, timeout=30000)

                    title = await self.page.title()
                    current_url = self.page.url

                    self.logger.info(f"✅ Page loaded: {title}")
                    self.logger.info(f"Current URL: {current_url}")

                    # Take screenshot
                    await self.page.screenshot(path=f"balance_page_{url.split('/')[-1]}.png")

                    # Extract balance
                    balance = await self.extract_balance_from_page()

                    if balance:
                        self.logger.info(f"✅ Balance extracted: {balance}")
                        return balance
                    else:
                        self.logger.warning(f"⚠️ No balance found on {url}")

                except Exception as e:
                    self.logger.warning(f"Failed to load {url}: {e}")
                    continue

            self.logger.error("❌ Could not extract balance from any page")
            return None

        except Exception as e:
            self.logger.error(f"❌ Session loading failed: {e}")
            return None

    async def extract_balance_from_page(self) -> str:
        """Extract balance from current page."""
        try:
            # Wait for page to load
            await self.page.wait_for_timeout(random.uniform(2000, 4000))

            # Look for balance elements with comprehensive selectors
            balance_selectors = [
                '[class*="balance"]',
                '[class*="account"]',
                '[class*="wallet"]',
                '[class*="funds"]',
                '[data-testid*="balance"]',
                '[data-testid*="account"]',
                '[data-testid*="wallet"]',
                'span:has-text("$")',
                'div:has-text("$")',
                '[class*="amount"]',
                '[class*="money"]',
                '[class*="balance-amount"]',
                '[class*="account-balance"]',
                '[class*="wallet-balance"]'
            ]

            balance_text = None

            # Try to find balance using selectors
            for selector in balance_selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    for element in elements:
                        text = await element.text_content()
                        if text and '$' in text:
                            # Extract numeric value
                            match = re.search(r'\$[\d,]+\.?\d*', text)
                            if match:
                                balance_text = match.group()
                                self.logger.info(f"Found balance: {balance_text}")
                                break
                    if balance_text:
                        break
                except Exception as e:
                    self.logger.debug(f"Balance selector failed: {selector} - {e}")
                    continue

            # If no balance found with selectors, try regex on page content
            if not balance_text:
                self.logger.info("Trying regex extraction on page content...")
                content = await self.page.text_content("body")

                # Look for balance patterns
                balance_patterns = [
                    r'\$[\d,]+\.?\d*',
                    r'[\d,]+\.?\d*\s*USD',
                    r'Balance:\s*\$[\d,]+\.?\d*',
                    r'Account:\s*\$[\d,]+\.?\d*',
                    r'Wallet:\s*\$[\d,]+\.?\d*',
                    r'Funds:\s*\$[\d,]+\.?\d*'
                ]

                for pattern in balance_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        balance_text = matches[0]
                        self.logger.info(f"Found balance via regex: {balance_text}")
                        break

            return balance_text

        except Exception as e:
            self.logger.error(f"❌ Balance extraction failed: {e}")
            return None

    async def take_screenshot(self, filename: str) -> bool:
        """Take screenshot for debugging."""
        try:
            if self.page:
                await self.page.screenshot(path=filename)
                self.logger.info(f"✅ Screenshot saved: {filename}")
                return True
        except Exception as e:
            self.logger.error(f"❌ Screenshot failed: {e}")
        return False

    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.page:
                await self.page.close()
                self.logger.info("Page closed")

            if self.context:
                await self.context.close()
                self.logger.info("Context closed")

            if self.browser:
                await self.browser.close()
                self.logger.info("Browser closed")

            if self.playwright:
                await self.playwright.stop()
                self.logger.info("Playwright stopped")

        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")

    async def monitor_balance_manual(self) -> str:
        """Main balance monitoring function with manual login."""
        try:
            self.logger.info("🔐 DraftKings Balance Monitor - Manual Login")
            self.logger.info("=" * 50)
            self.logger.info(f"🔐 Monitoring account: {self.username}")
            self.logger.info("=" * 50)

            # Check if session exists
            if os.path.exists(self.session_file):
                self.logger.info("✅ Found existing session, attempting automated extraction...")
                balance = await self.load_session_and_extract_balance()

                if balance:
                    self.logger.info(f"💰 Balance extracted: {balance}")
                    return balance
                else:
                    self.logger.warning("⚠️ Automated extraction failed, starting manual login...")

            # Manual login setup
            if not await self.manual_login_setup():
                self.logger.error("❌ Manual login setup failed")
                return None

            # Extract balance after manual login
            balance = await self.extract_balance_from_page()

            if balance:
                self.logger.info(f"💰 Balance extracted: {balance}")
                return balance
            else:
                self.logger.error("❌ Failed to extract balance")
                return None

        except Exception as e:
            self.logger.error(f"❌ Balance monitoring failed: {e}")
            return None
        finally:
            await self.cleanup()


def main():
    """Main function."""
    print("🔐 DraftKings Balance Monitor - Manual Login")
    print("=" * 50)

    # DraftKings credentials
    username = "paxtonedgar3@gmail.com"
    password = "Empireozarks@2013"

    # Create monitor
    monitor = DraftKingsManualLoginMonitor(username, password)

    # Run balance monitoring
    balance = asyncio.run(monitor.monitor_balance_manual())

    print("\n" + "=" * 50)
    print("MANUAL LOGIN RESULTS")
    print("=" * 50)

    if balance:
        print(f"✅ SUCCESS: Balance extracted: {balance}")
        print("🔐 Manual login + automated extraction worked!")
    else:
        print("❌ FAILED: Could not extract balance")
        print("💡 Check screenshots and try again")


if __name__ == "__main__":
    main()
