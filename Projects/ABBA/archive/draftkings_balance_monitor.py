"""
DraftKings Balance Monitor - Production Ready
Complete balance monitoring solution with stealth anti-detection
"""

import asyncio
import logging
import os
import re
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
    status: str = "success"
    message: str = ""

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class DraftKingsBalanceMonitor:
    """Production-ready DraftKings balance monitor with stealth capabilities."""

    def __init__(self, username: str, password: str, api_key: str | None = None, project_id: str | None = None):
        """Initialize the balance monitor."""
        self.username = username
        self.password = password
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

    async def setup_stealth_browser(self) -> bool:
        """Set up browser with stealth anti-detection."""
        try:
            if not self.session:
                if not await self.create_session():
                    return False

            self.logger.info("Setting up stealth browser...")
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.connect_over_cdp(self.session.connect_url)
            self.logger.info("✅ Connected to BrowserBase via Playwright!")

            # Create page with stealth settings
            self.page = await self.browser.new_page()

            # Set stealth user agent and viewport
            await self.page.set_extra_http_headers({
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })

            await self.page.set_viewport_size({"width": 1920, "height": 1080})

            # Add stealth scripts
            await self.page.add_init_script("""
                // Remove webdriver property
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
                
                // Override permissions
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
                
                // Override plugins
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5],
                });
                
                // Override languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en'],
                });
            """)

            self.logger.info("✅ Stealth settings applied!")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to setup stealth browser: {e}")
            return False

    async def navigate_to_draftkings(self) -> bool:
        """Navigate to DraftKings with stealth."""
        try:
            if not self.page:
                if not await self.setup_stealth_browser():
                    return False

            self.logger.info("Navigating to DraftKings with stealth...")

            # Try different DraftKings URLs, prioritizing login page
            draftkings_urls = [
                "https://www.draftkings.com/login",
                "https://sportsbook.draftkings.com/login",
                "https://www.draftkings.com/signin",
                "https://sportsbook.draftkings.com/signin",
                "https://www.draftkings.com",
                "https://sportsbook.draftkings.com",
                "https://www.draftkings.com/sportsbook"
            ]

            for url in draftkings_urls:
                try:
                    self.logger.info(f"Trying URL: {url}")
                    await self.page.goto(url, timeout=30000, wait_until="domcontentloaded")
                    await self.page.wait_for_timeout(3000)

                    title = await self.page.title()
                    current_url = self.page.url
                    self.logger.info(f"✅ Page loaded: {title}")
                    self.logger.info(f"Current URL: {current_url}")

                    # Check if we're on a login page
                    if "login" in current_url.lower() or "signin" in current_url.lower():
                        self.logger.info("🎯 Successfully navigated to login page!")
                        return True

                    # If we're on the main page, try to find and click login button
                    if "draftkings" in title.lower() and ("login" not in current_url.lower() and "signin" not in current_url.lower()):
                        self.logger.info("🏠 On main page, looking for login button...")

                        # Look for login buttons
                        login_button_selectors = [
                            'a[href*="login"]',
                            'a[href*="signin"]',
                            'button:has-text("Login")',
                            'button:has-text("Sign In")',
                            'button:has-text("Log In")',
                            'a:has-text("Login")',
                            'a:has-text("Sign In")',
                            '[class*="login"]',
                            '[class*="signin"]'
                        ]

                        for selector in login_button_selectors:
                            try:
                                login_button = await self.page.query_selector(selector)
                                if login_button:
                                    self.logger.info(f"Found login button: {selector}")
                                    await login_button.click()
                                    await self.page.wait_for_timeout(3000)

                                    # Check if we're now on login page
                                    new_url = self.page.url
                                    if "login" in new_url.lower() or "signin" in new_url.lower():
                                        self.logger.info("✅ Successfully clicked to login page!")
                                        return True
                                    else:
                                        self.logger.info(f"Clicked button but still on: {new_url}")
                            except Exception as e:
                                self.logger.debug(f"Login button selector failed: {selector} - {e}")
                                continue

                        # If no login button found, try direct navigation to login
                        self.logger.info("No login button found, trying direct login URL...")
                        try:
                            await self.page.goto("https://www.draftkings.com/login", timeout=15000, wait_until="domcontentloaded")
                            await self.page.wait_for_timeout(3000)

                            login_url = self.page.url
                            if "login" in login_url.lower() or "signin" in login_url.lower():
                                self.logger.info("✅ Direct login navigation successful!")
                                return True
                        except Exception as e:
                            self.logger.warning(f"Direct login navigation failed: {e}")

                    # If we found a working page, return True
                    if "draftkings" in title.lower():
                        self.logger.info("✅ DraftKings page loaded successfully")
                        return True

                except Exception as e:
                    self.logger.warning(f"Failed to navigate to {url}: {e}")
                    continue

            self.logger.error("❌ All navigation attempts failed")
            return False

        except Exception as e:
            self.logger.error(f"❌ Navigation failed: {e}")
            return False

    async def login_to_draftkings(self) -> bool:
        """Login to DraftKings account."""
        try:
            if not self.page:
                self.logger.error("No page available for login")
                return False

            self.logger.info("Attempting to login to DraftKings...")

            # First, ensure we're on a login page
            current_url = self.page.url
            if "login" not in current_url.lower() and "signin" not in current_url.lower():
                self.logger.warning("Not on login page, attempting to navigate to login...")
                try:
                    await self.page.goto("https://www.draftkings.com/login", timeout=15000, wait_until="domcontentloaded")
                    await self.page.wait_for_timeout(3000)
                except Exception as e:
                    self.logger.error(f"Failed to navigate to login page: {e}")
                    return False

            # Look for login elements with comprehensive selectors
            login_selectors = {
                'username': [
                    'input[type="email"]',
                    'input[name="username"]',
                    'input[name="email"]',
                    'input[id*="username"]',
                    'input[id*="email"]',
                    'input[placeholder*="email"]',
                    'input[placeholder*="Email"]',
                    'input[placeholder*="username"]',
                    'input[placeholder*="Username"]',
                    'input[data-testid*="email"]',
                    'input[data-testid*="username"]'
                ],
                'password': [
                    'input[type="password"]',
                    'input[name="password"]',
                    'input[id*="password"]',
                    'input[placeholder*="password"]',
                    'input[placeholder*="Password"]',
                    'input[data-testid*="password"]'
                ],
                'submit': [
                    'button[type="submit"]',
                    'input[type="submit"]',
                    'button:has-text("Login")',
                    'button:has-text("Sign In")',
                    'button:has-text("Log In")',
                    'button[class*="login"]',
                    'button[class*="signin"]',
                    'button[data-testid*="login"]',
                    'button[data-testid*="signin"]'
                ]
            }

            # Find and fill username
            username_field = None
            for selector in login_selectors['username']:
                try:
                    username_field = await self.page.query_selector(selector)
                    if username_field:
                        self.logger.info(f"Found username field: {selector}")
                        break
                except:
                    continue

            if not username_field:
                self.logger.error("❌ Username field not found")
                # Take screenshot for debugging
                await self.take_screenshot("login_page_no_username.png")
                return False

            # Find and fill password
            password_field = None
            for selector in login_selectors['password']:
                try:
                    password_field = await self.page.query_selector(selector)
                    if password_field:
                        self.logger.info(f"Found password field: {selector}")
                        break
                except:
                    continue

            if not password_field:
                self.logger.error("❌ Password field not found")
                # Take screenshot for debugging
                await self.take_screenshot("login_page_no_password.png")
                return False

            # Take screenshot before filling credentials
            await self.take_screenshot("login_page_before_fill.png")

            # Fill credentials with human-like delays
            self.logger.info("Filling username...")
            await username_field.click()
            await self.page.wait_for_timeout(500)
            await username_field.fill(self.username)
            await self.page.wait_for_timeout(1000)

            self.logger.info("Filling password...")
            await password_field.click()
            await self.page.wait_for_timeout(500)
            await password_field.fill(self.password)
            await self.page.wait_for_timeout(1000)

            # Take screenshot after filling credentials
            await self.take_screenshot("login_page_after_fill.png")

            # Find and click submit button
            submit_button = None
            for selector in login_selectors['submit']:
                try:
                    submit_button = await self.page.query_selector(selector)
                    if submit_button:
                        self.logger.info(f"Found submit button: {selector}")
                        break
                except:
                    continue

            if not submit_button:
                self.logger.error("❌ Submit button not found")
                # Take screenshot for debugging
                await self.take_screenshot("login_page_no_submit.png")
                return False

            # Click submit
            self.logger.info("Clicking submit button...")
            await submit_button.click()

            # Wait for navigation
            await self.page.wait_for_timeout(5000)

            # Take screenshot after submit
            await self.take_screenshot("login_page_after_submit.png")

            # Check if login was successful
            current_url = self.page.url
            title = await self.page.title()

            self.logger.info(f"Post-login URL: {current_url}")
            self.logger.info(f"Post-login title: {title}")

            if "login" not in current_url.lower() and "signin" not in current_url.lower():
                self.logger.info("✅ Login appears successful")
                return True
            else:
                self.logger.error("❌ Login may have failed - still on login page")
                return False

        except Exception as e:
            self.logger.error(f"❌ Login failed: {e}")
            # Take screenshot for debugging
            await self.take_screenshot("login_page_error.png")
            return False

    async def extract_balance_info(self) -> BalanceInfo:
        """Extract balance information from DraftKings."""
        balance_info = BalanceInfo()

        try:
            if not self.page:
                balance_info.status = "error"
                balance_info.message = "No page available for balance extraction"
                return balance_info

            self.logger.info("Extracting balance information...")

            # Navigate to account page
            try:
                await self.page.goto("https://www.draftkings.com/account", timeout=15000)
                await self.page.wait_for_load_state("domcontentloaded", timeout=10000)
                self.logger.info("✅ Navigated to account page")
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
                r'[\d,]+\.?\d*\s*USD',  # 1,234.56 USD
            ]

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

            if numeric_balances:
                # Use the highest balance as account balance
                balance_info.account_balance = max(numeric_balances)
                balance_info.available_balance = max(numeric_balances)
                balance_info.status = "success"
                balance_info.message = f"Balance extracted: ${balance_info.account_balance:,.2f}"

                self.logger.info(f"💰 {balance_info.message}")
            else:
                balance_info.status = "warning"
                balance_info.message = "No balance information found"
                self.logger.warning("No balance information found")

            return balance_info

        except Exception as e:
            balance_info.status = "error"
            balance_info.message = f"Balance extraction failed: {e}"
            self.logger.error(f"❌ {balance_info.message}")
            return balance_info

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

            if hasattr(self, 'playwright') and self.playwright:
                await self.playwright.stop()
                self.logger.info("Playwright stopped")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


async def monitor_draftkings_balance(username: str, password: str) -> BalanceInfo:
    """Monitor DraftKings balance with full automation."""
    print("💰 DraftKings Balance Monitor")
    print("=" * 50)

    monitor = DraftKingsBalanceMonitor(username, password)

    try:
        # Navigate to DraftKings
        if not await monitor.navigate_to_draftkings():
            balance_info = BalanceInfo(status="error", message="Failed to navigate to DraftKings")
            return balance_info

        # Take screenshot before login
        await monitor.take_screenshot("draftkings_before_login.png")

        # Login to DraftKings
        if not await monitor.login_to_draftkings():
            balance_info = BalanceInfo(status="error", message="Failed to login to DraftKings")
            await monitor.take_screenshot("draftkings_login_failed.png")
            return balance_info

        # Take screenshot after login
        await monitor.take_screenshot("draftkings_after_login.png")

        # Extract balance information
        balance_info = await monitor.extract_balance_info()

        # Take final screenshot
        await monitor.take_screenshot("draftkings_balance_extracted.png")

        return balance_info

    except Exception as e:
        balance_info = BalanceInfo(status="error", message=f"Monitor failed: {e}")
        return balance_info

    finally:
        await monitor.close()


def main():
    """Main function."""
    print("🧪 DraftKings Balance Monitor - Production Ready")
    print("=" * 50)

    # Set credentials
    username = "paxtonedgar3@gmail.com"
    password = "Empireozarks@2013"

    print(f"🔐 Monitoring account: {username}")
    print("🕵️ Using stealth anti-detection techniques")
    print("=" * 50)

    balance_info = asyncio.run(monitor_draftkings_balance(username, password))

    print("\n" + "=" * 50)
    print("BALANCE MONITOR RESULTS")
    print("=" * 50)

    if balance_info.status == "success":
        print("✅ Balance monitoring successful!")
        print(f"💰 Account Balance: ${balance_info.account_balance:,.2f}")
        print(f"💰 Available Balance: ${balance_info.available_balance:,.2f}")
        print(f"💰 Currency: {balance_info.currency}")
        print(f"💰 Source: {balance_info.source}")
        print(f"💰 Timestamp: {balance_info.timestamp}")
        print("\n🎉 Production ready!")
    elif balance_info.status == "warning":
        print("⚠️ Balance monitoring completed with warnings")
        print(f"💡 Message: {balance_info.message}")
        print("\n🔧 Check screenshots for debugging")
    else:
        print("❌ Balance monitoring failed!")
        print(f"💡 Error: {balance_info.message}")
        print("\n🔧 Check screenshots for debugging")


if __name__ == "__main__":
    main()
