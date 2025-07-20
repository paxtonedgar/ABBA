"""
DraftKings Balance Monitor with BrowserBase Stealth Mode
Uses BrowserBase's built-in stealth capabilities for enhanced anti-detection
"""

import asyncio
import logging
import os
import re

from browserbase import Browserbase
from playwright.async_api import async_playwright


class DraftKingsStealthMonitor:
    """DraftKings balance monitor using BrowserBase stealth mode."""

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.api_key = os.getenv("BROWSERBASE_API_KEY")
        self.project_id = os.getenv("BROWSERBASE_PROJECT_ID")
        self.session = None
        self.browser = None
        self.page = None

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    async def create_stealth_session(self) -> bool:
        """Create BrowserBase session with stealth mode enabled."""
        try:
            self.logger.info("Creating BrowserBase session with stealth mode...")
            bb = Browserbase(api_key=self.api_key)

            # Create session with stealth mode enabled
            self.session = bb.sessions.create(
                project_id=self.project_id,
                # Enable stealth mode with advanced settings
                stealth_mode=True,
                # Additional stealth options
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080},
                timezone="America/New_York",
                locale="en-US",
                # Enable additional stealth features
                webgl_vendor="Intel Inc.",
                webgl_renderer="Intel Iris OpenGL Engine",
                # Disable automation indicators
                webdriver_mode=False,
                # Enable realistic browser behavior
                realistic_mouse_movements=True,
                realistic_typing=True,
                # Additional privacy settings
                do_not_track=True,
                # Enable fingerprint randomization
                fingerprint_randomization=True
            )

            self.logger.info(f"✅ Stealth session created: {self.session.id}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to create stealth session: {e}")
            return False

    async def setup_stealth_browser(self) -> bool:
        """Setup Playwright browser with stealth session."""
        try:
            if not self.session:
                if not await self.create_stealth_session():
                    return False

            self.logger.info("Setting up stealth browser...")

            async with async_playwright() as p:
                # Connect to BrowserBase with stealth session
                self.browser = await p.chromium.connect_over_cdp(self.session.connect_url)
                self.logger.info("✅ Connected to BrowserBase stealth session!")

                # Create page (stealth settings already applied by BrowserBase)
                self.page = await self.browser.new_page()
                self.logger.info("✅ Stealth page created!")

                # Additional stealth enhancements
                await self.page.add_init_script("""
                    // Additional stealth overrides
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
                    
                    // Override chrome
                    Object.defineProperty(window, 'chrome', {
                        get: () => ({
                            runtime: {},
                        }),
                    });
                    
                    // Override automation indicators
                    delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
                    delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
                    delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
                """)

                self.logger.info("✅ Additional stealth enhancements applied!")
                return True

        except Exception as e:
            self.logger.error(f"❌ Failed to setup stealth browser: {e}")
            return False

    async def navigate_to_draftkings(self) -> bool:
        """Navigate to DraftKings with stealth mode."""
        try:
            if not self.page:
                if not await self.setup_stealth_browser():
                    return False

            self.logger.info("Navigating to DraftKings with stealth mode...")

            # Try different DraftKings URLs with stealth
            draftkings_urls = [
                "https://www.draftkings.com/login",
                "https://sportsbook.draftkings.com/login",
                "https://www.draftkings.com/signin",
                "https://sportsbook.draftkings.com/signin",
                "https://www.draftkings.com",
                "https://sportsbook.draftkings.com"
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

                        # Look for login buttons with stealth
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
        """Login to DraftKings account with stealth mode."""
        try:
            if not self.page:
                self.logger.error("No page available for login")
                return False

            self.logger.info("Attempting to login to DraftKings with stealth mode...")

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

            # Wait for page to fully load with stealth
            await self.page.wait_for_timeout(5000)

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
                await self.take_screenshot("stealth_login_no_username.png")
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
                await self.take_screenshot("stealth_login_no_password.png")
                return False

            # Take screenshot before filling credentials
            await self.take_screenshot("stealth_login_before_fill.png")

            # Fill credentials with human-like delays (stealth mode)
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
            await self.take_screenshot("stealth_login_after_fill.png")

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
                await self.take_screenshot("stealth_login_no_submit.png")
                return False

            # Click submit
            self.logger.info("Clicking submit button...")
            await submit_button.click()

            # Wait for navigation
            await self.page.wait_for_timeout(5000)

            # Take screenshot after submit
            await self.take_screenshot("stealth_login_after_submit.png")

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
            await self.take_screenshot("stealth_login_error.png")
            return False

    async def extract_balance(self) -> str:
        """Extract balance from DraftKings account."""
        try:
            if not self.page:
                self.logger.error("No page available for balance extraction")
                return None

            self.logger.info("Extracting balance from DraftKings...")

            # Wait for page to load
            await self.page.wait_for_timeout(3000)

            # Take screenshot before extraction
            await self.take_screenshot("stealth_balance_before.png")

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
                '[class*="money"]'
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
                    r'Account:\s*\$[\d,]+\.?\d*'
                ]

                for pattern in balance_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        balance_text = matches[0]
                        self.logger.info(f"Found balance via regex: {balance_text}")
                        break

            # Take screenshot after extraction
            await self.take_screenshot("stealth_balance_after.png")

            if balance_text:
                self.logger.info(f"✅ Balance extracted: {balance_text}")
                return balance_text
            else:
                self.logger.warning("⚠️ No balance found")
                return None

        except Exception as e:
            self.logger.error(f"❌ Balance extraction failed: {e}")
            await self.take_screenshot("stealth_balance_error.png")
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

            if self.browser:
                await self.browser.close()
                self.logger.info("Browser closed")

            if self.session:
                # Note: BrowserBase sessions auto-cleanup
                self.logger.info("Session cleanup initiated")

        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")

    async def monitor_balance(self) -> str:
        """Main balance monitoring function with stealth mode."""
        try:
            self.logger.info("🧪 DraftKings Balance Monitor - Stealth Mode")
            self.logger.info("=" * 50)
            self.logger.info(f"🔐 Monitoring account: {self.username}")
            self.logger.info("🕵️ Using BrowserBase stealth mode")
            self.logger.info("=" * 50)

            # Navigate to DraftKings
            if not await self.navigate_to_draftkings():
                self.logger.error("❌ Failed to navigate to DraftKings")
                return None

            # Take screenshot before login
            await self.take_screenshot("stealth_before_login.png")

            # Login to DraftKings
            if not await self.login_to_draftkings():
                self.logger.error("❌ Failed to login to DraftKings")
                await self.take_screenshot("stealth_login_failed.png")
                return None

            # Extract balance
            balance = await self.extract_balance()

            if balance:
                self.logger.info(f"💰 Balance: {balance}")
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
    print("🧪 DraftKings Balance Monitor - Stealth Mode")
    print("=" * 50)

    # DraftKings credentials
    username = "paxtonedgar3@gmail.com"
    password = "Empireozarks@2013"

    # Create monitor
    monitor = DraftKingsStealthMonitor(username, password)

    # Run balance monitoring
    balance = asyncio.run(monitor.monitor_balance())

    print("\n" + "=" * 50)
    print("STEALTH MODE RESULTS")
    print("=" * 50)

    if balance:
        print(f"✅ SUCCESS: Balance extracted: {balance}")
    else:
        print("❌ FAILED: Could not extract balance")
        print("💡 Check screenshots for debugging")


if __name__ == "__main__":
    main()
