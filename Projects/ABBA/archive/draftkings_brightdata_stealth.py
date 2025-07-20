"""
DraftKings Balance Monitor - Bright Data Residential Proxy Integration
Enhanced with Bright Data residential proxies for immediate form loading testing
"""

import asyncio
import logging
import os
import random
import re

from playwright.async_api import Page, async_playwright

# Try to import undetected-playwright if available
try:
    from undetected_playwright import stealth_async
    UNDETECTED_AVAILABLE = True
except ImportError:
    UNDETECTED_AVAILABLE = False
    print("⚠️ undetected-playwright not installed. Install with: pip install undetected-playwright")


class DraftKingsBrightDataMonitor:
    """DraftKings balance monitor using Bright Data residential proxies."""

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Bright Data residential proxy configuration
        self.brightdata_config = {
            'username': os.getenv('BRIGHTDATA_USERNAME', 'brd-customer-hl_1234567890-zone-residential'),
            'password': os.getenv('BRIGHTDATA_PASSWORD', 'your_brightdata_password'),
            'host': os.getenv('BRIGHTDATA_HOST', 'brd.superproxy.io'),
            'port': os.getenv('BRIGHTDATA_PORT', '22225'),
            'enabled': os.getenv('USE_BRIGHTDATA', 'true').lower() == 'true'
        }

        # Alternative proxy providers for testing
        self.proxy_configs = {
            'brightdata': {
                'server': f"http://{self.brightdata_config['username']}:{self.brightdata_config['password']}@{self.brightdata_config['host']}:{self.brightdata_config['port']}",
                'username': self.brightdata_config['username'],
                'password': self.brightdata_config['password']
            },
            'oxylabs': {
                'server': os.getenv('OXYLABS_PROXY', 'http://username:password@proxy.oxylabs.io:60000'),
                'username': os.getenv('OXYLABS_USERNAME', ''),
                'password': os.getenv('OXYLABS_PASSWORD', '')
            },
            'smartproxy': {
                'server': os.getenv('SMARTPROXY_SERVER', 'http://username:password@gate.smartproxy.com:7000'),
                'username': os.getenv('SMARTPROXY_USERNAME', ''),
                'password': os.getenv('SMARTPROXY_PASSWORD', '')
            }
        }

    async def create_brightdata_context(self) -> tuple:
        """Create context with Bright Data residential proxy."""
        try:
            self.logger.info("Creating Bright Data residential proxy context...")

            # Start Playwright
            self.playwright = await async_playwright().start()

            # Configure proxy based on availability
            proxy_config = None
            if self.brightdata_config['enabled'] and self.brightdata_config['username'] != 'brd-customer-hl_1234567890-zone-residential':
                proxy_config = self.proxy_configs['brightdata']
                self.logger.info("🌐 Using Bright Data residential proxy")
            elif os.getenv('OXYLABS_USERNAME'):
                proxy_config = self.proxy_configs['oxylabs']
                self.logger.info("🌐 Using Oxylabs residential proxy")
            elif os.getenv('SMARTPROXY_USERNAME'):
                proxy_config = self.proxy_configs['smartproxy']
                self.logger.info("🌐 Using SmartProxy residential proxy")
            else:
                self.logger.warning("⚠️ No residential proxy configured, using direct connection")

            # Launch browser with proxy if available
            launch_options = {
                'headless': False,  # Avoid headless detection
                'args': [
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--disable-gpu',
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding',
                    '--disable-features=TranslateUI',
                    '--disable-ipc-flooding-protection',
                    '--disable-default-apps',
                    '--disable-extensions',
                    '--disable-plugins',
                    '--disable-blink-features=AutomationControlled'
                ]
            }

            if proxy_config:
                launch_options['proxy'] = {
                    'server': proxy_config['server'],
                    'username': proxy_config['username'],
                    'password': proxy_config['password']
                }

            self.browser = await self.playwright.chromium.launch(**launch_options)

            # Create context with advanced stealth settings
            context_options = {
                'viewport': {'width': 1920, 'height': 1080},
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'locale': 'en-US',
                'timezone_id': 'America/New_York',
                'geolocation': {'latitude': 40.7128, 'longitude': -74.0060},  # NYC for US betting
                'permissions': ['geolocation'],
                'extra_http_headers': {
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-User': '?1',
                    'Cache-Control': 'max-age=0'
                }
            }

            self.context = await self.browser.new_context(**context_options)

            # Apply undetected-playwright if available
            if UNDETECTED_AVAILABLE:
                self.logger.info("🕵️ Applying undetected-playwright stealth...")
                await stealth_async(self.context)
            else:
                self.logger.info("⚠️ Using standard stealth techniques...")

            # Apply additional stealth enhancements
            await self.context.add_init_script("""
                // Bright Data enhanced stealth overrides
                
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
                
                // Additional stealth measures
                Object.defineProperty(navigator, 'hardwareConcurrency', {
                    get: () => 8,
                });
                
                Object.defineProperty(navigator, 'deviceMemory', {
                    get: () => 8,
                });
                
                // Override screen properties
                Object.defineProperty(screen, 'colorDepth', {
                    get: () => 24,
                });
                
                Object.defineProperty(screen, 'pixelDepth', {
                    get: () => 24,
                });
                
                // Override WebGL
                const getParameter = WebGLRenderingContext.prototype.getParameter;
                WebGLRenderingContext.prototype.getParameter = function(parameter) {
                    if (parameter === 37445) {
                        return 'Intel Inc.';
                    }
                    if (parameter === 37446) {
                        return 'Intel Iris OpenGL Engine';
                    }
                    return getParameter.call(this, parameter);
                };
                
                // Override canvas fingerprinting
                const originalGetContext = HTMLCanvasElement.prototype.getContext;
                HTMLCanvasElement.prototype.getContext = function(type, ...args) {
                    const context = originalGetContext.call(this, type, ...args);
                    if (type === '2d') {
                        const originalFillText = context.fillText;
                        context.fillText = function(...args) {
                            return originalFillText.apply(this, args);
                        };
                    }
                    return context;
                };
                
                // Override audio fingerprinting
                const originalGetChannelData = AudioBuffer.prototype.getChannelData;
                AudioBuffer.prototype.getChannelData = function(channel) {
                    const data = originalGetChannelData.call(this, channel);
                    return data;
                };
                
                // Override battery API
                if (navigator.getBattery) {
                    navigator.getBattery = function() {
                        return Promise.resolve({
                            charging: true,
                            chargingTime: Infinity,
                            dischargingTime: Infinity,
                            level: 1
                        });
                    };
                }
                
                // Override connection API
                if (navigator.connection) {
                    Object.defineProperty(navigator, 'connection', {
                        get: () => ({
                            effectiveType: '4g',
                            rtt: 50,
                            downlink: 10,
                            saveData: false
                        }),
                    });
                }
                
                // Override media devices
                if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
                    const originalEnumerateDevices = navigator.mediaDevices.enumerateDevices;
                    navigator.mediaDevices.enumerateDevices = function() {
                        return Promise.resolve([
                            {
                                deviceId: 'default',
                                kind: 'audioinput',
                                label: 'Default - MacBook Pro Microphone',
                                groupId: 'group1'
                            },
                            {
                                deviceId: 'default',
                                kind: 'audiooutput',
                                label: 'Default - MacBook Pro Speakers',
                                groupId: 'group1'
                            }
                        ]);
                    };
                }
                
                // Override timezone
                Object.defineProperty(Intl, 'DateTimeFormat', {
                    get: () => function(locale, options) {
                        return new Date().toLocaleString('en-US', {timeZone: 'America/New_York'});
                    },
                });
                
                // Override performance timing
                const originalGetEntries = Performance.prototype.getEntries;
                Performance.prototype.getEntries = function() {
                    const entries = originalGetEntries.call(this);
                    return entries.filter(entry => !entry.name.includes('chrome-extension'));
                };
                
                // Override console methods to avoid detection
                const originalLog = console.log;
                const originalWarn = console.warn;
                const originalError = console.error;
                
                console.log = function(...args) {
                    if (args.some(arg => typeof arg === 'string' && arg.includes('webdriver'))) {
                        return;
                    }
                    return originalLog.apply(this, args);
                };
                
                console.warn = function(...args) {
                    if (args.some(arg => typeof arg === 'string' && arg.includes('webdriver'))) {
                        return;
                    }
                    return originalWarn.apply(this, args);
                };
                
                console.error = function(...args) {
                    if (args.some(arg => typeof arg === 'string' && arg.includes('webdriver'))) {
                        return;
                    }
                    return originalError.apply(this, args);
                };
                
                // Additional CDP evasion
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
                
                // Override automation detection
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
                
                // Override automation properties
                Object.defineProperty(navigator, 'automation', {
                    get: () => undefined,
                });
                
                // Override automation detection methods
                if (window.navigator.permissions) {
                    const originalQuery = window.navigator.permissions.query;
                    window.navigator.permissions.query = (parameters) => (
                        parameters.name === 'notifications' ?
                            Promise.resolve({ state: Notification.permission }) :
                            originalQuery(parameters)
                    );
                }
            """)

            self.logger.info("✅ Bright Data residential proxy context created!")
            return self.context, self.browser

        except Exception as e:
            self.logger.error(f"❌ Failed to create Bright Data context: {e}")
            return None, None

    async def human_like_interaction(self, page: Page):
        """Simulate human-like behavior."""
        try:
            # Random mouse movement
            await page.mouse.move(
                random.randint(100, 800),
                random.randint(100, 600)
            )
            await asyncio.sleep(random.uniform(0.5, 1.5))

            # Random scroll
            await page.evaluate("window.scrollBy(0, window.innerHeight / 2);")
            await asyncio.sleep(random.uniform(1.0, 2.0))

            # Random scroll back
            await page.evaluate("window.scrollBy(0, -window.innerHeight / 4);")
            await asyncio.sleep(random.uniform(0.5, 1.0))

            self.logger.info("✅ Human-like interactions applied")

        except Exception as e:
            self.logger.warning(f"⚠️ Human-like interaction failed: {e}")

    async def navigate_to_draftkings_brightdata(self) -> bool:
        """Navigate to DraftKings with Bright Data residential proxy."""
        try:
            if not self.context:
                context, browser = await self.create_brightdata_context()
                if not context:
                    return False
                self.context = context
                self.browser = browser

            self.logger.info("Navigating to DraftKings with Bright Data residential proxy...")

            # Create page
            self.page = await self.context.new_page()
            self.logger.info("✅ Bright Data page created!")

            # Try different DraftKings URLs with residential proxy
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

                    # Navigate with residential proxy
                    await self.page.goto(url, timeout=60000, wait_until="domcontentloaded")

                    # Apply human-like interactions
                    await self.human_like_interaction(self.page)

                    # Wait for page to load with realistic timing
                    await self.page.wait_for_timeout(random.uniform(3000, 5000))

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

                        # Look for login buttons with residential proxy
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

                                    # Click with human-like behavior
                                    await login_button.hover()
                                    await self.page.wait_for_timeout(random.uniform(200, 500))
                                    await login_button.click()
                                    await self.page.wait_for_timeout(random.uniform(3000, 5000))

                                    # Apply human-like interactions after click
                                    await self.human_like_interaction(self.page)

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
                            await self.page.goto("https://www.draftkings.com/login", timeout=30000, wait_until="domcontentloaded")
                            await self.human_like_interaction(self.page)
                            await self.page.wait_for_timeout(random.uniform(3000, 5000))

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

                    # If we successfully loaded a page, break and don't try more URLs
                    break

                except Exception as e:
                    self.logger.warning(f"Failed to navigate to {url}: {e}")
                    continue

            if self.page and self.page.url:
                self.logger.info(f"✅ Successfully navigated to: {self.page.url}")
                return True
            else:
                self.logger.error("❌ All navigation attempts failed")
                return False

        except Exception as e:
            self.logger.error(f"❌ Navigation failed: {e}")
            return False

    async def login_to_draftkings_brightdata(self) -> bool:
        """Login to DraftKings account with Bright Data residential proxy."""
        try:
            if not self.page:
                self.logger.error("No page available for login")
                return False

            self.logger.info("Attempting to login to DraftKings with Bright Data residential proxy...")

            # First, ensure we're on a login page
            current_url = self.page.url
            if "login" not in current_url.lower() and "signin" not in current_url.lower():
                self.logger.warning("Not on login page, attempting to navigate to login...")
                try:
                    await self.page.goto("https://www.draftkings.com/login", timeout=30000, wait_until="domcontentloaded")
                    await self.human_like_interaction(self.page)
                    await self.page.wait_for_timeout(random.uniform(3000, 5000))
                except Exception as e:
                    self.logger.error(f"Failed to navigate to login page: {e}")
                    return False

            # Wait for page to fully load with residential proxy
            await self.page.wait_for_timeout(random.uniform(5000, 8000))

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
                await self.take_screenshot("brightdata_login_no_username.png")
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
                await self.take_screenshot("brightdata_login_no_password.png")
                return False

            # Take screenshot before filling credentials
            await self.take_screenshot("brightdata_login_before_fill.png")

            # Fill credentials with human-like delays (residential proxy)
            self.logger.info("Filling username...")
            await username_field.hover()
            await self.page.wait_for_timeout(random.uniform(200, 500))
            await username_field.click()
            await self.page.wait_for_timeout(random.uniform(500, 1000))

            # Type username character by character for realism
            for char in self.username:
                await username_field.type(char)
                await self.page.wait_for_timeout(random.uniform(50, 150))  # Random delay

            await self.page.wait_for_timeout(random.uniform(1000, 2000))

            self.logger.info("Filling password...")
            await password_field.hover()
            await self.page.wait_for_timeout(random.uniform(200, 500))
            await password_field.click()
            await self.page.wait_for_timeout(random.uniform(500, 1000))

            # Type password character by character for realism
            for char in self.password:
                await password_field.type(char)
                await self.page.wait_for_timeout(random.uniform(50, 150))  # Random delay

            await self.page.wait_for_timeout(random.uniform(1000, 2000))

            # Take screenshot after filling credentials
            await self.take_screenshot("brightdata_login_after_fill.png")

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
                await self.take_screenshot("brightdata_login_no_submit.png")
                return False

            # Click submit with human-like behavior
            self.logger.info("Clicking submit button...")
            await submit_button.hover()
            await self.page.wait_for_timeout(random.uniform(200, 500))
            await submit_button.click()

            # Wait for navigation
            await self.page.wait_for_timeout(random.uniform(5000, 8000))

            # Take screenshot after submit
            await self.take_screenshot("brightdata_login_after_submit.png")

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
            await self.take_screenshot("brightdata_login_error.png")
            return False

    async def extract_balance(self) -> str:
        """Extract balance from DraftKings account."""
        try:
            if not self.page:
                self.logger.error("No page available for balance extraction")
                return None

            self.logger.info("Extracting balance from DraftKings...")

            # Wait for page to load
            await self.page.wait_for_timeout(random.uniform(3000, 5000))

            # Take screenshot before extraction
            await self.take_screenshot("brightdata_balance_before.png")

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
            await self.take_screenshot("brightdata_balance_after.png")

            if balance_text:
                self.logger.info(f"✅ Balance extracted: {balance_text}")
                return balance_text
            else:
                self.logger.warning("⚠️ No balance found")
                return None

        except Exception as e:
            self.logger.error(f"❌ Balance extraction failed: {e}")
            await self.take_screenshot("brightdata_balance_error.png")
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

    async def monitor_balance(self) -> str:
        """Main balance monitoring function with Bright Data residential proxy."""
        try:
            self.logger.info("🌐 DraftKings Balance Monitor - Bright Data Residential Proxy")
            self.logger.info("=" * 60)
            self.logger.info(f"🔐 Monitoring account: {self.username}")
            self.logger.info("🌐 Using Bright Data residential proxy")
            if UNDETECTED_AVAILABLE:
                self.logger.info("🛡️ Using undetected-playwright")
            self.logger.info("=" * 60)

            # Navigate to DraftKings
            if not await self.navigate_to_draftkings_brightdata():
                self.logger.error("❌ Failed to navigate to DraftKings")
                return None

            # Take screenshot before login
            await self.take_screenshot("brightdata_before_login.png")

            # Login to DraftKings
            if not await self.login_to_draftkings_brightdata():
                self.logger.error("❌ Failed to login to DraftKings")
                await self.take_screenshot("brightdata_login_failed.png")
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
    print("🌐 DraftKings Balance Monitor - Bright Data Residential Proxy")
    print("=" * 60)

    # Check if Bright Data credentials are configured
    if os.getenv('BRIGHTDATA_USERNAME') == 'brd-customer-hl_1234567890-zone-residential':
        print("⚠️  Bright Data credentials not configured!")
        print("💡 Set environment variables:")
        print("   export BRIGHTDATA_USERNAME='your_brightdata_username'")
        print("   export BRIGHTDATA_PASSWORD='your_brightdata_password'")
        print("   export BRIGHTDATA_HOST='brd.superproxy.io'")
        print("   export BRIGHTDATA_PORT='22225'")
        print("   export USE_BRIGHTDATA='true'")
        print("\n🔗 Get Bright Data credentials at: https://brightdata.com")
        print("📧 Or use alternative providers: Oxylabs, SmartProxy")
        return

    # DraftKings credentials
    username = "paxtonedgar3@gmail.com"
    password = "Empireozarks@2013"

    # Create monitor
    monitor = DraftKingsBrightDataMonitor(username, password)

    # Run balance monitoring
    balance = asyncio.run(monitor.monitor_balance())

    print("\n" + "=" * 60)
    print("BRIGHT DATA RESIDENTIAL PROXY RESULTS")
    print("=" * 60)

    if balance:
        print(f"✅ SUCCESS: Balance extracted: {balance}")
        print("🌐 Residential proxy bypassed IP protection!")
    else:
        print("❌ FAILED: Could not extract balance")
        print("💡 Check screenshots for debugging")
        print("🔧 Consider implementing alternative approaches")


if __name__ == "__main__":
    main()
