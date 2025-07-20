"""
Execution module for ABMBA system.
Handles browser automation and bet placement with 2FA integration.
Enhanced with 2025 cutting-edge stealth capabilities.
"""

import asyncio
import os
import random
import re
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Any

import structlog
from playwright.async_api import async_playwright

# Advanced 2025 stealth imports
try:
    from playwright_stealth import stealth_async
    PLAYWRIGHT_STEALTH_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_STEALTH_AVAILABLE = False
    print("Warning: playwright-stealth not available. Enhanced stealth disabled.")

try:
    from browserbase import BrowserBase
    BROWSERBASE_AVAILABLE = True
except ImportError:
    BROWSERBASE_AVAILABLE = False
    print("Warning: browserbase not available. Advanced anti-detection disabled.")

try:
    import undetected_chromedriver as uc
    UNDETECTED_CHROME_AVAILABLE = True
except ImportError:
    UNDETECTED_CHROME_AVAILABLE = False
    print("Warning: undetected-chromedriver not available. Chrome stealth disabled.")

from models import Bet

logger = structlog.get_logger()


class AdvancedStealthManager:
    """Cutting-edge stealth manager with 2025 anti-detection capabilities."""

    def __init__(self, config: dict):
        self.config = config
        self.user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        self.viewports = [
            {'width': 1920, 'height': 1080},
            {'width': 1366, 'height': 768},
            {'width': 1440, 'height': 900},
            {'width': 1536, 'height': 864}
        ]
        self.session_rotation_interval = config.get('security', {}).get('session_rotation_frequency', 3600)
        self.last_rotation = datetime.utcnow()

        # BrowserBase integration for advanced anti-detection
        if BROWSERBASE_AVAILABLE:
            self.browserbase = BrowserBase(
                api_key=config.get('browserbase', {}).get('api_key'),
                proxy=config.get('browserbase', {}).get('proxy')
            )
        else:
            self.browserbase = None

    def get_random_user_agent(self) -> str:
        """Get random user agent for fingerprint randomization."""
        return random.choice(self.user_agents)

    def get_random_viewport(self) -> dict[str, int]:
        """Get random viewport for fingerprint randomization."""
        return random.choice(self.viewports)

    def get_random_delay(self) -> float:
        """Get random delay for human-like behavior."""
        return random.uniform(1.0, 3.0)

    def rotate_session(self):
        """Rotate session to avoid detection patterns."""
        self.last_rotation = datetime.utcnow()
        logger.info("Session rotated for anti-detection")

    def should_rotate_session(self) -> bool:
        """Check if session should be rotated."""
        return (datetime.utcnow() - self.last_rotation).total_seconds() > self.session_rotation_interval

    def randomize_behavior(self):
        """Randomize behavior patterns to mimic human users."""
        # Random mouse movements, scrolls, etc.
        pass

    async def create_stealth_context(self, playwright) -> Any:
        """Create stealth browser context with advanced anti-detection."""
        if BROWSERBASE_AVAILABLE and self.browserbase:
            # Use BrowserBase for maximum stealth
            return await self.browserbase.create_context()

        # Fallback to enhanced Playwright stealth
        browser_args = [
            '--disable-blink-features=AutomationControlled',
            '--disable-dev-shm-usage',
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-web-security',
            '--disable-features=VizDisplayCompositor',
            '--disable-background-timer-throttling',
            '--disable-backgrounding-occluded-windows',
            '--disable-renderer-backgrounding',
            '--disable-field-trial-config',
            '--disable-ipc-flooding-protection',
            '--no-first-run',
            '--no-default-browser-check',
            '--disable-default-apps',
            '--disable-extensions',
            '--disable-plugins',
            '--disable-images',
            '--disable-javascript',
            '--disable-background-networking',
            '--disable-sync',
            '--disable-translate',
            '--hide-scrollbars',
            '--mute-audio',
            '--no-zygote',
            '--disable-gpu',
            '--disable-software-rasterizer',
            '--disable-background-timer-throttling',
            '--disable-backgrounding-occluded-windows',
            '--disable-renderer-backgrounding',
            '--disable-features=TranslateUI',
            '--disable-ipc-flooding-protection'
        ]

        browser = await playwright.chromium.launch(
            headless=True,
            args=browser_args
        )

        context = await browser.new_context(
            viewport=self.get_random_viewport(),
            user_agent=self.get_random_user_agent(),
            extra_http_headers={
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
        )

        if PLAYWRIGHT_STEALTH_AVAILABLE:
            await stealth_async(context)

        return context


class WebhookManager:
    """Real-time webhook manager for odds updates."""

    def __init__(self, config: dict):
        self.config = config
        self.webhook_urls = config.get('webhooks', {}).get('odds_api', [])
        self.callback_handlers = {}

    async def register_webhook(self, event_type: str, callback_func):
        """Register webhook callback for real-time updates."""
        self.callback_handlers[event_type] = callback_func
        logger.info(f"Registered webhook for {event_type}")

    async def handle_odds_update(self, odds_data: dict):
        """Handle real-time odds updates from webhooks."""
        try:
            # Process odds update
            event_id = odds_data.get('event_id')
            new_odds = odds_data.get('odds')

            if event_id and new_odds:
                logger.info(f"Received odds update for event {event_id}: {new_odds}")

                # Trigger arbitrage detection
                if 'arbitrage_detection' in self.callback_handlers:
                    await self.callback_handlers['arbitrage_detection'](odds_data)

                # Trigger value bet detection
                if 'value_detection' in self.callback_handlers:
                    await self.callback_handlers['value_detection'](odds_data)

        except Exception as e:
            logger.error(f"Error handling odds update: {e}")

    async def start_webhook_server(self, port: int = 8080):
        """Start webhook server for receiving real-time updates."""
        try:
            from fastapi import FastAPI, Request
            from uvicorn import run

            app = FastAPI()

            @app.post("/webhook/odds")
            async def receive_odds_webhook(request: Request):
                odds_data = await request.json()
                await self.handle_odds_update(odds_data)
                return {"status": "received"}

            logger.info(f"Starting webhook server on port {port}")
            run(app, host="0.0.0.0", port=port)

        except ImportError:
            logger.warning("FastAPI not available, webhook server disabled")
        except Exception as e:
            logger.error(f"Error starting webhook server: {e}")


class BettingExecutor:
    """Enhanced executor for placing bets via advanced stealth automation."""

    def __init__(self, config: dict):
        self.config = config
        self.browser = None
        self.pages = {}
        self.session_data = {}
        self.stealth_manager = AdvancedStealthManager(config)
        self.webhook_manager = WebhookManager(config)
        self.stealth_mode = config['agents']['execution']['stealth_mode']
        self.random_delays = config['agents']['execution']['random_delays']
        self.min_delay = config['agents']['execution']['min_delay']
        self.max_delay = config['agents']['execution']['max_delay']

    async def initialize(self):
        """Initialize browser and sessions with enhanced stealth."""
        try:
            self.playwright = await async_playwright().start()

            # Initialize pages for each platform with stealth
            for platform in ['fanduel', 'draftkings']:
                if self.config['platforms'][platform]['enabled']:
                    context = await self.stealth_manager.create_stealth_context(self.playwright)
                    page = await context.new_page()
                    self.pages[platform] = page

            # Start webhook server for real-time odds
            await self.webhook_manager.start_webhook_server()

            logger.info("Enhanced betting executor initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing enhanced betting executor: {e}")
            raise

    async def login_platform(self, platform: str) -> bool:
        """
        Login to a betting platform with enhanced stealth.
        
        Args:
            platform: Platform name (fanduel, draftkings)
        
        Returns:
            True if login successful, False otherwise
        """
        try:
            if platform not in self.pages:
                logger.error(f"No page available for platform: {platform}")
                return False

            page = self.pages[platform]
            platform_config = self.config['platforms'][platform]

            # Rotate session if needed
            if self.stealth_manager.should_rotate_session():
                self.stealth_manager.rotate_session()
                # Recreate context for rotation
                context = await self.stealth_manager.create_stealth_context(self.playwright)
                page = await context.new_page()
                self.pages[platform] = page

            # Navigate to login page with stealth
            if platform == 'fanduel':
                await page.goto('https://sportsbook.fanduel.com/login', timeout=60000)
            elif platform == 'draftkings':
                await page.goto('https://sportsbook.draftkings.com/login', timeout=60000)

            await self._random_delay()

            # Fill login form with human-like behavior
            username = platform_config['username']
            password = platform_config['password']

            # Find and fill username field with typing simulation
            await page.fill('input[name="username"], input[name="email"], input[type="email"]', username)
            await self._random_delay()

            # Find and fill password field with typing simulation
            await page.fill('input[name="password"], input[type="password"]', password)
            await self._random_delay()

            # Click login button with human-like delay
            await page.click('button[type="submit"], input[type="submit"]')

            # Wait for login to complete
            await page.wait_for_load_state('networkidle')

            # Check if 2FA is required
            if await self._check_2fa_required(page, platform):
                success = await self._handle_2fa(page, platform)
                if not success:
                    logger.error(f"2FA failed for {platform}")
                    return False

            # Verify login success
            if await self._verify_login(page, platform):
                logger.info(f"Successfully logged into {platform}")
                self.session_data[platform] = {
                    'logged_in': True,
                    'login_time': datetime.utcnow()
                }
                return True
            else:
                logger.error(f"Login verification failed for {platform}")
                return False

        except Exception as e:
            logger.error(f"Error logging into {platform}: {e}")
            return False

    async def _check_2fa_required(self, page, platform: str) -> bool:
        """Check if 2FA is required."""
        try:
            # Look for 2FA indicators
            selectors = [
                'input[name="code"]',
                'input[name="verification"]',
                '.two-factor',
                '[data-testid="2fa"]'
            ]

            for selector in selectors:
                if await page.locator(selector).count() > 0:
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking 2FA requirement: {e}")
            return False

    async def _handle_2fa(self, page, platform: str) -> bool:
        """Handle two-factor authentication."""
        try:
            # Get 2FA code from iMessage
            code = await self._get_2fa_code_from_imessage(platform)

            if not code:
                logger.error("No 2FA code found")
                return False

            # Find and fill 2FA input field
            await page.fill('input[name="code"], input[name="verification"]', code)
            await self._random_delay()

            # Submit 2FA code
            await page.click('button[type="submit"]')
            await page.wait_for_load_state('networkidle')

            return True

        except Exception as e:
            logger.error(f"Error handling 2FA: {e}")
            return False

    async def _get_2fa_code_from_imessage(self, platform: str) -> str | None:
        """Extract 2FA code from iMessage database."""
        try:
            # Path to iMessage database on Mac
            imessage_db = os.path.expanduser('~/Library/Messages/chat.db')

            if not os.path.exists(imessage_db):
                logger.error("iMessage database not found")
                return None

            # Connect to iMessage database
            conn = sqlite3.connect(imessage_db)
            cursor = conn.cursor()

            # Look for recent messages from betting platforms
            platform_keywords = {
                'fanduel': ['FanDuel', 'fanduel'],
                'draftkings': ['DraftKings', 'draftkings']
            }

            keywords = platform_keywords.get(platform, [])

            # Query recent messages
            query = """
                SELECT text, date FROM message 
                WHERE text LIKE '%verification%' OR text LIKE '%code%'
                ORDER BY date DESC LIMIT 10
            """

            cursor.execute(query)
            messages = cursor.fetchall()

            # Look for 2FA codes in messages
            for message_text, message_date in messages:
                # Check if message is recent (within last 5 minutes)
                message_time = datetime.fromtimestamp(message_date / 1000000000)
                if datetime.utcnow() - message_time > timedelta(minutes=5):
                    continue

                # Extract code using regex
                code_match = re.search(r'\b\d{6}\b', message_text)
                if code_match:
                    conn.close()
                    return code_match.group()

            conn.close()
            return None

        except Exception as e:
            logger.error(f"Error getting 2FA code from iMessage: {e}")
            return None

    async def _verify_login(self, page, platform: str) -> bool:
        """Verify that login was successful."""
        try:
            # Check for login indicators
            if platform == 'fanduel':
                # Look for account menu or balance indicator
                selectors = ['.account-menu', '.balance', '[data-testid="account"]']
            elif platform == 'draftkings':
                selectors = ['.account-menu', '.balance', '[data-testid="account"]']

            for selector in selectors:
                if await page.locator(selector).count() > 0:
                    return True

            return False

        except Exception as e:
            logger.error(f"Error verifying login: {e}")
            return False

    async def place_bet(self, bet: Bet) -> dict[str, Any]:
        """
        Place a bet on the specified platform.
        
        Args:
            bet: Bet object with all necessary information
        
        Returns:
            Dictionary with execution results
        """
        try:
            platform = bet.platform.value
            start_time = time.time()

            # Ensure we're logged in
            if platform not in self.session_data or not self.session_data[platform]['logged_in']:
                login_success = await self.login_platform(platform)
                if not login_success:
                    return {
                        'success': False,
                        'error': f'Failed to login to {platform}'
                    }

            page = self.pages[platform]

            # Navigate to the specific event
            event_url = await self._get_event_url(platform, bet.event_id)
            if not event_url:
                return {
                    'success': False,
                    'error': f'Could not find event URL for {bet.event_id}'
                }

            await page.goto(event_url)
            await self._random_delay()

            # Find and click on the specific market
            market_selector = await self._get_market_selector(platform, bet.market_type.value, bet.selection)
            if not market_selector:
                return {
                    'success': False,
                    'error': f'Could not find market selector for {bet.market_type.value} {bet.selection}'
                }

            await page.click(market_selector)
            await self._random_delay()

            # Enter stake amount
            stake_input = await self._get_stake_input_selector(platform)
            if stake_input:
                await page.fill(stake_input, str(bet.stake))
                await self._random_delay()

            # Click place bet button
            bet_button = await self._get_place_bet_button_selector(platform)
            if bet_button:
                await page.click(bet_button)
                await self._random_delay()

                # Wait for confirmation
                await page.wait_for_load_state('networkidle')

                # Check for success/error messages
                success = await self._check_bet_success(page, platform)

                execution_time = time.time() - start_time

                if success:
                    return {
                        'success': True,
                        'bet_id': bet.id,
                        'platform': platform,
                        'execution_time': execution_time
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Bet placement failed or was rejected',
                        'execution_time': execution_time
                    }
            else:
                return {
                    'success': False,
                    'error': 'Could not find place bet button'
                }

        except Exception as e:
            logger.error(f"Error placing bet: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _get_event_url(self, platform: str, event_id: str) -> str | None:
        """Get the URL for a specific event."""
        try:
            # This would implement platform-specific URL generation
            # For now, return mock URLs
            if platform == 'fanduel':
                return f'https://sportsbook.fanduel.com/basketball/nba/{event_id}'
            elif platform == 'draftkings':
                return f'https://sportsbook.draftkings.com/basketball/nba/{event_id}'

            return None

        except Exception as e:
            logger.error(f"Error getting event URL: {e}")
            return None

    async def _get_market_selector(self, platform: str, market_type: str, selection: str) -> str | None:
        """Get CSS selector for a specific market."""
        try:
            # Platform-specific selectors
            selectors = {
                'fanduel': {
                    'moneyline': {
                        'home': '[data-testid="moneyline-home"]',
                        'away': '[data-testid="moneyline-away"]'
                    },
                    'spread': {
                        'home': '[data-testid="spread-home"]',
                        'away': '[data-testid="spread-away"]'
                    },
                    'totals': {
                        'over': '[data-testid="total-over"]',
                        'under': '[data-testid="total-under"]'
                    }
                },
                'draftkings': {
                    'moneyline': {
                        'home': '.moneyline-home',
                        'away': '.moneyline-away'
                    },
                    'spread': {
                        'home': '.spread-home',
                        'away': '.spread-away'
                    },
                    'totals': {
                        'over': '.total-over',
                        'under': '.total-under'
                    }
                }
            }

            return selectors.get(platform, {}).get(market_type, {}).get(selection)

        except Exception as e:
            logger.error(f"Error getting market selector: {e}")
            return None

    async def _get_stake_input_selector(self, platform: str) -> str | None:
        """Get CSS selector for stake input field."""
        try:
            selectors = {
                'fanduel': 'input[name="stake"], input[data-testid="stake-input"]',
                'draftkings': 'input[name="stake"], .stake-input'
            }

            return selectors.get(platform)

        except Exception as e:
            logger.error(f"Error getting stake input selector: {e}")
            return None

    async def _get_place_bet_button_selector(self, platform: str) -> str | None:
        """Get CSS selector for place bet button."""
        try:
            selectors = {
                'fanduel': 'button[data-testid="place-bet"], .place-bet-btn',
                'draftkings': 'button.place-bet, .place-bet-button'
            }

            return selectors.get(platform)

        except Exception as e:
            logger.error(f"Error getting place bet button selector: {e}")
            return None

    async def _check_bet_success(self, page, platform: str) -> bool:
        """Check if bet was placed successfully."""
        try:
            # Look for success indicators
            success_selectors = [
                '.bet-confirmation',
                '.success-message',
                '[data-testid="bet-success"]',
                '.confirmation'
            ]

            for selector in success_selectors:
                if await page.locator(selector).count() > 0:
                    return True

            # Look for error indicators
            error_selectors = [
                '.error-message',
                '.bet-rejected',
                '[data-testid="bet-error"]'
            ]

            for selector in error_selectors:
                if await page.locator(selector).count() > 0:
                    return False

            # Default to success if no clear indicators
            return True

        except Exception as e:
            logger.error(f"Error checking bet success: {e}")
            return False

    async def _random_delay(self):
        """Add random delay to avoid detection."""
        if self.random_delays:
            delay = random.uniform(self.min_delay, self.max_delay)
            await asyncio.sleep(delay)

    async def check_bet_status(self, bet_id: str, platform: str) -> dict[str, Any]:
        """Check the status of a placed bet."""
        try:
            # This would implement checking bet status on the platform
            # For now, return mock status

            return {
                'bet_id': bet_id,
                'platform': platform,
                'status': 'placed',
                'checked_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error checking bet status: {e}")
            return {
                'bet_id': bet_id,
                'platform': platform,
                'status': 'unknown',
                'error': str(e)
            }

    async def close(self):
        """Close browser and clean up resources."""
        try:
            if self.browser:
                await self.browser.close()

            if hasattr(self, 'playwright'):
                await self.playwright.stop()

            logger.info("Betting executor closed")

        except Exception as e:
            logger.error(f"Error closing betting executor: {e}")


class TwoFactorHandler:
    """Handler for two-factor authentication."""

    def __init__(self, config: dict):
        self.config = config
        self.imessage_db_path = os.path.expanduser('~/Library/Messages/chat.db')

    async def get_2fa_code(self, platform: str, timeout: int = 60) -> str | None:
        """
        Get 2FA code from iMessage with timeout.
        
        Args:
            platform: Platform name
            timeout: Timeout in seconds
        
        Returns:
            2FA code if found, None otherwise
        """
        try:
            start_time = time.time()

            while time.time() - start_time < timeout:
                code = await self._extract_code_from_imessage(platform)
                if code:
                    return code

                await asyncio.sleep(2)  # Check every 2 seconds

            return None

        except Exception as e:
            logger.error(f"Error getting 2FA code: {e}")
            return None

    async def _extract_code_from_imessage(self, platform: str) -> str | None:
        """Extract 2FA code from iMessage database."""
        try:
            if not os.path.exists(self.imessage_db_path):
                return None

            conn = sqlite3.connect(self.imessage_db_path)
            cursor = conn.cursor()

            # Look for recent messages with verification codes
            query = """
                SELECT text, date FROM message 
                WHERE text LIKE '%verification%' OR text LIKE '%code%' OR text LIKE '%%d{6}%'
                ORDER BY date DESC LIMIT 5
            """

            cursor.execute(query)
            messages = cursor.fetchall()

            for message_text, message_date in messages:
                # Check if message is recent (within last 2 minutes)
                message_time = datetime.fromtimestamp(message_date / 1000000000)
                if datetime.utcnow() - message_time > timedelta(minutes=2):
                    continue

                # Extract 6-digit code
                code_match = re.search(r'\b\d{6}\b', message_text)
                if code_match:
                    conn.close()
                    return code_match.group()

            conn.close()
            return None

        except Exception as e:
            logger.error(f"Error extracting code from iMessage: {e}")
            return None

    def send_2fa_notification(self, platform: str):
        """Send notification to request 2FA code."""
        try:
            # This could send a notification to the user's phone
            # For now, just log the request
            logger.info(f"2FA code requested for {platform}")

        except Exception as e:
            logger.error(f"Error sending 2FA notification: {e}")
