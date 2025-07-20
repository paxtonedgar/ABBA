"""
BrowserBase Executor - Official SDK Version
Updated to use the official BrowserBase Python SDK with proper authentication
"""

import asyncio
import os

import structlog
from browserbase import Browserbase
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logger = structlog.get_logger(__name__)


class BrowserBaseExecutor:
    """BrowserBase executor using official Python SDK."""

    def __init__(self):
        """Initialize BrowserBase executor."""
        self.api_key = os.getenv("BROWSERBASE_API_KEY")
        self.project_id = os.getenv("BROWSERBASE_PROJECT_ID")

        if not self.api_key or not self.project_id:
            raise ValueError("Missing BROWSERBASE_API_KEY or BROWSERBASE_PROJECT_ID environment variables")

        # Initialize BrowserBase client
        self.bb = Browserbase(api_key=self.api_key)
        self.session = None
        self.driver = None

        logger.info("BrowserBase executor initialized",
                   api_key=self.api_key[:10] + "...",
                   project_id=self.project_id)

    async def create_session(self, keep_alive: bool = False) -> bool:
        """Create a new browser session."""
        try:
            logger.info("Creating BrowserBase session", keep_alive=keep_alive)

            # Create session with optional keep_alive
            session_params = {"project_id": self.project_id}
            if keep_alive:
                session_params["keep_alive"] = True

            self.session = self.bb.sessions.create(**session_params)

            logger.info("Session created successfully",
                       session_id=self.session.id,
                       selenium_url=self.session.selenium_remote_url)

            return True

        except Exception as e:
            logger.error("Failed to create session", error=str(e))
            return False

    async def create_driver(self) -> bool:
        """Create WebDriver instance."""
        try:
            if not self.session:
                logger.error("No session available")
                return False

            logger.info("Creating WebDriver")

            # Use the SDK's create_driver helper
            self.driver = self.session.create_driver()

            logger.info("WebDriver created successfully")
            return True

        except Exception as e:
            logger.error("Failed to create WebDriver", error=str(e))
            return False

    async def navigate(self, url: str) -> bool:
        """Navigate to a URL."""
        try:
            if not self.driver:
                logger.error("No WebDriver available")
                return False

            logger.info("Navigating to URL", url=url)

            self.driver.get(url)

            # Wait for page to load
            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "title")))

            page_title = self.driver.title
            current_url = self.driver.current_url

            logger.info("Navigation successful",
                       title=page_title,
                       url=current_url)

            return True

        except Exception as e:
            logger.error("Navigation failed", url=url, error=str(e))
            return False

    async def find_element(self, by: By, value: str, timeout: int = 10):
        """Find an element on the page."""
        try:
            if not self.driver:
                logger.error("No WebDriver available")
                return None

            logger.info("Finding element", by=by, value=value)

            wait = WebDriverWait(self.driver, timeout)
            element = wait.until(EC.presence_of_element_located((by, value)))

            logger.info("Element found", by=by, value=value)
            return element

        except Exception as e:
            logger.error("Element not found", by=by, value=value, error=str(e))
            return None

    async def click_element(self, by: By, value: str) -> bool:
        """Click an element on the page."""
        try:
            element = await self.find_element(by, value)
            if not element:
                return False

            logger.info("Clicking element", by=by, value=value)
            element.click()

            logger.info("Element clicked successfully")
            return True

        except Exception as e:
            logger.error("Failed to click element", by=by, value=value, error=str(e))
            return False

    async def send_keys(self, by: By, value: str, keys: str) -> bool:
        """Send keys to an element."""
        try:
            element = await self.find_element(by, value)
            if not element:
                return False

            logger.info("Sending keys to element", by=by, value=value, keys=keys[:10] + "...")
            element.send_keys(keys)

            logger.info("Keys sent successfully")
            return True

        except Exception as e:
            logger.error("Failed to send keys", by=by, value=value, error=str(e))
            return False

    async def get_page_source(self) -> str | None:
        """Get the page source."""
        try:
            if not self.driver:
                logger.error("No WebDriver available")
                return None

            page_source = self.driver.page_source
            logger.info("Page source retrieved", length=len(page_source))
            return page_source

        except Exception as e:
            logger.error("Failed to get page source", error=str(e))
            return None

    async def get_element_text(self, by: By, value: str) -> str | None:
        """Get text from an element."""
        try:
            element = await self.find_element(by, value)
            if not element:
                return None

            text = element.text
            logger.info("Element text retrieved", by=by, value=value, text=text[:50] + "...")
            return text

        except Exception as e:
            logger.error("Failed to get element text", by=by, value=value, error=str(e))
            return None

    async def close_driver(self):
        """Close the WebDriver."""
        try:
            if self.driver:
                logger.info("Closing WebDriver")
                self.driver.quit()
                self.driver = None
                logger.info("WebDriver closed")
        except Exception as e:
            logger.error("Error closing WebDriver", error=str(e))

    async def close_session(self):
        """Close the session."""
        try:
            await self.close_driver()

            if self.session:
                logger.info("Closing session", session_id=self.session.id)
                # Sessions are automatically closed when they expire
                self.session = None
                logger.info("Session closed")

        except Exception as e:
            logger.error("Error closing session", error=str(e))

    async def __aenter__(self):
        """Async context manager entry."""
        await self.create_session()
        await self.create_driver()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_session()


# Example usage
async def test_browserbase_executor():
    """Test the BrowserBase executor."""
    logger.info("Testing BrowserBase executor")

    async with BrowserBaseExecutor() as executor:
        # Navigate to Google
        success = await executor.navigate("https://www.google.com")
        if not success:
            logger.error("Failed to navigate to Google")
            return False

        # Find and interact with search box
        success = await executor.send_keys(By.NAME, "q", "BrowserBase test")
        if not success:
            logger.error("Failed to send keys to search box")
            return False

        # Get page title
        title_element = await executor.find_element(By.TAG_NAME, "title")
        if title_element:
            title = title_element.get_attribute("textContent")
            logger.info("Page title", title=title)

        logger.info("BrowserBase executor test completed successfully")
        return True


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
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Run test
    asyncio.run(test_browserbase_executor())
