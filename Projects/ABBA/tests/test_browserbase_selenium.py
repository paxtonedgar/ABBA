"""
BrowserBase Selenium Test
Test Selenium WebDriver connection for browser control
"""

import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def test_browserbase_selenium():
    """Test BrowserBase Selenium WebDriver connection."""

    print("🚗 Testing BrowserBase Selenium WebDriver")
    print("=" * 50)

    # Get current session
    import asyncio

    import httpx

    async def get_session():
        api_key = os.getenv("BROWSERBASE_API_KEY")

        async with httpx.AsyncClient() as client:
            # Get sessions
            response = await client.get(
                "https://api.browserbase.com/v1/sessions",
                headers={"X-BB-API-Key": api_key},
            )

            if response.status_code != 200:
                print(f"❌ Failed to get sessions: {response.status_code}")
                return None

            sessions = response.json()
            if not sessions:
                print("❌ No sessions found")
                return None

            # Get the first running session
            for session in sessions:
                if session.get("status") == "RUNNING":
                    # Get full session details
                    session_response = await client.get(
                        f"https://api.browserbase.com/v1/sessions/{session['id']}",
                        headers={"X-BB-API-Key": api_key},
                    )

                    if session_response.status_code == 200:
                        return session_response.json()

            print("❌ No running session found")
            return None

    # Get session details
    session = asyncio.run(get_session())
    if not session:
        return False

    print(f"✅ Found running session: {session['id']}")
    selenium_url = session.get("seleniumRemoteUrl")
    signing_key = session.get("signingKey")

    if not selenium_url:
        print("❌ No Selenium URL found in session")
        return False

    print(f"🔗 Selenium URL: {selenium_url}")
    print(f"🔑 Signing Key: {signing_key[:20]}..." if signing_key else "No signing key")

    # Configure Chrome options
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-plugins")
    chrome_options.add_argument("--disable-images")
    chrome_options.add_argument("--disable-javascript")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--allow-running-insecure-content")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    # Add authentication using signing key
    if signing_key:
        chrome_options.add_argument(f"--header=Authorization: Bearer {signing_key}")

    try:
        print("\n🌐 Connecting to Selenium WebDriver...")

        # Try different authentication methods
        # Method 1: Add auth to URL
        auth_url = f"{selenium_url}?auth={signing_key}" if signing_key else selenium_url

        print(f"🔗 Trying URL: {auth_url}")

        # Connect to the remote WebDriver with authentication
        driver = webdriver.Remote(command_executor=auth_url, options=chrome_options)

        print("✅ Selenium WebDriver connected!")

        # Test navigation
        print("\n🧭 Testing navigation to Google...")
        driver.get("https://www.google.com")

        # Wait for page to load
        wait = WebDriverWait(driver, 10)
        title_element = wait.until(
            EC.presence_of_element_located((By.TAG_NAME, "title"))
        )

        page_title = driver.title
        current_url = driver.current_url

        print("✅ Navigation successful!")
        print(f"   Title: {page_title}")
        print(f"   URL: {current_url}")

        # Test DraftKings navigation
        print("\n🎯 Testing navigation to DraftKings...")
        driver.get("https://www.draftkings.com")

        # Wait for page to load
        wait = WebDriverWait(driver, 10)
        title_element = wait.until(
            EC.presence_of_element_located((By.TAG_NAME, "title"))
        )

        page_title = driver.title
        current_url = driver.current_url

        print("✅ DraftKings navigation successful!")
        print(f"   Title: {page_title}")
        print(f"   URL: {current_url}")

        # Close the browser
        driver.quit()
        print("\n🔒 Browser closed successfully!")

        return True

    except Exception as e:
        print(f"❌ Selenium WebDriver failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 BrowserBase Selenium Test")
    print("=" * 50)

    success = test_browserbase_selenium()

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)

    if success:
        print("✅ Selenium WebDriver successful!")
        print("🎉 BrowserBase is ready for live DraftKings testing!")
        print("\n🚀 Next steps:")
        print("1. Run: python test_live_balance_simple.py")
        print("2. The system will log into your DraftKings account")
        print("3. Check your balance and test fund management")
    else:
        print("❌ Selenium WebDriver failed!")
        print("\n💡 This suggests BrowserBase may require:")
        print("   - Different WebDriver configuration")
        print("   - Authentication in WebDriver connection")
        print("   - Different capabilities")


if __name__ == "__main__":
    main()
