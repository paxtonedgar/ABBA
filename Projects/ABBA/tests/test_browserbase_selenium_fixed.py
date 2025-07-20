"""
BrowserBase Selenium Test - Fixed Authentication
Test Selenium WebDriver connection with proper signing key authentication
"""

import asyncio
import http.client
import os

import httpx
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def test_browserbase_selenium_fixed():
    """Test BrowserBase Selenium WebDriver with proper authentication."""

    print("🚗 Testing BrowserBase Selenium WebDriver - Fixed Auth")
    print("=" * 60)

    # Get current session
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

    if not signing_key:
        print("❌ No signing key found in session")
        return False

    print(f"🔗 Selenium URL: {selenium_url}")
    print(f"🔑 Signing Key: {signing_key[:20]}...")

    try:
        print("\n🌐 Setting up authenticated HTTP connection...")

        # Parse the selenium URL to get hostname
        from urllib.parse import urlparse

        parsed_url = urlparse(selenium_url)
        hostname = parsed_url.hostname
        port = parsed_url.port or 80

        print(f"🔗 Hostname: {hostname}:{port}")

        # Create HTTP connection with signing key authentication
        agent = http.client.HTTPConnection(hostname, port)

        # Monkey-patch the putrequest method to add signing key header
        original_putrequest = agent.putrequest

        def add_signing_key(method, url, *args, **kwargs):
            result = original_putrequest(method, url, *args, **kwargs)
            agent.putheader("x-bb-signing-key", signing_key)
            return result

        agent.putrequest = add_signing_key

        print("✅ HTTP connection configured with signing key")

        # Connect to the remote WebDriver
        print("\n🚗 Connecting to Selenium WebDriver...")

        # Use modern Selenium API
        from selenium.webdriver.chrome.options import Options

        chrome_options = Options()
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

        driver = webdriver.Remote(
            command_executor=selenium_url,
            options=chrome_options,
            keep_alive=True,
            http_client=agent,
        )

        print("✅ Selenium WebDriver connected successfully!")

        # Test navigation to Google
        print("\n🧭 Testing navigation to Google...")
        driver.get("https://www.google.com")

        # Wait for page to load
        wait = WebDriverWait(driver, 10)
        title_element = wait.until(
            EC.presence_of_element_located((By.TAG_NAME, "title"))
        )

        page_title = driver.title
        current_url = driver.current_url

        print("✅ Google navigation successful!")
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

        # Test element interaction (find search box on Google)
        print("\n🔍 Testing element interaction...")
        driver.get("https://www.google.com")

        # Find and interact with search box
        search_box = wait.until(EC.presence_of_element_located((By.NAME, "q")))
        search_box.send_keys("BrowserBase test")

        print("✅ Element interaction successful!")
        print("   Found search box and entered text")

        # Close the browser
        driver.quit()
        print("\n🔒 Browser closed successfully!")

        return True

    except Exception as e:
        print(f"❌ Selenium WebDriver failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧪 BrowserBase Selenium Test - Fixed Authentication")
    print("=" * 60)

    success = test_browserbase_selenium_fixed()

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    if success:
        print("✅ Selenium WebDriver authentication successful!")
        print("🎉 BrowserBase is ready for live DraftKings testing!")
        print("\n🚀 Next steps:")
        print("1. Run: python test_live_balance_simple.py")
        print("2. The system will log into your DraftKings account")
        print("3. Check your balance and test fund management")
        print("\n💡 Browser control is now fully functional!")
    else:
        print("❌ Selenium WebDriver authentication failed!")
        print("\n💡 Troubleshooting:")
        print("   - Check if session is still running")
        print("   - Verify signing key is present")
        print("   - Check BrowserBase service status")


if __name__ == "__main__":
    main()
