"""
BrowserBase Test - Using Existing Session
Test using an existing running session instead of creating a new one
"""

import asyncio
import os

import httpx
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


async def test_existing_session():
    """Test using an existing BrowserBase session."""

    print("🚀 Testing BrowserBase with Existing Session")
    print("=" * 50)

    api_key = os.getenv("BROWSERBASE_API_KEY")

    try:
        # Get existing sessions
        print("📊 Getting existing sessions...")
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.browserbase.com/v1/sessions",
                headers={"X-BB-API-Key": api_key},
            )

            if response.status_code != 200:
                print(f"❌ Failed to get sessions: {response.status_code}")
                return False

            sessions = response.json()
            if not sessions:
                print("❌ No sessions found")
                return False

            # Find the first running session
            running_session = None
            for session in sessions:
                if session.get("status") == "RUNNING":
                    # Get full session details
                    session_response = await client.get(
                        f"https://api.browserbase.com/v1/sessions/{session['id']}",
                        headers={"X-BB-API-Key": api_key},
                    )

                    if session_response.status_code == 200:
                        running_session = session_response.json()
                        break

            if not running_session:
                print("❌ No running session found")
                return False

            print(f"✅ Found running session: {running_session['id']}")
            print(f"🔗 Selenium URL: {running_session['seleniumRemoteUrl']}")
            print(f"🔑 Signing Key: {running_session['signingKey'][:20]}...")

            # Create WebDriver with custom HTTP client
            print("\n🚗 Creating WebDriver...")

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
            chrome_options.add_experimental_option(
                "excludeSwitches", ["enable-automation"]
            )
            chrome_options.add_experimental_option("useAutomationExtension", False)

            # Try adding the signing key as a capability
            chrome_options.set_capability(
                "browserbase:signing-key", running_session["signingKey"]
            )
            chrome_options.set_capability("browserbase:api-key", api_key)

            # Try using the selenium URL with signing key as query parameter
            selenium_url_with_auth = f"{running_session['seleniumRemoteUrl']}?signingKey={running_session['signingKey']}"

            print(f"🔗 Trying URL: {selenium_url_with_auth}")

            driver = webdriver.Remote(
                command_executor=selenium_url_with_auth,
                options=chrome_options,
                keep_alive=True,
            )

            print("✅ WebDriver created successfully!")

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
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧪 BrowserBase Existing Session Test")
    print("=" * 50)

    success = asyncio.run(test_existing_session())

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)

    if success:
        print("✅ BrowserBase existing session test successful!")
        print("🎉 BrowserBase is ready for live DraftKings testing!")
        print("\n🚀 Next steps:")
        print("1. Run: python test_live_balance_simple.py")
        print("2. The system will log into your DraftKings account")
        print("3. Check your balance and test fund management")
        print("\n💡 Browser control is now fully functional!")
    else:
        print("❌ BrowserBase existing session test failed!")
        print("\n💡 Troubleshooting:")
        print("   - Check if session is still running")
        print("   - Verify signing key is present")
        print("   - Check BrowserBase service status")


if __name__ == "__main__":
    main()
