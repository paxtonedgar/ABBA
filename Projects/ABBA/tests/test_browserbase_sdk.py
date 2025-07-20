"""
BrowserBase SDK Test
Test using the official BrowserBase Python SDK
"""

import os

from browserbase import Browserbase
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def test_browserbase_sdk():
    """Test BrowserBase using the official Python SDK."""

    print("🚀 Testing BrowserBase Python SDK")
    print("=" * 50)

    # Initialize BrowserBase client
    api_key = os.getenv("BROWSERBASE_API_KEY")
    project_id = os.getenv("BROWSERBASE_PROJECT_ID")

    if not api_key or not project_id:
        print("❌ Missing environment variables:")
        print(f"   BROWSERBASE_API_KEY: {'✅' if api_key else '❌'}")
        print(f"   BROWSERBASE_PROJECT_ID: {'✅' if project_id else '❌'}")
        return False

    print(f"🔑 API Key: {api_key[:10]}...")
    print(f"📋 Project ID: {project_id}")

    try:
        # Initialize BrowserBase client
        print("\n🌐 Initializing BrowserBase client...")
        bb = Browserbase(api_key=api_key)

        print("✅ BrowserBase client initialized!")

        # Create a session
        print("\n📊 Creating session...")
        session = bb.sessions.create(project_id=project_id)

        print(f"✅ Session created: {session.id}")
        print(f"🔗 Selenium URL: {session.selenium_remote_url}")
        print(f"🔑 Signing Key: {session.signing_key[:20]}...")

        # Create WebDriver using the SDK helper
        print("\n🚗 Creating WebDriver...")

        # The session object doesn't have create_driver() method
        # We need to create the WebDriver manually with signing key authentication
        import http.client
        from urllib.parse import urlparse

        # Parse the selenium URL to get hostname
        parsed_url = urlparse(session.selenium_remote_url)
        hostname = parsed_url.hostname
        port = parsed_url.port or 80

        print(f"🔗 Hostname: {hostname}:{port}")

        # Create HTTP connection with signing key authentication
        agent = http.client.HTTPConnection(hostname, port)

        # Monkey-patch the putrequest method to add signing key header
        original_putrequest = agent.putrequest

        def add_signing_key(method, url, *args, **kwargs):
            result = original_putrequest(method, url, *args, **kwargs)
            agent.putheader("x-bb-signing-key", session.signing_key)
            return result

        agent.putrequest = add_signing_key

        print("✅ HTTP connection configured with signing key")

        # Create WebDriver with custom HTTP client
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
            command_executor=session.selenium_remote_url,
            options=chrome_options,
            keep_alive=True,
            http_client=agent,
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
        search_box.send_keys("BrowserBase SDK test")

        print("✅ Element interaction successful!")
        print("   Found search box and entered text")

        # Close the browser
        driver.quit()
        print("\n🔒 Browser closed successfully!")

        return True

    except Exception as e:
        print(f"❌ BrowserBase SDK test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧪 BrowserBase Python SDK Test")
    print("=" * 50)

    success = test_browserbase_sdk()

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)

    if success:
        print("✅ BrowserBase SDK test successful!")
        print("🎉 BrowserBase is ready for live DraftKings testing!")
        print("\n🚀 Next steps:")
        print("1. Run: python test_live_balance_simple.py")
        print("2. The system will log into your DraftKings account")
        print("3. Check your balance and test fund management")
        print("\n💡 Browser control is now fully functional with official SDK!")
    else:
        print("❌ BrowserBase SDK test failed!")
        print("\n💡 Troubleshooting:")
        print("   - Check environment variables")
        print("   - Verify API key and project ID")
        print("   - Check BrowserBase service status")


if __name__ == "__main__":
    main()
