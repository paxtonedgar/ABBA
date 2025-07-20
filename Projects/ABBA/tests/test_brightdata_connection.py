"""
Test Bright Data Proxy Connection
Simple test to verify Bright Data residential proxy is working
"""

import asyncio
import os

from playwright.async_api import async_playwright


async def test_brightdata_connection():
    """Test Bright Data proxy connection."""
    print("🌐 Testing Bright Data Proxy Connection")
    print("=" * 50)

    # Get Bright Data credentials
    username = os.getenv("BRIGHTDATA_USERNAME")
    password = os.getenv("BRIGHTDATA_PASSWORD")
    host = os.getenv("BRIGHTDATA_HOST", "brd.superproxy.io")
    port = os.getenv("BRIGHTDATA_PORT", "22225")

    print(f"Username: {username}")
    print(f"Password: {password[:10]}..." if password else "Password: Not set")
    print(f"Host: {host}")
    print(f"Port: {port}")

    if not username or not password:
        print("❌ Bright Data credentials not properly configured")
        return False

    try:
        # Start Playwright
        playwright = await async_playwright().start()

        # Configure proxy
        proxy_config = {
            "server": f"http://{username}:{password}@{host}:{port}",
            "username": username,
            "password": password,
        }

        print(f"🌐 Proxy server: {proxy_config['server']}")

        # Launch browser with proxy
        browser = await playwright.chromium.launch(headless=False, proxy=proxy_config)

        print("✅ Browser launched with Bright Data proxy")

        # Create context
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )

        print("✅ Context created")

        # Create page
        page = await context.new_page()
        print("✅ Page created")

        # Test IP address
        print("🌐 Testing IP address...")
        await page.goto("https://httpbin.org/ip", timeout=30000)

        # Get IP info
        ip_info = await page.text_content("body")
        print(f"🌐 IP Info: {ip_info}")

        # Test DraftKings navigation
        print("🎯 Testing DraftKings navigation...")
        await page.goto("https://www.draftkings.com/login", timeout=60000)

        title = await page.title()
        current_url = page.url

        print(f"✅ Page title: {title}")
        print(f"✅ Current URL: {current_url}")

        # Take screenshot
        await page.screenshot(path="brightdata_test_screenshot.png")
        print("✅ Screenshot saved: brightdata_test_screenshot.png")

        # Check if we're on login page
        if "login" in current_url.lower():
            print("🎯 Successfully navigated to DraftKings login page!")

            # Wait a bit and check for form elements
            await page.wait_for_timeout(5000)

            # Look for login form elements
            email_fields = await page.query_selector_all('input[type="email"]')
            password_fields = await page.query_selector_all('input[type="password"]')

            print(f"📧 Email fields found: {len(email_fields)}")
            print(f"🔒 Password fields found: {len(password_fields)}")

            if email_fields and password_fields:
                print("✅ Login form loaded successfully!")
                print("🌐 Bright Data residential proxy is working!")
                return True
            else:
                print("❌ Login form not loaded - still blocked by anti-bot")
                return False
        else:
            print("❌ Failed to navigate to login page")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        try:
            if "page" in locals():
                await page.close()
            if "context" in locals():
                await context.close()
            if "browser" in locals():
                await browser.close()
            if "playwright" in locals():
                await playwright.stop()
        except:
            pass


def main():
    """Main function."""
    print("🧪 Bright Data Connection Test")
    print("=" * 50)

    success = asyncio.run(test_brightdata_connection())

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)

    if success:
        print("✅ SUCCESS: Bright Data proxy is working!")
        print("🚀 Ready to test DraftKings login form loading")
    else:
        print("❌ FAILED: Bright Data proxy test failed")
        print("🔧 Check credentials and try again")


if __name__ == "__main__":
    main()
