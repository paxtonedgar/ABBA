"""
Simple DraftKings Test
Test basic navigation without proxy to verify our setup works
"""

import asyncio

from playwright.async_api import async_playwright


async def test_simple_draftkings():
    """Test simple DraftKings navigation."""
    print("🎯 Testing Simple DraftKings Navigation")
    print("=" * 50)

    try:
        # Start Playwright
        playwright = await async_playwright().start()

        # Launch browser without proxy
        browser = await playwright.chromium.launch(
            headless=False,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )

        print("✅ Browser launched")

        # Create context
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )

        print("✅ Context created")

        # Create page
        page = await context.new_page()
        print("✅ Page created")

        # Test IP address first
        print("🌐 Testing IP address...")
        await page.goto("https://httpbin.org/ip", timeout=30000)

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
        await page.screenshot(path="simple_test_screenshot.png")
        print("✅ Screenshot saved: simple_test_screenshot.png")

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
                return True
            else:
                print("❌ Login form not loaded - blocked by anti-bot")
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
    print("🧪 Simple DraftKings Test")
    print("=" * 50)

    success = asyncio.run(test_simple_draftkings())

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)

    if success:
        print("✅ SUCCESS: Basic navigation works!")
        print("🚀 Ready to add Bright Data proxy")
    else:
        print("❌ FAILED: Basic navigation failed")
        print("🔧 Check internet connection and try again")


if __name__ == "__main__":
    main()
