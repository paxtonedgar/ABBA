"""
Test Bright Data Integration Setup
Demonstrates the configuration and setup for Bright Data residential proxies
"""

import asyncio
import os

from playwright.async_api import async_playwright


def test_brightdata_configuration():
    """Test Bright Data configuration setup."""
    print("🌐 Bright Data Residential Proxy Integration Test")
    print("=" * 60)

    # Check environment variables
    brightdata_vars = {
        "BRIGHTDATA_USERNAME": os.getenv("BRIGHTDATA_USERNAME"),
        "BRIGHTDATA_PASSWORD": os.getenv("BRIGHTDATA_PASSWORD"),
        "BRIGHTDATA_HOST": os.getenv("BRIGHTDATA_HOST", "brd.superproxy.io"),
        "BRIGHTDATA_PORT": os.getenv("BRIGHTDATA_PORT", "22225"),
        "USE_BRIGHTDATA": os.getenv("USE_BRIGHTDATA", "true"),
    }

    print("📋 Environment Variables Check:")
    for var, value in brightdata_vars.items():
        if value:
            if "PASSWORD" in var:
                print(f"   ✅ {var}: {'*' * len(value)}")
            else:
                print(f"   ✅ {var}: {value}")
        else:
            print(f"   ❌ {var}: Not set")

    print("\n🔧 Alternative Proxy Providers:")
    alternative_providers = {
        "OXYLABS_USERNAME": os.getenv("OXYLABS_USERNAME"),
        "SMARTPROXY_USERNAME": os.getenv("SMARTPROXY_USERNAME"),
    }

    for var, value in alternative_providers.items():
        if value:
            print(f"   ✅ {var}: Configured")
        else:
            print(f"   ❌ {var}: Not configured")

    print("\n📚 Setup Instructions:")
    print("1. Get Bright Data credentials at: https://brightdata.com")
    print("2. Set environment variables:")
    print("   export BRIGHTDATA_USERNAME='your_brightdata_username'")
    print("   export BRIGHTDATA_PASSWORD='your_brightdata_password'")
    print("   export BRIGHTDATA_HOST='brd.superproxy.io'")
    print("   export BRIGHTDATA_PORT='22225'")
    print("   export USE_BRIGHTDATA='true'")

    print("\n🔄 Alternative Providers:")
    print("• Oxylabs: https://oxylabs.io")
    print("• SmartProxy: https://smartproxy.com")
    print("• IPRoyal: https://iproyal.com")

    return bool(
        brightdata_vars["BRIGHTDATA_USERNAME"]
        and brightdata_vars["BRIGHTDATA_PASSWORD"]
    )


async def test_playwright_with_proxy():
    """Test Playwright with proxy configuration."""
    print("\n🧪 Testing Playwright with Proxy Configuration...")

    try:
        playwright = await async_playwright().start()

        # Test browser launch options
        launch_options = {
            "headless": False,
            "args": [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        }

        # Add proxy if configured
        if os.getenv("BRIGHTDATA_USERNAME") and os.getenv("BRIGHTDATA_PASSWORD"):
            proxy_config = {
                "server": f"http://{os.getenv('BRIGHTDATA_USERNAME')}:{os.getenv('BRIGHTDATA_PASSWORD')}@{os.getenv('BRIGHTDATA_HOST', 'brd.superproxy.io')}:{os.getenv('BRIGHTDATA_PORT', '22225')}",
                "username": os.getenv("BRIGHTDATA_USERNAME"),
                "password": os.getenv("BRIGHTDATA_PASSWORD"),
            }
            launch_options["proxy"] = proxy_config
            print("✅ Proxy configuration added to launch options")
        else:
            print("⚠️ No proxy configured - using direct connection")

        # Launch browser
        browser = await playwright.chromium.launch(**launch_options)
        print("✅ Browser launched successfully")

        # Create context
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )
        print("✅ Context created successfully")

        # Create page
        page = await context.new_page()
        print("✅ Page created successfully")

        # Test navigation (without going to DraftKings)
        await page.goto("https://httpbin.org/ip", timeout=30000)
        ip_info = await page.text_content("body")
        print(f"✅ IP check successful: {ip_info[:100]}...")

        # Cleanup
        await page.close()
        await context.close()
        await browser.close()
        await playwright.stop()

        print("✅ Playwright test completed successfully")
        return True

    except Exception as e:
        print(f"❌ Playwright test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Bright Data Integration Test Suite")
    print("=" * 60)

    # Test configuration
    config_ok = test_brightdata_configuration()

    if config_ok:
        print("\n✅ Configuration looks good!")
        print("🚀 Ready to test with DraftKings")
    else:
        print("\n⚠️ Configuration incomplete")
        print("💡 Set up Bright Data credentials first")

    # Test Playwright
    playwright_ok = asyncio.run(test_playwright_with_proxy())

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    if config_ok and playwright_ok:
        print("✅ All tests passed!")
        print("🚀 Ready to run: python draftkings_brightdata_stealth.py")
    else:
        print("❌ Some tests failed")
        print("🔧 Check configuration and try again")


if __name__ == "__main__":
    main()
