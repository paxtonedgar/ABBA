"""
Alternative Bright Data Test
Test different authentication methods for Bright Data
"""

import asyncio
import os

from playwright.async_api import async_playwright


async def test_brightdata_alternative():
    """Test alternative Bright Data authentication methods."""
    print("🌐 Testing Alternative Bright Data Methods")
    print("=" * 50)

    # Get credentials
    username = os.getenv("BRIGHTDATA_USERNAME")
    password = os.getenv("BRIGHTDATA_PASSWORD")

    print(f"Username: {username}")
    print(f"Password: {password[:10]}..." if password else "Password: Not set")

    if not username or not password:
        print("❌ Credentials not set")
        return False

    # Try different proxy configurations
    proxy_configs = [
        {
            "name": "Method 1: Standard HTTP",
            "config": {
                "server": f"http://{username}:{password}@brd.superproxy.io:22225",
                "username": username,
                "password": password,
            },
        },
        {
            "name": "Method 2: HTTPS",
            "config": {
                "server": f"https://{username}:{password}@brd.superproxy.io:22225",
                "username": username,
                "password": password,
            },
        },
        {
            "name": "Method 3: Separate credentials",
            "config": {
                "server": "http://brd.superproxy.io:22225",
                "username": username,
                "password": password,
            },
        },
        {
            "name": "Method 4: Different port",
            "config": {
                "server": f"http://{username}:{password}@brd.superproxy.io:24000",
                "username": username,
                "password": password,
            },
        },
    ]

    for method in proxy_configs:
        print(f"\n🔧 Testing {method['name']}...")

        try:
            # Start Playwright
            playwright = await async_playwright().start()

            # Launch browser with this proxy config
            browser = await playwright.chromium.launch(
                headless=False, proxy=method["config"]
            )

            print(f"✅ Browser launched with {method['name']}")

            # Create context
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )

            # Create page
            page = await context.new_page()

            # Test IP address
            print("🌐 Testing IP address...")
            await page.goto("https://httpbin.org/ip", timeout=30000)

            ip_info = await page.text_content("body")
            print(f"🌐 IP Info: {ip_info}")

            # If IP test works, try DraftKings
            if "origin" in ip_info:
                print("🎯 Testing DraftKings navigation...")
                await page.goto("https://www.draftkings.com/login", timeout=60000)

                title = await page.title()
                current_url = page.url

                print(f"✅ Page title: {title}")
                print(f"✅ Current URL: {current_url}")

                # Take screenshot
                screenshot_name = f"brightdata_{method['name'].replace(' ', '_').replace(':', '')}.png"
                await page.screenshot(path=screenshot_name)
                print(f"✅ Screenshot saved: {screenshot_name}")

                # Check for login form
                if "login" in current_url.lower():
                    await page.wait_for_timeout(5000)

                    email_fields = await page.query_selector_all('input[type="email"]')
                    password_fields = await page.query_selector_all(
                        'input[type="password"]'
                    )

                    print(f"📧 Email fields found: {len(email_fields)}")
                    print(f"🔒 Password fields found: {len(password_fields)}")

                    if email_fields and password_fields:
                        print(f"✅ SUCCESS: {method['name']} worked!")
                        print("🌐 Login form loaded with residential proxy!")

                        # Cleanup
                        await page.close()
                        await context.close()
                        await browser.close()
                        await playwright.stop()

                        return True
                    else:
                        print(f"❌ {method['name']}: Form still blocked")
                else:
                    print(f"❌ {method['name']}: Failed to navigate to login")

            # Cleanup
            await page.close()
            await context.close()
            await browser.close()
            await playwright.stop()

        except Exception as e:
            print(f"❌ {method['name']} failed: {e}")
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
            continue

    return False


def main():
    """Main function."""
    print("🧪 Alternative Bright Data Test")
    print("=" * 50)

    success = asyncio.run(test_brightdata_alternative())

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)

    if success:
        print("✅ SUCCESS: Found working Bright Data configuration!")
        print("🚀 Ready to implement full balance monitoring")
    else:
        print("❌ FAILED: All Bright Data methods failed")
        print("🔧 Check credentials or try different proxy provider")


if __name__ == "__main__":
    main()
