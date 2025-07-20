"""
BrowserBase Playwright Simple Test
Test with simple website first, then DraftKings with better error handling
"""

import asyncio
import os

from browserbase import Browserbase
from playwright.async_api import async_playwright


async def test_browserbase_playwright_simple():
    """Test BrowserBase integration with Playwright - simple version."""
    print("🎭 BrowserBase Playwright Simple Test")
    print("=" * 50)

    api_key = os.getenv("BROWSERBASE_API_KEY")
    project_id = os.getenv("BROWSERBASE_PROJECT_ID")

    if not api_key or not project_id:
        print("❌ Missing BROWSERBASE_API_KEY or BROWSERBASE_PROJECT_ID")
        return False

    try:
        # 1. Create BrowserBase session
        print("📊 Creating BrowserBase session...")
        bb = Browserbase(api_key=api_key)
        session = bb.sessions.create(project_id=project_id)

        print(f"✅ Session created: {session.id}")
        print(f"🔗 Connect URL: {session.connect_url[:50]}...")

        # 2. Launch Playwright with BrowserBase
        print("\n🎭 Launching Playwright...")
        async with async_playwright() as p:
            # Connect to BrowserBase using the connect URL
            browser = await p.chromium.connect_over_cdp(session.connect_url)

            print("✅ Connected to BrowserBase via Playwright!")

            # 3. Create page and navigate to simple site first
            page = await browser.new_page()
            print("📄 New page created")

            # Test with a simple website first
            print("\n🌐 Testing with simple website...")
            await page.goto("https://httpbin.org/html", timeout=30000)

            # Get page title
            title = await page.title()
            print(f"✅ Simple page loaded: {title}")

            # Check content
            content = await page.text_content("body")
            print(f"📄 Content length: {len(content)} characters")

            # 4. Now try DraftKings with better error handling
            print("\n🎯 Testing DraftKings...")
            try:
                # Navigate to DraftKings with shorter timeout
                await page.goto("https://www.draftkings.com", timeout=15000)

                # Wait for basic load, not networkidle
                await page.wait_for_load_state("domcontentloaded", timeout=10000)

                # Get page title
                title = await page.title()
                print(f"✅ DraftKings page loaded: {title}")

                # Check page content
                content = await page.text_content("body")
                print(f"📄 DraftKings content length: {len(content)} characters")

                # Look for balance-related text
                balance_indicators = [
                    "balance",
                    "account",
                    "funds",
                    "deposit",
                    "withdraw",
                    "$",
                ]
                found_indicators = [
                    indicator
                    for indicator in balance_indicators
                    if indicator.lower() in content.lower()
                ]

                if found_indicators:
                    print(f"💰 Found balance indicators: {found_indicators}")

                # Check for login elements
                print("\n🔐 Checking login elements...")
                login_selectors = [
                    'input[type="email"]',
                    'input[name="username"]',
                    'input[type="password"]',
                    'input[name="password"]',
                    'button[type="submit"]',
                ]

                found_elements = {}
                for selector in login_selectors:
                    try:
                        element = await page.query_selector(selector)
                        if element:
                            found_elements[selector] = True
                            print(f"   ✅ Found: {selector}")
                    except:
                        pass

                if not found_elements:
                    print("   ❌ No login elements found")

                # Take a screenshot
                print("\n📸 Taking screenshot...")
                await page.screenshot(path="draftkings_playwright.png")
                print("✅ Screenshot saved: draftkings_playwright.png")

            except Exception as e:
                print(f"❌ DraftKings navigation failed: {e}")
                print("💡 This might be due to anti-bot protection")

                # Try to get any content that loaded
                try:
                    title = await page.title()
                    print(f"📄 Partial page title: {title}")

                    # Take screenshot of what loaded
                    await page.screenshot(path="draftkings_partial.png")
                    print("✅ Partial screenshot saved: draftkings_partial.png")
                except:
                    pass

            # 5. Test with a different sports betting site
            print("\n🏈 Testing alternative sports site...")
            try:
                await page.goto("https://www.espn.com", timeout=15000)
                await page.wait_for_load_state("domcontentloaded", timeout=10000)

                title = await page.title()
                print(f"✅ ESPN page loaded: {title}")

                content = await page.text_content("body")
                print(f"📄 ESPN content length: {len(content)} characters")

                # Look for sports-related content
                sports_indicators = [
                    "sports",
                    "football",
                    "basketball",
                    "baseball",
                    "scores",
                ]
                found_sports = [
                    indicator
                    for indicator in sports_indicators
                    if indicator.lower() in content.lower()
                ]

                if found_sports:
                    print(f"🏈 Found sports indicators: {found_sports}")

            except Exception as e:
                print(f"❌ ESPN navigation failed: {e}")

            # 6. Close browser
            await browser.close()
            print("🔒 Browser closed")

        print("\n✅ Playwright simple test completed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧪 BrowserBase Playwright Simple Integration")
    print("=" * 50)

    success = asyncio.run(test_browserbase_playwright_simple())

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)

    if success:
        print("✅ Playwright integration successful!")
        print("🎉 BrowserBase + Playwright is working!")
        print("\n🚀 Next steps:")
        print("1. Handle DraftKings anti-bot protection")
        print("2. Add your DraftKings credentials")
        print("3. Implement balance extraction logic")
        print("\n💡 Playwright is much better than Selenium!")
    else:
        print("❌ Playwright integration failed!")
        print("\n💡 Troubleshooting:")
        print("   - Check BrowserBase session status")
        print("   - Verify Playwright installation")
        print("   - Check network connectivity")


if __name__ == "__main__":
    main()
