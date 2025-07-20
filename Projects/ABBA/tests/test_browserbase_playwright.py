"""
BrowserBase Playwright Integration Test
Replace Selenium with Playwright for better browser automation
"""

import asyncio
import os

from browserbase import Browserbase
from playwright.async_api import async_playwright


async def test_browserbase_playwright():
    """Test BrowserBase integration with Playwright."""
    print("🎭 BrowserBase Playwright Integration Test")
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
        print(f"🌐 Selenium URL: {session.selenium_remote_url}")

        # 2. Launch Playwright with BrowserBase
        print("\n🎭 Launching Playwright...")
        async with async_playwright() as p:
            # Connect to BrowserBase using the connect URL
            browser = await p.chromium.connect_over_cdp(session.connect_url)

            print("✅ Connected to BrowserBase via Playwright!")

            # 3. Create page and navigate
            page = await browser.new_page()
            print("📄 New page created")

            # Navigate to DraftKings
            print("\n🎯 Navigating to DraftKings...")
            await page.goto("https://www.draftkings.com")

            # Wait for page to load
            await page.wait_for_load_state("networkidle")

            # Get page title
            title = await page.title()
            print(f"✅ Page loaded: {title}")

            # 4. Check page content
            print("\n🔍 Checking page content...")
            content = await page.text_content("body")
            print(f"📄 Page content length: {len(content)} characters")

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

            # 5. Check for login elements
            print("\n🔐 Checking login elements...")

            # Look for common login selectors
            login_selectors = [
                'input[type="email"]',
                'input[name="username"]',
                'input[id*="username"]',
                'input[id*="email"]',
                'input[type="password"]',
                'input[name="password"]',
                'input[id*="password"]',
                'button[type="submit"]',
                'input[type="submit"]',
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

            # 6. Check for balance elements
            print("\n💰 Checking for balance elements...")
            balance_selectors = [
                '[class*="balance"]',
                '[id*="balance"]',
                '[class*="account"]',
                '[id*="account"]',
                '[class*="funds"]',
                '[id*="funds"]',
                '[class*="money"]',
                '[id*="money"]',
                '[class*="amount"]',
                '[id*="amount"]',
            ]

            balance_elements = []
            for selector in balance_selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    for element in elements[:3]:  # Limit to first 3 per selector
                        text = await element.text_content()
                        if text and text.strip():
                            balance_elements.append(
                                {"selector": selector, "text": text.strip()[:50]}
                            )
                except:
                    pass

            if balance_elements:
                print(f"💰 Found {len(balance_elements)} potential balance elements:")
                for i, elem in enumerate(balance_elements[:5]):  # Show first 5
                    print(f"   {i+1}. {elem['selector']} - {elem['text']}")
            else:
                print("   ❌ No balance elements found")

            # 7. Navigate to account page
            print("\n🏦 Attempting to navigate to account page...")
            try:
                await page.goto("https://www.draftkings.com/account")
                await page.wait_for_load_state("networkidle")

                account_title = await page.title()
                print(f"✅ Account page loaded: {account_title}")

                # Check account page content
                account_content = await page.text_content("body")
                print(
                    f"📄 Account page content length: {len(account_content)} characters"
                )

                # Look for balance indicators on account page
                account_balance_indicators = [
                    indicator
                    for indicator in balance_indicators
                    if indicator.lower() in account_content.lower()
                ]
                if account_balance_indicators:
                    print(
                        f"💰 Account page balance indicators: {account_balance_indicators}"
                    )

            except Exception as e:
                print(f"❌ Failed to navigate to account page: {e}")

            # 8. Take a screenshot
            print("\n📸 Taking screenshot...")
            await page.screenshot(path="draftkings_screenshot.png")
            print("✅ Screenshot saved: draftkings_screenshot.png")

            # 9. Close browser
            await browser.close()
            print("🔒 Browser closed")

        print("\n✅ Playwright test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧪 BrowserBase Playwright Integration")
    print("=" * 50)

    success = asyncio.run(test_browserbase_playwright())

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)

    if success:
        print("✅ Playwright integration successful!")
        print("🎉 BrowserBase + Playwright is working perfectly!")
        print("\n🚀 Next steps:")
        print("1. Add your DraftKings credentials")
        print("2. Implement balance extraction logic")
        print("3. Set up automated monitoring")
        print("\n💡 Playwright is much better than Selenium for this!")
    else:
        print("❌ Playwright integration failed!")
        print("\n💡 Troubleshooting:")
        print("   - Check BrowserBase session status")
        print("   - Verify Playwright installation")
        print("   - Check DraftKings website availability")


if __name__ == "__main__":
    main()
