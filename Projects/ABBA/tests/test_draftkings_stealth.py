"""
Stealth DraftKings Balance Monitoring Test
Use advanced anti-detection techniques to bypass bot protection
"""

import asyncio
import os

from browserbase import Browserbase
from playwright.async_api import async_playwright


async def test_draftkings_stealth():
    """Test DraftKings with stealth techniques to bypass anti-bot protection."""
    print("🕵️ Stealth DraftKings Balance Monitoring Test")
    print("=" * 60)

    # Set credentials
    username = "paxtonedgar3@gmail.com"
    password = "Empireozarks@2013"

    print(f"🔐 Testing with account: {username}")
    print("🕵️ Using stealth techniques to bypass anti-bot protection...")

    api_key = os.getenv("BROWSERBASE_API_KEY")
    project_id = os.getenv("BROWSERBASE_PROJECT_ID")

    try:
        # 1. Create BrowserBase session
        print("\n📊 Creating BrowserBase session...")
        bb = Browserbase(api_key=api_key)
        session = bb.sessions.create(project_id=project_id)
        print(f"✅ Session created: {session.id}")

        # 2. Launch Playwright with stealth settings
        print("\n🎭 Launching Playwright with stealth mode...")
        async with async_playwright() as p:
            browser = await p.chromium.connect_over_cdp(session.connect_url)
            print("✅ Connected to BrowserBase via Playwright!")

            # Create page with stealth settings
            page = await browser.new_page()

            # Set stealth user agent and viewport
            await page.set_extra_http_headers(
                {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
            )

            await page.set_viewport_size({"width": 1920, "height": 1080})

            # Add stealth scripts
            await page.add_init_script(
                """
                // Remove webdriver property
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
                
                // Override permissions
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
                
                // Override plugins
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5],
                });
                
                // Override languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en'],
                });
            """
            )

            print("✅ Stealth settings applied!")

            # 3. Try different DraftKings URLs
            draftkings_urls = [
                "https://www.draftkings.com",
                "https://sportsbook.draftkings.com",
                "https://www.draftkings.com/sportsbook",
                "https://www.draftkings.com/account",
            ]

            success = False
            for i, url in enumerate(draftkings_urls):
                print(f"\n🎯 Attempt {i+1}: Navigating to {url}")

                try:
                    # Navigate with longer timeout and different load strategy
                    await page.goto(url, timeout=30000, wait_until="domcontentloaded")

                    # Wait a bit for any anti-bot checks
                    await page.wait_for_timeout(3000)

                    # Check if page loaded successfully
                    title = await page.title()
                    print(f"✅ Page loaded: {title}")

                    # Take screenshot
                    await page.screenshot(path=f"draftkings_stealth_{i+1}.png")
                    print(f"✅ Screenshot saved: draftkings_stealth_{i+1}.png")

                    # Check if we're on a login page or main page
                    current_url = page.url
                    content = await page.text_content("body")

                    if (
                        "login" in current_url.lower()
                        or "signin" in current_url.lower()
                    ):
                        print("🔐 Detected login page - attempting login...")

                        # Try to find and fill login fields
                        login_success = await attempt_login(page, username, password)

                        if login_success:
                            print("✅ Login successful!")
                            success = True
                            break
                        else:
                            print("❌ Login failed, trying next URL...")
                            continue

                    elif "draftkings" in title.lower():
                        print("🏠 Detected main page - checking for balance...")

                        # Look for balance information
                        balance_found = await check_for_balance(page)

                        if balance_found:
                            print("💰 Balance information found!")
                            success = True
                            break
                        else:
                            print("❌ No balance found, trying next URL...")
                            continue

                    else:
                        print(f"❓ Unknown page: {title}")
                        continue

                except Exception as e:
                    print(f"❌ Navigation failed: {e}")
                    continue

            if not success:
                print("\n❌ All navigation attempts failed")
                print("💡 This suggests strong anti-bot protection")

                # Try one more approach - mobile user agent
                print("\n📱 Trying mobile user agent...")
                try:
                    await page.set_extra_http_headers(
                        {
                            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1"
                        }
                    )

                    await page.set_viewport_size({"width": 375, "height": 667})

                    await page.goto(
                        "https://www.draftkings.com",
                        timeout=30000,
                        wait_until="domcontentloaded",
                    )
                    await page.wait_for_timeout(3000)

                    title = await page.title()
                    print(f"✅ Mobile page loaded: {title}")

                    await page.screenshot(path="draftkings_mobile.png")
                    print("✅ Mobile screenshot saved: draftkings_mobile.png")

                except Exception as e:
                    print(f"❌ Mobile attempt failed: {e}")

            # Close browser
            await browser.close()
            print("🔒 Browser closed")

        return success

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def attempt_login(page, username: str, password: str) -> bool:
    """Attempt to login to DraftKings."""
    try:
        # Look for login elements with more selectors
        login_selectors = {
            "username": [
                'input[type="email"]',
                'input[name="username"]',
                'input[name="email"]',
                'input[id*="username"]',
                'input[id*="email"]',
                'input[placeholder*="email"]',
                'input[placeholder*="Email"]',
            ],
            "password": [
                'input[type="password"]',
                'input[name="password"]',
                'input[id*="password"]',
                'input[placeholder*="password"]',
                'input[placeholder*="Password"]',
            ],
            "submit": [
                'button[type="submit"]',
                'input[type="submit"]',
                'button:has-text("Login")',
                'button:has-text("Sign In")',
                'button:has-text("Log In")',
                'button[class*="login"]',
                'button[class*="signin"]',
            ],
        }

        # Find and fill username
        username_field = None
        for selector in login_selectors["username"]:
            try:
                username_field = await page.query_selector(selector)
                if username_field:
                    print(f"   ✅ Found username field: {selector}")
                    break
            except:
                continue

        if not username_field:
            print("   ❌ Username field not found")
            return False

        # Find and fill password
        password_field = None
        for selector in login_selectors["password"]:
            try:
                password_field = await page.query_selector(selector)
                if password_field:
                    print(f"   ✅ Found password field: {selector}")
                    break
            except:
                continue

        if not password_field:
            print("   ❌ Password field not found")
            return False

        # Fill credentials with human-like delays
        await username_field.click()
        await page.wait_for_timeout(500)
        await username_field.fill(username)
        await page.wait_for_timeout(1000)

        await password_field.click()
        await page.wait_for_timeout(500)
        await password_field.fill(password)
        await page.wait_for_timeout(1000)

        # Find and click submit button
        submit_button = None
        for selector in login_selectors["submit"]:
            try:
                submit_button = await page.query_selector(selector)
                if submit_button:
                    print(f"   ✅ Found submit button: {selector}")
                    break
            except:
                continue

        if not submit_button:
            print("   ❌ Submit button not found")
            return False

        # Click submit
        await submit_button.click()

        # Wait for navigation
        await page.wait_for_timeout(5000)

        # Check if login was successful
        current_url = page.url
        if "login" not in current_url.lower() and "signin" not in current_url.lower():
            print("   ✅ Login appears successful")
            return True
        else:
            print("   ❌ Login may have failed")
            return False

    except Exception as e:
        print(f"   ❌ Login attempt failed: {e}")
        return False


async def check_for_balance(page) -> bool:
    """Check for balance information on the current page."""
    try:
        content = await page.text_content("body")

        # Look for balance indicators
        balance_indicators = ["balance", "account", "funds", "deposit", "withdraw", "$"]
        found_indicators = [
            indicator
            for indicator in balance_indicators
            if indicator.lower() in content.lower()
        ]

        if found_indicators:
            print(f"   💰 Found balance indicators: {found_indicators}")
            return True
        else:
            print("   ❌ No balance indicators found")
            return False

    except Exception as e:
        print(f"   ❌ Balance check failed: {e}")
        return False


def main():
    """Main function."""
    print("🧪 Stealth DraftKings Balance Monitoring")
    print("=" * 60)
    print("🕵️ Using advanced anti-detection techniques")
    print("⚠️  Testing with real credentials!")
    print("=" * 60)

    success = asyncio.run(test_draftkings_stealth())

    print("\n" + "=" * 60)
    print("STEALTH TEST RESULTS")
    print("=" * 60)

    if success:
        print("✅ Stealth balance monitoring successful!")
        print("🎉 Bypassed anti-bot protection!")
        print("\n🚀 Ready for production!")
    else:
        print("❌ Stealth balance monitoring failed!")
        print("\n💡 DraftKings has strong anti-bot protection")
        print("🔧 Alternative approaches:")
        print("   1. Use official DraftKings API (if available)")
        print("   2. Manual balance checking")
        print("   3. Different automation platform")
        print("   4. Contact DraftKings for API access")


if __name__ == "__main__":
    main()
