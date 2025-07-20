"""
Advanced DraftKings Login Page Analyzer
Analyze the login page with dynamic content waiting and anti-bot handling
"""

import asyncio
import os

from browserbase import Browserbase
from playwright.async_api import async_playwright


async def analyze_login_page_advanced():
    """Analyze the DraftKings login page with advanced techniques."""
    print("🔍 Advanced DraftKings Login Page Analyzer")
    print("=" * 50)

    api_key = os.getenv("BROWSERBASE_API_KEY")
    project_id = os.getenv("BROWSERBASE_PROJECT_ID")

    try:
        # Create BrowserBase session
        print("📊 Creating BrowserBase session...")
        bb = Browserbase(api_key=api_key)
        session = bb.sessions.create(project_id=project_id)
        print(f"✅ Session created: {session.id}")

        # Launch Playwright with stealth
        print("\n🎭 Launching Playwright with stealth...")
        async with async_playwright() as p:
            browser = await p.chromium.connect_over_cdp(session.connect_url)
            print("✅ Connected to BrowserBase via Playwright!")

            # Create page with stealth settings
            page = await browser.new_page()

            # Set stealth user agent
            await page.set_extra_http_headers({
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })

            await page.set_viewport_size({"width": 1920, "height": 1080})

            # Add comprehensive stealth scripts
            await page.add_init_script("""
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
                
                // Override chrome
                Object.defineProperty(window, 'chrome', {
                    get: () => ({
                        runtime: {},
                    }),
                });
            """)

            print("✅ Advanced stealth settings applied!")

            # Navigate to login page
            print("\n🎯 Navigating to login page...")
            await page.goto("https://www.draftkings.com/login", timeout=30000, wait_until="domcontentloaded")

            # Wait for page to fully load
            await page.wait_for_timeout(5000)

            title = await page.title()
            current_url = page.url
            print(f"✅ Page loaded: {title}")
            print(f"Current URL: {current_url}")

            # Take initial screenshot
            await page.screenshot(path="login_page_initial.png")
            print("✅ Initial screenshot saved: login_page_initial.png")

            # Wait for network to be idle
            print("\n⏳ Waiting for network to be idle...")
            try:
                await page.wait_for_load_state("networkidle", timeout=10000)
                print("✅ Network is idle")
            except:
                print("⚠️ Network idle timeout, continuing...")

            # Wait additional time for dynamic content
            print("\n⏳ Waiting for dynamic content...")
            await page.wait_for_timeout(5000)

            # Take screenshot after waiting
            await page.screenshot(path="login_page_after_wait.png")
            print("✅ After-wait screenshot saved: login_page_after_wait.png")

            # Check if we're redirected
            final_url = page.url
            if final_url != current_url:
                print(f"⚠️ Page redirected to: {final_url}")

            # Analyze all input fields with waiting
            print("\n🔍 Analyzing input fields (with waiting)...")

            # Wait for any input fields to appear
            try:
                await page.wait_for_selector("input", timeout=10000)
                print("✅ Input fields detected")
            except:
                print("⚠️ No input fields found after waiting")

            input_fields = await page.query_selector_all("input")
            print(f"Found {len(input_fields)} input fields")

            for i, input_field in enumerate(input_fields):
                try:
                    input_type = await input_field.get_attribute("type")
                    input_name = await input_field.get_attribute("name")
                    input_id = await input_field.get_attribute("id")
                    input_placeholder = await input_field.get_attribute("placeholder")
                    input_class = await input_field.get_attribute("class")
                    input_data_testid = await input_field.get_attribute("data-testid")

                    print(f"\nInput {i+1}:")
                    print(f"  Type: {input_type}")
                    print(f"  Name: {input_name}")
                    print(f"  ID: {input_id}")
                    print(f"  Placeholder: {input_placeholder}")
                    print(f"  Class: {input_class}")
                    print(f"  Data-testid: {input_data_testid}")

                except Exception as e:
                    print(f"  Error analyzing input {i+1}: {e}")

            # Analyze all buttons with waiting
            print("\n🔍 Analyzing buttons (with waiting)...")

            # Wait for any buttons to appear
            try:
                await page.wait_for_selector("button", timeout=10000)
                print("✅ Buttons detected")
            except:
                print("⚠️ No buttons found after waiting")

            buttons = await page.query_selector_all("button")
            print(f"Found {len(buttons)} buttons")

            for i, button in enumerate(buttons):
                try:
                    button_text = await button.text_content()
                    button_type = await button.get_attribute("type")
                    button_class = await button.get_attribute("class")
                    button_data_testid = await button.get_attribute("data-testid")

                    print(f"\nButton {i+1}:")
                    print(f"  Text: {button_text}")
                    print(f"  Type: {button_type}")
                    print(f"  Class: {button_class}")
                    print(f"  Data-testid: {button_data_testid}")

                except Exception as e:
                    print(f"  Error analyzing button {i+1}: {e}")

            # Check for iframes (common for login forms)
            print("\n🔍 Checking for iframes...")
            iframes = await page.query_selector_all("iframe")
            print(f"Found {len(iframes)} iframes")

            for i, iframe in enumerate(iframes):
                try:
                    iframe_src = await iframe.get_attribute("src")
                    iframe_id = await iframe.get_attribute("id")
                    iframe_class = await iframe.get_attribute("class")

                    print(f"\nIframe {i+1}:")
                    print(f"  Src: {iframe_src}")
                    print(f"  ID: {iframe_id}")
                    print(f"  Class: {iframe_class}")

                    # Try to access iframe content
                    if iframe_src:
                        print(f"  Iframe has src: {iframe_src}")

                except Exception as e:
                    print(f"  Error analyzing iframe {i+1}: {e}")

            # Get page content for text analysis
            print("\n🔍 Analyzing page content...")
            content = await page.text_content("body")
            print(f"Page content length: {len(content)} characters")

            # Look for login-related text
            login_keywords = ["login", "sign in", "email", "password", "username", "account", "log in"]
            found_keywords = []

            for keyword in login_keywords:
                if keyword.lower() in content.lower():
                    found_keywords.append(keyword)

            print(f"Found login keywords: {found_keywords}")

            # Check for common anti-bot indicators
            anti_bot_indicators = ["captcha", "verify", "robot", "automation", "blocked", "access denied"]
            found_anti_bot = []

            for indicator in anti_bot_indicators:
                if indicator.lower() in content.lower():
                    found_anti_bot.append(indicator)

            if found_anti_bot:
                print(f"⚠️ Anti-bot indicators found: {found_anti_bot}")
            else:
                print("✅ No anti-bot indicators detected")

            # Try to find any interactive elements
            print("\n🔍 Looking for any interactive elements...")

            # Check for any clickable elements
            clickable_selectors = ["a", "button", "input[type='submit']", "input[type='button']", "[role='button']"]
            total_clickable = 0

            for selector in clickable_selectors:
                elements = await page.query_selector_all(selector)
                total_clickable += len(elements)
                if len(elements) > 0:
                    print(f"Found {len(elements)} {selector} elements")

            print(f"Total clickable elements: {total_clickable}")

            # Close browser
            await browser.close()
            print("🔒 Browser closed")

        print("\n✅ Advanced login page analysis completed!")
        return True

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    print("🧪 Advanced DraftKings Login Page Analysis")
    print("=" * 50)

    success = asyncio.run(analyze_login_page_advanced())

    print("\n" + "=" * 50)
    print("ADVANCED ANALYSIS RESULTS")
    print("=" * 50)

    if success:
        print("✅ Advanced login page analysis successful!")
        print("📋 Check the output above for detailed element analysis")
        print("📸 Check login_page_initial.png and login_page_after_wait.png")
    else:
        print("❌ Advanced login page analysis failed!")


if __name__ == "__main__":
    main()
