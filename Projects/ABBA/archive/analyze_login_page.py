"""
DraftKings Login Page Analyzer
Analyze the login page structure to find correct selectors
"""

import asyncio
import os

from browserbase import Browserbase
from playwright.async_api import async_playwright


async def analyze_login_page():
    """Analyze the DraftKings login page structure."""
    print("🔍 DraftKings Login Page Analyzer")
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

            # Add stealth scripts
            await page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
            """)

            print("✅ Stealth settings applied!")

            # Navigate to login page
            print("\n🎯 Navigating to login page...")
            await page.goto("https://www.draftkings.com/login", timeout=30000, wait_until="domcontentloaded")
            await page.wait_for_timeout(3000)

            title = await page.title()
            current_url = page.url
            print(f"✅ Page loaded: {title}")
            print(f"Current URL: {current_url}")

            # Take screenshot
            await page.screenshot(path="login_page_analysis.png")
            print("✅ Screenshot saved: login_page_analysis.png")

            # Analyze all input fields
            print("\n🔍 Analyzing input fields...")
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

            # Analyze all buttons
            print("\n🔍 Analyzing buttons...")
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

            # Look for specific login-related elements
            print("\n🔍 Looking for login-specific elements...")

            # Check for forms
            forms = await page.query_selector_all("form")
            print(f"Found {len(forms)} forms")

            for i, form in enumerate(forms):
                try:
                    form_action = await form.get_attribute("action")
                    form_method = await form.get_attribute("method")
                    form_class = await form.get_attribute("class")
                    print(f"\nForm {i+1}:")
                    print(f"  Action: {form_action}")
                    print(f"  Method: {form_method}")
                    print(f"  Class: {form_class}")
                except Exception as e:
                    print(f"  Error analyzing form {i+1}: {e}")

            # Get page content for text analysis
            print("\n🔍 Analyzing page content...")
            content = await page.text_content("body")

            # Look for login-related text
            login_keywords = ["login", "sign in", "email", "password", "username", "account"]
            found_keywords = []

            for keyword in login_keywords:
                if keyword.lower() in content.lower():
                    found_keywords.append(keyword)

            print(f"Found login keywords: {found_keywords}")

            # Close browser
            await browser.close()
            print("🔒 Browser closed")

        print("\n✅ Login page analysis completed!")
        return True

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    print("🧪 DraftKings Login Page Analysis")
    print("=" * 50)

    success = asyncio.run(analyze_login_page())

    print("\n" + "=" * 50)
    print("ANALYSIS RESULTS")
    print("=" * 50)

    if success:
        print("✅ Login page analysis successful!")
        print("📋 Check the output above for element details")
        print("📸 Check login_page_analysis.png for visual reference")
    else:
        print("❌ Login page analysis failed!")


if __name__ == "__main__":
    main()
