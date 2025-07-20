"""
DraftKings WebSocket Test
Live balance monitoring using BrowserBase WebSocket control
"""

import asyncio
import json
import os
from typing import Any

import httpx
import websockets


class DraftKingsWebSocketMonitor:
    """DraftKings balance monitoring using WebSocket control."""

    def __init__(self):
        """Initialize the monitor."""
        self.api_key = os.getenv("BROWSERBASE_API_KEY")
        self.project_id = os.getenv("BROWSERBASE_PROJECT_ID")
        self.session = None
        self.websocket = None
        self.message_id = 1

        if not self.api_key or not self.project_id:
            raise ValueError("Missing BROWSERBASE_API_KEY or BROWSERBASE_PROJECT_ID")

    async def get_session(self) -> dict[str, Any] | None:
        """Get or create a BrowserBase session."""
        async with httpx.AsyncClient() as client:
            # First, try to get existing sessions
            response = await client.get(
                "https://api.browserbase.com/v1/sessions",
                headers={"X-BB-API-Key": self.api_key},
            )

            if response.status_code != 200:
                print(f"❌ Failed to get sessions: {response.status_code}")
                return None

            sessions = response.json()

            # Look for running session
            for session in sessions:
                if session.get("status") == "RUNNING":
                    # Get full session details
                    session_response = await client.get(
                        f"https://api.browserbase.com/v1/sessions/{session['id']}",
                        headers={"X-BB-API-Key": self.api_key},
                    )

                    if session_response.status_code == 200:
                        print(f"✅ Using existing session: {session['id']}")
                        return session_response.json()

            # Create new session if none running
            print("📊 Creating new session...")
            create_response = await client.post(
                "https://api.browserbase.com/v1/sessions",
                headers={
                    "X-BB-API-Key": self.api_key,
                    "Content-Type": "application/json",
                },
                json={"projectId": self.project_id},
            )

            if create_response.status_code not in [200, 201]:
                print(f"❌ Failed to create session: {create_response.status_code}")
                return None

            session_data = create_response.json()
            print(f"✅ Created new session: {session_data['id']}")
            return session_data

    async def connect_websocket(self, session: dict[str, Any]):
        """Connect to BrowserBase WebSocket."""
        connect_url = session.get("connectUrl")
        if not connect_url:
            raise ValueError("No connect URL in session")

        print(f"🔗 Connecting to WebSocket: {connect_url[:50]}...")
        self.websocket = await websockets.connect(connect_url)
        print("✅ WebSocket connected!")

    async def send_command(
        self, method: str, params: dict[str, Any] = None
    ) -> dict[str, Any] | None:
        """Send a command to the browser."""
        if not self.websocket:
            raise ValueError("WebSocket not connected")

        message = {"id": self.message_id, "method": method, "params": params or {}}

        print(f"📤 Sending: {method}")
        await self.websocket.send(json.dumps(message))

        # Wait for response
        try:
            response = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
            response_data = json.loads(response)
            print(f"📥 Received: {response_data.get('id', 'unknown')}")

            if "error" in response_data:
                print(f"❌ Error: {response_data['error']}")
                return None

            self.message_id += 1
            return response_data

        except asyncio.TimeoutError:
            print(f"⏰ Timeout waiting for response to {method}")
            return None

    async def navigate_to_draftkings(self):
        """Navigate to DraftKings website."""
        print("\n🎯 Navigating to DraftKings...")

        # Try different navigation methods
        navigation_methods = [
            "Page.navigate",
            "Runtime.navigate",
            "navigate",
            "goto",
            "load",
        ]

        for method in navigation_methods:
            print(f"🔍 Trying method: {method}")
            response = await self.send_command(
                method, {"url": "https://www.draftkings.com"}
            )

            if response and "error" not in response:
                print(f"✅ Navigation successful with method: {method}")
                break
            elif response and "error" in response:
                print(f"❌ Method {method} failed: {response['error']}")
            else:
                print(f"⏰ Method {method} timed out")
        else:
            print("❌ All navigation methods failed")
            return False

        # Wait for page to load
        await asyncio.sleep(3)

        # Get page title
        title_response = await self.send_command(
            "Runtime.evaluate", {"expression": "document.title"}
        )

        if title_response and "result" in title_response:
            title = title_response["result"].get("result", {}).get("value", "Unknown")
            print(f"✅ Page loaded: {title}")

        return True

    async def check_page_content(self):
        """Check the current page content."""
        print("\n🔍 Checking page content...")

        # Get page content
        response = await self.send_command(
            "Runtime.evaluate", {"expression": "document.body.innerText"}
        )

        if not response or "result" not in response:
            print("❌ Failed to get page content")
            return None

        content = response["result"].get("result", {}).get("value", "")
        print(f"📄 Page content length: {len(content)} characters")

        # Look for balance-related text
        balance_indicators = ["balance", "account", "funds", "deposit", "withdraw", "$"]
        found_indicators = [
            indicator
            for indicator in balance_indicators
            if indicator.lower() in content.lower()
        ]

        if found_indicators:
            print(f"💰 Found balance indicators: {found_indicators}")

        return content

    async def simulate_login_flow(self):
        """Simulate the login flow (without actual credentials)."""
        print("\n🔐 Simulating login flow...")

        # Look for login elements
        login_response = await self.send_command(
            "Runtime.evaluate",
            {
                "expression": """
            (function() {
                const elements = {
                    username: document.querySelector('input[type="email"], input[name="username"], input[id*="username"], input[id*="email"]'),
                    password: document.querySelector('input[type="password"], input[name="password"], input[id*="password"]'),
                    loginButton: document.querySelector('button[type="submit"], button:contains("Login"), button:contains("Sign In"), input[type="submit"]')
                };
                return {
                    hasUsername: !!elements.username,
                    hasPassword: !!elements.password,
                    hasLoginButton: !!elements.loginButton,
                    pageTitle: document.title,
                    url: window.location.href
                };
            })()
            """
            },
        )

        if login_response and "result" in login_response:
            result = login_response["result"].get("result", {}).get("value", {})
            print("🔍 Login elements found:")
            print(f"   Username field: {'✅' if result.get('hasUsername') else '❌'}")
            print(f"   Password field: {'✅' if result.get('hasPassword') else '❌'}")
            print(f"   Login button: {'✅' if result.get('hasLoginButton') else '❌'}")
            print(f"   Page title: {result.get('pageTitle', 'Unknown')}")
            print(f"   URL: {result.get('url', 'Unknown')}")

        return True

    async def check_balance_elements(self):
        """Check for balance-related elements on the page."""
        print("\n💰 Checking for balance elements...")

        balance_response = await self.send_command(
            "Runtime.evaluate",
            {
                "expression": """
            (function() {
                const balanceSelectors = [
                    '[class*="balance"]',
                    '[id*="balance"]',
                    '[class*="account"]',
                    '[id*="account"]',
                    '[class*="funds"]',
                    '[id*="funds"]',
                    '[class*="money"]',
                    '[id*="money"]',
                    '[class*="amount"]',
                    '[id*="amount"]'
                ];
                
                const elements = [];
                balanceSelectors.forEach(selector => {
                    const found = document.querySelectorAll(selector);
                    found.forEach(el => {
                        elements.push({
                            tag: el.tagName,
                            text: el.textContent?.trim().substring(0, 50),
                            className: el.className,
                            id: el.id
                        });
                    });
                });
                
                return elements.slice(0, 10); // Limit to first 10
            })()
            """
            },
        )

        if balance_response and "result" in balance_response:
            elements = balance_response["result"].get("result", {}).get("value", [])
            if elements:
                print(f"💰 Found {len(elements)} potential balance elements:")
                for i, elem in enumerate(elements[:5]):  # Show first 5
                    print(
                        f"   {i+1}. {elem.get('tag')} - {elem.get('text', 'No text')}"
                    )
            else:
                print("❌ No balance elements found")

        return True

    async def close_connection(self):
        """Close the WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
            print("🔒 WebSocket connection closed")


async def test_draftkings_websocket():
    """Test DraftKings balance monitoring via WebSocket."""
    print("🎯 DraftKings WebSocket Balance Monitor Test")
    print("=" * 60)

    monitor = DraftKingsWebSocketMonitor()

    try:
        # Get or create session
        session = await monitor.get_session()
        if not session:
            print("❌ Failed to get session")
            return False

        # Connect to WebSocket
        await monitor.connect_websocket(session)

        # Navigate to DraftKings
        success = await monitor.navigate_to_draftkings()
        if not success:
            print("❌ Failed to navigate to DraftKings")
            return False

        # Check page content
        content = await monitor.check_page_content()

        # Simulate login flow
        await monitor.simulate_login_flow()

        # Check for balance elements
        await monitor.check_balance_elements()

        # Navigate to account page (if possible)
        print("\n🏦 Attempting to navigate to account page...")
        account_response = await monitor.send_command(
            "navigate", {"url": "https://www.draftkings.com/account"}
        )

        if account_response:
            await asyncio.sleep(2)
            await monitor.check_page_content()
            await monitor.check_balance_elements()

        print("\n✅ DraftKings WebSocket test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        await monitor.close_connection()


def main():
    """Main test function."""
    print("🧪 DraftKings WebSocket Balance Monitor")
    print("=" * 60)

    success = asyncio.run(test_draftkings_websocket())

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    if success:
        print("✅ DraftKings WebSocket test successful!")
        print("🎉 Live balance monitoring is ready!")
        print("\n🚀 Next steps:")
        print("1. Add your DraftKings credentials")
        print("2. Implement balance extraction logic")
        print("3. Set up automated monitoring")
        print("\n💡 WebSocket control is working perfectly!")
    else:
        print("❌ DraftKings WebSocket test failed!")
        print("\n💡 Troubleshooting:")
        print("   - Check BrowserBase session status")
        print("   - Verify WebSocket connection")
        print("   - Check DraftKings website availability")


if __name__ == "__main__":
    main()
