"""
BrowserBase WebSocket Test
Test WebSocket connection for browser control
"""

import asyncio
import json
import os

import websockets


async def test_browserbase_websocket():
    """Test BrowserBase WebSocket connection."""

    # Get current session
    import httpx

    api_key = os.getenv("BROWSERBASE_API_KEY")
    project_id = os.getenv("BROWSERBASE_PROJECT_ID")

    print("🔌 Testing BrowserBase WebSocket Connection")
    print("=" * 50)

    # Get current session
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.browserbase.com/v1/sessions", headers={"X-BB-API-Key": api_key}
        )

        if response.status_code != 200:
            print(f"❌ Failed to get sessions: {response.status_code}")
            return False

        sessions = response.json()
        if not sessions:
            print("❌ No sessions found")
            return False

        # Get the first running session
        running_session = None
        for session in sessions:
            if session.get("status") == "RUNNING":
                # Get full session details
                session_response = await client.get(
                    f"https://api.browserbase.com/v1/sessions/{session['id']}",
                    headers={"X-BB-API-Key": api_key},
                )

                if session_response.status_code == 200:
                    running_session = session_response.json()
                    break

        if not running_session:
            print("❌ No running session found")
            return False

        print(f"✅ Found running session: {running_session['id']}")
        connect_url = running_session.get("connectUrl")

        if not connect_url:
            print("❌ No connect URL found in session")
            return False

        print(f"🔗 Connect URL: {connect_url[:50]}...")
        print(f"🔗 Selenium URL: {running_session.get('seleniumRemoteUrl', 'N/A')}")

        # Try to connect via WebSocket
        try:
            print("\n🌐 Attempting WebSocket connection...")

            # Connect to the WebSocket
            async with websockets.connect(connect_url) as websocket:
                print("✅ WebSocket connected!")

                # Send a simple message to test
                test_message = {
                    "id": 1,
                    "method": "ping",
                    "sessionId": running_session["id"],
                    "params": {},
                }

                await websocket.send(json.dumps(test_message))
                print("📤 Sent test message")

                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    print(f"📥 Received response: {response}")

                    # Try to send a navigation command
                    nav_message = {
                        "id": 2,
                        "method": "navigate",
                        "sessionId": running_session["id"],
                        "params": {"url": "https://www.google.com"},
                    }

                    await websocket.send(json.dumps(nav_message))
                    print("📤 Sent navigation command")

                    # Wait for response
                    try:
                        nav_response = await asyncio.wait_for(
                            websocket.recv(), timeout=10.0
                        )
                        print(f"📥 Navigation response: {nav_response}")

                        # Check if navigation was successful
                        nav_data = json.loads(nav_response)
                        if "error" not in nav_data:
                            print("✅ Navigation command sent successfully!")
                            return True
                        else:
                            print(f"❌ Navigation failed: {nav_data['error']}")
                            return False

                    except asyncio.TimeoutError:
                        print("⏰ Navigation response timeout")
                        return False

                except asyncio.TimeoutError:
                    print("⏰ Response timeout")
                    return False

        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
            return False


async def main():
    """Main test function."""
    print("🧪 BrowserBase WebSocket Test")
    print("=" * 50)

    success = await test_browserbase_websocket()

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)

    if success:
        print("✅ WebSocket connection successful!")
        print("🎉 BrowserBase is ready for live testing!")
    else:
        print("❌ WebSocket connection failed!")
        print("\n💡 This suggests BrowserBase may require:")
        print("   - Different WebSocket protocol")
        print("   - Authentication in WebSocket connection")
        print("   - Different message format")
        print("   - Selenium WebDriver instead of WebSocket")


if __name__ == "__main__":
    asyncio.run(main())
