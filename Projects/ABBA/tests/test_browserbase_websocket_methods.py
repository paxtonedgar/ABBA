"""
BrowserBase WebSocket Methods Discovery
Test to discover what methods are available in BrowserBase WebSocket
"""

import asyncio
import json
import os

import httpx
import websockets


async def discover_websocket_methods():
    """Discover available WebSocket methods in BrowserBase."""

    print("🔍 BrowserBase WebSocket Methods Discovery")
    print("=" * 50)

    api_key = os.getenv("BROWSERBASE_API_KEY")
    project_id = os.getenv("BROWSERBASE_PROJECT_ID")

    try:
        # Get or create session
        async with httpx.AsyncClient() as client:
            # Get existing sessions
            response = await client.get(
                "https://api.browserbase.com/v1/sessions",
                headers={"X-BB-API-Key": api_key},
            )

            if response.status_code != 200:
                print(f"❌ Failed to get sessions: {response.status_code}")
                return False

            sessions = response.json()

            # Find running session
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

            print(f"✅ Using session: {running_session['id']}")

            # Connect to WebSocket
            connect_url = running_session.get("connectUrl")
            if not connect_url:
                print("❌ No connect URL in session")
                return False

            print("🔗 Connecting to WebSocket...")
            async with websockets.connect(connect_url) as websocket:
                print("✅ WebSocket connected!")

                # Test basic ping
                print("\n📤 Testing ping...")
                await websocket.send(json.dumps({"id": 1, "method": "ping"}))

                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    print(f"📥 Ping response: {response}")
                except asyncio.TimeoutError:
                    print("⏰ Ping timeout")

                # Test different method patterns
                test_methods = [
                    "ping",
                    "Ping",
                    "PING",
                    "status",
                    "Status",
                    "info",
                    "Info",
                    "getInfo",
                    "get_info",
                    "version",
                    "Version",
                    "capabilities",
                    "Capabilities",
                    "methods",
                    "Methods",
                    "list",
                    "List",
                    "help",
                    "Help",
                ]

                print(f"\n🔍 Testing {len(test_methods)} potential methods...")

                for i, method in enumerate(test_methods):
                    print(f"  {i+1:2d}. Testing: {method}")

                    message = {"id": i + 2, "method": method, "params": {}}

                    await websocket.send(json.dumps(message))

                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                        response_data = json.loads(response)

                        if "error" in response_data:
                            error_code = response_data["error"].get("code", "unknown")
                            error_msg = response_data["error"].get("message", "unknown")
                            print(f"     ❌ Error {error_code}: {error_msg}")
                        else:
                            print(f"     ✅ Success: {response_data}")

                    except asyncio.TimeoutError:
                        print("     ⏰ Timeout")
                    except Exception as e:
                        print(f"     ❌ Exception: {e}")

                # Test with session ID
                print("\n🔍 Testing with session ID...")
                session_methods = [
                    "session.info",
                    "session.status",
                    "session.getInfo",
                    f"session.{running_session['id']}.info",
                    f"session.{running_session['id']}.status",
                ]

                for i, method in enumerate(session_methods):
                    print(f"  {i+1}. Testing: {method}")

                    message = {
                        "id": len(test_methods) + i + 2,
                        "method": method,
                        "params": {},
                    }

                    await websocket.send(json.dumps(message))

                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                        response_data = json.loads(response)

                        if "error" in response_data:
                            error_code = response_data["error"].get("code", "unknown")
                            error_msg = response_data["error"].get("message", "unknown")
                            print(f"     ❌ Error {error_code}: {error_msg}")
                        else:
                            print(f"     ✅ Success: {response_data}")

                    except asyncio.TimeoutError:
                        print("     ⏰ Timeout")
                    except Exception as e:
                        print(f"     ❌ Exception: {e}")

                print("\n✅ Method discovery completed!")
                return True

    except Exception as e:
        print(f"❌ Discovery failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function."""
    print("🧪 BrowserBase WebSocket Methods Discovery")
    print("=" * 50)

    success = asyncio.run(discover_websocket_methods())

    print("\n" + "=" * 50)
    print("DISCOVERY RESULTS")
    print("=" * 50)

    if success:
        print("✅ Method discovery completed!")
        print("📋 Check the output above for available methods")
    else:
        print("❌ Method discovery failed!")


if __name__ == "__main__":
    main()
