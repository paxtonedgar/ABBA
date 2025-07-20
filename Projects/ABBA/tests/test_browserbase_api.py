"""
Simple BrowserBase API Test
Test the API key directly with the correct endpoints
"""

import asyncio
import os

import httpx


async def test_browserbase_api():
    """Test BrowserBase API directly."""

    api_key = os.getenv("BROWSERBASE_API_KEY")
    project_id = os.getenv("BROWSERBASE_PROJECT_ID")

    print("🔍 Testing BrowserBase API Key")
    print("=" * 40)
    print(f"API Key: {api_key[:10]}..." if api_key else "NOT SET")
    print(f"Project ID: {project_id}")
    print()

    if not api_key:
        print("❌ BROWSERBASE_API_KEY not set")
        return False

    if not project_id:
        print("❌ BROWSERBASE_PROJECT_ID not set")
        return False

    # Test 1: Validate API key with project check
    print("🔑 Testing API key with project validation...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.browserbase.com/v1/projects/{project_id}",
                headers={"X-BB-API-Key": api_key, "Content-Type": "application/json"},
                timeout=10.0,
            )

            print(f"Project Validation Status Code: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print("✅ API key validation successful!")
                print(f"   Project: {data.get('name', 'N/A')}")
                print(f"   Project ID: {data.get('id', 'N/A')}")
                print(f"   Status: {data.get('status', 'N/A')}")
            else:
                print(f"❌ API key validation failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False

    except Exception as e:
        print(f"❌ Error validating API key: {e}")
        return False

    # Test 2: Create a session (this will test the API key)
    print("\n📊 Testing session creation...")
    try:
        async with httpx.AsyncClient() as client:
            session_config = {"projectId": project_id}

            response = await client.post(
                "https://api.browserbase.com/v1/sessions",
                headers={"X-BB-API-Key": api_key, "Content-Type": "application/json"},
                json=session_config,
                timeout=30.0,
            )

            print(f"Status Code: {response.status_code}")

            if response.status_code == 201 or response.status_code == 200:
                data = response.json()
                session_id = data.get("id")
                print("✅ Session creation successful!")
                print(f"   Session ID: {session_id}")
                print(f"   Status: {data.get('status', 'N/A')}")

                # Test 3: Navigate to a simple page
                print("\n🧭 Testing navigation...")
                nav_response = await client.post(
                    f"https://api.browserbase.com/v1/sessions/{session_id}/actions",
                    headers={
                        "X-BB-API-Key": api_key,
                        "Content-Type": "application/json",
                    },
                    json={"action": "navigate", "url": "https://www.google.com"},
                    timeout=30.0,
                )

                print(f"Navigation Status Code: {nav_response.status_code}")

                if nav_response.status_code == 200:
                    nav_data = nav_response.json()
                    print("✅ Navigation successful!")
                    print(f"   Response: {nav_data}")

                    # Test 4: Close the session
                    print("\n🔒 Closing session...")
                    close_response = await client.delete(
                        f"https://api.browserbase.com/v1/sessions/{session_id}",
                        headers={"X-BB-API-Key": api_key},
                        timeout=10.0,
                    )

                    if close_response.status_code == 200:
                        print("✅ Session closed successfully!")
                    else:
                        print(f"⚠️  Session close status: {close_response.status_code}")

                    return True
                else:
                    print(f"❌ Navigation failed: {nav_response.status_code}")
                    print(f"Response: {nav_response.text}")
                    return False

            else:
                print(f"❌ Session creation failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False

    except httpx.HTTPStatusError as e:
        print(f"❌ HTTP error: {e.response.status_code}")
        print(f"Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Error testing BrowserBase API: {e}")
        return False


async def main():
    """Main test function."""
    print("🧪 BrowserBase API Key Test")
    print("=" * 50)

    success = await test_browserbase_api()

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)

    if success:
        print("✅ BrowserBase API key is valid!")
        print("\n🎉 You can now run the live balance test.")
        print("\n🚀 Next steps:")
        print("1. Run: python test_live_balance_simple.py")
        print("2. The system will log into your DraftKings account")
        print("3. Check your balance and test fund management")
    else:
        print("❌ BrowserBase API key test failed!")
        print("\n🔧 Troubleshooting steps:")
        print("1. Check your BrowserBase account at https://browserbase.com")
        print("2. Verify your API key is correct")
        print("3. Make sure your account has available credits")
        print("4. Check if your project ID is correct")
        print("5. Contact BrowserBase support if issues persist")


if __name__ == "__main__":
    asyncio.run(main())
