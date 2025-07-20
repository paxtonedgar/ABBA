"""
Live DraftKings Balance Monitoring Test
Test with real credentials to extract actual balance information
"""

import asyncio

from browserbase_playwright_executor import BrowserBasePlaywrightExecutor


async def test_draftkings_live_login():
    """Test DraftKings login and balance extraction with real credentials."""
    print("🎯 Live DraftKings Balance Monitoring Test")
    print("=" * 60)

    # Set credentials
    username = "paxtonedgar3@gmail.com"
    password = "Empireozarks@2013"

    print(f"🔐 Testing with account: {username}")
    print("💰 Attempting to extract real balance information...")

    executor = BrowserBasePlaywrightExecutor()

    try:
        # 1. Navigate to DraftKings
        print("\n🎯 Step 1: Navigating to DraftKings...")
        if not await executor.navigate_to_draftkings():
            print("❌ Failed to navigate to DraftKings")
            return False

        # Take screenshot before login
        await executor.take_screenshot("draftkings_before_login.png")
        print("✅ Screenshot saved: draftkings_before_login.png")

        # 2. Login to DraftKings
        print("\n🔐 Step 2: Logging into DraftKings...")
        login_success = await executor.login_to_draftkings(username, password)

        if not login_success:
            print("❌ Login failed - this might be due to:")
            print("   - Anti-bot protection")
            print("   - Captcha requirements")
            print("   - Account security measures")
            print("   - Website changes")

            # Take screenshot of failed login
            await executor.take_screenshot("draftkings_login_failed.png")
            print("✅ Screenshot saved: draftkings_login_failed.png")
            return False

        print("✅ Login successful!")

        # Take screenshot after login
        await executor.take_screenshot("draftkings_after_login.png")
        print("✅ Screenshot saved: draftkings_after_login.png")

        # 3. Extract balance information
        print("\n💰 Step 3: Extracting balance information...")
        balance_info = await executor.extract_balance_info()

        if balance_info and balance_info.account_balance:
            print("\n" + "=" * 60)
            print("💰 BALANCE INFORMATION EXTRACTED!")
            print("=" * 60)
            print(f"   Account Balance: ${balance_info.account_balance:,.2f}")
            print(f"   Available Balance: ${balance_info.available_balance:,.2f}")
            print(
                f"   Pending Balance: ${balance_info.pending_balance:,.2f}"
                if balance_info.pending_balance
                else "   Pending Balance: Not detected"
            )
            print(f"   Currency: {balance_info.currency}")
            print(f"   Source: {balance_info.source}")
            print(f"   Timestamp: {balance_info.timestamp}")
            print("=" * 60)

            # Take final screenshot
            await executor.take_screenshot("draftkings_balance_extracted.png")
            print("✅ Screenshot saved: draftkings_balance_extracted.png")

            return True

        else:
            print("\n❌ No balance information extracted")
            print("💡 This might be due to:")
            print("   - Balance not visible on current page")
            print("   - Different page layout")
            print("   - Account restrictions")
            print("   - Need to navigate to specific balance page")

            # Take screenshot for debugging
            await executor.take_screenshot("draftkings_no_balance.png")
            print("✅ Screenshot saved: draftkings_no_balance.png")

            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        await executor.close()


def main():
    """Main function."""
    print("🧪 Live DraftKings Balance Monitoring")
    print("=" * 60)
    print("⚠️  WARNING: This test uses real credentials!")
    print("⚠️  Make sure you're in a secure environment!")
    print("=" * 60)

    success = asyncio.run(test_draftkings_live_login())

    print("\n" + "=" * 60)
    print("LIVE TEST RESULTS")
    print("=" * 60)

    if success:
        print("✅ Live balance monitoring successful!")
        print("🎉 Real balance information extracted!")
        print("\n🚀 Production ready!")
        print("💡 You can now:")
        print("   1. Set up automated monitoring")
        print("   2. Create balance alerts")
        print("   3. Integrate with your trading system")
        print("   4. Deploy to production")
    else:
        print("❌ Live balance monitoring failed!")
        print("\n💡 Possible reasons:")
        print("   - Anti-bot protection detected")
        print("   - Captcha or security challenge")
        print("   - Account security measures")
        print("   - Website layout changes")
        print("\n🔧 Next steps:")
        print("   1. Check the screenshots for visual clues")
        print("   2. Review DraftKings security settings")
        print("   3. Try manual login to verify account status")
        print("   4. Consider using different automation approach")


if __name__ == "__main__":
    main()
