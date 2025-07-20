#!/usr/bin/env python3
"""
Simple MLB Testing System Runner
Easy-to-use script to run MLB prediction system tests.
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))


def print_banner():
    """Print the test system banner."""
    print("=" * 70)
    print("🧪 MLB PREDICTION SYSTEM TESTING SUITE")
    print("=" * 70)
    print("Testing all games in the MLB season with real API data")
    print("Includes robust statistical validation and performance benchmarks")
    print("=" * 70)


def print_usage():
    """Print usage instructions."""
    print("\nUsage Options:")
    print("1. Quick Test (Basic functionality):")
    print("   python test_mlb_system.py --quick")
    print()
    print("2. Full Season Test (Comprehensive):")
    print("   python test_mlb_system.py --full")
    print()
    print("3. Custom Season Test:")
    print("   python test_mlb_system.py --season 2024")
    print()
    print("4. Verbose Mode:")
    print("   python test_mlb_system.py --full --verbose")
    print()
    print("5. Help:")
    print("   python test_mlb_system.py --help")


async def run_quick_test():
    """Run a quick test of the MLB system."""
    print("\n🏃 Running Quick MLB Test...")

    try:
        from mlb_season_testing_system import MLBSeasonTester

        # Initialize tester
        tester = MLBSeasonTester()

        # Test basic functionality
        print("• Testing data fetching...")
        season_data = await tester.fetch_mlb_season_data(2024)
        print(f"  ✅ Fetched {len(season_data)} records")

        print("• Testing API performance...")
        api_results = await tester.run_api_performance_tests()
        print(
            f"  ✅ API success rate: {api_results.get('overall_performance', {}).get('success_rate', 0):.2f}"
        )

        print("• Testing data quality...")
        if not season_data.empty:
            quality_results = await tester.run_data_quality_tests(season_data)
            print(
                f"  ✅ Data quality score: {quality_results.get('overall_score', 0):.3f}"
            )

        print("\n✅ Quick test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        return False


async def run_full_test(season=2024, verbose=False):
    """Run a full comprehensive test."""
    print(f"\n🚀 Running Full MLB {season} Season Test...")

    try:
        from run_mlb_tests import MLBTestRunner

        # Initialize test runner
        runner = MLBTestRunner()

        # Run comprehensive tests
        results = await runner.run_comprehensive_tests(season)

        # Print summary
        runner.print_summary()

        return results.get("summary", {}).get("overall_status") in ["excellent", "good"]

    except Exception as e:
        print(f"\n❌ Full test failed: {e}")
        return False


async def main():
    """Main function."""
    print_banner()

    # Parse command line arguments
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print_usage()
        return

    # Determine test type
    if "--quick" in args:
        success = await run_quick_test()
    elif "--full" in args or "--season" in args:
        # Extract season if specified
        season = 2024
        for i, arg in enumerate(args):
            if arg == "--season" and i + 1 < len(args):
                try:
                    season = int(args[i + 1])
                except ValueError:
                    print(f"Invalid season: {args[i + 1]}")
                    return

        verbose = "--verbose" in args
        success = await run_full_test(season, verbose)
    else:
        print_usage()
        return

    # Exit with appropriate code
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed - check the results for details")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
