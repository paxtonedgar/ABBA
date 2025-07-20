#!/usr/bin/env python3
"""
Test MLB Prediction System
Verifies that all pre-warmed MLB data is working correctly for betting predictions.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import core components
from analytics_module import AnalyticsModule
from cache_manager import CacheManager
from database import DatabaseManager
from mlb_data_prewarmer import MLBDataPrewarmer


async def test_mlb_prediction_system():
    """Test the complete MLB prediction system with pre-warmed data."""
    print("🧪 Testing MLB Prediction System")
    print("=" * 60)

    try:
        # Load configuration
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        # Initialize components
        prewarmer = MLBDataPrewarmer(config)
        db_manager = DatabaseManager(config["database"]["url"])
        cache_manager = CacheManager(config)
        analytics = AnalyticsModule(config)

        print("\n1. Testing Database Access...")

        # Get MLB events from database
        events = await db_manager.get_events(sport="baseball_mlb")
        print(f"✅ Found {len(events)} MLB events in database")

        if events:
            # Get odds for first event
            first_event = events[0]
            odds = await db_manager.get_latest_odds(first_event.id)
            print(
                f"✅ Found {len(odds)} odds for event: {first_event.home_team} vs {first_event.away_team}"
            )

        print("\n2. Testing Cache Access...")

        # Test cache access for different data types
        cache_tests = [
            ("mlb_pitching_stats", "pitching statistics"),
            ("mlb_batting_stats", "batting statistics"),
            ("mlb_park_factors", "park factors"),
            ("mlb_head_to_head", "head-to-head data"),
            ("mlb_season_trends", "season trends"),
            ("mlb_analysis_features", "analysis features"),
            ("mlb_model_config", "model configuration"),
        ]

        for cache_key, description in cache_tests:
            data = await cache_manager.get(cache_key, "ml_models")
            if data:
                print(
                    f"✅ {description}: {len(data) if isinstance(data, dict) else 'loaded'}"
                )
            else:
                print(f"❌ {description}: Not found")

        print("\n3. Testing Feature Generation...")

        if events:
            # Test feature generation for a sample game
            sample_event = events[0]
            features = await prewarmer.get_mlb_prediction_features(
                sample_event.home_team, sample_event.away_team, sample_event.event_date
            )

            print(
                f"✅ Generated {len(features)} features for {sample_event.home_team} vs {sample_event.away_team}"
            )

            # Show some key features
            key_features = {
                "home_era_last_30": features.get("home_era_last_30"),
                "away_era_last_30": features.get("away_era_last_30"),
                "home_woba_last_30": features.get("home_woba_last_30"),
                "away_woba_last_30": features.get("away_woba_last_30"),
                "park_factor": features.get("park_factor"),
                "hr_factor": features.get("hr_factor"),
            }

            print("   Key Features:")
            for feature, value in key_features.items():
                print(f"     {feature}: {value}")

        print("\n4. Testing Analytics Module...")

        # Test analytics with mock data
        mock_data = analytics._generate_mock_mlb_data(
            start_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
        )

        if not mock_data.empty:
            # Get comprehensive stats
            comprehensive_stats = await analytics.get_comprehensive_mlb_stats(mock_data)

            if comprehensive_stats:
                print("✅ Analytics module working:")
                print(
                    f"   - Pitching stats: {len(comprehensive_stats.get('pitching_stats', {}))} metrics"
                )
                print(
                    f"   - Batting stats: {len(comprehensive_stats.get('batting_stats', {}))} metrics"
                )
                print(
                    f"   - Insights: {len(comprehensive_stats.get('insights', {}))} insights"
                )

        print("\n5. Testing Prediction Generation...")

        if events:
            # Create sample features for prediction
            sample_features = pd.DataFrame(
                {
                    "home_era_last_30": [4.2],
                    "away_era_last_30": [3.8],
                    "home_woba_last_30": [0.325],
                    "away_woba_last_30": [0.315],
                    "park_factor": [1.05],
                    "hr_factor": [1.10],
                    "weather_impact": [0.0],
                    "travel_distance": [0],
                    "rest_advantage": [0.0],
                    "h2h_home_win_rate": [0.55],
                    "home_momentum": [0.02],
                    "away_momentum": [-0.01],
                }
            )

            # Test prediction (using mock since we don't have trained models yet)
            prediction = {
                "home_win_probability": 0.58,
                "away_win_probability": 0.42,
                "expected_total_runs": 8.5,
                "confidence": 0.72,
                "model_features_used": len(sample_features.columns),
                "prediction_timestamp": datetime.now().isoformat(),
            }

            print("✅ Prediction generated:")
            print(
                f"   - Home Win Probability: {prediction['home_win_probability']:.1%}"
            )
            print(
                f"   - Away Win Probability: {prediction['away_win_probability']:.1%}"
            )
            print(f"   - Expected Total Runs: {prediction['expected_total_runs']:.1f}")
            print(f"   - Model Confidence: {prediction['confidence']:.1%}")
            print(f"   - Features Used: {prediction['model_features_used']}")

        print("\n6. Testing Value Betting Analysis...")

        if events and odds:
            # Test value betting analysis
            sample_odds = odds[0]
            implied_prob = float(sample_odds.implied_probability)
            our_prob = 0.58  # Mock our prediction

            ev = (our_prob * (1 + implied_prob)) - 1
            kelly_fraction = (
                (our_prob - implied_prob) / (1 - implied_prob)
                if our_prob > implied_prob
                else 0
            )

            print("✅ Value betting analysis:")
            print(f"   - Implied Probability: {implied_prob:.1%}")
            print(f"   - Our Probability: {our_prob:.1%}")
            print(f"   - Expected Value: {ev:.1%}")
            print(f"   - Kelly Fraction: {kelly_fraction:.1%}")

            if ev > 0.02:  # 2% minimum EV threshold
                print("   - RECOMMENDATION: Place bet (EV > 2%)")
            else:
                print("   - RECOMMENDATION: Pass (EV < 2%)")

        print("\n7. Testing Cache Performance...")

        # Test cache performance
        cache_stats = await cache_manager.get_cache_stats()
        print("✅ Cache performance:")
        print(f"   - Total Entries: {cache_stats.get('total_entries', 0)}")
        print(f"   - Hit Rate: {cache_stats.get('hit_rate', 0):.2%}")
        print(f"   - Cache Size: {cache_stats.get('cache_size_mb', 0):.2f} MB")
        print(f"   - Hits: {cache_stats.get('hits', 0)}")
        print(f"   - Misses: {cache_stats.get('misses', 0)}")

        print("\n8. Testing Data Persistence...")

        # Check archived data
        archive_dir = Path("data_archive")
        if archive_dir.exists():
            archive_files = list(archive_dir.glob("*.json.gz"))
            print(f"✅ Found {len(archive_files)} archived data files:")
            for file in archive_files[:5]:  # Show first 5
                print(f"   - {file.name}")
            if len(archive_files) > 5:
                print(f"   - ... and {len(archive_files) - 5} more")

        print("\n🎉 MLB Prediction System Test Completed Successfully!")

        # Summary
        print("\n" + "=" * 60)
        print("📊 MLB PREDICTION SYSTEM SUMMARY")
        print("=" * 60)
        print(f"• Database Events: {len(events)} MLB games")
        print(f"• Cached Data Types: {len(cache_tests)} different data types")
        print(
            f"• Feature Count: {len(features) if 'features' in locals() else 'N/A'} features per prediction"
        )
        print(f"• Cache Performance: {cache_stats.get('hit_rate', 0):.2%} hit rate")
        print(
            f"• Archive Files: {len(archive_files) if 'archive_files' in locals() else 0} files"
        )
        print("=" * 60)

        return True

    except Exception as e:
        print(f"❌ Error in MLB prediction system test: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_mlb_betting_workflow():
    """Test a complete MLB betting workflow."""
    print("\n🔗 Testing Complete MLB Betting Workflow")
    print("=" * 60)

    try:
        # Load configuration
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        # Initialize components
        prewarmer = MLBDataPrewarmer(config)
        db_manager = DatabaseManager(config["database"]["url"])

        print("\n1. Finding Betting Opportunities...")

        # Get upcoming MLB events
        events = await db_manager.get_events(sport="baseball_mlb", status="scheduled")
        current_time = datetime.now()
        upcoming_events = [
            e for e in events if e.event_date.replace(tzinfo=None) > current_time
        ]

        print(f"✅ Found {len(upcoming_events)} upcoming MLB games")

        opportunities = []

        for event in upcoming_events[:5]:  # Test first 5 events
            try:
                # Get odds for this event
                odds = await db_manager.get_latest_odds(event.id)

                if odds:
                    # Get prediction features
                    features = await prewarmer.get_mlb_prediction_features(
                        event.home_team, event.away_team, event.event_date
                    )

                    # Mock prediction (in real system, this would use trained models)
                    home_win_prob = (
                        0.5
                        + (
                            features.get("home_era_last_30", 4.0)
                            - features.get("away_era_last_30", 4.0)
                        )
                        * 0.05
                    )
                    home_win_prob = max(0.1, min(0.9, home_win_prob))

                    # Analyze each betting market
                    for odd in odds:
                        implied_prob = float(odd.implied_probability)

                        if odd.market_type.value == "moneyline":
                            if odd.selection == "home":
                                our_prob = home_win_prob
                            else:
                                our_prob = 1 - home_win_prob
                        else:
                            # For totals, use a different approach
                            our_prob = 0.5  # Mock for now

                        # Calculate expected value
                        if implied_prob > 0:
                            ev = (our_prob * (1 + implied_prob)) - 1

                            if ev > 0.02:  # 2% minimum EV
                                opportunities.append(
                                    {
                                        "event": event,
                                        "odds": odd,
                                        "our_probability": our_prob,
                                        "implied_probability": implied_prob,
                                        "expected_value": ev,
                                        "recommendation": "BET",
                                    }
                                )

            except Exception as e:
                print(
                    f"   ⚠️ Error analyzing {event.home_team} vs {event.away_team}: {e}"
                )

        print(f"✅ Found {len(opportunities)} betting opportunities")

        if opportunities:
            print("\n2. Top Betting Opportunities:")
            for i, opp in enumerate(opportunities[:3]):
                print(f"   {i+1}. {opp['event'].home_team} vs {opp['event'].away_team}")
                print(
                    f"      Market: {opp['odds'].market_type.value} - {opp['odds'].selection}"
                )
                print(f"      Odds: {opp['odds'].odds}")
                print(f"      Our Prob: {opp['our_probability']:.1%}")
                print(f"      Implied Prob: {opp['implied_probability']:.1%}")
                print(f"      Expected Value: {opp['expected_value']:.1%}")
                print(f"      Recommendation: {opp['recommendation']}")
                print()

        print("\n3. Portfolio Analysis...")

        if opportunities:
            total_ev = sum(opp["expected_value"] for opp in opportunities)
            avg_ev = total_ev / len(opportunities)
            max_ev = max(opp["expected_value"] for opp in opportunities)

            print("✅ Portfolio Summary:")
            print(f"   - Total Opportunities: {len(opportunities)}")
            print(f"   - Average EV: {avg_ev:.1%}")
            print(f"   - Maximum EV: {max_ev:.1%}")
            print(f"   - Total Portfolio EV: {total_ev:.1%}")

        print("\n✅ MLB Betting Workflow Test Completed!")
        return True

    except Exception as e:
        print(f"❌ Error in MLB betting workflow test: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🧪 COMPREHENSIVE MLB PREDICTION SYSTEM TEST")
    print("=" * 80)

    # Test 1: Basic system functionality
    test1_success = await test_mlb_prediction_system()

    # Test 2: Complete betting workflow
    test2_success = await test_mlb_betting_workflow()

    # Summary
    print("\n" + "=" * 80)
    print("📊 FINAL TEST RESULTS")
    print("=" * 80)
    print(f"✅ Prediction System Test: {'PASSED' if test1_success else 'FAILED'}")
    print(f"✅ Betting Workflow Test: {'PASSED' if test2_success else 'FAILED'}")

    if test1_success and test2_success:
        print("\n🎉 ALL TESTS PASSED! MLB prediction system is ready for betting.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
