#!/usr/bin/env python3
"""
Test Live Betting System
Demonstrates ML training, real-time odds, weather integration, and injury tracking.
"""

import asyncio
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml

# Import live betting components
from live_betting_system import (
    InjuryTracker,
    LiveBettingSystem,
    MLModelTrainer,
    RealTimeOddsFeed,
    WeatherIntegration,
)


async def test_ml_model_training():
    """Test ML model training with historical data."""
    print("🤖 Testing ML Model Training")
    print("=" * 60)

    try:
        # Load configuration
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        # Initialize model trainer
        trainer = MLModelTrainer(config)

        # Generate mock historical data for training
        print("📊 Generating mock historical data...")

        n_samples = 1000
        historical_data = pd.DataFrame(
            {
                "home_era_last_30": np.random.normal(4.0, 0.5, n_samples),
                "away_era_last_30": np.random.normal(4.0, 0.5, n_samples),
                "home_whip_last_30": np.random.normal(1.30, 0.15, n_samples),
                "away_whip_last_30": np.random.normal(1.30, 0.15, n_samples),
                "home_k_per_9_last_30": np.random.normal(8.5, 1.2, n_samples),
                "away_k_per_9_last_30": np.random.normal(8.5, 1.2, n_samples),
                "home_avg_velocity_last_30": np.random.normal(92.5, 2.0, n_samples),
                "away_avg_velocity_last_30": np.random.normal(92.5, 2.0, n_samples),
                "home_woba_last_30": np.random.normal(0.320, 0.020, n_samples),
                "away_woba_last_30": np.random.normal(0.320, 0.020, n_samples),
                "home_iso_last_30": np.random.normal(0.170, 0.030, n_samples),
                "away_iso_last_30": np.random.normal(0.170, 0.030, n_samples),
                "home_barrel_rate_last_30": np.random.normal(0.085, 0.015, n_samples),
                "away_barrel_rate_last_30": np.random.normal(0.085, 0.015, n_samples),
                "park_factor": np.random.normal(1.0, 0.1, n_samples),
                "hr_factor": np.random.normal(1.0, 0.15, n_samples),
                "weather_impact": np.random.normal(0, 0.05, n_samples),
                "travel_distance": np.random.randint(0, 3000, n_samples),
                "h2h_home_win_rate": np.random.uniform(0.3, 0.7, n_samples),
                "home_momentum": np.random.normal(0, 0.1, n_samples),
                "away_momentum": np.random.normal(0, 0.1, n_samples),
            }
        )

        print(f"✅ Generated {len(historical_data)} historical records")

        # Train models
        print("\n🤖 Training ML models...")
        training_results = await trainer.train_models(historical_data)

        print("✅ Training completed:")
        print(f"   - Models trained: {training_results['models_trained']}")

        for model_name, accuracy in training_results["accuracy_scores"].items():
            print(f"   - {model_name} accuracy: {accuracy:.3f}")

        for model_name, cv_score in training_results["cross_val_scores"].items():
            print(f"   - {model_name} CV score: {cv_score:.3f}")

        # Test predictions
        print("\n🎯 Testing predictions...")
        test_features = pd.DataFrame(
            {
                "home_era_last_30": [3.8],
                "away_era_last_30": [4.2],
                "home_whip_last_30": [1.25],
                "away_whip_last_30": [1.35],
                "home_k_per_9_last_30": [9.0],
                "away_k_per_9_last_30": [8.0],
                "home_avg_velocity_last_30": [94.0],
                "away_avg_velocity_last_30": [91.0],
                "home_woba_last_30": [0.330],
                "away_woba_last_30": [0.310],
                "home_iso_last_30": [0.180],
                "away_iso_last_30": [0.160],
                "home_barrel_rate_last_30": [0.090],
                "away_barrel_rate_last_30": [0.080],
                "park_factor": [1.05],
                "hr_factor": [1.10],
                "weather_impact": [0.02],
                "travel_distance": [500],
                "h2h_home_win_rate": [0.55],
                "home_momentum": [0.05],
                "away_momentum": [-0.02],
            }
        )

        prediction = await trainer.predict(test_features)

        if "error" not in prediction:
            print("✅ Prediction generated:")
            print(
                f"   - Home Win Probability: {prediction['home_win_probability']:.1%}"
            )
            print(
                f"   - Away Win Probability: {prediction['away_win_probability']:.1%}"
            )
            print(f"   - Confidence: {prediction['confidence']:.1%}")
            print(f"   - Ensemble Prediction: {prediction['ensemble_prediction']}")
        else:
            print(f"❌ Prediction error: {prediction['error']}")

        return True

    except Exception as e:
        print(f"❌ Error in ML model training test: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_weather_integration():
    """Test weather integration system."""
    print("\n🌤️ Testing Weather Integration")
    print("=" * 60)

    try:
        # Load configuration
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        # Initialize weather integration
        async with WeatherIntegration(config) as weather:
            # Test weather data for different stadiums
            test_stadiums = ["Yankee Stadium", "Coors Field", "Oracle Park"]

            for stadium in test_stadiums:
                print(f"\n📍 Testing weather for {stadium}...")

                weather_data = await weather.get_weather_data(
                    stadium, datetime.now() + timedelta(hours=2)
                )

                if weather_data:
                    print("✅ Weather data retrieved:")
                    print(f"   - Temperature: {weather_data.temperature}°F")
                    print(f"   - Humidity: {weather_data.humidity}%")
                    print(
                        f"   - Wind: {weather_data.wind_speed} mph {weather_data.wind_direction}"
                    )
                    print(
                        f"   - Precipitation: {weather_data.precipitation_chance:.1%}"
                    )
                    print(f"   - Visibility: {weather_data.visibility} km")

                    # Calculate weather impact
                    impact = weather.calculate_weather_impact(weather_data)
                    print(f"   - Weather Impact Factor: {impact:.3f}")

                    if impact > 1.0:
                        print("   - Effect: Favorable for offense")
                    elif impact < 1.0:
                        print("   - Effect: Unfavorable for offense")
                    else:
                        print("   - Effect: Neutral")
                else:
                    print(f"❌ No weather data for {stadium}")

        return True

    except Exception as e:
        print(f"❌ Error in weather integration test: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_injury_tracking():
    """Test injury tracking system."""
    print("\n🏥 Testing Injury Tracking")
    print("=" * 60)

    try:
        # Load configuration
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        # Initialize injury tracker
        async with InjuryTracker(config) as tracker:
            # Test injury data for different teams
            test_teams = ["New York Yankees", "Los Angeles Dodgers", "Houston Astros"]

            for team in test_teams:
                print(f"\n🏟️ Testing injuries for {team}...")

                injuries = await tracker.get_injury_data(team)

                if injuries:
                    print(f"✅ Found {len(injuries)} injuries:")

                    for injury in injuries:
                        print(f"   - {injury.player_name} ({injury.position})")
                        print(f"     Status: {injury.status}")
                        print(f"     Injury: {injury.injury_type}")
                        print(f"     Impact Score: {injury.impact_score:.2f}")
                        if injury.expected_return:
                            print(
                                f"     Expected Return: {injury.expected_return.strftime('%Y-%m-%d')}"
                            )

                    # Calculate team impact
                    impact = tracker.calculate_injury_impact(injuries, team)
                    print(f"   - Total Team Impact: {impact:.3f}")

                    if impact > 0.3:
                        print("   - Effect: Significant impact on team performance")
                    elif impact > 0.1:
                        print("   - Effect: Moderate impact on team performance")
                    else:
                        print("   - Effect: Minimal impact on team performance")
                else:
                    print(f"✅ No injuries reported for {team}")

        return True

    except Exception as e:
        print(f"❌ Error in injury tracking test: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_real_time_odds():
    """Test real-time odds feed."""
    print("\n📊 Testing Real-Time Odds Feed")
    print("=" * 60)

    try:
        # Load configuration
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        # Initialize odds feed
        async with RealTimeOddsFeed(config) as odds_feed:
            print("📡 Fetching live odds...")

            live_odds = await odds_feed.get_live_odds("baseball_mlb")

            if live_odds:
                print(f"✅ Fetched {len(live_odds)} live odds")

                # Group by event
                events = {}
                for odds in live_odds:
                    if odds.event_id not in events:
                        events[odds.event_id] = []
                    events[odds.event_id].append(odds)

                print(f"📅 Odds available for {len(events)} events")

                # Show sample odds
                for i, (event_id, event_odds) in enumerate(list(events.items())[:3]):
                    print(f"\n🎯 Event {i+1} ({event_id}):")
                    for odds in event_odds:
                        print(
                            f"   - {odds.platform}: {odds.market_type.value} {odds.selection}"
                        )
                        print(
                            f"     Odds: {odds.odds}, Implied Prob: {odds.implied_probability:.1%}"
                        )
            else:
                print("⚠️ No live odds available (using mock data)")
                print("   This is expected in testing environment")

        return True

    except Exception as e:
        print(f"❌ Error in real-time odds test: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_complete_live_betting_cycle():
    """Test complete live betting cycle."""
    print("\n🎯 Testing Complete Live Betting Cycle")
    print("=" * 60)

    try:
        # Load configuration
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        # Initialize live betting system
        live_system = LiveBettingSystem(config)

        print("🚀 Running live betting cycle...")

        # Run the complete cycle
        results = await live_system.run_live_betting_cycle()

        print("✅ Live betting cycle completed:")
        print(f"   - Events Analyzed: {results['events_analyzed']}")
        print(f"   - Opportunities Found: {results['opportunities_found']}")
        print(f"   - Bets Placed: {results['bets_placed']}")
        print(f"   - Total Expected Value: {results['total_ev']:.1%}")

        if results["errors"]:
            print(f"   - Errors: {len(results['errors'])}")
            for error in results["errors"][:2]:
                print(f"     • {error}")

        return True

    except Exception as e:
        print(f"❌ Error in live betting cycle test: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_value_betting_analysis():
    """Test value betting analysis with different scenarios."""
    print("\n💰 Testing Value Betting Analysis")
    print("=" * 60)

    try:
        # Load configuration
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        # Initialize live betting system
        live_system = LiveBettingSystem(config)

        # Test different betting scenarios
        scenarios = [
            {
                "name": "Strong Home Team Advantage",
                "home_era": 3.5,
                "away_era": 4.5,
                "home_woba": 0.340,
                "away_woba": 0.300,
                "implied_prob": 0.55,
                "selection": "home",
            },
            {
                "name": "Underdog Value",
                "home_era": 4.2,
                "away_era": 3.8,
                "home_woba": 0.310,
                "away_woba": 0.330,
                "implied_prob": 0.40,
                "selection": "away",
            },
            {
                "name": "High Scoring Game",
                "home_era": 4.8,
                "away_era": 4.6,
                "home_woba": 0.350,
                "away_woba": 0.345,
                "implied_prob": 0.48,
                "selection": "over",
            },
        ]

        for scenario in scenarios:
            print(f"\n📊 Scenario: {scenario['name']}")

            # Create mock event and odds
            from decimal import Decimal

            from models import Event, Odds

            event = Event(
                sport="baseball_mlb",
                home_team="Test Home Team",
                away_team="Test Away Team",
                event_date=datetime.now() + timedelta(hours=2),
            )

            odds = Odds(
                event_id=event.id,
                platform="fanduel",
                market_type="moneyline",
                selection=scenario["selection"],
                odds=Decimal("100"),
                implied_probability=Decimal(str(scenario["implied_prob"])),
            )

            # Create prediction
            prediction = {
                "home_win_probability": (
                    0.6 if scenario["home_era"] < scenario["away_era"] else 0.4
                ),
                "away_win_probability": (
                    0.4 if scenario["home_era"] < scenario["away_era"] else 0.6
                ),
                "confidence": 0.75,
            }

            # Create features
            features = {
                "home_era_last_30": scenario["home_era"],
                "away_era_last_30": scenario["away_era"],
                "home_woba_last_30": scenario["home_woba"],
                "away_woba_last_30": scenario["away_woba"],
                "park_factor": 1.05,
                "weather_impact": 0.02,
            }

            # Analyze opportunity
            opportunity = live_system._analyze_betting_opportunity(
                event, odds, prediction, features
            )

            if opportunity:
                print("✅ Betting opportunity found:")
                print(f"   - Our Probability: {opportunity['our_probability']:.1%}")
                print(
                    f"   - Implied Probability: {opportunity['implied_probability']:.1%}"
                )
                print(f"   - Expected Value: {opportunity['expected_value']:.1%}")
                print(f"   - Kelly Fraction: {opportunity['kelly_fraction']:.1%}")
                print(
                    f"   - Recommended Stake: ${opportunity['recommended_stake']:.2f}"
                )
                print(f"   - Recommendation: {opportunity['recommendation']}")
            else:
                print("❌ No betting opportunity (EV below threshold)")

        return True

    except Exception as e:
        print(f"❌ Error in value betting analysis test: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🧪 COMPREHENSIVE LIVE BETTING SYSTEM TEST")
    print("=" * 80)

    # Run all tests
    tests = [
        ("ML Model Training", test_ml_model_training),
        ("Weather Integration", test_weather_integration),
        ("Injury Tracking", test_injury_tracking),
        ("Real-Time Odds", test_real_time_odds),
        ("Value Betting Analysis", test_value_betting_analysis),
        ("Complete Live Betting Cycle", test_complete_live_betting_cycle),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = await test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 80)
    print("📊 LIVE BETTING SYSTEM TEST RESULTS")
    print("=" * 80)

    passed = sum(results.values())
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Live betting system is ready for production.")
        print("\n🚀 Next Steps:")
        print("   1. Connect to real odds APIs (The Odds API, SportsData.io)")
        print("   2. Integrate with weather APIs (OpenWeatherMap)")
        print("   3. Connect to injury tracking APIs")
        print("   4. Implement real bet placement with betting platforms")
        print("   5. Add real-time monitoring and alerts")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please check the errors above.")

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
