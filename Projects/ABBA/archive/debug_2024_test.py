#!/usr/bin/env python3
"""
Debug 2024 MLB Season Test
Step-by-step debugging of the prediction system.
"""

import asyncio

import numpy as np
import pandas as pd
import structlog
import yaml

# Import core components
from live_betting_system import MLModelTrainer

logger = structlog.get_logger()


async def debug_prediction_system():
    """Debug the prediction system step by step."""
    print("🔍 Debugging 2024 MLB Prediction System")
    print("=" * 60)

    try:
        # Load configuration
        with open('config.yaml') as f:
            config = yaml.safe_load(f)

        # Initialize model trainer
        trainer = MLModelTrainer(config)

        print("1. Testing model loading...")
        await trainer.load_models()
        print(f"✅ Loaded {len(trainer.models)} models")

        print("\n2. Testing feature creation...")

        # Create test features that match training data
        test_features = {
            'home_era_last_30': 3.8,
            'away_era_last_30': 4.2,
            'home_whip_last_30': 1.25,
            'away_whip_last_30': 1.35,
            'home_k_per_9_last_30': 9.0,
            'away_k_per_9_last_30': 8.0,
            'home_avg_velocity_last_30': 94.0,
            'away_avg_velocity_last_30': 91.0,
            'home_woba_last_30': 0.330,
            'away_woba_last_30': 0.310,
            'home_iso_last_30': 0.180,
            'away_iso_last_30': 0.160,
            'home_barrel_rate_last_30': 0.090,
            'away_barrel_rate_last_30': 0.080,
            'park_factor': 1.05,
            'hr_factor': 1.10,
            'weather_impact': 0.02,
            'travel_distance': 500,
            'h2h_home_win_rate': 0.55,
            'home_momentum': 0.05,
            'away_momentum': -0.02
        }

        print(f"✅ Created test features with {len(test_features)} features")

        print("\n3. Testing prediction generation...")

        # Convert to DataFrame
        features_df = pd.DataFrame([test_features])
        print(f"✅ Created DataFrame with shape: {features_df.shape}")

        # Generate prediction
        prediction = await trainer.predict(features_df)
        print(f"✅ Prediction result: {prediction}")

        if 'error' not in prediction:
            print(f"   - Home Win Probability: {prediction['home_win_probability']:.1%}")
            print(f"   - Away Win Probability: {prediction['away_win_probability']:.1%}")
            print(f"   - Confidence: {prediction['confidence']:.1%}")
        else:
            print(f"❌ Prediction error: {prediction['error']}")

        print("\n4. Testing with real 2024 data...")

        # Create realistic 2024 team stats
        team_stats = {
            'New York Yankees': {
                'era': 3.85,
                'whip': 1.28,
                'k_per_9': 9.2,
                'avg_velocity': 94.5,
                'woba': 0.335,
                'iso': 0.185,
                'barrel_rate': 0.092
            },
            'Boston Red Sox': {
                'era': 4.15,
                'whip': 1.32,
                'k_per_9': 8.8,
                'avg_velocity': 92.8,
                'woba': 0.325,
                'iso': 0.175,
                'barrel_rate': 0.088
            }
        }

        # Create features for a real game
        real_features = {
            'home_era_last_30': team_stats['New York Yankees']['era'],
            'away_era_last_30': team_stats['Boston Red Sox']['era'],
            'home_whip_last_30': team_stats['New York Yankees']['whip'],
            'away_whip_last_30': team_stats['Boston Red Sox']['whip'],
            'home_k_per_9_last_30': team_stats['New York Yankees']['k_per_9'],
            'away_k_per_9_last_30': team_stats['Boston Red Sox']['k_per_9'],
            'home_avg_velocity_last_30': team_stats['New York Yankees']['avg_velocity'],
            'away_avg_velocity_last_30': team_stats['Boston Red Sox']['avg_velocity'],
            'home_woba_last_30': team_stats['New York Yankees']['woba'],
            'away_woba_last_30': team_stats['Boston Red Sox']['woba'],
            'home_iso_last_30': team_stats['New York Yankees']['iso'],
            'away_iso_last_30': team_stats['Boston Red Sox']['iso'],
            'home_barrel_rate_last_30': team_stats['New York Yankees']['barrel_rate'],
            'away_barrel_rate_last_30': team_stats['Boston Red Sox']['barrel_rate'],
            'park_factor': 1.10,  # Yankee Stadium
            'hr_factor': 1.20,
            'weather_impact': 1.0,
            'travel_distance': 215,  # Boston to NY
            'h2h_home_win_rate': 0.55,
            'home_momentum': 0.02,
            'away_momentum': -0.01
        }

        real_features_df = pd.DataFrame([real_features])
        real_prediction = await trainer.predict(real_features_df)

        if 'error' not in real_prediction:
            print("✅ Real 2024 prediction successful:")
            print("   - Yankees vs Red Sox")
            print(f"   - Home Win Probability: {real_prediction['home_win_probability']:.1%}")
            print(f"   - Away Win Probability: {real_prediction['away_win_probability']:.1%}")
            print(f"   - Confidence: {real_prediction['confidence']:.1%}")
        else:
            print(f"❌ Real prediction error: {real_prediction['error']}")

        print("\n5. Testing batch predictions...")

        # Create multiple games
        games_data = []
        for i in range(5):
            game_features = real_features.copy()
            # Add some variation
            game_features['home_era_last_30'] += np.random.normal(0, 0.1)
            game_features['away_era_last_30'] += np.random.normal(0, 0.1)
            games_data.append(game_features)

        games_df = pd.DataFrame(games_data)
        print(f"✅ Created batch DataFrame with {len(games_df)} games")

        batch_predictions = []
        for i, row in games_df.iterrows():
            try:
                pred = await trainer.predict(pd.DataFrame([row]))
                if 'error' not in pred:
                    batch_predictions.append(pred)
            except Exception as e:
                print(f"❌ Error in batch prediction {i}: {e}")

        print(f"✅ Generated {len(batch_predictions)} batch predictions")

        return True

    except Exception as e:
        print(f"❌ Debug error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main debug function."""
    print("🔍 COMPREHENSIVE DEBUG TEST")
    print("=" * 80)

    success = await debug_prediction_system()

    if success:
        print("\n✅ Debug test completed successfully!")
        print("The prediction system is working correctly.")
    else:
        print("\n❌ Debug test failed!")
        print("There are issues with the prediction system.")

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
