#!/usr/bin/env python3
"""
Basic Analytics Example for ABBA

This example demonstrates the core analytics functionality including:
- Biometric data processing
- Ensemble predictions
- User personalization
"""

import asyncio
import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abba.analytics.ensemble import EnsembleManager
from abba.analytics.manager import AdvancedAnalyticsManager
from abba.core.config import Config
from abba.core.logging import setup_logging


async def main():
    """Run the basic analytics example."""

    # Setup logging
    setup_logging(level="INFO")

    # Initialize configuration
    config = Config()

    # Mock database manager
    db_manager = None

    # Initialize analytics manager
    analytics = AdvancedAnalyticsManager(config.model_dump(), db_manager)

    print("🚀 ABBA Basic Analytics Example")
    print("=" * 50)

    # 1. Biometric Data Processing
    print("\n1. Processing Biometric Data...")

    biometric_data = {
        "heart_rate": [75, 78, 82, 79, 76, 80, 77, 81, 78, 75],
        "fatigue_metrics": {"sleep_quality": 0.8, "stress_level": 0.3, "workload": 0.6},
        "movement": {
            "total_distance": 5000,
            "avg_speed": 2.1,
            "max_speed": 3.5,
            "acceleration_count": 15,
        },
    }

    features = await analytics.integrate_biometrics(biometric_data)
    print(f"✅ Extracted {features.size} features from biometric data")

    # 2. Create Ensemble Model
    print("\n2. Creating Ensemble Model...")

    ensemble = await analytics.create_ensemble_model(
        ["random_forest", "gradient_boosting"]
    )

    print(f"✅ Created ensemble with {len(ensemble)} models")

    # 3. Make Predictions
    print("\n3. Making Ensemble Predictions...")

    # Mock some models for demonstration
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

    # Create simple mock models
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=10, random_state=42)

    # Mock training data
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)

    # Train models
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)

    models = [rf_model, gb_model]

    # Make prediction
    if features.size > 0:
        # Reshape features to match training data
        features_reshaped = np.random.rand(1, 10)  # Mock features
        prediction = await analytics.ensemble_predictions(models, features_reshaped)

        if prediction:
            print(f"✅ Prediction: {prediction.value:.4f}")
            print(f"   Confidence: {prediction.confidence:.4f}")
            print(f"   Error Margin: ±{prediction.error_margin:.4f}")
            print(f"   Models Used: {prediction.model_count}")

    # 4. User Personalization
    print("\n4. User Personalization...")

    # Mock user betting history
    from datetime import datetime
    from unittest.mock import Mock

    user_history = [
        Mock(
            outcome=True,
            sport="MLB",
            amount=50.0,
            odds=1.8,
            confidence=0.7,
            timestamp=datetime(2024, 1, 1, 14, 0),
        ),
        Mock(
            outcome=False,
            sport="NHL",
            amount=25.0,
            odds=2.1,
            confidence=0.6,
            timestamp=datetime(2024, 1, 2, 20, 0),
        ),
        Mock(
            outcome=True,
            sport="MLB",
            amount=75.0,
            odds=1.9,
            confidence=0.8,
            timestamp=datetime(2024, 1, 3, 15, 0),
        ),
    ]

    patterns = await analytics.personalization_engine.analyze_patterns(user_history)

    print(f"✅ User Success Rate: {patterns.success_rate:.2%}")
    print(f"   Preferred Sports: {patterns.preferred_sports}")
    print(f"   Risk Tolerance: {patterns.risk_tolerance:.2f}")
    print(f"   Average Bet Size: ${np.mean(patterns.bet_sizes):.2f}")

    # 5. Ensemble Validation
    print("\n5. Ensemble Validation...")

    ensemble_manager = EnsembleManager()
    predictions = [0.6, 0.7, 0.8, 0.65, 0.75]

    validation = await ensemble_manager.validate_ensemble(predictions)

    print(f"✅ Ensemble Valid: {validation['valid']}")
    print(f"   Reason: {validation['reason']}")
    print(f"   Agreement: {validation['metrics'].get('agreement', 0):.3f}")

    print("\n🎉 Example completed successfully!")
    print("\nThis demonstrates the core ABBA analytics capabilities:")
    print("- Biometric data processing and feature extraction")
    print("- Ensemble model creation and prediction")
    print("- User behavior pattern analysis")
    print("- Prediction validation and confidence estimation")


if __name__ == "__main__":
    asyncio.run(main())
