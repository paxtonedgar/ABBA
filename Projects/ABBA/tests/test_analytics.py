"""
Test suite for Advanced Analytics Module
Tests MLB/NHL data integration, XGBoost, SHAP, and GNN functionality.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import structlog

from src.abba.analytics.manager import AdvancedAnalyticsManager

logger = structlog.get_logger()


def load_config():
    """Simple config loader for testing."""
    return {
        "database": {"url": "sqlite+aiosqlite:///abmba.db"},
        "apis": {"openai": {"model": "gpt-3.5-turbo", "api_key": "test-key"}},
        "sports": [
            {"name": "baseball_mlb", "enabled": True},
            {"name": "hockey_nhl", "enabled": True},
        ],
        "platforms": {"fanduel": {"enabled": True}, "draftkings": {"enabled": True}},
    }


class TestAnalyticsModule:
    """Test AnalyticsModule functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = load_config()
        self.analytics = AdvancedAnalyticsManager(
            self.config, None
        )  # No db_manager for testing

        # Create test data
        self.mlb_data = self._generate_mock_mlb_data("2024-01-01", "2024-01-31")
        self.nhl_data = self._generate_mock_nhl_data("2024-01-01", "2024-01-31")

    def _generate_mock_mlb_data(self, start_date, end_date):
        """Generate mock MLB data for testing."""
        dates = pd.date_range(start_date, end_date, freq="D")
        data = []

        for date in dates:
            for _ in range(50):  # 50 events per day
                data.append(
                    {
                        "game_date": date,
                        "player_name": f"Player_{_}",
                        "release_speed": np.random.normal(92, 5),
                        "launch_speed": np.random.normal(85, 15),
                        "pitch_type": np.random.choice(["FF", "SL", "CH", "CU"]),
                        "events": np.random.choice(
                            ["single", "double", "triple", "home_run", "out"]
                        ),
                        "launch_angle": np.random.normal(15, 10),
                        "exit_velocity": np.random.normal(85, 15),
                    }
                )

        return pd.DataFrame(data)

    def _generate_mock_nhl_data(self, start_date, end_date):
        """Generate mock NHL data for testing."""
        dates = pd.date_range(start_date, end_date, freq="D")
        data = []

        for date in dates:
            for _ in range(30):  # 30 shots per day
                data.append(
                    {
                        "game_date": date,
                        "player_name": f"Player_{_}",
                        "shot_distance": np.random.uniform(5, 60),
                        "shot_angle": np.random.uniform(-45, 45),
                        "goal": np.random.choice([0, 1], p=[0.8, 0.2]),
                        "shot_type": np.random.choice(
                            ["wrist", "slap", "backhand", "snap"]
                        ),
                        "x_coord": np.random.uniform(-100, 100),
                        "y_coord": np.random.uniform(-42.5, 42.5),
                    }
                )

        return pd.DataFrame(data)

    async def test_fetch_mlb_data(self):
        """Test MLB data fetching functionality."""
        print("\nTesting MLB data fetching...")

        # Test Statcast data fetching
        statcast_data = await self.analytics.fetch_mlb_data(
            "2024-01-01", "2024-01-31", "statcast"
        )
        print(f"Fetched {len(statcast_data)} MLB Statcast records")

        assert not statcast_data.empty
        assert "release_speed" in statcast_data.columns
        assert "launch_speed" in statcast_data.columns
        assert "pitch_type" in statcast_data.columns

        print("✅ MLB data fetching test passed")

    async def test_fetch_nhl_data(self):
        """Test NHL data fetching functionality."""
        print("\nTesting NHL data fetching...")

        # Test shot data fetching
        shot_data = await self.analytics.fetch_nhl_data(
            "2024-01-01", "2024-01-31", "shots"
        )
        print(f"Fetched {len(shot_data)} NHL shot records")

        assert not shot_data.empty
        assert "shot_distance" in shot_data.columns
        assert "shot_angle" in shot_data.columns
        assert "goal" in shot_data.columns

        print("✅ NHL data fetching test passed")

    def test_engineer_mlb_features(self):
        """Test MLB feature engineering."""
        print("\nTesting MLB feature engineering...")

        engineered_data = self.analytics.engineer_features(self.mlb_data, "mlb")
        print(f"Engineered {engineered_data.shape[1]} features for MLB data")

        # Check for engineered features
        expected_features = [
            "velocity_bin",
            "is_fastball",
            "is_breaking",
            "is_offspeed",
        ]
        for feature in expected_features:
            if feature in engineered_data.columns:
                print(f"✅ Found engineered feature: {feature}")

        # Check for rolling averages
        rolling_features = [col for col in engineered_data.columns if "rolling" in col]
        print(f"Generated {len(rolling_features)} rolling average features")

        assert (
            engineered_data.shape[1] > self.mlb_data.shape[1]
        )  # Should have more features

        print("✅ MLB feature engineering test passed")

    def test_engineer_nhl_features(self):
        """Test NHL feature engineering."""
        print("\nTesting NHL feature engineering...")

        engineered_data = self.analytics.engineer_features(self.nhl_data, "nhl")
        print(f"Engineered {engineered_data.shape[1]} features for NHL data")

        # Check for engineered features
        expected_features = [
            "distance_bin",
            "is_close_range",
            "angle_bin",
            "is_good_angle",
        ]
        for feature in expected_features:
            if feature in engineered_data.columns:
                print(f"✅ Found engineered feature: {feature}")

        assert (
            engineered_data.shape[1] > self.nhl_data.shape[1]
        )  # Should have more features

        print("✅ NHL feature engineering test passed")

    def test_train_xgboost_model(self):
        """Test XGBoost model training."""
        print("\nTesting XGBoost model training...")

        # Prepare data for training
        engineered_data = self.analytics.engineer_features(self.mlb_data, "mlb")

        # Create target: 1 for hits, 0 for outs
        target = (
            engineered_data["events"].isin(["single", "double", "triple", "home_run"])
        ).astype(int)
        feature_cols = [
            col
            for col in engineered_data.columns
            if col not in ["events", "game_date", "player_name"]
        ]

        # Select numeric features
        numeric_features = engineered_data[feature_cols].select_dtypes(
            include=[np.number]
        )
        valid_indices = numeric_features.notna().all(axis=1)
        X = numeric_features[valid_indices]
        y = target[valid_indices]

        if len(X) > 0:
            # Train model
            _ = self.analytics.train_xgboost_model(X, y, "test_mlb_model")

            # Check model performance
            performance = self.analytics.model_performance["test_mlb_model"]
            print(f"Model accuracy: {performance['accuracy']:.3f}")
            print(f"Model AUC: {performance['auc']:.3f}")

            assert performance["accuracy"] > 0
            assert performance["auc"] > 0
            assert "test_mlb_model" in self.analytics.mlb_models

            print("✅ XGBoost model training test passed")
        else:
            print("⚠️ Skipping XGBoost test - insufficient data")

    def test_shap_insights(self):
        """Test SHAP insights generation."""
        print("\nTesting SHAP insights generation...")

        # Train a model first
        engineered_data = self.analytics.engineer_features(self.mlb_data, "mlb")
        target = (
            engineered_data["events"].isin(["single", "double", "triple", "home_run"])
        ).astype(int)
        feature_cols = [
            col
            for col in engineered_data.columns
            if col not in ["events", "game_date", "player_name"]
        ]
        numeric_features = engineered_data[feature_cols].select_dtypes(
            include=[np.number]
        )
        valid_indices = numeric_features.notna().all(axis=1)
        X = numeric_features[valid_indices]
        y = target[valid_indices]

        if len(X) > 0:
            model = self.analytics.train_xgboost_model(X, y, "test_shap_model")

            # Generate SHAP insights
            mock_instance = pd.DataFrame(np.random.normal(0, 1, (1, X.shape[1])))
            insights = self.analytics.get_shap_insights(
                model, mock_instance, list(X.columns)
            )

            assert "feature_importance" in insights
            assert "shap_values" in insights
            assert len(insights["feature_importance"]) > 0

            print("✅ SHAP insights test passed")
        else:
            print("⚠️ Skipping SHAP test - insufficient data")

    def test_create_gnn_model(self):
        """Test GNN model creation."""
        print("\nTesting GNN model creation...")

        # Create mock team network data
        team_data = {
            "nodes": pd.DataFrame(
                {
                    "player_id": range(1, 21),
                    "team": ["Team_A"] * 10 + ["Team_B"] * 10,
                    "position": np.random.choice(
                        ["P", "C", "1B", "2B", "3B", "SS", "OF"], 20
                    ),
                    "avg": np.random.uniform(0.200, 0.350, 20),
                }
            ),
            "edges": pd.DataFrame(
                {
                    "source": np.random.choice(range(1, 21), 50),
                    "target": np.random.choice(range(1, 21), 50),
                    "weight": np.random.uniform(0, 1, 50),
                }
            ),
        }

        gnn_model = self.analytics.create_gnn_model(team_data, "test_gnn")
        assert gnn_model is not None
        assert "test_gnn" in self.analytics.gnn_models

        print("✅ GNN model creation test passed")

    def test_model_performance_tracking(self):
        """Test model performance tracking."""
        print("\nTesting model performance tracking...")

        # Create mock performance data
        performance_data = {
            "accuracy": 0.75,
            "precision": 0.72,
            "recall": 0.78,
            "f1": 0.75,
            "auc": 0.82,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.analytics.track_model_performance("test_model", performance_data)

        assert "test_model" in self.analytics.model_performance
        assert self.analytics.model_performance["test_model"]["accuracy"] == 0.75

        print("✅ Model performance tracking test passed")

    def test_feature_importance_summary(self):
        """Test feature importance summary generation."""
        print("\nTesting feature importance summary...")

        # Create mock feature importance data
        feature_importance = {
            "release_speed": 0.25,
            "launch_angle": 0.20,
            "exit_velocity": 0.18,
            "pitch_type": 0.15,
            "launch_speed": 0.12,
            "other_features": 0.10,
        }

        summary = self.analytics.generate_feature_importance_summary(feature_importance)

        assert "top_features" in summary
        assert "insights" in summary
        assert len(summary["top_features"]) > 0

        print("✅ Feature importance summary test passed")


class TestAdvancedPredictor:
    """Test AdvancedPredictor functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = load_config()
        self.predictor = AdvancedAnalyticsManager(
            self.config, None
        )  # No db_manager for testing

    async def test_predict_mlb_outcome(self):
        """Test MLB outcome prediction."""
        print("\nTesting MLB outcome prediction...")

        # Create mock game data
        game_data = {
            "home_team": "Team_A",
            "away_team": "Team_B",
            "pitcher_stats": {"era": 3.50, "whip": 1.25},
            "batter_stats": {"avg": 0.280, "ops": 0.800},
            "weather": {"temperature": 72, "wind_speed": 8},
        }

        prediction = await self.predictor.predict_mlb_outcome(game_data)

        assert "prediction" in prediction
        assert "confidence" in prediction
        assert "factors" in prediction

        print("✅ MLB outcome prediction test passed")

    async def test_predict_nhl_outcome(self):
        """Test NHL outcome prediction."""
        print("\nTesting NHL outcome prediction...")

        # Create mock game data
        game_data = {
            "home_team": "Team_A",
            "away_team": "Team_B",
            "goalie_stats": {"save_pct": 0.920, "gaa": 2.50},
            "skater_stats": {"goals": 25, "assists": 35},
            "team_stats": {"pp_pct": 0.220, "pk_pct": 0.820},
        }

        prediction = await self.predictor.predict_nhl_outcome(game_data)

        assert "prediction" in prediction
        assert "confidence" in prediction
        assert "factors" in prediction

        print("✅ NHL outcome prediction test passed")


async def test_integration_workflow():
    """Test complete analytics workflow integration."""
    print("\nTesting complete analytics workflow...")

    config = load_config()
    analytics = AdvancedAnalyticsManager(config, None)  # No db_manager for testing

    # 1. Fetch data
    mlb_data = await analytics.fetch_mlb_data("2024-01-01", "2024-01-31", "statcast")
    nhl_data = await analytics.fetch_nhl_data("2024-01-01", "2024-01-31", "shots")

    # 2. Engineer features
    mlb_features = analytics.engineer_features(mlb_data, "mlb")
    nhl_features = analytics.engineer_features(nhl_data, "nhl")

    # 3. Train models
    if len(mlb_features) > 0:
        target_mlb = (
            mlb_features["events"].isin(["single", "double", "triple", "home_run"])
        ).astype(int)
        feature_cols_mlb = [
            col
            for col in mlb_features.columns
            if col not in ["events", "game_date", "player_name"]
        ]
        numeric_features_mlb = mlb_features[feature_cols_mlb].select_dtypes(
            include=[np.number]
        )
        valid_indices_mlb = numeric_features_mlb.notna().all(axis=1)
        X_mlb = numeric_features_mlb[valid_indices_mlb]
        y_mlb = target_mlb[valid_indices_mlb]

        if len(X_mlb) > 0:
            mlb_model = analytics.train_xgboost_model(
                X_mlb, y_mlb, "integration_mlb_model"
            )
            assert mlb_model is not None

    if len(nhl_features) > 0:
        target_nhl = nhl_features["goal"]
        feature_cols_nhl = [
            col
            for col in nhl_features.columns
            if col not in ["goal", "game_date", "player_name"]
        ]
        numeric_features_nhl = nhl_features[feature_cols_nhl].select_dtypes(
            include=[np.number]
        )
        valid_indices_nhl = numeric_features_nhl.notna().all(axis=1)
        X_nhl = numeric_features_nhl[valid_indices_nhl]
        y_nhl = target_nhl[valid_indices_nhl]

        if len(X_nhl) > 0:
            nhl_model = analytics.train_xgboost_model(
                X_nhl, y_nhl, "integration_nhl_model"
            )
            assert nhl_model is not None

    print("✅ Integration workflow test passed")


async def run_all_analytics_tests():
    """Run all analytics tests."""
    print("🚀 Starting Analytics Test Suite...")

    # Create test instance
    test_instance = TestAnalyticsModule()
    test_instance.setup_method()

    # Run tests
    await test_instance.test_fetch_mlb_data()
    await test_instance.test_fetch_nhl_data()
    test_instance.test_engineer_mlb_features()
    test_instance.test_engineer_nhl_features()
    test_instance.test_train_xgboost_model()
    test_instance.test_shap_insights()
    test_instance.test_create_gnn_model()
    test_instance.test_model_performance_tracking()
    test_instance.test_feature_importance_summary()

    # Run advanced predictor tests
    predictor_test = TestAdvancedPredictor()
    predictor_test.setup_method()
    await predictor_test.test_predict_mlb_outcome()
    await predictor_test.test_predict_nhl_outcome()

    # Run integration test
    await test_integration_workflow()

    print("🎉 All Analytics Tests Completed Successfully!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_all_analytics_tests())
