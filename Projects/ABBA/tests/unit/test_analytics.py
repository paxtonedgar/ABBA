"""Unit tests for analytics module."""

from unittest.mock import Mock

import numpy as np
import pytest

from abba.analytics.biometrics import BiometricsProcessor
from abba.analytics.ensemble import EnsembleManager
from abba.analytics.manager import AdvancedAnalyticsManager
from abba.analytics.models import BiometricData, Prediction, UserPatterns
from abba.analytics.personalization import PersonalizationEngine


class TestBiometricsProcessor:
    """Test biometrics processing."""

    @pytest.fixture
    def processor(self):
        return BiometricsProcessor()

    @pytest.mark.asyncio
    async def test_process_valid_data(self, processor):
        """Test processing valid biometric data."""
        data = {
            "heart_rate": [75, 78, 82, 79, 76],
            "fatigue_metrics": {"sleep_quality": 0.8, "stress_level": 0.3},
            "movement": {"total_distance": 5000, "avg_speed": 2.1},
        }

        result = await processor.process(data)

        assert result is not None
        assert isinstance(result, BiometricData)
        assert result.heart_rate["mean_hr"] > 0
        assert 0 <= result.fatigue_level <= 1
        assert result.recovery_status > 0

    @pytest.mark.asyncio
    async def test_process_empty_data(self, processor):
        """Test processing empty data."""
        result = await processor.process({})

        assert result is not None
        assert isinstance(result, BiometricData)

    def test_calculate_trend(self, processor):
        """Test trend calculation."""
        data = np.array([1, 2, 3, 4, 5])
        trend = processor._calculate_trend(data)

        assert isinstance(trend, float)
        assert -1 <= trend <= 1


class TestPersonalizationEngine:
    """Test personalization engine."""

    @pytest.fixture
    def engine(self):
        return PersonalizationEngine()

    @pytest.mark.asyncio
    async def test_analyze_patterns_empty_history(self, engine):
        """Test pattern analysis with empty history."""
        patterns = await engine.analyze_patterns([])

        assert isinstance(patterns, UserPatterns)
        assert patterns.success_rate == 0.0
        assert patterns.risk_tolerance == 0.5

    @pytest.mark.asyncio
    async def test_analyze_patterns_with_history(self, engine):
        """Test pattern analysis with betting history."""
        # Mock betting history
        history = [
            Mock(
                outcome=True,
                sport="MLB",
                amount=50.0,
                timestamp=Mock(hour=14, weekday=1),
            ),
            Mock(
                outcome=False,
                sport="NHL",
                amount=25.0,
                timestamp=Mock(hour=20, weekday=5),
            ),
            Mock(
                outcome=True,
                sport="MLB",
                amount=75.0,
                timestamp=Mock(hour=15, weekday=2),
            ),
        ]

        patterns = await engine.analyze_patterns(history)

        assert isinstance(patterns, UserPatterns)
        assert patterns.success_rate == 2 / 3
        assert "MLB" in patterns.preferred_sports
        assert len(patterns.bet_sizes) == 3

    @pytest.mark.asyncio
    async def test_create_model(self, engine):
        """Test model creation."""
        patterns = UserPatterns(
            success_rate=0.6,
            preferred_sports=["MLB"],
            bet_sizes=[50.0, 75.0],
            time_patterns={},
            risk_tolerance=0.7,
        )

        model = await engine.create_model(patterns)

        assert model is not None
        assert hasattr(model, "fit")


class TestEnsembleManager:
    """Test ensemble manager."""

    @pytest.fixture
    def manager(self):
        return EnsembleManager()

    @pytest.mark.asyncio
    async def test_combine_predictions_average(self, manager):
        """Test average combination method."""
        predictions = [0.6, 0.7, 0.8]
        result = await manager.combine_predictions(predictions, "average")

        assert result == pytest.approx(0.7, rel=1e-2)

    @pytest.mark.asyncio
    async def test_combine_predictions_median(self, manager):
        """Test median combination method."""
        predictions = [0.6, 0.7, 0.8]
        result = await manager.combine_predictions(predictions, "median")

        assert result == 0.7

    @pytest.mark.asyncio
    async def test_combine_predictions_voting(self, manager):
        """Test voting combination method."""
        predictions = [0.6, 0.7, 0.3]  # 2 above 0.5, 1 below
        result = await manager.combine_predictions(predictions, "voting")

        assert result == pytest.approx(2 / 3, rel=1e-2)

    @pytest.mark.asyncio
    async def test_calculate_error_bars(self, manager):
        """Test error bar calculation."""
        predictions = [0.6, 0.7, 0.8]
        result = await manager.calculate_error_bars(predictions)

        assert "confidence" in result
        assert "margin" in result
        assert 0 <= result["confidence"] <= 1
        assert result["margin"] >= 0

    @pytest.mark.asyncio
    async def test_validate_ensemble(self, manager):
        """Test ensemble validation."""
        predictions = [0.6, 0.7, 0.8]
        result = await manager.validate_ensemble(predictions)

        assert "valid" in result
        assert "reason" in result
        assert "metrics" in result
        assert isinstance(result["valid"], bool)


class TestAdvancedAnalyticsManager:
    """Test advanced analytics manager."""

    @pytest.fixture
    def config(self):
        return {"model_cache_dir": "./test_models", "log_level": "INFO"}

    @pytest.fixture
    def db_manager(self):
        return Mock()

    @pytest.fixture
    def manager(self, config, db_manager):
        return AdvancedAnalyticsManager(config, db_manager)

    @pytest.mark.asyncio
    async def test_integrate_biometrics(self, manager):
        """Test biometric integration."""
        player_data = {
            "heart_rate": [75, 78, 82, 79, 76],
            "fatigue_metrics": {"sleep_quality": 0.8, "stress_level": 0.3},
            "movement": {"total_distance": 5000, "avg_speed": 2.1},
        }

        result = await manager.integrate_biometrics(player_data)

        assert isinstance(result, np.ndarray)
        assert result.size > 0

    @pytest.mark.asyncio
    async def test_create_ensemble_model(self, manager):
        """Test ensemble model creation."""
        ensemble = await manager.create_ensemble_model(
            ["random_forest", "gradient_boosting"]
        )

        assert isinstance(ensemble, dict)
        assert len(ensemble) > 0

    @pytest.mark.asyncio
    async def test_ensemble_predictions(self, manager):
        """Test ensemble predictions."""
        # Mock models
        models = [Mock(), Mock()]
        for model in models:
            model.predict_proba.return_value = np.array([[0.3, 0.7]])

        data = np.array([[1.0, 2.0, 3.0]])

        result = await manager.ensemble_predictions(models, data)

        assert result is not None
        assert isinstance(result, Prediction)
        assert result.model_count == 2


class TestPrediction:
    """Test prediction model."""

    def test_prediction_creation(self):
        """Test prediction creation."""
        prediction = Prediction(
            value=0.75, confidence=0.8, error_margin=0.1, model_count=3
        )

        assert prediction.value == 0.75
        assert prediction.confidence == 0.8
        assert prediction.error_margin == 0.1
        assert prediction.model_count == 3
        assert prediction.timestamp is not None


class TestBiometricData:
    """Test biometric data model."""

    def test_biometric_data_creation(self):
        """Test biometric data creation."""
        data = BiometricData(
            heart_rate={"mean_hr": 75.0},
            fatigue_level=0.3,
            movement_metrics={"total_distance": 5000.0},
            recovery_status=0.8,
        )

        assert data.heart_rate["mean_hr"] == 75.0
        assert data.fatigue_level == 0.3
        assert data.recovery_status == 0.8
        assert data.timestamp is not None


class TestUserPatterns:
    """Test user patterns model."""

    def test_user_patterns_creation(self):
        """Test user patterns creation."""
        patterns = UserPatterns(
            success_rate=0.6,
            preferred_sports=["MLB", "NHL"],
            bet_sizes=[50.0, 75.0],
            time_patterns={"hourly": {}},
            risk_tolerance=0.7,
        )

        assert patterns.success_rate == 0.6
        assert "MLB" in patterns.preferred_sports
        assert len(patterns.bet_sizes) == 2
        assert patterns.risk_tolerance == 0.7
