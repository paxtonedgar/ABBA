"""Tests for extracted services — no toolkit, no mixins.

Verifies that PredictionService, MarketService, and DataService
work independently with injected dependencies.
"""

import pytest

from abba.connectors.seed import seed_sample_data
from abba.engine.elo import EloRatings
from abba.engine.ensemble import EnsembleEngine
from abba.engine.features import FeatureEngine
from abba.engine.hockey import HockeyAnalytics
from abba.engine.kelly import KellyEngine
from abba.engine.ml_model import NHLGameModel
from abba.engine.value import ValueEngine
from abba.services.data import DataService
from abba.services.market import MarketService
from abba.services.prediction import PredictionService
from abba.storage import Storage


@pytest.fixture
def db():
    s = Storage(":memory:")
    seed_sample_data(s)
    yield s
    s.close()


@pytest.fixture
def prediction_service(db):
    hockey = HockeyAnalytics()
    ensemble = EnsembleEngine()
    features = FeatureEngine()
    elo = EloRatings(k=4, home_advantage=50)
    ml_model = NHLGameModel()

    # Initialize Elo from completed games
    completed = db.query_games(sport="NHL", status="final", limit=500)
    if completed:
        games = sorted(completed, key=lambda g: g.get("date", ""))
        elo.initialize_from_games(games)

    return PredictionService(
        storage=db,
        hockey=hockey,
        ensemble=ensemble,
        features=features,
        elo=elo,
        ml_model=ml_model,
    )


@pytest.fixture
def market_service(db, prediction_service):
    def predict_fn(gid):
        return prediction_service.predict_generic(gid)

    return MarketService(
        storage=db,
        value=ValueEngine(),
        kelly=KellyEngine(),
        predict_fn=predict_fn,
    )


@pytest.fixture
def data_service(db):
    return DataService(db)


class TestPredictionService:
    """Test PredictionService in isolation (no toolkit)."""

    def test_predict_nhl_returns_prediction(self, prediction_service, db):
        games = db.query_games(sport="NHL", status="scheduled")
        if not games:
            pytest.skip("No scheduled NHL games in seed data")
        game_id = games[0]["game_id"]
        result = prediction_service.predict_nhl(game_id)
        assert "error" not in result
        assert "prediction" in result
        assert "features" in result
        assert result["sport"] == "NHL"

    def test_predict_nhl_exposes_prediction_input_contract(self, prediction_service, db):
        games = db.query_games(sport="NHL", status="scheduled")
        if not games:
            pytest.skip("No scheduled NHL games in seed data")

        result = prediction_service.predict_nhl(games[0]["game_id"])
        assert "error" not in result
        assert "prediction_input" in result
        assert "model_features_used" in result

        prediction_input = result["prediction_input"]
        assert "required_features" in prediction_input
        assert "optional_features" in prediction_input
        assert "provenance" in prediction_input
        assert "team_stats" in prediction_input["provenance"]
        assert "odds" in prediction_input["provenance"]

    def test_predict_nhl_missing_game(self, prediction_service):
        result = prediction_service.predict_nhl("nonexistent-game-id")
        assert "error" in result

    def test_predict_nhl_non_nhl_game(self, prediction_service, db):
        games = db.query_games(sport="MLB")
        if not games:
            pytest.skip("No MLB games in seed data")
        result = prediction_service.predict_nhl(games[0]["game_id"])
        assert "error" in result

    def test_predict_generic_routes_nhl(self, prediction_service, db):
        games = db.query_games(sport="NHL", status="scheduled")
        if not games:
            pytest.skip("No scheduled NHL games")
        result = prediction_service.predict_generic(games[0]["game_id"])
        assert "error" not in result
        assert result["sport"] == "NHL"

    def test_player_impact_healthy_roster(self, prediction_service, db):
        # Query a team that has roster data
        roster = db.query_roster()
        if not roster:
            pytest.skip("No roster data")
        team = roster[0]["team"]
        impact = prediction_service.player_impact(team)
        assert "injury_impact" in impact
        assert "top_scorer_available" in impact
        assert "roster_completeness" in impact
        assert 0.0 <= impact["injury_impact"] <= 0.10

    def test_player_impact_no_roster(self, prediction_service):
        impact = prediction_service.player_impact("NONEXISTENT")
        assert impact["injury_impact"] == 0.0
        assert impact["roster_completeness"] == 1.0

    def test_explain_returns_factors(self, prediction_service, db):
        games = db.query_games(sport="NHL", status="scheduled")
        if not games:
            pytest.skip("No scheduled NHL games")
        result = prediction_service.explain(games[0]["game_id"])
        assert "error" not in result
        assert "top_factors" in result
        assert len(result["top_factors"]) > 0

    def test_build_model_types(self, prediction_service):
        preds = [0.55, 0.52, 0.48, 0.53, 0.50, 0.54]
        features = {"market_implied_prob": 0.0}
        types = prediction_service._build_model_types(preds, None, features)
        assert len(types) == len(preds)
        assert types[0] == "points_log5"

    def test_build_model_types_with_elo(self, prediction_service):
        preds = [0.55, 0.52, 0.48, 0.53, 0.50, 0.54, 0.56]
        features = {"market_implied_prob": 0.0}
        types = prediction_service._build_model_types(preds, 0.56, features)
        assert "elo" in types


class TestMarketService:
    """Test MarketService in isolation."""

    def test_find_value_no_games(self, market_service):
        result = market_service.find_value(sport="NONEXISTENT")
        assert result["count"] == 0
        assert result["games_scanned"] == 0

    def test_compare_odds_no_odds(self, market_service):
        result = market_service.compare_odds("nonexistent-game")
        # Should return without error (just empty)
        assert isinstance(result, dict)


class TestDataService:
    """Test DataService in isolation."""

    def test_initial_refresh_ts_none(self):
        """Fresh database has no refresh timestamp."""
        s = Storage(":memory:")
        ds = DataService(s)
        assert ds.last_refresh_ts is None
        s.close()

    def test_data_service_has_storage(self, data_service):
        assert data_service.storage is not None
