"""Tests for extracted services — no toolkit, no mixins.

Verifies that PredictionService, MarketService, and DataService
work independently with injected dependencies.
"""

import pytest

from abba.connectors.live import NHLLiveConnector
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

    def test_predict_nhl_derives_rest_edge_from_schedule(self, prediction_service, db):
        games = db.query_games(sport="NHL", status="scheduled")
        if not games:
            pytest.skip("No scheduled NHL games in seed data")

        result = prediction_service.predict_nhl(games[0]["game_id"])
        assert "error" not in result

        prediction_input = result["prediction_input"]
        rest_provenance = prediction_input["provenance"]["rest"]
        assert rest_provenance["status"] == "present"
        assert "rest_edge" not in (result.get("defaulted_features") or [])
        assert "rest_edge" in prediction_input["features"]
        assert "rest_info" in result["context_only"]["context_only_features"]

    def test_predict_nhl_prefers_game_level_goalie_override(self, prediction_service, db):
        games = db.query_games(sport="NHL", status="scheduled")
        if not games:
            pytest.skip("No scheduled NHL games in seed data")

        game = games[0]
        db.upsert_goaltender_stats([
            {
                "goaltender_id": "override-home-goalie",
                "team": game["home_team"],
                "season": "2025-26",
                "stats": {
                    "name": "Override Home",
                    "games_played": 4,
                    "games_started": 2,
                    "save_pct": 0.931,
                    "gaa": 2.10,
                    "gsaa": 8.0,
                },
            },
            {
                "goaltender_id": "override-away-goalie",
                "team": game["away_team"],
                "season": "2025-26",
                "stats": {
                    "name": "Override Away",
                    "games_played": 3,
                    "games_started": 1,
                    "save_pct": 0.889,
                    "gaa": 3.60,
                    "gsaa": -7.0,
                },
            },
        ])
        db.upsert_games([{
            "game_id": game["game_id"],
            "sport": game["sport"],
            "date": game["date"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "home_score": game.get("home_score"),
            "away_score": game.get("away_score"),
            "venue": game.get("venue"),
            "status": game.get("status", "scheduled"),
            "metadata": {
                **(game.get("metadata") or {}),
                "home_goalie_id": "override-home-goalie",
                "away_goalie_id": "override-away-goalie",
                "goalie_source": "test_override",
            },
            "source": game.get("source", "seed"),
        }])

        result = prediction_service.predict_nhl(game["game_id"])

        assert result["home_goaltender"] == "Override Home"
        assert result["away_goaltender"] == "Override Away"
        assert result["data_provenance"]["home_goaltender"]["selection_method"] == "game_metadata_override"
        assert result["data_provenance"]["away_goaltender"]["selection_method"] == "game_metadata_override"
        assert result["prediction_input"]["provenance"]["goaltenders"]["source"] == "game_metadata_override"

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


class TestNHLLiveConnector:
    def test_extract_confirmed_goalies_from_play_by_play(self):
        connector = NHLLiveConnector()
        payload = {
            "rosterSpots": [
                {"teamId": 6, "playerId": 1001, "positionCode": "G"},
                {"teamId": 6, "playerId": 1002, "positionCode": "G"},
                {"teamId": 28, "playerId": 2001, "positionCode": "G"},
                {"teamId": 28, "playerId": 2002, "positionCode": "G"},
            ],
            "plays": [
                {"eventId": 1, "details": {"goalieInNetId": 2001}},
                {"eventId": 2, "details": {"goalieInNetId": 1002}},
                {"eventId": 3, "details": {"goalieInNetId": 2002}},
            ],
        }

        starters = connector._extract_confirmed_goalies_from_play_by_play(payload)

        assert starters == {"28": "2001", "6": "1002"}

    def test_fetch_special_teams_by_team_scales_percentages(self, monkeypatch):
        connector = NHLLiveConnector()

        def fake_fetch_json(url):
            assert "20252026" in url
            return {
                "data": [
                    {
                        "teamFullName": "New York Rangers",
                        "powerPlayPct": 0.24321,
                        "penaltyKillPct": 0.81234,
                        "faceoffWinPct": 0.51789,
                    },
                    {
                        "teamFullName": "Boston Bruins",
                        "powerPlayPct": 0.19876,
                        "penaltyKillPct": 0.775,
                        "faceoffWinPct": None,
                    },
                ],
            }

        monkeypatch.setattr(connector, "_fetch_json", fake_fetch_json)

        stats = connector._fetch_special_teams_by_team("2025-26")

        assert stats["NYR"]["power_play_percentage"] == 24.32
        assert stats["NYR"]["penalty_kill_percentage"] == 81.23
        assert stats["NYR"]["faceoff_win_percentage"] == 51.79
        assert stats["BOS"]["power_play_percentage"] == 19.88
        assert stats["BOS"]["penalty_kill_percentage"] == 77.5
        assert "faceoff_win_percentage" not in stats["BOS"]

    def test_fetch_standings_merges_special_teams(self, db, monkeypatch):
        connector = NHLLiveConnector()

        def fake_fetch_json(url):
            if "standings/now" in url:
                return {
                    "standings": [
                        {
                            "teamAbbrev": {"default": "NYR"},
                            "seasonId": 20252026,
                            "wins": 40,
                            "losses": 20,
                            "otLosses": 5,
                            "points": 85,
                            "goalFor": 220,
                            "goalAgainst": 180,
                            "goalDifferential": 40,
                            "gamesPlayed": 65,
                            "regulationWins": 30,
                            "streakCode": "W",
                            "streakCount": 3,
                            "homeWins": 20,
                            "homeLosses": 10,
                            "homeOtLosses": 2,
                            "roadWins": 20,
                            "roadLosses": 10,
                            "roadOtLosses": 3,
                            "l10Wins": 7,
                            "l10Losses": 2,
                            "l10OtLosses": 1,
                            "divisionName": "Metropolitan",
                            "conferenceName": "Eastern",
                            "teamName": {"default": "New York Rangers"},
                            "wildcardSequence": 0,
                            "pointPctg": 0.654,
                        }
                    ]
                }
            if "stats/rest/en/team/summary" in url:
                return {
                    "data": [
                        {
                            "teamFullName": "New York Rangers",
                            "powerPlayPct": 0.241,
                            "penaltyKillPct": 0.815,
                            "faceoffWinPct": 0.521,
                        }
                    ]
                }
            raise AssertionError(f"Unexpected URL {url}")

        monkeypatch.setattr(connector, "_fetch_json", fake_fetch_json)

        result = connector._fetch_standings(db)
        stored = db.query_team_stats(team_id="NYR", sport="NHL", season="2025-26")

        assert result["status"] == "ok"
        assert result["special_teams_teams_updated"] == 1
        assert stored[0]["stats"]["power_play_percentage"] == 24.1
        assert stored[0]["stats"]["penalty_kill_percentage"] == 81.5
        assert stored[0]["stats"]["faceoff_win_percentage"] == 52.1
