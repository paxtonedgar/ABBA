"""Tests for DuckDB storage layer."""

import pytest
from abba.storage import Storage


@pytest.fixture
def db():
    s = Storage(":memory:")
    yield s
    s.close()


class TestGames:
    def test_upsert_and_query(self, db):
        games = [
            {"game_id": "g1", "sport": "MLB", "date": "2026-04-10",
             "home_team": "NYY", "away_team": "BOS", "status": "scheduled"},
            {"game_id": "g2", "sport": "NHL", "date": "2026-04-10",
             "home_team": "NYR", "away_team": "BOS", "status": "final",
             "home_score": 3, "away_score": 2},
        ]
        assert db.upsert_games(games) == 2

        all_games = db.query_games()
        assert len(all_games) == 2

    def test_filter_by_sport(self, db):
        db.upsert_games([
            {"game_id": "m1", "sport": "MLB", "date": "2026-04-10",
             "home_team": "NYY", "away_team": "BOS"},
            {"game_id": "n1", "sport": "NHL", "date": "2026-04-10",
             "home_team": "NYR", "away_team": "CAR"},
        ])
        mlb = db.query_games(sport="MLB")
        assert len(mlb) == 1
        assert mlb[0]["sport"] == "MLB"

    def test_filter_by_team(self, db):
        db.upsert_games([
            {"game_id": "g1", "sport": "MLB", "date": "2026-04-10",
             "home_team": "NYY", "away_team": "BOS"},
            {"game_id": "g2", "sport": "MLB", "date": "2026-04-10",
             "home_team": "LAD", "away_team": "HOU"},
        ])
        nyy = db.query_games(team="NYY")
        assert len(nyy) == 1

    def test_upsert_replaces(self, db):
        db.upsert_games([
            {"game_id": "g1", "sport": "MLB", "date": "2026-04-10",
             "home_team": "NYY", "away_team": "BOS", "status": "scheduled"},
        ])
        db.upsert_games([
            {"game_id": "g1", "sport": "MLB", "date": "2026-04-10",
             "home_team": "NYY", "away_team": "BOS", "status": "final",
             "home_score": 5, "away_score": 3},
        ])
        games = db.query_games()
        assert len(games) == 1
        assert games[0]["status"] == "final"


class TestOdds:
    def test_insert_and_query(self, db):
        db.upsert_games([{"game_id": "g1", "sport": "MLB", "date": "2026-04-10",
                          "home_team": "NYY", "away_team": "BOS"}])
        db.insert_odds([{
            "game_id": "g1", "sportsbook": "DraftKings", "market_type": "moneyline",
            "home_odds": 1.85, "away_odds": 2.05,
        }])
        odds = db.query_odds(game_id="g1")
        assert len(odds) == 1
        assert odds[0]["sportsbook"] == "DraftKings"


class TestSessions:
    def test_create_and_charge(self, db):
        db.create_session("s1", 100.0)
        session = db.get_session("s1")
        assert session["budget_remaining"] == 100.0

        remaining = db.charge_session("s1", 10.0)
        assert remaining == 90.0

    def test_tool_call_log(self, db):
        db.create_session("s1", 100.0)
        db.log_tool_call("s1", "query_games", {"sport": "MLB"},
                         {"count": 5}, 0.01, 15.2)
        # No assertion on content, just verify it doesn't crash


class TestSchemaDiscovery:
    def test_list_tables(self, db):
        tables = db.list_tables()
        names = [t["table"] for t in tables]
        assert "games" in names
        assert "odds_snapshots" in names
        assert "predictions_cache" in names

    def test_describe_table(self, db):
        cols = db.describe_table("games")
        col_names = [c["column_name"] for c in cols]
        assert "game_id" in col_names
        assert "sport" in col_names


class TestPredictionCache:
    def test_cache_and_retrieve(self, db):
        db.upsert_games([{"game_id": "g1", "sport": "MLB", "date": "2026-04-10",
                          "home_team": "NYY", "away_team": "BOS"}])
        pred = {"home_win_prob": 0.62, "confidence": 0.85}
        db.cache_prediction("g1", "v1", "abc123", pred, ttl_minutes=60)

        cached = db.get_cached_prediction("g1", "v1", "abc123")
        assert cached is not None
        assert cached["home_win_prob"] == 0.62

    def test_cache_miss(self, db):
        cached = db.get_cached_prediction("nonexistent", "v1", "xxx")
        assert cached is None
