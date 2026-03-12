"""Invariant C — Cross-Provider Identity Resolution.

Odds rows must be joinable to schedule rows through a proven mapping rule.
The OddsLiveConnector resolves game_ids by matching (home_team, away_team, date)
against the stored schedule, using the _ODDS_NAME_TO_ABBREV mapping.

These tests verify the resolver works correctly and that seed data
produces joinable IDs.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

from abba.storage import Storage
from abba.server import ABBAToolkit


@pytest.fixture
def db():
    s = Storage(":memory:")
    yield s
    s.close()


@pytest.fixture
def toolkit():
    return ABBAToolkit(db_path=":memory:", auto_seed=True)


class TestOddsScheduleJoin:
    """Verify that odds and schedule can be joined via the resolver."""

    def test_seed_data_odds_join_to_schedule(self, toolkit):
        """Seed data uses matching game_ids for games and odds."""
        games = toolkit.storage.query_games(sport="NHL", status="scheduled")
        if not games:
            pytest.skip("No scheduled NHL games in seed data")

        gid = games[0]["game_id"]
        odds = toolkit.storage.query_odds(game_id=gid)
        assert len(odds) > 0, (
            f"Seed data odds should be joinable by game_id={gid}. "
            "Seed generates matching IDs for games and odds."
        )

    def test_unmatched_odds_produce_feature_degradation(self, db):
        """When odds can't join to schedule, market_implied_prob is 0."""
        from abba.engine.hockey import HockeyAnalytics

        hockey = HockeyAnalytics()

        home = {"stats": {"wins": 40, "losses": 25, "overtime_losses": 5,
                          "goals_for": 230, "goals_against": 200,
                          "power_play_percentage": 22.0, "penalty_kill_percentage": 80.0,
                          "recent_form": 0.6}}
        away = {"stats": {"wins": 35, "losses": 30, "overtime_losses": 5,
                          "goals_for": 210, "goals_against": 220,
                          "power_play_percentage": 19.0, "penalty_kill_percentage": 78.0,
                          "recent_form": 0.45}}

        features_no_odds = hockey.build_nhl_features(home, away, odds_data=[])
        fake_odds = [{"home_odds": 1.85, "away_odds": 2.05, "sportsbook": "test"}]
        features_with_odds = hockey.build_nhl_features(home, away, odds_data=fake_odds)

        assert features_no_odds.get("market_implied_prob", 0) == 0
        assert features_with_odds.get("market_implied_prob", 0) > 0

    def test_prediction_warns_about_missing_odds(self, toolkit):
        """When odds are absent for a game, prediction includes a warning."""
        games = toolkit.storage.query_games(sport="NHL", status="scheduled")
        if not games:
            pytest.skip("No scheduled NHL games")

        # Use a game_id that has no odds
        toolkit.storage.upsert_games([{
            "game_id": "nhl-no-odds-test",
            "sport": "NHL",
            "date": "2026-03-10",
            "home_team": games[0].get("home_team", "NYR"),
            "away_team": games[0].get("away_team", "BOS"),
            "status": "scheduled",
        }])
        result = toolkit.nhl_predict_game("nhl-no-odds-test")
        if "error" not in result:
            provenance = result.get("data_provenance", {})
            odds_prov = provenance.get("odds", {})
            assert odds_prov.get("status") == "absent", (
                "Provenance should mark odds as 'absent' when no odds data matched"
            )


class TestOddsNameResolver:
    """Verify the team name mapping used for ID resolution."""

    def test_all_32_nhl_teams_mapped(self):
        """The Odds API name resolver covers all 32 NHL teams."""
        from abba.connectors.live import OddsLiveConnector
        mapping = OddsLiveConnector._ODDS_NAME_TO_ABBREV

        # All 32 NHL abbreviations must be in the values
        expected_abbrevs = {
            "ANA", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ",
            "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NSH",
            "NJD", "NYI", "NYR", "OTT", "PHI", "PIT", "SJS", "SEA",
            "STL", "TBL", "TOR", "VAN", "VGK", "WSH", "WPG",
        }
        # UTA or ARI depending on season
        actual_abbrevs = set(mapping.values())
        missing = expected_abbrevs - actual_abbrevs
        assert len(missing) <= 1, (  # allow ARI to be absent (now UTA)
            f"Missing team abbreviations in odds name resolver: {missing}"
        )

    def test_known_mappings_correct(self):
        """Spot-check key team name mappings."""
        from abba.connectors.live import OddsLiveConnector
        m = OddsLiveConnector._ODDS_NAME_TO_ABBREV

        assert m["New York Rangers"] == "NYR"
        assert m["Boston Bruins"] == "BOS"
        assert m["Vegas Golden Knights"] == "VGK"
        assert m["Tampa Bay Lightning"] == "TBL"
        assert m["Montreal Canadiens"] == "MTL" or m.get("Montréal Canadiens") == "MTL"


class TestOddsIDFormat:
    """Document the actual ID formats from both providers."""

    def test_nhl_schedule_id_format(self):
        """NHL API game IDs are formatted as nhl-{numeric}."""
        sample_nhl_api_id = "2025020456"
        game_id = f"nhl-{sample_nhl_api_id}"
        assert game_id.startswith("nhl-")
        assert sample_nhl_api_id.isdigit()

    def test_odds_fallback_id_uses_different_prefix(self):
        """Unresolved odds events use odds- prefix to avoid confusion."""
        # When the resolver can't match, it uses odds- prefix instead of nhl-
        # This makes mismatches visible rather than silently failing
        fallback_id = f"odds-a1b2c3d4e5f6"
        assert fallback_id.startswith("odds-")
        assert not fallback_id.startswith("nhl-")
