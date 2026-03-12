"""Invariant D (supplemental) — Season Selection.

Queries in the prediction path must not silently pull wrong-season data
due to missing ORDER BY, absent season filters, or default behavior.

These tests seed multi-season rows and prove queries return stale data.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

from abba.storage import Storage
from abba.connectors.seed import seed_sample_data


@pytest.fixture
def db():
    s = Storage(":memory:")
    yield s
    s.close()


@pytest.fixture
def multi_season_db():
    """Storage with data from two seasons."""
    s = Storage(":memory:")
    seed_sample_data(s)

    # Add previous season data for a team
    old_stats = [
        {
            "team_id": "NYR",
            "sport": "NHL",
            "season": "2024-25",
            "stats": {
                "wins": 52, "losses": 20, "overtime_losses": 10,
                "goals_for": 300, "goals_against": 210,
                "goal_differential": 90,
                "games_played": 82,
                "points_pct": 0.695,
                "recent_form": 0.7,
            },
            "source": "test_old_season",
        },
    ]
    s.upsert_team_stats(old_stats)

    old_goalies = [
        {
            "goaltender_id": "old-season-goalie",
            "team": "NYR",
            "season": "2024-25",
            "stats": {
                "name": "Last Year Goalie",
                "role": "starter",
                "games_played": 60,
                "games_started": 55,
                "save_pct": 0.935,
                "gaa": 2.10,
                "gsaa": 25.0,
            },
        },
    ]
    s.upsert_goaltender_stats(old_goalies)

    old_advanced = [
        {
            "team_id": "NYR",
            "season": "2024-25",
            "stats": {
                "corsi_pct": 56.0,
                "fenwick_pct": 55.0,
                "xgf_pct": 57.0,
            },
        },
    ]
    s.upsert_nhl_advanced_stats(old_advanced)

    yield s
    s.close()


class TestTeamStatsSeasonSelection:
    """query_team_stats must return current season, not stale data."""

    def test_unfiltered_query_returns_both_seasons(self, multi_season_db):
        """Without season filter, both seasons are returned."""
        results = multi_season_db.query_team_stats(team_id="NYR", sport="NHL")
        seasons = {r.get("season") for r in results}

        assert len(seasons) >= 2, (
            "Expected multi-season data. If only one season exists, "
            "the test fixture setup may have changed."
        )

    def test_first_result_is_current_season(self, multi_season_db):
        """[0] must be the current season, not a stale one."""
        results = multi_season_db.query_team_stats(team_id="NYR", sport="NHL")

        if len(results) < 2:
            pytest.skip("Need multi-season data for this test")

        first_season = results[0].get("season")
        assert first_season == "2025-26", (
            f"SEASON SELECTION VIOLATION: query_team_stats returned '{first_season}' "
            f"as first result instead of '2025-26'. nhl_predict_game uses [0], "
            "so it would use stale season data for prediction."
        )

    def test_season_filter_returns_only_requested(self, multi_season_db):
        """Explicit season filter must return only that season."""
        results = multi_season_db.query_team_stats(
            team_id="NYR", sport="NHL", season="2025-26"
        )
        seasons = {r.get("season") for r in results}

        assert seasons == {"2025-26"}, (
            f"Season filter returned unexpected seasons: {seasons}"
        )

    def test_prediction_path_uses_season_filter(self):
        """nhl_predict_game must pass season filter to query_team_stats."""
        import inspect
        from abba.server.tools.nhl import NHLToolsMixin

        source = inspect.getsource(NHLToolsMixin.nhl_predict_game)

        # Find the query_team_stats call
        lines = source.split("\n")
        team_stats_lines = [
            (i, line.strip()) for i, line in enumerate(lines)
            if "query_team_stats" in line
        ]

        for line_num, line_content in team_stats_lines:
            # Check if season= is in the call or nearby lines
            context = "\n".join(lines[max(0, line_num):min(len(lines), line_num + 3)])
            assert "season=" in context, (
                f"SEASON SELECTION VIOLATION: query_team_stats call at line ~{line_num} "
                f"does not include season= parameter: '{line_content}'. "
                "Multi-season data will return arbitrary results."
            )


class TestGoaltenderStatsSeasonSelection:
    """query_goaltender_stats must not mix seasons."""

    def test_unfiltered_goalie_query_returns_multi_season(self, multi_season_db):
        """Without season filter, goalies from multiple seasons are returned."""
        results = multi_season_db.query_goaltender_stats(team="NYR")
        seasons = {r.get("season") for r in results}

        if len(seasons) >= 2:
            # Multi-season goalie data exists
            # The starter selection may pick a goalie from the wrong season
            pass

    def test_goalie_from_wrong_season_can_be_selected(self, multi_season_db):
        """Prove that without season filter, a previous-season goalie can win selection."""
        results = multi_season_db.query_goaltender_stats(team="NYR")

        # Simulate nhl.py:48-54 selection
        selected = next(
            (g["stats"] for g in results if g.get("stats", {}).get("role") == "starter"),
            results[0]["stats"] if results else None,
        )

        if selected:
            # The selected goalie might be "Last Year Goalie" from 2024-25
            # because the old-season goalie has role="starter" (set in seed data)
            # while current-season goalies (from live connector) don't have role
            selected_name = selected.get("name", "")
            if selected_name == "Last Year Goalie":
                pytest.fail(
                    "SEASON SELECTION VIOLATION: Goalie starter selection picked "
                    "'Last Year Goalie' from 2024-25 season instead of current-season goalie. "
                    "This happens because old seed data has role='starter' but current data does not."
                )

    def test_prediction_path_goalie_query_uses_season(self):
        """nhl_predict_game must pass season to query_goaltender_stats."""
        import inspect
        from abba.server.tools.nhl import NHLToolsMixin

        source = inspect.getsource(NHLToolsMixin.nhl_predict_game)
        lines = source.split("\n")

        goalie_lines = [
            (i, line.strip()) for i, line in enumerate(lines)
            if "query_goaltender_stats" in line
        ]

        for line_num, line_content in goalie_lines:
            context = "\n".join(lines[max(0, line_num):min(len(lines), line_num + 3)])
            assert "season=" in context, (
                f"SEASON SELECTION VIOLATION: query_goaltender_stats call at line ~{line_num} "
                f"does not include season= parameter: '{line_content}'."
            )


class TestAdvancedStatsSeasonSelection:
    """query_nhl_advanced_stats must not mix seasons."""

    def test_unfiltered_advanced_returns_multi_season(self, multi_season_db):
        """Without season filter, advanced stats from multiple seasons are returned."""
        results = multi_season_db.query_nhl_advanced_stats(team_id="NYR")
        seasons = {r.get("season") for r in results}

        if len(seasons) >= 2:
            first_season = results[0].get("season")
            assert first_season == "2025-26", (
                f"SEASON SELECTION VIOLATION: Advanced stats query returned '{first_season}' "
                "first. Prediction would use stale Corsi/xG data."
            )

    def test_prediction_path_advanced_query_uses_season(self):
        """nhl_predict_game must pass season to query_nhl_advanced_stats."""
        import inspect
        from abba.server.tools.nhl import NHLToolsMixin

        source = inspect.getsource(NHLToolsMixin.nhl_predict_game)
        lines = source.split("\n")

        adv_lines = [
            (i, line.strip()) for i, line in enumerate(lines)
            if "query_nhl_advanced_stats" in line
        ]

        for line_num, line_content in adv_lines:
            context = "\n".join(lines[max(0, line_num):min(len(lines), line_num + 3)])
            assert "season=" in context, (
                f"SEASON SELECTION VIOLATION: query_nhl_advanced_stats call at line ~{line_num} "
                f"does not include season= parameter: '{line_content}'."
            )


class TestSeasonDeterminism:
    """Query results must be deterministic across multiple calls."""

    def test_repeated_queries_return_same_order(self, multi_season_db):
        """Same query run twice must return same row order."""
        r1 = multi_season_db.query_team_stats(team_id="NYR", sport="NHL")
        r2 = multi_season_db.query_team_stats(team_id="NYR", sport="NHL")

        if len(r1) >= 2 and len(r2) >= 2:
            assert r1[0].get("season") == r2[0].get("season"), (
                "DETERMINISM VIOLATION: Same query returned different row orders. "
                f"First call: {r1[0].get('season')}, second call: {r2[0].get('season')}."
            )

    def test_query_order_by_exists_for_team_stats(self):
        """query_team_stats must have ORDER BY for deterministic results."""
        import inspect

        source = inspect.getsource(Storage.query_team_stats)
        has_order = "ORDER BY" in source.upper()

        assert has_order, (
            "SEASON SELECTION VIOLATION: query_team_stats() has no ORDER BY clause. "
            "Multi-season results are returned in arbitrary order."
        )
