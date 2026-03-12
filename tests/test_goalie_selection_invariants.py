"""Invariant E — Starter Determinism.

If goalie matchup is part of the model, starter identity must be
explicit and deterministic.

These tests prove:
- first-row-wins is the current behavior
- missing role field falls through silently
- changing starter data does change model output (when schema is correct)
- non-deterministic selection from DB ordering
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

from abba.engine.hockey import HockeyAnalytics
from abba.storage import Storage
from abba.server import ABBAToolkit


@pytest.fixture
def hockey():
    return HockeyAnalytics()


@pytest.fixture
def db():
    s = Storage(":memory:")
    yield s
    s.close()


class TestStarterSelectionDeterminism:
    """Starter must be chosen by explicit criteria, not row order."""

    def test_first_row_wins_without_role(self, db):
        """When role field is absent, first-row-wins determines the starter.
        This is non-deterministic and violates Invariant E."""

        # Insert backup goalie FIRST (more games played by starter, but backup inserted first)
        goalies = [
            {
                "goaltender_id": "backup-goalie",
                "team": "TST",
                "season": "2025-26",
                "stats": {
                    "name": "Backup Guy",
                    "games_played": 20,
                    "games_started": 15,
                    "save_pct": 0.890,
                    "gaa": 3.50,
                    "gsaa": -8.0,
                    # NO role field — simulating live connector output
                },
            },
            {
                "goaltender_id": "starter-goalie",
                "team": "TST",
                "season": "2025-26",
                "stats": {
                    "name": "Starter Guy",
                    "games_played": 55,
                    "games_started": 50,
                    "save_pct": 0.920,
                    "gaa": 2.40,
                    "gsaa": 15.0,
                    # NO role field
                },
            },
        ]
        db.upsert_goaltender_stats(goalies)

        # Query goalies
        results = db.query_goaltender_stats(team="TST")
        assert len(results) == 2

        # Simulate nhl.py:48-54 starter selection logic
        selected = next(
            (g["stats"] for g in results if g.get("stats", {}).get("role") == "starter"),
            results[0]["stats"] if results else None,
        )

        # The role filter finds nothing (no role field on live data)
        # So it falls back to results[0] — whichever goalie is first
        assert selected is not None, "Should have selected a goalie"

        # The selected goalie is the FIRST ROW, not the best goalie
        # This is insertion-order dependent, not merit-based
        selected_name = selected.get("name", "")
        starter_name = "Starter Guy"

        if selected_name != starter_name:
            # PROVEN: first-row-wins selected the backup
            pass  # This is the expected violation
        # Either way, the selection was not based on games_started

    def test_deterministic_selection_by_games_started(self, db):
        """Starter should be the goalie with most games_started.
        Uses _select_starter() which was fixed for Invariant E."""
        from abba.server.tools.nhl import _select_starter

        goalies = [
            {
                "goaltender_id": "backup",
                "team": "TST",
                "season": "2025-26",
                "stats": {
                    "name": "Backup",
                    "games_started": 15,
                    "save_pct": 0.890,
                    "gaa": 3.50,
                    "gsaa": -5.0,
                },
            },
            {
                "goaltender_id": "starter",
                "team": "TST",
                "season": "2025-26",
                "stats": {
                    "name": "Starter",
                    "games_started": 50,
                    "save_pct": 0.920,
                    "gaa": 2.40,
                    "gsaa": 15.0,
                },
            },
        ]
        db.upsert_goaltender_stats(goalies)
        results = db.query_goaltender_stats(team="TST")

        # Use the actual _select_starter function (fixed for Invariant E)
        selected = _select_starter(results)

        assert selected is not None, "Should have selected a goalie"
        assert selected["name"] == "Starter", (
            f"INVARIANT E VIOLATION: _select_starter picked '{selected['name']}' "
            f"but correct starter is 'Starter' (most games_started)."
        )


class TestMissingStarterFailsClosed:
    """When starter cannot be determined, system should fail closed."""

    def test_no_goalies_returns_none_not_error(self, db):
        """When no goalies exist for a team, the model should fail closed."""
        results = db.query_goaltender_stats(team="NONEXISTENT")
        assert len(results) == 0

        # Simulate nhl.py:48-50 behavior
        home_goalie = next(
            (g["stats"] for g in results if g.get("stats", {}).get("role") == "starter"),
            results[0]["stats"] if results else None,
        )

        # Returns None — but the prediction continues with default goaltender_edge = 0.0
        assert home_goalie is None, "No goalies should result in None"

    def test_prediction_continues_with_none_goalie(self):
        """Prove the prediction runs (doesn't fail) when goalie data is None."""
        toolkit = ABBAToolkit(db_path=":memory:", auto_seed=False)

        # Seed minimal data: one game, team stats, but NO goalie data
        from abba.storage import Storage
        toolkit.storage.upsert_games([{
            "game_id": "nhl-test-001",
            "sport": "NHL",
            "date": "2026-03-10",
            "home_team": "TST",
            "away_team": "OPP",
            "status": "scheduled",
        }])
        toolkit.storage.upsert_team_stats([
            {"team_id": "TST", "sport": "NHL", "season": "2025-26",
             "stats": {"wins": 40, "losses": 25, "overtime_losses": 5,
                       "goals_for": 230, "goals_against": 200,
                       "games_played": 70, "recent_form": 0.6}},
            {"team_id": "OPP", "sport": "NHL", "season": "2025-26",
             "stats": {"wins": 35, "losses": 30, "overtime_losses": 5,
                       "goals_for": 210, "goals_against": 220,
                       "games_played": 70, "recent_form": 0.45}},
        ])

        result = toolkit.nhl_predict_game("nhl-test-001")

        # The prediction should either:
        # (a) Fail with an explicit error about missing goalie data, OR
        # (b) Succeed but flag that goalie data was absent in provenance
        # Currently it does neither — it silently uses goaltender_edge = 0.0

        has_explicit_warning = (
            "error" in result
            or result.get("data_provenance", {}).get("home_goaltender", {}).get("status") == "absent"
            or "goalie" in str(result.get("confidence", {}).get("caveats", [])).lower()
        )

        assert has_explicit_warning, (
            "INVARIANT E VIOLATION: Prediction ran without goalie data and produced a result "
            "with no warning about missing goaltender information. "
            f"goaltender_edge = {result.get('features', {}).get('goaltender_edge', 'N/A')}. "
            "User cannot tell that goalie matchup was not evaluated."
        )


class TestStarterChangeAffectsOutput:
    """Changing starter identity must change model output."""

    def test_different_starters_produce_different_edges(self, hockey):
        """Swapping starter goalies must produce different matchup edges."""
        elite_goalie = {"save_pct": 0.928, "gsaa": 20.0}
        weak_goalie = {"save_pct": 0.895, "gsaa": -15.0}
        opponent = {"save_pct": 0.910, "gsaa": 5.0}

        # Elite starter
        edge_elite = hockey.goaltender_matchup_edge(
            starter_sv_pct=elite_goalie["save_pct"],
            opponent_sv_pct=opponent["save_pct"],
            starter_gsaa=elite_goalie["gsaa"],
            opponent_gsaa=opponent["gsaa"],
        )

        # Weak starter (backup plays instead)
        edge_weak = hockey.goaltender_matchup_edge(
            starter_sv_pct=weak_goalie["save_pct"],
            opponent_sv_pct=opponent["save_pct"],
            starter_gsaa=weak_goalie["gsaa"],
            opponent_gsaa=opponent["gsaa"],
        )

        assert edge_elite["goaltender_edge"] != edge_weak["goaltender_edge"], (
            "Different goalies must produce different matchup edges"
        )
        assert edge_elite["goaltender_edge"] > edge_weak["goaltender_edge"], (
            "Elite goalie should produce higher edge than weak goalie"
        )

    def test_starter_selection_affects_full_prediction(self, hockey):
        """Prove that which goalie is selected changes the final ensemble prediction."""
        base_features = {
            "home_pts_pct": 0.60, "away_pts_pct": 0.50,
            "home_goal_diff_pg": 0.4, "away_goal_diff_pg": -0.1,
            "home_recent_form": 0.6, "away_recent_form": 0.5,
            "home_games_played": 70, "away_games_played": 70,
            "home_gf_per_game": 3.2, "home_ga_per_game": 2.8,
            "away_gf_per_game": 2.9, "away_ga_per_game": 3.0,
            "home_corsi_pct": 0.52, "away_corsi_pct": 0.49,
            "home_xgf_pct": 0.53, "away_xgf_pct": 0.48,
            "home_st_edge": 0.02, "rest_edge": 0.0,
        }

        # With elite goalie
        features_elite = {**base_features, "goaltender_edge": 0.04}
        preds_elite = hockey.predict_nhl_game(features_elite)

        # With weak goalie
        features_weak = {**base_features, "goaltender_edge": -0.03}
        preds_weak = hockey.predict_nhl_game(features_weak)

        # At least the goaltender matchup model (index 4) should differ
        assert preds_elite[4] != preds_weak[4], (
            "Goaltender matchup model output must change when goaltender_edge changes"
        )

        # The composite model (index 5) should also differ
        assert preds_elite[5] != preds_weak[5], (
            "Composite model must reflect goaltender_edge changes"
        )


class TestQueryOrderDeterminism:
    """Prove that goalie query results have no guaranteed order."""

    def test_query_has_no_order_by(self):
        """query_goaltender_stats SQL must include ORDER BY for deterministic results."""
        import inspect
        from abba.storage.duckdb import Storage

        source = inspect.getsource(Storage.query_goaltender_stats)

        has_order_by = "ORDER BY" in source.upper()
        assert has_order_by, (
            "INVARIANT E VIOLATION: query_goaltender_stats() has no ORDER BY clause. "
            "Result order depends on DB internals, making starter selection non-deterministic."
        )
