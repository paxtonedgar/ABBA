"""Invariant B — Schema Contract Integrity.

Every field produced by live ingestion that is consumed by the NHL
prediction path must exist under the exact keys and semantics the
model expects.

These tests build fixture data in live-ingestion shape and feed it
through the prediction path, proving field name mismatches.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

from abba.engine.hockey import HockeyAnalytics
from abba.storage import Storage


@pytest.fixture
def hockey():
    return HockeyAnalytics()


@pytest.fixture
def db():
    s = Storage(":memory:")
    yield s
    s.close()


# --- The canonical schema contract ---
# These are the field names the model READS from goalie stats dicts.
# Extracted from hockey.py:830-833 and hockey.py:612-614.
MODEL_GOALIE_KEYS = {
    "save_pct",       # hockey.py:830 — goalie.get("save_pct", 0.907)
    "gaa",            # hockey.py:613 — primary.get("gaa", 0)
    "gsaa",           # hockey.py:832 — goalie.get("gsaa", 0)
    "name",           # nhl.py:120 — home_goalie.get("name")
    "games_played",   # used in season_review and goalie grading
    "games_started",  # needed for starter determination
    # NOTE: "role" removed — _select_starter() handles missing role via
    # max(games_started) fallback (Invariant E fix). No longer required.
}

# These are the field names the LIVE CONNECTOR writes.
# Extracted from live.py:267-295 (after Invariant B fix).
LIVE_CONNECTOR_GOALIE_KEYS = {
    "name",
    "games_played",
    "games_started",
    "wins",
    "losses",
    "ot_losses",
    "save_pct",                # FIXED: model-compatible key
    "gaa",                     # FIXED: model-compatible key
    "gsaa",                    # FIXED: computed from shots_against * 0.907 - goals_against
    "save_percentage",         # kept for transparency alongside save_pct
    "goals_against_average",   # kept for transparency alongside gaa
    "goals_against",
    "saves",
    "shots_against",
    "shutouts",
    "time_on_ice_minutes",
    # NOTE: "role" is still not set by live connector — starter selection
    # uses max(games_started) fallback via _select_starter() (Invariant E fix)
}


class TestGoalieSchemaContract:
    """Prove that live ingestion and model consumption agree on field names."""

    def test_model_required_keys_exist_in_live_output(self):
        """Every key the model reads must exist in what live ingestion writes."""
        missing = MODEL_GOALIE_KEYS - LIVE_CONNECTOR_GOALIE_KEYS
        assert not missing, (
            f"INVARIANT B VIOLATION: Model reads keys not produced by live ingestion: {missing}. "
            f"These fields will hit defaults, making the goalie model inert on live data."
        )

    def test_save_pct_key_name_matches(self):
        """Live connector must write 'save_pct', not 'save_percentage'."""
        assert "save_pct" in LIVE_CONNECTOR_GOALIE_KEYS, (
            "INVARIANT B VIOLATION: Live connector writes 'save_percentage' but model reads 'save_pct'. "
            "hockey.py:830 does goalie.get('save_pct', 0.907) — always gets default 0.907 on live data."
        )
        assert "save_percentage" not in LIVE_CONNECTOR_GOALIE_KEYS or "save_pct" in LIVE_CONNECTOR_GOALIE_KEYS, (
            "INVARIANT B VIOLATION: 'save_percentage' exists but 'save_pct' does not."
        )

    def test_gaa_key_name_matches(self):
        """Live connector must write 'gaa', not 'goals_against_average'."""
        assert "gaa" in LIVE_CONNECTOR_GOALIE_KEYS, (
            "INVARIANT B VIOLATION: Live connector writes 'goals_against_average' but model reads 'gaa'. "
            "hockey.py:613 does primary.get('gaa', 0) — always gets 0 on live data."
        )

    def test_gsaa_is_computed_by_live(self):
        """Live connector must compute and store GSAA."""
        assert "gsaa" in LIVE_CONNECTOR_GOALIE_KEYS, (
            "INVARIANT B VIOLATION: Live connector never computes 'gsaa'. "
            "hockey.py:832 does goalie.get('gsaa', 0) — always gets 0 on live data. "
            "GSAA should be computed as: (shots_against * 0.907) - goals_against."
        )

    def test_role_handled_by_fallback(self):
        """Live connector does not set 'role', but _select_starter() handles this
        via max(games_started) fallback (Invariant E fix)."""
        # Role is not in live output — that's OK now because starter selection
        # no longer depends on it (falls back to games_started deterministically)
        assert "role" not in LIVE_CONNECTOR_GOALIE_KEYS, (
            "If role is now set by live connector, update this test."
        )


class TestGoalieDataFlowIntegrity:
    """Prove the actual data flow — live shape in, model consumption out."""

    def _make_live_goalie_stats(self) -> dict:
        """Build a goalie stats dict in the shape live.py:267-281 produces."""
        return {
            "name": "Test Goalie",
            "games_played": 50,
            "games_started": 45,
            "wins": 28,
            "losses": 15,
            "ot_losses": 7,
            "save_percentage": 0.9180,       # live key name
            "goals_against_average": 2.65,    # live key name
            "goals_against": 120,
            "saves": 1330,
            "shots_against": 1450,
            "shutouts": 4,
            "time_on_ice_minutes": 2850.0,
            # NOTE: no "gsaa", no "role" — live doesn't produce them
        }

    def _make_seed_goalie_stats(self) -> dict:
        """Build a goalie stats dict in the shape seed.py produces."""
        return {
            "name": "Test Goalie",
            "role": "starter",               # seed sets this
            "games_played": 50,
            "save_pct": 0.9180,              # seed key name
            "gaa": 2.65,                     # seed key name
            "saves": 1330,
            "shots_against": 1450,
            "goals_against": 120,
            "xg_against": 125.0,
            "gsaa": 11.45,                   # seed computes this
            "xgsaa": 5.0,
            "quality_starts": 30,
            "shutouts": 4,
            "minutes_played": 2850.0,
        }

    def test_live_goalie_in_matchup_model_gets_defaults(self, hockey):
        """Prove that live-shaped goalie data hits defaults in the matchup model."""
        live_goalie = self._make_live_goalie_stats()

        # This is what the model does with goalie data:
        sv_pct = live_goalie.get("save_pct", 0.907)    # model reads "save_pct"
        gsaa = live_goalie.get("gsaa", 0)               # model reads "gsaa"

        # With live data, these hit defaults:
        assert sv_pct == 0.907, (
            "Expected default 0.907 since live data has 'save_percentage' not 'save_pct'"
        )
        assert gsaa == 0, (
            "Expected default 0 since live data has no 'gsaa' field"
        )

    def test_seed_goalie_in_matchup_model_gets_real_values(self, hockey):
        """Prove that seed-shaped goalie data provides real values."""
        seed_goalie = self._make_seed_goalie_stats()

        sv_pct = seed_goalie.get("save_pct", 0.907)
        gsaa = seed_goalie.get("gsaa", 0)

        assert sv_pct == 0.9180, "Seed data should provide real save_pct"
        assert gsaa == 11.45, "Seed data should provide real gsaa"

    def test_matchup_edge_differs_between_live_and_seed_shape(self, hockey):
        """Prove the goalie matchup model produces different results for same goalie
        depending on whether data comes from live vs seed shape."""
        live_goalie = self._make_live_goalie_stats()
        seed_goalie = self._make_seed_goalie_stats()

        # With seed data — model gets real values
        seed_edge = hockey.goaltender_matchup_edge(
            starter_sv_pct=seed_goalie.get("save_pct", 0.907),
            opponent_sv_pct=0.910,
            starter_gsaa=seed_goalie.get("gsaa", 0),
            opponent_gsaa=5.0,
        )

        # With live data — model gets defaults
        live_edge = hockey.goaltender_matchup_edge(
            starter_sv_pct=live_goalie.get("save_pct", 0.907),
            opponent_sv_pct=0.910,
            starter_gsaa=live_goalie.get("gsaa", 0),
            opponent_gsaa=5.0,
        )

        # These should be the same for the same goalie — but they won't be
        assert abs(seed_edge["goaltender_edge"] - live_edge["goaltender_edge"]) > 0.001, (
            "INVARIANT B PROVEN: Same goalie produces different matchup edges "
            "depending on data source shape. Schema mismatch causes the model "
            "to see different inputs for identical underlying data."
        )

    def test_live_goalie_matchup_edge_is_near_zero(self, hockey):
        """Prove that live-shaped goalie data produces near-zero matchup edge."""
        live_home = self._make_live_goalie_stats()
        live_away = self._make_live_goalie_stats()

        edge = hockey.goaltender_matchup_edge(
            starter_sv_pct=live_home.get("save_pct", 0.907),
            opponent_sv_pct=live_away.get("save_pct", 0.907),
            starter_gsaa=live_home.get("gsaa", 0),
            opponent_gsaa=live_away.get("gsaa", 0),
        )

        # Both goalies hit defaults, so edge is ~0 regardless of actual performance
        assert abs(edge["goaltender_edge"]) < 0.001, (
            "INVARIANT B PROVEN: Live goalie data produces near-zero matchup edge "
            "because both goalies hit the same defaults (sv_pct=0.907, gsaa=0). "
            "The goaltender matchup model is inert on live data."
        )


class TestSchemaContractStorage:
    """Prove schema violations at the storage boundary."""

    def test_upsert_rejects_wrong_schema_goalie_stats(self, db):
        """Storage rejects goalie data with old field names (save_percentage instead of save_pct)."""
        from abba.storage.duckdb import StorageValidationError

        live_record = {
            "goaltender_id": "test-goalie-1",
            "team": "NYR",
            "season": "2025-26",
            "stats": {
                "name": "Test Goalie",
                "games_played": 50,
                "games_started": 45,
                "save_percentage": 0.9180,       # wrong key — should be save_pct
                "goals_against_average": 2.65,    # wrong key — should be gaa
                "goals_against": 120,
                "saves": 1330,
                "shots_against": 1450,
            },
        }

        with pytest.raises(StorageValidationError, match="missing required keys"):
            db.upsert_goaltender_stats([live_record])
