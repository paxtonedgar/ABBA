"""Invariant D — Snapshot Coherence.

Every prediction must carry coherent as_of, season, source, and
default/missing metadata for every dataset used.

These tests prove the system currently produces predictions with
no provenance, no freshness tracking, and silent cross-season mixing.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

from abba.server import ABBAToolkit
from abba.storage import Storage


@pytest.fixture
def toolkit():
    return ABBAToolkit(db_path=":memory:", auto_seed=True)


@pytest.fixture
def db():
    s = Storage(":memory:")
    yield s
    s.close()


class TestPredictionProvenance:
    """Every prediction output must carry provenance metadata."""

    def test_prediction_has_data_provenance(self, toolkit):
        """Prediction output must include a data_provenance field."""
        games = toolkit.storage.query_games(sport="NHL", status="scheduled")
        if not games:
            pytest.skip("No scheduled NHL games")

        result = toolkit.nhl_predict_game(games[0]["game_id"])
        if "error" in result:
            pytest.skip(f"Prediction error: {result['error']}")

        assert "data_provenance" in result, (
            "INVARIANT D VIOLATION: Prediction output has no 'data_provenance' field. "
            "Cannot determine which data sources were used, their freshness, or their season."
        )

    def test_provenance_has_required_sources(self, toolkit):
        """Provenance must document at least team_stats and goaltender_stats."""
        games = toolkit.storage.query_games(sport="NHL", status="scheduled")
        if not games:
            pytest.skip("No scheduled NHL games")

        result = toolkit.nhl_predict_game(games[0]["game_id"])
        if "error" in result:
            pytest.skip(f"Prediction error: {result['error']}")

        provenance = result.get("data_provenance", {})
        required = ["home_team_stats", "away_team_stats", "home_goaltender", "away_goaltender"]

        for source in required:
            assert source in provenance, (
                f"INVARIANT D VIOLATION: Provenance missing '{source}'. "
                "Cannot determine if this data source was present, absent, or defaulted."
            )

    def test_provenance_has_as_of_per_source(self, toolkit):
        """Each provenance entry must include an as_of timestamp."""
        games = toolkit.storage.query_games(sport="NHL", status="scheduled")
        if not games:
            pytest.skip("No scheduled NHL games")

        result = toolkit.nhl_predict_game(games[0]["game_id"])
        if "error" in result:
            pytest.skip(f"Prediction error: {result['error']}")

        provenance = result.get("data_provenance", {})
        for source_name, source_meta in provenance.items():
            assert "as_of" in source_meta, (
                f"INVARIANT D VIOLATION: Provenance for '{source_name}' has no 'as_of' timestamp. "
                "Cannot determine data freshness."
            )

    def test_provenance_has_season_per_source(self, toolkit):
        """Each provenance entry must declare which season the data is from."""
        games = toolkit.storage.query_games(sport="NHL", status="scheduled")
        if not games:
            pytest.skip("No scheduled NHL games")

        result = toolkit.nhl_predict_game(games[0]["game_id"])
        if "error" in result:
            pytest.skip(f"Prediction error: {result['error']}")

        provenance = result.get("data_provenance", {})
        for source_name, source_meta in provenance.items():
            assert "season" in source_meta, (
                f"INVARIANT D VIOLATION: Provenance for '{source_name}' has no 'season' field. "
                "Cannot detect cross-season data mixing."
            )

    def test_provenance_declares_defaults(self, toolkit):
        """Provenance must flag when values were defaulted due to missing data."""
        games = toolkit.storage.query_games(sport="NHL", status="scheduled")
        if not games:
            pytest.skip("No scheduled NHL games")

        result = toolkit.nhl_predict_game(games[0]["game_id"])
        if "error" in result:
            pytest.skip(f"Prediction error: {result['error']}")

        provenance = result.get("data_provenance", {})
        for source_name, source_meta in provenance.items():
            assert "status" in source_meta, (
                f"INVARIANT D VIOLATION: Provenance for '{source_name}' has no 'status' field. "
                "Must be one of: 'present', 'absent', 'defaulted', 'stale'."
            )


class TestDefaultDetection:
    """Prove the system cannot distinguish real data from defaults."""

    def test_features_dont_flag_defaults(self, toolkit):
        """Feature dict should flag which values are defaults vs measured."""
        games = toolkit.storage.query_games(sport="NHL", status="scheduled")
        if not games:
            pytest.skip("No scheduled NHL games")

        result = toolkit.nhl_predict_game(games[0]["game_id"])
        if "error" in result:
            pytest.skip(f"Prediction error: {result['error']}")

        features = result.get("features", {})

        # Check for known default values that indicate missing data
        # goaltender_edge = 0.0 when goalie data is missing
        # rest_edge = 0.0 when rest data is never passed
        # home_corsi_pct = 0.0 when advanced stats are absent

        # The system should flag these as defaults, not present them as measurements
        defaulted_features = []
        for key in ("goaltender_edge", "rest_edge", "home_corsi_pct", "away_corsi_pct",
                     "home_xgf_pct", "away_xgf_pct"):
            val = features.get(key, None)
            if val is not None and val == 0.0:
                defaulted_features.append(key)

        if defaulted_features:
            # These 0.0 values are indistinguishable from measured zeros
            defaults_meta = result.get("defaulted_features")
            assert defaults_meta is not None, (
                f"INVARIANT D VIOLATION: Features {defaulted_features} are 0.0 (likely defaults) "
                "but no metadata distinguishes them from measured zeros. "
                "A real 0.0 goaltender_edge (equal goalies) is indistinguishable from "
                "missing goalie data."
            )


class TestCrossSeasonMixing:
    """Prove the system can silently mix data from different seasons."""

    def test_queries_lack_season_filter(self):
        """nhl_predict_game queries must include season parameter."""
        import inspect
        from abba.server.tools.nhl import NHLToolsMixin

        source = inspect.getsource(NHLToolsMixin.nhl_predict_game)

        # Count query calls that lack season parameter
        query_calls = [
            ("query_team_stats", "nhl.py:35"),
            ("query_nhl_advanced_stats", "nhl.py:40-41"),
            ("query_goaltender_stats", "nhl.py:46-47"),
        ]

        unfiltered = []
        for method_name, location in query_calls:
            # Find the call in source
            if method_name in source:
                # Check if season= is passed in that call
                # Simple heuristic: find the line with the method call
                lines = source.split("\n")
                for i, line in enumerate(lines):
                    if method_name in line:
                        # Check this line and next few lines for season=
                        context = "\n".join(lines[max(0, i):min(len(lines), i + 3)])
                        if "season=" not in context:
                            unfiltered.append(f"{method_name} at {location}")

        assert not unfiltered, (
            f"INVARIANT D VIOLATION: These prediction-path queries lack season filter: "
            f"{unfiltered}. With multi-season data, they may return stale season results."
        )

    def test_multi_season_data_is_not_rejected(self, db):
        """Prove that mixed-season data is silently accepted."""
        from abba.connectors.seed import seed_sample_data

        seed_sample_data(db)

        # Insert team stats for a different season
        old_season_stats = [{
            "team_id": "NYR",
            "sport": "NHL",
            "season": "2024-25",  # Previous season
            "stats": {
                "wins": 50, "losses": 22, "overtime_losses": 10,
                "goals_for": 280, "goals_against": 200,
                "goal_differential": 80,
                "games_played": 82,
            },
            "source": "test",
        }]
        db.upsert_team_stats(old_season_stats)

        # Query without season filter — both seasons returned
        results = db.query_team_stats(team_id="NYR", sport="NHL")
        seasons = {r.get("season") for r in results}

        if len(seasons) > 1:
            # Multiple seasons exist — the system must handle this
            # Currently: [0] picks an arbitrary one
            first_result_season = results[0].get("season")
            assert first_result_season == "2025-26", (
                f"INVARIANT D VIOLATION: Multi-season query returned '{first_result_season}' first. "
                "Without ORDER BY or season filter, the prediction may use stale season data."
            )


class TestRefreshTimestampPersistence:
    """Verify that freshness tracking persists across restarts."""

    def test_last_refresh_ts_is_none_on_cold_start(self):
        """_last_refresh_ts is None on cold start (no prior refresh recorded)."""
        toolkit = ABBAToolkit(db_path=":memory:", auto_seed=True)
        # auto_seed does not call refresh_data, so no timestamp is recorded
        assert toolkit._last_refresh_ts is None

    def test_freshness_table_exists(self, db):
        """data_freshness table exists for per-table refresh tracking."""
        tables = db.list_tables()
        table_names = {t["table"] for t in tables}
        assert "data_freshness" in table_names

    def test_record_and_retrieve_refresh(self, db):
        """record_refresh() persists and get_last_refresh() retrieves it."""
        db.record_refresh("team_stats", source="nhl", row_count=32)
        ts = db.get_last_refresh("team_stats")
        assert ts is not None
        assert isinstance(ts, float)
        assert ts > 0
