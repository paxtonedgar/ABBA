"""Tests for MoneyPuck advanced stats connector."""

import pytest

from abba.connectors.moneypuck import MoneyPuckConnector, _MONEYPUCK_TO_ABBREV
from abba.storage import Storage


# --- Inline CSV for unit tests (mimics MoneyPuck column structure) ---

SAMPLE_CSV_ROWS = [
    {
        "team": "BOS",
        "situation": "5on5",
        "corsiPercentage": "54.2",
        "fenwickPercentage": "53.8",
        "xGoalsFor": "85.3",
        "xGoalsAgainst": "72.1",
        "shotsOnGoalFor": "1050",
        "shotsOnGoalAgainst": "920",
        "goalsFor": "95",
        "goalsAgainst": "78",
        "iceTime": "120000",  # seconds
        "flurryScoreVenueAdjustedxGoalsFor": "54.2",
        "flurryScoreVenueAdjustedxGoalsAgainst": "45.8",
    },
    {
        "team": "TOR",
        "situation": "5on5",
        "corsiPercentage": "48.5",
        "fenwickPercentage": "49.1",
        "xGoalsFor": "70.2",
        "xGoalsAgainst": "80.5",
        "shotsOnGoalFor": "900",
        "shotsOnGoalAgainst": "1000",
        "goalsFor": "72",
        "goalsAgainst": "88",
        "iceTime": "118000",
        "flurryScoreVenueAdjustedxGoalsFor": "48.5",
        "flurryScoreVenueAdjustedxGoalsAgainst": "51.5",
    },
    # Non-5v5 row — should be filtered out
    {
        "team": "BOS",
        "situation": "all",
        "corsiPercentage": "55.0",
        "fenwickPercentage": "54.0",
        "xGoalsFor": "100.0",
        "xGoalsAgainst": "80.0",
        "shotsOnGoalFor": "1200",
        "shotsOnGoalAgainst": "1000",
        "goalsFor": "110",
        "goalsAgainst": "85",
        "iceTime": "150000",
        "flurryScoreVenueAdjustedxGoalsFor": "55.0",
        "flurryScoreVenueAdjustedxGoalsAgainst": "45.0",
    },
    # Dotted abbreviation team
    {
        "team": "L.A",
        "situation": "5on5",
        "corsiPercentage": "51.0",
        "fenwickPercentage": "50.5",
        "xGoalsFor": "75.0",
        "xGoalsAgainst": "74.0",
        "shotsOnGoalFor": "950",
        "shotsOnGoalAgainst": "940",
        "goalsFor": "80",
        "goalsAgainst": "79",
        "iceTime": "119000",
        "flurryScoreVenueAdjustedxGoalsFor": "51.0",
        "flurryScoreVenueAdjustedxGoalsAgainst": "49.0",
    },
]


@pytest.fixture
def connector():
    return MoneyPuckConnector()


@pytest.fixture
def db():
    s = Storage(":memory:")
    yield s
    s.close()


class TestParseTeamStats:
    """Test CSV parsing logic."""

    def test_filters_to_5on5_only(self, connector):
        records = connector.parse_team_stats(SAMPLE_CSV_ROWS, "2025-26")
        teams = [r["team_id"] for r in records]
        # BOS appears twice in CSV (5on5 + all), but should only appear once
        assert teams.count("BOS") == 1

    def test_output_shape(self, connector):
        records = connector.parse_team_stats(SAMPLE_CSV_ROWS, "2025-26")
        assert len(records) == 3  # BOS, TOR, LAK (5on5 only)
        for r in records:
            assert "team_id" in r
            assert "season" in r
            assert "stats" in r
            stats = r["stats"]
            assert "corsi_pct" in stats
            assert "fenwick_pct" in stats
            assert "xgf_pct" in stats
            assert "pdo" in stats
            assert "shooting_pct" in stats
            assert "source" in stats
            assert stats["source"] == "moneypuck"

    def test_corsi_values(self, connector):
        records = connector.parse_team_stats(SAMPLE_CSV_ROWS, "2025-26")
        bos = next(r for r in records if r["team_id"] == "BOS")
        assert bos["stats"]["corsi_pct"] == 54.2

    def test_xgf_pct_computed(self, connector):
        records = connector.parse_team_stats(SAMPLE_CSV_ROWS, "2025-26")
        bos = next(r for r in records if r["team_id"] == "BOS")
        # xGF% = 85.3 / (85.3 + 72.1) * 100 ≈ 54.17
        assert 54.0 <= bos["stats"]["xgf_pct"] <= 54.3

    def test_pdo_is_shooting_plus_save(self, connector):
        records = connector.parse_team_stats(SAMPLE_CSV_ROWS, "2025-26")
        bos = next(r for r in records if r["team_id"] == "BOS")
        expected_pdo = bos["stats"]["shooting_pct"] + bos["stats"]["save_pct_5v5"]
        assert abs(bos["stats"]["pdo"] - expected_pdo) < 0.01

    def test_dotted_team_names(self, connector):
        records = connector.parse_team_stats(SAMPLE_CSV_ROWS, "2025-26")
        teams = {r["team_id"] for r in records}
        assert "LAK" in teams  # "L.A" → "LAK"

    def test_season_passed_through(self, connector):
        records = connector.parse_team_stats(SAMPLE_CSV_ROWS, "2024-25")
        for r in records:
            assert r["season"] == "2024-25"


class TestTeamNameMapping:
    """Verify all 32 NHL teams are covered."""

    EXPECTED_ABBREVS = {
        "ANA", "ARI", "BOS", "BUF", "CGY", "CAR", "CHI", "COL",
        "CBJ", "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL",
        "NSH", "NJD", "NYI", "NYR", "OTT", "PHI", "PIT", "SJS",
        "SEA", "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WSH", "WPG",
    }

    def test_all_32_teams_mapped(self):
        mapped_abbrevs = set(_MONEYPUCK_TO_ABBREV.values())
        missing = self.EXPECTED_ABBREVS - mapped_abbrevs
        assert not missing, f"Missing teams in mapping: {missing}"

    def test_dotted_variants_resolve(self):
        assert _MONEYPUCK_TO_ABBREV["L.A"] == "LAK"
        assert _MONEYPUCK_TO_ABBREV["N.J"] == "NJD"
        assert _MONEYPUCK_TO_ABBREV["S.J"] == "SJS"
        assert _MONEYPUCK_TO_ABBREV["T.B"] == "TBL"

    def test_plain_variants_resolve(self):
        assert _MONEYPUCK_TO_ABBREV["LA"] == "LAK"
        assert _MONEYPUCK_TO_ABBREV["NJ"] == "NJD"


class TestStorageRoundTrip:
    """Integration: connector → storage → query round-trip."""

    def test_parse_upsert_query(self, connector, db):
        records = connector.parse_team_stats(SAMPLE_CSV_ROWS, "2025-26")
        stored = db.upsert_nhl_advanced_stats(records)
        assert stored == 3

        # Query back
        bos = db.query_nhl_advanced_stats(team_id="BOS", season="2025-26")
        assert len(bos) == 1
        assert bos[0]["stats"]["corsi_pct"] == 54.2
        assert bos[0]["stats"]["source"] == "moneypuck"

    def test_query_all_teams(self, connector, db):
        records = connector.parse_team_stats(SAMPLE_CSV_ROWS, "2025-26")
        db.upsert_nhl_advanced_stats(records)
        all_stats = db.query_nhl_advanced_stats(season="2025-26")
        assert len(all_stats) == 3

    def test_upsert_replaces(self, connector, db):
        """Second upsert replaces first."""
        records = connector.parse_team_stats(SAMPLE_CSV_ROWS, "2025-26")
        db.upsert_nhl_advanced_stats(records)

        # Mutate and re-upsert
        records[0]["stats"]["corsi_pct"] = 99.9
        db.upsert_nhl_advanced_stats([records[0]])

        bos = db.query_nhl_advanced_stats(team_id="BOS", season="2025-26")
        assert bos[0]["stats"]["corsi_pct"] == 99.9


class TestRefreshGracefulFailure:
    """Test that refresh returns clean status dicts on failure."""

    def test_invalid_season_format(self, connector, db):
        result = connector.refresh(db, season="invalid")
        assert result["status"] == "error"
        assert "Invalid season" in result["error"]

    def test_empty_rows_returns_no_data(self, connector, db):
        # Simulate: CSV fetched but no 5on5 rows
        records = connector.parse_team_stats([], "2025-26")
        assert records == []
