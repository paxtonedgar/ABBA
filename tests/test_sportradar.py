"""Tests for SportsRadar NHL connector.

Unit tests use mocked data; integration tests hit the real API.
"""

import pytest

from abba.connectors.sportradar import (
    SportsRadarConnector,
    _SR_ALIAS_TO_ABBREV,
)
from abba.storage import Storage


@pytest.fixture
def connector():
    return SportsRadarConnector(api_key="test-key")


@pytest.fixture
def db():
    s = Storage(":memory:")
    yield s
    s.close()


class TestAliasMapping:
    """All 32 NHL teams must be mapped."""

    EXPECTED_ABBREVS = {
        "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
        "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD",
        "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
        "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
    }

    def test_all_32_teams(self):
        mapped = set(_SR_ALIAS_TO_ABBREV.values())
        missing = self.EXPECTED_ABBREVS - mapped
        assert not missing, f"Missing teams: {missing}"

    def test_divergent_aliases(self):
        """SR uses abbreviated forms that differ from ABBA's 3-letter codes."""
        assert _SR_ALIAS_TO_ABBREV["LA"] == "LAK"
        assert _SR_ALIAS_TO_ABBREV["NJ"] == "NJD"
        assert _SR_ALIAS_TO_ABBREV["SJ"] == "SJS"
        assert _SR_ALIAS_TO_ABBREV["TB"] == "TBL"


class TestConnectorInit:
    def test_no_api_key(self):
        c = SportsRadarConnector()
        result = c.refresh(Storage(":memory:"))
        assert result["status"] == "no_api_key"

    def test_custom_api_key(self):
        c = SportsRadarConnector(api_key="my-key")
        assert c.api_key == "my-key"


class TestAliasToAbbrev:
    def test_identity_aliases(self, connector):
        assert connector._alias_to_abbrev("BOS") == "BOS"
        assert connector._alias_to_abbrev("TOR") == "TOR"

    def test_divergent_aliases(self, connector):
        assert connector._alias_to_abbrev("LA") == "LAK"
        assert connector._alias_to_abbrev("TB") == "TBL"

    def test_unknown_alias_passthrough(self, connector):
        assert connector._alias_to_abbrev("UNKNOWN") == "UNKNOWN"
