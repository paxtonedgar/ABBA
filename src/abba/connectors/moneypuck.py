"""MoneyPuck advanced stats connector.

Fetches team-level 5v5 analytics (Corsi, Fenwick, xG, PDO) from
MoneyPuck's free public CSV files. No auth required.

Source: moneypuck.com -- CC-BY licensed, updated daily during season.
"""

from __future__ import annotations

import csv
import io
import urllib.request
import urllib.error
from datetime import datetime
from typing import Any

from ..storage import Storage


# MoneyPuck team name → NHL 3-letter abbreviation.
# MoneyPuck uses inconsistent naming (sometimes abbreviations, sometimes short names).
_MONEYPUCK_TO_ABBREV: dict[str, str] = {
    "ANA": "ANA", "ARI": "ARI", "BOS": "BOS", "BUF": "BUF",
    "CGY": "CGY", "CAR": "CAR", "CHI": "CHI", "COL": "COL",
    "CBJ": "CBJ", "DAL": "DAL", "DET": "DET", "EDM": "EDM",
    "FLA": "FLA", "L.A": "LAK", "MIN": "MIN", "MTL": "MTL",
    "NSH": "NSH", "N.J": "NJD", "NYI": "NYI", "NYR": "NYR",
    "OTT": "OTT", "PHI": "PHI", "PIT": "PIT", "S.J": "SJS",
    "SEA": "SEA", "STL": "STL", "T.B": "TBL", "TOR": "TOR",
    "UTA": "UTA", "VAN": "VAN", "VGK": "VGK", "WSH": "WSH",
    "WPG": "WPG",
    # Legacy / alternate forms seen in MoneyPuck data
    "L.A.": "LAK", "N.J.": "NJD", "S.J.": "SJS", "T.B.": "TBL",
    "LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL",
}

# CSV URL template. {year} is the starting year of the season (e.g. 2025 for 2025-26).
_CSV_URL = "https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/teams.csv"


class MoneyPuckConnector:
    """Fetches team-level 5v5 advanced stats from MoneyPuck CSV files.

    Free, no auth. Provides Corsi%, Fenwick%, xGF%, shooting%, PDO --
    the metrics the NHL API doesn't expose but predictions need.
    """

    def _fetch_csv(self, year: int) -> list[dict[str, str]] | None:
        """Fetch and parse the MoneyPuck team summary CSV for a season.

        Returns list of row dicts from csv.DictReader, or None on failure.
        """
        url = _CSV_URL.format(year=year)
        self._last_error: str | None = None
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ABBA/2.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                text = resp.read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(text))
            return list(reader)
        except urllib.error.HTTPError as e:
            self._last_error = f"HTTP {e.code}: {e.reason} ({url})"
            return None
        except urllib.error.URLError as e:
            self._last_error = f"URL error: {e.reason} ({url})"
            return None
        except (TimeoutError, UnicodeDecodeError, csv.Error) as e:
            self._last_error = f"{type(e).__name__}: {e} ({url})"
            return None

    @staticmethod
    def _safe_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
        """Safely extract a float from a CSV row dict."""
        try:
            return float(row.get(key, default))
        except (ValueError, TypeError):
            return default

    def parse_team_stats(self, rows: list[dict[str, str]], season: str) -> list[dict[str, Any]]:
        """Parse MoneyPuck CSV rows into advanced stats records.

        Filters to situation == '5on5' (even-strength) and extracts
        per-team analytics.
        """
        records: list[dict[str, Any]] = []

        for row in rows:
            # Only 5v5 situation for team-level analytics
            if row.get("situation") != "5on5":
                continue

            team_raw = row.get("team", "")
            abbrev = _MONEYPUCK_TO_ABBREV.get(team_raw)
            if not abbrev:
                continue

            sf = self._safe_float
            # Shot attempts
            cf = sf(row, "flurryScoreVenueAdjustedxGoalsFor")  # score-venue adjusted xGF
            ca = sf(row, "flurryScoreVenueAdjustedxGoalsAgainst")

            # Raw Corsi (shot attempts for/against)
            corsi_for = sf(row, "corsiPercentage", 50.0)
            # MoneyPuck provides corsiPercentage directly as 0-100
            # But some seasons use different column names -- fall back to computing
            shots_for = sf(row, "shotsOnGoalFor")
            shots_against = sf(row, "shotsOnGoalAgainst")

            # Fenwick
            fenwick_pct = sf(row, "fenwickPercentage", 50.0)

            # xG
            xgf = sf(row, "xGoalsFor")
            xga = sf(row, "xGoalsAgainst")
            xgf_total = xgf + xga
            xgf_pct = (xgf / xgf_total * 100) if xgf_total > 0 else 50.0

            # Shooting and save percentages
            goals_for = sf(row, "goalsFor")
            goals_against = sf(row, "goalsAgainst")
            sh_pct = (goals_for / shots_for * 100) if shots_for > 0 else 8.0
            sv_pct = (1 - goals_against / shots_against) * 100 if shots_against > 0 else 91.0
            pdo = sh_pct + sv_pct

            # Ice time
            toi = sf(row, "iceTime", 1.0)  # total 5v5 ice time in seconds
            toi_minutes = toi / 60 if toi > 0 else 1.0

            # Shots per 60
            sf_per60 = (shots_for / toi_minutes * 60) if toi_minutes > 0 else 30.0
            sa_per60 = (shots_against / toi_minutes * 60) if toi_minutes > 0 else 30.0

            records.append({
                "team_id": abbrev,
                "season": season,
                "stats": {
                    "corsi_pct": round(corsi_for, 2),
                    "fenwick_pct": round(fenwick_pct, 2),
                    "xgf_pct": round(xgf_pct, 2),
                    "xgf": round(xgf, 2),
                    "xga": round(xga, 2),
                    "shots_for_per60": round(sf_per60, 2),
                    "shots_against_per60": round(sa_per60, 2),
                    "shooting_pct": round(sh_pct, 2),
                    "save_pct_5v5": round(sv_pct, 2),
                    "pdo": round(pdo, 2),
                    "goals_for_5v5": int(goals_for),
                    "goals_against_5v5": int(goals_against),
                    "toi_minutes_5v5": round(toi_minutes, 1),
                    "source": "moneypuck",
                },
            })

        return records

    def refresh(
        self,
        storage: Storage,
        season: str = "2025-26",
        team: str | None = None,
    ) -> dict[str, Any]:
        """Fetch MoneyPuck advanced stats and write to storage.

        Args:
            storage: DuckDB storage instance.
            season: Season string like "2025-26".
            team: Optional team abbreviation to filter to.

        Returns:
            Status dict with teams_updated count.
        """
        # Extract start year from season string (e.g. "2025-26" → 2025)
        try:
            year = int(season.split("-")[0])
        except (ValueError, IndexError):
            return {"status": "error", "error": f"Invalid season format: {season}. Expected YYYY-YY."}

        rows = self._fetch_csv(year)
        if rows is None:
            return {
                "status": "no_data",
                "error": self._last_error or "Could not fetch MoneyPuck CSV",
                "season": season,
                "fetched_at": datetime.now().isoformat(),
            }

        records = self.parse_team_stats(rows, season)

        if team:
            records = [r for r in records if r["team_id"] == team.upper()]

        if not records:
            return {
                "status": "no_data",
                "error": f"No 5on5 team data found for season {season}",
                "season": season,
                "fetched_at": datetime.now().isoformat(),
            }

        stored = storage.upsert_nhl_advanced_stats(records)

        return {
            "status": "ok",
            "teams_updated": stored,
            "season": season,
            "source": "moneypuck",
            "fetched_at": datetime.now().isoformat(),
        }
