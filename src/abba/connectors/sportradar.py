"""SportsRadar NHL connector.

Fetches standings, schedule, rosters, goaltender stats, advanced analytics,
and injuries from the SportsRadar NHL v7 API.

Replaces the free NHL Stats API and MoneyPuck connectors with a single,
comprehensive, paid data source.

API docs: https://developer.sportradar.com/ice-hockey/reference/nhl-overview
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime, date
from typing import Any

from ..storage import Storage

BASE_URL = "https://api.sportradar.com/nhl/{access}/v7/en"

# SportsRadar alias → ABBA 3-letter abbreviation.
# SR uses mostly standard NHL abbreviations but a few differ.
_SR_ALIAS_TO_ABBREV: dict[str, str] = {
    "ANA": "ANA", "BOS": "BOS", "BUF": "BUF", "CAR": "CAR",
    "CBJ": "CBJ", "CGY": "CGY", "CHI": "CHI", "COL": "COL",
    "DAL": "DAL", "DET": "DET", "EDM": "EDM", "FLA": "FLA",
    "LA": "LAK", "MIN": "MIN", "MTL": "MTL", "NJ": "NJD",
    "NSH": "NSH", "NYI": "NYI", "NYR": "NYR", "OTT": "OTT",
    "PHI": "PHI", "PIT": "PIT", "SEA": "SEA", "SJ": "SJS",
    "STL": "STL", "TB": "TBL", "TOR": "TOR", "UTA": "UTA",
    "VAN": "VAN", "VGK": "VGK", "WPG": "WPG", "WSH": "WSH",
}

# Trial tier: 1 request per second
_RATE_LIMIT_SECONDS = 1.1


class SportsRadarConnector:
    """Comprehensive NHL data from SportsRadar v7 API.

    Provides standings, schedule, goaltender stats, advanced analytics
    (Corsi, Fenwick, PDO), and injuries — all from one source.

    Requires SPORTRADAR_API_KEY env var (or pass api_key to __init__).
    Trial tier: 1 req/sec, 1000 req/mo.
    """

    def __init__(self, api_key: str | None = None, access: str = "trial"):
        self.api_key = api_key or os.environ.get("SPORTRADAR_API_KEY", "")
        self.access = access
        self._base = BASE_URL.format(access=access)
        self._last_request: float = 0
        self._last_error: str | None = None
        # Cached hierarchy: alias → team_id (UUID)
        self._team_ids: dict[str, str] = {}

    def _rate_limit(self) -> None:
        """Enforce trial-tier rate limit (1 req/sec)."""
        elapsed = time.time() - self._last_request
        if elapsed < _RATE_LIMIT_SECONDS:
            time.sleep(_RATE_LIMIT_SECONDS - elapsed)
        self._last_request = time.time()

    def _fetch_json(self, path: str) -> dict[str, Any] | list[Any] | None:
        """Fetch JSON from SportsRadar API. Returns None on failure."""
        self._last_error = None
        url = f"{self._base}/{path}"
        separator = "&" if "?" in url else "?"
        url += f"{separator}api_key={self.api_key}"
        self._rate_limit()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ABBA/2.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            self._last_error = f"HTTP {e.code}: {e.reason} ({path})"
            return None
        except urllib.error.URLError as e:
            self._last_error = f"URL error: {e.reason} ({path})"
            return None
        except (json.JSONDecodeError, TimeoutError) as e:
            self._last_error = f"{type(e).__name__}: {e} ({path})"
            return None

    @staticmethod
    def _alias_to_abbrev(alias: str) -> str:
        """Convert SportsRadar alias to ABBA abbreviation."""
        return _SR_ALIAS_TO_ABBREV.get(alias, alias)

    # ------------------------------------------------------------------
    # Hierarchy — build team UUID lookup
    # ------------------------------------------------------------------

    def _ensure_hierarchy(self) -> None:
        """Fetch league hierarchy to build alias → UUID map if not cached."""
        if self._team_ids:
            return
        data = self._fetch_json("league/hierarchy.json")
        if not data:
            return
        for conf in data.get("conferences", []):
            for div in conf.get("divisions", []):
                for t in div.get("teams", []):
                    alias = t.get("alias", "")
                    self._team_ids[alias] = t["id"]

    def _team_uuid(self, alias: str) -> str | None:
        """Get SportsRadar UUID for a team alias."""
        self._ensure_hierarchy()
        return self._team_ids.get(alias)

    # ------------------------------------------------------------------
    # Standings
    # ------------------------------------------------------------------

    def fetch_standings(
        self, storage: Storage, season_year: int = 2025
    ) -> dict[str, Any]:
        """Fetch NHL standings and store as team_stats.

        Args:
            season_year: Start year of the NHL season (e.g. 2025 for 2025-26).
        """
        data = self._fetch_json(f"seasons/{season_year}/REG/standings.json")
        if not data or "conferences" not in data:
            return {"status": "failed", "error": self._last_error or "no standings data"}

        season_str = f"{season_year}-{str(season_year + 1)[-2:]}"
        team_stats: list[dict[str, Any]] = []

        for conf in data.get("conferences", []):
            conf_name = conf.get("name", "")
            for div in conf.get("divisions", []):
                div_name = div.get("name", "")
                for t in div.get("teams", []):
                    alias = t.get("alias", "")
                    abbrev = self._alias_to_abbrev(alias)
                    # Cache UUID while we have it
                    if "id" in t:
                        self._team_ids[alias] = t["id"]

                    wins = t.get("wins", 0)
                    losses = t.get("losses", 0)
                    otl = t.get("overtime_losses", 0)

                    # Extract home/road/L10 from records
                    records = {r["record_type"]: r for r in t.get("records", [])}
                    home_rec = records.get("home", {})
                    road_rec = records.get("road", {})
                    l10_rec = records.get("last_10", {})
                    l10_wins = l10_rec.get("wins", 0)
                    l10_losses = l10_rec.get("losses", 0)
                    l10_otl = l10_rec.get("overtime_losses", 0)
                    l10_total = l10_wins + l10_losses + l10_otl
                    recent_form = l10_wins / max(l10_total, 1)

                    team_stats.append({
                        "team_id": abbrev,
                        "sport": "NHL",
                        "season": season_str,
                        "stats": {
                            "wins": wins,
                            "losses": losses,
                            "overtime_losses": otl,
                            "points": t.get("points", 0),
                            "goals_for": t.get("goals_for", 0),
                            "goals_against": t.get("goals_against", 0),
                            "goal_differential": t.get("goal_diff", 0),
                            "games_played": t.get("games_played", 0),
                            "regulation_wins": t.get("regulation_wins", 0),
                            "points_pct": t.get("points_pct", 0) / 100,
                            "power_play_percentage": t.get("powerplay_pct", 0),
                            "penalty_kill_percentage": t.get("penalty_killing_pct", 0),
                            "home_wins": home_rec.get("wins", 0),
                            "home_losses": home_rec.get("losses", 0),
                            "home_ot_losses": home_rec.get("overtime_losses", 0),
                            "road_wins": road_rec.get("wins", 0),
                            "road_losses": road_rec.get("losses", 0),
                            "road_ot_losses": road_rec.get("overtime_losses", 0),
                            "l10_wins": l10_wins,
                            "l10_losses": l10_losses,
                            "l10_ot_losses": l10_otl,
                            "recent_form": recent_form,
                            "streak_code": t.get("streak", {}).get("kind", ""),
                            "streak_count": t.get("streak", {}).get("length", 0),
                            "division_name": div_name,
                            "conference_name": conf_name,
                            "division_rank": t.get("rank", {}).get("division", 0),
                            "conference_rank": t.get("rank", {}).get("conference", 0),
                            "clinched": t.get("rank", {}).get("clinched", ""),
                        },
                        "source": "sportradar",
                    })

        stored = storage.upsert_team_stats(team_stats)
        return {"status": "ok", "teams_updated": stored, "season": season_str}

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def fetch_schedule(
        self, storage: Storage, game_date: str | None = None
    ) -> dict[str, Any]:
        """Fetch daily schedule and store games.

        Args:
            game_date: ISO date string (YYYY-MM-DD). Defaults to today.
        """
        if game_date is None:
            game_date = date.today().isoformat()

        parts = game_date.split("-")
        path = f"games/{parts[0]}/{parts[1]}/{parts[2]}/schedule.json"
        data = self._fetch_json(path)
        if not data or "games" not in data:
            return {"status": "failed", "error": self._last_error or "no schedule data"}

        games: list[dict[str, Any]] = []
        for g in data.get("games", []):
            home_alias = g.get("home", {}).get("alias", "")
            away_alias = g.get("away", {}).get("alias", "")
            home_abbrev = self._alias_to_abbrev(home_alias)
            away_abbrev = self._alias_to_abbrev(away_alias)

            sr_status = g.get("status", "")
            if sr_status in ("closed", "complete"):
                status = "final"
            elif sr_status in ("inprogress", "halftime"):
                status = "live"
            else:
                status = "scheduled"

            game_entry: dict[str, Any] = {
                "game_id": f"nhl-sr-{g['id'][:8]}",
                "sport": "NHL",
                "date": game_date,
                "home_team": home_abbrev,
                "away_team": away_abbrev,
                "venue": g.get("venue", {}).get("name", ""),
                "status": status,
                "metadata": {
                    "sr_id": g.get("id", ""),
                    "start_time": g.get("scheduled", ""),
                    "coverage": g.get("coverage", ""),
                    "broadcasts": [
                        b.get("network", "")
                        for b in g.get("broadcasts", [])
                    ],
                },
                "source": "sportradar",
            }

            if status == "final":
                game_entry["home_score"] = g.get("home_points", 0)
                game_entry["away_score"] = g.get("away_points", 0)

            games.append(game_entry)

        stored = storage.upsert_games(games)
        return {"status": "ok", "games_stored": stored, "date": game_date}

    # ------------------------------------------------------------------
    # Team statistics + goaltender stats
    # ------------------------------------------------------------------

    def fetch_team_statistics(
        self,
        storage: Storage,
        team_alias: str,
        season_year: int = 2025,
    ) -> dict[str, Any]:
        """Fetch seasonal statistics for a team — includes goalie stats.

        Writes goaltender_stats to storage for all goalies on the roster.
        """
        uuid = self._team_uuid(team_alias)
        if not uuid:
            # If hierarchy not loaded, try to load it
            self._ensure_hierarchy()
            uuid = self._team_uuid(team_alias)
            if not uuid:
                return {"status": "failed", "error": f"Unknown team alias: {team_alias}"}

        data = self._fetch_json(
            f"seasons/{season_year}/REG/teams/{uuid}/statistics.json"
        )
        if not data:
            return {"status": "failed", "error": self._last_error or "no stats data"}

        season_str = f"{season_year}-{str(season_year + 1)[-2:]}"
        abbrev = self._alias_to_abbrev(team_alias)

        # Extract goaltender stats
        goalie_stats: list[dict[str, Any]] = []
        players = data.get("players", [])
        goalies = [p for p in players if p.get("primary_position") == "G"]

        for g in goalies:
            gt = g.get("goaltending", {}).get("total", {})
            if not gt or gt.get("games_played", 0) == 0:
                continue

            pid = g.get("id", "")
            sa = gt.get("shots_against", 0)
            ga = gt.get("goals_against", 0)
            save_pct = gt.get("saves_pct", 0.0)
            gaa = gt.get("avg_goals_against", 0.0)
            # GSAA = expected_GA - actual_GA (positive = saved more than avg)
            expected_ga = sa * (1 - 0.907)  # 0.093 = league avg GA rate
            gsaa = round(expected_ga - ga, 2) if sa > 0 else 0.0
            gs = g.get("statistics", {}).get("total", {}).get("games_started", 0)

            goalie_stats.append({
                "goaltender_id": pid,
                "team": abbrev,
                "season": season_str,
                "stats": {
                    "name": g.get("full_name", ""),
                    "games_played": gt.get("games_played", 0),
                    "games_started": gs,
                    "wins": gt.get("wins", 0),
                    "losses": gt.get("losses", 0),
                    "ot_losses": gt.get("overtime_losses", 0),
                    "save_pct": round(save_pct, 4),
                    "gaa": round(gaa, 2),
                    "gsaa": gsaa,
                    "save_percentage": round(save_pct, 4),
                    "goals_against_average": round(gaa, 2),
                    "goals_against": ga,
                    "saves": gt.get("saves", 0),
                    "shots_against": sa,
                    "shutouts": gt.get("shutouts", 0),
                },
            })

        # Tag starter (most games started)
        if goalie_stats:
            max_gs = max(g["stats"].get("games_started", 0) for g in goalie_stats)
            for g in goalie_stats:
                if g["stats"].get("games_started", 0) == max_gs and max_gs > 0:
                    g["stats"]["role"] = "starter"
                    break

        goalies_stored = storage.upsert_goaltender_stats(goalie_stats)
        return {
            "status": "ok",
            "team": abbrev,
            "goalies_stored": goalies_stored,
            "season": season_str,
        }

    # ------------------------------------------------------------------
    # Advanced analytics (Corsi, Fenwick, PDO)
    # ------------------------------------------------------------------

    def fetch_analytics(
        self,
        storage: Storage,
        team_alias: str,
        season_year: int = 2025,
    ) -> dict[str, Any]:
        """Fetch seasonal analytics (Corsi, Fenwick, PDO) for a team."""
        uuid = self._team_uuid(team_alias)
        if not uuid:
            self._ensure_hierarchy()
            uuid = self._team_uuid(team_alias)
            if not uuid:
                return {"status": "failed", "error": f"Unknown team alias: {team_alias}"}

        data = self._fetch_json(
            f"seasons/{season_year}/REG/teams/{uuid}/analytics.json"
        )
        if not data:
            return {"status": "failed", "error": self._last_error or "no analytics data"}

        season_str = f"{season_year}-{str(season_year + 1)[-2:]}"
        abbrev = self._alias_to_abbrev(team_alias)

        own = data.get("own_record", {}).get("statistics", {})
        totals = own.get("total", {})
        averages = own.get("average", {})

        if not totals:
            return {"status": "no_data", "error": "No analytics totals", "team": abbrev}

        gp = totals.get("games_played", 1)
        sf = totals.get("on_ice_shots_for", 0)
        sa = totals.get("on_ice_shots_against", 0)

        record = {
            "team_id": abbrev,
            "season": season_str,
            "stats": {
                "corsi_pct": round(totals.get("corsi_pct", 50.0) * 100, 2),
                "fenwick_pct": round(totals.get("fenwick_pct", 50.0) * 100, 2),
                "corsi_for": totals.get("corsi_for", 0),
                "corsi_against": totals.get("corsi_against", 0),
                "fenwick_for": totals.get("fenwick_for", 0),
                "fenwick_against": totals.get("fenwick_against", 0),
                "on_ice_shots_for": sf,
                "on_ice_shots_against": sa,
                # SR PDO is on a non-standard scale; store raw for reference
                "pdo_raw": totals.get("pdo", 0),
                "shots_for_per60": round(averages.get("on_ice_shots_for", 0) * 3, 2),
                "shots_against_per60": round(averages.get("on_ice_shots_against", 0) * 3, 2),
                "avg_shot_distance": averages.get("average_shot_distance", 0),
                "source": "sportradar",
            },
        }

        # Compute xGF% proxy from shot differential + PDO decomposition
        # SR doesn't provide xG directly, but corsi_pct is the best proxy
        # Set xgf_pct equal to corsi_pct for feature compatibility
        record["stats"]["xgf_pct"] = record["stats"]["corsi_pct"]

        stored = storage.upsert_nhl_advanced_stats([record])
        return {"status": "ok", "team": abbrev, "stored": stored, "season": season_str}

    # ------------------------------------------------------------------
    # Injuries
    # ------------------------------------------------------------------

    def fetch_injuries(self, storage: Storage) -> dict[str, Any]:
        """Fetch league-wide injuries and update roster injury status."""
        data = self._fetch_json("league/injuries.json")
        if not data or "teams" not in data:
            return {"status": "failed", "error": self._last_error or "no injury data"}

        updates = 0
        for team in data.get("teams", []):
            alias = team.get("alias", "")
            abbrev = self._alias_to_abbrev(alias)

            for player in team.get("players", []):
                injuries = player.get("injuries", [])
                if not injuries:
                    continue

                injury = injuries[0]  # Most recent
                status = injury.get("status", "")
                pid = player.get("id", "")

                # Map SR injury status to ABBA format
                if status in ("Out", "Out For Season"):
                    injury_status = "injured"
                elif status == "Day To Day":
                    injury_status = "day-to-day"
                else:
                    injury_status = "questionable"

                # Update roster entry if it exists
                existing = storage.query_roster(team=abbrev)
                for p in existing:
                    if p.get("player_id") == pid:
                        storage.upsert_roster([{
                            **p,
                            "injury_status": injury_status,
                        }])
                        updates += 1
                        break

        return {"status": "ok", "injury_updates": updates}

    # ------------------------------------------------------------------
    # Full refresh
    # ------------------------------------------------------------------

    def refresh(
        self,
        storage: Storage,
        team: str | None = None,
        season_year: int = 2025,
    ) -> dict[str, Any]:
        """Full refresh: standings + schedule + per-team stats/analytics + injuries.

        Args:
            storage: DuckDB storage.
            team: Optional single team to refresh (abbreviation). If None, refreshes all.
            season_year: NHL season start year (e.g. 2025 for 2025-26).
        """
        if not self.api_key:
            return {
                "status": "no_api_key",
                "error": "Set SPORTRADAR_API_KEY env var or pass api_key to constructor.",
            }

        results: dict[str, Any] = {
            "source": "sportradar",
            "fetched_at": datetime.now().isoformat(),
        }

        # 1. Standings (all teams, 1 API call)
        results["standings"] = self.fetch_standings(storage, season_year)

        # 2. Schedule (today, 1 API call)
        results["schedule"] = self.fetch_schedule(storage)

        # 3. Determine teams to fetch detailed stats for
        teams_to_fetch: set[str] = set()
        if team:
            teams_to_fetch.add(team.upper())
        else:
            # Fetch for teams with scheduled games
            scheduled = storage.query_games(sport="NHL", status="scheduled", limit=200)
            for g in scheduled:
                teams_to_fetch.add(g["home_team"])
                teams_to_fetch.add(g["away_team"])

        # 4. Per-team statistics + analytics
        if teams_to_fetch:
            stats_results: dict[str, Any] = {}
            analytics_results: dict[str, Any] = {}
            for t in sorted(teams_to_fetch):
                # Map ABBA abbreviation back to SR alias
                sr_alias = t
                for alias, abbrev in _SR_ALIAS_TO_ABBREV.items():
                    if abbrev == t:
                        sr_alias = alias
                        break
                stats_results[t] = self.fetch_team_statistics(
                    storage, sr_alias, season_year
                )
                analytics_results[t] = self.fetch_analytics(
                    storage, sr_alias, season_year
                )
            results["team_statistics"] = stats_results
            results["analytics"] = analytics_results

        # 5. Injuries (1 API call)
        results["injuries"] = self.fetch_injuries(storage)

        return results
