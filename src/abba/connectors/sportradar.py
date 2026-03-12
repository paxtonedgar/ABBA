"""SportsRadar NHL v7 connector — schema-complete extraction.

Designed backwards from the official XSD schemas to extract every field the API
provides per call, maximizing value from each of the 1000 trial-tier queries.

Endpoints (queries used):
  - hierarchy.json           (1)  → team UUIDs, alias mapping
  - standings.json           (1)  → 32 teams: W/L/OTL, PP%, PK%, home/road/L10, ranks, clinch
  - schedule.json            (1)  → daily games with venue, broadcasts, scores
  - teams/{id}/statistics    (1/team) → skater totals, strength splits, faceoffs, goalie stats
  - teams/{id}/analytics     (1/team) → Corsi, Fenwick, PDO, zone starts, shot types
  - teams/{id}/depth_chart   (1/team) → line combos, depth order, coaches
  - league/injuries          (1)  → all teams: injury desc, status, dates
  - league/transfers         (1)  → trades, signings, waivers

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

# Reverse: ABBA abbreviation → SR alias (for API calls)
_ABBREV_TO_SR_ALIAS: dict[str, str] = {v: k for k, v in _SR_ALIAS_TO_ABBREV.items()}

# Trial tier: 1 request per second
_RATE_LIMIT_SECONDS = 1.1


class SportsRadarConnector:
    """Schema-complete NHL data from SportsRadar v7 API.

    Extracts every field defined in the v7 XSD schemas to maximize the value
    of each API call against the trial-tier budget (1000 req/mo, 1 req/sec).
    """

    def __init__(self, api_key: str | None = None, access: str = "trial"):
        self.api_key = api_key or os.environ.get("SPORTRADAR_API_KEY", "")
        self.access = access
        self._base = BASE_URL.format(access=access)
        self._last_request: float = 0
        self._last_error: str | None = None
        self._team_ids: dict[str, str] = {}      # alias → UUID
        self._uuid_to_alias: dict[str, str] = {}  # UUID → alias

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < _RATE_LIMIT_SECONDS:
            time.sleep(_RATE_LIMIT_SECONDS - elapsed)
        self._last_request = time.time()

    def _fetch_json(self, path: str) -> dict[str, Any] | list[Any] | None:
        self._last_error = None
        url = f"{self._base}/{path}"
        sep = "&" if "?" in url else "?"
        url += f"{sep}api_key={self.api_key}"
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
        return _SR_ALIAS_TO_ABBREV.get(alias, alias)

    @staticmethod
    def _abbrev_to_alias(abbrev: str) -> str:
        return _ABBREV_TO_SR_ALIAS.get(abbrev, abbrev)

    # ------------------------------------------------------------------
    # Hierarchy
    # ------------------------------------------------------------------

    def _ensure_hierarchy(self) -> None:
        if self._team_ids:
            return
        data = self._fetch_json("league/hierarchy.json")
        if not data:
            return
        for conf in data.get("conferences", []):
            for div in conf.get("divisions", []):
                for t in div.get("teams", []):
                    alias = t.get("alias", "")
                    tid = t.get("id", "")
                    if alias and tid:
                        self._team_ids[alias] = tid
                        self._uuid_to_alias[tid] = alias

    def _team_uuid(self, alias: str) -> str | None:
        self._ensure_hierarchy()
        return self._team_ids.get(alias)

    # ------------------------------------------------------------------
    # Standings (schema: standings-v4.0.xsd)
    # ------------------------------------------------------------------

    def fetch_standings(
        self, storage: Storage, season_year: int = 2025
    ) -> dict[str, Any]:
        """Fetch all 32 teams' standings. 1 API call.

        Schema fields extracted:
          - recordAttributes: wins, losses, overtime_losses, win_pct, goals_for/against, points
          - team-level: games_played, regulation_wins, shootout_wins/losses, goal_diff,
            powerplay stats, penalty_killing_pct, points_pct, points_per_game
          - records[]: home, road, last_10, last_10_home, last_10_road, division, conference
          - rank: division, conference, league, wildcard, clinched
          - calc_rank: div_rank, div_tiebreak, conf_rank, conf_tiebreak
          - streak: win/loss + length
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
                    tid = t.get("id", "")
                    alias = t.get("alias", "") or self._uuid_to_alias.get(tid, "")
                    abbrev = self._alias_to_abbrev(alias)
                    if alias and tid:
                        self._team_ids[alias] = tid
                        self._uuid_to_alias[tid] = alias

                    # --- Core record ---
                    wins = t.get("wins", 0)
                    losses = t.get("losses", 0)
                    otl = t.get("overtime_losses", 0)
                    gp = t.get("games_played", 0)

                    # --- Situational records ---
                    records = {r.get("record_type", ""): r for r in t.get("records", [])}
                    home = records.get("home", {})
                    road = records.get("road", {})
                    l10 = records.get("last_10", {})
                    l10h = records.get("last_10_home", {})
                    l10r = records.get("last_10_road", {})
                    div_rec = records.get("division", {})
                    conf_rec = records.get("conference", {})

                    l10_w = l10.get("wins", 0)
                    l10_l = l10.get("losses", 0)
                    l10_otl = l10.get("overtime_losses", 0)
                    l10_total = l10_w + l10_l + l10_otl
                    recent_form = l10_w / max(l10_total, 1)

                    # --- Rank ---
                    rank = t.get("rank", {})
                    calc = t.get("calc_rank", {})

                    # --- Streak ---
                    streak = t.get("streak", {})
                    streak_code = streak.get("kind", "")
                    streak_len = streak.get("length", 0)

                    # --- Power play ---
                    pp_pct = t.get("powerplay_pct", 0.0)
                    pk_pct = t.get("penalty_killing_pct", 0.0)
                    pp_opps = t.get("powerplays", 0)
                    pp_goals = t.get("powerplay_goals", 0)
                    ppa = t.get("powerplays_against", 0)
                    ppga = t.get("powerplay_goals_against", 0)

                    stats: dict[str, Any] = {
                        # Core
                        "wins": wins,
                        "losses": losses,
                        "overtime_losses": otl,
                        "points": t.get("points", 0),
                        "games_played": gp,
                        "goals_for": t.get("goals_for", 0),
                        "goals_against": t.get("goals_against", 0),
                        "goal_differential": t.get("goal_diff", 0),
                        "regulation_wins": t.get("regulation_wins", 0),
                        "regulation_overtime_wins": t.get("regulation_overtime_wins", 0),
                        "shootout_wins": t.get("shootout_wins", 0),
                        "shootout_losses": t.get("shootout_losses", 0),
                        "win_pct": t.get("win_pct", 0.0),
                        "points_pct": t.get("points_pct", 0.0),
                        "points_per_game": t.get("points_per_game", 0.0),
                        # Special teams
                        "power_play_percentage": pp_pct,
                        "penalty_kill_percentage": pk_pct,
                        "powerplay_opportunities": pp_opps,
                        "powerplay_goals": pp_goals,
                        "powerplays_against": ppa,
                        "powerplay_goals_against": ppga,
                        # Home
                        "home_wins": home.get("wins", 0),
                        "home_losses": home.get("losses", 0),
                        "home_ot_losses": home.get("overtime_losses", 0),
                        "home_goals_for": home.get("goals_for", 0),
                        "home_goals_against": home.get("goals_against", 0),
                        "home_points": home.get("points", 0),
                        # Road
                        "road_wins": road.get("wins", 0),
                        "road_losses": road.get("losses", 0),
                        "road_ot_losses": road.get("overtime_losses", 0),
                        "road_goals_for": road.get("goals_for", 0),
                        "road_goals_against": road.get("goals_against", 0),
                        "road_points": road.get("points", 0),
                        # L10
                        "l10_wins": l10_w,
                        "l10_losses": l10_l,
                        "l10_ot_losses": l10_otl,
                        "l10_home_wins": l10h.get("wins", 0),
                        "l10_home_losses": l10h.get("losses", 0),
                        "l10_road_wins": l10r.get("wins", 0),
                        "l10_road_losses": l10r.get("losses", 0),
                        "recent_form": recent_form,
                        # Division/conference record
                        "div_wins": div_rec.get("wins", 0),
                        "div_losses": div_rec.get("losses", 0),
                        "div_ot_losses": div_rec.get("overtime_losses", 0),
                        "conf_wins": conf_rec.get("wins", 0),
                        "conf_losses": conf_rec.get("losses", 0),
                        "conf_ot_losses": conf_rec.get("overtime_losses", 0),
                        # Streak
                        "streak_code": streak_code,
                        "streak_count": streak_len,
                        # Rank
                        "division_rank": rank.get("division", 0),
                        "conference_rank": rank.get("conference", 0),
                        "league_rank": rank.get("league", 0),
                        "wildcard_rank": rank.get("wildcard", 0),
                        "clinched": rank.get("clinched", ""),
                        "calc_div_rank": calc.get("div_rank", 0),
                        "calc_conf_rank": calc.get("conf_rank", 0),
                        "calc_div_tiebreak": calc.get("div_tiebreak", ""),
                        "calc_conf_tiebreak": calc.get("conf_tiebreak", ""),
                        # Organization
                        "division_name": div_name,
                        "conference_name": conf_name,
                    }

                    team_stats.append({
                        "team_id": abbrev,
                        "sport": "NHL",
                        "season": season_str,
                        "stats": stats,
                        "source": "sportradar",
                    })

        stored = storage.upsert_team_stats(team_stats)
        return {"status": "ok", "teams_updated": stored, "season": season_str}

    # ------------------------------------------------------------------
    # Schedule (schema: schedule-v6.0.xsd)
    # ------------------------------------------------------------------

    def fetch_schedule(
        self, storage: Storage, game_date: str | None = None
    ) -> dict[str, Any]:
        """Fetch daily schedule. 1 API call.

        Schema fields: game id, status, scheduled time, home/away with alias,
        venue (name, city, capacity, timezone), broadcasts (network, cable, satellite).
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
            home = g.get("home", {})
            away = g.get("away", {})
            venue = g.get("venue", {})
            home_abbrev = self._alias_to_abbrev(home.get("alias", ""))
            away_abbrev = self._alias_to_abbrev(away.get("alias", ""))

            sr_status = g.get("status", "")
            if sr_status in ("closed", "complete"):
                status = "final"
            elif sr_status in ("inprogress", "halftime"):
                status = "live"
            else:
                status = "scheduled"

            broadcasts = g.get("broadcasts", [])
            broadcast_info = {}
            for b in broadcasts:
                for key in ("network", "cable", "satellite", "radio", "internet"):
                    if b.get(key):
                        broadcast_info[key] = b[key]

            game_entry: dict[str, Any] = {
                "game_id": f"nhl-sr-{g['id'][:8]}",
                "sport": "NHL",
                "date": game_date,
                "home_team": home_abbrev,
                "away_team": away_abbrev,
                "venue": venue.get("name", ""),
                "status": status,
                "metadata": {
                    "sr_id": g.get("id", ""),
                    "sr_home_id": home.get("id", ""),
                    "sr_away_id": away.get("id", ""),
                    "start_time": g.get("scheduled", ""),
                    "coverage": g.get("coverage", ""),
                    "broadcasts": broadcast_info,
                    "venue_city": venue.get("city", ""),
                    "venue_capacity": venue.get("capacity", 0),
                    "venue_timezone": venue.get("time_zone", ""),
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
    # Team Statistics (schema: statistics-v7.0 + common-season-v4.0)
    # ------------------------------------------------------------------

    def fetch_team_statistics(
        self,
        storage: Storage,
        team_alias: str,
        season_year: int = 2025,
    ) -> dict[str, Any]:
        """Fetch seasonal statistics for a team. 1 API call.

        Extracts from schema:
          Team-level: goals, assists, shots, missed_shots, hits, blocked_shots,
            giveaways, takeaways, faceoff_win_pct, penalties, PIM, +/-,
            plus powerplay/shorthanded/evenstrength splits with strength breakdowns.
          Goaltenders: full stats + PP/SH/ES/penalty splits.
          Skaters: per-player G/A/P/+- (stored to roster for player impact).
        """
        uuid = self._team_uuid(team_alias)
        if not uuid:
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
        players = data.get("players", [])

        # ---- Team-level skater stats (from team_records.overall) ----
        team_rec = data.get("team_records", {}).get("overall", {})
        team_total = team_rec.get("statistics", {}).get("total", {})

        team_skater_stats = {}
        if team_total:
            # baseSkaterAttributes + faceoffAttributes
            team_skater_stats = {
                "goals": team_total.get("goals", 0),
                "assists": team_total.get("assists", 0),
                "shots": team_total.get("shots", 0),
                "missed_shots": team_total.get("missed_shots", 0),
                "points": team_total.get("points", 0),
                "shooting_pct": team_total.get("shooting_pct", 0.0),
                "hits": team_total.get("hits", 0),
                "blocked_shots": team_total.get("blocked_shots", 0),
                "blocked_att": team_total.get("blocked_att", 0),
                "giveaways": team_total.get("giveaways", 0),
                "takeaways": team_total.get("takeaways", 0),
                "penalties": team_total.get("penalties", 0),
                "penalty_minutes": team_total.get("penalty_minutes", 0),
                "team_penalties": team_total.get("team_penalties", 0),
                "team_penalty_minutes": team_total.get("team_penalty_minutes", 0),
                "plus_minus": team_total.get("plus_minus", 0),
                "faceoffs_won": team_total.get("faceoffs_won", 0),
                "faceoffs_lost": team_total.get("faceoffs_lost", 0),
                "faceoff_win_pct": team_total.get("faceoff_win_pct", 0.0),
                "games_played": team_total.get("games_played", 0),
                "powerplays": team_total.get("powerplays", 0),
                "winning_goals": team_total.get("winning_goals", 0),
                "overtime_goals": team_total.get("overtime_goals", 0),
                "emptynet_goals": team_total.get("emptynet_goals", 0),
            }

            # Strength splits (5v5, PP, SH)
            for situation in ("powerplay", "shorthanded", "evenstrength"):
                sit_data = team_total.get(situation, {})
                if sit_data:
                    prefix = {"powerplay": "pp", "shorthanded": "sh", "evenstrength": "es"}[situation]
                    team_skater_stats[f"{prefix}_goals"] = sit_data.get("goals", 0)
                    team_skater_stats[f"{prefix}_assists"] = sit_data.get("assists", 0)
                    team_skater_stats[f"{prefix}_shots"] = sit_data.get("shots", 0)

                    # Strength sub-splits (5v5, 5v4, etc.)
                    for strength in sit_data.get("strength", []):
                        st = strength.get("type", "")
                        if st:
                            team_skater_stats[f"{prefix}_{st}_goals"] = strength.get("goals", 0)

        # ---- Goaltender stats ----
        goalie_stats: list[dict[str, Any]] = []
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
            expected_ga = sa * (1 - 0.907)
            gsaa = round(expected_ga - ga, 2) if sa > 0 else 0.0
            gs = g.get("statistics", {}).get("total", {}).get("games_started", 0)

            goalie_stat: dict[str, Any] = {
                "name": g.get("full_name", ""),
                "jersey_number": g.get("jersey_number", ""),
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
            }

            # Situation splits for goalie (PP/SH/ES/penalty)
            for situation in ("powerplay", "shorthanded", "evenstrength", "penalty"):
                sit = gt.get(situation, {})
                if sit:
                    prefix = {"powerplay": "pp", "shorthanded": "sh",
                              "evenstrength": "es", "penalty": "pen"}[situation]
                    goalie_stat[f"{prefix}_sa"] = sit.get("shots_against", 0)
                    goalie_stat[f"{prefix}_ga"] = sit.get("goals_against", 0)
                    goalie_stat[f"{prefix}_saves"] = sit.get("saves", 0)
                    goalie_stat[f"{prefix}_sv_pct"] = sit.get("saves_pct", 0.0)

            goalie_stats.append({
                "goaltender_id": pid,
                "team": abbrev,
                "season": season_str,
                "stats": goalie_stat,
            })

        # Tag starter
        if goalie_stats:
            max_gs = max(g["stats"].get("games_started", 0) for g in goalie_stats)
            for g in goalie_stats:
                if g["stats"].get("games_started", 0) == max_gs and max_gs > 0:
                    g["stats"]["role"] = "starter"
                    break

        goalies_stored = storage.upsert_goaltender_stats(goalie_stats)

        # ---- Skater roster population ----
        skaters = [p for p in players if p.get("primary_position") != "G"]
        roster_entries: list[dict[str, Any]] = []
        for p in skaters:
            st = p.get("statistics", {}).get("total", {})
            if not st or st.get("games_played", 0) == 0:
                continue
            roster_entries.append({
                "player_id": p.get("id", ""),
                "team": abbrev,
                "season": season_str,
                "name": p.get("full_name", ""),
                "position": p.get("primary_position", ""),
                "stats": {
                    "jersey_number": p.get("jersey_number", ""),
                    "games_played": st.get("games_played", 0),
                    "goals": st.get("goals", 0),
                    "assists": st.get("assists", 0),
                    "points": st.get("points", 0),
                    "plus_minus": st.get("plus_minus", 0),
                    "shots": st.get("shots", 0),
                    "shooting_pct": st.get("shooting_pct", 0.0),
                    "hits": st.get("hits", 0),
                    "blocked_shots": st.get("blocked_shots", 0),
                    "giveaways": st.get("giveaways", 0),
                    "takeaways": st.get("takeaways", 0),
                    "penalty_minutes": st.get("penalty_minutes", 0),
                    "faceoff_win_pct": st.get("faceoff_win_pct", 0.0),
                    "pp_goals": st.get("powerplay", {}).get("goals", 0),
                    "pp_assists": st.get("powerplay", {}).get("assists", 0),
                    "sh_goals": st.get("shorthanded", {}).get("goals", 0),
                    "es_goals": st.get("evenstrength", {}).get("goals", 0),
                    "winning_goals": st.get("winning_goals", 0),
                    "overtime_goals": st.get("overtime_goals", 0),
                },
                "injury_status": "healthy",
            })

        roster_stored = 0
        if roster_entries:
            roster_stored = storage.upsert_roster(roster_entries)

        return {
            "status": "ok",
            "team": abbrev,
            "goalies_stored": goalies_stored,
            "skaters_stored": roster_stored,
            "team_stats": team_skater_stats,
            "season": season_str,
        }

    # ------------------------------------------------------------------
    # Analytics (schema: analytics-v6.0.xsd)
    # ------------------------------------------------------------------

    def fetch_analytics(
        self,
        storage: Storage,
        team_alias: str,
        season_year: int = 2025,
    ) -> dict[str, Any]:
        """Fetch Corsi, Fenwick, PDO, zone starts, shot types. 1 API call.

        Schema fields extracted:
          total: corsi_for/against/pct, fenwick_for/against/pct, pdo,
            on_ice_shots_for/against/differential/pct, average_shot_distance
          shots: wrist/slap/backhand/tip/snap/wrap_around (shots + goals each)
          starts: offensive_zone/defensive_zone/neutral_zone starts
          average: per-game versions of all above
        """
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

        stats: dict[str, Any] = {
            # Core analytics
            "corsi_pct": round(totals.get("corsi_pct", 50.0) * 100, 2),
            "fenwick_pct": round(totals.get("fenwick_pct", 50.0) * 100, 2),
            "corsi_for": totals.get("corsi_for", 0),
            "corsi_against": totals.get("corsi_against", 0),
            "corsi_total": totals.get("corsi_total", 0),
            "fenwick_for": totals.get("fenwick_for", 0),
            "fenwick_against": totals.get("fenwick_against", 0),
            "fenwick_total": totals.get("fenwick_total", 0),
            "on_ice_shots_for": totals.get("on_ice_shots_for", 0),
            "on_ice_shots_against": totals.get("on_ice_shots_against", 0),
            "on_ice_shots_differential": totals.get("on_ice_shots_differential", 0),
            "on_ice_shots_pct": totals.get("on_ice_shots_pct", 0.0),
            "pdo_raw": totals.get("pdo", 0),
            "avg_shot_distance": totals.get("average_shot_distance", 0),
            "games_played": totals.get("games_played", 0),
            # Per-game averages
            "avg_corsi_for": averages.get("corsi_for", 0),
            "avg_corsi_against": averages.get("corsi_against", 0),
            "avg_fenwick_for": averages.get("fenwick_for", 0),
            "avg_fenwick_against": averages.get("fenwick_against", 0),
            "shots_for_per60": round(averages.get("on_ice_shots_for", 0) * 3, 2),
            "shots_against_per60": round(averages.get("on_ice_shots_against", 0) * 3, 2),
            "source": "sportradar",
        }

        # Zone starts (from total.starts)
        starts = totals.get("starts", {})
        if starts:
            oz = starts.get("offensive_zone_starts", 0)
            dz = starts.get("defensive_zone_starts", 0)
            nz = starts.get("neutral_zone_starts", 0)
            total_starts = oz + dz + nz
            stats["oz_starts"] = oz
            stats["dz_starts"] = dz
            stats["nz_starts"] = nz
            stats["oz_start_pct"] = round(oz / max(total_starts, 1) * 100, 1)
            stats["dz_start_pct"] = round(dz / max(total_starts, 1) * 100, 1)

        # Shot type breakdown (from total.shots)
        shots = totals.get("shots", {})
        if shots:
            for shot_type in ("wrist_shot", "slap_shot", "backhand_shot",
                              "tip_shot", "snap_shot", "wrap_around_shot"):
                stats[f"{shot_type}_shots"] = shots.get(f"{shot_type}_shots", 0)
                stats[f"{shot_type}_goals"] = shots.get(f"{shot_type}_goals", 0)

        # xGF% proxy — corsi_pct is best available from SR
        stats["xgf_pct"] = stats["corsi_pct"]

        record = {"team_id": abbrev, "season": season_str, "stats": stats}
        stored = storage.upsert_nhl_advanced_stats([record])
        return {"status": "ok", "team": abbrev, "stored": stored, "season": season_str}

    # ------------------------------------------------------------------
    # Depth chart (schema: team-depth-chart-v6.0.xsd)
    # ------------------------------------------------------------------

    def fetch_depth_chart(
        self,
        storage: Storage,
        team_alias: str,
        season_year: int = 2025,
    ) -> dict[str, Any]:
        """Fetch line combos and depth ordering. 1 API call per team.

        Schema fields: positions[].players[] with depth ordering, coaches[].
        """
        uuid = self._team_uuid(team_alias)
        if not uuid:
            self._ensure_hierarchy()
            uuid = self._team_uuid(team_alias)
            if not uuid:
                return {"status": "failed", "error": f"Unknown team alias: {team_alias}"}

        data = self._fetch_json(f"teams/{uuid}/depth_chart.json")
        if not data:
            return {"status": "failed", "error": self._last_error or "no depth chart data"}

        season_str = f"{season_year}-{str(season_year + 1)[-2:]}"
        abbrev = self._alias_to_abbrev(team_alias)

        # Build depth info per player, then merge into existing roster entries
        positions_raw = data.get("positions", {})
        positions = positions_raw.values() if isinstance(positions_raw, dict) else positions_raw

        depth_info: dict[str, dict[str, Any]] = {}  # player_id → depth data
        for pos in positions:
            if isinstance(pos, str):
                continue
            pos_name = pos.get("name", "")
            for p in pos.get("players", []):
                pid = p.get("id", "")
                if not pid:
                    continue
                # Only store primary position depth (skip PP1/PP2 etc.)
                if pid not in depth_info or pos_name in ("LW", "C", "RW", "LD", "RD", "G"):
                    depth_info[pid] = {
                        "depth": p.get("depth", 0),
                        "depth_chart_position": pos_name,
                        "jersey_number": p.get("jersey_number", ""),
                        "handedness": p.get("handedness", ""),
                        "status": p.get("status", ""),
                        "full_name": p.get("full_name", ""),
                        "primary_position": p.get("primary_position", pos_name),
                    }

        # Merge into existing roster (preserve stats from statistics endpoint)
        existing = {p["player_id"]: p for p in storage.query_roster(team=abbrev)}
        roster_updates: list[dict[str, Any]] = []

        for pid, info in depth_info.items():
            if pid in existing:
                entry = {**existing[pid]}
                entry["line_number"] = info["depth"]
                entry.setdefault("stats", {})
                entry["stats"]["depth_chart_position"] = info["depth_chart_position"]
                entry["stats"]["depth"] = info["depth"]
                entry["stats"]["handedness"] = info["handedness"]
            else:
                entry = {
                    "player_id": pid,
                    "team": abbrev,
                    "season": season_str,
                    "name": info["full_name"],
                    "position": info["primary_position"],
                    "line_number": info["depth"],
                    "stats": {
                        "depth_chart_position": info["depth_chart_position"],
                        "depth": info["depth"],
                        "jersey_number": info["jersey_number"],
                        "handedness": info["handedness"],
                    },
                    "injury_status": "healthy",
                }
            roster_updates.append(entry)

        updated = 0
        if roster_updates:
            updated = storage.upsert_roster(roster_updates)

        # Coaches
        coaches = []
        for c in data.get("coaches", []):
            coaches.append({
                "name": c.get("full_name", ""),
                "position": c.get("position", ""),
                "experience": c.get("experience", 0),
            })

        return {
            "status": "ok",
            "team": abbrev,
            "roster_updated": updated,
            "coaches": coaches,
            "positions": len(positions),
        }

    # ------------------------------------------------------------------
    # Injuries (schema: injuries-v2.0.xsd + common-v7.0.xsd)
    # ------------------------------------------------------------------

    def fetch_injuries(self, storage: Storage) -> dict[str, Any]:
        """Fetch league-wide injuries. 1 API call.

        Schema fields: injury.status (Unknown/Day To Day/Out/Out For Season/Out Indefinitely),
        injury.desc, injury.comment, injury.start_date, injury.update_date.
        """
        data = self._fetch_json("league/injuries.json")
        if not data:
            return {"status": "failed", "error": self._last_error or "no injury data"}

        # Handle both nested and flat formats
        teams = data.get("teams", [])
        if not teams:
            injuries_node = data.get("injuries", {})
            teams = injuries_node.get("teams", []) if isinstance(injuries_node, dict) else []

        all_injuries: list[dict[str, Any]] = []
        roster_updates = 0

        for team in teams:
            alias = team.get("alias", "")
            abbrev = self._alias_to_abbrev(alias)

            for player in team.get("players", []):
                injuries = player.get("injuries", [])
                if not injuries:
                    continue

                injury = injuries[0]  # Most recent
                sr_status = injury.get("status", "")

                # Map to ABBA format
                if sr_status in ("Out", "Out For Season", "Out Indefinitely"):
                    injury_status = "injured"
                elif sr_status == "Day To Day":
                    injury_status = "day-to-day"
                elif sr_status == "Unknown":
                    injury_status = "questionable"
                else:
                    injury_status = "questionable"

                pid = player.get("id", "")

                all_injuries.append({
                    "player_id": pid,
                    "team": abbrev,
                    "name": player.get("full_name", ""),
                    "position": player.get("primary_position", ""),
                    "status": injury_status,
                    "sr_status": sr_status,
                    "desc": injury.get("desc", ""),
                    "comment": injury.get("comment", ""),
                    "start_date": injury.get("start_date", ""),
                    "update_date": injury.get("update_date", ""),
                })

                # Update roster entry if it exists
                existing = storage.query_roster(team=abbrev)
                for p in existing:
                    if p.get("player_id") == pid:
                        storage.upsert_roster([{
                            **p,
                            "injury_status": injury_status,
                        }])
                        roster_updates += 1
                        break

        return {
            "status": "ok",
            "injury_updates": roster_updates,
            "injuries": all_injuries,
            "teams_with_injuries": len([t for t in teams if t.get("players")]),
        }

    # ------------------------------------------------------------------
    # Transfers (schema: transfers-v7.0.xsd)
    # ------------------------------------------------------------------

    def fetch_transfers(self, storage: Storage | None = None) -> dict[str, Any]:
        """Fetch recent league-wide transfers/transactions. 1 API call.

        Schema fields: transaction_type, transaction_code, effective_date,
        from_team, to_team, notes, desc.
        """
        data = self._fetch_json("league/transfers.json")
        if not data:
            return {"status": "failed", "error": self._last_error or "no transfer data"}

        transfers_data = data.get("transfers", {})
        players = transfers_data.get("players", []) if isinstance(transfers_data, dict) else []

        transactions: list[dict[str, Any]] = []
        for p in players:
            for t in p.get("transfers", []):
                from_team = t.get("from_team", {})
                to_team = t.get("to_team", {})
                transactions.append({
                    "player_id": p.get("id", ""),
                    "player_name": p.get("full_name", ""),
                    "position": p.get("primary_position", ""),
                    "transaction_type": t.get("transaction_type", ""),
                    "transaction_code": t.get("transaction_code", ""),
                    "effective_date": t.get("effective_date", ""),
                    "from_team": self._alias_to_abbrev(from_team.get("alias", "")),
                    "to_team": self._alias_to_abbrev(to_team.get("alias", "")),
                    "notes": t.get("notes", ""),
                    "desc": t.get("desc", ""),
                })

        return {
            "status": "ok",
            "transactions": transactions,
            "count": len(transactions),
        }

    # ------------------------------------------------------------------
    # Full refresh
    # ------------------------------------------------------------------

    def refresh(
        self,
        storage: Storage,
        team: str | None = None,
        season_year: int = 2025,
    ) -> dict[str, Any]:
        """Orchestrated refresh with query-budget awareness.

        Default (no team specified):
          hierarchy(1) + standings(1) + schedule(1) + injuries(1) = 4 queries
          + per-team stats(1) + analytics(1) for teams playing today

        With team specified:
          hierarchy(1) + standings(1) + schedule(1) + stats(1) + analytics(1)
          + depth_chart(1) + injuries(1) = 7 queries
        """
        if not self.api_key:
            return {
                "status": "no_api_key",
                "error": "Set SPORTRADAR_API_KEY env var or pass api_key to constructor.",
            }

        results: dict[str, Any] = {
            "source": "sportradar",
            "fetched_at": datetime.now().isoformat(),
            "queries_used": 0,
        }

        # 0. Hierarchy (1 call, cached after first)
        pre_cached = bool(self._team_ids)
        self._ensure_hierarchy()
        if not pre_cached:
            results["queries_used"] += 1

        # 1. Standings (1 call — all 32 teams)
        results["standings"] = self.fetch_standings(storage, season_year)
        results["queries_used"] += 1

        # 2. Schedule (1 call — today's games)
        results["schedule"] = self.fetch_schedule(storage)
        results["queries_used"] += 1

        # 3. Determine teams for detailed fetch
        teams_to_fetch: set[str] = set()
        if team:
            teams_to_fetch.add(team.upper())
        else:
            scheduled = storage.query_games(sport="NHL", status="scheduled", limit=200)
            for g in scheduled:
                teams_to_fetch.add(g["home_team"])
                teams_to_fetch.add(g["away_team"])

        # 4. Per-team statistics + analytics
        if teams_to_fetch:
            stats_results: dict[str, Any] = {}
            analytics_results: dict[str, Any] = {}
            for t in sorted(teams_to_fetch):
                sr_alias = self._abbrev_to_alias(t)
                stats_results[t] = self.fetch_team_statistics(storage, sr_alias, season_year)
                results["queries_used"] += 1
                analytics_results[t] = self.fetch_analytics(storage, sr_alias, season_year)
                results["queries_used"] += 1
            results["team_statistics"] = stats_results
            results["analytics"] = analytics_results

        # 5. Depth chart (only for single-team refresh)
        if team:
            sr_alias = self._abbrev_to_alias(team.upper())
            results["depth_chart"] = self.fetch_depth_chart(storage, sr_alias, season_year)
            results["queries_used"] += 1

        # 6. Injuries (1 call — league-wide)
        results["injuries"] = self.fetch_injuries(storage)
        results["queries_used"] += 1

        return results
