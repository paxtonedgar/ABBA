"""Live data connectors that pull from real APIs.

These connectors solve the core problem: LLMs hallucinate sports data.
They get rosters wrong, records wrong, outcomes wrong. ABBA provides
a live data layer that agents query instead of guessing.

Each connector fetches from a real API and writes to the DuckDB store.
Every response includes freshness metadata so agents know data staleness.

Sources:
- NHL Stats API (api-web.nhle.com) -- FREE, no auth, real-time
- MLB Stats API (statsapi.mlb.com) -- FREE, no auth, real-time
- The Odds API (the-odds-api.com) -- API key required, 500 req/mo free
- OpenWeather (openweathermap.org) -- API key required, 1000/day free
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, date
from typing import Any

from ..storage import Storage


# --- NHL Live Connector (FREE, no auth) ---

class NHLLiveConnector:
    """Pulls real data from the NHL Stats API.

    api-web.nhle.com/v1/ -- free, no auth, official NHL data.
    Standings, schedules, rosters, player stats -- all real.
    """

    BASE_URL = "https://api-web.nhle.com/v1"
    STATS_BASE_URL = "https://api.nhle.com/stats/rest/en/team"

    # NHL team abbreviation -> full name mapping
    TEAMS = {
        "ANA": "Anaheim Ducks", "ARI": "Arizona Coyotes", "BOS": "Boston Bruins",
        "BUF": "Buffalo Sabres", "CGY": "Calgary Flames", "CAR": "Carolina Hurricanes",
        "CHI": "Chicago Blackhawks", "COL": "Colorado Avalanche", "CBJ": "Columbus Blue Jackets",
        "DAL": "Dallas Stars", "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers",
        "FLA": "Florida Panthers", "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild",
        "MTL": "Montreal Canadiens", "NSH": "Nashville Predators", "NJD": "New Jersey Devils",
        "NYI": "New York Islanders", "NYR": "New York Rangers", "OTT": "Ottawa Senators",
        "PHI": "Philadelphia Flyers", "PIT": "Pittsburgh Penguins", "SJS": "San Jose Sharks",
        "SEA": "Seattle Kraken", "STL": "St. Louis Blues", "TBL": "Tampa Bay Lightning",
        "TOR": "Toronto Maple Leafs", "UTA": "Utah Hockey Club", "VAN": "Vancouver Canucks",
        "VGK": "Vegas Golden Knights", "WSH": "Washington Capitals", "WPG": "Winnipeg Jets",
    }
    FULL_NAME_TO_ABBREV = {full_name: abbrev for abbrev, full_name in TEAMS.items()}
    FULL_NAME_TO_ABBREV.update({
        "Montréal Canadiens": "MTL",
        "Utah Mammoth": "UTA",
    })

    def _fetch_json(self, url: str) -> dict[str, Any] | list[Any] | None:
        """Fetch JSON from URL. Returns None on failure, stores last error."""
        self._last_error: str | None = None
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ABBA/2.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            self._last_error = f"HTTP {e.code}: {e.reason} ({url})"
            return None
        except urllib.error.URLError as e:
            self._last_error = f"URL error: {e.reason} ({url})"
            return None
        except json.JSONDecodeError as e:
            self._last_error = f"JSON decode error: {e} ({url})"
            return None
        except TimeoutError:
            self._last_error = f"Timeout after 15s ({url})"
            return None

    @staticmethod
    def _season_to_stats_season_id(season: str) -> str | None:
        """Convert YYYY-YY season string to NHL stats seasonId form."""
        try:
            start_year, end_suffix = season.split("-")
            end_year = start_year[:2] + end_suffix
            return f"{start_year}{end_year}"
        except (ValueError, IndexError):
            return None

    def _fetch_special_teams_by_team(self, season: str) -> dict[str, dict[str, float]]:
        """Fetch current-season PP/PK from the NHL stats REST endpoint."""
        season_id = self._season_to_stats_season_id(season)
        if not season_id:
            return {}

        query = urllib.parse.urlencode({"cayenneExp": f"seasonId={season_id} and gameTypeId=2"})
        data = self._fetch_json(f"{self.STATS_BASE_URL}/summary?{query}")
        if not isinstance(data, dict):
            return {}

        stats_by_team: dict[str, dict[str, float]] = {}
        for row in data.get("data", []):
            full_name = row.get("teamFullName")
            abbrev = self.FULL_NAME_TO_ABBREV.get(full_name, "")
            if not abbrev:
                continue
            pp_pct_raw = row.get("powerPlayPct")
            pk_pct_raw = row.get("penaltyKillPct")
            faceoff_pct_raw = row.get("faceoffWinPct")

            pp_pct = float(pp_pct_raw) * 100 if pp_pct_raw is not None else 22.0
            pk_pct = float(pk_pct_raw) * 100 if pk_pct_raw is not None else 80.0
            faceoff_pct = float(faceoff_pct_raw) * 100 if faceoff_pct_raw is not None else None

            stats_by_team[abbrev] = {
                "power_play_percentage": round(pp_pct, 2),
                "penalty_kill_percentage": round(pk_pct, 2),
                **({"faceoff_win_percentage": round(faceoff_pct, 2)} if faceoff_pct is not None else {}),
            }
        return stats_by_team

    def refresh(self, storage: Storage, team: str | None = None) -> dict[str, Any]:
        """Pull fresh data from NHL API and store it.

        Fetches standings, schedule, and goaltender stats + rosters for
        teams with games on the schedule (or a specific team if provided).
        """
        results: dict[str, Any] = {"source": "nhl_stats_api", "fetched_at": datetime.now().isoformat()}

        # 1. Standings (all teams)
        standings = self._fetch_standings(storage)
        results["standings"] = standings

        # 2. Today's schedule
        schedule = self._fetch_schedule(storage)
        results["schedule"] = schedule

        # 2b. For live/final games, infer the actual goalie in net from gamecenter
        # play-by-play and persist that as a per-game override.
        confirmed_goalies = self._refresh_confirmed_goalies(storage)
        if confirmed_goalies["games_updated"] or confirmed_goalies["games_checked"]:
            results["confirmed_goalies"] = confirmed_goalies

        # 3. Determine which teams need roster + goalie data
        teams_to_fetch: set[str] = set()
        if team:
            teams_to_fetch.add(team.upper())
        # Auto-fetch for all teams with games on the schedule
        scheduled_games = storage.query_games(sport="NHL", status="scheduled", limit=200)
        for g in scheduled_games:
            teams_to_fetch.add(g["home_team"])
            teams_to_fetch.add(g["away_team"])

        # Cold-start guard — if no scheduled games yet (first refresh),
        # fetch all 32 NHL teams so goalie/roster data is never empty
        if not teams_to_fetch:
            teams_to_fetch = set(self.TEAMS.keys())

        # 4. Fetch rosters and goaltender stats for those teams
        if teams_to_fetch:
            roster_results = {}
            goalie_results = {}
            for t in sorted(teams_to_fetch):
                roster_results[t] = self._fetch_roster(storage, t)
                goalie_results[t] = self._fetch_goaltender_stats(storage, t)
            results["rosters"] = roster_results
            results["goaltender_stats"] = goalie_results

        return results

    @staticmethod
    def _merge_metadata(existing: Any, updates: dict[str, Any]) -> dict[str, Any]:
        """Merge game metadata dictionaries without dropping existing fields."""
        base = existing if isinstance(existing, dict) else {}
        return {**base, **updates}

    @staticmethod
    def _toi_to_seconds(toi: str | None) -> int:
        """Convert MM:SS or HH:MM:SS strings to seconds."""
        if not toi:
            return 0
        try:
            parts = [int(part) for part in toi.split(":")]
        except ValueError:
            return 0
        if len(parts) == 2:
            minutes, seconds = parts
            return minutes * 60 + seconds
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return hours * 3600 + minutes * 60 + seconds
        return 0

    def _extract_confirmed_goalies_from_play_by_play(self, payload: dict[str, Any]) -> dict[str, str]:
        """Infer current goalies from earliest play-by-play events with goalieInNetId."""
        goalie_team: dict[str, str] = {}
        for roster_spot in payload.get("rosterSpots", []):
            if roster_spot.get("positionCode") == "G":
                goalie_team[str(roster_spot.get("playerId", ""))] = str(roster_spot.get("teamId", ""))

        starters: dict[str, str] = {}
        for play in payload.get("plays", []):
            goalie_id = play.get("details", {}).get("goalieInNetId")
            if goalie_id is None:
                continue
            goalie_id_str = str(goalie_id)
            team_id = goalie_team.get(goalie_id_str)
            if not team_id or team_id in starters:
                continue
            starters[team_id] = goalie_id_str
            if len(starters) >= 2:
                break
        return starters

    def _extract_confirmed_goalies_from_boxscore(self, payload: dict[str, Any]) -> dict[str, str]:
        """Fallback: infer goalies in use from boxscore TOI/shots faced."""
        starters: dict[str, str] = {}
        for side in ("homeTeam", "awayTeam"):
            goalies = payload.get("playerByGameStats", {}).get(side, {}).get("goalies", [])
            best_goalie = None
            best_signal = -1
            for goalie in goalies:
                signal = max(
                    self._toi_to_seconds(goalie.get("toi")),
                    int(goalie.get("shotsAgainst", 0) or 0),
                    int(goalie.get("saves", 0) or 0),
                )
                if signal > best_signal:
                    best_signal = signal
                    best_goalie = goalie
            team_id = payload.get(side, {}).get("id")
            if best_goalie and team_id is not None and best_signal > 0:
                starters[str(team_id)] = str(best_goalie.get("playerId", ""))
        return starters

    def _fetch_confirmed_goalies_for_game(self, game_id: str) -> dict[str, Any] | None:
        """Fetch explicit in-net goalie IDs for a live/final game if the feed exposes them."""
        raw_game_id = game_id.removeprefix("nhl-")

        play_payload = self._fetch_json(f"{self.BASE_URL}/gamecenter/{raw_game_id}/play-by-play")
        if isinstance(play_payload, dict):
            from_play = self._extract_confirmed_goalies_from_play_by_play(play_payload)
            if len(from_play) >= 2:
                return {
                    "team_goalies": from_play,
                    "goalie_source": "play_by_play",
                }

        box_payload = self._fetch_json(f"{self.BASE_URL}/gamecenter/{raw_game_id}/boxscore")
        if isinstance(box_payload, dict):
            from_box = self._extract_confirmed_goalies_from_boxscore(box_payload)
            if len(from_box) >= 2:
                return {
                    "team_goalies": from_box,
                    "goalie_source": "boxscore",
                }

        return None

    def _refresh_confirmed_goalies(self, storage: Storage) -> dict[str, Any]:
        """Persist per-game goalie overrides for today's live/final games."""
        today = date.today().isoformat()
        games = storage.query_games(sport="NHL", date=today, limit=200)
        target_games = [g for g in games if g.get("status") in ("live", "final")]
        updated = 0
        details: list[dict[str, Any]] = []

        for game in target_games:
            inferred = self._fetch_confirmed_goalies_for_game(game["game_id"])
            if not inferred:
                details.append({"game_id": game["game_id"], "status": "unresolved"})
                continue

            home_team_id = str(game.get("home_team", ""))
            away_team_id = str(game.get("away_team", ""))
            team_goalies = inferred["team_goalies"]
            # Map NHL numeric team ids from the feed back to stored abbreviations.
            # We use the play-by-play payload only to identify goalie ids; the
            # abbreviations still come from the schedule row.
            if len(team_goalies) != 2:
                details.append({"game_id": game["game_id"], "status": "partial"})
                continue

            # Resolve home/away goalie ids by looking at actual roster rows for the teams.
            home_goalies = {
                str(goalie["goaltender_id"])
                for goalie in storage.query_goaltender_stats(team=home_team_id, season="2025-26")
            }
            away_goalies = {
                str(goalie["goaltender_id"])
                for goalie in storage.query_goaltender_stats(team=away_team_id, season="2025-26")
            }
            home_goalie_id = next((gid for gid in team_goalies.values() if gid in home_goalies), None)
            away_goalie_id = next((gid for gid in team_goalies.values() if gid in away_goalies), None)
            if not home_goalie_id or not away_goalie_id:
                details.append({"game_id": game["game_id"], "status": "unmatched"})
                continue

            metadata = self._merge_metadata(game.get("metadata"), {
                "home_goalie_id": home_goalie_id,
                "away_goalie_id": away_goalie_id,
                "goalie_source": inferred["goalie_source"],
                "goalie_confirmed_at": datetime.now().isoformat(),
            })
            storage.upsert_games([{
                "game_id": game["game_id"],
                "sport": game["sport"],
                "date": game["date"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_score": game.get("home_score"),
                "away_score": game.get("away_score"),
                "venue": game.get("venue"),
                "status": game.get("status", "scheduled"),
                "metadata": metadata,
                "source": game.get("source", "nhl_api"),
            }])
            updated += 1
            details.append({
                "game_id": game["game_id"],
                "status": "ok",
                "home_goalie_id": home_goalie_id,
                "away_goalie_id": away_goalie_id,
                "source": inferred["goalie_source"],
            })

        return {
            "games_checked": len(target_games),
            "games_updated": updated,
            "details": details,
        }

    def _fetch_standings(self, storage: Storage) -> dict[str, Any]:
        """Fetch current NHL standings and store as team stats."""
        data = self._fetch_json(f"{self.BASE_URL}/standings/now")
        if not data or "standings" not in data:
            return {"status": "failed", "error": self._last_error or "could not fetch standings"}

        season_str = None
        if data["standings"]:
            season_id = str(data["standings"][0].get("seasonId", "20252026"))
            season_str = f"{season_id[:4]}-{season_id[6:8]}"
        special_teams = self._fetch_special_teams_by_team(season_str or "2025-26")

        team_stats = []
        for entry in data["standings"]:
            abbrev = entry.get("teamAbbrev", {}).get("default", "")
            entry_season_id = str(entry.get("seasonId", "20252026"))
            team_special = special_teams.get(abbrev, {})
            team_stats.append({
                "team_id": abbrev,
                "sport": "NHL",
                "season": entry_season_id[:4] + "-" + entry_season_id[6:8],
                "stats": {
                    "wins": entry.get("wins", 0),
                    "losses": entry.get("losses", 0),
                    "overtime_losses": entry.get("otLosses", 0),
                    "points": entry.get("points", 0),
                    "goals_for": entry.get("goalFor", 0),
                    "goals_against": entry.get("goalAgainst", 0),
                    "goal_differential": entry.get("goalDifferential", 0),
                    "games_played": entry.get("gamesPlayed", 0),
                    "regulation_wins": entry.get("regulationWins", 0),
                    "streak_code": entry.get("streakCode", ""),
                    "streak_count": entry.get("streakCount", 0),
                    "home_wins": entry.get("homeWins", 0),
                    "home_losses": entry.get("homeLosses", 0),
                    "home_ot_losses": entry.get("homeOtLosses", 0),
                    "road_wins": entry.get("roadWins", 0),
                    "road_losses": entry.get("roadLosses", 0),
                    "road_ot_losses": entry.get("roadOtLosses", 0),
                    "l10_wins": entry.get("l10Wins", 0),
                    "l10_losses": entry.get("l10Losses", 0),
                    "l10_ot_losses": entry.get("l10OtLosses", 0),
                    "division_name": entry.get("divisionName", ""),
                    "conference_name": entry.get("conferenceName", ""),
                    "team_name": entry.get("teamName", {}).get("default", ""),
                    "wild_card_sequence": entry.get("wildcardSequence", 0),
                    "points_pct": entry.get("pointPctg", 0),
                    "recent_form": entry.get("l10Wins", 0) / max(entry.get("l10Wins", 0) + entry.get("l10Losses", 0) + entry.get("l10OtLosses", 0), 1),
                    "power_play_percentage": team_special.get("power_play_percentage", 22.0),
                    "penalty_kill_percentage": team_special.get("penalty_kill_percentage", 80.0),
                    **({"faceoff_win_percentage": team_special["faceoff_win_percentage"]} if "faceoff_win_percentage" in team_special else {}),
                },
                "source": "nhl_api",
            })

        stored = storage.upsert_team_stats(team_stats)
        teams_with_special_teams = sum(
            1
            for team in team_stats
            if "power_play_percentage" in team["stats"] and "penalty_kill_percentage" in team["stats"]
        )
        return {
            "status": "ok",
            "teams_updated": stored,
            "special_teams_teams_updated": teams_with_special_teams,
        }

    def _fetch_schedule(self, storage: Storage, game_date: str | None = None) -> dict[str, Any]:
        """Fetch NHL schedule for today (or specified date) and store games."""
        if game_date is None:
            game_date = date.today().isoformat()

        data = self._fetch_json(f"{self.BASE_URL}/schedule/{game_date}")
        if not data or "gameWeek" not in data:
            return {"status": "failed", "error": self._last_error or "could not fetch schedule"}

        games = []
        for day_data in data.get("gameWeek", []):
            for game in day_data.get("games", []):
                game_id = f"nhl-{game.get('id', '')}"
                home_abbrev = game.get("homeTeam", {}).get("abbrev", "")
                away_abbrev = game.get("awayTeam", {}).get("abbrev", "")
                game_state = game.get("gameState", "")

                status = "scheduled"
                if game_state in ("LIVE", "CRIT"):
                    status = "live"
                elif game_state in ("FINAL", "OFF"):
                    status = "final"

                game_entry = {
                    "game_id": game_id,
                    "sport": "NHL",
                    "date": day_data.get("date", game_date),
                    "home_team": home_abbrev,
                    "away_team": away_abbrev,
                    "venue": game.get("venue", {}).get("default", ""),
                    "status": status,
                    "metadata": {
                        "start_time": game.get("startTimeUTC", ""),
                        "game_type": game.get("gameType", 0),
                        "tv_broadcasts": [b.get("network", "") for b in game.get("tvBroadcasts", [])],
                    },
                    "source": "nhl_api",
                }

                if status == "final":
                    game_entry["home_score"] = game.get("homeTeam", {}).get("score", 0)
                    game_entry["away_score"] = game.get("awayTeam", {}).get("score", 0)

                games.append(game_entry)

        stored = storage.upsert_games(games)
        return {"status": "ok", "games_stored": stored}

    def _fetch_roster(self, storage: Storage, team: str) -> dict[str, Any]:
        """Fetch current roster for a team and store it."""
        data = self._fetch_json(f"{self.BASE_URL}/roster/{team}/current")
        if not data:
            return {"status": "failed", "error": self._last_error or f"could not fetch roster for {team}"}

        players = []
        for position_group in ["forwards", "defensemen", "goalies"]:
            for player in data.get(position_group, []):
                pid = str(player.get("id", ""))
                first = player.get("firstName", {}).get("default", "")
                last = player.get("lastName", {}).get("default", "")
                pos = player.get("positionCode", "")

                players.append({
                    "player_id": pid,
                    "team": team,
                    "season": "2025-26",
                    "name": f"{first} {last}",
                    "position": pos,
                    "stats": {
                        "sweater_number": player.get("sweaterNumber", 0),
                        "shoots_catches": player.get("shootsCatches", ""),
                        "height_inches": player.get("heightInInches", 0),
                        "weight_pounds": player.get("weightInPounds", 0),
                        "birth_date": player.get("birthDate", ""),
                        "birth_country": player.get("birthCountry", ""),
                    },
                    "injury_status": "healthy",
                })

        stored = storage.upsert_roster(players)
        return {"status": "ok", "players_stored": stored, "team": team}

    def _fetch_goaltender_stats(self, storage: Storage, team: str) -> dict[str, Any]:
        """Fetch goaltender season stats for a team from the NHL club stats endpoint."""
        data = self._fetch_json(f"{self.BASE_URL}/club-stats/{team}/now")
        if not data:
            return {"status": "failed", "error": self._last_error or f"could not fetch club stats for {team}"}

        goalies = data.get("goalies", [])
        if not goalies:
            # Flag empty goalie response as warning, not success
            return {"status": "warning", "goalies_stored": 0, "warning": f"NHL API returned no goalie data for {team}"}

        goalie_stats = []
        for g in goalies:
            pid = str(g.get("playerId", ""))
            first = g.get("firstName", {}).get("default", "")
            last = g.get("lastName", {}).get("default", "")
            gp = g.get("gamesPlayed", 0)

            toi_seconds = g.get("timeOnIce", 0)
            toi_minutes = toi_seconds / 60.0 if toi_seconds else 0

            # Compute derived fields the model requires
            goals_against = g.get("goalsAgainst", 0)
            saves = g.get("saves", 0)
            shots_against = g.get("shotsAgainst", 0)
            save_pct = round(g.get("savePercentage", 0), 4)
            gaa = round(g.get("goalsAgainstAverage", 0), 2)
            # GSAA = (shots_against * league_avg_sv_pct) - goals_against
            gsaa = round(shots_against * 0.907 - goals_against, 2) if shots_against > 0 else 0.0
            games_started = g.get("gamesStarted", 0)

            goalie_stats.append({
                "goaltender_id": pid,
                "team": team,
                "season": "2025-26",
                "stats": {
                    "name": f"{first} {last}",
                    "games_played": gp,
                    "games_started": games_started,
                    "wins": g.get("wins", 0),
                    "losses": g.get("losses", 0),
                    "ot_losses": g.get("overtimeLosses", 0),
                    # Model-compatible keys (save_pct not save_percentage)
                    "save_pct": save_pct,
                    "gaa": gaa,
                    "gsaa": gsaa,
                    # Keep originals for transparency
                    "save_percentage": save_pct,
                    "goals_against_average": gaa,
                    "goals_against": goals_against,
                    "saves": saves,
                    "shots_against": shots_against,
                    "shutouts": g.get("shutouts", 0),
                    "time_on_ice_minutes": round(toi_minutes, 1),
                },
            })

        # Infer role from games_started — goalie with most starts is "starter"
        if goalie_stats:
            max_gs = max(g["stats"].get("games_started", 0) for g in goalie_stats)
            for g in goalie_stats:
                if g["stats"].get("games_started", 0) == max_gs and max_gs > 0:
                    g["stats"]["role"] = "starter"
                    break  # only tag one starter

        stored = storage.upsert_goaltender_stats(goalie_stats)
        return {"status": "ok", "goalies_stored": stored, "team": team}


# --- Odds Live Connector (API key required) ---

class OddsLiveConnector:
    """Pulls real odds from The Odds API.

    Requires ODDS_API_KEY env var. Free tier: 500 req/mo.
    """

    BASE_URL = "https://api.the-odds-api.com/v4"

    # Map The Odds API full team names → NHL 3-letter abbreviations.
    # Used to resolve game_id by matching against stored schedule.
    _ODDS_NAME_TO_ABBREV: dict[str, str] = {
        "Anaheim Ducks": "ANA", "Arizona Coyotes": "ARI", "Utah Hockey Club": "UTA",
        "Boston Bruins": "BOS", "Buffalo Sabres": "BUF",
        "Calgary Flames": "CGY", "Carolina Hurricanes": "CAR",
        "Chicago Blackhawks": "CHI", "Colorado Avalanche": "COL",
        "Columbus Blue Jackets": "CBJ", "Dallas Stars": "DAL",
        "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM",
        "Florida Panthers": "FLA", "Los Angeles Kings": "LAK",
        "Minnesota Wild": "MIN", "Montreal Canadiens": "MTL",
        "Montréal Canadiens": "MTL",
        "Nashville Predators": "NSH", "New Jersey Devils": "NJD",
        "New York Islanders": "NYI", "New York Rangers": "NYR",
        "Ottawa Senators": "OTT", "Philadelphia Flyers": "PHI",
        "Pittsburgh Penguins": "PIT", "San Jose Sharks": "SJS",
        "Seattle Kraken": "SEA", "St Louis Blues": "STL",
        "St. Louis Blues": "STL",
        "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR",
        "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK",
        "Washington Capitals": "WSH", "Winnipeg Jets": "WPG",
    }

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY", "")

    def refresh(self, storage: Storage, sport: str = "icehockey_nhl") -> dict[str, Any]:
        """Fetch current odds and store them."""
        if not self.api_key:
            return {
                "status": "no_api_key",
                "error": "Set ODDS_API_KEY environment variable. Free tier: 500 req/mo at the-odds-api.com",
            }

        url = (
            f"{self.BASE_URL}/sports/{sport}/odds"
            f"?apiKey={self.api_key}"
            f"&regions=us"
            f"&markets=h2h,spreads,totals"
            f"&oddsFormat=decimal"
        )

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ABBA/2.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                remaining = resp.headers.get("x-requests-remaining", "unknown")
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            return {"status": "failed", "error": str(e)}

        # Build lookup from (home_abbrev, away_abbrev, date) → game_id
        # so we can resolve odds events to schedule game_ids.
        schedule_games = storage.query_games(sport="NHL", status="scheduled")
        schedule_lookup: dict[tuple[str, str, str], str] = {}
        for g in schedule_games:
            gdate = str(g.get("date", ""))[:10]
            schedule_lookup[(g.get("home_team", ""), g.get("away_team", ""), gdate)] = g["game_id"]

        odds_records = []
        resolved_count = 0
        for event in data:
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")

            # Resolve game_id via schedule match instead of using Odds API's own id
            home_abbrev = self._ODDS_NAME_TO_ABBREV.get(home_team, "")
            away_abbrev = self._ODDS_NAME_TO_ABBREV.get(away_team, "")
            commence = event.get("commence_time", "")[:10]  # ISO date portion
            game_id = schedule_lookup.get((home_abbrev, away_abbrev, commence))
            if game_id:
                resolved_count += 1
            else:
                # Fallback: use Odds API id (won't join to schedule, but data isn't lost)
                game_id = f"odds-{event.get('id', '')}"

            for bookmaker in event.get("bookmakers", []):
                book_name = bookmaker.get("title", "unknown")
                for market in bookmaker.get("markets", []):
                    market_key = market.get("key", "")
                    outcomes = {o.get("name", ""): o.get("price", 0) for o in market.get("outcomes", [])}

                    record = {
                        "game_id": game_id,
                        "sportsbook": book_name,
                        "market_type": market_key,
                    }

                    if market_key == "h2h":
                        record["home_odds"] = outcomes.get(home_team, 0)
                        record["away_odds"] = outcomes.get(away_team, 0)
                    elif market_key == "spreads":
                        for o in market.get("outcomes", []):
                            if o.get("name") == home_team:
                                record["spread"] = o.get("point", 0)
                                record["home_odds"] = o.get("price", 0)
                            else:
                                record["away_odds"] = o.get("price", 0)
                    elif market_key == "totals":
                        for o in market.get("outcomes", []):
                            if o.get("name") == "Over":
                                record["total"] = o.get("point", 0)
                                record["over_odds"] = o.get("price", 0)
                            else:
                                record["under_odds"] = o.get("price", 0)

                    odds_records.append(record)

        stored = storage.insert_odds(odds_records)
        return {
            "status": "ok",
            "odds_stored": stored,
            "events": len(data),
            "events_resolved": resolved_count,
            "events_unresolved": len(data) - resolved_count,
            "requests_remaining": remaining,
        }


# --- MLB Stats Connector (FREE, no auth) ---

class MLBLiveConnector:
    """Pulls real data from MLB Stats API (statsapi.mlb.com). Free, no auth."""

    BASE_URL = "https://statsapi.mlb.com/api/v1"

    def _fetch_json(self, url: str) -> dict[str, Any] | None:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ABBA/2.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            return None

    def refresh(self, storage: Storage, team: str | None = None) -> dict[str, Any]:
        """Fetch MLB standings and schedule."""
        results: dict[str, Any] = {"source": "mlb_stats_api", "fetched_at": datetime.now().isoformat()}

        # Standings
        data = self._fetch_json(f"{self.BASE_URL}/standings?leagueId=103,104&season=2026")
        if data and "records" in data:
            team_stats = []
            for division in data["records"]:
                for entry in division.get("teamRecords", []):
                    abbrev = entry.get("team", {}).get("abbreviation", "")
                    team_stats.append({
                        "team_id": abbrev,
                        "sport": "MLB",
                        "season": "2026",
                        "stats": {
                            "wins": entry.get("wins", 0),
                            "losses": entry.get("losses", 0),
                            "win_percentage": float(entry.get("winningPercentage", ".500")),
                            "runs_scored": entry.get("runsScored", 0),
                            "runs_allowed": entry.get("runsAllowed", 0),
                            "run_differential": entry.get("runDifferential", 0),
                            "streak": entry.get("streak", {}).get("streakCode", ""),
                            "division": division.get("division", {}).get("name", ""),
                        },
                    })
            stored = storage.upsert_team_stats(team_stats)
            results["standings"] = {"status": "ok", "teams_updated": stored}
        else:
            results["standings"] = {"status": "failed"}

        # Schedule (today)
        today = date.today().isoformat()
        sched = self._fetch_json(f"{self.BASE_URL}/schedule?sportId=1&date={today}")
        if sched and "dates" in sched:
            games = []
            for d in sched["dates"]:
                for g in d.get("games", []):
                    status_code = g.get("status", {}).get("statusCode", "")
                    status = "final" if status_code == "F" else "live" if status_code in ("I", "PW") else "scheduled"
                    game_entry = {
                        "game_id": f"mlb-{g.get('gamePk', '')}",
                        "sport": "MLB",
                        "date": d.get("date", today),
                        "home_team": g.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", ""),
                        "away_team": g.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", ""),
                        "venue": g.get("venue", {}).get("name", ""),
                        "status": status,
                    }
                    if status == "final":
                        game_entry["home_score"] = g.get("teams", {}).get("home", {}).get("score", 0)
                        game_entry["away_score"] = g.get("teams", {}).get("away", {}).get("score", 0)
                    games.append(game_entry)
            stored = storage.upsert_games(games)
            results["schedule"] = {"status": "ok", "games_stored": stored}
        else:
            results["schedule"] = {"status": "no_games_today"}

        return results


def list_connectors() -> list[dict[str, Any]]:
    """List all available data connectors and their status."""
    sr_key = bool(os.environ.get("SPORTRADAR_API_KEY"))
    odds_key = bool(os.environ.get("ODDS_API_KEY"))
    weather_key = bool(os.environ.get("OPENWEATHER_API_KEY"))

    return [
        {
            "name": "sportradar",
            "sport": "NHL",
            "provides": ["standings", "schedule", "goaltender_stats", "advanced_stats (Corsi, Fenwick, PDO)", "injuries"],
            "auth": True,
            "cost": "trial: 1000 req/mo",
            "freshness": "real-time",
            "status": "active" if sr_key else "needs SPORTRADAR_API_KEY env var",
            "endpoint": "api.sportradar.com",
            "note": "Comprehensive — replaces nhl_stats_api + moneypuck when active",
        },
        {
            "name": "nhl_stats_api",
            "sport": "NHL",
            "provides": ["standings", "schedule", "rosters", "player_stats"],
            "auth": False,
            "cost": "free",
            "freshness": "real-time",
            "status": "active (fallback)" if sr_key else "active",
            "endpoint": "api-web.nhle.com",
        },
        {
            "name": "moneypuck",
            "sport": "NHL",
            "provides": ["advanced_stats (Corsi, Fenwick, xG, PDO, shooting%)"],
            "auth": False,
            "cost": "free",
            "freshness": "daily",
            "status": "active",
            "endpoint": "moneypuck.com",
        },
        {
            "name": "mlb_stats_api",
            "sport": "MLB",
            "provides": ["standings", "schedule", "rosters"],
            "auth": False,
            "cost": "free",
            "freshness": "real-time",
            "status": "active",
            "endpoint": "statsapi.mlb.com",
        },
        {
            "name": "the_odds_api",
            "sport": "MLB, NHL, NBA, NFL",
            "provides": ["odds (h2h, spreads, totals)"],
            "auth": True,
            "cost": "free tier (500 req/mo)",
            "freshness": "30 seconds",
            "status": "active" if odds_key else "needs ODDS_API_KEY env var",
            "endpoint": "api.the-odds-api.com",
        },
        {
            "name": "openweather",
            "sport": "all outdoor",
            "provides": ["weather"],
            "auth": True,
            "cost": "free tier (1000/day)",
            "freshness": "hourly",
            "status": "active" if weather_key else "needs OPENWEATHER_API_KEY env var",
            "endpoint": "openweathermap.org",
        },
    ]
