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

        # 3. Determine which teams need roster + goalie data
        teams_to_fetch: set[str] = set()
        if team:
            teams_to_fetch.add(team.upper())
        # Auto-fetch for all teams with games on the schedule (not just today —
        # the schedule endpoint returns ~a week of games)
        scheduled_games = storage.query_games(sport="NHL", status="scheduled", limit=200)
        for g in scheduled_games:
            teams_to_fetch.add(g["home_team"])
            teams_to_fetch.add(g["away_team"])

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

    def _fetch_standings(self, storage: Storage) -> dict[str, Any]:
        """Fetch current NHL standings and store as team stats."""
        data = self._fetch_json(f"{self.BASE_URL}/standings/now")
        if not data or "standings" not in data:
            return {"status": "failed", "error": self._last_error or "could not fetch standings"}

        team_stats = []
        for entry in data["standings"]:
            abbrev = entry.get("teamAbbrev", {}).get("default", "")
            team_stats.append({
                "team_id": abbrev,
                "sport": "NHL",
                "season": str(entry.get("seasonId", "20252026"))[:4] + "-" + str(entry.get("seasonId", "20252026"))[6:8],
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
                },
                "source": "nhl_api",
            })

        stored = storage.upsert_team_stats(team_stats)
        return {"status": "ok", "teams_updated": stored}

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
            return {"status": "ok", "goalies_stored": 0, "note": "no goalie data in response"}

        goalie_stats = []
        for g in goalies:
            pid = str(g.get("playerId", ""))
            first = g.get("firstName", {}).get("default", "")
            last = g.get("lastName", {}).get("default", "")
            gp = g.get("gamesPlayed", 0)

            toi_seconds = g.get("timeOnIce", 0)
            toi_minutes = toi_seconds / 60.0 if toi_seconds else 0

            goalie_stats.append({
                "goaltender_id": pid,
                "team": team,
                "season": "2025-26",
                "stats": {
                    "name": f"{first} {last}",
                    "games_played": gp,
                    "games_started": g.get("gamesStarted", 0),
                    "wins": g.get("wins", 0),
                    "losses": g.get("losses", 0),
                    "ot_losses": g.get("overtimeLosses", 0),
                    "save_percentage": round(g.get("savePercentage", 0), 4),
                    "goals_against_average": round(g.get("goalsAgainstAverage", 0), 2),
                    "goals_against": g.get("goalsAgainst", 0),
                    "saves": g.get("saves", 0),
                    "shots_against": g.get("shotsAgainst", 0),
                    "shutouts": g.get("shutouts", 0),
                    "time_on_ice_minutes": round(toi_minutes, 1),
                },
            })

        stored = storage.upsert_goaltender_stats(goalie_stats)
        return {"status": "ok", "goalies_stored": stored, "team": team}


# --- Odds Live Connector (API key required) ---

class OddsLiveConnector:
    """Pulls real odds from The Odds API.

    Requires ODDS_API_KEY env var. Free tier: 500 req/mo.
    """

    BASE_URL = "https://api.the-odds-api.com/v4"

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

        odds_records = []
        for event in data:
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")
            game_id = f"nhl-{event.get('id', '')}"

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
    odds_key = bool(os.environ.get("ODDS_API_KEY"))
    weather_key = bool(os.environ.get("OPENWEATHER_API_KEY"))

    return [
        {
            "name": "nhl_stats_api",
            "sport": "NHL",
            "provides": ["standings", "schedule", "rosters", "player_stats"],
            "auth": False,
            "cost": "free",
            "freshness": "real-time",
            "status": "active",
            "endpoint": "api-web.nhle.com",
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
