"""Seed data connector.

Populates the DuckDB store with realistic sample data for development
and demonstration. In production, this would be replaced by real API
connectors (MLB Stats API, The Odds API, OpenWeather, etc.).

The seed data is deterministic (fixed random seed) so tests are reproducible.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np

from ..storage import Storage

# Real MLB teams with 2024-ish realistic stats
MLB_TEAMS = {
    "NYY": {"name": "New York Yankees", "venue": "Yankee Stadium"},
    "BOS": {"name": "Boston Red Sox", "venue": "Fenway Park"},
    "LAD": {"name": "Los Angeles Dodgers", "venue": "Dodger Stadium"},
    "HOU": {"name": "Houston Astros", "venue": "Minute Maid Park"},
    "ATL": {"name": "Atlanta Braves", "venue": "Truist Park"},
    "PHI": {"name": "Philadelphia Phillies", "venue": "Citizens Bank Park"},
    "SD": {"name": "San Diego Padres", "venue": "Petco Park"},
    "CHC": {"name": "Chicago Cubs", "venue": "Wrigley Field"},
    "SEA": {"name": "Seattle Mariners", "venue": "T-Mobile Park"},
    "BAL": {"name": "Baltimore Orioles", "venue": "Camden Yards"},
}

NHL_TEAMS = {
    "NYR": {"name": "New York Rangers", "venue": "Madison Square Garden"},
    "BOS": {"name": "Boston Bruins", "venue": "TD Garden"},
    "CAR": {"name": "Carolina Hurricanes", "venue": "PNC Arena"},
    "COL": {"name": "Colorado Avalanche", "venue": "Ball Arena"},
    "DAL": {"name": "Dallas Stars", "venue": "American Airlines Center"},
    "FLA": {"name": "Florida Panthers", "venue": "Amerant Bank Arena"},
    "VGK": {"name": "Vegas Golden Knights", "venue": "T-Mobile Arena"},
    "WPG": {"name": "Winnipeg Jets", "venue": "Canada Life Centre"},
}

SPORTSBOOKS = ["DraftKings", "FanDuel", "BetMGM", "Caesars"]


def seed_sample_data(storage: Storage, days: int = 7) -> dict[str, int]:
    """Seed storage with realistic sample data.

    Returns counts of records created per table.
    """
    rng = np.random.default_rng(42)  # deterministic
    counts: dict[str, int] = {}

    # --- Games ---
    games = []
    today = datetime.now().date()
    teams_list = list(MLB_TEAMS.keys())

    for day_offset in range(-3, days + 1):
        date = today + timedelta(days=day_offset)
        # 5 MLB games per day
        rng.shuffle(teams_list)
        for i in range(0, min(10, len(teams_list)), 2):
            home = teams_list[i]
            away = teams_list[i + 1]
            status = "final" if day_offset < 0 else "scheduled"
            game = {
                "game_id": f"mlb-{date.isoformat()}-{home}-{away}",
                "sport": "MLB",
                "date": date.isoformat(),
                "home_team": home,
                "away_team": away,
                "venue": MLB_TEAMS[home]["venue"],
                "status": status,
                "metadata": {},
            }
            if status == "final":
                game["home_score"] = int(rng.integers(0, 10))
                game["away_score"] = int(rng.integers(0, 10))
            games.append(game)

    # NHL games
    nhl_list = list(NHL_TEAMS.keys())
    for day_offset in range(-3, days + 1):
        date = today + timedelta(days=day_offset)
        rng.shuffle(nhl_list)
        for i in range(0, min(6, len(nhl_list)), 2):
            home = nhl_list[i]
            away = nhl_list[i + 1]
            status = "final" if day_offset < 0 else "scheduled"
            game = {
                "game_id": f"nhl-{date.isoformat()}-{home}-{away}",
                "sport": "NHL",
                "date": date.isoformat(),
                "home_team": home,
                "away_team": away,
                "venue": NHL_TEAMS[home]["venue"],
                "status": status,
                "metadata": {},
            }
            if status == "final":
                game["home_score"] = int(rng.integers(0, 6))
                game["away_score"] = int(rng.integers(0, 6))
            games.append(game)

    counts["games"] = storage.upsert_games(games)

    # --- Team stats ---
    team_stats = []
    for tid, info in MLB_TEAMS.items():
        wins = int(rng.integers(65, 100))
        losses = 162 - wins
        rs = int(rng.integers(600, 850))
        ra = int(rng.integers(580, 830))
        team_stats.append({
            "team_id": tid,
            "sport": "MLB",
            "season": "2026",
            "stats": {
                "wins": wins,
                "losses": losses,
                "win_percentage": round(wins / 162, 3),
                "runs_scored": rs,
                "runs_allowed": ra,
                "run_differential": rs - ra,
                "batting_average": round(float(rng.uniform(0.235, 0.275)), 3),
                "era": round(float(rng.uniform(3.40, 4.60)), 2),
                "recent_form": round(float(rng.uniform(0.35, 0.70)), 3),
            },
        })
    for tid, info in NHL_TEAMS.items():
        wins = int(rng.integers(35, 55))
        losses = int(rng.integers(20, 40))
        otl = 82 - wins - losses
        gf = int(rng.integers(220, 320))
        ga = int(rng.integers(210, 310))
        team_stats.append({
            "team_id": tid,
            "sport": "NHL",
            "season": "2025-26",
            "stats": {
                "wins": wins,
                "losses": losses,
                "overtime_losses": otl,
                "points": wins * 2 + otl,
                "goals_for": gf,
                "goals_against": ga,
                "goal_differential": gf - ga,
                "power_play_percentage": round(float(rng.uniform(17, 27)), 1),
                "penalty_kill_percentage": round(float(rng.uniform(76, 88)), 1),
                "recent_form": round(float(rng.uniform(0.35, 0.70)), 3),
            },
        })
    counts["team_stats"] = storage.upsert_team_stats(team_stats)

    # --- Odds ---
    odds = []
    for game in games:
        for book in SPORTSBOOKS:
            # Realistic moneyline odds with vig (~4.5%)
            true_home = float(rng.uniform(0.35, 0.65))
            vig = 1.045
            home_decimal = round(vig / true_home, 2)
            away_decimal = round(vig / (1.0 - true_home), 2)
            total = round(float(rng.uniform(7.5, 10.5)), 1) if game["sport"] == "MLB" else round(float(rng.uniform(5.5, 7.0)), 1)

            odds.append({
                "game_id": game["game_id"],
                "sportsbook": book,
                "market_type": "moneyline",
                "home_odds": home_decimal,
                "away_odds": away_decimal,
                "spread": round(float(rng.uniform(-2.5, 2.5)), 1),
                "total": total,
                "over_odds": round(float(rng.uniform(1.85, 1.95)), 2),
                "under_odds": round(float(rng.uniform(1.85, 1.95)), 2),
            })
    counts["odds"] = storage.insert_odds(odds)

    # --- Weather ---
    weather_count = 0
    for game in games:
        if game["sport"] == "MLB":  # weather matters more for baseball
            storage.insert_weather({
                "game_id": game["game_id"],
                "temperature": round(float(rng.uniform(55, 92)), 1),
                "humidity": round(float(rng.uniform(30, 75)), 1),
                "wind_speed": round(float(rng.uniform(2, 18)), 1),
                "wind_direction": str(rng.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])),
                "precipitation_chance": round(float(rng.uniform(0, 0.4)), 2),
                "conditions": str(rng.choice(["Clear", "Partly Cloudy", "Cloudy", "Overcast"])),
            })
            weather_count += 1
    counts["weather"] = weather_count

    return counts
