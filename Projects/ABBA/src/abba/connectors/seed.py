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

    # --- NHL Goaltender stats ---
    NHL_GOALIES = {
        "NYR": [
            {"id": "shesterkin-igor", "name": "Igor Shesterkin", "role": "starter"},
            {"id": "quick-jonathan", "name": "Jonathan Quick", "role": "backup"},
        ],
        "BOS": [
            {"id": "swayman-jeremy", "name": "Jeremy Swayman", "role": "starter"},
            {"id": "korpisalo-joonas", "name": "Joonas Korpisalo", "role": "backup"},
        ],
        "CAR": [
            {"id": "andersen-frederik", "name": "Frederik Andersen", "role": "starter"},
            {"id": "kochetkov-pyotr", "name": "Pyotr Kochetkov", "role": "backup"},
        ],
        "COL": [
            {"id": "georgiev-alexandar", "name": "Alexandar Georgiev", "role": "starter"},
            {"id": "annunen-justus", "name": "Justus Annunen", "role": "backup"},
        ],
        "DAL": [
            {"id": "oettinger-jake", "name": "Jake Oettinger", "role": "starter"},
            {"id": "wedgewood-scott", "name": "Scott Wedgewood", "role": "backup"},
        ],
        "FLA": [
            {"id": "bobrovsky-sergei", "name": "Sergei Bobrovsky", "role": "starter"},
            {"id": "knight-spencer", "name": "Spencer Knight", "role": "backup"},
        ],
        "VGK": [
            {"id": "hill-adin", "name": "Adin Hill", "role": "starter"},
            {"id": "thompson-logan", "name": "Logan Thompson", "role": "backup"},
        ],
        "WPG": [
            {"id": "hellebuyck-connor", "name": "Connor Hellebuyck", "role": "starter"},
            {"id": "comrie-eric", "name": "Eric Comrie", "role": "backup"},
        ],
    }

    goalie_stats = []
    for team_code, goalies in NHL_GOALIES.items():
        for g in goalies:
            is_starter = g["role"] == "starter"
            gp = int(rng.integers(45, 62)) if is_starter else int(rng.integers(18, 30))
            sv_pct = round(float(rng.uniform(0.910, 0.928)), 4) if is_starter else round(float(rng.uniform(0.895, 0.915)), 4)
            sa = int(gp * rng.integers(28, 34))
            ga = int(round(sa * (1 - sv_pct)))
            saves = sa - ga
            mins = gp * 60.0
            gaa = round(ga / gp * 60 / 60, 3) if gp > 0 else 0
            xga = round(float(ga + rng.uniform(-8, 8)), 1)
            gsaa = round((0.907 * sa - ga), 2)
            xgsaa = round(xga - ga, 2)
            qs = int(rng.integers(int(gp * 0.45), int(gp * 0.75) + 1)) if is_starter else int(rng.integers(int(gp * 0.3), int(gp * 0.6) + 1))
            shutouts = int(rng.integers(1, 7)) if is_starter else int(rng.integers(0, 3))

            goalie_stats.append({
                "goaltender_id": g["id"],
                "team": team_code,
                "season": "2025-26",
                "stats": {
                    "name": g["name"],
                    "role": g["role"],
                    "games_played": gp,
                    "save_pct": sv_pct,
                    "gaa": gaa,
                    "saves": saves,
                    "shots_against": sa,
                    "goals_against": ga,
                    "xg_against": xga,
                    "gsaa": gsaa,
                    "xgsaa": xgsaa,
                    "quality_starts": qs,
                    "shutouts": shutouts,
                    "minutes_played": round(mins, 1),
                },
            })
    counts["goaltender_stats"] = storage.upsert_goaltender_stats(goalie_stats)

    # --- NHL Advanced stats (Corsi, Fenwick, xG, score-adjusted) ---
    nhl_advanced = []
    for tid in NHL_TEAMS:
        cf = int(rng.integers(2800, 3500))
        ca = int(rng.integers(2700, 3400))
        cf_pct = round(cf / (cf + ca) * 100, 2)

        ff = int(rng.integers(2200, 2800))
        fa = int(rng.integers(2100, 2700))
        ff_pct = round(ff / (ff + fa) * 100, 2)

        xgf = round(float(rng.uniform(140, 200)), 1)
        xga = round(float(rng.uniform(135, 195)), 1)
        xgf_pct = round(xgf / (xgf + xga) * 100, 2)

        # Score-adjusted (close games are truest measure)
        adj_cf_pct = round(cf_pct + float(rng.uniform(-2, 2)), 2)

        nhl_advanced.append({
            "team_id": tid,
            "season": "2025-26",
            "stats": {
                "corsi_for": cf,
                "corsi_against": ca,
                "corsi_pct": cf_pct,
                "fenwick_for": ff,
                "fenwick_against": fa,
                "fenwick_pct": ff_pct,
                "xgf": xgf,
                "xga": xga,
                "xgf_pct": xgf_pct,
                "adj_corsi_pct": adj_cf_pct,
                "shots_for_per60": round(float(rng.uniform(28, 36)), 2),
                "shots_against_per60": round(float(rng.uniform(27, 35)), 2),
                "shooting_pct": round(float(rng.uniform(8.5, 11.5)), 2),
                "save_pct_5v5": round(float(rng.uniform(0.915, 0.935)), 4),
                "pdo": round(float(rng.uniform(98, 103)), 1),
            },
        })
    counts["nhl_advanced_stats"] = storage.upsert_nhl_advanced_stats(nhl_advanced)

    # --- Salary cap data ---
    POSITIONS = ["C", "LW", "RW", "LD", "RD", "G"]
    # Realistic cap hit ranges by position (in dollars)
    CAP_RANGES = {
        "C": (900_000, 12_500_000), "LW": (800_000, 10_000_000),
        "RW": (800_000, 10_500_000), "LD": (800_000, 9_500_000),
        "RD": (800_000, 9_000_000), "G": (800_000, 11_500_000),
    }
    FIRST_NAMES = ["Alex", "Connor", "Nathan", "Auston", "Cale", "Jack", "Brady",
                   "Leon", "Mika", "Artemi", "Andrei", "Mikko", "Sebastian",
                   "Brayden", "Mark", "Filip", "Elias", "Quinn", "Adam", "Sam"]
    LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller",
                  "Wilson", "Anderson", "Thomas", "Jackson", "White", "Harris",
                  "Martin", "Thompson", "Garcia", "Martinez", "Robinson", "Clark",
                  "Lewis", "Lee", "Walker", "Hall", "Allen", "Young", "King"]

    cap_data = []
    roster_data = []
    for tid in NHL_TEAMS:
        # 23-man roster per team
        used_names: set[str] = set()
        for i in range(23):
            pos = POSITIONS[min(i // 4, 5)] if i < 20 else "G"
            if i >= 20:
                pos = "G"

            # Generate unique name
            while True:
                fname = str(rng.choice(FIRST_NAMES))
                lname = str(rng.choice(LAST_NAMES))
                full_name = f"{fname} {lname}"
                if full_name not in used_names:
                    used_names.add(full_name)
                    break

            player_id = f"{tid}-{lname.lower()}-{fname.lower()}"
            cap_low, cap_high = CAP_RANGES[pos]
            cap_hit = round(float(rng.uniform(cap_low, cap_high)), 0)
            years = int(rng.integers(1, 8))
            status = "active"
            if float(rng.random()) < 0.05:
                status = "ltir"
            elif float(rng.random()) < 0.03:
                status = "buried"

            cap_data.append({
                "player_id": player_id,
                "team": tid,
                "season": "2025-26",
                "name": full_name,
                "position": pos,
                "cap_hit": cap_hit,
                "aav": cap_hit,
                "contract_years_remaining": years,
                "status": status,
            })

            # Line assignment
            if pos in ("C", "LW", "RW"):
                line = (i // 3) + 1
            elif pos in ("LD", "RD"):
                line = ((i - 12) // 2) + 1 if i >= 12 else 1
            else:
                line = 1 if i == 20 else 2

            injury = "healthy"
            if float(rng.random()) < 0.08:
                injury = str(rng.choice(["day-to-day", "IR", "LTIR"]))

            gp = int(rng.integers(40, 80))
            if pos == "G":
                player_stats_json = {
                    "games_played": gp,
                    "save_pct": round(float(rng.uniform(0.895, 0.930)), 4),
                    "gaa": round(float(rng.uniform(2.2, 3.5)), 3),
                }
            else:
                goals = int(rng.integers(2, 40))
                assists = int(rng.integers(5, 55))
                player_stats_json = {
                    "games_played": gp,
                    "goals": goals,
                    "assists": assists,
                    "points": goals + assists,
                    "plus_minus": int(rng.integers(-20, 25)),
                    "pim": int(rng.integers(4, 80)),
                    "toi_per_game": round(float(rng.uniform(10, 22)), 1),
                }

            roster_data.append({
                "player_id": player_id,
                "team": tid,
                "season": "2025-26",
                "name": full_name,
                "position": pos,
                "line_number": min(line, 4),
                "stats": player_stats_json,
                "injury_status": injury,
            })

    counts["salary_cap"] = storage.upsert_salary_cap(cap_data)
    counts["roster"] = storage.upsert_roster(roster_data)

    return counts
