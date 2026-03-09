"""Expected value and market edge detection.

Scans available odds across sportsbooks against model predictions
to find +EV opportunities. The core loop:

1. For each game, get the ensemble prediction (home win probability)
2. For each sportsbook's odds on that game, compute implied probability
3. If our predicted prob exceeds implied prob by min_edge, it's +EV
4. Rank by EV per dollar wagered, highest first
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ValueEngine:
    """Finds expected value opportunities in betting markets."""

    def __init__(self, min_ev: float = 0.03, min_edge: float = 0.02):
        """
        Args:
            min_ev: Minimum expected value per dollar wagered (default 3%)
            min_edge: Minimum probability edge over implied odds (default 2%)
        """
        self.min_ev = min_ev
        self.min_edge = min_edge

    def find_value(
        self,
        games: list[dict[str, Any]],
        predictions: dict[str, float],
        odds: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Scan for +EV opportunities.

        Args:
            games: List of game dicts with game_id, home_team, away_team
            predictions: Dict of game_id -> home win probability
            odds: List of odds snapshots with game_id, home_odds, away_odds, sportsbook
        """
        opportunities = []

        # Index odds by game_id
        odds_by_game: dict[str, list[dict]] = {}
        for o in odds:
            gid = o.get("game_id", "")
            odds_by_game.setdefault(gid, []).append(o)

        for game in games:
            gid = game.get("game_id", "")
            home_prob = predictions.get(gid)
            if home_prob is None:
                continue

            away_prob = 1.0 - home_prob
            game_odds = odds_by_game.get(gid, [])

            for o in game_odds:
                home_decimal = o.get("home_odds", 0)
                away_decimal = o.get("away_odds", 0)
                book = o.get("sportsbook", "unknown")

                # Check home bet
                if home_decimal and home_decimal > 1.0:
                    home_ev = self._calculate_ev(home_prob, home_decimal)
                    home_edge = home_prob - (1.0 / home_decimal)
                    if home_ev >= self.min_ev and home_edge >= self.min_edge:
                        opportunities.append({
                            "game_id": gid,
                            "home_team": game.get("home_team", ""),
                            "away_team": game.get("away_team", ""),
                            "selection": "home",
                            "team": game.get("home_team", ""),
                            "sportsbook": book,
                            "decimal_odds": home_decimal,
                            "implied_probability": round(1.0 / home_decimal, 4),
                            "model_probability": round(home_prob, 4),
                            "edge": round(home_edge, 4),
                            "expected_value": round(home_ev, 4),
                        })

                # Check away bet
                if away_decimal and away_decimal > 1.0:
                    away_ev = self._calculate_ev(away_prob, away_decimal)
                    away_edge = away_prob - (1.0 / away_decimal)
                    if away_ev >= self.min_ev and away_edge >= self.min_edge:
                        opportunities.append({
                            "game_id": gid,
                            "home_team": game.get("home_team", ""),
                            "away_team": game.get("away_team", ""),
                            "selection": "away",
                            "team": game.get("away_team", ""),
                            "sportsbook": book,
                            "decimal_odds": away_decimal,
                            "implied_probability": round(1.0 / away_decimal, 4),
                            "model_probability": round(away_prob, 4),
                            "edge": round(away_edge, 4),
                            "expected_value": round(away_ev, 4),
                        })

        # Sort by EV descending
        opportunities.sort(key=lambda x: x["expected_value"], reverse=True)
        return opportunities

    def compare_odds(
        self, odds: list[dict[str, Any]], game_id: str
    ) -> dict[str, Any]:
        """Compare odds across sportsbooks for a game."""
        game_odds = [o for o in odds if o.get("game_id") == game_id]
        if not game_odds:
            return {"game_id": game_id, "books": [], "best_home": None, "best_away": None}

        books = []
        best_home = None
        best_away = None

        for o in game_odds:
            book = o.get("sportsbook", "unknown")
            home = o.get("home_odds", 0)
            away = o.get("away_odds", 0)
            books.append({"sportsbook": book, "home_odds": home, "away_odds": away})

            if home and (best_home is None or home > best_home["odds"]):
                best_home = {"sportsbook": book, "odds": home}
            if away and (best_away is None or away > best_away["odds"]):
                best_away = {"sportsbook": book, "odds": away}

        return {
            "game_id": game_id,
            "books": books,
            "best_home": best_home,
            "best_away": best_away,
        }

    def _calculate_ev(self, probability: float, decimal_odds: float) -> float:
        """Expected value per dollar wagered.

        EV = P(win) * (odds - 1) - P(loss)
        Positive EV means the bet is profitable long-term.
        """
        p = np.clip(probability, 0.001, 0.999)
        return float(p * (decimal_odds - 1.0) - (1.0 - p))
