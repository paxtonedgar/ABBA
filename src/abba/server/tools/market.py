"""Market / betting tools mixin."""

from __future__ import annotations

import time
from typing import Any


class MarketToolsMixin:
    """Value scanning, odds comparison, EV calculation, and Kelly sizing."""

    def find_value(
        self,
        sport: str | None = None,
        date: str | None = None,
        min_ev: float = 0.03,
    ) -> dict[str, Any]:
        """Scan for +EV betting opportunities."""
        start = time.time()

        games = self.storage.query_games(sport=sport, date=date, status="scheduled")
        if not games:
            return self._track("find_value", {"sport": sport, "date": date},
                               {"opportunities": [], "count": 0, "games_scanned": 0}, start)

        predictions: dict[str, float] = {}
        for g in games:
            gid = g.get("game_id", "")
            pred = self.predict_game(gid)
            pred_data = pred.get("prediction", {})
            if isinstance(pred_data, dict) and "value" in pred_data:
                predictions[gid] = pred_data["value"]

        all_odds = self.storage.query_odds(latest_only=True)

        self.value.min_ev = min_ev
        opportunities = self.value.find_value(games, predictions, all_odds)

        result = {
            "opportunities": opportunities[:20],
            "count": len(opportunities),
            "games_scanned": len(games),
        }
        return self._track("find_value", {"sport": sport, "min_ev": min_ev}, result, start)

    def compare_odds(self, game_id: str) -> dict[str, Any]:
        """Compare odds across sportsbooks for a game."""
        start = time.time()
        all_odds = self.storage.query_odds(game_id=game_id, latest_only=True)
        result = self.value.compare_odds(all_odds, game_id)
        return self._track("compare_odds", {"game_id": game_id}, result, start)

    def calculate_ev(
        self,
        win_probability: float,
        decimal_odds: float,
    ) -> dict[str, Any]:
        """Calculate expected value for a specific bet."""
        start = time.time()
        ev = win_probability * (decimal_odds - 1.0) - (1.0 - win_probability)
        implied = 1.0 / decimal_odds if decimal_odds > 0 else 0
        edge = win_probability - implied

        result = {
            "win_probability": round(win_probability, 4),
            "decimal_odds": round(decimal_odds, 4),
            "implied_probability": round(implied, 4),
            "edge": round(edge, 4),
            "expected_value": round(ev, 4),
            "is_positive_ev": ev > 0,
        }
        return self._track("calculate_ev",
                           {"win_probability": win_probability, "decimal_odds": decimal_odds},
                           result, start)

    def kelly_sizing(
        self,
        win_probability: float,
        decimal_odds: float,
        bankroll: float = 10000.0,
    ) -> dict[str, Any]:
        """Calculate optimal position size using Kelly Criterion."""
        start = time.time()
        sizing = self.kelly.calculate(win_probability, decimal_odds, bankroll)
        result = sizing.to_dict()
        result["bankroll"] = bankroll
        result["decimal_odds"] = decimal_odds
        result["win_probability"] = round(win_probability, 4)
        return self._track("kelly_sizing",
                           {"win_probability": win_probability, "decimal_odds": decimal_odds,
                            "bankroll": bankroll}, result, start)
