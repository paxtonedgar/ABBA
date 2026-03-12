"""Market service — owns value scanning and odds comparison.

Extracted from MarketToolsMixin.
"""

from __future__ import annotations

from typing import Any

from ..engine.kelly import KellyEngine
from ..engine.value import ValueEngine
from ..storage import Storage


class MarketService:
    """Owns market analysis: value scanning, odds comparison, EV, Kelly."""

    def __init__(
        self,
        storage: Storage,
        value: ValueEngine,
        kelly: KellyEngine,
        predict_fn: Any = None,
        nhl_predict_fn: Any = None,
    ):
        self.storage = storage
        self.value = value
        self.kelly = kelly
        self._predict_fn = predict_fn
        self._nhl_predict_fn = nhl_predict_fn

    def find_value(
        self,
        sport: str | None = None,
        date: str | None = None,
        min_ev: float = 0.03,
    ) -> dict[str, Any]:
        """Scan for +EV betting opportunities.

        Routes NHL games through the NHL-specific predictor via predict_fn,
        generic games through generic_predict_fn.
        """
        games = self.storage.query_games(sport=sport, date=date, status="scheduled")
        if not games:
            return {"opportunities": [], "count": 0, "games_scanned": 0}

        predictions: dict[str, float] = {}
        for g in games:
            gid = g.get("game_id", "")
            game_sport = g.get("sport", "").upper()
            # Route NHL games through the NHL-specific predictor
            if game_sport == "NHL" and self._nhl_predict_fn:
                pred = self._nhl_predict_fn(gid)
            elif self._predict_fn:
                pred = self._predict_fn(gid)
            else:
                continue
            pred_data = pred.get("prediction", {})
            if isinstance(pred_data, dict) and "value" in pred_data:
                predictions[gid] = pred_data["value"]

        all_odds = self.storage.query_odds(latest_only=True)

        self.value.min_ev = min_ev
        opportunities = self.value.find_value(games, predictions, all_odds)

        return {
            "opportunities": opportunities[:20],
            "count": len(opportunities),
            "games_scanned": len(games),
        }

    def compare_odds(self, game_id: str) -> dict[str, Any]:
        """Compare odds across sportsbooks for a game."""
        all_odds = self.storage.query_odds(game_id=game_id, latest_only=True)
        return self.value.compare_odds(all_odds, game_id)
