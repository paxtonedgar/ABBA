"""Analytics tools mixin (predict, explain, graph)."""

from __future__ import annotations

import time
from typing import Any


class AnalyticsToolsMixin:
    """Prediction and analysis methods."""

    def predict_game(
        self,
        game_id: str,
        method: str = "weighted",
    ) -> dict[str, Any]:
        """Ensemble prediction for a game. Returns home win probability.

        Routes NHL → nhl_predict_game, others → PredictionService.predict_generic.
        """
        start = time.time()

        game = self.storage.get_game_by_id(game_id)
        if not game:
            return self._track("predict_game", {"game_id": game_id},
                               {"error": f"game not found: {game_id}"}, start)

        sport = game.get("sport", "MLB")

        # NHL games must use the NHL-specific predictor
        if sport == "NHL":
            return self.nhl_predict_game(game_id, method=method)

        result = self.prediction.predict_generic(game_id, method=method, version=self.VERSION)
        return self._track("predict_game", {"game_id": game_id}, result, start)

    def explain_prediction(self, game_id: str) -> dict[str, Any]:
        """Explain what's driving a prediction -- feature importance."""
        start = time.time()
        # Use self.predict_game to ensure NHL path includes player impact
        pred = self.predict_game(game_id)
        if "error" in pred:
            return self._track("explain_prediction", {"game_id": game_id}, pred, start)

        features = pred.get("features", {})
        sport = pred.get("sport", "MLB")

        if sport == "NHL":
            neutral = {
                "home_pts_pct": 0.5, "away_pts_pct": 0.5,
                "home_goal_diff_pg": 0.0, "away_goal_diff_pg": 0.0,
                "home_recent_form": 0.5, "away_recent_form": 0.5,
                "home_gf_per_game": 3.0, "home_ga_per_game": 3.0,
                "away_gf_per_game": 3.0, "away_ga_per_game": 3.0,
                "home_corsi_pct": 0.50, "away_corsi_pct": 0.50,
                "home_xgf_pct": 0.50, "away_xgf_pct": 0.50,
                "goaltender_edge": 0.0, "home_st_edge": 0.0,
                "rest_edge": 0.0, "market_implied_prob": 0.0,
                "home_games_played": 82, "away_games_played": 82,
            }
            schema = self.hockey.NHL_FEATURE_SCHEMA
        else:
            neutral = {
                "home_win_pct": 0.5, "away_win_pct": 0.5,
                "home_run_diff_per_game": 0.0, "away_run_diff_per_game": 0.0,
                "home_recent_form": 0.5, "away_recent_form": 0.5,
                "home_advantage": 0.54,
                "temp_impact": 0.0, "wind_impact": 0.0, "precip_risk": 0.0,
            }
            schema = self.features.FEATURE_SCHEMA

        importance = []
        for feat, val in features.items():
            n = neutral.get(feat, 0.0)
            deviation = abs(val - n)
            direction = "favors_home" if val > n else "favors_away" if val < n else "neutral"
            importance.append({
                "feature": feat,
                "value": val,
                "neutral_value": n,
                "deviation": round(deviation, 4),
                "direction": direction,
                "description": schema.get(feat, ""),
            })

        importance.sort(key=lambda x: x["deviation"], reverse=True)

        result = {
            "game_id": game_id,
            "home_team": pred.get("home_team"),
            "away_team": pred.get("away_team"),
            "prediction": pred.get("prediction"),
            "top_factors": importance[:5],
            "all_factors": importance,
        }
        return self._track("explain_prediction", {"game_id": game_id}, result, start)

    def graph_analysis(self, team_data: dict[str, Any]) -> dict[str, Any]:
        """Team network analysis. Pass players + relationships."""
        start = time.time()
        result = self.graph.analyze_team(team_data)
        return self._track("graph_analysis", {"player_count": len(team_data.get("players", []))},
                           result, start)
