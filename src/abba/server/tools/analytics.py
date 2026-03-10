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
        """Ensemble prediction for a game. Returns home win probability."""
        start = time.time()

        game = self.storage.get_game_by_id(game_id)
        if not game:
            return self._track("predict_game", {"game_id": game_id},
                               {"error": f"game not found: {game_id}"}, start)

        sport = game.get("sport", "MLB")
        home = game.get("home_team", "")
        away = game.get("away_team", "")

        home_stats_list = self.storage.query_team_stats(team_id=home, sport=sport)
        away_stats_list = self.storage.query_team_stats(team_id=away, sport=sport)

        home_stats = home_stats_list[0] if home_stats_list else {"stats": {}}
        away_stats = away_stats_list[0] if away_stats_list else {"stats": {}}

        weather = self.storage.get_weather(game_id)
        features = self.features.build_features(home_stats, away_stats, weather, sport)
        model_preds = self.features.predict_from_features(features)

        data_hash = self.ensemble.data_hash(game_id, self.VERSION, features)
        cached = self.storage.get_cached_prediction(game_id, self.VERSION, data_hash)
        if cached:
            cached["_cache_hit"] = True
            return self._track("predict_game", {"game_id": game_id}, cached, start)

        prediction = self.ensemble.combine(model_preds, method=method)

        result = {
            "game_id": game_id,
            "home_team": home,
            "away_team": away,
            "sport": sport,
            "prediction": prediction.to_dict(),
            "features": {k: round(v, 4) for k, v in features.items()},
            "_cache_hit": False,
        }

        self.storage.cache_prediction(game_id, self.VERSION, data_hash, result)
        return self._track("predict_game", {"game_id": game_id}, result, start)

    def explain_prediction(self, game_id: str) -> dict[str, Any]:
        """Explain what's driving a prediction -- feature importance."""
        start = time.time()
        pred = self.predict_game(game_id)
        if "error" in pred:
            return self._track("explain_prediction", {"game_id": game_id}, pred, start)

        features = pred.get("features", {})

        neutral = {
            "home_win_pct": 0.5, "away_win_pct": 0.5,
            "home_run_diff_per_game": 0.0, "away_run_diff_per_game": 0.0,
            "home_recent_form": 0.5, "away_recent_form": 0.5,
            "home_advantage": 0.54,
            "temp_impact": 0.0, "wind_impact": 0.0, "precip_risk": 0.0,
        }

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
                "description": self.features.FEATURE_SCHEMA.get(feat, ""),
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
