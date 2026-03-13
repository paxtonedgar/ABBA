"""Prediction service — owns NHL and generic prediction orchestration.

Extracted from NHLToolsMixin.nhl_predict_game and AnalyticsToolsMixin.predict_game.
Can be tested without instantiating ABBAToolkit or any mixins.
"""

from __future__ import annotations

from typing import Any

from ..engine.calibration import apply_temperature
from ..engine.confidence import build_prediction_meta, get_calibration_artifact
from ..engine.elo import EloRatings
from ..engine.ensemble import EnsembleEngine
from ..engine.features import FeatureEngine
from ..engine.hockey import HockeyAnalytics
from ..engine.ml_model import NHLGameModel
from ..services.prediction_input import CURRENT_SEASON, build_nhl_prediction_input
from ..storage import Storage
from ..types import PredictionOutput


class PredictionService:
    """Owns prediction orchestration for NHL and generic sports.

    Dependencies are injected — no toolkit or mixin references.
    """

    def __init__(
        self,
        storage: Storage,
        hockey: HockeyAnalytics,
        ensemble: EnsembleEngine,
        features: FeatureEngine,
        elo: EloRatings,
        ml_model: NHLGameModel,
    ):
        self.storage = storage
        self.hockey = hockey
        self.ensemble = ensemble
        self.features = features
        self.elo = elo
        self.ml_model = ml_model

    def predict_nhl(
        self,
        game_id: str,
        method: str = "weighted",
        version: str = "2.0.0",
        last_refresh_ts: float | None = None,
        player_impact_fn: Any = None,
    ) -> dict[str, Any]:
        """Full NHL prediction pipeline with fail-closed guards.

        Args:
            game_id: Game to predict.
            method: Ensemble method (weighted, average, median, voting).
            version: Model version string for cache keying.
            last_refresh_ts: Unix timestamp of last data refresh (for confidence).
            player_impact_fn: Callable(team) -> impact dict. If None, uses neutral defaults.
        """
        prepared = build_nhl_prediction_input(
            storage=self.storage,
            hockey=self.hockey,
            game_id=game_id,
            last_refresh_ts=last_refresh_ts,
            player_impact_fn=player_impact_fn,
            season=CURRENT_SEASON,
        )
        if isinstance(prepared, dict):
            return prepared

        game = self.storage.get_game_by_id(game_id)
        if not game:
            return {"error": f"game not found: {game_id}"}

        home = prepared.prediction_input["home_team"]
        away = prepared.prediction_input["away_team"]
        season = prepared.prediction_input["season"]
        features = dict(prepared.prediction_input["features"])
        defaulted_features = list(prepared.prediction_input.get("defaulted_features", []))
        data_warnings = list(prepared.data_warnings)
        home_goalie = prepared.home_goalie
        away_goalie = prepared.away_goalie
        home_player_impact = prepared.home_player_impact
        away_player_impact = prepared.away_player_impact

        # Elo
        elo_pred = self.elo.predict(home, away)
        elo_prob = elo_pred.get("home_win_prob")

        # NHL model predictions
        model_preds = self.hockey.predict_nhl_game(features, elo_prob=elo_prob)

        # ML model (optional)
        if self.ml_model.ready:
            ml_prob = self.ml_model.predict(features)
            if ml_prob is not None:
                model_preds.append(ml_prob)

        # Cache check
        data_hash = self.ensemble.data_hash(game_id, version, features)
        cached = self.storage.get_cached_prediction(game_id, version + "-nhl", data_hash)
        if cached:
            cached["_cache_hit"] = True
            return cached

        # Combine
        prediction = self.ensemble.combine(model_preds, method=method)

        # Temperature scaling: recalibrate raw probability using empirical T
        raw_value = prediction.to_dict()["value"]
        cal_artifact = get_calibration_artifact()
        if cal_artifact and cal_artifact.temperature != 1.0:
            from dataclasses import replace
            calibrated = float(apply_temperature([raw_value], cal_artifact.temperature)[0])
            prediction = replace(prediction, value=calibrated)

        # Confidence metadata
        data_source = prepared.data_source
        has_goalie = home_goalie is not None and away_goalie is not None

        extra_caveats = list(data_warnings)
        total_injury = home_player_impact["injury_impact"] + away_player_impact["injury_impact"]
        if total_injury > 0.04:
            extra_caveats.append(f"Significant injuries affecting prediction (combined impact: {total_injury:.1%})")

        confidence_meta = build_prediction_meta(
            features=features,
            prediction_value=prediction.to_dict().get("value", 0.5),
            data_source=data_source,
            has_goalie_data=has_goalie,
            last_refresh_ts=last_refresh_ts,
            extra_caveats=extra_caveats if extra_caveats else None,
        )

        result: PredictionOutput = {
            "game_id": game_id,
            "home_team": home,
            "away_team": away,
            "sport": "NHL",
            "season": season,
            "prediction": prediction.to_dict(),
            "raw_model_prob": round(raw_value, 4),
            "calibration": {
                "temperature": round(cal_artifact.temperature, 4) if cal_artifact else 1.0,
                "applied": cal_artifact is not None and cal_artifact.temperature != 1.0,
                "backtest_sample": cal_artifact.sample_size if cal_artifact else 0,
            },
            "features": {k: round(v, 4) for k, v in features.items()},
            "home_goaltender": home_goalie.get("name") if home_goalie else "unknown",
            "away_goaltender": away_goalie.get("name") if away_goalie else "unknown",
            "model_count": len(model_preds),
            "model_types": self._build_model_types(model_preds, elo_prob, features),
            "elo": {
                "home_rating": round(elo_pred.get("home_rating", 1500), 1),
                "away_rating": round(elo_pred.get("away_rating", 1500), 1),
                "elo_home_prob": round(elo_pred.get("home_win_prob", 0.5), 4),
            },
            "player_impact": {
                "home": home_player_impact,
                "away": away_player_impact,
            },
            "confidence": confidence_meta,
            "defaulted_features": defaulted_features if defaulted_features else None,
            "data_provenance": prepared.data_provenance,
            "prediction_input": prepared.prediction_input,
            "model_features_used": sorted(prepared.prediction_input["features"].keys()),
            "context_only": prepared.prediction_input.get("context_only", {}),
            "_cache_hit": False,
        }

        self.storage.cache_prediction(game_id, version + "-nhl", data_hash, result)
        return result

    @staticmethod
    def _build_model_types(
        model_preds: list[float],
        elo_prob: float | None,
        features: dict[str, float],
    ) -> list[str]:
        """Build the list of model type labels matching predict_nhl_game output.

        Model order (from predict_nhl_game):
        1. points_log5
        2. pythagorean_situational (Pythagorean + Corsi/xG + ST + rest)
        3. goaltender_matchup
        4. (optional) elo
        5. (optional) market_implied
        + gradient_boosting (from ML model, appended by caller)
        """
        types = ["points_log5", "pythagorean_situational",
                 "goaltender_matchup"]
        if elo_prob is not None and 0.01 <= elo_prob <= 0.99:
            types.append("elo")
        market = features.get("market_implied_prob", 0)
        if market > 0 and 0.15 <= market <= 0.85:
            types.append("market_implied")
        while len(types) < len(model_preds):
            types.append("gradient_boosting")
        return types[:len(model_preds)]

    def predict_generic(
        self,
        game_id: str,
        method: str = "weighted",
        version: str = "2.0.0",
    ) -> dict[str, Any]:
        """Generic (non-NHL) prediction pipeline."""
        game = self.storage.get_game_by_id(game_id)
        if not game:
            return {"error": f"game not found: {game_id}"}

        sport = game.get("sport", "MLB")

        # NHL games must use the NHL-specific predictor
        if sport == "NHL":
            return self.predict_nhl(game_id, method=method, version=version)

        home = game.get("home_team", "")
        away = game.get("away_team", "")

        home_stats_list = self.storage.query_team_stats(team_id=home, sport=sport)
        away_stats_list = self.storage.query_team_stats(team_id=away, sport=sport)

        home_stats = home_stats_list[0] if home_stats_list else {"stats": {}}
        away_stats = away_stats_list[0] if away_stats_list else {"stats": {}}

        weather = self.storage.get_weather(game_id)
        features = self.features.build_features(home_stats, away_stats, weather, sport)
        model_preds = self.features.predict_from_features(features)

        data_hash = self.ensemble.data_hash(game_id, version, features)
        cached = self.storage.get_cached_prediction(game_id, version, data_hash)
        if cached:
            cached["_cache_hit"] = True
            return cached

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

        self.storage.cache_prediction(game_id, version, data_hash, result)
        return result

    def explain(
        self,
        game_id: str,
        version: str = "2.0.0",
    ) -> dict[str, Any]:
        """Feature importance breakdown for a prediction."""
        pred = self.predict_generic(game_id, version=version)
        if "error" in pred:
            return pred

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

        return {
            "game_id": game_id,
            "home_team": pred.get("home_team"),
            "away_team": pred.get("away_team"),
            "prediction": pred.get("prediction"),
            "top_factors": importance[:5],
            "all_factors": importance,
        }

    def player_impact(self, team: str) -> dict[str, float]:
        """Compute player-level impact features for a team."""
        roster = self.storage.query_roster(team=team)
        if not roster:
            return {"injury_impact": 0.0, "top_scorer_available": 1.0, "roster_completeness": 1.0}

        total = len(roster)
        healthy = [p for p in roster if p.get("injury_status", "healthy") == "healthy"]

        skaters = [p for p in roster if p.get("position", "") not in ("G",)]
        skaters_by_pts = sorted(
            skaters,
            key=lambda p: (p.get("stats") or {}).get("points", 0)
            if isinstance(p.get("stats"), dict) else 0,
            reverse=True,
        )

        top_players = skaters_by_pts[:10]
        top_healthy = sum(
            1 for p in top_players if p.get("injury_status", "healthy") == "healthy"
        )
        top_scorer_available = top_healthy / max(len(top_players), 1)

        injury_impact = 0.0
        for i, p in enumerate(skaters_by_pts):
            if p.get("injury_status", "healthy") != "healthy":
                weight = max(0.015 - i * 0.001, 0.003)
                injury_impact += weight

        goalies = [p for p in roster if p.get("position") == "G"]
        starter_injured = any(
            g.get("injury_status", "healthy") != "healthy"
            for g in goalies[:1]
        )
        if starter_injured:
            injury_impact += 0.03

        return {
            "injury_impact": round(min(injury_impact, 0.10), 4),
            "top_scorer_available": round(top_scorer_available, 3),
            "roster_completeness": round(len(healthy) / max(total, 1), 3),
        }
