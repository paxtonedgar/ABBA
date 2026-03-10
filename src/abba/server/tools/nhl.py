"""NHL-specific tools mixin."""

from __future__ import annotations

import time
from typing import Any


class NHLToolsMixin:
    """NHL prediction, goaltender, advanced stats, cap, roster, season review, and playoff tools."""

    def nhl_predict_game(
        self,
        game_id: str,
        method: str = "weighted",
    ) -> dict[str, Any]:
        """NHL-specific prediction using Corsi, xG, goaltender, special teams, Elo, and player impact."""
        from ...engine.confidence import build_prediction_meta

        start = time.time()

        game = self.storage.get_game_by_id(game_id)
        if not game:
            return self._track("nhl_predict_game", {"game_id": game_id},
                               {"error": f"game not found: {game_id}"}, start)

        if game.get("sport") != "NHL":
            return self._track("nhl_predict_game", {"game_id": game_id},
                               {"error": "not an NHL game, use predict_game for other sports"}, start)

        home = game.get("home_team", "")
        away = game.get("away_team", "")

        # Get all data sources
        home_stats_list = self.storage.query_team_stats(team_id=home, sport="NHL")
        away_stats_list = self.storage.query_team_stats(team_id=away, sport="NHL")
        home_stats = home_stats_list[0] if home_stats_list else {"stats": {}}
        away_stats = away_stats_list[0] if away_stats_list else {"stats": {}}

        home_adv_list = self.storage.query_nhl_advanced_stats(team_id=home)
        away_adv_list = self.storage.query_nhl_advanced_stats(team_id=away)
        home_adv = home_adv_list[0].get("stats", {}) if home_adv_list else None
        away_adv = away_adv_list[0].get("stats", {}) if away_adv_list else None

        # Get starter goaltenders
        home_goalies = self.storage.query_goaltender_stats(team=home)
        away_goalies = self.storage.query_goaltender_stats(team=away)
        home_goalie = next(
            (g["stats"] for g in home_goalies if g.get("stats", {}).get("role") == "starter"),
            home_goalies[0]["stats"] if home_goalies else None,
        )
        away_goalie = next(
            (g["stats"] for g in away_goalies if g.get("stats", {}).get("role") == "starter"),
            away_goalies[0]["stats"] if away_goalies else None,
        )

        # Player-level impact (injuries, depth)
        home_player_impact = self._player_impact(home)
        away_player_impact = self._player_impact(away)

        # Get odds data for market-implied probability
        game_odds = self.storage.query_odds(game_id=game_id, latest_only=True)

        # Build comprehensive NHL features
        features = self.hockey.build_nhl_features(
            home_stats, away_stats,
            home_advanced=home_adv, away_advanced=away_adv,
            home_goalie=home_goalie, away_goalie=away_goalie,
            odds_data=game_odds,
        )

        # Add player-level features
        features["home_injury_impact"] = home_player_impact["injury_impact"]
        features["away_injury_impact"] = away_player_impact["injury_impact"]
        features["home_roster_completeness"] = home_player_impact["roster_completeness"]
        features["away_roster_completeness"] = away_player_impact["roster_completeness"]

        # Get Elo prediction
        elo_pred = self.elo.predict(home, away)
        elo_prob = elo_pred.get("home_win_prob")

        # Generate NHL-specific model predictions
        model_preds = self.hockey.predict_nhl_game(features, elo_prob=elo_prob)

        # Check cache
        data_hash = self.ensemble.data_hash(game_id, self.VERSION, features)
        cached = self.storage.get_cached_prediction(game_id, self.VERSION + "-nhl", data_hash)
        if cached:
            cached["_cache_hit"] = True
            return self._track("nhl_predict_game", {"game_id": game_id}, cached, start)

        # Combine
        prediction = self.ensemble.combine(model_preds, method=method)

        # Determine data source for confidence metadata
        data_source = home_stats.get("source", "unknown") if isinstance(home_stats, dict) else "unknown"
        has_goalie = home_goalie is not None and away_goalie is not None

        extra_caveats = []
        total_injury = home_player_impact["injury_impact"] + away_player_impact["injury_impact"]
        if total_injury > 0.04:
            extra_caveats.append(f"Significant injuries affecting prediction (combined impact: {total_injury:.1%})")

        confidence_meta = build_prediction_meta(
            features=features,
            prediction_value=prediction.to_dict().get("value", 0.5),
            data_source=data_source,
            has_goalie_data=has_goalie,
            last_refresh_ts=getattr(self, '_last_refresh_ts', None),
            extra_caveats=extra_caveats if extra_caveats else None,
        )

        result = {
            "game_id": game_id,
            "home_team": home,
            "away_team": away,
            "sport": "NHL",
            "prediction": prediction.to_dict(),
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
            "_cache_hit": False,
        }

        self.storage.cache_prediction(game_id, self.VERSION + "-nhl", data_hash, result)
        return self._track("nhl_predict_game", {"game_id": game_id}, result, start)

    def query_goaltender_stats(
        self,
        team: str | None = None,
        goaltender_id: str | None = None,
        season: str | None = None,
    ) -> dict[str, Any]:
        """Query NHL goaltender stats."""
        start = time.time()
        params = {k: v for k, v in locals().items() if k != "self" and v is not None}
        goalies = self.storage.query_goaltender_stats(
            goaltender_id=goaltender_id, team=team, season=season,
        )
        result = {"goaltenders": goalies, "count": len(goalies)}
        return self._track("query_goaltender_stats", params, result, start)

    def query_advanced_stats(
        self,
        team_id: str | None = None,
        season: str | None = None,
    ) -> dict[str, Any]:
        """Query NHL advanced stats."""
        start = time.time()
        params = {k: v for k, v in locals().items() if k != "self" and v is not None}
        stats = self.storage.query_nhl_advanced_stats(team_id=team_id, season=season)
        result = {"teams": stats, "count": len(stats)}
        return self._track("query_advanced_stats", params, result, start)

    def query_cap_data(
        self,
        team: str | None = None,
        season: str | None = None,
        position: str | None = None,
    ) -> dict[str, Any]:
        """Query salary cap data with cap analysis."""
        start = time.time()
        params = {k: v for k, v in locals().items() if k != "self" and v is not None}
        contracts = self.storage.query_salary_cap(team=team, season=season, position=position)

        analysis = None
        if team and contracts:
            roster_for_analysis = [
                {"name": c["name"], "position": c.get("position", ""),
                 "cap_hit": c.get("cap_hit", 0),
                 "contract_years_remaining": c.get("contract_years_remaining", 1),
                 "status": c.get("status", "active")}
                for c in contracts
            ]
            analysis = self.hockey.cap_analysis(roster_for_analysis)

        result = {"contracts": contracts, "count": len(contracts)}
        if analysis:
            result["cap_analysis"] = analysis
        return self._track("query_cap_data", params, result, start)

    def query_roster(
        self,
        team: str | None = None,
        season: str | None = None,
        position: str | None = None,
    ) -> dict[str, Any]:
        """Query team roster."""
        start = time.time()
        params = {k: v for k, v in locals().items() if k != "self" and v is not None}
        players = self.storage.query_roster(team=team, season=season, position=position)
        result = {"players": players, "count": len(players)}
        return self._track("query_roster", params, result, start)

    def season_review(
        self,
        team_id: str,
        season: str | None = None,
    ) -> dict[str, Any]:
        """Comprehensive NHL season review."""
        start = time.time()
        season = season or "2025-26"

        team_stats_list = self.storage.query_team_stats(team_id=team_id, sport="NHL", season=season)
        if not team_stats_list:
            return self._track("season_review", {"team_id": team_id},
                               {"error": f"no stats found for {team_id}"}, start)

        team_stats = team_stats_list[0].get("stats", {})

        adv_list = self.storage.query_nhl_advanced_stats(team_id=team_id, season=season)
        advanced = adv_list[0].get("stats", {}) if adv_list else None

        goalies = self.storage.query_goaltender_stats(team=team_id, season=season)
        goalie_stats = [g.get("stats", {}) for g in goalies] if goalies else None

        review = self.hockey.season_review(team_stats, advanced, goalie_stats)
        review["team_id"] = team_id
        review["season"] = season

        return self._track("season_review", {"team_id": team_id, "season": season}, review, start)

    def playoff_odds(
        self,
        team_id: str,
        season: str | None = None,
        division_cutline: int = 90,
        wildcard_cutline: int = 95,
    ) -> dict[str, Any]:
        """Estimate playoff probability from current points pace."""
        start = time.time()
        season = season or "2025-26"

        team_stats_list = self.storage.query_team_stats(team_id=team_id, sport="NHL", season=season)
        if not team_stats_list:
            return self._track("playoff_odds", {"team_id": team_id},
                               {"error": f"no stats found for {team_id}"}, start)

        stats = team_stats_list[0].get("stats", {})
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        otl = stats.get("overtime_losses", 0)
        gp = wins + losses + otl
        points = wins * 2 + otl
        remaining = 82 - gp

        result = self.hockey.playoff_probability(
            current_points=points,
            games_remaining=remaining,
            games_played=gp,
            division_cutline=division_cutline,
            wildcard_cutline=wildcard_cutline,
        )
        result["team_id"] = team_id
        result["season"] = season
        return self._track("playoff_odds", {"team_id": team_id}, result, start)
