"""Workflow engine -- chains tool calls into multi-step analytical pipelines.

Each workflow:
1. Refreshes live data from APIs
2. Runs a structured chain of calculations
3. Returns a narrative-ready result dict that agents present to users

Workflows handle the "thinking" so the agent just presents results.
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any

from ..engine.confidence import build_workflow_meta
from ..server.toolkit import ABBAToolkit


class WorkflowEngine:
    """Executes multi-step analytical workflows."""

    def __init__(self, toolkit: ABBAToolkit | None = None):
        self.toolkit = toolkit or ABBAToolkit(auto_seed=False)

    def run(self, workflow_name: str, **params: Any) -> dict[str, Any]:
        """Run a named workflow."""
        workflows = {
            "game_prediction": self.game_prediction,
            "tonights_slate": self.tonights_slate,
            "season_story": self.season_story,
            "value_scan": self.value_scan,
            "cap_strategy": self.cap_strategy,
            "playoff_race": self.playoff_race,
            "goaltender_duel": self.goaltender_duel,
            "team_comparison": self.team_comparison,
            "betting_strategy": self.betting_strategy,
        }

        fn = workflows.get(workflow_name)
        if not fn:
            return {"error": f"unknown workflow: {workflow_name}", "available": list(workflows.keys())}

        start = time.time()
        result = fn(**params)
        result["_workflow"] = {
            "name": workflow_name,
            "elapsed_ms": round((time.time() - start) * 1000, 1),
            "params": params,
        }

        # Attach confidence metadata from the weakest-link in the workflow.
        # Single-game workflows: use the prediction's own confidence metadata.
        # Multi-step workflows: build from inferred data sources.
        if "_confidence" not in result:
            # Check if a sub-step already produced confidence metadata
            confidence_from_pred = result.get("confidence")  # from nhl_predict_game
            if isinstance(confidence_from_pred, dict) and "reliability_grade" in confidence_from_pred:
                result["_confidence"] = confidence_from_pred
            else:
                has_goalie = any(
                    k in result for k in ("home_goaltender", "goaltending", "goalie1")
                )
                data_sources = self._infer_data_sources(result)
                result["_confidence"] = build_workflow_meta(
                    workflow_name=workflow_name,
                    data_sources_used=data_sources,
                    steps_completed=len([k for k in result if not k.startswith("_")]),
                    steps_total=len([k for k in result if not k.startswith("_")]),
                    has_goalie_data=has_goalie,
                )

        return result

    @staticmethod
    def _infer_data_sources(result: dict) -> list[str]:
        """Infer which data sources were used from workflow result structure."""
        sources: list[str] = []
        # Check nested prediction confidence for source hints
        # Check top-level confidence metadata (dict), not prediction.confidence (float)
        conf_meta = result.get("confidence", {})
        if isinstance(conf_meta, dict):
            if conf_meta.get("data_freshness") == "seed":
                sources.append("seed")
            elif conf_meta.get("reliability_grade") in ("A", "B", "C"):
                sources.append("live")
        if not sources:
            sources.append("seed")  # conservative default
        return sources

    # =====================================================================
    # GAME PREDICTION WORKFLOW
    # "Who wins tonight's Rangers game?"
    # "Predict NYR vs BOS"
    # "What are the chances the Avs win?"
    # =====================================================================

    def game_prediction(self, team: str | None = None, game_id: str | None = None) -> dict[str, Any]:
        """Full game prediction pipeline with narrative context.

        Steps:
        1. Refresh live data
        2. Find the game (by team or game_id)
        3. Pull both teams' stats, advanced stats, goaltender stats
        4. Compute rest/B2B from schedule
        5. Run 6-model NHL prediction
        6. Compare odds and find value
        7. Size the bet if +EV
        8. Build narrative summary
        """
        # 1. Refresh
        self.toolkit.refresh_data(source="nhl", team=team)

        season = getattr(self.toolkit, "_CURRENT_SEASON", "2025-26")

        # 2. Find the game
        if game_id:
            games = self.toolkit.query_games(sport="NHL")
            game = next((g for g in games.get("games", []) if g.get("game_id") == game_id), None)
        elif team:
            games = self.toolkit.query_games(sport="NHL", team=team, status="scheduled")
            game = _select_target_game(games.get("games", []))
        else:
            return {"error": "provide team or game_id"}

        if not game:
            return {"error": f"no scheduled game found for {team or game_id}"}

        gid = game["game_id"]
        home = game["home_team"]
        away = game["away_team"]

        # 3. Team profiles
        home_stats = self.toolkit.query_team_stats(team_id=home, sport="NHL", season=season)
        away_stats = self.toolkit.query_team_stats(team_id=away, sport="NHL", season=season)
        home_adv = self.toolkit.query_advanced_stats(team_id=home, season=season)
        away_adv = self.toolkit.query_advanced_stats(team_id=away, season=season)
        home_goalies = self.toolkit.query_goaltender_stats(team=home, season=season)
        away_goalies = self.toolkit.query_goaltender_stats(team=away, season=season)
        home_roster = self.toolkit.query_roster(team=home, season=season)
        away_roster = self.toolkit.query_roster(team=away, season=season)

        # 3b. Head-to-head season series
        h2h = self._head_to_head(home, away)

        # 4. Rest/B2B detection
        rest_info = self._compute_rest(home, away)

        # 5. Run prediction
        prediction = self.toolkit.nhl_predict_game(gid)

        # 6. Odds and value
        odds = self.toolkit.compare_odds(gid)
        pred_value = prediction.get("prediction", {}).get("value", 0.5)
        market_eval = _evaluate_market_sides(
            toolkit=self.toolkit,
            home_team=home,
            away_team=away,
            home_win_prob=pred_value,
            odds=odds,
        )

        # 7. Build narrative
        hs = home_stats.get("teams", [{}])[0].get("stats", {}) if home_stats.get("count") else {}
        as_ = away_stats.get("teams", [{}])[0].get("stats", {}) if away_stats.get("count") else {}

        home_record = f"{hs.get('wins', 0)}-{hs.get('losses', 0)}-{hs.get('overtime_losses', 0)}"
        away_record = f"{as_.get('wins', 0)}-{as_.get('losses', 0)}-{as_.get('overtime_losses', 0)}"

        home_starter = _find_named_goalie(home_goalies, prediction.get("home_goaltender"))
        away_starter = _find_named_goalie(away_goalies, prediction.get("away_goaltender"))
        defaulted = set(prediction.get("defaulted_features") or [])
        features = prediction.get("features", {})
        best_bet = market_eval.get("best_bet") or {}

        narrative = {
            "headline": f"{home} ({home_record}) vs {away} ({away_record})",
            "home_team": _build_team_snapshot(
                team=home,
                record=home_record,
                stats=hs,
                advanced=home_adv.get("teams", [{}])[0].get("stats", {}) if home_adv.get("count") else {},
                roster=home_roster.get("players", []),
                starter=home_starter,
            ),
            "away_team": _build_team_snapshot(
                team=away,
                record=away_record,
                stats=as_,
                advanced=away_adv.get("teams", [{}])[0].get("stats", {}) if away_adv.get("count") else {},
                roster=away_roster.get("players", []),
                starter=away_starter,
            ),
            "prediction": prediction.get("prediction", {}),
            "features": features,
            "confidence": prediction.get("confidence"),
            "defaulted_features": prediction.get("defaulted_features"),
            "data_provenance": prediction.get("data_provenance"),
            "head_to_head": h2h,
            "rest": rest_info,
            "odds": odds,
            "market_evaluation": market_eval,
            "best_bet": market_eval.get("best_bet"),
            "ev": best_bet.get("ev"),
            "sizing": best_bet.get("kelly"),
        }

        # Key factors narrative — only mention factors the model actually used.
        factors = []
        context_notes = []
        if pred_value > 0.55:
            factors.append(f"{home} favored at {pred_value:.0%}")
        elif pred_value < 0.45:
            factors.append(f"{away} favored at {1-pred_value:.0%}")
        else:
            factors.append("Toss-up game")

        st_edge = features.get("home_st_edge", 0.0)
        if abs(st_edge) >= 0.015:
            advantaged_team = home if st_edge > 0 else away
            factors.append(f"Special teams edge leans {advantaged_team}")

        if "goaltender_edge" not in defaulted:
            if home_starter and home_starter.get("save_pct", 0) > 0.920:
                factors.append(f"Elite goaltending: {home_starter['name']} ({home_starter['save_pct']:.3f} Sv%)")
            if away_starter and away_starter.get("save_pct", 0) > 0.920:
                factors.append(f"Elite goaltending: {away_starter['name']} ({away_starter['save_pct']:.3f} Sv%)")

        if "rest_edge" not in defaulted:
            if rest_info.get("home_b2b"):
                factors.append(f"{home} on a back-to-back (fatigue penalty)")
            if rest_info.get("away_b2b"):
                factors.append(f"{away} on a back-to-back (fatigue penalty)")

        if h2h["games_played"] > 0:
            context_notes.append(
                f"Season series: {home} {h2h['home_team_wins']}-{h2h['away_team_wins']} vs {away}"
                + (f" (last meeting: {h2h['last_meeting']})" if h2h["last_meeting"] else "")
            )

        for team_code, adv_stats in (
            (home, home_adv.get("teams", [{}])[0].get("stats", {}) if home_adv.get("count") else {}),
            (away, away_adv.get("teams", [{}])[0].get("stats", {}) if away_adv.get("count") else {}),
        ):
            analytics_note = _build_analytics_note(team_code, adv_stats)
            if analytics_note:
                context_notes.append(analytics_note)

        for team_code, roster_result in ((home, home_roster), (away, away_roster)):
            roster_note = _build_roster_note(team_code, roster_result.get("players", []))
            if roster_note:
                context_notes.append(roster_note)

        if market_eval.get("best_bet"):
            best_bet = market_eval["best_bet"]
            context_notes.append(
                f"Best market side: {best_bet['team']} {best_bet['side']} at {best_bet['sportsbook']} "
                f"({best_bet['probability']:.0%} model vs {best_bet['implied_probability']:.0%} implied)"
            )

        if defaulted:
            context_notes.append(
                f"Model defaulted {len(defaulted)} neutral features: {', '.join(sorted(defaulted))}"
            )

        narrative["key_factors"] = factors
        narrative["context_notes"] = context_notes
        return narrative

    # =====================================================================
    # TONIGHT'S SLATE
    # "What NHL games are on tonight?"
    # "Give me the full slate with predictions"
    # =====================================================================

    def tonights_slate(self, sport: str = "NHL") -> dict[str, Any]:
        """Full tonight's slate: all games with predictions, odds, and picks."""
        self.toolkit.refresh_data(source="nhl")

        games_result = self.toolkit.query_games(sport=sport, status="scheduled")
        games = games_result.get("games", [])

        if not games:
            return {"games": [], "count": 0, "message": f"No scheduled {sport} games found"}

        slate = []
        value_picks = []

        for game in games[:15]:  # cap at 15 games
            gid = game["game_id"]
            home = game["home_team"]
            away = game["away_team"]

            # Quick prediction
            pred = self.toolkit.nhl_predict_game(gid) if sport == "NHL" else self.toolkit.predict_game(gid)
            pred_val = pred.get("prediction", {}).get("value", 0.5)

            # Odds check
            odds = self.toolkit.compare_odds(gid)
            best_home = odds.get("best_home", {})
            best_away = odds.get("best_away", {})

            confidence_meta = pred.get("confidence", {})
            market_eval = _evaluate_market_sides(
                toolkit=self.toolkit,
                home_team=home,
                away_team=away,
                home_win_prob=pred_val,
                odds=odds,
            )

            entry = {
                "game_id": gid,
                "matchup": f"{away} @ {home}",
                "home_team": home,
                "away_team": away,
                "home_win_prob": round(pred_val, 3),
                "away_win_prob": round(1 - pred_val, 3),
                "pick": home if pred_val > 0.5 else away,
                "confidence": _confidence_sort_key(confidence_meta),
                "confidence_grade": confidence_meta.get("reliability_grade", "unknown"),
                "confidence_interval": confidence_meta.get("confidence_interval"),
                "defaulted_features": pred.get("defaulted_features"),
                "data_provenance": pred.get("data_provenance"),
                "best_home_odds": best_home.get("odds") if best_home else None,
                "best_away_odds": best_away.get("odds") if best_away else None,
                "best_bet": market_eval.get("best_bet"),
            }

            if market_eval.get("best_bet"):
                entry["value_bet"] = market_eval["best_bet"]
                value_picks.append(entry["value_bet"])

            slate.append(entry)

        # Sort by confidence
        slate.sort(key=lambda x: x["confidence"], reverse=True)
        value_picks.sort(key=lambda x: x.get("ev", {}).get("expected_value", 0), reverse=True)

        return {
            "date": date.today().isoformat(),
            "sport": sport,
            "games": slate,
            "game_count": len(slate),
            "value_picks": value_picks,
            "value_pick_count": len(value_picks),
            "best_bet": value_picks[0] if value_picks else None,
        }

    # =====================================================================
    # SEASON STORY
    # "Tell me the story of the Rangers' season"
    # "How have the Bruins been doing?"
    # "Give me the full picture on Colorado"
    # =====================================================================

    def season_story(self, team: str, season: str | None = None) -> dict[str, Any]:
        """Comprehensive season narrative with analytics, trends, and context.

        This is storytelling -- not just data, but the narrative arc of a season.
        """
        self.toolkit.refresh_data(source="nhl", team=team)
        season = season or "2025-26"

        # Core data
        review = self.toolkit.season_review(team_id=team, season=season)
        advanced = self.toolkit.query_advanced_stats(team_id=team, season=season)
        goalies = self.toolkit.query_goaltender_stats(team=team, season=season)
        roster = self.toolkit.query_roster(team=team, season=season)
        cap = self.toolkit.query_cap_data(team=team, season=season)
        playoff = self.toolkit.playoff_odds(team_id=team, season=season)

        # Recent games
        recent = self.toolkit.query_games(sport="NHL", team=team, status="final")

        # Build the story
        story: dict[str, Any] = {"team": team, "season": season}

        # Chapter 1: The record
        story["record_summary"] = {
            "record": review.get("record", ""),
            "points": review.get("points", 0),
            "points_pct": review.get("points_pct", 0),
            "goal_differential": review.get("goal_differential", 0),
            "goals_for_per_game": review.get("goals_for_per_game", 0),
            "goals_against_per_game": review.get("goals_against_per_game", 0),
        }

        # Chapter 2: Luck or skill?
        story["underlying_numbers"] = {
            "pythagorean_wins": review.get("pythagorean_wins", 0),
            "luck_factor": review.get("luck_factor", 0),
            "luck_narrative": (
                "overperforming their underlying numbers -- regression risk"
                if review.get("luck_factor", 0) > 3 else
                "underperforming -- due for positive regression"
                if review.get("luck_factor", 0) < -3 else
                "performing in line with expectations"
            ),
        }

        # Chapter 3: Process (analytics)
        adv_stats = advanced.get("teams", [{}])[0].get("stats", {}) if advanced.get("count") else {}
        story["analytics"] = {
            "corsi_pct": adv_stats.get("corsi_pct", 50),
            "xgf_pct": adv_stats.get("xgf_pct", 50),
            "pdo": adv_stats.get("pdo", 100),
            "shooting_pct": adv_stats.get("shooting_pct", 0),
            "save_pct_5v5": adv_stats.get("save_pct_5v5", 0),
            "grade": review.get("analytics_grade", "n/a"),
            "process_narrative": _analytics_narrative(adv_stats),
        }

        # Chapter 4: Goaltending
        goalie_list = goalies.get("goaltenders", [])
        story["goaltending"] = {
            "grade": review.get("goaltending_grade", "n/a"),
            "goaltenders": [
                {
                    "name": g.get("stats", {}).get("name", ""),
                    "role": g.get("stats", {}).get("role", ""),
                    "sv_pct": g.get("stats", {}).get("save_pct", 0),
                    "gaa": g.get("stats", {}).get("gaa", 0),
                    "gsaa": g.get("stats", {}).get("gsaa", 0),
                    "games": g.get("stats", {}).get("games_played", 0),
                }
                for g in goalie_list[:3]
            ],
        }

        # Chapter 5: Special teams
        story["special_teams"] = {
            "pp_pct": review.get("pp_pct", 0),
            "pk_pct": review.get("pk_pct", 0),
            "grade": review.get("special_teams_grade", "n/a"),
        }

        # Chapter 6: Roster composition
        roster_list = roster.get("players", [])
        forwards = [p for p in roster_list if p.get("position") in ("C", "L", "R", "LW", "RW")]
        defense = [p for p in roster_list if p.get("position") in ("D", "LD", "RD")]
        goalies_r = [p for p in roster_list if p.get("position") == "G"]
        injured = [p for p in roster_list if p.get("injury_status", "healthy") != "healthy"]

        story["roster"] = {
            "forward_count": len(forwards),
            "defense_count": len(defense),
            "goalie_count": len(goalies_r),
            "injured": [{"name": p["name"], "status": p["injury_status"]} for p in injured],
            "injury_count": len(injured),
        }

        # Chapter 7: Cap situation
        cap_analysis = cap.get("cap_analysis", {})
        story["cap_situation"] = {
            "cap_space": cap_analysis.get("cap_space", 0),
            "effective_space": cap_analysis.get("effective_cap_space", 0),
            "health": cap_analysis.get("cap_health", ""),
            "top5_pct": cap_analysis.get("top5_cap_pct", 0),
            "expiring_contracts": cap_analysis.get("expiring_contracts", 0),
            "dead_cap": cap_analysis.get("dead_cap", 0),
            "trade_flexibility": (
                "significant cap space for acquisitions"
                if cap_analysis.get("effective_cap_space", 0) > 10_000_000 else
                "limited flexibility, need to move money out"
                if cap_analysis.get("effective_cap_space", 0) > 2_000_000 else
                "cap-strapped, limited to minimum salary additions"
            ),
        }

        # Chapter 8: Playoff outlook
        story["playoff_outlook"] = {
            "projected_points": playoff.get("projected_points", 0),
            "wildcard_probability": playoff.get("wildcard_probability", 0),
            "division_probability": playoff.get("division_probability", 0),
            "status": playoff.get("status", ""),
            "games_remaining": playoff.get("games_remaining", 0),
            "points_needed": playoff.get("points_needed_wildcard", 0),
        }

        # Chapter 9: Recent results
        recent_games = recent.get("games", [])[:10]
        wins = sum(1 for g in recent_games if _is_win(g, team))
        story["recent_results"] = {
            "last_10_record": f"{wins}-{len(recent_games) - wins}",
            "games": [
                {
                    "date": g.get("date", ""),
                    "opponent": g["away_team"] if g["home_team"] == team else g["home_team"],
                    "score": f"{g.get('home_score', 0)}-{g.get('away_score', 0)}",
                    "result": "W" if _is_win(g, team) else "L",
                }
                for g in recent_games[:5]
            ],
        }

        return story

    # =====================================================================
    # VALUE SCAN
    # "Any good bets tonight?"
    # "Where's the value in NHL tonight?"
    # "Find me +EV plays"
    # =====================================================================

    def value_scan(self, sport: str = "NHL", min_ev: float = 0.02, bankroll: float = 10000.0) -> dict[str, Any]:
        """Scan all games for +EV opportunities with Kelly sizing."""
        self.toolkit.refresh_data(source="nhl")

        value_result = self.toolkit.find_value(sport=sport, min_ev=min_ev)
        opportunities = value_result.get("opportunities", [])

        sized_bets = []
        total_recommended = 0.0

        for opp in opportunities[:10]:
            prob = opp.get("model_probability", 0.5)
            odds = opp.get("decimal_odds", 2.0)
            sizing = self.toolkit.kelly_sizing(prob, odds, bankroll)
            prediction = (
                self.toolkit.nhl_predict_game(opp.get("game_id", ""))
                if sport.upper() == "NHL"
                else self.toolkit.predict_game(opp.get("game_id", ""))
            )
            confidence_meta = prediction.get("confidence", {})

            sized_bets.append({
                "game_id": opp.get("game_id"),
                "team": opp.get("team"),
                "side": opp.get("selection"),
                "sportsbook": opp.get("sportsbook"),
                "odds": odds,
                "model_prob": prob,
                "implied_prob": opp.get("implied_probability"),
                "edge": opp.get("edge"),
                "ev_per_dollar": opp.get("expected_value"),
                "recommended_stake": sizing.get("recommended_stake", 0),
                "kelly_fraction": sizing.get("fraction", 0),
                "confidence_grade": confidence_meta.get("reliability_grade", "unknown"),
                "confidence_interval": confidence_meta.get("confidence_interval"),
                "defaulted_features": prediction.get("defaulted_features"),
                "requires_manual_review": confidence_meta.get("reliability_grade") in {"D", "F"},
            })
            total_recommended += sizing.get("recommended_stake", 0)

        return {
            "sport": sport,
            "games_scanned": value_result.get("games_scanned", 0),
            "opportunities_found": len(opportunities),
            "sized_bets": sized_bets,
            "total_recommended_stake": round(total_recommended, 2),
            "bankroll": bankroll,
            "bankroll_pct_at_risk": round(total_recommended / bankroll * 100, 1) if bankroll > 0 else 0,
        }

    # =====================================================================
    # CAP STRATEGY
    # "Can the Panthers afford to make a trade?"
    # "What's the Rangers' cap situation?"
    # "Who can they trade at the deadline?"
    # =====================================================================

    def cap_strategy(self, team: str) -> dict[str, Any]:
        """Full cap analysis with trade deadline strategy."""
        self.toolkit.refresh_data(source="nhl", team=team)
        season = getattr(self.toolkit, "_CURRENT_SEASON", "2025-26")

        cap = self.toolkit.query_cap_data(team=team, season=season)
        roster = self.toolkit.query_roster(team=team, season=season)
        playoff = self.toolkit.playoff_odds(team_id=team, season=season)

        analysis = cap.get("cap_analysis", {})
        contracts = cap.get("contracts", [])
        players = roster.get("players", [])

        # Identify trade chips (expiring contracts on non-contending teams)
        is_contender = (playoff.get("wildcard_probability", 0) > 0.5)

        expendable = []
        if is_contender:
            # Contenders look to add -- find cap they can move
            for c in contracts:
                if c.get("contract_years_remaining", 0) <= 1 and c.get("cap_hit", 0) < 3_000_000:
                    expendable.append({"name": c["name"], "cap_hit": c.get("cap_hit", 0), "reason": "expiring depth piece"})
        else:
            # Sellers look to move UFAs for picks
            for c in contracts:
                if c.get("contract_years_remaining", 0) <= 1 and c.get("cap_hit", 0) > 2_000_000:
                    expendable.append({"name": c["name"], "cap_hit": c.get("cap_hit", 0), "reason": "rental -- sell for picks"})

        # Biggest contracts
        top_contracts = sorted(contracts, key=lambda x: x.get("cap_hit", 0), reverse=True)[:5]

        return {
            "team": team,
            "mode": "buyer" if is_contender else "seller",
            "cap_ceiling": analysis.get("cap_ceiling", 88_000_000),
            "total_cap_hit": analysis.get("total_cap_hit", 0),
            "cap_space": analysis.get("cap_space", 0),
            "effective_space": analysis.get("effective_cap_space", 0),
            "cap_health": analysis.get("cap_health", ""),
            "top_contracts": [
                {"name": c["name"], "position": c.get("position"), "cap_hit": c.get("cap_hit", 0),
                 "years_left": c.get("contract_years_remaining", 0)}
                for c in top_contracts
            ],
            "trade_chips": expendable,
            "expiring_total": analysis.get("expiring_cap_total", 0),
            "dead_cap": analysis.get("dead_cap", 0),
            "ltir_relief": analysis.get("ltir_relief", 0),
            "playoff_probability": playoff.get("wildcard_probability", 0),
            "strategy_summary": (
                f"{'Buyer' if is_contender else 'Seller'} mode. "
                f"{'$' + str(int(analysis.get('effective_cap_space', 0)/1_000_000)) + 'M effective space.' if is_contender else ''} "
                f"{len(expendable)} {'pieces to move for space' if is_contender else 'rentals to sell for picks'}."
            ),
        }

    # =====================================================================
    # PLAYOFF RACE
    # "Will the Rangers make the playoffs?"
    # "How tight is the Eastern Conference race?"
    # "What does the playoff picture look like?"
    # =====================================================================

    def playoff_race(self, team: str | None = None, conference: str | None = None) -> dict[str, Any]:
        """Playoff race analysis with standings context."""
        self.toolkit.refresh_data(source="nhl")
        season = getattr(self.toolkit, "_CURRENT_SEASON", "2025-26")

        all_teams = self.toolkit.query_team_stats(sport="NHL", season=season)
        teams_data = all_teams.get("teams", [])

        # Build standings
        standings = []
        for t in teams_data:
            stats = t.get("stats", {})
            tid = t.get("team_id", "")
            playoff = self.toolkit.playoff_odds(team_id=tid, season=season)

            standings.append({
                "team": tid,
                "points": stats.get("points", 0),
                "games_played": stats.get("games_played", stats.get("wins", 0) + stats.get("losses", 0) + stats.get("overtime_losses", 0)),
                "wins": stats.get("wins", 0),
                "losses": stats.get("losses", 0),
                "otl": stats.get("overtime_losses", 0),
                "goal_diff": stats.get("goal_differential", stats.get("goals_for", 0) - stats.get("goals_against", 0)),
                "pts_pct": stats.get("points_pct", 0),
                "projected_points": playoff.get("projected_points", 0),
                "playoff_prob": playoff.get("wildcard_probability", 0),
                "status": playoff.get("status", ""),
                "conference": stats.get("conference_name", ""),
                "division": stats.get("division_name", ""),
                "recent_form": stats.get("recent_form", 0),
            })

        standings.sort(key=lambda x: x["points"], reverse=True)

        # Focus on specific team if requested
        team_detail = None
        if team:
            team_detail = next((s for s in standings if s["team"] == team), None)

        # Filter by conference if requested
        if conference:
            standings = [s for s in standings if conference.lower() in s.get("conference", "").lower()]

        # Bubble teams (within 10 points of wildcard line)
        wildcard_line = standings[15]["points"] if len(standings) > 15 else 0
        bubble = [s for s in standings if abs(s["points"] - wildcard_line) <= 10]

        return {
            "standings": standings[:20],
            "bubble_teams": bubble,
            "wildcard_cutline_approx": wildcard_line,
            "team_focus": team_detail,
            "updated": date.today().isoformat(),
        }

    # =====================================================================
    # GOALTENDER DUEL
    # "Compare Shesterkin vs Bobrovsky"
    # "Who's the better goalie, Swayman or Oettinger?"
    # =====================================================================

    def goaltender_duel(self, goalie1_team: str, goalie2_team: str) -> dict[str, Any]:
        """Head-to-head goaltender comparison."""
        season = getattr(self.toolkit, "_CURRENT_SEASON", "2025-26")
        g1 = self.toolkit.query_goaltender_stats(team=goalie1_team, season=season)
        g2 = self.toolkit.query_goaltender_stats(team=goalie2_team, season=season)
        g1_data, g1_selection = _select_goalie_profile(g1)
        g2_data, g2_selection = _select_goalie_profile(g2)

        matchup = self.toolkit.hockey.goaltender_matchup_edge(
            g1_data.get("save_pct", 0.907), g2_data.get("save_pct", 0.907),
            g1_data.get("gsaa", 0), g2_data.get("gsaa", 0),
        )

        def _goalie_profile(stats: dict, selection_method: str) -> dict:
            return {
                "name": stats.get("name", ""),
                "sv_pct": stats.get("save_pct", 0),
                "gaa": stats.get("gaa", 0),
                "gsaa": stats.get("gsaa", 0),
                "xgsaa": stats.get("xgsaa", 0),
                "games": stats.get("games_played", 0),
                "quality_starts": stats.get("quality_starts", 0),
                "shutouts": stats.get("shutouts", 0),
                "selection_method": selection_method,
                "situational_splits": _goalie_situational_splits(stats),
            }

        categories_won = {"goalie1": 0, "goalie2": 0}
        for metric in ["save_pct", "gsaa", "xgsaa"]:
            if g1_data.get(metric, 0) > g2_data.get(metric, 0):
                categories_won["goalie1"] += 1
            else:
                categories_won["goalie2"] += 1
        for metric in ["gaa"]:  # lower is better
            if g1_data.get(metric, 99) < g2_data.get(metric, 99):
                categories_won["goalie1"] += 1
            else:
                categories_won["goalie2"] += 1

        return {
            "goalie1": _goalie_profile(g1_data, g1_selection),
            "goalie2": _goalie_profile(g2_data, g2_selection),
            "matchup_edge": matchup,
            "categories_won": categories_won,
            "verdict": (
                f"{g1_data.get('name', goalie1_team)} has the edge"
                if matchup["goaltender_edge"] > 0.05 else
                f"{g2_data.get('name', goalie2_team)} has the edge"
                if matchup["goaltender_edge"] < -0.05 else
                "Too close to call"
            ),
        }

    # =====================================================================
    # TEAM COMPARISON
    # "Compare the Rangers and Bruins"
    # "NYR vs BOS season comparison"
    # =====================================================================

    def team_comparison(self, team1: str, team2: str) -> dict[str, Any]:
        """Side-by-side team comparison across all dimensions."""
        self.toolkit.refresh_data(source="nhl")
        season = getattr(self.toolkit, "_CURRENT_SEASON", "2025-26")

        r1 = self.toolkit.season_review(team_id=team1, season=season)
        r2 = self.toolkit.season_review(team_id=team2, season=season)
        a1 = self.toolkit.query_advanced_stats(team_id=team1, season=season)
        a2 = self.toolkit.query_advanced_stats(team_id=team2, season=season)
        g1 = self.toolkit.query_goaltender_stats(team=team1, season=season)
        g2 = self.toolkit.query_goaltender_stats(team=team2, season=season)
        r1_roster = self.toolkit.query_roster(team=team1, season=season)
        r2_roster = self.toolkit.query_roster(team=team2, season=season)

        a1_stats = a1.get("teams", [{}])[0].get("stats", {}) if a1.get("count") else {}
        a2_stats = a2.get("teams", [{}])[0].get("stats", {}) if a2.get("count") else {}

        categories = {
            "record": {"team1": r1.get("points", 0), "team2": r2.get("points", 0), "higher_better": True},
            "goal_diff": {"team1": r1.get("goal_differential", 0), "team2": r2.get("goal_differential", 0), "higher_better": True},
            "corsi": {"team1": a1_stats.get("corsi_pct", 50), "team2": a2_stats.get("corsi_pct", 50), "higher_better": True},
            "xgf": {"team1": a1_stats.get("xgf_pct", 50), "team2": a2_stats.get("xgf_pct", 50), "higher_better": True},
            "pp": {"team1": r1.get("pp_pct", 0), "team2": r2.get("pp_pct", 0), "higher_better": True},
            "pk": {"team1": r1.get("pk_pct", 0), "team2": r2.get("pk_pct", 0), "higher_better": True},
        }

        t1_wins = sum(1 for c in categories.values()
                      if (c["team1"] > c["team2"]) == c["higher_better"])
        t2_wins = len(categories) - t1_wins

        return {
            "team1": {"abbrev": team1, "record": r1.get("record", ""), "points": r1.get("points", 0)},
            "team2": {"abbrev": team2, "record": r2.get("record", ""), "points": r2.get("points", 0)},
            "categories": categories,
            "category_wins": {team1: t1_wins, team2: t2_wins},
            "goaltending": {
                team1: _goalie_team_summary(g1.get("goaltenders", [])),
                team2: _goalie_team_summary(g2.get("goaltenders", [])),
            },
            "roster_depth": {
                team1: _roster_summary(r1_roster.get("players", [])),
                team2: _roster_summary(r2_roster.get("players", [])),
            },
            "analytics_context": {
                team1: _analytics_context(a1_stats),
                team2: _analytics_context(a2_stats),
            },
            "review1": r1,
            "review2": r2,
        }

    # =====================================================================
    # BETTING STRATEGY DISTILLATION
    # "What's my betting strategy for tonight?"
    # "Give me a disciplined approach to NHL betting"
    # "Build me a betting plan for the week"
    # =====================================================================

    def betting_strategy(
        self, sport: str = "NHL", bankroll: float = 10000.0,
        risk_tolerance: str = "moderate", timeframe: str = "tonight",
    ) -> dict[str, Any]:
        """Distilled betting strategy with discipline rules.

        Combines value scanning, Kelly sizing, and risk management
        into a coherent strategy document an agent can present.
        """
        self.toolkit.refresh_data(source="nhl")

        # Adjust parameters by risk tolerance
        kelly_mult = {"conservative": 0.25, "moderate": 0.5, "aggressive": 0.75}.get(risk_tolerance, 0.5)
        min_ev = {"conservative": 0.05, "moderate": 0.03, "aggressive": 0.02}.get(risk_tolerance, 0.03)
        max_daily_risk = {"conservative": 0.03, "moderate": 0.05, "aggressive": 0.10}.get(risk_tolerance, 0.05)

        # Scan for value
        value = self.toolkit.find_value(sport=sport, min_ev=min_ev)
        opps = value.get("opportunities", [])

        # Size bets
        plays = []
        total_stake = 0.0
        max_stake = bankroll * max_daily_risk

        for opp in opps:
            if total_stake >= max_stake:
                break

            prob = opp.get("model_probability", 0.5)
            odds = opp.get("decimal_odds", 2.0)
            sizing = self.toolkit.kelly_sizing(prob, odds, bankroll)
            prediction = (
                self.toolkit.nhl_predict_game(opp.get("game_id", ""))
                if sport.upper() == "NHL"
                else self.toolkit.predict_game(opp.get("game_id", ""))
            )
            confidence_meta = prediction.get("confidence", {})

            stake = sizing.get("recommended_stake", 0) * (kelly_mult / 0.5)  # adjust for risk
            stake = min(stake, max_stake - total_stake)  # don't exceed daily limit
            if stake <= 0:
                continue

            plays.append({
                "game": f"{opp.get('home_team', '')} vs {opp.get('away_team', '')}",
                "team": opp.get("team"),
                "side": opp.get("selection"),
                "book": opp.get("sportsbook"),
                "odds": odds,
                "model_prob": round(prob, 3),
                "edge": round(opp.get("edge", 0), 3),
                "ev_per_dollar": round(opp.get("expected_value", 0), 3),
                "stake": round(stake, 2),
                "to_win": round(stake * (odds - 1), 2),
                "confidence_grade": confidence_meta.get("reliability_grade", "unknown"),
                "defaulted_features": prediction.get("defaulted_features"),
                "requires_manual_review": confidence_meta.get("reliability_grade") in {"D", "F"},
            })
            total_stake += stake

        # Expected daily P&L
        expected_profit = sum(p["stake"] * p["ev_per_dollar"] for p in plays)

        return {
            "strategy": {
                "risk_tolerance": risk_tolerance,
                "kelly_multiplier": kelly_mult,
                "min_ev_threshold": min_ev,
                "max_daily_risk_pct": max_daily_risk * 100,
                "bankroll": bankroll,
            },
            "discipline_rules": [
                f"Max {max_daily_risk*100:.0f}% of bankroll at risk per day (${max_stake:.0f})",
                f"Using {kelly_mult:.0%} Kelly (half-Kelly = aggressive, quarter-Kelly = conservative)",
                f"Only play edges >{min_ev*100:.0f}% EV",
                "Never chase losses -- stick to the model",
                "Track every bet for long-term ROI validation",
                "If model accuracy drops below 55%, pause and investigate",
            ],
            "todays_plays": plays,
            "play_count": len(plays),
            "total_stake": round(total_stake, 2),
            "total_to_win": round(sum(p["to_win"] for p in plays), 2),
            "expected_profit": round(expected_profit, 2),
            "games_scanned": value.get("games_scanned", 0),
            "manual_review_play_count": sum(1 for p in plays if p["requires_manual_review"]),
        }

    # =====================================================================
    # HELPERS
    # =====================================================================

    def _head_to_head(self, home: str, away: str) -> dict[str, Any]:
        """Look up the season series between two teams."""
        home_finished = self.toolkit.query_games(sport="NHL", team=home, status="final")
        games = home_finished.get("games", [])

        # Filter for games where the opponent is the other team
        h2h_games = [
            g for g in games
            if (g["home_team"] == home and g["away_team"] == away)
            or (g["home_team"] == away and g["away_team"] == home)
        ]

        home_wins = 0
        away_wins = 0
        last_meeting = ""

        for g in h2h_games:
            if _is_win(g, home):
                home_wins += 1
            else:
                away_wins += 1
            game_date = g.get("date", "")
            if game_date and (not last_meeting or str(game_date) > last_meeting):
                last_meeting = str(game_date)[:10]

        return {
            "games_played": len(h2h_games),
            "home_team_wins": home_wins,
            "away_team_wins": away_wins,
            "last_meeting": last_meeting,
        }

    def _compute_rest(self, home: str, away: str) -> dict[str, Any]:
        """Compute rest days and B2B status from recent games."""
        today = date.today()

        home_games = self.toolkit.query_games(sport="NHL", team=home, status="final")
        away_games = self.toolkit.query_games(sport="NHL", team=away, status="final")

        def _days_since_last(games_result: dict) -> int:
            games = games_result.get("games", [])
            if not games:
                return 3
            last_date_str = games[0].get("date", "")
            if not last_date_str:
                return 3
            try:
                last = date.fromisoformat(str(last_date_str)[:10])
                return (today - last).days
            except (ValueError, TypeError):
                return 3

        home_rest = _days_since_last(home_games)
        away_rest = _days_since_last(away_games)

        return {
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "home_b2b": home_rest <= 1,
            "away_b2b": away_rest <= 1,
        }


# --- Standalone helpers ---

def _select_target_game(games: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the nearest scheduled game, not an arbitrary first row."""
    if not games:
        return None
    return min(games, key=lambda g: (str(g.get("date", "")), str(g.get("game_id", ""))))


def _confidence_sort_key(confidence_meta: dict[str, Any]) -> float:
    """Numeric sort key derived from workflow reliability metadata."""
    if not isinstance(confidence_meta, dict):
        return 0.0
    grade_score = {"A": 90.0, "B": 75.0, "C": 60.0, "D": 35.0, "F": 10.0}.get(
        confidence_meta.get("reliability_grade"),
        0.0,
    )
    interval = confidence_meta.get("confidence_interval", {})
    width = interval.get("width", 1.0) if isinstance(interval, dict) else 1.0
    width_penalty = min(max(float(width), 0.0), 1.0) * 20.0
    return round(max(grade_score - width_penalty, 0.0), 1)


def _evaluate_market_sides(
    toolkit: ABBAToolkit,
    home_team: str,
    away_team: str,
    home_win_prob: float,
    odds: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate both market sides and pick the better positive-EV option."""
    sides = []
    candidates = [
        ("home", home_team, home_win_prob, odds.get("best_home")),
        ("away", away_team, 1.0 - home_win_prob, odds.get("best_away")),
    ]

    for side, team, probability, market in candidates:
        if not market or not market.get("odds"):
            continue
        ev = toolkit.calculate_ev(probability, market["odds"])
        kelly = toolkit.kelly_sizing(probability, market["odds"]) if ev.get("is_positive_ev") else None
        sides.append({
            "side": side,
            "team": team,
            "sportsbook": market.get("sportsbook", ""),
            "odds": market["odds"],
            "probability": probability,
            "implied_probability": round(1.0 / market["odds"], 4),
            "ev": ev,
            "kelly": kelly,
        })

    positive = [side for side in sides if side.get("ev", {}).get("is_positive_ev")]
    best = max(positive, key=lambda side: side["ev"].get("expected_value", 0.0)) if positive else None
    return {"sides": sides, "best_bet": best}


def _find_named_goalie(goalies_result: dict[str, Any], goalie_name: str | None) -> dict[str, Any] | None:
    """Resolve the specific goalie used by the model from a query result set."""
    goalies = goalies_result.get("goaltenders", [])
    if goalie_name:
        for goalie in goalies:
            stats = goalie.get("stats", {})
            if stats.get("name") == goalie_name:
                return stats
    return _select_goalie_profile(goalies_result)[0]


def _select_goalie_profile(goalies_result: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Select a goalie profile deterministically for display."""
    goalies = goalies_result.get("goaltenders", [])
    if not goalies:
        return {}, "none"
    for goalie in goalies:
        stats = goalie.get("stats", {})
        if stats.get("role") == "starter":
            return stats, "role_tag"
    best = max(goalies, key=lambda g: g.get("stats", {}).get("games_played", 0))
    return best.get("stats", {}), "max_games_played"


def _build_team_snapshot(
    team: str,
    record: str,
    stats: dict[str, Any],
    advanced: dict[str, Any],
    roster: list[dict[str, Any]],
    starter: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compact workflow-facing snapshot of a team's current state."""
    roster_summary = _roster_summary(roster)
    return {
        "abbrev": team,
        "record": record,
        "points": stats.get("points", 0),
        "recent_form": stats.get("recent_form", ""),
        "starter": starter.get("name", "TBD") if starter else "TBD",
        "starter_sv_pct": starter.get("save_pct", 0) if starter else 0,
        "special_teams": {
            "pp_pct": stats.get("power_play_percentage", 0),
            "pk_pct": stats.get("penalty_kill_percentage", 0),
        },
        "analytics": _analytics_context(advanced),
        "roster_depth": roster_summary,
    }


def _analytics_context(advanced: dict[str, Any]) -> dict[str, Any]:
    """Return only the advanced metrics that actually exist in storage."""
    return {
        "corsi_pct": advanced.get("corsi_pct"),
        "xgf_pct": advanced.get("xgf_pct"),
        "pdo": advanced.get("pdo"),
        "shooting_pct": advanced.get("shooting_pct"),
        "save_pct_5v5": advanced.get("save_pct_5v5"),
    }


def _build_analytics_note(team: str, advanced: dict[str, Any]) -> str | None:
    """Human-readable analytics note for context-only workflow output."""
    if not advanced:
        return None
    notes = []
    corsi = advanced.get("corsi_pct")
    xgf = advanced.get("xgf_pct")
    if corsi is not None:
        notes.append(f"{team} Corsi {corsi:.1f}%")
    if xgf is not None:
        notes.append(f"xGF {xgf:.1f}%")
    return " | ".join(notes) if notes else None


def _roster_summary(players: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize top-end production, injuries, and line distribution."""
    if not players:
        return {"top_scorers": [], "injury_count": 0, "line_distribution": {}}

    sorted_players = sorted(
        players,
        key=lambda player: (
            player.get("stats", {}).get("points", 0),
            player.get("stats", {}).get("goals", 0),
        ),
        reverse=True,
    )
    top_scorers = [
        {
            "name": player.get("name", ""),
            "position": player.get("position", ""),
            "line_number": player.get("line_number"),
            "points": player.get("stats", {}).get("points", 0),
            "goals": player.get("stats", {}).get("goals", 0),
            "assists": player.get("stats", {}).get("assists", 0),
        }
        for player in sorted_players[:3]
    ]

    injured = [
        {"name": player.get("name", ""), "status": player.get("injury_status", "unknown")}
        for player in players
        if player.get("injury_status", "healthy") != "healthy"
    ]

    line_distribution: dict[str, int] = {}
    for player in players:
        line_number = player.get("line_number")
        if line_number is None:
            continue
        key = str(line_number)
        line_distribution[key] = line_distribution.get(key, 0) + 1

    return {
        "top_scorers": top_scorers,
        "injury_count": len(injured),
        "injuries": injured[:5],
        "line_distribution": line_distribution,
    }


def _build_roster_note(team: str, players: list[dict[str, Any]]) -> str | None:
    """Context-only note summarizing roster health and top-end production."""
    summary = _roster_summary(players)
    top = summary.get("top_scorers", [])
    if not top and not summary.get("injury_count"):
        return None

    note_parts = []
    if top:
        leaders = ", ".join(f"{player['name']} ({player['points']} pts)" for player in top[:2])
        note_parts.append(f"{team} scoring leaders: {leaders}")
    if summary.get("injury_count"):
        note_parts.append(f"{summary['injury_count']} injured skaters on current roster")
    return " | ".join(note_parts)


def _goalie_situational_splits(stats: dict[str, Any]) -> dict[str, Any]:
    """Expose situation split fields only when they really exist."""
    split_keys = {
        "even_strength_save_pct": "even_strength_save_pct",
        "power_play_save_pct": "power_play_save_pct",
        "short_handed_save_pct": "short_handed_save_pct",
    }
    available = {
        alias: stats[key]
        for alias, key in split_keys.items()
        if stats.get(key) is not None
    }
    return {
        "status": "present" if available else "absent",
        "splits": available if available else None,
    }


def _goalie_team_summary(goalies: list[dict[str, Any]]) -> dict[str, Any]:
    """Simple team-level goalie summary for comparison workflows."""
    if not goalies:
        return {"starter": None, "backup": None}
    ordered = sorted(
        (goalie.get("stats", {}) for goalie in goalies),
        key=lambda stats: (stats.get("role") == "starter", stats.get("games_played", 0)),
        reverse=True,
    )
    return {
        "starter": ordered[0] if ordered else None,
        "backup": ordered[1] if len(ordered) > 1 else None,
    }


def _is_win(game: dict, team: str) -> bool:
    """Did the team win this game?"""
    if game["home_team"] == team:
        return (game.get("home_score", 0) or 0) > (game.get("away_score", 0) or 0)
    else:
        return (game.get("away_score", 0) or 0) > (game.get("home_score", 0) or 0)


def _analytics_narrative(adv: dict) -> str:
    """Generate a narrative sentence from advanced stats."""
    cf = adv.get("corsi_pct", 50)
    xgf = adv.get("xgf_pct", 50)
    pdo = adv.get("pdo", 100)

    parts = []
    if cf > 53:
        parts.append("dominant possession team")
    elif cf > 51:
        parts.append("above-average possession")
    elif cf < 47:
        parts.append("struggling to control play")
    elif cf < 49:
        parts.append("slightly outpossessed")
    else:
        parts.append("average possession")

    if xgf > 53:
        parts.append("generating high-quality chances")
    elif xgf < 47:
        parts.append("not creating enough quality scoring chances")

    if pdo > 102:
        parts.append("running hot (high PDO -- potential regression)")
    elif pdo < 98:
        parts.append("running cold (low PDO -- expect improvement)")

    return "; ".join(parts) if parts else "analytics within normal range"


# --- Public API ---

def run_workflow(name: str, toolkit: ABBAToolkit | None = None, **params: Any) -> dict[str, Any]:
    """Convenience function to run a workflow."""
    engine = WorkflowEngine(toolkit)
    return engine.run(name, **params)


def list_workflows() -> list[dict[str, Any]]:
    """List available workflows for agent discovery."""
    return [
        {
            "name": "game_prediction",
            "triggers": ["who wins", "predict", "game prediction", "matchup", "tonight's game"],
            "description": "Full game prediction with 6-model NHL analysis, goaltender matchup, rest factors, odds comparison, and Kelly sizing",
            "params": {"team": "team abbreviation (e.g., NYR)", "game_id": "specific game ID (optional)"},
        },
        {
            "name": "tonights_slate",
            "triggers": ["tonight", "slate", "schedule", "what's on", "all games", "full card"],
            "description": "All scheduled games with predictions, odds, and value picks",
            "params": {"sport": "NHL or MLB (default NHL)"},
        },
        {
            "name": "season_story",
            "triggers": ["season", "how are they doing", "story", "overview", "full picture", "tell me about"],
            "description": "Comprehensive season narrative: record, analytics grades, goaltending, cap, playoff outlook, recent results",
            "params": {"team": "team abbreviation (e.g., BOS)"},
        },
        {
            "name": "value_scan",
            "triggers": ["value", "bets", "+EV", "edges", "where's the value", "good bets"],
            "description": "Scan all games for +EV opportunities with Kelly sizing",
            "params": {"sport": "NHL", "min_ev": "minimum EV threshold (default 0.02)", "bankroll": "default 10000"},
        },
        {
            "name": "cap_strategy",
            "triggers": ["cap", "salary", "trade", "deadline", "afford", "contracts"],
            "description": "Salary cap analysis with trade deadline strategy -- buyer vs seller, trade chips, flexibility",
            "params": {"team": "team abbreviation"},
        },
        {
            "name": "playoff_race",
            "triggers": ["playoffs", "standings", "race", "bubble", "clinch", "eliminated", "make the playoffs"],
            "description": "Playoff race analysis with Monte Carlo projections for all teams",
            "params": {"team": "focus team (optional)", "conference": "Eastern/Western (optional)"},
        },
        {
            "name": "goaltender_duel",
            "triggers": ["compare goalies", "goaltender", "goalie vs", "better goalie", "netminder"],
            "description": "Head-to-head goaltender comparison (Sv%, GSAA, xGSAA, matchup edge)",
            "params": {"goalie1_team": "team abbreviation", "goalie2_team": "team abbreviation"},
        },
        {
            "name": "team_comparison",
            "triggers": ["compare teams", "vs", "head to head", "which team", "better team"],
            "description": "Side-by-side team comparison across record, analytics, special teams, and goaltending",
            "params": {"team1": "team abbreviation", "team2": "team abbreviation"},
        },
        {
            "name": "betting_strategy",
            "triggers": ["strategy", "betting plan", "discipline", "approach", "system", "methodology"],
            "description": "Distilled betting strategy with risk management, Kelly sizing, and discipline rules",
            "params": {"bankroll": "default 10000", "risk_tolerance": "conservative/moderate/aggressive"},
        },
    ]
