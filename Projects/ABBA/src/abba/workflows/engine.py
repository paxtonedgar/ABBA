"""Workflow engine -- chains tool calls into multi-step analytical pipelines.

Each workflow:
1. Refreshes live data from APIs
2. Runs a structured chain of calculations
3. Returns a narrative-ready result dict that agents present to users

Workflows handle the "thinking" so the agent just presents results.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any

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
        return result

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

        # 2. Find the game
        if game_id:
            games = self.toolkit.query_games(sport="NHL")
            game = next((g for g in games.get("games", []) if g.get("game_id") == game_id), None)
        elif team:
            games = self.toolkit.query_games(sport="NHL", team=team, status="scheduled")
            game = games.get("games", [None])[0] if games.get("count", 0) > 0 else None
        else:
            return {"error": "provide team or game_id"}

        if not game:
            return {"error": f"no scheduled game found for {team or game_id}"}

        gid = game["game_id"]
        home = game["home_team"]
        away = game["away_team"]

        # 3. Team profiles
        home_stats = self.toolkit.query_team_stats(team_id=home, sport="NHL")
        away_stats = self.toolkit.query_team_stats(team_id=away, sport="NHL")
        home_adv = self.toolkit.query_advanced_stats(team_id=home)
        away_adv = self.toolkit.query_advanced_stats(team_id=away)
        home_goalies = self.toolkit.query_goaltender_stats(team=home)
        away_goalies = self.toolkit.query_goaltender_stats(team=away)

        # 4. Rest/B2B detection
        rest_info = self._compute_rest(home, away)

        # 5. Run prediction
        prediction = self.toolkit.nhl_predict_game(gid)

        # 6. Odds and value
        odds = self.toolkit.compare_odds(gid)
        pred_value = prediction.get("prediction", {}).get("value", 0.5)

        ev_result = None
        kelly_result = None
        if odds.get("best_home") and odds["best_home"].get("odds"):
            ev_result = self.toolkit.calculate_ev(pred_value, odds["best_home"]["odds"])
            if ev_result.get("is_positive_ev"):
                kelly_result = self.toolkit.kelly_sizing(pred_value, odds["best_home"]["odds"])

        # 7. Build narrative
        hs = home_stats.get("teams", [{}])[0].get("stats", {}) if home_stats.get("count") else {}
        as_ = away_stats.get("teams", [{}])[0].get("stats", {}) if away_stats.get("count") else {}

        home_record = f"{hs.get('wins', 0)}-{hs.get('losses', 0)}-{hs.get('overtime_losses', 0)}"
        away_record = f"{as_.get('wins', 0)}-{as_.get('losses', 0)}-{as_.get('overtime_losses', 0)}"

        home_starter = _find_starter(home_goalies)
        away_starter = _find_starter(away_goalies)

        narrative = {
            "headline": f"{home} ({home_record}) vs {away} ({away_record})",
            "home_team": {
                "abbrev": home,
                "record": home_record,
                "points": hs.get("points", 0),
                "recent_form": hs.get("recent_form", ""),
                "starter": home_starter.get("name", "TBD") if home_starter else "TBD",
                "starter_sv_pct": home_starter.get("save_pct", 0) if home_starter else 0,
            },
            "away_team": {
                "abbrev": away,
                "record": away_record,
                "points": as_.get("points", 0),
                "recent_form": as_.get("recent_form", ""),
                "starter": away_starter.get("name", "TBD") if away_starter else "TBD",
                "starter_sv_pct": away_starter.get("save_pct", 0) if away_starter else 0,
            },
            "prediction": prediction.get("prediction", {}),
            "features": prediction.get("features", {}),
            "rest": rest_info,
            "odds": odds,
            "ev": ev_result,
            "sizing": kelly_result,
        }

        # Key factors narrative
        factors = []
        if pred_value > 0.55:
            factors.append(f"{home} favored at {pred_value:.0%}")
        elif pred_value < 0.45:
            factors.append(f"{away} favored at {1-pred_value:.0%}")
        else:
            factors.append("Toss-up game")

        if home_starter and home_starter.get("save_pct", 0) > 0.920:
            factors.append(f"Elite goaltending: {home_starter['name']} ({home_starter['save_pct']:.3f} Sv%)")
        if away_starter and away_starter.get("save_pct", 0) > 0.920:
            factors.append(f"Elite goaltending: {away_starter['name']} ({away_starter['save_pct']:.3f} Sv%)")

        if rest_info.get("home_b2b"):
            factors.append(f"{home} on a back-to-back (fatigue penalty)")
        if rest_info.get("away_b2b"):
            factors.append(f"{away} on a back-to-back (fatigue penalty)")

        if ev_result and ev_result.get("is_positive_ev"):
            factors.append(f"+EV opportunity: {ev_result['expected_value']:.1%} edge")

        narrative["key_factors"] = factors
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

            entry = {
                "game_id": gid,
                "matchup": f"{away} @ {home}",
                "home_team": home,
                "away_team": away,
                "home_win_prob": round(pred_val, 3),
                "away_win_prob": round(1 - pred_val, 3),
                "pick": home if pred_val > 0.5 else away,
                "confidence": round(abs(pred_val - 0.5) * 200, 1),  # 0-100 scale
                "best_home_odds": best_home.get("odds") if best_home else None,
                "best_away_odds": best_away.get("odds") if best_away else None,
            }

            # Check for value
            if best_home and best_home.get("odds"):
                ev = self.toolkit.calculate_ev(pred_val, best_home["odds"])
                if ev.get("is_positive_ev"):
                    entry["value_bet"] = {
                        "side": "home",
                        "team": home,
                        "ev": ev["expected_value"],
                        "edge": ev["edge"],
                        "book": best_home.get("sportsbook", ""),
                    }
                    value_picks.append(entry["value_bet"])

            if best_away and best_away.get("odds"):
                ev = self.toolkit.calculate_ev(1 - pred_val, best_away["odds"])
                if ev.get("is_positive_ev"):
                    entry["value_bet"] = {
                        "side": "away",
                        "team": away,
                        "ev": ev["expected_value"],
                        "edge": ev["edge"],
                        "book": best_away.get("sportsbook", ""),
                    }
                    value_picks.append(entry["value_bet"])

            slate.append(entry)

        # Sort by confidence
        slate.sort(key=lambda x: x["confidence"], reverse=True)

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

        cap = self.toolkit.query_cap_data(team=team)
        roster = self.toolkit.query_roster(team=team)
        playoff = self.toolkit.playoff_odds(team_id=team)

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

        all_teams = self.toolkit.query_team_stats(sport="NHL")
        teams_data = all_teams.get("teams", [])

        # Build standings
        standings = []
        for t in teams_data:
            stats = t.get("stats", {})
            tid = t.get("team_id", "")
            playoff = self.toolkit.playoff_odds(team_id=tid)

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
        g1 = self.toolkit.query_goaltender_stats(team=goalie1_team)
        g2 = self.toolkit.query_goaltender_stats(team=goalie2_team)

        g1_starters = [g for g in g1.get("goaltenders", []) if g.get("stats", {}).get("role") == "starter"]
        g2_starters = [g for g in g2.get("goaltenders", []) if g.get("stats", {}).get("role") == "starter"]

        g1_data = g1_starters[0]["stats"] if g1_starters else g1.get("goaltenders", [{}])[0].get("stats", {})
        g2_data = g2_starters[0]["stats"] if g2_starters else g2.get("goaltenders", [{}])[0].get("stats", {})

        matchup = self.toolkit.hockey.goaltender_matchup_edge(
            g1_data.get("save_pct", 0.907), g2_data.get("save_pct", 0.907),
            g1_data.get("gsaa", 0), g2_data.get("gsaa", 0),
        )

        def _goalie_profile(stats: dict) -> dict:
            return {
                "name": stats.get("name", ""),
                "sv_pct": stats.get("save_pct", 0),
                "gaa": stats.get("gaa", 0),
                "gsaa": stats.get("gsaa", 0),
                "xgsaa": stats.get("xgsaa", 0),
                "games": stats.get("games_played", 0),
                "quality_starts": stats.get("quality_starts", 0),
                "shutouts": stats.get("shutouts", 0),
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
            "goalie1": _goalie_profile(g1_data),
            "goalie2": _goalie_profile(g2_data),
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

        r1 = self.toolkit.season_review(team_id=team1)
        r2 = self.toolkit.season_review(team_id=team2)
        a1 = self.toolkit.query_advanced_stats(team_id=team1)
        a2 = self.toolkit.query_advanced_stats(team_id=team2)
        g1 = self.toolkit.query_goaltender_stats(team=team1)
        g2 = self.toolkit.query_goaltender_stats(team=team2)

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
        }

    # =====================================================================
    # HELPERS
    # =====================================================================

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

def _find_starter(goalies_result: dict) -> dict | None:
    """Find the starting goaltender from query results."""
    goalies = goalies_result.get("goaltenders", [])
    for g in goalies:
        if g.get("stats", {}).get("role") == "starter":
            return g.get("stats", {})
    return goalies[0].get("stats", {}) if goalies else None


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
