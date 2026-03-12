"""NHL-specific analytics engine.

Comprehensive hockey analytics: Corsi, Fenwick, expected goals (xG),
goaltender models, special teams, rest/schedule effects, score-state
adjusted stats, salary cap analysis, season reviews, and playoff models.

All formulas use real hockey analytics math -- not LLM guesses.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..types import OddsSnapshot, TeamStatsRecord
from scipy import stats as scipy_stats


class HockeyAnalytics:
    """NHL-specific analytics engine with real hockey math."""

    # xG model coefficients -- defaults from published nhlscraper xG_v2.
    # Override via `load_xg_coefficients()` when you train on real play-by-play.
    XG_COEFFICIENTS: dict[str, Any] = {
        "intercept": -1.9963,
        "distance": -0.0316,      # per foot
        "angle": -0.0081,         # per degree from center
        "rebound": 0.4133,        # ~1.5x odds ratio
        "rush": -0.0658,
        "pp": 0.4090,
        "sh": -0.3500,
        "shot_types": {
            "wrist": 0.0, "snap": 0.05, "slap": -0.15,
            "backhand": -0.30, "tip": 0.35, "wrap": -0.80,
        },
        "source": "nhlscraper_xG_v2",
    }

    @classmethod
    def load_xg_coefficients(cls, path: str) -> None:
        """Load xG coefficients from a JSON file.

        File format: same keys as XG_COEFFICIENTS.
        This allows updating the model when trained on real play-by-play data
        without changing source code.
        """
        import json
        from pathlib import Path
        config = json.loads(Path(path).read_text())
        for key in ("intercept", "distance", "angle", "rebound", "rush", "pp", "sh"):
            if key in config:
                cls.XG_COEFFICIENTS[key] = float(config[key])
        if "shot_types" in config:
            cls.XG_COEFFICIENTS["shot_types"].update(config["shot_types"])
        cls.XG_COEFFICIENTS["source"] = config.get("source", path)

    # --- Shot-based metrics (5v5) ---

    @staticmethod
    def corsi(
        shots_for: int,
        blocks_for: int,
        missed_for: int,
        shots_against: int,
        blocks_against: int,
        missed_against: int,
        minutes_5v5: float = 1.0,
    ) -> dict[str, float]:
        """Corsi (all shot attempts at 5v5).

        CF = shots on goal + blocked shots + missed shots (for)
        CA = same (against)
        CF% = CF / (CF + CA) * 100

        The single best possession proxy in hockey. >50% means you're
        outchancing your opponent at even strength.
        """
        cf = shots_for + blocks_for + missed_for
        ca = shots_against + blocks_against + missed_against
        total = cf + ca

        cf_pct = (cf / total * 100) if total > 0 else 50.0
        cf_per60 = (cf / minutes_5v5 * 60) if minutes_5v5 > 0 else 0.0
        ca_per60 = (ca / minutes_5v5 * 60) if minutes_5v5 > 0 else 0.0
        cf_rel = cf_pct - 50.0  # CF% centered at zero (not true CF%Rel which needs off-ice data)

        return {
            "corsi_for": cf,
            "corsi_against": ca,
            "corsi_pct": round(cf_pct, 2),
            "corsi_for_per60": round(cf_per60, 2),
            "corsi_against_per60": round(ca_per60, 2),
            "corsi_rel": round(cf_rel, 2),
        }

    @staticmethod
    def fenwick(
        shots_for: int,
        missed_for: int,
        shots_against: int,
        missed_against: int,
        minutes_5v5: float = 1.0,
    ) -> dict[str, float]:
        """Fenwick (unblocked shot attempts at 5v5).

        FF = shots on goal + missed shots (for)
        FA = same (against)
        FF% = FF / (FF + FA) * 100

        Like Corsi but excludes blocked shots. Better signal when you
        suspect shot-blocking is inflating Corsi against.
        """
        ff = shots_for + missed_for
        fa = shots_against + missed_against
        total = ff + fa

        ff_pct = (ff / total * 100) if total > 0 else 50.0
        ff_per60 = (ff / minutes_5v5 * 60) if minutes_5v5 > 0 else 0.0
        fa_per60 = (fa / minutes_5v5 * 60) if minutes_5v5 > 0 else 0.0

        return {
            "fenwick_for": ff,
            "fenwick_against": fa,
            "fenwick_pct": round(ff_pct, 2),
            "fenwick_for_per60": round(ff_per60, 2),
            "fenwick_against_per60": round(fa_per60, 2),
        }

    # --- Expected Goals (xG) ---

    @classmethod
    def expected_goals(cls, shots: list[dict[str, Any]]) -> dict[str, Any]:
        """Expected goals model using shot quality.

        Each shot has: distance, angle, shot_type, is_rebound, is_rush,
        strength (even/pp/sh).

        Uses a logistic regression with coefficients from published NHL xG
        models (nhlscraper xG_v2, Evolving Hockey). All adjustments are
        additive in log-odds space, then converted to probability once.

        Reference probabilities (even strength, wrist shot, no rebound/rush):
          10ft, 0°: ~9.0%    20ft, 0°: ~6.7%    30ft, 0°: ~5.0%
          40ft, 0°: ~3.7%    50ft, 0°: ~2.7%    60ft, 0°: ~2.0%
        """
        if not shots:
            return {"xg_total": 0.0, "shot_count": 0, "xg_per_shot": 0.0, "shots": []}

        # Load coefficients from class-level config (overridable via load_xg_coefficients)
        coefs = cls.XG_COEFFICIENTS
        INTERCEPT = coefs["intercept"]
        COEF_DISTANCE = coefs["distance"]
        COEF_ANGLE = coefs["angle"]
        COEF_REBOUND = coefs["rebound"]
        COEF_RUSH = coefs["rush"]
        COEF_PP = coefs["pp"]
        COEF_SH = coefs["sh"]
        SHOT_TYPE_COEFS = coefs["shot_types"]

        shot_details = []
        total_xg = 0.0

        for shot in shots:
            dist = max(shot.get("distance", 30.0), 0.0)
            angle = abs(shot.get("angle", 0.0))
            stype = shot.get("shot_type", "wrist")
            is_rebound = shot.get("is_rebound", False)
            is_rush = shot.get("is_rush", False)
            strength = shot.get("strength", "even")

            # Build log-odds additively
            z = INTERCEPT + COEF_DISTANCE * dist + COEF_ANGLE * angle
            z += SHOT_TYPE_COEFS.get(stype, 0.0)

            if is_rebound:
                z += COEF_REBOUND
            if is_rush:
                z += COEF_RUSH
            if strength == "pp":
                z += COEF_PP
            elif strength == "sh":
                z += COEF_SH

            # Convert log-odds to probability (logistic function)
            # Clamp z to prevent math.exp overflow on extreme inputs
            z = max(-500, min(500, z))
            xg = 1.0 / (1.0 + math.exp(-z))
            total_xg += xg

            shot_details.append({
                "distance": dist,
                "angle": angle,
                "shot_type": stype,
                "xg": round(xg, 4),
            })

        return {
            "xg_total": round(total_xg, 3),
            "shot_count": len(shots),
            "xg_per_shot": round(total_xg / len(shots), 4),
            "shots": shot_details,
        }

    # --- Goaltender model ---

    @staticmethod
    def goaltender_metrics(
        saves: int,
        shots_against: int,
        goals_against: int,
        xg_against: float,
        games_played: int,
        minutes_played: float,
        quality_starts: int = 0,
        shutouts: int = 0,
    ) -> dict[str, float]:
        """Comprehensive goaltender evaluation.

        - Sv%: saves / shots against
        - GAA: goals against * 60 / minutes played
        - GSAA: goals saved above average = (league_avg_sv% * SA) - GA
        - xGSAA: xG against - actual goals against (positive = outperforming)
        - QS%: quality starts / games played
        """
        sv_pct = saves / shots_against if shots_against > 0 else 0.0
        gaa = (goals_against / minutes_played * 60) if minutes_played > 0 else 0.0

        # League average Sv% is approximately 0.907 (2023-24 season)
        league_avg_sv = 0.907
        # GSAA = expected GA at league avg - actual GA. Positive = saved more than average.
        # Expected GA = (1 - league_avg_sv) * SA
        expected_ga = (1 - league_avg_sv) * shots_against
        gsaa = (expected_ga - goals_against) if shots_against > 0 else 0.0

        # xG-based: positive means goalie is outperforming shot quality
        xgsaa = xg_against - goals_against

        qs_pct = quality_starts / games_played if games_played > 0 else 0.0

        return {
            "save_pct": round(sv_pct, 4),
            "gaa": round(gaa, 3),
            "gsaa": round(gsaa, 2),
            "xgsaa": round(xgsaa, 2),
            "quality_start_pct": round(qs_pct, 3),
            "shutouts": shutouts,
            "games_played": games_played,
            "shots_against": shots_against,
            "saves": saves,
        }

    @staticmethod
    def goaltender_matchup_edge(
        starter_sv_pct: float,
        opponent_sv_pct: float,
        starter_gsaa: float,
        opponent_gsaa: float,
    ) -> dict[str, float]:
        """Compare two goaltenders for a game prediction.

        Returns an edge factor (-1 to 1) where positive favors the starter.
        Based on save percentage and GSAA differential.
        """
        sv_diff = starter_sv_pct - opponent_sv_pct
        gsaa_diff = starter_gsaa - opponent_gsaa

        # Calibrated from research: 0.01 Sv% diff on ~30 shots/game = ~0.3 goals
        # = ~3.75% win probability. So 0.01 Sv% -> 0.0375 edge.
        # GSAA is a season cumulative stat with high variance game-to-game,
        # so it gets lower weight. 10 GSAA diff -> ~0.05 edge.
        sv_edge = sv_diff / 0.01 * 0.0375
        gsaa_edge = gsaa_diff / 10.0 * 0.05

        # Weighted: Sv% is more predictive for single-game context
        combined = 0.7 * np.clip(sv_edge, -0.5, 0.5) + 0.3 * np.clip(gsaa_edge, -0.5, 0.5)
        combined = float(np.clip(combined, -0.5, 0.5))

        return {
            "goaltender_edge": round(combined, 4),
            "sv_pct_edge": round(float(np.clip(sv_edge, -1, 1)), 4),
            "gsaa_edge": round(float(np.clip(gsaa_edge, -1, 1)), 4),
            "starter_sv_pct": round(starter_sv_pct, 4),
            "opponent_sv_pct": round(opponent_sv_pct, 4),
        }

    # --- Special teams ---

    @staticmethod
    def special_teams_rating(
        pp_goals: int,
        pp_opportunities: int,
        pk_kills: int,
        pk_times_shorthanded: int,
        pp_shots: int = 0,
        pp_xg: float = 0.0,
        pk_shots_against: int = 0,
        pk_xg_against: float = 0.0,
    ) -> dict[str, float]:
        """Special teams analysis with shot quality metrics.

        PP% and PK% alone miss shot quality. A team can have 20% PP%
        but generate high-danger chances that are being stopped. This
        model adds shot-based metrics to separate process from results.
        """
        pp_pct = (pp_goals / pp_opportunities * 100) if pp_opportunities > 0 else 0.0
        pk_pct = (pk_kills / pk_times_shorthanded * 100) if pk_times_shorthanded > 0 else 0.0

        # Shot quality on PP
        pp_conversion = (pp_goals / pp_shots * 100) if pp_shots > 0 else 0.0
        pp_xg_per_opp = (pp_xg / pp_opportunities) if pp_opportunities > 0 else 0.0

        # PK goals allowed and save percentage
        pk_goals_against = pk_times_shorthanded - pk_kills
        pk_save_pct = (1 - pk_goals_against / pk_shots_against) if pk_shots_against > 0 else 0.0
        pk_xg_against_per_opp = (pk_xg_against / pk_times_shorthanded) if pk_times_shorthanded > 0 else 0.0

        # League averages: PP ~22%, PK ~80%
        pp_above_avg = pp_pct - 22.0
        pk_above_avg = pk_pct - 80.0

        # Combined special teams index (higher = better)
        # Weighted: PK slightly more important historically
        st_index = 0.45 * pp_above_avg + 0.55 * pk_above_avg

        return {
            "pp_pct": round(pp_pct, 2),
            "pk_pct": round(pk_pct, 2),
            "pp_above_avg": round(pp_above_avg, 2),
            "pk_above_avg": round(pk_above_avg, 2),
            "pp_conversion_rate": round(pp_conversion, 2),
            "pp_xg_per_opportunity": round(pp_xg_per_opp, 4),
            "pk_xg_against_per_opportunity": round(pk_xg_against_per_opp, 4),
            "special_teams_index": round(st_index, 2),
        }

    # --- Rest and schedule effects ---

    @staticmethod
    def rest_advantage(
        home_rest_days: int,
        away_rest_days: int,
        home_is_back_to_back: bool,
        away_is_back_to_back: bool,
        home_travel_km: float = 0.0,
        away_travel_km: float = 0.0,
        home_games_last_7: int = 3,
        away_games_last_7: int = 3,
    ) -> dict[str, Any]:
        """Schedule-based advantage/disadvantage.

        Back-to-back games reduce NHL win probability by ~4-5%.
        Travel distance further compounds fatigue.
        Rest advantage (2+ days vs B2B) worth ~3-4% edge.
        """
        # Base rest factor: 0 = neutral, positive = favors home
        home_fatigue = 0.0
        away_fatigue = 0.0

        # Back-to-back penalty: ~0.045 win probability reduction
        if home_is_back_to_back:
            home_fatigue += 0.045
        if away_is_back_to_back:
            away_fatigue += 0.045

        # Rest days bonus (beyond 1 day, diminishing returns)
        home_rest_bonus = min(home_rest_days - 1, 3) * 0.01 if home_rest_days > 1 else 0.0
        away_rest_bonus = min(away_rest_days - 1, 3) * 0.01 if away_rest_days > 1 else 0.0

        # Travel penalty: long travel (>2000km) adds ~0.01-0.02 fatigue
        home_travel_penalty = min(home_travel_km / 2000, 1.0) * 0.02
        away_travel_penalty = min(away_travel_km / 2000, 1.0) * 0.02

        # Schedule density: >4 games in 7 days = compressed
        home_density_penalty = max(0, (home_games_last_7 - 3)) * 0.008
        away_density_penalty = max(0, (away_games_last_7 - 3)) * 0.008

        home_total = home_fatigue + home_travel_penalty + home_density_penalty - home_rest_bonus
        away_total = away_fatigue + away_travel_penalty + away_density_penalty - away_rest_bonus

        # Net advantage: positive favors home
        rest_edge = away_total - home_total

        return {
            "rest_edge": round(rest_edge, 4),
            "home_fatigue_total": round(home_total, 4),
            "away_fatigue_total": round(away_total, 4),
            "home_is_back_to_back": home_is_back_to_back,
            "away_is_back_to_back": away_is_back_to_back,
            "home_rest_days": home_rest_days,
            "away_rest_days": away_rest_days,
            "schedule_concern": "high" if max(home_total, away_total) > 0.04 else "low",
        }

    # --- Score-state adjusted stats ---

    @staticmethod
    def score_adjusted_corsi(
        corsi_leading: dict[str, int],
        corsi_trailing: dict[str, int],
        corsi_tied: dict[str, int],
        minutes_leading: float,
        minutes_trailing: float,
        minutes_tied: float,
    ) -> dict[str, float]:
        """Score-state adjusted Corsi.

        Teams leading protect leads (fewer CF, more CA).
        Teams trailing press (more CF, fewer CA).
        Tied is the cleanest game state.

        Uses score-state-specific factors from Evolving Hockey's published
        methodology. Leading teams generate fewer attempts (each is worth more),
        trailing teams generate more (each is worth less).

        Factors from Evolving Hockey (home perspective, 5v5 shots):
          Down 1: CF*0.915, CA*1.102 | Up 1: CF*1.037, CA*0.966
          Down 2+: CF*0.876, CA*1.166 | Up 2+: CF*1.091, CA*0.924
          Tied: CF*0.972, CA*1.029

        Simplified to 3 states (leading/trailing/tied) using up-1/down-1 as
        typical cases, with slight scaling for multi-goal leads baked in.
        """
        # Adjustment factors (Evolving Hockey, averaged across score states)
        adj = {
            "leading": {"cf": 1.05, "ca": 0.95},   # up 1-2 avg
            "trailing": {"cf": 0.90, "ca": 1.13},   # down 1-2 avg
            "tied": {"cf": 0.972, "ca": 1.029},
        }

        states = [
            (corsi_leading, minutes_leading, adj["leading"]),
            (corsi_trailing, minutes_trailing, adj["trailing"]),
            (corsi_tied, minutes_tied, adj["tied"]),
        ]

        total_adj_cf = 0.0
        total_adj_ca = 0.0
        total_minutes = 0.0

        for state_data, minutes, factors in states:
            if minutes <= 0:
                continue
            cf = state_data.get("cf", 0)
            ca = state_data.get("ca", 0)
            total_adj_cf += cf * factors["cf"]
            total_adj_ca += ca * factors["ca"]
            total_minutes += minutes

        total = total_adj_cf + total_adj_ca
        adj_cf_pct = (total_adj_cf / total * 100) if total > 0 else 50.0

        adj_cf_per60 = (total_adj_cf / total_minutes * 60) if total_minutes > 0 else 0.0
        adj_ca_per60 = (total_adj_ca / total_minutes * 60) if total_minutes > 0 else 0.0

        return {
            "adj_corsi_for": round(total_adj_cf, 1),
            "adj_corsi_against": round(total_adj_ca, 1),
            "adj_corsi_pct": round(adj_cf_pct, 2),
            "adj_corsi_for_per60": round(adj_cf_per60, 2),
            "adj_corsi_against_per60": round(adj_ca_per60, 2),
            "minutes_leading": round(minutes_leading, 1),
            "minutes_trailing": round(minutes_trailing, 1),
            "minutes_tied": round(minutes_tied, 1),
        }

    # --- Salary cap analysis ---

    @staticmethod
    def cap_analysis(
        roster: list[dict[str, Any]],
        cap_ceiling: float = 95_500_000,
        cap_floor: float = 70_600_000,
    ) -> dict[str, Any]:
        """Salary cap analysis.

        Evaluates cap space, dead cap, position spending, and identifies
        cap concerns (LTIR, bonus overage, trade deadline flexibility).
        """
        if not roster:
            return {"error": "no roster data"}

        total_cap_hit = sum(p.get("cap_hit", 0) for p in roster)
        cap_space = cap_ceiling - total_cap_hit

        # Position breakdown
        position_spend: dict[str, float] = {}
        position_counts: dict[str, int] = {}
        for p in roster:
            pos = p.get("position", "unknown")
            cap_hit = p.get("cap_hit", 0)
            position_spend[pos] = position_spend.get(pos, 0) + cap_hit
            position_counts[pos] = position_counts.get(pos, 0) + 1

        # Top-heavy analysis: what % of cap is in top 5 earners
        sorted_by_cap = sorted(roster, key=lambda x: x.get("cap_hit", 0), reverse=True)
        top5_hit = sum(p.get("cap_hit", 0) for p in sorted_by_cap[:5])
        top5_pct = (top5_hit / total_cap_hit * 100) if total_cap_hit > 0 else 0

        # Expiring contracts (useful for trade deadline / offseason)
        expiring = [p for p in roster if p.get("contract_years_remaining", 999) <= 1]
        expiring_cap = sum(p.get("cap_hit", 0) for p in expiring)

        # Dead cap (buried contracts, LTIR)
        dead_cap = sum(p.get("cap_hit", 0) for p in roster if p.get("status") == "buried")
        ltir_hits = [p.get("cap_hit", 0) for p in roster if p.get("status") == "ltir"]
        ltir_total = sum(ltir_hits)

        # LTIR relief: team can exceed cap by the LTIR player's hit, minus
        # any cap space they had before activating LTIR. This is NOT simply
        # additive cap space -- teams on LTIR do not accrue daily cap space.
        # Under the 2025 CBA, relief per player is capped at the prior-season
        # league average salary (~$3.82M) unless SELTIR is designated.
        # Simplified model: LTIR pool = sum of LTIR hits - available cap space at activation.
        # Since we don't track activation timing, approximate as: LTIR hits - max(cap_space, 0).
        # If team was already over the cap, the pool is just how far over they are.
        ltir_relief = max(ltir_total - max(cap_space, 0), 0) if ltir_total > 0 else 0

        # Effective cap space: if on LTIR, team can spend up to their LTIR pool
        # above the cap ceiling. If not on LTIR, it's just cap_space.
        effective_space = ltir_relief if ltir_total > 0 else cap_space

        return {
            "total_cap_hit": round(total_cap_hit, 0),
            "cap_ceiling": cap_ceiling,
            "cap_space": round(cap_space, 0),
            "effective_cap_space": round(effective_space, 0),
            "roster_size": len(roster),
            "position_spending": {k: round(v, 0) for k, v in position_spend.items()},
            "position_counts": position_counts,
            "top5_earners": [
                {"name": p.get("name", ""), "cap_hit": p.get("cap_hit", 0), "position": p.get("position", "")}
                for p in sorted_by_cap[:5]
            ],
            "top5_cap_pct": round(top5_pct, 1),
            "expiring_contracts": len(expiring),
            "expiring_cap_total": round(expiring_cap, 0),
            "dead_cap": round(dead_cap, 0),
            "ltir_relief": round(ltir_relief, 0),
            "cap_health": "healthy" if effective_space > 5_000_000 else "tight" if effective_space > 0 else "over",
        }

    # --- Season review ---

    @staticmethod
    def season_review(
        team_stats: dict[str, Any],
        advanced_stats: dict[str, Any] | None = None,
        goaltender_stats: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Comprehensive season review for a team.

        Combines record, analytics (Corsi, xG), goaltending, and special
        teams into a structured assessment an agent can reason over.
        """
        wins = team_stats.get("wins", 0)
        losses = team_stats.get("losses", 0)
        otl = team_stats.get("overtime_losses", 0)
        gp = wins + losses + otl
        points = wins * 2 + otl
        pts_pct = points / (gp * 2) if gp > 0 else 0.0
        gf = team_stats.get("goals_for", 0)
        ga = team_stats.get("goals_against", 0)
        gf_per_game = gf / gp if gp > 0 else 0.0
        ga_per_game = ga / gp if gp > 0 else 0.0

        # Pythagorean expectation (hockey exponent ≈ 2.05, Dayaratna & Miller 2013)
        exp = 2.05
        pyth_pct = gf ** exp / (gf ** exp + ga ** exp) if (gf + ga) > 0 else 0.5
        pyth_wins = pyth_pct * gp
        luck_factor = wins - pyth_wins  # positive = lucky, negative = unlucky

        review: dict[str, Any] = {
            "record": f"{wins}-{losses}-{otl}",
            "points": points,
            "points_pct": round(pts_pct, 3),
            "games_played": gp,
            "goals_for_per_game": round(gf_per_game, 2),
            "goals_against_per_game": round(ga_per_game, 2),
            "goal_differential": gf - ga,
            "pythagorean_wins": round(pyth_wins, 1),
            "luck_factor": round(luck_factor, 1),
        }

        # Advanced stats summary
        if advanced_stats:
            cf_pct = advanced_stats.get("corsi_pct", 50.0)
            xgf_pct = advanced_stats.get("xgf_pct", 50.0)
            review["corsi_pct"] = cf_pct
            review["xgf_pct"] = xgf_pct
            review["analytics_grade"] = (
                "elite" if cf_pct > 53 and xgf_pct > 53 else
                "good" if cf_pct > 51 and xgf_pct > 51 else
                "average" if cf_pct > 48 and xgf_pct > 48 else
                "below_average" if cf_pct > 46 else "poor"
            )

        # Special teams summary
        pp_pct = team_stats.get("power_play_percentage", 0)
        pk_pct = team_stats.get("penalty_kill_percentage", 0)
        review["pp_pct"] = pp_pct
        review["pk_pct"] = pk_pct
        review["special_teams_grade"] = (
            "elite" if pp_pct > 25 and pk_pct > 83 else
            "good" if pp_pct > 22 and pk_pct > 80 else
            "average" if pp_pct > 18 and pk_pct > 77 else
            "poor"
        )

        # Goaltending summary
        if goaltender_stats:
            primary = goaltender_stats[0]
            review["primary_goaltender"] = {
                "name": primary.get("name", ""),
                "sv_pct": primary.get("save_pct", 0),
                "gaa": primary.get("gaa", 0),
                "gsaa": primary.get("gsaa", 0),
            }
            avg_sv = np.mean([g.get("save_pct", 0.900) for g in goaltender_stats])
            review["goaltending_grade"] = (
                "elite" if avg_sv > 0.920 else
                "good" if avg_sv > 0.912 else
                "average" if avg_sv > 0.905 else
                "below_average" if avg_sv > 0.895 else "poor"
            )

        return review

    # --- Playoff probability ---

    @staticmethod
    def playoff_probability(
        current_points: int,
        games_remaining: int,
        games_played: int,
        points_pace: float | None = None,
        division_cutline: int = 90,
        wildcard_cutline: int = 95,
        opponent_win_probs: list[float] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Estimate playoff probability using points pace and Monte Carlo.

        Standard rule of thumb: 96-98 points to make playoffs.
        Uses per-game simulation with opponent-specific win probabilities
        when available, falling back to uniform true-talent estimate.

        Parameters
        ----------
        opponent_win_probs : list[float] or None
            Per-game win probability for each remaining game, ordered by date.
            Length must equal games_remaining. If None, uses regressed true-talent
            for all games uniformly.
        """
        if games_played == 0:
            return {"error": "no games played"}

        # Current pace
        pts_per_game = current_points / games_played
        if points_pace is None:
            points_pace = pts_per_game
        projected_points = current_points + points_pace * games_remaining

        # Regress win rate toward .500 for true-talent estimate
        # Palmer-Tango: k ≈ 55 for NHL points percentage
        win_rate = pts_per_game / 2.0  # convert pts/game to win-equivalent rate
        true_talent = 0.5 + (win_rate - 0.5) * (games_played / (games_played + 55))

        rng = np.random.default_rng(seed)
        n_sims = 50000

        # Use opponent-specific probabilities if provided and valid
        schedule_aware = (
            opponent_win_probs is not None
            and len(opponent_win_probs) == games_remaining
            and games_remaining > 0
        )

        if schedule_aware and games_remaining > 0:
            # Per-game simulation with variable opponent strength
            sim_points = np.zeros(n_sims)
            for game_idx in range(games_remaining):
                p_win = opponent_win_probs[game_idx]
                p_otl = (1 - p_win) * 0.25
                p_reg_loss = (1 - p_win) * 0.75
                outcomes = rng.multinomial(1, [p_win, p_otl, p_reg_loss], size=n_sims)
                sim_points += outcomes[:, 0] * 2 + outcomes[:, 1] * 1
            sim_final_points = current_points + sim_points
        else:
            # Uniform true-talent for all remaining games
            p_win = true_talent
            p_otl = (1 - true_talent) * 0.25
            p_reg_loss = (1 - true_talent) * 0.75

            outcomes = rng.multinomial(
                games_remaining,
                [p_win, p_otl, p_reg_loss],
                size=n_sims,
            )
            sim_points_earned = outcomes[:, 0] * 2 + outcomes[:, 1] * 1
            sim_final_points = current_points + sim_points_earned

        # Expected points per game from true talent
        expected_pts_pg = true_talent * 2 + (1 - true_talent) * 0.25 * 1

        div_prob = float(np.mean(sim_final_points >= division_cutline))
        wc_prob = float(np.mean(sim_final_points >= wildcard_cutline))

        # Points needed calculation
        pts_for_division = max(0, division_cutline - current_points)
        pts_for_wildcard = max(0, wildcard_cutline - current_points)
        win_rate_needed_div = pts_for_division / (games_remaining * 2) if games_remaining > 0 else 0
        win_rate_needed_wc = pts_for_wildcard / (games_remaining * 2) if games_remaining > 0 else 0

        # Max possible points
        max_possible = current_points + games_remaining * 2

        return {
            "current_points": current_points,
            "games_remaining": games_remaining,
            "points_per_game": round(pts_per_game, 3),
            "projected_points": round(projected_points, 1),
            "true_talent_pts_pg": round(expected_pts_pg, 3),
            "division_cutline": division_cutline,
            "wildcard_cutline": wildcard_cutline,
            "division_probability": round(div_prob, 3),
            "wildcard_probability": round(wc_prob, 3),
            "points_needed_division": pts_for_division,
            "points_needed_wildcard": pts_for_wildcard,
            "win_rate_needed_division": round(win_rate_needed_div, 3),
            "win_rate_needed_wildcard": round(win_rate_needed_wc, 3),
            "simulations": n_sims,
            "schedule_aware": schedule_aware,
            "status": (
                "eliminated" if max_possible < wildcard_cutline else
                "on_pace" if wc_prob >= 0.95 else
                "likely" if wc_prob >= 0.75 else
                "contending" if wc_prob >= 0.50 else
                "bubble" if wc_prob >= 0.25 else
                "long_shot" if wc_prob >= 0.05 else
                "fading"
            ),
        }

    # --- NHL-enhanced prediction features ---

    def build_nhl_features(
        self,
        home_stats: TeamStatsRecord | dict[str, Any],
        away_stats: TeamStatsRecord | dict[str, Any],
        home_advanced: dict[str, Any] | None = None,
        away_advanced: dict[str, Any] | None = None,
        home_goalie: dict[str, Any] | None = None,
        away_goalie: dict[str, Any] | None = None,
        rest_info: dict[str, Any] | None = None,
        odds_data: list[OddsSnapshot] | None = None,
    ) -> dict[str, float]:
        """Build comprehensive NHL feature vector for prediction.

        Goes beyond win/loss to include Corsi, xG, goaltending,
        special teams, and rest -- the features that actually
        predict NHL outcomes.
        """
        features: dict[str, float] = {}
        hs = home_stats.get("stats", home_stats)
        as_ = away_stats.get("stats", away_stats)

        # Basic record
        h_wins = hs.get("wins", 40)
        h_losses = hs.get("losses", 30)
        h_otl = hs.get("overtime_losses", 10)
        a_wins = as_.get("wins", 40)
        a_losses = as_.get("losses", 30)
        a_otl = as_.get("overtime_losses", 10)

        h_gp = max(h_wins + h_losses + h_otl, 1)
        a_gp = max(a_wins + a_losses + a_otl, 1)

        features["home_pts_pct"] = (h_wins * 2 + h_otl) / (h_gp * 2)
        features["away_pts_pct"] = (a_wins * 2 + a_otl) / (a_gp * 2)

        # Goal differential per game
        h_gf = hs.get("goals_for", 0)
        h_ga = hs.get("goals_against", 0)
        a_gf = as_.get("goals_for", 0)
        a_ga = as_.get("goals_against", 0)
        features["home_goal_diff_pg"] = (h_gf - h_ga) / h_gp
        features["away_goal_diff_pg"] = (a_gf - a_ga) / a_gp

        # Recent form (L10)
        features["home_recent_form"] = hs.get("recent_form", features["home_pts_pct"])
        features["away_recent_form"] = as_.get("recent_form", features["away_pts_pct"])

        # Home/road splits (if available from API)
        h_home_wins = hs.get("home_wins", 0)
        h_home_losses = hs.get("home_losses", 0)
        h_home_otl = hs.get("home_ot_losses", 0)
        h_home_gp = h_home_wins + h_home_losses + h_home_otl
        if h_home_gp > 10:
            features["home_pts_pct"] = (h_home_wins * 2 + h_home_otl) / (h_home_gp * 2)

        a_road_wins = as_.get("road_wins", 0)
        a_road_losses = as_.get("road_losses", 0)
        a_road_otl = as_.get("road_ot_losses", 0)
        a_road_gp = a_road_wins + a_road_losses + a_road_otl
        if a_road_gp > 10:
            features["away_pts_pct"] = (a_road_wins * 2 + a_road_otl) / (a_road_gp * 2)

        # Games played (for regression to mean)
        features["home_games_played"] = h_gp
        features["away_games_played"] = a_gp

        # Per-game goal rates (for Pythagorean model)
        features["home_gf_per_game"] = h_gf / h_gp if h_gp > 0 else 3.0
        features["home_ga_per_game"] = h_ga / h_gp if h_gp > 0 else 3.0
        features["away_gf_per_game"] = a_gf / a_gp if a_gp > 0 else 3.0
        features["away_ga_per_game"] = a_ga / a_gp if a_gp > 0 else 3.0

        # Advanced stats (Corsi, xG)
        if home_advanced and away_advanced:
            features["home_corsi_pct"] = home_advanced.get("corsi_pct", 50.0) / 100
            features["away_corsi_pct"] = away_advanced.get("corsi_pct", 50.0) / 100
            features["home_xgf_pct"] = home_advanced.get("xgf_pct", 50.0) / 100
            features["away_xgf_pct"] = away_advanced.get("xgf_pct", 50.0) / 100
        else:
            features["home_corsi_pct"] = 0.50
            features["away_corsi_pct"] = 0.50
            features["home_xgf_pct"] = 0.50
            features["away_xgf_pct"] = 0.50

        # Goaltender edge
        if home_goalie and away_goalie:
            matchup = self.goaltender_matchup_edge(
                home_goalie.get("save_pct", 0.907),
                away_goalie.get("save_pct", 0.907),
                home_goalie.get("gsaa", 0),
                away_goalie.get("gsaa", 0),
            )
            features["goaltender_edge"] = matchup["goaltender_edge"]
        else:
            features["goaltender_edge"] = 0.0

        # Special teams differential
        h_pp = hs.get("power_play_percentage", 22.0)
        h_pk = hs.get("penalty_kill_percentage", 80.0)
        a_pp = as_.get("power_play_percentage", 22.0)
        a_pk = as_.get("penalty_kill_percentage", 80.0)
        # Home PP vs Away PK, and Away PP vs Home PK
        features["home_st_edge"] = ((h_pp - a_pk) + (h_pk - a_pp)) / 100

        # Rest/schedule edge
        if rest_info:
            features["rest_edge"] = rest_info.get("rest_edge", 0.0)
        else:
            features["rest_edge"] = 0.0

        # Market-implied probability from odds data (de-vigged)
        features["market_implied_prob"] = 0.0
        if odds_data:
            # Find the best (highest) home and away odds across books
            best_home_odds = 0.0
            best_away_odds = 0.0
            for o in odds_data:
                ho = o.get("home_odds", 0)
                ao = o.get("away_odds", 0)
                if ho and ho > best_home_odds:
                    best_home_odds = ho
                if ao and ao > best_away_odds:
                    best_away_odds = ao
            # Compute de-vigged implied probability for the home team
            if best_home_odds > 1.0 and best_away_odds > 1.0:
                raw_home = 1.0 / best_home_odds
                raw_away = 1.0 / best_away_odds
                total_implied = raw_home + raw_away
                if total_implied > 0:
                    features["market_implied_prob"] = raw_home / total_implied

        return features

    @staticmethod
    def regress_to_mean(observed: float, league_avg: float, games_played: int, k: int = 55) -> float:
        """Regress an observed stat toward the league average.

        Uses the Palmer-Tango empirical Bayes method:
          true_talent = league_avg + (observed - league_avg) * n / (n + k)

        k ≈ 55 for NHL points percentage (midpoint of published 36-73 range).
        At 20 GP: weight=0.27. At 41 GP: 0.43. At 82 GP: 0.60.
        """
        weight = games_played / (games_played + k)
        return league_avg + (observed - league_avg) * weight

    def predict_nhl_game(self, features: dict[str, float], elo_prob: float | None = None) -> list[float]:
        """Generate NHL-specific model predictions from features.

        Models with calibrated sensitivity:
        1. Points percentage log5 + home ice (baseline)
        2. Pythagorean expectation via log5 (goal-based)
        3. Recent form weighted (momentum, low weight)
        4. Goal differential strength model
        5. Goaltender matchup model
        6. Composite with special teams + rest adjustments
        7. (optional) Market-implied probability
        8. (optional) Elo rating prediction

        All inputs are regressed to the mean based on games played.
        NHL game outcomes have ~58-62% max predictability.
        """
        hp = features.get("home_pts_pct", 0.5)
        ap = features.get("away_pts_pct", 0.5)
        hgd = features.get("home_goal_diff_pg", 0.0)
        agd = features.get("away_goal_diff_pg", 0.0)
        hrf = features.get("home_recent_form", 0.5)
        arf = features.get("away_recent_form", 0.5)
        ge = features.get("goaltender_edge", 0.0)
        ste = features.get("home_st_edge", 0.0)
        re = features.get("rest_edge", 0.0)
        h_gp = features.get("home_games_played", 82)
        a_gp = features.get("away_games_played", 82)
        h_gf_pg = features.get("home_gf_per_game", 3.0)
        h_ga_pg = features.get("home_ga_per_game", 3.0)
        a_gf_pg = features.get("away_gf_per_game", 3.0)
        a_ga_pg = features.get("away_ga_per_game", 3.0)

        # Regress stats to the mean based on sample size
        hp = self.regress_to_mean(hp, 0.5, int(h_gp))
        ap = self.regress_to_mean(ap, 0.5, int(a_gp))

        # Home ice: ~54% in recent NHL seasons, = 0.04 additive boost
        home_boost = 0.04

        # Model 1: Points percentage log5 (standard head-to-head formula)
        # log5: P(A beats B) = (pA - pA*pB) / (pA + pB - 2*pA*pB)
        # Clamp inputs away from 0 and 1 to prevent singularity
        hp_c = max(0.01, min(0.99, hp))
        ap_c = max(0.01, min(0.99, ap))
        denom = hp_c + ap_c - 2 * hp_c * ap_c
        m1 = (hp_c - hp_c * ap_c) / denom if abs(denom) > 1e-8 else 0.5
        m1 += home_boost

        # Model 2: Pythagorean expectation via log5
        # Use actual GF/GA per game to compute Pythagorean win%, then log5
        exp = 2.05
        h_gf = max(h_gf_pg, 0.01)
        h_ga = max(h_ga_pg, 0.01)
        a_gf = max(a_gf_pg, 0.01)
        a_ga = max(a_ga_pg, 0.01)
        h_pyth = h_gf ** exp / (h_gf ** exp + h_ga ** exp)
        a_pyth = a_gf ** exp / (a_gf ** exp + a_ga ** exp)
        # Regress Pythagorean estimates too
        h_pyth = self.regress_to_mean(h_pyth, 0.5, int(h_gp))
        a_pyth = self.regress_to_mean(a_pyth, 0.5, int(a_gp))
        h_pyth_c = max(0.01, min(0.99, h_pyth))
        a_pyth_c = max(0.01, min(0.99, a_pyth))
        denom2 = h_pyth_c + a_pyth_c - 2 * h_pyth_c * a_pyth_c
        m2 = (h_pyth_c - h_pyth_c * a_pyth_c) / denom2 if abs(denom2) > 1e-8 else 0.5
        m2 += home_boost

        # Model 3: Recent form (low weight — noisy, low predictive power)
        form_diff = hrf - arf
        m3 = 0.5 + form_diff * 0.25 + home_boost

        # Model 4: Goal differential strength
        # Scale: +0.5 GD/game diff -> ~0.08 probability edge
        gd_diff = hgd - agd
        m4 = 0.5 + gd_diff * 0.16 + home_boost

        # Model 5: Goaltender matchup
        # Edge from goaltender_matchup_edge() already calibrated to realistic range
        m5 = 0.5 + ge + home_boost

        # Model 6: Composite with special teams + rest
        base = (m1 + m2 + m3 + m4 + m5) / 5
        # Special teams: typical range is -0.10 to +0.10, worth ~3% probability per 0.10
        st_adj = ste * 0.3
        # Rest: B2B penalty = ~0.045, directly additive
        m6 = base + st_adj + re

        models = [float(np.clip(x, 0.01, 0.99)) for x in [m1, m2, m3, m4, m5, m6]]

        # Model 7 (optional): Market-implied probability
        # The market knows things models don't -- injuries, lineup changes, sharp money.
        # Only include if the probability is between 0.15 and 0.85 (filter out broken odds).
        # Use inverse-variance weighting: market gets 0.30, model average gets 0.70.
        market_prob = features.get("market_implied_prob")
        if market_prob is not None and market_prob > 0 and 0.15 <= market_prob <= 0.85:
            model_avg = sum(models) / len(models)
            blended = 0.70 * model_avg + 0.30 * market_prob
            models.append(float(np.clip(blended, 0.01, 0.99)))

        # Model 8 (optional): Elo rating prediction
        # FiveThirtyEight-style Elo with K=6, home ice +50 Elo points.
        # Elo captures long-term team strength independent of the other models'
        # feature engineering. Include if a valid probability was passed.
        if elo_prob is not None and 0.01 <= elo_prob <= 0.99:
            models.append(float(np.clip(elo_prob, 0.01, 0.99)))

        return models

    # Feature schema for documentation
    NHL_FEATURE_SCHEMA = {
        "home_pts_pct": "Home team points percentage (0-1)",
        "away_pts_pct": "Away team points percentage (0-1)",
        "home_goal_diff_pg": "Home team goal differential per game",
        "away_goal_diff_pg": "Away team goal differential per game",
        "home_recent_form": "Home team recent form (0-1)",
        "away_recent_form": "Away team recent form (0-1)",
        "home_games_played": "Home team games played (for regression to mean)",
        "away_games_played": "Away team games played (for regression to mean)",
        "home_gf_per_game": "Home team goals for per game",
        "home_ga_per_game": "Home team goals against per game",
        "away_gf_per_game": "Away team goals for per game",
        "away_ga_per_game": "Away team goals against per game",
        "home_corsi_pct": "Home team 5v5 Corsi% (0-1)",
        "away_corsi_pct": "Away team 5v5 Corsi% (0-1)",
        "home_xgf_pct": "Home team expected goals for% (0-1)",
        "away_xgf_pct": "Away team expected goals for% (0-1)",
        "goaltender_edge": "Goaltender matchup edge (-0.5 to 0.5)",
        "home_st_edge": "Special teams differential",
        "rest_edge": "Rest/schedule advantage (-0.1 to 0.1)",
        "market_implied_prob": "De-vigged market-implied home win probability (0-1, 0 if unavailable)",
    }
