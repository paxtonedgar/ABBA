"""Prediction confidence and accuracy metadata for LLM tool responses.

When an LLM calls ABBA tools and gets back predictions, it needs to know
HOW MUCH to trust the numbers so it doesn't launder uncertainty into false
precision. This module attaches calibration metadata to every prediction
and workflow response.

All computations use stdlib + numpy only (no additional dependencies).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# LLM Interpretation Guide
# ---------------------------------------------------------------------------

LLM_INTERPRETATION_GUIDE: dict[str, str] = {
    "unreliable_grades": (
        "When reliability is D or F, explicitly tell the user this prediction "
        "is unreliable and should not be used for decision-making."
    ),
    "confidence_interval": (
        "Always mention the confidence interval, not just the point estimate. "
        "For example, say '55% (80% CI: 42%-68%)' instead of just '55%'."
    ),
    "stale_data": (
        "If data is stale (more than 24 hours old), say so. Stale data means "
        "injuries, lineup changes, or other developments may not be reflected."
    ),
    "seed_data": (
        "If using seed data, do not present predictions as real. Seed data is "
        "synthetic/illustrative and does not reflect actual team performance."
    ),
    "coin_flip_rule": (
        "A 55% prediction with 80% CI [0.42, 0.68] means this is basically a "
        "coin flip -- say so. Any prediction where the CI spans both sides of "
        "0.50 should be described as uncertain, not as a lean."
    ),
    "sample_size": (
        "Small sample sizes (under 30 games) mean the prediction is heavily "
        "regressed toward the mean. Mention this to the user."
    ),
    "missing_features": (
        "If goaltender data, advanced stats, or other key features are missing, "
        "the model is operating with less information than intended. Say so."
    ),
}


# ---------------------------------------------------------------------------
# Historical calibration baselines
# ---------------------------------------------------------------------------

# These represent the model's known calibration performance from backtesting.
# They are used to compute confidence intervals and reliability grades.
# In production these would be loaded from a calibration artifact; here we
# hardcode conservative estimates from internal backtests.
_DEFAULT_ACCURACY_HISTORY: dict[str, Any] = {
    "log_loss": 0.68,
    "brier_score": 0.24,
    "accuracy": 0.57,
    "sample_size": 820,
    "date_range": "2023-10-01 to 2024-04-30",
}

# Calibration error: how far off the model's probability estimates tend to be.
# Used to widen confidence intervals. Derived from reliability diagrams on
# the backtest set. 0.08 means "on average, when we say 60% we're really
# somewhere in [52%, 68%]".
_BASE_CALIBRATION_ERROR: float = 0.08


# ---------------------------------------------------------------------------
# PredictionConfidence dataclass
# ---------------------------------------------------------------------------

@dataclass
class PredictionConfidence:
    """Metadata attached to every prediction response.

    Provides the calling LLM (and downstream users) with everything needed
    to judge how much to trust a point estimate.
    """

    accuracy_history: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_ACCURACY_HISTORY))
    data_freshness: str | float = "unknown"
    sample_size: dict[str, int] = field(default_factory=dict)
    confidence_interval: dict[str, float] = field(default_factory=dict)
    reliability_grade: str = "F"
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy_history": self.accuracy_history,
            "data_freshness": self.data_freshness,
            "sample_size": self.sample_size,
            "confidence_interval": self.confidence_interval,
            "reliability_grade": self.reliability_grade,
            "caveats": list(self.caveats),
            "interpretation_guide": LLM_INTERPRETATION_GUIDE,
        }


# ---------------------------------------------------------------------------
# Reliability grade assignment
# ---------------------------------------------------------------------------

def _compute_reliability_grade(
    data_source: str,
    home_gp: int,
    away_gp: int,
    has_goalie_data: bool,
    staleness_seconds: float,
) -> str:
    """Assign a letter grade based on data quality signals.

    Grades:
        A: live data, 50+ GP both teams, goalie data, <1hr stale
        B: live data, 30+ GP both teams, <24hr stale
        C: live data, <30 GP OR >24hr stale OR missing goalie data
        D: seed data OR <10 GP either team
        F: no data at all
    """
    if data_source == "none":
        return "F"

    is_seed = data_source == "seed"
    min_gp = min(home_gp, away_gp)

    if is_seed or min_gp < 10:
        return "D"

    # From here, data_source is "live" (or any non-seed, non-none value)
    one_hour = 3600.0
    twenty_four_hours = 86400.0

    if min_gp >= 50 and has_goalie_data and staleness_seconds < one_hour:
        return "A"

    if min_gp >= 30 and staleness_seconds < twenty_four_hours:
        return "B"

    # Anything else with live data but degraded quality
    return "C"


# ---------------------------------------------------------------------------
# Caveat generation
# ---------------------------------------------------------------------------

def _generate_caveats(
    data_source: str,
    home_gp: int,
    away_gp: int,
    has_goalie_data: bool,
    staleness_seconds: float,
    extra_caveats: list[str] | None = None,
) -> list[str]:
    """Build a list of plain-English caveats the LLM should relay to users."""
    caveats: list[str] = []

    if data_source == "seed":
        caveats.append("Using seed data \u2014 not real")
    elif data_source == "none":
        caveats.append("No data available \u2014 prediction is meaningless")

    min_gp = min(home_gp, away_gp)
    if min_gp < 10:
        team_label = "Home" if home_gp < away_gp else "Away"
        gp = min(home_gp, away_gp)
        caveats.append(
            f"{team_label} team has only {gp} games played \u2014 high regression"
        )
    elif min_gp < 30:
        caveats.append(
            f"Small sample: min games played is {min_gp} \u2014 moderate regression applied"
        )

    if not has_goalie_data:
        caveats.append("No goaltender data available")

    if staleness_seconds >= 172800:  # 48 hours
        caveats.append("Data is 48+ hours stale")
    elif staleness_seconds >= 86400:  # 24 hours
        caveats.append("Data is 24+ hours stale")

    if extra_caveats:
        caveats.extend(extra_caveats)

    return caveats


# ---------------------------------------------------------------------------
# Confidence interval computation
# ---------------------------------------------------------------------------

def _compute_confidence_interval(
    prediction_value: float,
    calibration_error: float,
    min_gp: int,
    has_goalie_data: bool,
    data_source: str,
) -> dict[str, float]:
    """Compute an 80% confidence interval around the point estimate.

    The width is driven by:
    - Base calibration error from backtesting
    - Sample size (fewer games = wider)
    - Missing features (no goalie data = wider)
    - Data source quality (seed = much wider)

    Returns dict with keys: point, lower, upper, width.
    """
    half_width = calibration_error

    # Sample size penalty: below 50 GP, widen proportionally.
    # At 10 GP the interval is ~2x wider; at 50+ it's baseline.
    if min_gp > 0:
        sample_factor = math.sqrt(50.0 / max(min_gp, 1))
        sample_factor = max(sample_factor, 1.0)  # never narrow below baseline
    else:
        sample_factor = 3.0  # no games at all

    half_width *= sample_factor

    # Missing goalie data adds uncertainty
    if not has_goalie_data:
        half_width *= 1.25

    # Seed data: much wider
    if data_source == "seed":
        half_width *= 2.0
    elif data_source == "none":
        half_width = 0.50  # effectively 0-100%

    # 80% CI uses z=1.28 (vs 1.96 for 95%)
    ci_half = half_width * 1.28

    lower = max(prediction_value - ci_half, 0.0)
    upper = min(prediction_value + ci_half, 1.0)

    return {
        "point": round(prediction_value, 4),
        "lower": round(lower, 4),
        "upper": round(upper, 4),
        "width": round(upper - lower, 4),
    }


# ---------------------------------------------------------------------------
# build_prediction_meta -- the main entry point for predictions
# ---------------------------------------------------------------------------

def build_prediction_meta(
    features: dict[str, Any],
    prediction_value: float,
    data_source: str = "live",
    last_refresh_ts: float | None = None,
    has_goalie_data: bool = True,
    accuracy_history: dict[str, Any] | None = None,
    calibration_error: float | None = None,
    extra_caveats: list[str] | None = None,
) -> dict[str, Any]:
    """Build confidence metadata for a single prediction response.

    Parameters
    ----------
    features : dict
        The feature dict that was fed into the model. Used to extract
        games-played and other quality signals.
    prediction_value : float
        The point estimate (home win probability, 0-1).
    data_source : str
        One of "live", "seed", or "none".
    last_refresh_ts : float or None
        Unix timestamp of the most recent data refresh. If None and
        data_source is "seed", freshness is reported as "seed".
    has_goalie_data : bool
        Whether goaltender stats were available for the prediction.
    accuracy_history : dict or None
        Override the default historical accuracy metrics.
    calibration_error : float or None
        Override the default calibration error.
    extra_caveats : list[str] or None
        Additional caveats to include.

    Returns
    -------
    dict
        A metadata dict suitable for attaching to any prediction response.
    """
    home_gp = int(features.get("home_games_played", 0))
    away_gp = int(features.get("away_games_played", 0))
    cal_err = calibration_error if calibration_error is not None else _BASE_CALIBRATION_ERROR
    acc_hist = accuracy_history if accuracy_history is not None else dict(_DEFAULT_ACCURACY_HISTORY)

    # Data freshness
    if data_source == "seed":
        data_freshness: str | float = "seed"
        staleness: float = float("inf")
    elif last_refresh_ts is not None:
        staleness = time.time() - last_refresh_ts
        data_freshness = round(staleness, 1)
    else:
        staleness = float("inf")
        data_freshness = "unknown"

    # Reliability grade
    grade = _compute_reliability_grade(
        data_source=data_source,
        home_gp=home_gp,
        away_gp=away_gp,
        has_goalie_data=has_goalie_data,
        staleness_seconds=staleness,
    )

    # Caveats
    caveats = _generate_caveats(
        data_source=data_source,
        home_gp=home_gp,
        away_gp=away_gp,
        has_goalie_data=has_goalie_data,
        staleness_seconds=staleness,
        extra_caveats=extra_caveats,
    )

    # Confidence interval
    min_gp = min(home_gp, away_gp)
    ci = _compute_confidence_interval(
        prediction_value=prediction_value,
        calibration_error=cal_err,
        min_gp=min_gp,
        has_goalie_data=has_goalie_data,
        data_source=data_source,
    )

    confidence = PredictionConfidence(
        accuracy_history=acc_hist,
        data_freshness=data_freshness,
        sample_size={"home_games_played": home_gp, "away_games_played": away_gp},
        confidence_interval=ci,
        reliability_grade=grade,
        caveats=caveats,
    )

    return confidence.to_dict()


# ---------------------------------------------------------------------------
# build_workflow_meta -- metadata for workflow (multi-step) responses
# ---------------------------------------------------------------------------

def build_workflow_meta(
    workflow_name: str,
    data_sources_used: list[str],
    steps_completed: int = 0,
    steps_total: int = 0,
    overall_data_source: str | None = None,
    last_refresh_ts: float | None = None,
    min_games_played: int = 0,
    has_goalie_data: bool = True,
    extra_caveats: list[str] | None = None,
) -> dict[str, Any]:
    """Build confidence metadata for a workflow (multi-step) response.

    Workflows aggregate several tool calls, so the metadata reflects the
    weakest link in the chain.

    Parameters
    ----------
    workflow_name : str
        Name of the workflow (e.g. "game_prediction", "tonights_slate").
    data_sources_used : list[str]
        Which data sources were touched (e.g. ["live_api", "seed", "cache"]).
    steps_completed : int
        How many workflow steps ran successfully.
    steps_total : int
        Total number of steps in the workflow.
    overall_data_source : str or None
        The weakest data source in the chain. If any step used "seed",
        the overall source is "seed". If None, inferred from data_sources_used.
    last_refresh_ts : float or None
        Oldest data refresh timestamp among all steps.
    min_games_played : int
        Minimum GP across all teams involved in the workflow.
    has_goalie_data : bool
        Whether goaltender data was available for all relevant steps.
    extra_caveats : list[str] or None
        Additional caveats.

    Returns
    -------
    dict
        Workflow metadata dict.
    """
    # Infer overall data source from the weakest link
    if overall_data_source is None:
        if "none" in data_sources_used:
            overall_data_source = "none"
        elif "seed" in data_sources_used:
            overall_data_source = "seed"
        elif data_sources_used:
            overall_data_source = "live"
        else:
            overall_data_source = "none"

    # Staleness
    if overall_data_source == "seed":
        staleness = float("inf")
        data_freshness: str | float = "seed"
    elif last_refresh_ts is not None:
        staleness = time.time() - last_refresh_ts
        data_freshness = round(staleness, 1)
    else:
        staleness = float("inf")
        data_freshness = "unknown"

    grade = _compute_reliability_grade(
        data_source=overall_data_source,
        home_gp=min_games_played,
        away_gp=min_games_played,
        has_goalie_data=has_goalie_data,
        staleness_seconds=staleness,
    )

    caveats = _generate_caveats(
        data_source=overall_data_source,
        home_gp=min_games_played,
        away_gp=min_games_played,
        has_goalie_data=has_goalie_data,
        staleness_seconds=staleness,
        extra_caveats=extra_caveats,
    )

    if steps_completed < steps_total:
        caveats.append(
            f"Workflow incomplete: {steps_completed}/{steps_total} steps finished"
        )

    return {
        "workflow": workflow_name,
        "data_sources_used": data_sources_used,
        "steps_completed": steps_completed,
        "steps_total": steps_total,
        "data_freshness": data_freshness,
        "reliability_grade": grade,
        "caveats": caveats,
        "interpretation_guide": LLM_INTERPRETATION_GUIDE,
    }
