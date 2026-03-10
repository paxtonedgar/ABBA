"""Kelly Criterion position sizing.

Implements the Kelly Criterion for optimal bet sizing:
  f* = (bp - q) / b
where:
  b = decimal odds - 1 (net odds received on a win)
  p = probability of winning
  q = 1 - p (probability of losing)

We use fractional Kelly (default half-Kelly) because:
1. Full Kelly assumes perfect probability estimates (we don't have those)
2. Half-Kelly has ~75% of the growth rate with much lower variance
3. It's standard practice in quantitative betting
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KellyResult:
    """Position sizing output."""
    fraction: float          # Kelly fraction (0-1)
    recommended_stake: float # Dollar amount to bet
    expected_value: float    # Expected profit per dollar wagered
    edge: float             # Probability edge over implied odds
    bankroll_risk: float    # % of bankroll at risk

    def to_dict(self) -> dict:
        return {
            "fraction": round(self.fraction, 6),
            "recommended_stake": round(self.recommended_stake, 2),
            "expected_value": round(self.expected_value, 4),
            "edge": round(self.edge, 4),
            "bankroll_risk": round(self.bankroll_risk, 4),
        }


class KellyEngine:
    """Kelly Criterion position sizing with safety constraints."""

    def __init__(
        self,
        kelly_fraction: float = 0.5,
        max_bet_pct: float = 0.05,
        min_edge: float = 0.02,
        min_ev: float = 0.03,
    ):
        """
        Args:
            kelly_fraction: Fraction of full Kelly to use (0.5 = half-Kelly)
            max_bet_pct: Maximum bet as % of bankroll (hard cap)
            min_edge: Minimum probability edge to place a bet
            min_ev: Minimum expected value per dollar to place a bet
        """
        self.kelly_fraction = kelly_fraction
        self.max_bet_pct = max_bet_pct
        self.min_edge = min_edge
        self.min_ev = min_ev

    def calculate(
        self,
        win_probability: float,
        decimal_odds: float,
        bankroll: float,
    ) -> KellyResult:
        """Calculate optimal position size.

        Args:
            win_probability: Estimated probability of winning (0-1)
            decimal_odds: Decimal odds (e.g., 2.10 means +110)
            bankroll: Current bankroll in dollars
        """
        # Validate inputs
        p = np.clip(win_probability, 0.001, 0.999)
        q = 1.0 - p
        b = decimal_odds - 1.0  # net odds

        if b <= 0:
            return KellyResult(0, 0, 0, 0, 0)

        # Implied probability from odds
        implied_prob = 1.0 / decimal_odds

        # Edge: how much our estimated prob exceeds the market's
        edge = p - implied_prob

        # Expected value per dollar wagered: EV = p * b - q
        ev = p * b - q

        # Full Kelly fraction
        full_kelly = (b * p - q) / b

        # Apply fractional Kelly
        fraction = full_kelly * self.kelly_fraction

        # Safety constraints
        fraction = max(0.0, fraction)  # Never negative (never bet against yourself)
        fraction = min(fraction, self.max_bet_pct)  # Hard cap

        # Check minimums
        if edge < self.min_edge or ev < self.min_ev:
            fraction = 0.0

        stake = fraction * bankroll

        return KellyResult(
            fraction=fraction,
            recommended_stake=stake,
            expected_value=ev,
            edge=edge,
            bankroll_risk=fraction,
        )

    def american_to_decimal(self, american_odds: int) -> float:
        """Convert American odds to decimal.

        +150 → 2.50 (bet $100 to win $150)
        -150 → 1.667 (bet $150 to win $100)
        """
        if american_odds > 0:
            return 1.0 + american_odds / 100.0
        else:
            return 1.0 + 100.0 / abs(american_odds)

    def implied_probability(self, decimal_odds: float) -> float:
        """Get implied probability from decimal odds."""
        if decimal_odds <= 0:
            return 0.0
        return 1.0 / decimal_odds
