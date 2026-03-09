"""NHL Elo rating system.

Implements a FiveThirtyEight-style Elo model tuned for hockey:
- K-factor of 6 (high-variance sport)
- Home ice advantage of +50 Elo points
- Between-season reversion of 1/3 toward 1500
- Optional margin-of-victory multiplier for additional signal
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


class EloRatings:
    """Elo rating tracker for NHL teams.

    Parameters
    ----------
    k : float
        K-factor controlling how much a single game moves ratings.
        FiveThirtyEight uses 6 for the NHL.
    home_advantage : float
        Elo-point bonus added to the home team's rating when computing
        win probability. 50 is a reasonable NHL default.
    initial_rating : float
        Starting Elo for every team that has not yet been seen.
    """

    def __init__(
        self,
        k: float = 6,
        home_advantage: float = 50,
        initial_rating: float = 1500,
    ) -> None:
        self.k = k
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self._ratings: dict[str, float] = defaultdict(lambda: self.initial_rating)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_rating(self, team: str) -> float:
        """Return the current Elo rating for *team*.

        If the team has never been seen, the initial rating is returned
        (and stored for future lookups).
        """
        return self._ratings[team]

    def get_all_ratings(self) -> dict[str, float]:
        """Return a plain ``dict`` snapshot of every tracked team's rating."""
        return dict(self._ratings)

    def predict(self, home_team: str, away_team: str) -> dict[str, Any]:
        """Predict the outcome of a game between *home_team* and *away_team*.

        Returns
        -------
        dict
            ``home_win_prob``  – probability that home wins (0-1)
            ``away_win_prob``  – probability that away wins (0-1)
            ``home_rating``    – current Elo of home team
            ``away_rating``    – current Elo of away team
        """
        home_rating = self._ratings[home_team]
        away_rating = self._ratings[away_team]

        home_win_prob = self._win_probability(
            home_rating + self.home_advantage, away_rating
        )

        return {
            "home_win_prob": home_win_prob,
            "away_win_prob": 1.0 - home_win_prob,
            "home_rating": home_rating,
            "away_rating": away_rating,
        }

    def update(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
    ) -> dict[str, Any]:
        """Record a completed game and update both teams' ratings.

        The margin-of-victory multiplier is applied automatically when the
        scores differ.

        Parameters
        ----------
        home_team, away_team : str
            Team identifiers.
        home_score, away_score : int
            Final goals for each side.

        Returns
        -------
        dict
            Pre-game and post-game ratings for both teams, the pre-game
            prediction, and the margin-of-victory multiplier used.
        """
        prediction = self.predict(home_team, away_team)

        # Actual result: 1 = home win, 0 = home loss, 0.5 = draw
        if home_score > away_score:
            actual_home = 1.0
        elif home_score < away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        expected_home = prediction["home_win_prob"]

        goal_diff = abs(home_score - away_score)
        elo_diff = abs(
            (prediction["home_rating"] + self.home_advantage) - prediction["away_rating"]
        )
        mov_multiplier = self._margin_of_victory_multiplier(goal_diff, elo_diff)

        shift = self.k * mov_multiplier * (actual_home - expected_home)

        pre_home = prediction["home_rating"]
        pre_away = prediction["away_rating"]

        self._ratings[home_team] = pre_home + shift
        self._ratings[away_team] = pre_away - shift

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_pre": pre_home,
            "away_pre": pre_away,
            "home_post": self._ratings[home_team],
            "away_post": self._ratings[away_team],
            "home_win_prob": expected_home,
            "mov_multiplier": mov_multiplier,
            "shift": shift,
        }

    def season_reset(self) -> None:
        """Revert every team's rating 1/3 of the way toward *initial_rating*.

        Call this between seasons to account for roster turnover and
        regression to the mean.
        """
        for team in self._ratings:
            self._ratings[team] = (
                self._ratings[team]
                + (self.initial_rating - self._ratings[team]) / 3.0
            )

    def initialize_from_games(self, games: list[dict[str, Any]]) -> dict[str, float]:
        """Replay a sequence of completed games to build current ratings.

        Each element of *games* must be a dict with at least:

        - ``home_team`` (str)
        - ``away_team`` (str)
        - ``home_score`` (int)
        - ``away_score`` (int)

        An optional ``season`` key triggers ``season_reset()`` whenever
        the season value changes from one game to the next.

        Returns
        -------
        dict[str, float]
            The final ratings after all games have been processed.
        """
        current_season = None

        for game in games:
            season = game.get("season")
            if season is not None and current_season is not None and season != current_season:
                self.season_reset()
            current_season = season

            self.update(
                home_team=game["home_team"],
                away_team=game["away_team"],
                home_score=game["home_score"],
                away_score=game["away_score"],
            )

        return self.get_all_ratings()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _win_probability(rating_a: float, rating_b: float) -> float:
        """P(A wins) = 1 / (1 + 10^((Rb - Ra) / 400))."""
        return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))

    @staticmethod
    def _margin_of_victory_multiplier(goal_diff: int, elo_diff: float) -> float:
        """Amplify updates for blowouts, dampened by rating gap.

        Formula: ln(|goal_diff| + 1) * (2.2 / (elo_diff * 0.001 + 2.2))

        When ``goal_diff`` is 0 (a draw), the multiplier is 0 — but
        a draw still produces a rating change via the base K * (actual - expected).
        We clamp the minimum to 1.0 so that one-goal games are not *shrunk*.
        """
        if goal_diff == 0:
            return 1.0
        raw = math.log(goal_diff + 1) * (2.2 / (elo_diff * 0.001 + 2.2))
        return max(raw, 1.0)
