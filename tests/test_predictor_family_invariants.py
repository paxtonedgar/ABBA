"""Invariant A — Predictor Family Consistency.

If a workflow is labeled NHL and is used for staking/value decisions,
it must use the NHL predictor family or explicitly declare why it does not.

These tests PROVE the current system routes NHL value scanning through
the generic predictor. They will pass once the routing is fixed.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

pytestmark = pytest.mark.contract

from abba.server import ABBAToolkit


@pytest.fixture
def toolkit():
    return ABBAToolkit(db_path=":memory:", auto_seed=True)


class TestFindValueUsesCorrectPredictor:
    """find_value(sport="NHL") must use nhl_predict_game, not generic predict_game."""

    def test_find_value_nhl_calls_nhl_predictor(self, toolkit):
        """Patch both predictors, call find_value for NHL, assert NHL predictor was used."""
        nhl_pred_result = {
            "game_id": "test",
            "prediction": {"value": 0.60, "method": "weighted"},
            "sport": "NHL",
            "_meta": {"tool": "nhl_predict_game"},
        }
        generic_pred_result = {
            "game_id": "test",
            "prediction": {"value": 0.55, "method": "weighted"},
            "sport": "NHL",
            "_meta": {"tool": "predict_game"},
        }

        with patch.object(toolkit, "nhl_predict_game", return_value=nhl_pred_result) as nhl_mock, \
             patch.object(toolkit, "predict_game", return_value=generic_pred_result) as generic_mock:

            toolkit.find_value(sport="NHL")

            # The NHL predictor should have been called for NHL games
            # The generic predictor should NOT have been called for NHL games
            nhl_games_exist = toolkit.storage.query_games(sport="NHL", status="scheduled")
            if nhl_games_exist:
                assert nhl_mock.call_count > 0, (
                    "INVARIANT A VIOLATION: find_value(sport='NHL') did not call nhl_predict_game. "
                    "NHL value scanning is using the wrong model."
                )
                assert generic_mock.call_count == 0, (
                    "INVARIANT A VIOLATION: find_value(sport='NHL') called generic predict_game. "
                    "NHL games are being routed through the MLB/generic pseudo-model."
                )

    def test_find_value_nhl_never_uses_feature_engine(self, toolkit):
        """The generic FeatureEngine.predict_from_features must not be in the NHL value path."""
        with patch.object(toolkit.features, "predict_from_features", wraps=toolkit.features.predict_from_features) as fe_mock:
            toolkit.find_value(sport="NHL")

            nhl_games = toolkit.storage.query_games(sport="NHL", status="scheduled")
            if nhl_games:
                assert fe_mock.call_count == 0, (
                    "INVARIANT A VIOLATION: find_value(sport='NHL') used FeatureEngine.predict_from_features. "
                    "This is the generic pseudo-model, not the NHL 6-model ensemble."
                )

    def test_find_value_nhl_predictions_match_nhl_predict_game(self, toolkit):
        """Probabilities from find_value for NHL games must match nhl_predict_game output."""
        nhl_games = toolkit.storage.query_games(sport="NHL", status="scheduled")
        if not nhl_games:
            pytest.skip("No scheduled NHL games in seed data")

        game = nhl_games[0]
        gid = game["game_id"]

        # Get prediction from the NHL-specific path
        nhl_result = toolkit.nhl_predict_game(gid)
        if "error" in nhl_result:
            pytest.skip(f"nhl_predict_game returned error: {nhl_result['error']}")

        nhl_prob = nhl_result.get("prediction", {}).get("value")

        # Get prediction from find_value's internal path
        generic_result = toolkit.predict_game(gid)
        generic_prob = generic_result.get("prediction", {}).get("value")

        # These should be the same if find_value uses nhl_predict_game
        # They will differ if find_value uses the generic path
        if nhl_prob is not None and generic_prob is not None:
            if abs(nhl_prob - generic_prob) > 0.01:
                pytest.fail(
                    f"INVARIANT A VIOLATION: NHL-specific predictor returns {nhl_prob:.4f} "
                    f"but generic predictor returns {generic_prob:.4f} for game {gid}. "
                    f"find_value is using the wrong model for NHL games."
                )


class TestPredictorFamilyDeclaration:
    """If generic path is used for NHL, it must be explicitly declared."""

    def test_predict_game_checks_sport_for_nhl(self, toolkit):
        """predict_game should route NHL games to nhl_predict_game or declare why not."""
        # Read the source of predict_game to check for sport-based routing
        import inspect
        source = inspect.getsource(toolkit.predict_game)

        has_nhl_routing = (
            "nhl_predict_game" in source
            or "sport == \"NHL\"" in source
            or "sport == 'NHL'" in source
        )

        if not has_nhl_routing:
            pytest.fail(
                "INVARIANT A VIOLATION: predict_game() has no NHL-specific routing. "
                "All games go through the generic FeatureEngine path regardless of sport. "
                "NHL games need the hockey analytics ensemble, not the generic pseudo-model."
            )

    def test_market_tools_document_predictor_family(self, toolkit):
        """MarketToolsMixin.find_value must document which predictor family it uses."""
        import inspect
        source = inspect.getsource(type(toolkit).find_value)

        # Either routes by sport or has an explicit comment about using generic
        routes_by_sport = "sport" in source and ("nhl_predict" in source or "predict_game" in source)
        declares_generic = "generic" in source.lower() or "all sports" in source.lower()

        assert routes_by_sport or declares_generic, (
            "INVARIANT A VIOLATION: find_value() neither routes by sport "
            "nor documents that it intentionally uses the generic predictor for all sports."
        )
