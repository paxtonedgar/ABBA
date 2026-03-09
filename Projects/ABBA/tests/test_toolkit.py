"""Integration tests for the ABBAToolkit.

Tests the full pipeline: seed data -> query -> predict -> find value -> size bets.
This is what an agent would actually do.
"""

import pytest

from abba.server import ABBAToolkit


@pytest.fixture
def toolkit():
    """Fresh toolkit with seeded data for each test."""
    return ABBAToolkit(db_path=":memory:", auto_seed=True)


class TestToolDiscovery:
    def test_list_tools(self, toolkit):
        tools = toolkit.list_tools()
        assert len(tools) >= 20
        names = [t["name"] for t in tools]
        assert "query_games" in names
        assert "predict_game" in names
        assert "find_value" in names
        assert "kelly_sizing" in names

    def test_every_tool_has_description(self, toolkit):
        for tool in toolkit.list_tools():
            assert "description" in tool
            assert "params" in tool
            assert "category" in tool

    def test_call_tool_dispatch(self, toolkit):
        result = toolkit.call_tool("list_sources")
        assert "sources" in result

    def test_call_tool_unknown(self, toolkit):
        result = toolkit.call_tool("nonexistent_tool")
        assert "error" in result


class TestDataTools:
    def test_query_games_returns_data(self, toolkit):
        result = toolkit.query_games()
        assert result["count"] > 0
        assert "_meta" in result

    def test_query_games_filter_sport(self, toolkit):
        mlb = toolkit.query_games(sport="MLB")
        nhl = toolkit.query_games(sport="NHL")
        assert mlb["count"] > 0
        assert nhl["count"] > 0
        for g in mlb["games"]:
            assert g["sport"] == "MLB"
        for g in nhl["games"]:
            assert g["sport"] == "NHL"

    def test_query_odds(self, toolkit):
        result = toolkit.query_odds()
        assert result["count"] > 0

    def test_query_team_stats(self, toolkit):
        result = toolkit.query_team_stats(sport="MLB")
        assert result["count"] > 0

    def test_list_sources(self, toolkit):
        result = toolkit.list_sources()
        tables = [s["table"] for s in result["sources"]]
        assert "games" in tables
        assert "odds_snapshots" in tables

    def test_describe_dataset(self, toolkit):
        result = toolkit.describe_dataset("games")
        col_names = [c["column_name"] for c in result["columns"]]
        assert "game_id" in col_names
        assert "sport" in col_names


class TestAnalyticsTools:
    def test_predict_game(self, toolkit):
        games = toolkit.query_games(sport="MLB", status="scheduled")
        assert games["count"] > 0

        game_id = games["games"][0]["game_id"]
        pred = toolkit.predict_game(game_id)

        assert "prediction" in pred
        assert "features" in pred
        p = pred["prediction"]
        assert 0.0 < p["value"] < 1.0
        assert 0.0 <= p["confidence"] <= 1.0
        assert p["model_count"] == 4

    def test_predict_game_not_found(self, toolkit):
        result = toolkit.predict_game("nonexistent-game-id")
        assert "error" in result

    def test_prediction_caching(self, toolkit):
        games = toolkit.query_games(sport="MLB", status="scheduled")
        game_id = games["games"][0]["game_id"]

        # First call
        pred1 = toolkit.predict_game(game_id)
        assert pred1.get("_cache_hit") is False

        # Second call should be cached
        pred2 = toolkit.predict_game(game_id)
        assert pred2.get("_cache_hit") is True

    def test_explain_prediction(self, toolkit):
        games = toolkit.query_games(sport="MLB", status="scheduled")
        game_id = games["games"][0]["game_id"]

        result = toolkit.explain_prediction(game_id)
        assert "top_factors" in result
        assert len(result["top_factors"]) > 0
        # Top factors should be sorted by deviation
        devs = [f["deviation"] for f in result["top_factors"]]
        assert devs == sorted(devs, reverse=True)

    def test_graph_analysis(self, toolkit):
        team = {
            "players": ["Player1", "Player2", "Player3", "Player4"],
            "relationships": [
                {"player1_idx": 0, "player2_idx": 1, "weight": 0.8},
                {"player1_idx": 1, "player2_idx": 2, "weight": 0.9},
                {"player1_idx": 2, "player2_idx": 3, "weight": 0.7},
                {"player1_idx": 0, "player2_idx": 3, "weight": 0.6},
                {"player1_idx": 0, "player2_idx": 2, "weight": 0.5},
            ],
        }
        result = toolkit.graph_analysis(team)
        assert result["player_count"] == 4
        assert result["key_player_count"] >= 1
        assert 0 <= result["team_cohesion"] <= 1


class TestMarketTools:
    def test_find_value(self, toolkit):
        result = toolkit.find_value(sport="MLB", min_ev=0.01)
        # With random seed data, there should be some opportunities
        assert "opportunities" in result
        assert "games_scanned" in result
        assert result["games_scanned"] > 0

    def test_compare_odds(self, toolkit):
        games = toolkit.query_games(sport="MLB", status="scheduled")
        game_id = games["games"][0]["game_id"]
        result = toolkit.compare_odds(game_id)
        assert "books" in result
        assert len(result["books"]) > 0

    def test_calculate_ev(self, toolkit):
        # 60% at 2.0 odds: EV = 0.60*1.0 - 0.40 = 0.20
        result = toolkit.calculate_ev(0.60, 2.0)
        assert abs(result["expected_value"] - 0.20) < 0.001
        assert result["is_positive_ev"] is True

        # 40% at 2.0 odds: EV = 0.40*1.0 - 0.60 = -0.20
        result = toolkit.calculate_ev(0.40, 2.0)
        assert abs(result["expected_value"] - (-0.20)) < 0.001
        assert result["is_positive_ev"] is False

    def test_kelly_sizing(self, toolkit):
        result = toolkit.kelly_sizing(0.65, 2.0, 10000)
        assert result["recommended_stake"] > 0
        assert result["expected_value"] > 0
        assert result["edge"] > 0
        # Stake should be reasonable (half-Kelly, capped at 5%)
        assert result["recommended_stake"] <= 500  # 5% of 10000


class TestNHLTools:
    def test_nhl_predict_game(self, toolkit):
        games = toolkit.query_games(sport="NHL", status="scheduled")
        if games["count"] == 0:
            pytest.skip("no scheduled NHL games in seed data")
        game_id = games["games"][0]["game_id"]
        result = toolkit.nhl_predict_game(game_id)
        assert "prediction" in result
        assert result["model_count"] == 6

    def test_nhl_predict_wrong_sport(self, toolkit):
        games = toolkit.query_games(sport="MLB", status="scheduled")
        game_id = games["games"][0]["game_id"]
        result = toolkit.nhl_predict_game(game_id)
        assert "error" in result

    def test_query_goaltender_stats(self, toolkit):
        result = toolkit.query_goaltender_stats()
        assert result["count"] > 0
        goalie = result["goaltenders"][0]
        assert "stats" in goalie
        assert "save_pct" in goalie["stats"]

    def test_query_goaltender_by_team(self, toolkit):
        result = toolkit.query_goaltender_stats(team="NYR")
        assert result["count"] >= 1

    def test_query_advanced_stats(self, toolkit):
        result = toolkit.query_advanced_stats()
        assert result["count"] > 0
        team = result["teams"][0]
        assert "corsi_pct" in team["stats"]
        assert "xgf_pct" in team["stats"]

    def test_query_cap_data(self, toolkit):
        result = toolkit.query_cap_data(team="NYR")
        assert result["count"] > 0
        assert "cap_analysis" in result
        assert result["cap_analysis"]["roster_size"] > 0

    def test_query_roster(self, toolkit):
        result = toolkit.query_roster(team="NYR")
        assert result["count"] > 0

    def test_season_review(self, toolkit):
        result = toolkit.season_review(team_id="NYR")
        assert "record" in result
        assert "points" in result
        assert "goal_differential" in result

    def test_playoff_odds(self, toolkit):
        result = toolkit.playoff_odds(team_id="NYR")
        assert "wildcard_probability" in result
        assert "projected_points" in result


class TestSessionManagement:
    def test_session_budget(self, toolkit):
        result = toolkit.session_budget()
        assert result["budget_remaining"] > 0
        assert result["tool_calls"] >= 0

    def test_budget_decreases_with_calls(self, toolkit):
        b1 = toolkit.session_budget()["budget_remaining"]
        toolkit.query_games()
        b2 = toolkit.session_budget()["budget_remaining"]
        assert b2 < b1  # budget decreased

    def test_meta_on_every_response(self, toolkit):
        result = toolkit.query_games()
        assert "_meta" in result
        assert "latency_ms" in result["_meta"]
        assert result["_meta"]["tool"] == "query_games"


class TestAgentWorkflow:
    """Simulate what an agent would actually do end-to-end."""

    def test_full_analysis_workflow(self, toolkit):
        # 1. Discover what's available
        sources = toolkit.list_sources()
        assert any(s["table"] == "games" for s in sources["sources"])

        # 2. Find today's games
        games = toolkit.query_games(sport="MLB", status="scheduled")
        assert games["count"] > 0

        # 3. Pick a game and predict
        game_id = games["games"][0]["game_id"]
        prediction = toolkit.predict_game(game_id)
        home_prob = prediction["prediction"]["value"]
        assert 0 < home_prob < 1

        # 4. Explain the prediction
        explanation = toolkit.explain_prediction(game_id)
        assert len(explanation["top_factors"]) > 0

        # 5. Compare odds
        odds = toolkit.compare_odds(game_id)
        assert len(odds["books"]) > 0

        # 6. Calculate EV if we have odds
        if odds["best_home"]:
            ev = toolkit.calculate_ev(home_prob, odds["best_home"]["odds"])
            assert "expected_value" in ev

            # 7. Size the bet if +EV
            if ev["is_positive_ev"]:
                sizing = toolkit.kelly_sizing(home_prob, odds["best_home"]["odds"], 10000)
                assert sizing["recommended_stake"] >= 0

        # 8. Check budget
        budget = toolkit.session_budget()
        assert budget["tool_calls"] >= 6  # We made at least 6 calls
