"""Tests for workflow engine -- validates multi-step analytical pipelines."""

import pytest

from abba.server.toolkit import ABBAToolkit
from abba.workflows.engine import WorkflowEngine, list_workflows, run_workflow


@pytest.fixture
def toolkit():
    return ABBAToolkit(db_path=":memory:", auto_seed=True)


@pytest.fixture
def engine(toolkit):
    return WorkflowEngine(toolkit)


class TestWorkflowRegistry:
    def test_list_workflows_returns_all(self):
        wfs = list_workflows()
        names = [w["name"] for w in wfs]
        assert "game_prediction" in names
        assert "tonights_slate" in names
        assert "season_story" in names
        assert "value_scan" in names
        assert "cap_strategy" in names
        assert "playoff_race" in names
        assert "goaltender_duel" in names
        assert "team_comparison" in names
        assert "betting_strategy" in names
        assert len(wfs) == 9

    def test_each_workflow_has_triggers(self):
        for wf in list_workflows():
            assert len(wf["triggers"]) > 0, f"{wf['name']} has no triggers"
            assert wf["description"], f"{wf['name']} has no description"
            assert isinstance(wf["params"], dict), f"{wf['name']} params not a dict"

    def test_unknown_workflow(self, engine):
        result = engine.run("nonexistent_workflow")
        assert "error" in result
        assert "available" in result
        assert "game_prediction" in result["available"]


class TestWorkflowMeta:
    """Every workflow should attach _workflow metadata."""

    def test_meta_on_tonights_slate(self, engine):
        result = engine.run("tonights_slate")
        assert "_workflow" in result
        assert result["_workflow"]["name"] == "tonights_slate"
        assert "elapsed_ms" in result["_workflow"]
        assert result["_workflow"]["elapsed_ms"] >= 0

    def test_meta_on_season_story(self, engine):
        result = engine.run("season_story", team="NYR")
        assert "_workflow" in result
        assert result["_workflow"]["name"] == "season_story"
        assert result["_workflow"]["params"]["team"] == "NYR"


class TestTonightsSlate:
    def test_returns_games(self, engine):
        result = engine.run("tonights_slate")
        assert "games" in result
        assert "game_count" in result
        assert "value_picks" in result
        assert isinstance(result["games"], list)

    def test_games_sorted_by_confidence(self, engine):
        result = engine.run("tonights_slate")
        games = result["games"]
        if len(games) >= 2:
            for i in range(len(games) - 1):
                assert games[i]["confidence"] >= games[i + 1]["confidence"]

    def test_game_entry_structure(self, engine):
        result = engine.run("tonights_slate")
        for game in result["games"]:
            assert "game_id" in game
            assert "matchup" in game
            assert "home_team" in game
            assert "away_team" in game
            assert "home_win_prob" in game
            assert "pick" in game
            assert "confidence" in game
            assert "confidence_grade" in game
            assert "data_provenance" in game
            assert 0 <= game["home_win_prob"] <= 1


class TestSeasonStory:
    def test_full_story_structure(self, engine):
        result = engine.run("season_story", team="NYR")
        expected_chapters = [
            "team", "season", "record_summary", "underlying_numbers",
            "analytics", "goaltending", "special_teams", "roster",
            "cap_situation", "playoff_outlook", "recent_results",
        ]
        for chapter in expected_chapters:
            assert chapter in result, f"Missing chapter: {chapter}"

    def test_record_summary(self, engine):
        result = engine.run("season_story", team="NYR")
        rec = result["record_summary"]
        assert "record" in rec
        assert "points" in rec
        assert rec["points"] > 0

    def test_underlying_numbers(self, engine):
        result = engine.run("season_story", team="NYR")
        un = result["underlying_numbers"]
        assert "pythagorean_wins" in un
        assert "luck_factor" in un
        assert "luck_narrative" in un
        assert isinstance(un["luck_narrative"], str)

    def test_analytics_chapter(self, engine):
        result = engine.run("season_story", team="NYR")
        ana = result["analytics"]
        assert "corsi_pct" in ana
        assert "xgf_pct" in ana
        assert "grade" in ana
        assert "process_narrative" in ana

    def test_goaltending_chapter(self, engine):
        result = engine.run("season_story", team="NYR")
        gt = result["goaltending"]
        assert "grade" in gt
        assert "goaltenders" in gt
        assert isinstance(gt["goaltenders"], list)

    def test_cap_situation(self, engine):
        result = engine.run("season_story", team="NYR")
        cap = result["cap_situation"]
        assert "cap_space" in cap
        assert "trade_flexibility" in cap
        assert isinstance(cap["trade_flexibility"], str)

    def test_playoff_outlook(self, engine):
        result = engine.run("season_story", team="NYR")
        po = result["playoff_outlook"]
        assert "projected_points" in po
        assert "wildcard_probability" in po


class TestCapStrategy:
    def test_buyer_or_seller(self, engine):
        result = engine.run("cap_strategy", team="NYR")
        assert result["mode"] in ("buyer", "seller")
        assert "cap_space" in result
        assert "top_contracts" in result
        assert "trade_chips" in result
        assert "strategy_summary" in result

    def test_top_contracts_structure(self, engine):
        result = engine.run("cap_strategy", team="NYR")
        for c in result["top_contracts"]:
            assert "name" in c
            assert "cap_hit" in c
            assert "years_left" in c


class TestPlayoffRace:
    def test_returns_standings(self, engine):
        result = engine.run("playoff_race")
        assert "standings" in result
        assert "bubble_teams" in result
        assert len(result["standings"]) > 0

    def test_team_focus(self, engine):
        result = engine.run("playoff_race", team="NYR")
        assert result["team_focus"] is not None
        assert result["team_focus"]["team"] == "NYR"

    def test_standings_sorted_by_points(self, engine):
        result = engine.run("playoff_race")
        standings = result["standings"]
        for i in range(len(standings) - 1):
            assert standings[i]["points"] >= standings[i + 1]["points"]


class TestGoaltenderDuel:
    def test_duel_structure(self, engine):
        result = engine.run("goaltender_duel", goalie1_team="NYR", goalie2_team="FLA")
        assert "goalie1" in result
        assert "goalie2" in result
        assert "matchup_edge" in result
        assert "categories_won" in result
        assert "verdict" in result

    def test_goalie_profile_fields(self, engine):
        result = engine.run("goaltender_duel", goalie1_team="NYR", goalie2_team="BOS")
        for key in ("sv_pct", "gaa", "gsaa", "games", "selection_method", "situational_splits"):
            assert key in result["goalie1"]
            assert key in result["goalie2"]

    def test_verdict_is_string(self, engine):
        result = engine.run("goaltender_duel", goalie1_team="NYR", goalie2_team="FLA")
        assert isinstance(result["verdict"], str)
        assert len(result["verdict"]) > 0


class TestTeamComparison:
    def test_comparison_structure(self, engine):
        result = engine.run("team_comparison", team1="NYR", team2="BOS")
        assert "team1" in result
        assert "team2" in result
        assert "categories" in result
        assert "category_wins" in result
        assert "goaltending" in result
        assert "roster_depth" in result
        assert result["team1"]["abbrev"] == "NYR"
        assert result["team2"]["abbrev"] == "BOS"

    def test_six_categories(self, engine):
        result = engine.run("team_comparison", team1="NYR", team2="BOS")
        cats = result["categories"]
        assert "record" in cats
        assert "goal_diff" in cats
        assert "corsi" in cats
        assert "xgf" in cats
        assert "pp" in cats
        assert "pk" in cats

    def test_category_wins_sum(self, engine):
        result = engine.run("team_comparison", team1="NYR", team2="BOS")
        wins = result["category_wins"]
        assert wins["NYR"] + wins["BOS"] == 6


class TestValueScan:
    def test_scan_structure(self, engine):
        result = engine.run("value_scan")
        assert "games_scanned" in result
        assert "opportunities_found" in result
        assert "sized_bets" in result
        assert "bankroll" in result
        assert isinstance(result["sized_bets"], list)

    def test_custom_bankroll(self, engine):
        result = engine.run("value_scan", bankroll=5000.0)
        assert result["bankroll"] == 5000.0

    def test_sized_bets_surface_review_metadata(self, engine):
        result = engine.run("value_scan")
        for bet in result["sized_bets"]:
            assert "confidence_grade" in bet
            assert "defaulted_features" in bet
            assert "requires_manual_review" in bet


class TestBettingStrategy:
    def test_strategy_structure(self, engine):
        result = engine.run("betting_strategy")
        assert "strategy" in result
        assert "discipline_rules" in result
        assert "todays_plays" in result
        assert "play_count" in result
        assert "manual_review_play_count" in result
        assert isinstance(result["discipline_rules"], list)
        assert len(result["discipline_rules"]) >= 4

    def test_risk_tolerance_conservative(self, engine):
        result = engine.run("betting_strategy", risk_tolerance="conservative")
        assert result["strategy"]["kelly_multiplier"] == 0.25
        assert result["strategy"]["min_ev_threshold"] == 0.05

    def test_risk_tolerance_aggressive(self, engine):
        result = engine.run("betting_strategy", risk_tolerance="aggressive")
        assert result["strategy"]["kelly_multiplier"] == 0.75
        assert result["strategy"]["min_ev_threshold"] == 0.02


class TestGamePrediction:
    def test_no_args_returns_error(self, engine):
        result = engine.run("game_prediction")
        assert "error" in result

    def test_with_team(self, engine):
        result = engine.run("game_prediction", team="NYR")
        # May or may not find a game depending on seed data
        if "error" not in result:
            assert "headline" in result
            assert "prediction" in result
            assert "key_factors" in result
            assert "confidence" in result
            assert "data_provenance" in result
            assert "best_bet" in result
            assert "context_notes" in result

    def test_selected_goalie_matches_prediction_provenance(self, engine):
        result = engine.run("game_prediction", team="NYR")
        if "error" not in result:
            home_goalie = result["data_provenance"]["home_goaltender"]["name"]
            away_goalie = result["data_provenance"]["away_goaltender"]["name"]
            assert result["home_team"]["starter"] == home_goalie
            assert result["away_team"]["starter"] == away_goalie


class TestConvenienceFunction:
    def test_run_workflow_function(self, toolkit):
        result = run_workflow("tonights_slate", toolkit=toolkit)
        assert "games" in result
        assert "_workflow" in result
