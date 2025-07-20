"""
Test suite for sport-specific statistical analysis functionality.
Tests MLB and NHL statistical analysis, player performance analysis, and insights generation.
"""

import os
import sys
from unittest.mock import AsyncMock, Mock

import numpy as np
import pandas as pd
import pytest
from analytics import AnalyticsModule, SportSpecificStats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the main files
from agents import AnalyticsAgent
from database import DatabaseManager


class TestSportSpecificStats:
    """Test the SportSpecificStats class for MLB and NHL analysis."""

    def setup_method(self):
        """Set up test data and objects."""
        self.stats_analyzer = SportSpecificStats()

        # Create mock MLB data
        self.mlb_data = pd.DataFrame(
            {
                "game_date": pd.date_range("2024-01-01", periods=100, freq="D"),
                "player_name": ["Player A", "Player B"] * 50,
                "pitcher": ["Pitcher A", "Pitcher B"] * 50,
                "batter": ["Batter A", "Batter B"] * 50,
                "release_speed": np.random.normal(92, 3, 100),
                "launch_speed": np.random.normal(88, 8, 100),
                "launch_angle": np.random.normal(15, 8, 100),
                "pitch_type": np.random.choice(["FF", "SL", "CH", "CB"], 100),
                "release_spin_rate": np.random.normal(2200, 300, 100),
                "pfx_x": np.random.normal(0, 2, 100),
                "pfx_z": np.random.normal(0, 2, 100),
                "plate_x": np.random.normal(0, 1, 100),
                "plate_z": np.random.normal(2.5, 0.5, 100),
                "zone": np.random.randint(1, 10, 100),
                "balls": np.random.randint(0, 4, 100),
                "strikes": np.random.randint(0, 3, 100),
                "home_team": ["Team A", "Team B"] * 50,
                "away_team": ["Team B", "Team A"] * 50,
                "stand": np.random.choice(["L", "R"], 100),
                "p_throws": np.random.choice(["L", "R"], 100),
                "estimated_ba_using_speedangle": np.random.uniform(0.2, 0.4, 100),
                "estimated_woba_using_speedangle": np.random.uniform(0.3, 0.5, 100),
            }
        )

        # Create mock NHL data
        self.nhl_data = pd.DataFrame(
            {
                "game_date": pd.date_range("2024-01-01", periods=100, freq="D"),
                "player_name": ["Player A", "Player B"] * 50,
                "shooter": ["Shooter A", "Shooter B"] * 50,
                "goalie": ["Goalie A", "Goalie B"] * 50,
                "shot_distance": np.random.normal(20, 8, 100),
                "shot_angle": np.random.normal(25, 10, 100),
                "x_coordinate": np.random.normal(0, 50, 100),
                "y_coordinate": np.random.normal(0, 50, 100),
                "manpower_situation": np.random.choice(["5v5", "PP", "SH"], 100),
                "game_seconds_remaining": np.random.randint(0, 3600, 100),
                "goal": np.random.choice([0, 1], 100, p=[0.9, 0.1]),
                "save_percentage": np.random.uniform(0.85, 0.95, 100),
                "home_team": ["Team A", "Team B"] * 50,
                "away_team": ["Team B", "Team A"] * 50,
            }
        )

    def test_analyze_mlb_pitching_stats(self):
        """Test MLB pitching statistics analysis."""
        stats = self.stats_analyzer.analyze_mlb_pitching_stats(self.mlb_data)

        assert isinstance(stats, dict)
        assert "avg_velocity" in stats
        assert "velocity_std" in stats
        assert "max_velocity" in stats
        assert "velocity_percentiles" in stats
        assert "pitch_type_distribution" in stats
        assert "fastball_percentage" in stats
        assert "breaking_percentage" in stats
        assert "offspeed_percentage" in stats
        assert "avg_spin_rate" in stats
        assert "pitch_quality_score" in stats
        assert "velocity_consistency" in stats

        # Test specific calculations
        assert stats["avg_velocity"] > 0
        assert stats["velocity_std"] > 0
        assert stats["max_velocity"] > stats["avg_velocity"]
        assert 0 <= stats["fastball_percentage"] <= 100
        assert 0 <= stats["breaking_percentage"] <= 100
        assert 0 <= stats["offspeed_percentage"] <= 100

    def test_analyze_mlb_batting_stats(self):
        """Test MLB batting statistics analysis."""
        stats = self.stats_analyzer.analyze_mlb_batting_stats(self.mlb_data)

        assert isinstance(stats, dict)
        assert "avg_exit_velocity" in stats
        assert "exit_velocity_std" in stats
        assert "barrel_percentage" in stats
        assert "hard_hit_percentage" in stats
        assert "avg_launch_angle" in stats
        assert "launch_angle_distribution" in stats
        assert "expected_batting_average" in stats
        assert "expected_woba" in stats
        assert "plate_discipline" in stats

        # Test specific calculations
        assert stats["avg_exit_velocity"] > 0
        assert stats["exit_velocity_std"] > 0
        assert 0 <= stats["barrel_percentage"] <= 100
        assert 0 <= stats["hard_hit_percentage"] <= 100
        assert isinstance(stats["launch_angle_distribution"], dict)
        assert "ground_balls" in stats["launch_angle_distribution"]
        assert "line_drives" in stats["launch_angle_distribution"]
        assert "fly_balls" in stats["launch_angle_distribution"]

    def test_analyze_nhl_shot_stats(self):
        """Test NHL shot statistics analysis."""
        stats = self.stats_analyzer.analyze_nhl_shot_stats(self.nhl_data)

        assert isinstance(stats, dict)
        assert "avg_shot_distance" in stats
        assert "shot_distance_distribution" in stats
        assert "avg_shot_angle" in stats
        assert "high_danger_percentage" in stats
        assert "shot_location_analysis" in stats
        assert "powerplay_performance" in stats
        assert "even_strength_performance" in stats
        assert "period_performance" in stats
        assert "clutch_performance" in stats
        assert "shot_quality_score" in stats
        assert "scoring_chance_percentage" in stats

        # Test specific calculations
        assert stats["avg_shot_distance"] > 0
        assert isinstance(stats["shot_distance_distribution"], dict)
        assert "close_range" in stats["shot_distance_distribution"]
        assert "medium_range" in stats["shot_distance_distribution"]
        assert "long_range" in stats["shot_distance_distribution"]
        assert 0 <= stats["high_danger_percentage"] <= 100

    def test_analyze_nhl_goaltending_stats(self):
        """Test NHL goaltending statistics analysis."""
        stats = self.stats_analyzer.analyze_nhl_goaltending_stats(self.nhl_data)

        assert isinstance(stats, dict)
        assert "avg_save_percentage" in stats
        assert "save_percentage_by_situation" in stats
        assert "expected_save_percentage" in stats
        assert "save_percentage_vs_expected" in stats

        # Test specific calculations
        assert 0.8 <= stats["avg_save_percentage"] <= 1.0
        assert isinstance(stats["save_percentage_by_situation"], dict)

    def test_generate_mlb_insights(self):
        """Test MLB insights generation."""
        insights = self.stats_analyzer.generate_mlb_insights(self.mlb_data)

        assert isinstance(insights, dict)
        assert (
            "velocity_trend" in insights
            or "barrel_analysis" in insights
            or "plate_discipline" in insights
        )

        # Test that insights provide actionable information
        if "velocity_trend" in insights:
            assert "change" in insights["velocity_trend"]
            assert "trend" in insights["velocity_trend"]
            assert "recommendation" in insights["velocity_trend"]

    def test_generate_nhl_insights(self):
        """Test NHL insights generation."""
        insights = self.stats_analyzer.generate_nhl_insights(self.nhl_data)

        assert isinstance(insights, dict)
        assert (
            "shot_quality" in insights
            or "power_play" in insights
            or "period_analysis" in insights
        )

        # Test that insights provide actionable information
        if "shot_quality" in insights:
            assert "avg_distance" in insights["shot_quality"]
            assert "high_danger_percentage" in insights["shot_quality"]
            assert "quality_level" in insights["shot_quality"]
            assert "recommendation" in insights["shot_quality"]

    def test_calculate_barrel_percentage(self):
        """Test barrel percentage calculation."""
        # Create data with known barrel conditions
        barrel_data = self.mlb_data.copy()
        barrel_data["launch_angle"] = 28  # Perfect barrel angle
        barrel_data["launch_speed"] = 100  # Perfect barrel speed

        barrel_pct = self.stats_analyzer._calculate_barrel_percentage(barrel_data)
        assert barrel_pct > 0

    def test_calculate_strike_zone_accuracy(self):
        """Test strike zone accuracy calculation."""
        accuracy = self.stats_analyzer._calculate_strike_zone_accuracy(self.mlb_data)
        assert 0 <= accuracy <= 100

    def test_calculate_edge_percentage(self):
        """Test edge percentage calculation."""
        edge_pct = self.stats_analyzer._calculate_edge_percentage(self.mlb_data)
        assert 0 <= edge_pct <= 100

    def test_calculate_pitch_quality_score(self):
        """Test pitch quality score calculation."""
        score = self.stats_analyzer._calculate_pitch_quality_score(self.mlb_data)
        assert isinstance(score, float)

    def test_calculate_shot_quality_score(self):
        """Test shot quality score calculation."""
        score = self.stats_analyzer._calculate_shot_quality_score(self.nhl_data)
        assert isinstance(score, float)

    def test_calculate_scoring_chance_percentage(self):
        """Test scoring chance percentage calculation."""
        chance_pct = self.stats_analyzer._calculate_scoring_chance_percentage(
            self.nhl_data
        )
        assert 0 <= chance_pct <= 100

    def test_calculate_expected_save_percentage(self):
        """Test expected save percentage calculation."""
        expected_save = self.stats_analyzer._calculate_expected_save_percentage(
            self.nhl_data
        )
        assert 0.8 <= expected_save <= 1.0


class TestAnalyticsModuleSportStats:
    """Test the AnalyticsModule's sport-specific statistics integration."""

    def setup_method(self):
        """Set up test configuration and objects."""
        self.config = {
            "mlb_api_key": "test_key",
            "nhl_api_key": "test_key",
            "feature_engineering": {
                "rolling_windows": [5, 10, 20],
                "temporal_features": True,
                "situational_features": True,
                "interaction_features": True,
            },
            "apis": {"openai": {"model": "gpt-4", "api_key": "test_key"}},
        }
        self.analytics_module = AnalyticsModule(self.config)

        # Create test data
        self.mlb_data = pd.DataFrame(
            {
                "game_date": pd.date_range("2024-01-01", periods=50, freq="D"),
                "player_name": ["Player A", "Player B"] * 25,
                "release_speed": np.random.normal(92, 3, 50),
                "launch_speed": np.random.normal(88, 8, 50),
                "launch_angle": np.random.normal(15, 8, 50),
                "home_team": ["Team A", "Team B"] * 25,
            }
        )

        self.nhl_data = pd.DataFrame(
            {
                "game_date": pd.date_range("2024-01-01", periods=50, freq="D"),
                "player_name": ["Player A", "Player B"] * 25,
                "shot_distance": np.random.normal(20, 8, 50),
                "shot_angle": np.random.normal(25, 10, 50),
                "home_team": ["Team A", "Team B"] * 25,
            }
        )

    @pytest.mark.asyncio
    async def test_get_comprehensive_mlb_stats(self):
        """Test comprehensive MLB statistics generation."""
        stats = await self.analytics_module.get_comprehensive_mlb_stats(self.mlb_data)

        assert isinstance(stats, dict)
        assert "pitching_stats" in stats
        assert "batting_stats" in stats
        assert "insights" in stats
        assert "summary" in stats

        # Test summary
        summary = stats["summary"]
        assert summary["total_records"] == 50
        assert summary["teams_analyzed"] == 2
        assert summary["players_analyzed"] == 2
        assert "data_quality" in summary

    @pytest.mark.asyncio
    async def test_get_comprehensive_nhl_stats(self):
        """Test comprehensive NHL statistics generation."""
        stats = await self.analytics_module.get_comprehensive_nhl_stats(self.nhl_data)

        assert isinstance(stats, dict)
        assert "shot_stats" in stats
        assert "goaltending_stats" in stats
        assert "insights" in stats
        assert "summary" in stats

        # Test summary
        summary = stats["summary"]
        assert summary["total_records"] == 50
        assert summary["teams_analyzed"] == 2
        assert summary["players_analyzed"] == 2
        assert "data_quality" in summary

    @pytest.mark.asyncio
    async def test_analyze_player_performance_mlb(self):
        """Test individual player performance analysis for MLB."""
        player_analysis = await self.analytics_module.analyze_player_performance(
            self.mlb_data, "Player A", "mlb"
        )

        assert isinstance(player_analysis, dict)
        assert player_analysis["player_name"] == "Player A"
        assert player_analysis["sport"] == "MLB"
        assert "pitching_analysis" in player_analysis
        assert "batting_analysis" in player_analysis
        assert "insights" in player_analysis
        assert "performance_trends" in player_analysis

    @pytest.mark.asyncio
    async def test_analyze_player_performance_nhl(self):
        """Test individual player performance analysis for NHL."""
        player_analysis = await self.analytics_module.analyze_player_performance(
            self.nhl_data, "Player A", "nhl"
        )

        assert isinstance(player_analysis, dict)
        assert player_analysis["player_name"] == "Player A"
        assert player_analysis["sport"] == "NHL"
        assert "shot_analysis" in player_analysis
        assert "goaltending_analysis" in player_analysis
        assert "insights" in player_analysis
        assert "performance_trends" in player_analysis

    @pytest.mark.asyncio
    async def test_compare_players(self):
        """Test player comparison functionality."""
        players = ["Player A", "Player B"]
        comparison = await self.analytics_module.compare_players(
            self.mlb_data, players, "mlb"
        )

        assert isinstance(comparison, dict)
        assert "Player A" in comparison
        assert "Player B" in comparison
        assert "comparative_analysis" in comparison

    def test_analyze_player_trends(self):
        """Test player trend analysis."""
        trends = self.analytics_module._analyze_player_trends(self.mlb_data, "mlb")

        assert isinstance(trends, dict)
        if "velocity_trend" in trends:
            assert "slope" in trends["velocity_trend"]
            assert "trend" in trends["velocity_trend"]

    def test_generate_comparative_analysis(self):
        """Test comparative analysis generation."""
        players = ["Player A", "Player B"]
        comparative = self.analytics_module._generate_comparative_analysis(
            self.mlb_data, players, "mlb"
        )

        assert isinstance(comparative, dict)
        if "velocity_comparison" in comparative:
            assert "Player A" in comparative["velocity_comparison"]
            assert "Player B" in comparative["velocity_comparison"]


class TestAnalyticsAgentSportStats:
    """Test the AnalyticsAgent's sport-specific statistics tools."""

    def setup_method(self):
        """Set up test configuration and objects."""
        self.config = {
            "mlb_api_key": "test_key",
            "nhl_api_key": "test_key",
            "feature_engineering": {
                "rolling_windows": [5, 10, 20],
                "temporal_features": True,
                "situational_features": True,
                "interaction_features": True,
            },
            "apis": {"openai": {"model": "gpt-4", "api_key": "test_key"}},
        }

        # Mock database manager
        self.db_manager = Mock(spec=DatabaseManager)

        # Mock analytics module
        self.analytics_module = Mock(spec=AnalyticsModule)
        self.analytics_module.get_comprehensive_mlb_stats = AsyncMock()
        self.analytics_module.get_comprehensive_nhl_stats = AsyncMock()
        self.analytics_module.analyze_player_performance = AsyncMock()
        self.analytics_module.compare_players = AsyncMock()
        self.analytics_module.fetch_mlb_data = AsyncMock()
        self.analytics_module.fetch_nhl_data = AsyncMock()

        self.agent = AnalyticsAgent(self.config, self.db_manager, self.analytics_module)

    @pytest.mark.asyncio
    async def test_analyze_mlb_statcast_enhanced(self):
        """Test enhanced MLB Statcast analysis."""
        # Mock data
        mock_data = pd.DataFrame(
            {
                "player_name": ["Player A", "Player B"],
                "release_speed": [95, 88],
                "home_team": ["Team A", "Team B"],
            }
        )

        mock_stats = {
            "pitching_stats": {"avg_velocity": 91.5},
            "batting_stats": {"barrel_percentage": 8.5},
            "insights": {"velocity_trend": {"trend": "stable"}},
        }

        self.analytics_module.fetch_mlb_data.return_value = mock_data
        self.analytics_module.get_comprehensive_mlb_stats.return_value = mock_stats

        result = await self.agent._analyze_mlb_statcast()

        assert isinstance(result, dict)
        assert "data_summary" in result
        assert "pitching_analysis" in result
        assert "batting_analysis" in result
        assert "insights" in result
        assert "key_findings" in result

        # Test key findings extraction
        assert isinstance(result["key_findings"], list)

    @pytest.mark.asyncio
    async def test_analyze_nhl_shots_enhanced(self):
        """Test enhanced NHL shot analysis."""
        # Mock data
        mock_data = pd.DataFrame(
            {
                "player_name": ["Player A", "Player B"],
                "shot_distance": [15, 25],
                "home_team": ["Team A", "Team B"],
            }
        )

        mock_stats = {
            "shot_stats": {"avg_shot_distance": 20},
            "goaltending_stats": {"avg_save_percentage": 0.92},
            "insights": {"shot_quality": {"quality_level": "Medium"}},
        }

        self.analytics_module.fetch_nhl_data.return_value = mock_data
        self.analytics_module.get_comprehensive_nhl_stats.return_value = mock_stats

        result = await self.agent._analyze_nhl_shots()

        assert isinstance(result, dict)
        assert "data_summary" in result
        assert "shot_analysis" in result
        assert "goaltending_analysis" in result
        assert "insights" in result
        assert "key_findings" in result

        # Test key findings extraction
        assert isinstance(result["key_findings"], list)

    @pytest.mark.asyncio
    async def test_analyze_player_performance_tool(self):
        """Test the analyze_player_performance tool."""
        # Mock data and analysis
        mock_data = pd.DataFrame({"player_name": ["Player A"]})
        mock_analysis = {
            "player_name": "Player A",
            "sport": "MLB",
            "pitching_analysis": {"avg_velocity": 92},
            "batting_analysis": {"barrel_percentage": 10},
        }

        self.analytics_module.fetch_mlb_data.return_value = mock_data
        self.analytics_module.analyze_player_performance.return_value = mock_analysis

        result = await self.agent.analyze_player_performance("Player A", "mlb")

        assert isinstance(result, dict)
        assert result["player_name"] == "Player A"
        assert result["sport"] == "MLB"
        assert "recommendations" in result

    @pytest.mark.asyncio
    async def test_compare_players_tool(self):
        """Test the compare_players tool."""
        # Mock data and comparison
        mock_data = pd.DataFrame({"player_name": ["Player A", "Player B"]})
        mock_comparison = {
            "Player A": {"player_name": "Player A"},
            "Player B": {"player_name": "Player B"},
            "comparative_analysis": {"velocity_comparison": {}},
        }

        self.analytics_module.fetch_mlb_data.return_value = mock_data
        self.analytics_module.compare_players.return_value = mock_comparison

        result = await self.agent.compare_players(["Player A", "Player B"], "mlb")

        assert isinstance(result, dict)
        assert "Player A" in result
        assert "Player B" in result
        assert "rankings" in result

    @pytest.mark.asyncio
    async def test_generate_sport_insights_tool(self):
        """Test the generate_sport_insights tool."""
        # Mock data and stats
        mock_data = pd.DataFrame({"player_name": ["Player A"]})
        mock_stats = {
            "pitching_stats": {"avg_velocity": 92},
            "batting_stats": {"barrel_percentage": 10},
        }

        self.analytics_module.fetch_mlb_data.return_value = mock_data
        self.analytics_module.get_comprehensive_mlb_stats.return_value = mock_stats

        result = await self.agent.generate_sport_insights("mlb", "comprehensive")

        assert isinstance(result, dict)
        assert "betting_implications" in result

    def test_extract_mlb_key_findings(self):
        """Test MLB key findings extraction."""
        stats = {
            "pitching_stats": {"avg_velocity": 96},
            "batting_stats": {"barrel_percentage": 12},
        }

        findings = self.agent._extract_mlb_key_findings(stats)

        assert isinstance(findings, list)
        assert len(findings) > 0
        assert any("High average velocity" in finding for finding in findings)
        assert any("High barrel rate" in finding for finding in findings)

    def test_extract_nhl_key_findings(self):
        """Test NHL key findings extraction."""
        stats = {"shot_stats": {"high_danger_percentage": 45, "avg_shot_distance": 12}}

        findings = self.agent._extract_nhl_key_findings(stats)

        assert isinstance(findings, list)
        assert len(findings) > 0
        assert any("High danger shot percentage" in finding for finding in findings)
        assert any("Close-range shooting" in finding for finding in findings)

    def test_generate_player_recommendations(self):
        """Test player recommendation generation."""
        player_analysis = {
            "pitching_analysis": {
                "velocity_consistency": 4,
                "strike_zone_accuracy": 55,
            },
            "batting_analysis": {
                "barrel_percentage": 6,
                "plate_discipline": {"strikeout_rate": 28},
            },
        }

        recommendations = self.agent._generate_player_recommendations(
            player_analysis, "mlb"
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("velocity consistency" in rec.lower() for rec in recommendations)
        assert any("command" in rec.lower() for rec in recommendations)

    def test_generate_player_rankings(self):
        """Test player ranking generation."""
        data = pd.DataFrame(
            {
                "player_name": ["Player A", "Player B", "Player A", "Player B"],
                "release_speed": [95, 88, 94, 89],
                "launch_speed": [90, 85, 91, 86],
            }
        )

        rankings = self.agent._generate_player_rankings(
            data, ["Player A", "Player B"], "mlb"
        )

        assert isinstance(rankings, dict)
        assert "velocity_ranking" in rankings
        assert "exit_velocity_ranking" in rankings
        assert len(rankings["velocity_ranking"]) == 2
        assert len(rankings["exit_velocity_ranking"]) == 2

    def test_generate_comprehensive_insights(self):
        """Test comprehensive insights generation."""
        stats = {
            "pitching_stats": {"velocity_consistency": 4},
            "batting_stats": {"barrel_percentage": 6},
        }

        insights = self.agent._generate_comprehensive_insights(stats, "mlb")

        assert isinstance(insights, dict)
        assert "sport" in insights
        assert "key_metrics" in insights
        assert "recommendations" in insights
        assert insights["sport"] == "MLB"

    def test_generate_betting_implications(self):
        """Test betting implications generation."""
        insights = {"key_metrics": {"barrel_percentage": 15, "avg_velocity": 88}}

        implications = self.agent._generate_betting_implications(insights, "mlb")

        assert isinstance(implications, dict)
        assert "value_opportunities" in implications
        assert "risk_factors" in implications
        assert "confidence_level" in implications


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
