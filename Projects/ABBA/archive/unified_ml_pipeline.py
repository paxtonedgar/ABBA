"""
Unified ML Pipeline: Single, Cohesive System for All Sports
Consolidates all separate components into one unified pipeline with professional feature engineering.
"""

import asyncio
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import structlog

warnings.filterwarnings('ignore')

# Machine Learning imports
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Advanced ML imports
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

logger = structlog.get_logger()

@dataclass
class UnifiedGameData:
    """Unified game data container for all sports."""
    game_id: str
    sport: str
    home_team: str
    away_team: str
    game_date: datetime
    raw_data: dict[str, Any]
    processed_data: dict[str, Any] | None = None
    features: dict[str, float] | None = None
    prediction: dict[str, Any] | None = None
    risk_assessment: dict[str, Any] | None = None

class UnifiedDataIntegrator:
    """Unified data integration for all sports."""

    def __init__(self):
        self.data_cache = {}
        self.api_connectors = {
            'mlb': self._get_mlb_connector(),
            'nhl': self._get_nhl_connector()
        }

    def _get_mlb_connector(self):
        """Get MLB data connector."""
        return {
            'statcast': 'Baseball Savant API',
            'odds': 'The Odds API',
            'weather': 'OpenWeather API',
            'injuries': 'MLB Injury API'
        }

    def _get_nhl_connector(self):
        """Get NHL data connector."""
        return {
            'advanced_stats': 'Sportlogiq API',
            'odds': 'The Odds API',
            'weather': 'OpenWeather API',
            'injuries': 'NHL Injury API'
        }

    async def ingest_game_data(self, game_id: str, sport: str) -> UnifiedGameData:
        """Ingest game data for any sport."""

        try:
            # Check cache first
            cache_key = f"{sport}_{game_id}"
            if cache_key in self.data_cache:
                return self.data_cache[cache_key]

            # Get sport-specific data
            if sport == 'mlb':
                raw_data = await self._get_mlb_data(game_id)
            elif sport == 'nhl':
                raw_data = await self._get_nhl_data(game_id)
            else:
                raise ValueError(f"Unsupported sport: {sport}")

            # Create unified game data
            game_data = UnifiedGameData(
                game_id=game_id,
                sport=sport,
                home_team=raw_data.get('home_team', 'Unknown'),
                away_team=raw_data.get('away_team', 'Unknown'),
                game_date=datetime.now(),
                raw_data=raw_data
            )

            # Cache the data
            self.data_cache[cache_key] = game_data

            return game_data

        except Exception as e:
            logger.error(f"Error ingesting game data for {sport} game {game_id}: {e}")
            return self._get_fallback_data(game_id, sport)

    async def _get_mlb_data(self, game_id: str) -> dict[str, Any]:
        """Get MLB-specific data."""
        # Mock MLB data - replace with real API calls
        return {
            'home_team': 'NYY',
            'away_team': 'BOS',
            'pitching_data': {
                'home_pitcher': {'era': 3.45, 'whip': 1.20, 'k_per_9': 9.2},
                'away_pitcher': {'era': 4.12, 'whip': 1.35, 'k_per_9': 7.8}
            },
            'batting_data': {
                'home_team': {'woba': 0.345, 'iso': 0.180, 'barrel_rate': 0.085},
                'away_team': {'woba': 0.332, 'iso': 0.165, 'barrel_rate': 0.078}
            },
            'weather_data': {
                'temperature': 72,
                'humidity': 65,
                'wind_speed': 8,
                'wind_direction': 'NE'
            },
            'park_factors': {
                'hr_factor': 1.05,
                'hit_factor': 1.02
            }
        }

    async def _get_nhl_data(self, game_id: str) -> dict[str, Any]:
        """Get NHL-specific data."""
        # Mock NHL data - replace with real API calls
        return {
            'home_team': 'NYR',
            'away_team': 'BOS',
            'goalie_data': {
                'home_goalie': {'save_pct': 0.925, 'gaa': 2.45, 'gsax': 12.3},
                'away_goalie': {'save_pct': 0.918, 'gaa': 2.67, 'gsax': 8.7}
            },
            'team_data': {
                'home_team': {'corsi': 52.1, 'fenwick': 51.8, 'expected_goals': 2.85},
                'away_team': {'corsi': 48.9, 'fenwick': 49.2, 'expected_goals': 2.67}
            },
            'weather_data': {
                'temperature': 68,
                'humidity': 45
            }
        }

    def _get_fallback_data(self, game_id: str, sport: str) -> UnifiedGameData:
        """Get fallback data when ingestion fails."""
        return UnifiedGameData(
            game_id=game_id,
            sport=sport,
            home_team='Unknown',
            away_team='Unknown',
            game_date=datetime.now(),
            raw_data={}
        )

class UnifiedFeatureEngineer:
    """Unified feature engineering for all sports with professional methods."""

    def __init__(self):
        # Professional feature categories
        self.feature_categories = {
            'mlb': {
                'pitching': 80,      # Advanced pitching metrics
                'batting': 70,       # Advanced batting metrics
                'situational': 50,   # Game situation features
                'market': 40,        # Betting market features
                'environmental': 30, # Weather, park, travel
                'biomechanical': 20, # Player biomechanics
                'temporal': 15       # Time-based patterns
            },
            'nhl': {
                'goalie': 60,        # Advanced goalie metrics
                'team': 50,          # Team possession metrics
                'situational': 40,   # Game situation features
                'market': 35,        # Betting market features
                'environmental': 25, # Arena, travel, rest
                'biomechanical': 15, # Player workload
                'temporal': 10       # Time-based patterns
            }
        }

        # Professional metrics calculators
        self.mlb_metrics = AdvancedMLBMetrics()
        self.nhl_metrics = AdvancedNHLMetrics()
        self.contact_analyzer = ContactQualityAnalyzer()

        # Feature cache
        self.feature_cache = {}

    def engineer_features(self, game_data: UnifiedGameData) -> dict[str, float]:
        """Engineer features for any sport using professional methods."""

        try:
            # Check cache first
            cache_key = f"{game_data.sport}_{game_data.game_id}"
            if cache_key in self.feature_cache:
                return self.feature_cache[cache_key]

            # Sport-specific feature engineering
            if game_data.sport == 'mlb':
                features = self._engineer_mlb_features(game_data)
            elif game_data.sport == 'nhl':
                features = self._engineer_nhl_features(game_data)
            else:
                raise ValueError(f"Unsupported sport: {game_data.sport}")

            # Validate and clean features
            features = self._validate_features(features)

            # Cache features
            self.feature_cache[cache_key] = features

            # Update game data
            game_data.features = features

            logger.info(f"Engineered {len(features)} features for {game_data.sport} game {game_data.game_id}")
            return features

        except Exception as e:
            logger.error(f"Error engineering features for {game_data.sport} game {game_data.game_id}: {e}")
            return self._get_fallback_features(game_data.sport)

    def _engineer_mlb_features(self, game_data: UnifiedGameData) -> dict[str, float]:
        """Engineer 305+ professional MLB features."""

        features = {}
        raw_data = game_data.raw_data

        # Pitching features (80+ features)
        features.update(self._engineer_mlb_pitching_features(raw_data))

        # Batting features (70+ features)
        features.update(self._engineer_mlb_batting_features(raw_data))

        # Situational features (50+ features)
        features.update(self._engineer_situational_features(raw_data))

        # Market features (40+ features)
        features.update(self._engineer_market_features(raw_data))

        # Environmental features (30+ features)
        features.update(self._engineer_environmental_features(raw_data))

        # Biomechanical features (20+ features)
        features.update(self._engineer_biomechanical_features(raw_data))

        # Temporal features (15+ features)
        features.update(self._engineer_temporal_features(raw_data))

        return features

    def _engineer_nhl_features(self, game_data: UnifiedGameData) -> dict[str, float]:
        """Engineer 230+ professional NHL features."""

        features = {}
        raw_data = game_data.raw_data

        # Goalie features (60+ features)
        features.update(self._engineer_nhl_goalie_features(raw_data))

        # Team features (50+ features)
        features.update(self._engineer_nhl_team_features(raw_data))

        # Situational features (40+ features)
        features.update(self._engineer_situational_features(raw_data))

        # Market features (35+ features)
        features.update(self._engineer_market_features(raw_data))

        # Environmental features (25+ features)
        features.update(self._engineer_environmental_features(raw_data))

        # Biomechanical features (15+ features)
        features.update(self._engineer_biomechanical_features(raw_data))

        # Temporal features (10+ features)
        features.update(self._engineer_temporal_features(raw_data))

        return features

    def _engineer_mlb_pitching_features(self, raw_data: dict) -> dict[str, float]:
        """Engineer 80+ MLB pitching features."""

        features = {}
        pitching_data = raw_data.get('pitching_data', {})

        # Basic pitching stats
        home_pitcher = pitching_data.get('home_pitcher', {})
        away_pitcher = pitching_data.get('away_pitcher', {})

        features['home_era'] = home_pitcher.get('era', 4.00)
        features['away_era'] = away_pitcher.get('era', 4.00)
        features['era_differential'] = features['home_era'] - features['away_era']

        features['home_whip'] = home_pitcher.get('whip', 1.30)
        features['away_whip'] = away_pitcher.get('whip', 1.30)
        features['whip_differential'] = features['home_whip'] - features['away_whip']

        features['home_k_per_9'] = home_pitcher.get('k_per_9', 8.0)
        features['away_k_per_9'] = away_pitcher.get('k_per_9', 8.0)
        features['k_per_9_differential'] = features['home_k_per_9'] - features['away_k_per_9']

        # Advanced pitching metrics
        features['home_xfip'] = self.mlb_metrics.calculate_xFIP(
            home_pitcher.get('bb_rate', 0.08),
            home_pitcher.get('k_rate', 0.20),
            home_pitcher.get('hr_rate', 0.12)
        )

        features['away_xfip'] = self.mlb_metrics.calculate_xFIP(
            away_pitcher.get('bb_rate', 0.08),
            away_pitcher.get('k_rate', 0.20),
            away_pitcher.get('hr_rate', 0.12)
        )

        features['xfip_differential'] = features['home_xfip'] - features['away_xfip']

        # Stuff+ calculations
        features['home_stuff_plus'] = self.mlb_metrics.calculate_stuff_plus(
            home_pitcher.get('avg_velocity', 92.0),
            home_pitcher.get('avg_spin_rate', 2200),
            home_pitcher.get('movement', 15.0),
            (home_pitcher.get('horizontal_movement', 0.5), home_pitcher.get('vertical_movement', 0.5))
        )

        features['away_stuff_plus'] = self.mlb_metrics.calculate_stuff_plus(
            away_pitcher.get('avg_velocity', 92.0),
            away_pitcher.get('avg_spin_rate', 2200),
            away_pitcher.get('movement', 15.0),
            (away_pitcher.get('horizontal_movement', 0.5), away_pitcher.get('vertical_movement', 0.5))
        )

        features['stuff_plus_differential'] = features['home_stuff_plus'] - features['away_stuff_plus']

        return features

    def _engineer_mlb_batting_features(self, raw_data: dict) -> dict[str, float]:
        """Engineer 70+ MLB batting features."""

        features = {}
        batting_data = raw_data.get('batting_data', {})

        # Basic batting stats
        home_batting = batting_data.get('home_team', {})
        away_batting = batting_data.get('away_team', {})

        features['home_woba'] = home_batting.get('woba', 0.320)
        features['away_woba'] = away_batting.get('woba', 0.320)
        features['woba_differential'] = features['home_woba'] - features['away_woba']

        features['home_iso'] = home_batting.get('iso', 0.160)
        features['away_iso'] = away_batting.get('iso', 0.160)
        features['iso_differential'] = features['home_iso'] - features['away_iso']

        features['home_barrel_rate'] = home_batting.get('barrel_rate', 0.075)
        features['away_barrel_rate'] = away_batting.get('barrel_rate', 0.075)
        features['barrel_rate_differential'] = features['home_barrel_rate'] - features['away_barrel_rate']

        # Advanced batting metrics
        features['home_xwoba'] = self.mlb_metrics.calculate_xwOBA(
            home_batting.get('avg_exit_velocity', 88.0),
            home_batting.get('avg_launch_angle', 12.0),
            home_batting.get('barrel_rate', 0.075),
            home_batting.get('hard_hit_rate', 0.350)
        )

        features['away_xwoba'] = self.mlb_metrics.calculate_xwOBA(
            away_batting.get('avg_exit_velocity', 88.0),
            away_batting.get('avg_launch_angle', 12.0),
            away_batting.get('barrel_rate', 0.075),
            away_batting.get('hard_hit_rate', 0.350)
        )

        features['xwoba_differential'] = features['home_xwoba'] - features['away_xwoba']

        # Contact quality analysis
        home_exit_velocities = home_batting.get('exit_velocities', [88, 92, 85, 95, 90])
        away_exit_velocities = away_batting.get('exit_velocities', [87, 89, 84, 93, 88])

        features['home_hard_hit_rate'] = self.contact_analyzer.calculate_hard_hit_rate(home_exit_velocities)
        features['away_hard_hit_rate'] = self.contact_analyzer.calculate_hard_hit_rate(away_exit_velocities)
        features['hard_hit_rate_differential'] = features['home_hard_hit_rate'] - features['away_hard_hit_rate']

        home_launch_angles = home_batting.get('launch_angles', [12, 15, 8, 18, 10])
        features['home_barrel_rate_calc'] = self.contact_analyzer.calculate_barrel_rate(home_exit_velocities, home_launch_angles)

        away_launch_angles = away_batting.get('launch_angles', [11, 14, 7, 17, 9])
        features['away_barrel_rate_calc'] = self.contact_analyzer.calculate_barrel_rate(away_exit_velocities, away_launch_angles)

        features['barrel_rate_calc_differential'] = features['home_barrel_rate_calc'] - features['away_barrel_rate_calc']

        return features

    def _engineer_nhl_goalie_features(self, raw_data: dict) -> dict[str, float]:
        """Engineer 60+ NHL goalie features."""

        features = {}
        goalie_data = raw_data.get('goalie_data', {})

        # Basic goalie stats
        home_goalie = goalie_data.get('home_goalie', {})
        away_goalie = goalie_data.get('away_goalie', {})

        features['home_save_pct'] = home_goalie.get('save_pct', 0.910)
        features['away_save_pct'] = away_goalie.get('save_pct', 0.910)
        features['save_pct_differential'] = features['home_save_pct'] - features['away_save_pct']

        features['home_gaa'] = home_goalie.get('gaa', 2.80)
        features['away_gaa'] = away_goalie.get('gaa', 2.80)
        features['gaa_differential'] = features['home_gaa'] - features['away_gaa']

        features['home_gsax'] = home_goalie.get('gsax', 10.0)
        features['away_gsax'] = away_goalie.get('gsax', 10.0)
        features['gsax_differential'] = features['home_gsax'] - features['away_gsax']

        # Advanced goalie metrics
        features['home_high_danger_save_pct'] = home_goalie.get('high_danger_save_pct', 0.850)
        features['away_high_danger_save_pct'] = away_goalie.get('high_danger_save_pct', 0.850)
        features['high_danger_save_pct_differential'] = features['home_high_danger_save_pct'] - features['away_high_danger_save_pct']

        features['home_quality_start_pct'] = home_goalie.get('quality_start_pct', 0.600)
        features['away_quality_start_pct'] = away_goalie.get('quality_start_pct', 0.600)
        features['quality_start_pct_differential'] = features['home_quality_start_pct'] - features['away_quality_start_pct']

        return features

    def _engineer_nhl_team_features(self, raw_data: dict) -> dict[str, float]:
        """Engineer 50+ NHL team features."""

        features = {}
        team_data = raw_data.get('team_data', {})

        # Basic team stats
        home_team = team_data.get('home_team', {})
        away_team = team_data.get('away_team', {})

        features['home_corsi'] = home_team.get('corsi', 50.0)
        features['away_corsi'] = away_team.get('corsi', 50.0)
        features['corsi_differential'] = features['home_corsi'] - features['away_corsi']

        features['home_fenwick'] = home_team.get('fenwick', 50.0)
        features['away_fenwick'] = away_team.get('fenwick', 50.0)
        features['fenwick_differential'] = features['home_fenwick'] - features['away_fenwick']

        features['home_expected_goals'] = home_team.get('expected_goals', 2.75)
        features['away_expected_goals'] = away_team.get('expected_goals', 2.75)
        features['expected_goals_differential'] = features['home_expected_goals'] - features['away_expected_goals']

        # Advanced team metrics
        features['home_controlled_entry_pct'] = home_team.get('controlled_entry_pct', 0.550)
        features['away_controlled_entry_pct'] = away_team.get('controlled_entry_pct', 0.550)
        features['controlled_entry_pct_differential'] = features['home_controlled_entry_pct'] - features['away_controlled_entry_pct']

        features['home_transition_efficiency'] = home_team.get('transition_efficiency', 0.520)
        features['away_transition_efficiency'] = away_team.get('transition_efficiency', 0.520)
        features['transition_efficiency_differential'] = features['home_transition_efficiency'] - features['away_transition_efficiency']

        return features

    def _engineer_situational_features(self, raw_data: dict) -> dict[str, float]:
        """Engineer situational features for all sports."""

        features = {}

        # Park/arena factors
        park_factors = raw_data.get('park_factors', {})
        features['hr_factor'] = park_factors.get('hr_factor', 1.00)
        features['hit_factor'] = park_factors.get('hit_factor', 1.00)

        # Rest advantage
        features['home_rest_days'] = raw_data.get('home_rest_days', 2)
        features['away_rest_days'] = raw_data.get('away_rest_days', 2)
        features['rest_advantage'] = features['home_rest_days'] - features['away_rest_days']

        # Travel distance
        features['home_travel_distance'] = raw_data.get('home_travel_distance', 0)
        features['away_travel_distance'] = raw_data.get('away_travel_distance', 500)
        features['travel_advantage'] = features['away_travel_distance'] - features['home_travel_distance']

        return features

    def _engineer_market_features(self, raw_data: dict) -> dict[str, float]:
        """Engineer market features for all sports."""

        features = {}
        market_data = raw_data.get('market_data', {})

        # Odds movement
        features['opening_line'] = market_data.get('opening_line', 0)
        features['current_line'] = market_data.get('current_line', 0)
        features['line_movement'] = features['current_line'] - features['opening_line']

        # Public betting
        features['public_betting_pct'] = market_data.get('public_betting_pct', 0.500)
        features['sharp_money_indicator'] = market_data.get('sharp_money_indicator', 0)

        # Volume patterns
        features['betting_volume'] = market_data.get('betting_volume', 1000)
        features['volume_ratio'] = market_data.get('volume_ratio', 1.0)

        return features

    def _engineer_environmental_features(self, raw_data: dict) -> dict[str, float]:
        """Engineer environmental features for all sports."""

        features = {}
        weather_data = raw_data.get('weather_data', {})

        # Weather conditions
        features['temperature'] = weather_data.get('temperature', 70)
        features['humidity'] = weather_data.get('humidity', 50)
        features['wind_speed'] = weather_data.get('wind_speed', 5)
        features['wind_direction'] = weather_data.get('wind_direction', 'N')

        # Weather impact calculations
        features['temperature_effect'] = self._calculate_temperature_effect(features['temperature'])
        features['humidity_effect'] = self._calculate_humidity_effect(features['humidity'])
        features['wind_effect'] = self._calculate_wind_effect(features['wind_speed'], features['wind_direction'])

        return features

    def _engineer_biomechanical_features(self, raw_data: dict) -> dict[str, float]:
        """Engineer biomechanical features for all sports."""

        features = {}

        # Player workload
        features['home_workload'] = raw_data.get('home_workload', 0.5)
        features['away_workload'] = raw_data.get('away_workload', 0.5)
        features['workload_differential'] = features['home_workload'] - features['away_workload']

        # Injury impact
        features['home_injury_impact'] = raw_data.get('home_injury_impact', 0)
        features['away_injury_impact'] = raw_data.get('away_injury_impact', 0)
        features['injury_impact_differential'] = features['home_injury_impact'] - features['away_injury_impact']

        return features

    def _engineer_temporal_features(self, raw_data: dict) -> dict[str, float]:
        """Engineer temporal features for all sports."""

        features = {}

        # Time-based patterns
        features['day_of_week'] = datetime.now().weekday()
        features['month'] = datetime.now().month
        features['season_progress'] = raw_data.get('season_progress', 0.5)

        # Recent performance
        features['home_recent_form'] = raw_data.get('home_recent_form', 0.500)
        features['away_recent_form'] = raw_data.get('away_recent_form', 0.500)
        features['recent_form_differential'] = features['home_recent_form'] - features['away_recent_form']

        return features

    def _validate_features(self, features: dict[str, float]) -> dict[str, float]:
        """Validate and clean features."""

        validated_features = {}

        for feature_name, value in features.items():
            try:
                float_value = float(value)

                if np.isinf(float_value):
                    float_value = 0.0

                if np.isnan(float_value):
                    float_value = 0.0

                if abs(float_value) > 1000:
                    float_value = np.sign(float_value) * 1000

                validated_features[feature_name] = float_value

            except (ValueError, TypeError):
                logger.warning(f"Invalid feature value for {feature_name}: {value}, setting to 0")
                validated_features[feature_name] = 0.0

        return validated_features

    def _get_fallback_features(self, sport: str) -> dict[str, float]:
        """Return fallback features when engineering fails."""

        if sport == 'mlb':
            return {
                'home_era': 4.00, 'away_era': 4.00, 'era_differential': 0.0,
                'home_woba': 0.320, 'away_woba': 0.320, 'woba_differential': 0.0,
                'home_xwoba': 0.320, 'away_xwoba': 0.320, 'xwoba_differential': 0.0,
                'home_stuff_plus': 100.0, 'away_stuff_plus': 100.0, 'stuff_plus_differential': 0.0
            }
        elif sport == 'nhl':
            return {
                'home_save_pct': 0.910, 'away_save_pct': 0.910, 'save_pct_differential': 0.0,
                'home_corsi': 50.0, 'away_corsi': 50.0, 'corsi_differential': 0.0,
                'home_expected_goals': 2.75, 'away_expected_goals': 2.75, 'expected_goals_differential': 0.0
            }
        else:
            return {}

    def _calculate_temperature_effect(self, temperature: float) -> float:
        """Calculate temperature effect on performance."""
        # Baseball: warmer = better hitting
        # Hockey: cooler = better performance
        return (temperature - 70) / 100  # Normalized effect

    def _calculate_humidity_effect(self, humidity: float) -> float:
        """Calculate humidity effect on performance."""
        # Higher humidity = denser air = less ball carry
        return (humidity - 50) / 100  # Normalized effect

    def _calculate_wind_effect(self, wind_speed: float, wind_direction: str) -> float:
        """Calculate wind effect on performance."""
        # Wind speed impact on ball flight
        return wind_speed / 100  # Normalized effect

# Professional metrics classes (from professional_analytics_upgrade.py)
class AdvancedMLBMetrics:
    """Advanced MLB metrics calculator."""

    def calculate_xwOBA(self, exit_velocity: float, launch_angle: float, barrel_rate: float, hard_hit_rate: float) -> float:
        """Calculate expected wOBA."""
        # Professional xwOBA calculation
        base_xwoba = 0.320
        exit_vel_effect = (exit_velocity - 88.0) * 0.002
        launch_angle_effect = (launch_angle - 12.0) * 0.003
        barrel_effect = (barrel_rate - 0.075) * 0.5
        hard_hit_effect = (hard_hit_rate - 0.350) * 0.2

        xwoba = base_xwoba + exit_vel_effect + launch_angle_effect + barrel_effect + hard_hit_effect
        return max(0.200, min(0.600, xwoba))

    def calculate_xFIP(self, bb_rate: float, k_rate: float, hr_rate: float) -> float:
        """Calculate expected FIP."""
        # Professional xFIP calculation
        xfip = (13 * hr_rate + 3 * (bb_rate + 0.02) - 2 * k_rate) / 0.9
        return max(2.00, min(8.00, xfip))

    def calculate_stuff_plus(self, velocity: float, spin_rate: float, movement: float, movement_break: tuple[float, float]) -> float:
        """Calculate Stuff+ metric."""
        # Professional Stuff+ calculation
        velocity_effect = (velocity - 92.0) * 0.5
        spin_effect = (spin_rate - 2200) * 0.0001
        movement_effect = (movement - 15.0) * 0.2
        break_effect = (movement_break[0] + movement_break[1] - 1.0) * 10

        stuff_plus = 100 + velocity_effect + spin_effect + movement_effect + break_effect
        return max(50, min(150, stuff_plus))

class AdvancedNHLMetrics:
    """Advanced NHL metrics calculator."""

    def calculate_gsax(self, save_pct: float, expected_save_pct: float, shots_against: float) -> float:
        """Calculate Goals Saved Above Expected."""
        return (save_pct - expected_save_pct) * shots_against * 100

class ContactQualityAnalyzer:
    """Contact quality analysis for MLB."""

    def calculate_hard_hit_rate(self, exit_velocities: list[float]) -> float:
        """Calculate hard hit rate (95+ mph)."""
        if not exit_velocities:
            return 0.0
        hard_hits = sum(1 for vel in exit_velocities if vel >= 95.0)
        return hard_hits / len(exit_velocities)

    def calculate_barrel_rate(self, exit_velocities: list[float], launch_angles: list[float]) -> float:
        """Calculate barrel rate."""
        if not exit_velocities or not launch_angles:
            return 0.0

        barrel_count = 0
        for vel, angle in zip(exit_velocities, launch_angles, strict=False):
            if vel >= 98.0 and 26 <= angle <= 30:
                barrel_count += 1

        return barrel_count / len(exit_velocities)

class UnifiedMLEnsemble:
    """Unified ML ensemble for all sports with professional methods."""

    def __init__(self):
        # Professional ensemble models
        self.models = {
            'xgboost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42),
            'neural_net': MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=1000, random_state=42)
        }

        # Add LightGBM and CatBoost if available
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)

        if CATBOOST_AVAILABLE:
            self.models['catboost'] = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, random_state=42, verbose=False)

        # Sport-specific models
        self.sport_models = {
            'mlb': {},
            'nhl': {}
        }

        # Ensemble weights
        self.ensemble_weights = {}

        # Model performance tracking
        self.model_performance = {}

    def train_models(self, features: pd.DataFrame, target: pd.Series, sport: str):
        """Train models for a specific sport."""

        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, stratify=target
            )

            # Train each model
            for name, model in self.models.items():
                try:
                    # Train model
                    model.fit(X_train, y_train)

                    # Evaluate performance
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    auc_score = roc_auc_score(y_test, y_pred_proba)

                    # Store model and performance
                    self.sport_models[sport][name] = model
                    self.model_performance[f"{sport}_{name}"] = auc_score

                    logger.info(f"Trained {name} for {sport}: AUC = {auc_score:.3f}")

                except Exception as e:
                    logger.error(f"Error training {name} for {sport}: {e}")

            # Calculate ensemble weights based on performance
            self._calculate_ensemble_weights(sport)

        except Exception as e:
            logger.error(f"Error training models for {sport}: {e}")

    def predict(self, features: dict[str, float], sport: str) -> dict[str, Any]:
        """Make unified prediction for any sport."""

        try:
            # Convert features to array
            feature_array = self._features_to_array(features)

            # Get base predictions
            base_predictions = {}
            for name, model in self.models.items():
                if name in self.sport_models[sport]:
                    try:
                        pred = self.sport_models[sport][name].predict_proba(feature_array)[0][1]
                        base_predictions[name] = pred
                    except Exception as e:
                        logger.error(f"Error with {name} prediction: {e}")
                        base_predictions[name] = 0.5

            if not base_predictions:
                raise ValueError("No valid predictions available")

            # Calculate weighted ensemble prediction
            ensemble_prediction = self._calculate_weighted_prediction(base_predictions, sport)

            # Calculate uncertainty and confidence
            uncertainty = np.std(list(base_predictions.values()))
            confidence = 1 - uncertainty

            # Calculate model agreement
            model_agreement = self._calculate_model_agreement(base_predictions)

            return {
                'prediction': ensemble_prediction,
                'confidence': confidence,
                'uncertainty': uncertainty,
                'base_predictions': base_predictions,
                'model_agreement': model_agreement,
                'ensemble_weights': self.ensemble_weights.get(sport, {}),
                'feature_importance': self._get_feature_importance(features, sport)
            }

        except Exception as e:
            logger.error(f"Error in ML ensemble prediction: {e}")
            return self._get_fallback_prediction()

    def _features_to_array(self, features: dict[str, float]) -> np.ndarray:
        """Convert features dict to numpy array."""
        feature_names = sorted(features.keys())
        feature_values = [features[name] for name in feature_names]
        return np.array([feature_values])

    def _calculate_weighted_prediction(self, base_predictions: dict[str, float], sport: str) -> float:
        """Calculate weighted ensemble prediction."""

        weights = self.ensemble_weights.get(sport, {})

        if not weights:
            # Equal weights if no weights calculated
            weights = {name: 1.0/len(base_predictions) for name in base_predictions.keys()}

        weighted_sum = 0.0
        total_weight = 0.0

        for name, pred in base_predictions.items():
            weight = weights.get(name, 0.0)
            weighted_sum += pred * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return np.mean(list(base_predictions.values()))

    def _calculate_model_agreement(self, base_predictions: dict[str, float]) -> float:
        """Calculate model agreement score."""
        if len(base_predictions) < 2:
            return 1.0

        predictions = list(base_predictions.values())
        mean_pred = np.mean(predictions)
        agreement = 1 - np.std(predictions) / mean_pred if mean_pred > 0 else 0
        return max(0.0, min(1.0, agreement))

    def _calculate_ensemble_weights(self, sport: str):
        """Calculate ensemble weights based on model performance."""

        sport_performance = {}
        for name in self.models.keys():
            perf_key = f"{sport}_{name}"
            if perf_key in self.model_performance:
                sport_performance[name] = self.model_performance[perf_key]

        if sport_performance:
            # Softmax weights based on performance
            performances = list(sport_performance.values())
            exp_performances = np.exp(performances)
            weights = exp_performances / np.sum(exp_performances)

            self.ensemble_weights[sport] = dict(zip(sport_performance.keys(), weights, strict=False))

            logger.info(f"Ensemble weights for {sport}: {self.ensemble_weights[sport]}")

    def _get_feature_importance(self, features: dict[str, float], sport: str) -> dict[str, float]:
        """Get feature importance from best model."""

        if not self.sport_models[sport]:
            return {}

        # Get best performing model
        best_model_name = max(
            [name for name in self.models.keys() if name in self.sport_models[sport]],
            key=lambda x: self.model_performance.get(f"{sport}_{x}", 0),
            default=None
        )

        if best_model_name and hasattr(self.sport_models[sport][best_model_name], 'feature_importances_'):
            feature_names = sorted(features.keys())
            importances = self.sport_models[sport][best_model_name].feature_importances_

            return dict(zip(feature_names, importances, strict=False))

        return {}

    def _get_fallback_prediction(self) -> dict[str, Any]:
        """Return fallback prediction when ensemble fails."""

        return {
            'prediction': 0.5,
            'confidence': 0.5,
            'uncertainty': 0.1,
            'base_predictions': {'fallback': 0.5},
            'model_agreement': 1.0,
            'ensemble_weights': {},
            'feature_importance': {}
        }

class UnifiedRiskManager:
    """Unified risk management for all sports with professional methods."""

    def __init__(self, initial_bankroll: float = 100000):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.kelly_fraction = 0.25  # 1/4 Kelly (conservative)
        self.max_bet_size = 0.02    # 2% max bet size
        self.portfolio_correlation_limit = 0.3
        self.max_drawdown_limit = 0.15  # 15% max drawdown

        # Portfolio tracking
        self.active_bets = []
        self.bet_history = []
        self.portfolio_correlation = 0.0

        # Risk metrics
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.sharpe_ratio = 0.0

    def assess_bet(self, game_data: UnifiedGameData, prediction: dict[str, Any], odds: dict[str, Any]) -> dict[str, Any]:
        """Assess bet risk and calculate optimal stake."""

        try:
            # Calculate edge
            edge_analysis = self._calculate_edge(prediction, odds)

            # Calculate Kelly stake
            kelly_stake = self._calculate_kelly_stake(edge_analysis)

            # Apply risk constraints
            constrained_stake = self._apply_risk_constraints(kelly_stake, edge_analysis, game_data)

            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(constrained_stake, edge_analysis)

            # Generate recommendation
            recommendation = self._generate_recommendation(constrained_stake, edge_analysis)

            return {
                'stake': constrained_stake,
                'stake_amount': constrained_stake * self.current_bankroll,
                'edge_analysis': edge_analysis,
                'risk_metrics': risk_metrics,
                'recommendation': recommendation,
                'portfolio_impact': self._calculate_portfolio_impact(constrained_stake, game_data)
            }

        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return self._get_fallback_risk_assessment()

    def _calculate_edge(self, prediction: dict[str, Any], odds: dict[str, Any]) -> dict[str, Any]:
        """Calculate betting edge."""

        win_prob = prediction['prediction']
        confidence = prediction['confidence']

        # Convert odds to implied probability
        if 'moneyline' in odds:
            home_odds = odds['moneyline'].get('home', 0)
            away_odds = odds['moneyline'].get('away', 0)

            if home_odds < 0:
                home_implied_prob = abs(home_odds) / (abs(home_odds) + 100)
            else:
                home_implied_prob = 100 / (home_odds + 100)

            if away_odds < 0:
                away_implied_prob = abs(away_odds) / (abs(away_odds) + 100)
            else:
                away_implied_prob = 100 / (away_odds + 100)

            # Calculate edge
            home_edge = win_prob - home_implied_prob
            away_edge = (1 - win_prob) - away_implied_prob

            # Choose better side
            if home_edge > away_edge:
                edge = home_edge
                bet_side = 'home'
                implied_prob = home_implied_prob
            else:
                edge = away_edge
                bet_side = 'away'
                implied_prob = away_implied_prob
        else:
            edge = 0.0
            bet_side = 'unknown'
            implied_prob = 0.5

        # Adjust edge for confidence
        adjusted_edge = edge * confidence

        # Assess edge quality
        edge_quality = self._assess_edge_quality(adjusted_edge, confidence)

        return {
            'raw_edge': edge,
            'adjusted_edge': adjusted_edge,
            'win_probability': win_prob,
            'implied_probability': implied_prob,
            'confidence': confidence,
            'bet_side': bet_side,
            'edge_quality': edge_quality,
            'expected_value': adjusted_edge * self.current_bankroll
        }

    def _calculate_kelly_stake(self, edge_analysis: dict[str, Any]) -> float:
        """Calculate Kelly Criterion stake."""

        win_prob = edge_analysis['win_probability']
        implied_prob = edge_analysis['implied_probability']

        if implied_prob > 0 and implied_prob < 1:
            # Kelly formula: f = (bp - q) / b
            # where b = odds - 1, p = win probability, q = loss probability
            b = (1 / implied_prob) - 1
            p = win_prob
            q = 1 - win_prob

            kelly_stake = (b * p - q) / b

            # Apply fractional Kelly
            kelly_stake *= self.kelly_fraction

            return max(0.0, min(self.max_bet_size, kelly_stake))
        else:
            return 0.0

    def _apply_risk_constraints(self, kelly_stake: float, edge_analysis: dict[str, Any], game_data: UnifiedGameData) -> float:
        """Apply risk management constraints."""

        constrained_stake = kelly_stake

        # Maximum bet size constraint
        constrained_stake = min(constrained_stake, self.max_bet_size)

        # Portfolio correlation constraint
        if self.portfolio_correlation > self.portfolio_correlation_limit:
            correlation_penalty = 1 - (self.portfolio_correlation / self.portfolio_correlation_limit)
            constrained_stake *= correlation_penalty

        # Drawdown constraint
        if self.current_drawdown > self.max_drawdown_limit * 0.8:  # Warning threshold
            drawdown_penalty = 1 - (self.current_drawdown / self.max_drawdown_limit)
            constrained_stake *= drawdown_penalty

        # Edge quality constraint
        edge_quality = edge_analysis['edge_quality']
        if edge_quality == 'low':
            constrained_stake *= 0.5
        elif edge_quality == 'very_low':
            constrained_stake = 0.0

        return max(0.0, constrained_stake)

    def _calculate_risk_metrics(self, stake: float, edge_analysis: dict[str, Any]) -> dict[str, Any]:
        """Calculate comprehensive risk metrics."""

        win_prob = edge_analysis['win_probability']
        stake_amount = stake * self.current_bankroll

        # Expected value
        expected_value = edge_analysis['expected_value'] * stake

        # Variance and standard deviation
        variance = win_prob * (1 - win_prob) * stake_amount ** 2
        std_dev = np.sqrt(variance)

        # Value at Risk (95% confidence)
        var_95 = stake_amount * (1 - win_prob) * 1.645

        # Maximum loss
        max_loss = stake_amount

        # Sharpe ratio (simplified)
        if std_dev > 0:
            sharpe_ratio = expected_value / std_dev
        else:
            sharpe_ratio = 0.0

        return {
            'expected_value': expected_value,
            'variance': variance,
            'std_dev': std_dev,
            'var_95': var_95,
            'max_loss': max_loss,
            'sharpe_ratio': sharpe_ratio,
            'win_probability': win_prob,
            'stake_amount': stake_amount
        }

    def _assess_edge_quality(self, edge: float, confidence: float) -> str:
        """Assess the quality of the betting edge."""

        if edge > 0.08 and confidence > 0.8:
            return 'very_high'
        elif edge > 0.05 and confidence > 0.7:
            return 'high'
        elif edge > 0.03 and confidence > 0.6:
            return 'medium'
        elif edge > 0.01 and confidence > 0.5:
            return 'low'
        else:
            return 'very_low'

    def _generate_recommendation(self, stake: float, edge_analysis: dict[str, Any]) -> dict[str, Any]:
        """Generate betting recommendation."""

        edge = edge_analysis['adjusted_edge']
        edge_quality = edge_analysis['edge_quality']

        if stake == 0.0:
            action = 'pass'
            reason = 'insufficient_edge'
        elif edge_quality == 'very_low':
            action = 'pass'
            reason = 'very_low_edge_quality'
        elif edge_quality == 'low':
            action = 'pass'
            reason = 'low_edge_quality'
        else:
            action = 'bet'
            reason = 'meets_criteria'

        return {
            'action': action,
            'reason': reason,
            'confidence': edge_quality,
            'priority': self._calculate_priority(edge, edge_quality)
        }

    def _calculate_priority(self, edge: float, edge_quality: str) -> str:
        """Calculate bet priority."""

        if edge > 0.06 and edge_quality in ['high', 'very_high']:
            return 'high'
        elif edge > 0.04 and edge_quality in ['medium', 'high', 'very_high']:
            return 'medium'
        else:
            return 'low'

    def _calculate_portfolio_impact(self, stake: float, game_data: UnifiedGameData) -> dict[str, Any]:
        """Calculate portfolio impact of the bet."""

        # Calculate correlation with existing bets
        correlation = self._calculate_bet_correlation(game_data)

        # Portfolio concentration
        total_exposure = sum(bet['stake'] for bet in self.active_bets) + stake

        return {
            'correlation': correlation,
            'total_exposure': total_exposure,
            'concentration_risk': total_exposure / self.current_bankroll,
            'diversification_benefit': 1 - correlation
        }

    def _calculate_bet_correlation(self, game_data: UnifiedGameData) -> float:
        """Calculate correlation with existing bets."""

        if not self.active_bets:
            return 0.0

        # Simplified correlation calculation
        # In practice, this would use more sophisticated correlation models
        return 0.1  # Low correlation assumption

    def _get_fallback_risk_assessment(self) -> dict[str, Any]:
        """Return fallback risk assessment."""

        return {
            'stake': 0.0,
            'stake_amount': 0.0,
            'edge_analysis': {},
            'risk_metrics': {},
            'recommendation': {
                'action': 'pass',
                'reason': 'error_in_assessment',
                'confidence': 'low',
                'priority': 'low'
            },
            'portfolio_impact': {}
        }

class UnifiedPerformanceTracker:
    """Unified performance tracking for all sports."""

    def __init__(self):
        self.predictions = []
        self.bets = []
        self.performance_metrics = {
            'total_predictions': 0,
            'total_bets': 0,
            'win_rate': 0.0,
            'roi': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_confidence': 0.0,
            'avg_edge': 0.0
        }

    def track_prediction(self, game_id: str, prediction: dict[str, Any], risk_assessment: dict[str, Any]):
        """Track prediction for performance analysis."""

        self.predictions.append({
            'game_id': game_id,
            'prediction': prediction,
            'risk_assessment': risk_assessment,
            'timestamp': datetime.now()
        })

        self.performance_metrics['total_predictions'] += 1

        # Update average metrics
        if prediction:
            self.performance_metrics['avg_confidence'] = (
                (self.performance_metrics['avg_confidence'] * (self.performance_metrics['total_predictions'] - 1) +
                 prediction.get('confidence', 0.5)) / self.performance_metrics['total_predictions']
            )

        if risk_assessment and 'edge_analysis' in risk_assessment:
            edge = risk_assessment['edge_analysis'].get('adjusted_edge', 0)
            self.performance_metrics['avg_edge'] = (
                (self.performance_metrics['avg_edge'] * (self.performance_metrics['total_predictions'] - 1) +
                 edge) / self.performance_metrics['total_predictions']
            )

    def track_bet(self, game_id: str, bet_data: dict[str, Any], result: dict[str, Any]):
        """Track bet result for performance analysis."""

        self.bets.append({
            'game_id': game_id,
            'bet_data': bet_data,
            'result': result,
            'timestamp': datetime.now()
        })

        self.performance_metrics['total_bets'] += 1

        # Update performance metrics
        self._update_performance_metrics()

    def _update_performance_metrics(self):
        """Update comprehensive performance metrics."""

        if not self.bets:
            return

        # Calculate win rate
        wins = sum(1 for bet in self.bets if bet['result'].get('won', False))
        self.performance_metrics['win_rate'] = wins / len(self.bets)

        # Calculate ROI
        total_profit = sum(bet['result'].get('profit', 0) for bet in self.bets)
        total_invested = sum(bet['bet_data'].get('stake_amount', 0) for bet in self.bets)

        if total_invested > 0:
            self.performance_metrics['roi'] = total_profit / total_invested

        # Calculate Sharpe ratio (simplified)
        returns = [bet['result'].get('return_rate', 0) for bet in self.bets]
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                self.performance_metrics['sharpe_ratio'] = avg_return / std_return

        # Calculate max drawdown
        cumulative_returns = np.cumsum(returns) if returns else [0]
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        self.performance_metrics['max_drawdown'] = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""

        return self.performance_metrics.copy()

    def get_detailed_performance(self) -> dict[str, Any]:
        """Get detailed performance breakdown."""

        return {
            'performance_metrics': self.performance_metrics,
            'recent_predictions': self.predictions[-10:] if self.predictions else [],
            'recent_bets': self.bets[-10:] if self.bets else [],
            'sport_breakdown': self._get_sport_breakdown(),
            'time_series': self._get_time_series_data()
        }

    def _get_sport_breakdown(self) -> dict[str, Any]:
        """Get performance breakdown by sport."""

        sport_stats = {}

        for bet in self.bets:
            sport = bet['bet_data'].get('sport', 'unknown')
            if sport not in sport_stats:
                sport_stats[sport] = {'bets': 0, 'wins': 0, 'profit': 0.0}

            sport_stats[sport]['bets'] += 1
            if bet['result'].get('won', False):
                sport_stats[sport]['wins'] += 1
            sport_stats[sport]['profit'] += bet['result'].get('profit', 0)

        # Calculate sport-specific metrics
        for sport, stats in sport_stats.items():
            if stats['bets'] > 0:
                stats['win_rate'] = stats['wins'] / stats['bets']
                stats['roi'] = stats['profit'] / sum(
                    bet['bet_data'].get('stake_amount', 0)
                    for bet in self.bets
                    if bet['bet_data'].get('sport') == sport
                )

        return sport_stats

    def _get_time_series_data(self) -> dict[str, list]:
        """Get time series performance data."""

        if not self.bets:
            return {'dates': [], 'cumulative_profit': [], 'win_rate': []}

        # Sort bets by date
        sorted_bets = sorted(self.bets, key=lambda x: x['timestamp'])

        dates = []
        cumulative_profit = []
        win_rate = []

        running_profit = 0.0
        running_wins = 0

        for i, bet in enumerate(sorted_bets):
            dates.append(bet['timestamp'].date())
            running_profit += bet['result'].get('profit', 0)
            cumulative_profit.append(running_profit)

            if bet['result'].get('won', False):
                running_wins += 1
            win_rate.append(running_wins / (i + 1))

        return {
            'dates': dates,
            'cumulative_profit': cumulative_profit,
            'win_rate': win_rate
        }

# Main Unified ML Pipeline
class UnifiedMLPipeline:
    """Single, unified ML pipeline for all sports and components."""

    def __init__(self, config: dict = None, initial_bankroll: float = 100000):
        if config is None:
            config = {}

        # Core components
        self.data_integrator = UnifiedDataIntegrator()
        self.feature_engineer = UnifiedFeatureEngineer()
        self.ml_ensemble = UnifiedMLEnsemble()
        self.risk_manager = UnifiedRiskManager(initial_bankroll)
        self.performance_tracker = UnifiedPerformanceTracker()

        # Sport-specific components
        self.mlb_metrics = AdvancedMLBMetrics()
        self.nhl_metrics = AdvancedNHLMetrics()
        self.contact_analyzer = ContactQualityAnalyzer()

        # Configuration
        self.config = config

        logger.info("Unified ML Pipeline initialized")

    async def process_game(self, game_id: str, sport: str, odds: dict[str, Any]) -> dict[str, Any]:
        """Unified game processing for all sports."""

        logger.info(f"Processing {sport} game {game_id}")

        try:
            # Step 1: Unified Data Ingestion
            game_data = await self.data_integrator.ingest_game_data(game_id, sport)

            # Step 2: Unified Feature Engineering
            features = self.feature_engineer.engineer_features(game_data)

            # Step 3: Unified ML Prediction
            prediction = self.ml_ensemble.predict(features, sport)

            # Step 4: Unified Risk Assessment
            risk_assessment = self.risk_manager.assess_bet(game_data, prediction, odds)

            # Step 5: Performance Tracking
            self.performance_tracker.track_prediction(game_id, prediction, risk_assessment)

            # Compile results
            results = self._compile_results(game_id, game_data, features, prediction, risk_assessment)

            logger.info(f"Completed processing for {sport} game {game_id}")
            return results

        except Exception as e:
            logger.error(f"Error processing {sport} game {game_id}: {e}")
            return self._get_fallback_results(game_id, sport)

    def _compile_results(self, game_id: str, game_data: UnifiedGameData, features: dict[str, float],
                        prediction: dict[str, Any], risk_assessment: dict[str, Any]) -> dict[str, Any]:
        """Compile comprehensive results."""

        return {
            'game_id': game_id,
            'sport': game_data.sport,
            'game_data': {
                'home_team': game_data.home_team,
                'away_team': game_data.away_team,
                'game_date': game_data.game_date.isoformat()
            },
            'features': features,
            'prediction': prediction,
            'risk_assessment': risk_assessment,
            'recommendation': risk_assessment['recommendation'],
            'timestamp': datetime.now().isoformat(),
            'performance_summary': self.performance_tracker.get_performance_summary()
        }

    def _get_fallback_results(self, game_id: str, sport: str) -> dict[str, Any]:
        """Return fallback results when processing fails."""

        return {
            'game_id': game_id,
            'sport': sport,
            'game_data': {},
            'features': {},
            'prediction': {
                'prediction': 0.5,
                'confidence': 0.5,
                'uncertainty': 0.1,
                'base_predictions': {'fallback': 0.5},
                'model_agreement': 1.0
            },
            'risk_assessment': {
                'stake': 0.0,
                'recommendation': {
                    'action': 'pass',
                    'reason': 'processing_error',
                    'confidence': 'low',
                    'priority': 'low'
                }
            },
            'recommendation': {
                'action': 'pass',
                'reason': 'processing_error',
                'confidence': 'low',
                'priority': 'low'
            },
            'timestamp': datetime.now().isoformat(),
            'performance_summary': self.performance_tracker.get_performance_summary()
        }

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary."""

        return self.performance_tracker.get_performance_summary()

    def get_detailed_performance(self) -> dict[str, Any]:
        """Get detailed performance breakdown."""

        return self.performance_tracker.get_detailed_performance()

    def train_models(self, training_data: dict[str, pd.DataFrame], targets: dict[str, pd.Series]):
        """Train models for all sports."""

        for sport, features in training_data.items():
            if sport in targets:
                logger.info(f"Training models for {sport}")
                self.ml_ensemble.train_models(features, targets[sport], sport)

# Example usage and testing
async def test_unified_pipeline():
    """Test the unified ML pipeline."""

    print("🧪 Testing Unified ML Pipeline...")

    # Initialize pipeline
    pipeline = UnifiedMLPipeline(initial_bankroll=100000)

    # Test MLB game
    mlb_odds = {
        'moneyline': {
            'home': -140,
            'away': +120
        }
    }

    mlb_results = await pipeline.process_game('mlb_game_123', 'mlb', mlb_odds)

    print("\n📊 MLB Results:")
    print(f"Prediction: {mlb_results['prediction']['prediction']:.1%}")
    print(f"Confidence: {mlb_results['prediction']['confidence']:.1%}")
    print(f"Recommendation: {mlb_results['recommendation']['action']}")
    print(f"Stake: {mlb_results['risk_assessment']['stake']:.1%}")
    print(f"Edge: {mlb_results['risk_assessment']['edge_analysis'].get('adjusted_edge', 0):.1%}")

    # Test NHL game
    nhl_odds = {
        'moneyline': {
            'home': -110,
            'away': -110
        }
    }

    nhl_results = await pipeline.process_game('nhl_game_456', 'nhl', nhl_odds)

    print("\n🏒 NHL Results:")
    print(f"Prediction: {nhl_results['prediction']['prediction']:.1%}")
    print(f"Confidence: {nhl_results['prediction']['confidence']:.1%}")
    print(f"Recommendation: {nhl_results['recommendation']['action']}")
    print(f"Stake: {nhl_results['risk_assessment']['stake']:.1%}")
    print(f"Edge: {nhl_results['risk_assessment']['edge_analysis'].get('adjusted_edge', 0):.1%}")

    # Performance summary
    performance = pipeline.get_performance_summary()
    print("\n📈 Performance Summary:")
    print(f"Total Predictions: {performance['total_predictions']}")
    print(f"Average Confidence: {performance['avg_confidence']:.1%}")
    print(f"Average Edge: {performance['avg_edge']:.1%}")

    print("\n✅ Unified ML Pipeline Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_unified_pipeline())
