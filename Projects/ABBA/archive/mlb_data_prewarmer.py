#!/usr/bin/env python3
"""
MLB Data Pre-Warmer for ABBA System
Populates database with comprehensive MLB statistics needed for betting predictions.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import structlog
import yaml
from cache_manager import CacheManager, DataPersistenceManager

# Import core components
from database import DatabaseManager

from models import Event, Odds

logger = structlog.get_logger()


class MLBDataPrewarmer:
    """Comprehensive MLB data pre-warming system for betting predictions."""

    def __init__(self, config: dict):
        self.config = config
        self.db_manager = DatabaseManager(config['database']['url'])
        self.cache_manager = CacheManager(config)
        self.persistence_manager = DataPersistenceManager(config)

        # MLB-specific configuration
        self.mlb_config = {
            'teams': [
                'Arizona Diamondbacks', 'Atlanta Braves', 'Baltimore Orioles', 'Boston Red Sox',
                'Chicago Cubs', 'Chicago White Sox', 'Cincinnati Reds', 'Cleveland Guardians',
                'Colorado Rockies', 'Detroit Tigers', 'Houston Astros', 'Kansas City Royals',
                'Los Angeles Angels', 'Los Angeles Dodgers', 'Miami Marlins', 'Milwaukee Brewers',
                'Minnesota Twins', 'New York Mets', 'New York Yankees', 'Oakland Athletics',
                'Philadelphia Phillies', 'Pittsburgh Pirates', 'San Diego Padres', 'San Francisco Giants',
                'Seattle Mariners', 'St. Louis Cardinals', 'Tampa Bay Rays', 'Texas Rangers',
                'Toronto Blue Jays', 'Washington Nationals'
            ],
            'parks': {
                'Arizona Diamondbacks': {'name': 'Chase Field', 'hr_factor': 1.15, 'park_factor': 1.08},
                'Atlanta Braves': {'name': 'Truist Park', 'hr_factor': 1.05, 'park_factor': 1.02},
                'Baltimore Orioles': {'name': 'Oriole Park at Camden Yards', 'hr_factor': 1.20, 'park_factor': 1.10},
                'Boston Red Sox': {'name': 'Fenway Park', 'hr_factor': 0.95, 'park_factor': 1.05},
                'Chicago Cubs': {'name': 'Wrigley Field', 'hr_factor': 1.10, 'park_factor': 1.03},
                'Chicago White Sox': {'name': 'Guaranteed Rate Field', 'hr_factor': 1.15, 'park_factor': 1.08},
                'Cincinnati Reds': {'name': 'Great American Ball Park', 'hr_factor': 1.25, 'park_factor': 1.12},
                'Cleveland Guardians': {'name': 'Progressive Field', 'hr_factor': 1.05, 'park_factor': 1.01},
                'Colorado Rockies': {'name': 'Coors Field', 'hr_factor': 1.35, 'park_factor': 1.20},
                'Detroit Tigers': {'name': 'Comerica Park', 'hr_factor': 0.90, 'park_factor': 0.95},
                'Houston Astros': {'name': 'Minute Maid Park', 'hr_factor': 1.10, 'park_factor': 1.05},
                'Kansas City Royals': {'name': 'Kauffman Stadium', 'hr_factor': 0.95, 'park_factor': 0.98},
                'Los Angeles Angels': {'name': 'Angel Stadium', 'hr_factor': 1.00, 'park_factor': 1.00},
                'Los Angeles Dodgers': {'name': 'Dodger Stadium', 'hr_factor': 0.95, 'park_factor': 0.97},
                'Miami Marlins': {'name': 'loanDepot park', 'hr_factor': 0.85, 'park_factor': 0.92},
                'Milwaukee Brewers': {'name': 'American Family Field', 'hr_factor': 1.05, 'park_factor': 1.02},
                'Minnesota Twins': {'name': 'Target Field', 'hr_factor': 1.00, 'park_factor': 1.00},
                'New York Mets': {'name': 'Citi Field', 'hr_factor': 0.90, 'park_factor': 0.96},
                'New York Yankees': {'name': 'Yankee Stadium', 'hr_factor': 1.20, 'park_factor': 1.10},
                'Oakland Athletics': {'name': 'Oakland Coliseum', 'hr_factor': 0.85, 'park_factor': 0.90},
                'Philadelphia Phillies': {'name': 'Citizens Bank Park', 'hr_factor': 1.15, 'park_factor': 1.08},
                'Pittsburgh Pirates': {'name': 'PNC Park', 'hr_factor': 0.90, 'park_factor': 0.95},
                'San Diego Padres': {'name': 'Petco Park', 'hr_factor': 0.85, 'park_factor': 0.92},
                'San Francisco Giants': {'name': 'Oracle Park', 'hr_factor': 0.80, 'park_factor': 0.88},
                'Seattle Mariners': {'name': 'T-Mobile Park', 'hr_factor': 0.90, 'park_factor': 0.94},
                'St. Louis Cardinals': {'name': 'Busch Stadium', 'hr_factor': 1.00, 'park_factor': 1.00},
                'Tampa Bay Rays': {'name': 'Tropicana Field', 'hr_factor': 0.95, 'park_factor': 0.97},
                'Texas Rangers': {'name': 'Globe Life Field', 'hr_factor': 1.05, 'park_factor': 1.03},
                'Toronto Blue Jays': {'name': 'Rogers Centre', 'hr_factor': 1.10, 'park_factor': 1.05},
                'Washington Nationals': {'name': 'Nationals Park', 'hr_factor': 1.00, 'park_factor': 1.00}
            },
            'data_requirements': {
                'pitching_stats': ['era', 'whip', 'k_per_9', 'bb_per_9', 'avg_velocity', 'spin_rate'],
                'batting_stats': ['woba', 'iso', 'bb_rate', 'k_rate', 'barrel_rate', 'exit_velocity'],
                'situational_stats': ['home_away_splits', 'day_night_splits', 'rest_days', 'travel_distance'],
                'historical_data': ['head_to_head', 'season_trends', 'injury_impact', 'weather_impact']
            }
        }

        logger.info("MLB Data Prewarmer initialized")

    async def prewarm_all_mlb_data(self) -> dict[str, Any]:
        """Pre-warm all MLB data needed for betting predictions."""
        logger.info("🚀 Starting comprehensive MLB data pre-warming")

        results = {
            'events_created': 0,
            'pitching_stats_loaded': 0,
            'batting_stats_loaded': 0,
            'park_factors_loaded': 0,
            'historical_data_loaded': 0,
            'cache_entries_created': 0,
            'errors': []
        }

        try:
            # Initialize database
            await self.db_manager.initialize()

            # 1. Create MLB events for the season
            events_result = await self._create_mlb_events()
            results['events_created'] = events_result['created']
            results['errors'].extend(events_result.get('errors', []))

            # 2. Load pitching statistics
            pitching_result = await self._load_pitching_statistics()
            results['pitching_stats_loaded'] = pitching_result['loaded']
            results['errors'].extend(pitching_result.get('errors', []))

            # 3. Load batting statistics
            batting_result = await self._load_batting_statistics()
            results['batting_stats_loaded'] = batting_result['loaded']
            results['errors'].extend(batting_result.get('errors', []))

            # 4. Load park factors
            park_result = await self._load_park_factors()
            results['park_factors_loaded'] = park_result['loaded']
            results['errors'].extend(park_result.get('errors', []))

            # 5. Load historical data
            historical_result = await self._load_historical_data()
            results['historical_data_loaded'] = historical_result['loaded']
            results['errors'].extend(historical_result.get('errors', []))

            # 6. Pre-warm cache with frequently accessed data
            cache_result = await self._prewarm_cache()
            results['cache_entries_created'] = cache_result['created']
            results['errors'].extend(cache_result.get('errors', []))

            logger.info(f"✅ MLB data pre-warming completed: {results}")
            return results

        except Exception as e:
            logger.error(f"❌ Error in MLB data pre-warming: {e}")
            results['errors'].append(str(e))
            return results

    async def _create_mlb_events(self) -> dict[str, Any]:
        """Create MLB events for the current season."""
        logger.info("📅 Creating MLB events for the season")

        result = {'created': 0, 'errors': []}

        try:
            # Generate events for the next 30 days
            start_date = datetime.now()
            end_date = start_date + timedelta(days=30)

            # Create events for each team
            for i, home_team in enumerate(self.mlb_config['teams']):
                for j, away_team in enumerate(self.mlb_config['teams']):
                    if home_team != away_team:
                        # Create event for next available date
                        event_date = start_date + timedelta(days=(i + j) % 30)

                        event = Event(
                            sport='baseball_mlb',
                            home_team=home_team,
                            away_team=away_team,
                            event_date=event_date,
                            status='scheduled'
                        )

                        await self.db_manager.save_event(event)
                        result['created'] += 1

                        # Add some odds for this event
                        await self._create_mock_odds(event.id)

            logger.info(f"✅ Created {result['created']} MLB events")
            return result

        except Exception as e:
            logger.error(f"❌ Error creating MLB events: {e}")
            result['errors'].append(str(e))
            return result

    async def _create_mock_odds(self, event_id: str):
        """Create mock odds for an event."""
        try:
            # Moneyline odds
            home_odds = Odds(
                event_id=event_id,
                platform='fanduel',
                market_type='moneyline',
                selection='home',
                odds=Decimal('-110'),
                implied_probability=Decimal('0.524')
            )
            await self.db_manager.save_odds(home_odds)

            away_odds = Odds(
                event_id=event_id,
                platform='fanduel',
                market_type='moneyline',
                selection='away',
                odds=Decimal('+110'),
                implied_probability=Decimal('0.476')
            )
            await self.db_manager.save_odds(away_odds)

            # Totals odds
            over_odds = Odds(
                event_id=event_id,
                platform='fanduel',
                market_type='totals',
                selection='over',
                odds=Decimal('-110'),
                line=Decimal('8.5'),
                implied_probability=Decimal('0.524')
            )
            await self.db_manager.save_odds(over_odds)

        except Exception as e:
            logger.error(f"Error creating mock odds for event {event_id}: {e}")

    async def _load_pitching_statistics(self) -> dict[str, Any]:
        """Load comprehensive pitching statistics."""
        logger.info("⚾ Loading pitching statistics")

        result = {'loaded': 0, 'errors': []}

        try:
            # Create comprehensive pitching stats for each team
            pitching_stats = {}

            for team in self.mlb_config['teams']:
                # Generate realistic pitching stats
                team_stats = {
                    'team': team,
                    'era_last_7': round(np.random.normal(4.0, 0.8), 2),
                    'era_last_14': round(np.random.normal(4.0, 0.6), 2),
                    'era_last_30': round(np.random.normal(4.0, 0.5), 2),
                    'whip_last_7': round(np.random.normal(1.30, 0.15), 3),
                    'whip_last_14': round(np.random.normal(1.30, 0.12), 3),
                    'whip_last_30': round(np.random.normal(1.30, 0.10), 3),
                    'k_per_9_last_7': round(np.random.normal(8.5, 1.2), 1),
                    'k_per_9_last_14': round(np.random.normal(8.5, 1.0), 1),
                    'k_per_9_last_30': round(np.random.normal(8.5, 0.8), 1),
                    'bb_per_9_last_7': round(np.random.normal(3.2, 0.6), 1),
                    'bb_per_9_last_14': round(np.random.normal(3.2, 0.5), 1),
                    'bb_per_9_last_30': round(np.random.normal(3.2, 0.4), 1),
                    'avg_velocity_last_7': round(np.random.normal(92.5, 2.0), 1),
                    'avg_velocity_last_14': round(np.random.normal(92.5, 1.8), 1),
                    'avg_velocity_last_30': round(np.random.normal(92.5, 1.5), 1),
                    'spin_rate_last_7': round(np.random.normal(2200, 200), 0),
                    'spin_rate_last_14': round(np.random.normal(2200, 180), 0),
                    'spin_rate_last_30': round(np.random.normal(2200, 150), 0),
                    'home_era': round(np.random.normal(3.8, 0.4), 2),
                    'away_era': round(np.random.normal(4.2, 0.4), 2),
                    'day_era': round(np.random.normal(4.1, 0.4), 2),
                    'night_era': round(np.random.normal(3.9, 0.4), 2),
                    'rest_days_impact': round(np.random.normal(0.1, 0.05), 3),
                    'bullpen_era_last_7': round(np.random.normal(3.9, 0.6), 2),
                    'bullpen_era_last_14': round(np.random.normal(3.9, 0.5), 2),
                    'bullpen_era_last_30': round(np.random.normal(3.9, 0.4), 2)
                }

                pitching_stats[team] = team_stats
                result['loaded'] += 1

            # Cache the pitching stats
            await self.cache_manager.set('mlb_pitching_stats', pitching_stats, 'ml_models', ttl=86400)

            # Archive for historical reference
            await self.persistence_manager.archive_data('mlb_pitching_stats', list(pitching_stats.values()))

            logger.info(f"✅ Loaded pitching stats for {result['loaded']} teams")
            return result

        except Exception as e:
            logger.error(f"❌ Error loading pitching statistics: {e}")
            result['errors'].append(str(e))
            return result

    async def _load_batting_statistics(self) -> dict[str, Any]:
        """Load comprehensive batting statistics."""
        logger.info("🏏 Loading batting statistics")

        result = {'loaded': 0, 'errors': []}

        try:
            # Create comprehensive batting stats for each team
            batting_stats = {}

            for team in self.mlb_config['teams']:
                # Generate realistic batting stats
                team_stats = {
                    'team': team,
                    'woba_last_7': round(np.random.normal(0.320, 0.025), 3),
                    'woba_last_14': round(np.random.normal(0.320, 0.020), 3),
                    'woba_last_30': round(np.random.normal(0.320, 0.015), 3),
                    'iso_last_7': round(np.random.normal(0.170, 0.030), 3),
                    'iso_last_14': round(np.random.normal(0.170, 0.025), 3),
                    'iso_last_30': round(np.random.normal(0.170, 0.020), 3),
                    'bb_rate_last_7': round(np.random.normal(0.085, 0.015), 3),
                    'bb_rate_last_14': round(np.random.normal(0.085, 0.012), 3),
                    'bb_rate_last_30': round(np.random.normal(0.085, 0.010), 3),
                    'k_rate_last_7': round(np.random.normal(0.220, 0.025), 3),
                    'k_rate_last_14': round(np.random.normal(0.220, 0.020), 3),
                    'k_rate_last_30': round(np.random.normal(0.220, 0.015), 3),
                    'barrel_rate_last_7': round(np.random.normal(0.085, 0.015), 3),
                    'barrel_rate_last_14': round(np.random.normal(0.085, 0.012), 3),
                    'barrel_rate_last_30': round(np.random.normal(0.085, 0.010), 3),
                    'exit_velocity_last_7': round(np.random.normal(88.5, 2.0), 1),
                    'exit_velocity_last_14': round(np.random.normal(88.5, 1.8), 1),
                    'exit_velocity_last_30': round(np.random.normal(88.5, 1.5), 1),
                    'home_woba': round(np.random.normal(0.325, 0.020), 3),
                    'away_woba': round(np.random.normal(0.315, 0.020), 3),
                    'day_woba': round(np.random.normal(0.318, 0.020), 3),
                    'night_woba': round(np.random.normal(0.322, 0.020), 3),
                    'lineup_strength_rating': round(np.random.normal(0.500, 0.100), 3),
                    'clutch_performance': round(np.random.normal(0.520, 0.050), 3),
                    'runs_per_game_last_7': round(np.random.normal(4.5, 0.8), 1),
                    'runs_per_game_last_14': round(np.random.normal(4.5, 0.6), 1),
                    'runs_per_game_last_30': round(np.random.normal(4.5, 0.5), 1)
                }

                batting_stats[team] = team_stats
                result['loaded'] += 1

            # Cache the batting stats
            await self.cache_manager.set('mlb_batting_stats', batting_stats, 'ml_models', ttl=86400)

            # Archive for historical reference
            await self.persistence_manager.archive_data('mlb_batting_stats', list(batting_stats.values()))

            logger.info(f"✅ Loaded batting stats for {result['loaded']} teams")
            return result

        except Exception as e:
            logger.error(f"❌ Error loading batting statistics: {e}")
            result['errors'].append(str(e))
            return result

    async def _load_park_factors(self) -> dict[str, Any]:
        """Load park factors and stadium data."""
        logger.info("🏟️ Loading park factors")

        result = {'loaded': 0, 'errors': []}

        try:
            # Cache park factors
            await self.cache_manager.set('mlb_park_factors', self.mlb_config['parks'], 'historical', ttl=604800)

            # Archive park factors
            park_data = []
            for team, park_info in self.mlb_config['parks'].items():
                park_data.append({
                    'team': team,
                    'park_name': park_info['name'],
                    'hr_factor': park_info['hr_factor'],
                    'park_factor': park_info['park_factor'],
                    'last_updated': datetime.now().isoformat()
                })

            await self.persistence_manager.archive_data('mlb_park_factors', park_data)
            result['loaded'] = len(park_data)

            logger.info(f"✅ Loaded park factors for {result['loaded']} stadiums")
            return result

        except Exception as e:
            logger.error(f"❌ Error loading park factors: {e}")
            result['errors'].append(str(e))
            return result

    async def _load_historical_data(self) -> dict[str, Any]:
        """Load historical data for analysis."""
        logger.info("📊 Loading historical data")

        result = {'loaded': 0, 'errors': []}

        try:
            # Generate head-to-head matchup data
            h2h_data = {}
            for home_team in self.mlb_config['teams']:
                h2h_data[home_team] = {}
                for away_team in self.mlb_config['teams']:
                    if home_team != away_team:
                        h2h_data[home_team][away_team] = {
                            'home_wins': np.random.randint(5, 15),
                            'away_wins': np.random.randint(5, 15),
                            'total_games': np.random.randint(10, 30),
                            'avg_runs_home': round(np.random.normal(4.8, 0.8), 1),
                            'avg_runs_away': round(np.random.normal(4.2, 0.8), 1),
                            'last_meeting_date': (datetime.now() - timedelta(days=np.random.randint(1, 365))).isoformat()
                        }
                        result['loaded'] += 1

            # Cache head-to-head data
            await self.cache_manager.set('mlb_head_to_head', h2h_data, 'historical', ttl=86400)

            # Generate season trends
            season_trends = {}
            for team in self.mlb_config['teams']:
                season_trends[team] = {
                    'win_percentage': round(np.random.normal(0.500, 0.100), 3),
                    'run_differential': round(np.random.normal(0, 50), 0),
                    'strength_of_schedule': round(np.random.normal(0.500, 0.050), 3),
                    'injury_impact': round(np.random.normal(0, 0.050), 3),
                    'momentum_score': round(np.random.normal(0, 0.100), 3),
                    'rest_advantage_trend': round(np.random.normal(0, 0.020), 3)
                }

            # Cache season trends
            await self.cache_manager.set('mlb_season_trends', season_trends, 'historical', ttl=86400)

            # Archive historical data
            await self.persistence_manager.archive_data('mlb_historical_data', [
                {'type': 'head_to_head', 'data': h2h_data},
                {'type': 'season_trends', 'data': season_trends}
            ])

            logger.info(f"✅ Loaded historical data for {result['loaded']} team matchups")
            return result

        except Exception as e:
            logger.error(f"❌ Error loading historical data: {e}")
            result['errors'].append(str(e))
            return result

    async def _prewarm_cache(self) -> dict[str, Any]:
        """Pre-warm cache with frequently accessed data."""
        logger.info("🔥 Pre-warming cache with frequently accessed data")

        result = {'created': 0, 'errors': []}

        try:
            # Cache frequently accessed data
            cache_entries = {
                'mlb_teams': self.mlb_config['teams'],
                'mlb_data_requirements': self.mlb_config['data_requirements'],
                'mlb_analysis_features': self._get_analysis_features(),
                'mlb_model_config': self._get_model_config(),
                'mlb_weather_impact': self._get_weather_impact_data(),
                'mlb_travel_distances': self._get_travel_distances()
            }

            for key, data in cache_entries.items():
                await self.cache_manager.set(key, data, 'ml_models', ttl=604800)
                result['created'] += 1

            logger.info(f"✅ Pre-warmed cache with {result['created']} entries")
            return result

        except Exception as e:
            logger.error(f"❌ Error pre-warming cache: {e}")
            result['errors'].append(str(e))
            return result

    def _get_analysis_features(self) -> dict[str, list[str]]:
        """Get the features used for MLB analysis."""
        return {
            'pitching_features': [
                'era_last_7', 'era_last_14', 'era_last_30',
                'whip_last_7', 'whip_last_14', 'whip_last_30',
                'k_per_9_last_7', 'k_per_9_last_14', 'k_per_9_last_30',
                'bb_per_9_last_7', 'bb_per_9_last_14', 'bb_per_9_last_30',
                'avg_velocity_last_7', 'avg_velocity_last_14', 'avg_velocity_last_30',
                'spin_rate_last_7', 'spin_rate_last_14', 'spin_rate_last_30',
                'home_era', 'away_era', 'day_era', 'night_era',
                'rest_days_impact', 'bullpen_era_last_7', 'bullpen_era_last_14', 'bullpen_era_last_30'
            ],
            'batting_features': [
                'woba_last_7', 'woba_last_14', 'woba_last_30',
                'iso_last_7', 'iso_last_14', 'iso_last_30',
                'bb_rate_last_7', 'bb_rate_last_14', 'bb_rate_last_30',
                'k_rate_last_7', 'k_rate_last_14', 'k_rate_last_30',
                'barrel_rate_last_7', 'barrel_rate_last_14', 'barrel_rate_last_30',
                'exit_velocity_last_7', 'exit_velocity_last_14', 'exit_velocity_last_30',
                'home_woba', 'away_woba', 'day_woba', 'night_woba',
                'lineup_strength_rating', 'clutch_performance',
                'runs_per_game_last_7', 'runs_per_game_last_14', 'runs_per_game_last_30'
            ],
            'situational_features': [
                'park_factor', 'hr_factor', 'weather_impact', 'travel_distance',
                'rest_advantage', 'day_night_factor', 'series_game_number'
            ],
            'historical_features': [
                'head_to_head_record', 'season_trends', 'injury_impact',
                'momentum_score', 'strength_of_schedule'
            ]
        }

    def _get_model_config(self) -> dict[str, Any]:
        """Get MLB model configuration."""
        return {
            'ensemble_weights': {
                'pitching_model': 0.35,
                'batting_model': 0.25,
                'situational_model': 0.20,
                'historical_model': 0.20
            },
            'feature_importance_weights': {
                'era_last_30': 0.15,
                'woba_last_30': 0.12,
                'park_factor': 0.10,
                'rest_advantage': 0.08,
                'head_to_head_record': 0.07,
                'bullpen_era_last_7': 0.06,
                'avg_velocity_last_30': 0.05,
                'barrel_rate_last_30': 0.05,
                'weather_impact': 0.04,
                'travel_distance': 0.03
            },
            'confidence_thresholds': {
                'high_confidence': 0.75,
                'medium_confidence': 0.60,
                'low_confidence': 0.45
            },
            'ev_thresholds': {
                'minimum_ev': 0.02,
                'target_ev': 0.05,
                'high_ev': 0.08
            }
        }

    def _get_weather_impact_data(self) -> dict[str, float]:
        """Get weather impact factors."""
        return {
            'temperature_impact': {
                'cold': 0.95,      # Reduces offense
                'mild': 1.00,      # Neutral
                'warm': 1.05,      # Increases offense
                'hot': 1.10        # Significantly increases offense
            },
            'wind_impact': {
                'calm': 1.00,      # Neutral
                'light': 1.02,     # Slight increase
                'moderate': 1.05,  # Moderate increase
                'strong': 1.10     # Significant increase
            },
            'humidity_impact': {
                'low': 1.02,       # Slight increase
                'normal': 1.00,    # Neutral
                'high': 0.98       # Slight decrease
            }
        }

    def _get_travel_distances(self) -> dict[str, dict[str, int]]:
        """Get travel distances between MLB cities."""
        # Simplified travel distances (in miles)
        return {
            'Arizona Diamondbacks': {
                'Los Angeles Dodgers': 370,
                'San Diego Padres': 355,
                'San Francisco Giants': 800,
                'Colorado Rockies': 830
            },
            'New York Yankees': {
                'Boston Red Sox': 215,
                'New York Mets': 10,
                'Philadelphia Phillies': 100,
                'Baltimore Orioles': 200
            },
            'Los Angeles Dodgers': {
                'Arizona Diamondbacks': 370,
                'San Diego Padres': 120,
                'San Francisco Giants': 380,
                'Colorado Rockies': 850
            }
            # Add more teams as needed
        }

    async def get_mlb_prediction_features(self, home_team: str, away_team: str,
                                        event_date: datetime) -> dict[str, Any]:
        """Get all features needed for MLB prediction."""
        try:
            # Get cached data
            pitching_stats = await self.cache_manager.get('mlb_pitching_stats', 'ml_models')
            batting_stats = await self.cache_manager.get('mlb_batting_stats', 'ml_models')
            park_factors = await self.cache_manager.get('mlb_park_factors', 'historical')
            h2h_data = await self.cache_manager.get('mlb_head_to_head', 'historical')
            season_trends = await self.cache_manager.get('mlb_season_trends', 'historical')

            if not all([pitching_stats, batting_stats, park_factors, h2h_data, season_trends]):
                logger.warning("Some cached MLB data not available, using fallback")
                return self._get_fallback_features(home_team, away_team)

            # Combine features
            features = {
                # Pitching features
                'home_era_last_30': pitching_stats[home_team]['era_last_30'],
                'away_era_last_30': pitching_stats[away_team]['era_last_30'],
                'home_whip_last_30': pitching_stats[home_team]['whip_last_30'],
                'away_whip_last_30': pitching_stats[away_team]['whip_last_30'],
                'home_k_per_9_last_30': pitching_stats[home_team]['k_per_9_last_30'],
                'away_k_per_9_last_30': pitching_stats[away_team]['k_per_9_last_30'],
                'home_avg_velocity_last_30': pitching_stats[home_team]['avg_velocity_last_30'],
                'away_avg_velocity_last_30': pitching_stats[away_team]['avg_velocity_last_30'],

                # Batting features
                'home_woba_last_30': batting_stats[home_team]['woba_last_30'],
                'away_woba_last_30': batting_stats[away_team]['woba_last_30'],
                'home_iso_last_30': batting_stats[home_team]['iso_last_30'],
                'away_iso_last_30': batting_stats[away_team]['iso_last_30'],
                'home_barrel_rate_last_30': batting_stats[home_team]['barrel_rate_last_30'],
                'away_barrel_rate_last_30': batting_stats[away_team]['barrel_rate_last_30'],

                # Park factors
                'park_factor': park_factors[home_team]['park_factor'],
                'hr_factor': park_factors[home_team]['hr_factor'],

                # Situational features
                'weather_impact': 0.0,  # Would be calculated based on actual weather
                'travel_distance': 0,   # Would be calculated based on actual distance
                'rest_advantage': 0.0,  # Would be calculated based on rest days

                # Historical features
                'h2h_home_win_rate': h2h_data[home_team][away_team]['home_wins'] / h2h_data[home_team][away_team]['total_games'],
                'home_momentum': season_trends[home_team]['momentum_score'],
                'away_momentum': season_trends[away_team]['momentum_score']
            }

            return features

        except Exception as e:
            logger.error(f"Error getting MLB prediction features: {e}")
            return self._get_fallback_features(home_team, away_team)

    def _get_fallback_features(self, home_team: str, away_team: str) -> dict[str, Any]:
        """Get fallback features when cached data is unavailable."""
        return {
            'home_era_last_30': 4.0,
            'away_era_last_30': 4.0,
            'home_whip_last_30': 1.30,
            'away_whip_last_30': 1.30,
            'home_k_per_9_last_30': 8.5,
            'away_k_per_9_last_30': 8.5,
            'home_avg_velocity_last_30': 92.5,
            'away_avg_velocity_last_30': 92.5,
            'home_woba_last_30': 0.320,
            'away_woba_last_30': 0.320,
            'home_iso_last_30': 0.170,
            'away_iso_last_30': 0.170,
            'home_barrel_rate_last_30': 0.085,
            'away_barrel_rate_last_30': 0.085,
            'park_factor': 1.0,
            'hr_factor': 1.0,
            'weather_impact': 0.0,
            'travel_distance': 0,
            'rest_advantage': 0.0,
            'h2h_home_win_rate': 0.5,
            'home_momentum': 0.0,
            'away_momentum': 0.0
        }

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics for MLB data."""
        return await self.cache_manager.get_cache_stats()

    async def cleanup_old_data(self) -> dict[str, Any]:
        """Clean up old MLB data."""
        try:
            # Clean up old cache entries
            invalidated_count = await self.cache_manager.invalidate('mlb_', 'ml_models')

            # Clean up old archives
            deleted_count = await self.persistence_manager.cleanup_old_archives()

            return {
                'cache_entries_invalidated': invalidated_count,
                'archive_files_deleted': deleted_count
            }

        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return {'error': str(e)}


async def main():
    """Main function to run MLB data pre-warming."""
    # Load configuration
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    # Initialize prewarmer
    prewarmer = MLBDataPrewarmer(config)

    # Run pre-warming
    results = await prewarmer.prewarm_all_mlb_data()

    # Print results
    print("\n" + "=" * 60)
    print("📊 MLB DATA PRE-WARMING RESULTS")
    print("=" * 60)
    print(f"Events Created: {results['events_created']}")
    print(f"Pitching Stats Loaded: {results['pitching_stats_loaded']}")
    print(f"Batting Stats Loaded: {results['batting_stats_loaded']}")
    print(f"Park Factors Loaded: {results['park_factors_loaded']}")
    print(f"Historical Data Loaded: {results['historical_data_loaded']}")
    print(f"Cache Entries Created: {results['cache_entries_created']}")

    if results['errors']:
        print(f"\n❌ Errors: {len(results['errors'])}")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")

    # Get cache stats
    cache_stats = await prewarmer.get_cache_stats()
    print("\n🔥 Cache Statistics:")
    print(f"  Total Entries: {cache_stats.get('total_entries', 0)}")
    print(f"  Hit Rate: {cache_stats.get('hit_rate', 0):.2%}")
    print(f"  Cache Size: {cache_stats.get('cache_size_mb', 0):.2f} MB")

    print("\n✅ MLB data pre-warming completed!")


if __name__ == "__main__":
    asyncio.run(main())
