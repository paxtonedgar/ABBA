"""
Enhanced Feature Engineer for ABBA System
Optimized feature engineering with caching and batch processing.
"""

import json
import uuid
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import structlog
from database import DatabaseManager

logger = structlog.get_logger()


class OptimizedFeatureEngineer:
    """Optimized feature engineering with caching and batch processing."""

    def __init__(self, db_manager: DatabaseManager, config: dict):
        self.db_manager = db_manager
        self.config = config
        self.feature_cache = {}
        self.computation_graph = {}
        self.feature_registry = {}

        # Initialize sport-specific feature sets
        self.mlb_features = self._initialize_mlb_features()
        self.nhl_features = self._initialize_nhl_features()

        logger.info("OptimizedFeatureEngineer initialized")

    def _initialize_mlb_features(self) -> dict[str, list[str]]:
        """Initialize MLB-specific feature categories."""
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
            'market_features': [
                'odds_movement', 'line_movement', 'volume_pattern',
                'sharp_action_indicator', 'public_betting_percentage'
            ]
        }

    def _initialize_nhl_features(self) -> dict[str, list[str]]:
        """Initialize NHL-specific feature categories."""
        return {
            'goalie_features': [
                'save_percentage_last_7', 'save_percentage_last_14', 'save_percentage_last_30',
                'goals_against_average_last_7', 'goals_against_average_last_14', 'goals_against_average_last_30',
                'quality_starts_last_7', 'quality_starts_last_14', 'quality_starts_last_30',
                'home_save_percentage', 'away_save_percentage'
            ],
            'team_features': [
                'corsi_for_percentage_last_7', 'corsi_for_percentage_last_14', 'corsi_for_percentage_last_30',
                'fenwick_for_percentage_last_7', 'fenwick_for_percentage_last_14', 'fenwick_for_percentage_last_30',
                'expected_goals_for_last_7', 'expected_goals_for_last_14', 'expected_goals_for_last_30',
                'power_play_percentage_last_7', 'power_play_percentage_last_14', 'power_play_percentage_last_30',
                'penalty_kill_percentage_last_7', 'penalty_kill_percentage_last_14', 'penalty_kill_percentage_last_30'
            ],
            'situational_features': [
                'rest_advantage', 'travel_distance', 'back_to_back',
                'home_ice_advantage', 'season_series_record'
            ],
            'market_features': [
                'odds_movement', 'line_movement', 'volume_pattern',
                'sharp_action_indicator', 'public_betting_percentage'
            ]
        }

    @lru_cache(maxsize=1000)
    def get_cached_features(self, event_id: str, feature_set: str) -> pd.DataFrame | None:
        """Get cached features for an event."""
        cache_key = f"{event_id}_{feature_set}"
        return self.feature_cache.get(cache_key)

    async def precompute_features_batch(self, events: list[dict]) -> dict[str, pd.DataFrame]:
        """Pre-compute features for multiple events in batch."""
        logger.info(f"Pre-computing features for {len(events)} events")

        results = {}

        # Group events by sport for efficient processing
        mlb_events = [e for e in events if e.get('sport') == 'baseball_mlb']
        nhl_events = [e for e in events if e.get('sport') == 'hockey_nhl']

        # Process MLB events
        if mlb_events:
            mlb_features = await self._compute_mlb_features_batch(mlb_events)
            results.update(mlb_features)

        # Process NHL events
        if nhl_events:
            nhl_features = await self._compute_nhl_features_batch(nhl_events)
            results.update(nhl_features)

        # Cache computed features
        await self._cache_features_batch(results)

        logger.info(f"Pre-computed features for {len(results)} event sets")
        return results

    async def _compute_mlb_features_batch(self, events: list[dict]) -> dict[str, pd.DataFrame]:
        """Compute MLB features for multiple events efficiently."""
        features_dict = {}

        # Get all event IDs
        event_ids = [e['id'] for e in events]

        # Fetch all required data in batch
        statcast_data = await self._fetch_statcast_data_batch(event_ids)
        odds_data = await self._fetch_odds_data_batch(event_ids)
        weather_data = await self._fetch_weather_data_batch(events)

        for event in events:
            event_id = event['id']

            # Compute features for this event
            event_features = await self._compute_single_mlb_features(
                event, statcast_data.get(event_id, []),
                odds_data.get(event_id, []), weather_data.get(event_id, {})
            )

            features_dict[event_id] = event_features

        return features_dict

    async def _compute_single_mlb_features(self, event: dict, statcast_data: list,
                                         odds_data: list, weather_data: dict) -> pd.DataFrame:
        """Compute features for a single MLB event."""
        features = {}

        # Basic event features
        features['event_id'] = event['id']
        features['home_team'] = event['home_team']
        features['away_team'] = event['away_team']
        features['event_date'] = event['event_date']

        # Compute pitching features from Statcast data
        if statcast_data:
            features.update(self._compute_pitching_features(statcast_data))

        # Compute batting features from Statcast data
        if statcast_data:
            features.update(self._compute_batting_features(statcast_data))

        # Compute situational features
        features.update(self._compute_situational_features(event, weather_data))

        # Compute market features from odds data
        if odds_data:
            features.update(self._compute_market_features(odds_data))

        return pd.DataFrame([features])

    def _compute_pitching_features(self, statcast_data: list) -> dict[str, float]:
        """Compute pitching features from Statcast data."""
        features = {}

        # Convert to DataFrame for easier processing
        df = pd.DataFrame(statcast_data)

        if not df.empty:
            # ERA calculations (simplified)
            features['avg_release_speed'] = df['release_speed'].mean() if 'release_speed' in df.columns else 92.0
            features['avg_spin_rate'] = df['spin_rate'].mean() if 'spin_rate' in df.columns else 2200.0

            # Pitch type distribution
            if 'pitch_type' in df.columns:
                pitch_counts = df['pitch_type'].value_counts()
                features['fastball_percentage'] = pitch_counts.get('FF', 0) / len(df) if len(df) > 0 else 0.6
                features['breaking_percentage'] = (pitch_counts.get('SL', 0) + pitch_counts.get('CB', 0)) / len(df) if len(df) > 0 else 0.3
                features['offspeed_percentage'] = pitch_counts.get('CH', 0) / len(df) if len(df) > 0 else 0.1

        return features

    def _compute_batting_features(self, statcast_data: list) -> dict[str, float]:
        """Compute batting features from Statcast data."""
        features = {}

        df = pd.DataFrame(statcast_data)

        if not df.empty:
            # Exit velocity features
            features['avg_exit_velocity'] = df['launch_speed'].mean() if 'launch_speed' in df.columns else 88.0
            features['hard_hit_percentage'] = (df['launch_speed'] >= 95).mean() if 'launch_speed' in df.columns else 0.3

            # Launch angle features
            features['avg_launch_angle'] = df['launch_angle'].mean() if 'launch_angle' in df.columns else 12.0
            features['barrel_percentage'] = ((df['launch_angle'] >= 26) & (df['launch_angle'] <= 30) &
                                           (df['launch_speed'] >= 98)).mean() if all(col in df.columns for col in ['launch_angle', 'launch_speed']) else 0.08

        return features

    def _compute_situational_features(self, event: dict, weather_data: dict) -> dict[str, float]:
        """Compute situational features."""
        features = {}

        # Park factors (simplified)
        features['park_factor'] = 1.0  # Would be calculated from historical data
        features['hr_factor'] = 1.0    # Would be calculated from historical data

        # Weather impact
        if weather_data:
            features['temperature'] = weather_data.get('temperature', 70.0)
            features['wind_speed'] = weather_data.get('wind_speed', 5.0)
            features['wind_direction'] = weather_data.get('wind_direction', 'neutral')
            features['precipitation_chance'] = weather_data.get('precipitation_chance', 0.0)
        else:
            features['temperature'] = 70.0
            features['wind_speed'] = 5.0
            features['wind_direction'] = 'neutral'
            features['precipitation_chance'] = 0.0

        # Rest advantage
        features['rest_advantage'] = 0.0  # Would be calculated from schedule data

        return features

    def _compute_market_features(self, odds_data: list) -> dict[str, float]:
        """Compute market features from odds data."""
        features = {}

        if odds_data:
            # Calculate odds movement
            odds_values = [odds['odds'] for odds in odds_data if 'odds' in odds]
            if len(odds_values) > 1:
                features['odds_movement'] = odds_values[-1] - odds_values[0]
                features['odds_volatility'] = np.std(odds_values)
            else:
                features['odds_movement'] = 0.0
                features['odds_volatility'] = 0.0

            # Calculate implied probabilities
            implied_probs = [odds.get('implied_probability', 0.5) for odds in odds_data]
            features['avg_implied_probability'] = np.mean(implied_probs)
            features['implied_probability_std'] = np.std(implied_probs)

        return features

    async def _compute_nhl_features_batch(self, events: list[dict]) -> dict[str, pd.DataFrame]:
        """Compute NHL features for multiple events efficiently."""
        features_dict = {}

        # Get all event IDs
        event_ids = [e['id'] for e in events]

        # Fetch all required data in batch
        shot_data = await self._fetch_shot_data_batch(event_ids)
        odds_data = await self._fetch_odds_data_batch(event_ids)
        weather_data = await self._fetch_weather_data_batch(events)

        for event in events:
            event_id = event['id']

            # Compute features for this event
            event_features = await self._compute_single_nhl_features(
                event, shot_data.get(event_id, []),
                odds_data.get(event_id, []), weather_data.get(event_id, {})
            )

            features_dict[event_id] = event_features

        return features_dict

    async def _compute_single_nhl_features(self, event: dict, shot_data: list,
                                         odds_data: list, weather_data: dict) -> pd.DataFrame:
        """Compute features for a single NHL event."""
        features = {}

        # Basic event features
        features['event_id'] = event['id']
        features['home_team'] = event['home_team']
        features['away_team'] = event['away_team']
        features['event_date'] = event['event_date']

        # Compute goalie features from shot data
        if shot_data:
            features.update(self._compute_goalie_features(shot_data))

        # Compute team features from shot data
        if shot_data:
            features.update(self._compute_team_features(shot_data))

        # Compute situational features
        features.update(self._compute_nhl_situational_features(event, weather_data))

        # Compute market features from odds data
        if odds_data:
            features.update(self._compute_market_features(odds_data))

        return pd.DataFrame([features])

    def _compute_goalie_features(self, shot_data: list) -> dict[str, float]:
        """Compute goalie features from shot data."""
        features = {}

        df = pd.DataFrame(shot_data)

        if not df.empty:
            # Save percentage
            if 'goal' in df.columns:
                features['save_percentage'] = (1 - df['goal'].mean()) * 100
            else:
                features['save_percentage'] = 91.0  # League average

            # Shot distance analysis
            if 'shot_distance' in df.columns:
                features['avg_shot_distance'] = df['shot_distance'].mean()
                features['close_shots_percentage'] = (df['shot_distance'] <= 20).mean()
            else:
                features['avg_shot_distance'] = 35.0
                features['close_shots_percentage'] = 0.3

        return features

    def _compute_team_features(self, shot_data: list) -> dict[str, float]:
        """Compute team features from shot data."""
        features = {}

        df = pd.DataFrame(shot_data)

        if not df.empty:
            # Shot quality metrics
            if 'shot_angle' in df.columns:
                features['avg_shot_angle'] = df['shot_angle'].mean()
                features['high_danger_shots'] = (df['shot_angle'] >= 45).mean()
            else:
                features['avg_shot_angle'] = 25.0
                features['high_danger_shots'] = 0.2

            # Goal scoring efficiency
            if 'goal' in df.columns:
                features['shooting_percentage'] = df['goal'].mean() * 100
            else:
                features['shooting_percentage'] = 9.5  # League average

        return features

    def _compute_nhl_situational_features(self, event: dict, weather_data: dict) -> dict[str, float]:
        """Compute NHL situational features."""
        features = {}

        # Home ice advantage
        features['home_ice_advantage'] = 1.0

        # Weather impact (less significant for indoor sports)
        if weather_data:
            features['temperature'] = weather_data.get('temperature', 70.0)
            features['humidity'] = weather_data.get('humidity', 50.0)
        else:
            features['temperature'] = 70.0
            features['humidity'] = 50.0

        # Rest advantage
        features['rest_advantage'] = 0.0  # Would be calculated from schedule data

        return features

    async def _fetch_statcast_data_batch(self, event_ids: list[str]) -> dict[str, list]:
        """Fetch Statcast data for multiple events in batch."""
        # This would integrate with your existing data fetcher
        # For now, return mock data
        return {event_id: [] for event_id in event_ids}

    async def _fetch_shot_data_batch(self, event_ids: list[str]) -> dict[str, list]:
        """Fetch shot data for multiple events in batch."""
        # This would integrate with your existing data fetcher
        # For now, return mock data
        return {event_id: [] for event_id in event_ids}

    async def _fetch_odds_data_batch(self, event_ids: list[str]) -> dict[str, list]:
        """Fetch odds data for multiple events in batch."""
        # This would integrate with your existing data fetcher
        # For now, return mock data
        return {event_id: [] for event_id in event_ids}

    async def _fetch_weather_data_batch(self, events: list[dict]) -> dict[str, dict]:
        """Fetch weather data for multiple events in batch."""
        # This would integrate with your existing weather integration
        # For now, return mock data
        return {event['id']: {} for event in events}

    async def _cache_features_batch(self, features_dict: dict[str, pd.DataFrame]):
        """Cache computed features in database."""
        for event_id, features_df in features_dict.items():
            if not features_df.empty:
                # Store in database
                await self._store_features_in_db(event_id, features_df)

                # Store in memory cache
                cache_key = f"{event_id}_mlb"
                self.feature_cache[cache_key] = features_df

    async def _store_features_in_db(self, event_id: str, features_df: pd.DataFrame):
        """Store features in the engineered_features table."""
        try:
            features_json = features_df.to_json(orient='records')
            feature_id = str(uuid.uuid4())

            # Store in database using raw SQL for now
            # In a full implementation, this would use your database manager
            import sqlite3
            conn = sqlite3.connect('abmba.db')
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO engineered_features 
                (id, event_id, sport, feature_set_name, features, feature_version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feature_id, event_id, 'baseball_mlb', 'mlb_advanced',
                features_json, '1.0', datetime.utcnow(), datetime.utcnow()
            ))

            conn.commit()
            conn.close()

            logger.debug(f"Stored features for event {event_id}")

        except Exception as e:
            logger.error(f"Error storing features for event {event_id}: {e}")

    async def get_features_for_prediction(self, event_id: str, sport: str) -> pd.DataFrame:
        """Get features for prediction, using cache if available."""
        try:
            # Try to get from cache first
            cache_key = f"{event_id}_{sport}"
            if cache_key in self.feature_cache:
                logger.info(f"Using cached features for event {event_id}")
                return self.feature_cache[cache_key]

            # Try to get from database
            features_data = await self._get_features_from_db(event_id, sport)
            if features_data:
                logger.info(f"Using database features for event {event_id}")
                return pd.DataFrame(features_data)

            # Compute features if not available
            logger.info(f"Computing features for event {event_id}")
            event_data = await self._get_event_data(event_id)
            if event_data:
                features = await self._compute_single_mlb_features(
                    event_data, [], [], {}
                ) if sport == 'baseball_mlb' else await self._compute_single_nhl_features(
                    event_data, [], [], {}
                )

                # Cache the computed features
                self.feature_cache[cache_key] = features
                await self._store_features_in_db(event_id, features)

                return features

            # Return empty DataFrame if no data available
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error getting features for event {event_id}: {e}")
            return pd.DataFrame()

    async def _get_features_from_db(self, event_id: str, sport: str) -> list[dict] | None:
        """Get features from database."""
        try:
            import sqlite3
            conn = sqlite3.connect('abmba.db')
            cursor = conn.cursor()

            cursor.execute("""
                SELECT features FROM engineered_features 
                WHERE event_id = ? AND sport = ? 
                ORDER BY created_at DESC LIMIT 1
            """, (event_id, sport))

            result = cursor.fetchone()
            conn.close()

            if result:
                return json.loads(result[0])

            return None

        except Exception as e:
            logger.error(f"Error getting features from database: {e}")
            return None

    async def _get_event_data(self, event_id: str) -> dict | None:
        """Get event data from database."""
        try:
            import sqlite3
            conn = sqlite3.connect('abmba.db')
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, sport, home_team, away_team, event_date 
                FROM events WHERE id = ?
            """, (event_id,))

            result = cursor.fetchone()
            conn.close()

            if result:
                return {
                    'id': result[0],
                    'sport': result[1],
                    'home_team': result[2],
                    'away_team': result[3],
                    'event_date': result[4]
                }

            return None

        except Exception as e:
            logger.error(f"Error getting event data: {e}")
            return None
