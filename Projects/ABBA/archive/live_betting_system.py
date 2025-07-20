#!/usr/bin/env python3
"""
Live Betting System for ABBA
Implements ML model training, real-time odds, weather integration, and injury tracking.
"""

import asyncio
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import pandas as pd
import structlog
import xgboost as xgb
import yaml
from cache_manager import CacheManager
from database import DatabaseManager

# Import core components
from mlb_data_prewarmer import MLBDataPrewarmer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from models import Event, Odds

logger = structlog.get_logger()


@dataclass
class WeatherData:
    """Weather data for game analysis."""
    temperature: float
    humidity: float
    wind_speed: float
    wind_direction: str
    precipitation_chance: float
    pressure: float
    visibility: float
    game_time: datetime
    stadium: str


@dataclass
class InjuryData:
    """Injury and player availability data."""
    player_name: str
    team: str
    position: str
    injury_type: str
    status: str  # 'active', 'questionable', 'doubtful', 'out'
    expected_return: datetime | None
    impact_score: float  # 0-1 scale of impact on team performance
    last_updated: datetime


class MLModelTrainer:
    """Trains and manages ML models for MLB predictions."""

    def __init__(self, config: dict):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}

        # Model configuration
        self.model_config = {
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 150,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42
            }
        }

        logger.info("ML Model Trainer initialized")

    async def train_models(self, historical_data: pd.DataFrame) -> dict[str, Any]:
        """Train ensemble of ML models for MLB predictions."""
        logger.info("🤖 Training ML models for MLB predictions")

        results = {
            'models_trained': 0,
            'accuracy_scores': {},
            'feature_importance': {},
            'cross_val_scores': {},
            'errors': []
        }

        try:
            # Prepare features and target
            features, target = self._prepare_training_data(historical_data)

            if len(features) < 100:
                raise ValueError("Insufficient training data")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, stratify=target
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            self.scalers['standard'] = scaler

            # Train models
            for model_name, params in self.model_config.items():
                try:
                    logger.info(f"Training {model_name} model...")

                    if model_name == 'xgboost':
                        model = xgb.XGBClassifier(**params)
                    elif model_name == 'random_forest':
                        model = RandomForestClassifier(**params)
                    elif model_name == 'gradient_boosting':
                        model = GradientBoostingClassifier(**params)
                    else:
                        continue

                    # Train model
                    model.fit(X_train_scaled, y_train)

                    # Evaluate model
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

                    accuracy = accuracy_score(y_test, y_pred)
                    auc_score = roc_auc_score(y_test, y_pred_proba)

                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')

                    # Feature importance
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                        feature_names = features.columns
                        importance_dict = dict(zip(feature_names, importance, strict=False))
                        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

                    # Store results
                    self.models[model_name] = model
                    results['accuracy_scores'][model_name] = accuracy
                    results['feature_importance'][model_name] = importance_dict
                    results['cross_val_scores'][model_name] = cv_scores.mean()
                    results['models_trained'] += 1

                    logger.info(f"✅ {model_name} trained - Accuracy: {accuracy:.3f}, AUC: {auc_score:.3f}")

                except Exception as e:
                    logger.error(f"❌ Error training {model_name}: {e}")
                    results['errors'].append(f"{model_name}: {str(e)}")

            # Save models
            await self._save_models()

            logger.info(f"✅ Training completed - {results['models_trained']} models trained")
            return results

        except Exception as e:
            logger.error(f"❌ Error in model training: {e}")
            results['errors'].append(str(e))
            return results

    def _prepare_training_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training."""
        # Select numeric features
        feature_columns = [
            'home_era_last_30', 'away_era_last_30',
            'home_whip_last_30', 'away_whip_last_30',
            'home_k_per_9_last_30', 'away_k_per_9_last_30',
            'home_avg_velocity_last_30', 'away_avg_velocity_last_30',
            'home_woba_last_30', 'away_woba_last_30',
            'home_iso_last_30', 'away_iso_last_30',
            'home_barrel_rate_last_30', 'away_barrel_rate_last_30',
            'park_factor', 'hr_factor',
            'weather_impact', 'travel_distance',
            'h2h_home_win_rate', 'home_momentum', 'away_momentum'
        ]

        # Filter available features
        available_features = [col for col in feature_columns if col in data.columns]

        if len(available_features) < 10:
            # Use fallback features if not enough available
            available_features = data.select_dtypes(include=[np.number]).columns.tolist()

        features = data[available_features].fillna(0)

        # Create target: 1 if home team wins, 0 if away team wins
        # For now, use a simple heuristic based on ERA difference
        target = (features['home_era_last_30'] < features['away_era_last_30']).astype(int)

        return features, target

    async def _save_models(self):
        """Save trained models to disk."""
        try:
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)

            # Save models
            for model_name, model in self.models.items():
                model_path = models_dir / f"{model_name}_mlb_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)

            # Save scaler
            scaler_path = models_dir / "mlb_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers['standard'], f)

            logger.info("✅ Models saved to disk")

        except Exception as e:
            logger.error(f"❌ Error saving models: {e}")

    async def load_models(self):
        """Load trained models from disk."""
        try:
            models_dir = Path("models")

            # Load models
            for model_name in self.model_config.keys():
                model_path = models_dir / f"{model_name}_mlb_model.pkl"
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)

            # Load scaler
            scaler_path = models_dir / "mlb_scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scalers['standard'] = pickle.load(f)

            logger.info(f"✅ Loaded {len(self.models)} models from disk")

        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")

    async def predict(self, features: pd.DataFrame) -> dict[str, Any]:
        """Generate predictions using ensemble of models."""
        try:
            if not self.models:
                await self.load_models()

            if not self.models:
                return {'error': 'No trained models available'}

            # Scale features
            if 'standard' in self.scalers:
                features_scaled = self.scalers['standard'].transform(features)
            else:
                features_scaled = features.values

            # Get predictions from all models
            predictions = {}
            probabilities = {}

            for model_name, model in self.models.items():
                try:
                    pred = model.predict(features_scaled)
                    proba = model.predict_proba(features_scaled)[:, 1]

                    predictions[model_name] = pred[0]
                    probabilities[model_name] = proba[0]

                except Exception as e:
                    logger.error(f"Error with {model_name}: {e}")

            if not predictions:
                return {'error': 'No valid predictions generated'}

            # Ensemble prediction
            avg_probability = np.mean(list(probabilities.values()))
            confidence = 1 - np.std(list(probabilities.values()))

            # Determine prediction
            home_win_prob = avg_probability
            away_win_prob = 1 - avg_probability

            result = {
                'home_win_probability': float(home_win_prob),
                'away_win_probability': float(away_win_prob),
                'confidence': float(confidence),
                'model_predictions': {k: float(v) for k, v in probabilities.items()},
                'ensemble_prediction': 'home' if home_win_prob > 0.5 else 'away',
                'prediction_timestamp': datetime.now().isoformat()
            }

            return result

        except Exception as e:
            logger.error(f"❌ Error in prediction: {e}")
            return {'error': str(e)}


class RealTimeOddsFeed:
    """Real-time odds feed integration."""

    def __init__(self, config: dict):
        self.config = config
        self.api_keys = {
            'the_odds_api': config.get('apis', {}).get('the_odds_api_key'),
            'sportsdataio': config.get('apis', {}).get('sportsdataio_key'),
            'fanduel': config.get('apis', {}).get('fanduel_key'),
            'draftkings': config.get('apis', {}).get('draftkings_key')
        }
        self.session = None
        self.last_update = {}

        logger.info("Real-time odds feed initialized")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_live_odds(self, sport: str = 'baseball_mlb') -> list[Odds]:
        """Fetch live odds from multiple sources."""
        logger.info("📊 Fetching live odds")

        all_odds = []

        try:
            # Fetch from The Odds API
            if self.api_keys['the_odds_api']:
                odds_api_odds = await self._fetch_odds_api(sport)
                all_odds.extend(odds_api_odds)

            # Fetch from SportsData.io
            if self.api_keys['sportsdataio']:
                sportsdata_odds = await self._fetch_sportsdataio(sport)
                all_odds.extend(sportsdata_odds)

            # Fetch from FanDuel
            if self.api_keys['fanduel']:
                fanduel_odds = await self._fetch_fanduel(sport)
                all_odds.extend(fanduel_odds)

            logger.info(f"✅ Fetched {len(all_odds)} live odds")
            return all_odds

        except Exception as e:
            logger.error(f"❌ Error fetching live odds: {e}")
            return []

    async def _fetch_odds_api(self, sport: str) -> list[Odds]:
        """Fetch odds from The Odds API."""
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
            params = {
                'apiKey': self.api_keys['the_odds_api'],
                'regions': 'us',
                'markets': 'h2h,totals,spreads',
                'oddsFormat': 'american'
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    odds_list = []

                    for game in data:
                        event_id = game.get('id')

                        for bookmaker in game.get('bookmakers', []):
                            platform = bookmaker.get('title', '').lower()

                            for market in bookmaker.get('markets', []):
                                market_type = market.get('key', '')

                                for outcome in market.get('outcomes', []):
                                    odds_obj = Odds(
                                        event_id=event_id,
                                        platform=platform,
                                        market_type=market_type,
                                        selection=outcome.get('name', ''),
                                        odds=Decimal(str(outcome.get('price', 0))),
                                        line=Decimal(str(market.get('point', 0))) if market.get('point') else None,
                                        implied_probability=Decimal(str(outcome.get('price', 0)))
                                    )
                                    odds_list.append(odds_obj)

                    return odds_list
                else:
                    logger.warning(f"Odds API returned status {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Error fetching from Odds API: {e}")
            return []

    async def _fetch_sportsdataio(self, sport: str) -> list[Odds]:
        """Fetch odds from SportsData.io."""
        # Implementation would depend on SportsData.io API structure
        # For now, return empty list
        return []

    async def _fetch_fanduel(self, sport: str) -> list[Odds]:
        """Fetch odds from FanDuel API."""
        # Implementation would depend on FanDuel API structure
        # For now, return empty list
        return []


class WeatherIntegration:
    """Weather data integration for game analysis."""

    def __init__(self, config: dict):
        self.config = config
        self.api_key = config.get('apis', {}).get('openweather_api_key')
        self.session = None

        # Weather impact factors
        self.weather_impacts = {
            'temperature': {
                'cold': {'factor': 0.95, 'description': 'Reduces offense'},
                'mild': {'factor': 1.00, 'description': 'Neutral'},
                'warm': {'factor': 1.05, 'description': 'Increases offense'},
                'hot': {'factor': 1.10, 'description': 'Significantly increases offense'}
            },
            'wind': {
                'calm': {'factor': 1.00, 'description': 'Neutral'},
                'light': {'factor': 1.02, 'description': 'Slight increase'},
                'moderate': {'factor': 1.05, 'description': 'Moderate increase'},
                'strong': {'factor': 1.10, 'description': 'Significant increase'}
            },
            'humidity': {
                'low': {'factor': 1.02, 'description': 'Slight increase'},
                'normal': {'factor': 1.00, 'description': 'Neutral'},
                'high': {'factor': 0.98, 'description': 'Slight decrease'}
            }
        }

        logger.info("Weather integration initialized")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_weather_data(self, stadium: str, game_time: datetime) -> WeatherData | None:
        """Get weather data for a specific stadium and time."""
        try:
            if not self.api_key:
                logger.warning("No weather API key configured")
                return self._get_mock_weather_data(stadium, game_time)

            # Get stadium coordinates
            coords = self._get_stadium_coordinates(stadium)
            if not coords:
                return self._get_mock_weather_data(stadium, game_time)

            # Fetch weather data
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': coords['lat'],
                'lon': coords['lon'],
                'appid': self.api_key,
                'units': 'imperial'
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    weather = WeatherData(
                        temperature=data['main']['temp'],
                        humidity=data['main']['humidity'],
                        wind_speed=data['wind']['speed'],
                        wind_direction=self._get_wind_direction(data['wind'].get('deg', 0)),
                        precipitation_chance=data.get('rain', {}).get('1h', 0),
                        pressure=data['main']['pressure'],
                        visibility=data.get('visibility', 10000) / 1000,  # Convert to km
                        game_time=game_time,
                        stadium=stadium
                    )

                    return weather
                else:
                    logger.warning(f"Weather API returned status {response.status}")
                    return self._get_mock_weather_data(stadium, game_time)

        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return self._get_mock_weather_data(stadium, game_time)

    def _get_stadium_coordinates(self, stadium: str) -> dict[str, float] | None:
        """Get coordinates for MLB stadiums."""
        stadium_coords = {
            'Chase Field': {'lat': 33.4453, 'lon': -112.0667},
            'Truist Park': {'lat': 33.8907, 'lon': -84.4677},
            'Oriole Park at Camden Yards': {'lat': 39.2839, 'lon': -76.6217},
            'Fenway Park': {'lat': 42.3467, 'lon': -71.0972},
            'Wrigley Field': {'lat': 41.9484, 'lon': -87.6553},
            'Guaranteed Rate Field': {'lat': 41.8300, 'lon': -87.6338},
            'Great American Ball Park': {'lat': 39.0979, 'lon': -84.5082},
            'Progressive Field': {'lat': 41.4962, 'lon': -81.6852},
            'Coors Field': {'lat': 39.7561, 'lon': -104.9941},
            'Comerica Park': {'lat': 42.3390, 'lon': -83.0495},
            'Minute Maid Park': {'lat': 29.7569, 'lon': -95.3556},
            'Kauffman Stadium': {'lat': 39.0511, 'lon': -94.4803},
            'Angel Stadium': {'lat': 33.8003, 'lon': -117.8827},
            'Dodger Stadium': {'lat': 34.0736, 'lon': -118.2400},
            'loanDepot park': {'lat': 25.7781, 'lon': -80.2197},
            'American Family Field': {'lat': 43.0389, 'lon': -87.9715},
            'Target Field': {'lat': 44.9817, 'lon': -93.2783},
            'Citi Field': {'lat': 40.7569, 'lon': -73.8458},
            'Yankee Stadium': {'lat': 40.8296, 'lon': -73.9262},
            'Oakland Coliseum': {'lat': 37.7516, 'lon': -122.2006},
            'Citizens Bank Park': {'lat': 39.9058, 'lon': -75.1666},
            'PNC Park': {'lat': 40.4469, 'lon': -80.0058},
            'Petco Park': {'lat': 32.7075, 'lon': -117.1570},
            'Oracle Park': {'lat': 37.7786, 'lon': -122.3893},
            'T-Mobile Park': {'lat': 47.5914, 'lon': -122.3321},
            'Busch Stadium': {'lat': 38.6226, 'lon': -90.1928},
            'Tropicana Field': {'lat': 27.7684, 'lon': -82.6534},
            'Globe Life Field': {'lat': 32.7511, 'lon': -97.0824},
            'Rogers Centre': {'lat': 43.6414, 'lon': -79.3891},
            'Nationals Park': {'lat': 38.8730, 'lon': -77.0075}
        }

        return stadium_coords.get(stadium)

    def _get_wind_direction(self, degrees: float) -> str:
        """Convert wind degrees to direction."""
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                     'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        index = round(degrees / 22.5) % 16
        return directions[index]

    def _get_mock_weather_data(self, stadium: str, game_time: datetime) -> WeatherData:
        """Generate mock weather data for testing."""
        return WeatherData(
            temperature=75.0,
            humidity=60.0,
            wind_speed=8.0,
            wind_direction='SW',
            precipitation_chance=0.1,
            pressure=1013.0,
            visibility=10.0,
            game_time=game_time,
            stadium=stadium
        )

    def calculate_weather_impact(self, weather: WeatherData) -> float:
        """Calculate weather impact factor on game performance."""
        try:
            # Temperature impact
            if weather.temperature < 50:
                temp_factor = self.weather_impacts['temperature']['cold']['factor']
            elif weather.temperature < 70:
                temp_factor = self.weather_impacts['temperature']['mild']['factor']
            elif weather.temperature < 85:
                temp_factor = self.weather_impacts['temperature']['warm']['factor']
            else:
                temp_factor = self.weather_impacts['temperature']['hot']['factor']

            # Wind impact
            if weather.wind_speed < 5:
                wind_factor = self.weather_impacts['wind']['calm']['factor']
            elif weather.wind_speed < 10:
                wind_factor = self.weather_impacts['wind']['light']['factor']
            elif weather.wind_speed < 15:
                wind_factor = self.weather_impacts['wind']['moderate']['factor']
            else:
                wind_factor = self.weather_impacts['wind']['strong']['factor']

            # Humidity impact
            if weather.humidity < 40:
                humidity_factor = self.weather_impacts['humidity']['low']['factor']
            elif weather.humidity < 70:
                humidity_factor = self.weather_impacts['humidity']['normal']['factor']
            else:
                humidity_factor = self.weather_impacts['humidity']['high']['factor']

            # Combined impact
            total_impact = (temp_factor + wind_factor + humidity_factor) / 3

            return total_impact

        except Exception as e:
            logger.error(f"Error calculating weather impact: {e}")
            return 1.0  # Neutral impact


class InjuryTracker:
    """Real-time injury and player availability tracking."""

    def __init__(self, config: dict):
        self.config = config
        self.api_key = config.get('apis', {}).get('sportsdataio_key')
        self.session = None
        self.injuries_cache = {}
        self.last_update = {}

        logger.info("Injury tracker initialized")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_injury_data(self, team: str) -> list[InjuryData]:
        """Get injury data for a specific team."""
        try:
            # Check cache first
            cache_key = f"injuries_{team}_{datetime.now().strftime('%Y%m%d')}"
            if cache_key in self.injuries_cache:
                return self.injuries_cache[cache_key]

            # Fetch from API (mock implementation for now)
            injuries = await self._fetch_injury_data(team)

            # Cache the results
            self.injuries_cache[cache_key] = injuries

            logger.info(f"✅ Fetched {len(injuries)} injuries for {team}")
            return injuries

        except Exception as e:
            logger.error(f"❌ Error fetching injury data for {team}: {e}")
            return []

    async def _fetch_injury_data(self, team: str) -> list[InjuryData]:
        """Fetch injury data from API."""
        # Mock implementation - in production would use real API
        mock_injuries = [
            InjuryData(
                player_name="Mock Player 1",
                team=team,
                position="P",
                injury_type="Elbow",
                status="questionable",
                expected_return=datetime.now() + timedelta(days=7),
                impact_score=0.3,
                last_updated=datetime.now()
            ),
            InjuryData(
                player_name="Mock Player 2",
                team=team,
                position="OF",
                injury_type="Hamstring",
                status="out",
                expected_return=datetime.now() + timedelta(days=14),
                impact_score=0.5,
                last_updated=datetime.now()
            )
        ]

        return mock_injuries

    def calculate_injury_impact(self, injuries: list[InjuryData], team: str) -> float:
        """Calculate the impact of injuries on team performance."""
        try:
            if not injuries:
                return 0.0

            # Calculate weighted impact based on player importance and injury severity
            total_impact = 0.0
            total_weight = 0.0

            for injury in injuries:
                if injury.team == team:
                    # Weight by position importance
                    position_weight = self._get_position_weight(injury.position)

                    # Weight by injury status
                    status_weight = self._get_status_weight(injury.status)

                    # Weight by impact score
                    impact_weight = injury.impact_score

                    # Combined weight
                    weight = position_weight * status_weight * impact_weight

                    total_impact += weight
                    total_weight += weight

            if total_weight > 0:
                return total_impact / total_weight
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Error calculating injury impact: {e}")
            return 0.0

    def _get_position_weight(self, position: str) -> float:
        """Get weight for player position importance."""
        position_weights = {
            'P': 0.4,    # Pitchers are most important
            'C': 0.3,    # Catchers are important
            '1B': 0.2,   # First basemen
            '2B': 0.2,   # Second basemen
            '3B': 0.2,   # Third basemen
            'SS': 0.25,  # Shortstops
            'OF': 0.15,  # Outfielders
            'DH': 0.1    # Designated hitters
        }
        return position_weights.get(position, 0.1)

    def _get_status_weight(self, status: str) -> float:
        """Get weight for injury status severity."""
        status_weights = {
            'active': 0.0,
            'questionable': 0.3,
            'doubtful': 0.7,
            'out': 1.0
        }
        return status_weights.get(status, 0.5)


class LiveBettingSystem:
    """Complete live betting system integrating all components."""

    def __init__(self, config: dict):
        self.config = config
        self.mlb_prewarmer = MLBDataPrewarmer(config)
        self.db_manager = DatabaseManager(config['database']['url'])
        self.cache_manager = CacheManager(config)
        self.model_trainer = MLModelTrainer(config)
        self.odds_feed = RealTimeOddsFeed(config)
        self.weather_integration = WeatherIntegration(config)
        self.injury_tracker = InjuryTracker(config)

        # Betting configuration
        self.betting_config = {
            'min_ev_threshold': 0.02,  # 2% minimum expected value
            'max_risk_per_bet': 0.05,  # 5% max risk per bet
            'kelly_fraction': 0.25,    # Conservative Kelly fraction
            'min_confidence': 0.65,    # Minimum model confidence
            'max_bets_per_day': 10,    # Maximum bets per day
            'bankroll': Decimal('10000')  # Starting bankroll
        }

        logger.info("Live betting system initialized")

    async def run_live_betting_cycle(self) -> dict[str, Any]:
        """Run a complete live betting cycle."""
        logger.info("🚀 Starting live betting cycle")

        results = {
            'events_analyzed': 0,
            'opportunities_found': 0,
            'bets_placed': 0,
            'total_ev': 0.0,
            'errors': []
        }

        try:
            # 1. Get upcoming MLB events
            events = await self.db_manager.get_events(sport='baseball_mlb', status='scheduled')
            current_time = datetime.now()
            upcoming_events = [e for e in events if e.event_date.replace(tzinfo=None) > current_time]

            logger.info(f"📅 Analyzing {len(upcoming_events)} upcoming events")

            # 2. Get live odds
            async with self.odds_feed:
                live_odds = await self.odds_feed.get_live_odds()

            # 3. Analyze each event
            for event in upcoming_events[:10]:  # Limit to first 10 for testing
                try:
                    analysis_result = await self._analyze_event(event, live_odds)

                    if analysis_result:
                        results['events_analyzed'] += 1

                        if analysis_result.get('opportunities'):
                            results['opportunities_found'] += len(analysis_result['opportunities'])
                            results['total_ev'] += sum(opp['expected_value'] for opp in analysis_result['opportunities'])

                            # Place bets (mock for now)
                            for opportunity in analysis_result['opportunities']:
                                if opportunity['expected_value'] > self.betting_config['min_ev_threshold']:
                                    bet_result = await self._place_bet(opportunity)
                                    if bet_result:
                                        results['bets_placed'] += 1

                except Exception as e:
                    logger.error(f"Error analyzing event {event.id}: {e}")
                    results['errors'].append(f"Event {event.id}: {str(e)}")

            logger.info(f"✅ Live betting cycle completed: {results}")
            return results

        except Exception as e:
            logger.error(f"❌ Error in live betting cycle: {e}")
            results['errors'].append(str(e))
            return results

    async def _analyze_event(self, event: Event, live_odds: list[Odds]) -> dict[str, Any] | None:
        """Analyze a single event for betting opportunities."""
        try:
            # Get event odds
            event_odds = [o for o in live_odds if o.event_id == event.id]

            if not event_odds:
                return None

            # Get prediction features
            features = await self.mlb_prewarmer.get_mlb_prediction_features(
                event.home_team,
                event.away_team,
                event.event_date
            )

            # Get weather data
            async with self.weather_integration:
                weather = await self.weather_integration.get_weather_data(
                    self._get_stadium_name(event.home_team),
                    event.event_date
                )

                if weather:
                    features['weather_impact'] = self.weather_integration.calculate_weather_impact(weather)

            # Get injury data
            async with self.injury_tracker:
                home_injuries = await self.injury_tracker.get_injury_data(event.home_team)
                away_injuries = await self.injury_tracker.get_injury_data(event.away_team)

                home_injury_impact = self.injury_tracker.calculate_injury_impact(home_injuries, event.home_team)
                away_injury_impact = self.injury_tracker.calculate_injury_impact(away_injuries, event.away_team)

                features['home_injury_impact'] = home_injury_impact
                features['away_injury_impact'] = away_injury_impact

            # Generate prediction
            features_df = pd.DataFrame([features])
            prediction = await self.model_trainer.predict(features_df)

            if 'error' in prediction:
                logger.warning(f"Prediction error for event {event.id}: {prediction['error']}")
                return None

            # Analyze betting opportunities
            opportunities = []

            for odds in event_odds:
                opportunity = self._analyze_betting_opportunity(
                    event, odds, prediction, features
                )

                if opportunity:
                    opportunities.append(opportunity)

            return {
                'event': event,
                'prediction': prediction,
                'opportunities': opportunities,
                'weather': weather,
                'injuries': {
                    'home': home_injuries,
                    'away': away_injuries
                }
            }

        except Exception as e:
            logger.error(f"Error analyzing event {event.id}: {e}")
            return None

    def _analyze_betting_opportunity(self, event: Event, odds: Odds,
                                   prediction: dict, features: dict) -> dict[str, Any] | None:
        """Analyze a specific betting opportunity."""
        try:
            implied_prob = float(odds.implied_probability)

            # Determine our probability based on market type
            if odds.market_type.value == 'moneyline':
                if odds.selection == 'home':
                    our_prob = prediction['home_win_probability']
                else:
                    our_prob = prediction['away_win_probability']
            elif odds.market_type.value == 'totals':
                # For totals, use a different approach
                expected_runs = features.get('home_runs_per_game', 4.5) + features.get('away_runs_per_game', 4.5)
                if odds.selection == 'over':
                    our_prob = 0.5 + (expected_runs - float(odds.line or 8.5)) * 0.1
                else:
                    our_prob = 0.5 - (expected_runs - float(odds.line or 8.5)) * 0.1
            else:
                our_prob = 0.5  # Default for other markets

            # Calculate expected value
            if implied_prob > 0:
                ev = (our_prob * (1 + implied_prob)) - 1

                # Calculate Kelly fraction
                if our_prob > implied_prob:
                    kelly_fraction = (our_prob - implied_prob) / (1 - implied_prob)
                else:
                    kelly_fraction = 0.0

                # Apply Kelly fraction limit
                kelly_fraction = min(kelly_fraction, self.betting_config['kelly_fraction'])

                # Check if opportunity meets criteria
                if (ev > self.betting_config['min_ev_threshold'] and
                    prediction['confidence'] > self.betting_config['min_confidence']):

                    return {
                        'event': event,
                        'odds': odds,
                        'our_probability': our_prob,
                        'implied_probability': implied_prob,
                        'expected_value': ev,
                        'kelly_fraction': kelly_fraction,
                        'confidence': prediction['confidence'],
                        'recommended_stake': self.betting_config['bankroll'] * kelly_fraction,
                        'recommendation': 'BET'
                    }

            return None

        except Exception as e:
            logger.error(f"Error analyzing betting opportunity: {e}")
            return None

    async def _place_bet(self, opportunity: dict) -> bool:
        """Place a bet (mock implementation)."""
        try:
            logger.info(f"🎯 Placing bet: {opportunity['event'].home_team} vs {opportunity['event'].away_team}")
            logger.info(f"   Market: {opportunity['odds'].market_type.value} - {opportunity['odds'].selection}")
            logger.info(f"   Odds: {opportunity['odds'].odds}")
            logger.info(f"   Expected Value: {opportunity['expected_value']:.1%}")
            logger.info(f"   Stake: ${opportunity['recommended_stake']:.2f}")

            # In production, this would integrate with betting platforms
            # For now, just log the bet
            return True

        except Exception as e:
            logger.error(f"Error placing bet: {e}")
            return False

    def _get_stadium_name(self, team: str) -> str:
        """Get stadium name for a team."""
        stadiums = {
            'Arizona Diamondbacks': 'Chase Field',
            'Atlanta Braves': 'Truist Park',
            'Baltimore Orioles': 'Oriole Park at Camden Yards',
            'Boston Red Sox': 'Fenway Park',
            'Chicago Cubs': 'Wrigley Field',
            'Chicago White Sox': 'Guaranteed Rate Field',
            'Cincinnati Reds': 'Great American Ball Park',
            'Cleveland Guardians': 'Progressive Field',
            'Colorado Rockies': 'Coors Field',
            'Detroit Tigers': 'Comerica Park',
            'Houston Astros': 'Minute Maid Park',
            'Kansas City Royals': 'Kauffman Stadium',
            'Los Angeles Angels': 'Angel Stadium',
            'Los Angeles Dodgers': 'Dodger Stadium',
            'Miami Marlins': 'loanDepot park',
            'Milwaukee Brewers': 'American Family Field',
            'Minnesota Twins': 'Target Field',
            'New York Mets': 'Citi Field',
            'New York Yankees': 'Yankee Stadium',
            'Oakland Athletics': 'Oakland Coliseum',
            'Philadelphia Phillies': 'Citizens Bank Park',
            'Pittsburgh Pirates': 'PNC Park',
            'San Diego Padres': 'Petco Park',
            'San Francisco Giants': 'Oracle Park',
            'Seattle Mariners': 'T-Mobile Park',
            'St. Louis Cardinals': 'Busch Stadium',
            'Tampa Bay Rays': 'Tropicana Field',
            'Texas Rangers': 'Globe Life Field',
            'Toronto Blue Jays': 'Rogers Centre',
            'Washington Nationals': 'Nationals Park'
        }
        return stadiums.get(team, 'Unknown Stadium')


async def main():
    """Main function to run the live betting system."""
    # Load configuration
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    # Initialize live betting system
    live_system = LiveBettingSystem(config)

    # Run live betting cycle
    results = await live_system.run_live_betting_cycle()

    # Print results
    print("\n" + "=" * 80)
    print("🎯 LIVE BETTING SYSTEM RESULTS")
    print("=" * 80)
    print(f"Events Analyzed: {results['events_analyzed']}")
    print(f"Opportunities Found: {results['opportunities_found']}")
    print(f"Bets Placed: {results['bets_placed']}")
    print(f"Total Expected Value: {results['total_ev']:.1%}")

    if results['errors']:
        print(f"\n❌ Errors: {len(results['errors'])}")
        for error in results['errors'][:3]:
            print(f"  - {error}")

    print("\n✅ Live betting system completed!")


if __name__ == "__main__":
    asyncio.run(main())
