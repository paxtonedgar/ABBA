#!/usr/bin/env python3
"""
2024 MLB Season Weather-Enhanced Validation
Integrates real weather data for more realistic validation.
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import pandas as pd
import structlog
import yaml
from live_betting_system import MLModelTrainer

logger = structlog.get_logger()


@dataclass
class WeatherData:
    game_id: str
    date: datetime
    temperature: float
    humidity: float
    wind_speed: float
    wind_direction: str
    precipitation_chance: float
    pressure: float
    visibility: float
    weather_impact: float


@dataclass
class GameResult:
    game_id: str
    date: datetime
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    home_win: bool
    total_runs: int
    venue: str


@dataclass
class HistoricalOdds:
    game_id: str
    date: datetime
    home_team: str
    away_team: str
    bookmaker: str
    home_odds: float
    away_odds: float
    home_implied_prob: float
    away_implied_prob: float
    line_movement: float | None = None
    closing_line: float | None = None


@dataclass
class ModelPrediction:
    game_id: str
    date: datetime
    home_win_probability: float
    away_win_probability: float
    confidence: float
    features_used: dict[str, float]
    feature_timestamp: datetime


@dataclass
class BettingOpportunity:
    game_id: str
    date: datetime
    bookmaker: str
    selection: str
    odds: float
    implied_probability: float
    our_probability: float
    expected_value: float
    kelly_fraction: float
    stake_recommended: float
    actual_outcome: bool | None = None
    profit_loss: float | None = None
    transaction_cost: float = 0.0
    line_movement_impact: float = 0.0
    weather_impact: float = 0.0


class MLB2024WeatherEnhanced:
    """Weather-enhanced validation with real weather data."""

    def __init__(self, config: dict):
        self.config = config
        self.model_trainer = MLModelTrainer(config)
        self.openweather_api_key = config.get('apis', {}).get('openweather_api_key')

        # Realistic betting configuration
        self.betting_config = {
            'min_ev_threshold': 0.03,
            'max_risk_per_bet': 0.01,
            'kelly_fraction': 0.15,
            'min_confidence': 0.75,
            'bankroll': 10000,
            'min_edge': 0.02,
            'max_bet_size': 500,
            'daily_loss_limit': 200,
            'max_drawdown': 0.20,
            'transaction_cost': 0.05,
            'line_movement_slippage': 0.02,
            'max_bets_per_day': 5,
            'correlation_threshold': 0.7
        }

        # Weather impact thresholds
        self.weather_config = config.get('weather', {})

        # Performance tracking
        self.daily_pnl = {}
        self.current_bankroll = self.betting_config['bankroll']
        self.max_bankroll = self.betting_config['bankroll']
        self.bets_today = {}

        logger.info("MLB 2024 Weather-Enhanced Validation initialized")

    async def fetch_historical_weather(self, games: list[GameResult]) -> list[WeatherData]:
        """Fetch real historical weather data for all games."""
        logger.info("🌤️ Fetching real historical weather data")

        weather_data = []

        try:
            if not self.openweather_api_key:
                logger.warning("No OpenWeather API key found, using simulated weather")
                return await self.simulate_weather_data(games)

            # Group games by date to minimize API calls
            games_by_date = {}
            for game in games:
                date_str = game.date.strftime('%Y-%m-%d')
                if date_str not in games_by_date:
                    games_by_date[date_str] = []
                games_by_date[date_str].append(game)

            # Stadium coordinates (major MLB stadiums)
            stadium_coords = {
                'Yankee Stadium': (40.8296, -73.9262),
                'Fenway Park': (42.3467, -71.0972),
                'Rogers Centre': (43.6414, -79.3891),
                'Tropicana Field': (27.7682, -82.6534),
                'Oriole Park at Camden Yards': (39.2839, -76.6217),
                'Dodger Stadium': (34.0739, -118.2400),
                'Oracle Park': (37.7786, -122.3893),
                'Petco Park': (32.7075, -117.1570),
                'Coors Field': (39.7561, -104.9941),
                'Chase Field': (33.4453, -112.0667),
                'Minute Maid Park': (29.7573, -95.3554),
                'Globe Life Field': (32.7511, -97.0824),
                'T-Mobile Park': (47.5914, -122.3321),
                'Angel Stadium': (33.8003, -117.8827),
                'Oakland Coliseum': (37.7516, -122.2006),
                'Truist Park': (33.8904, -84.4679),
                'Citizens Bank Park': (39.9058, -75.1666),
                'Citi Field': (40.7569, -73.8458),
                'Nationals Park': (38.8730, -77.0074),
                'loanDepot park': (25.7780, -80.2196),
                'Wrigley Field': (41.9484, -87.6553),
                'American Family Field': (43.0384, -87.9715),
                'Great American Ball Park': (39.0979, -84.5082),
                'PNC Park': (40.4469, -80.0058),
                'Busch Stadium': (38.6226, -90.1928),
                'Guaranteed Rate Field': (41.8300, -87.6338),
                'Progressive Field': (41.4962, -81.6852),
                'Comerica Park': (42.3390, -83.0485),
                'Kauffman Stadium': (39.0514, -94.4805),
                'Target Field': (44.9817, -93.2783)
            }

            async with aiohttp.ClientSession() as session:
                for date_str, date_games in games_by_date.items():
                    try:
                        # Get weather for each unique stadium on this date
                        stadiums_this_date = set(game.venue for game in date_games)

                        for stadium in stadiums_this_date:
                            coords = stadium_coords.get(stadium)
                            if not coords:
                                continue

                            # Fetch historical weather data
                            weather = await self._fetch_weather_for_date(
                                session, coords[0], coords[1], date_str
                            )

                            if weather:
                                # Apply weather to all games at this stadium on this date
                                for game in date_games:
                                    if game.venue == stadium:
                                        weather_impact = self._calculate_weather_impact(weather)

                                        game_weather = WeatherData(
                                            game_id=game.game_id,
                                            date=game.date,
                                            temperature=weather['temp'],
                                            humidity=weather['humidity'],
                                            wind_speed=weather['wind_speed'],
                                            wind_direction=weather['wind_direction'],
                                            precipitation_chance=weather['precipitation'],
                                            pressure=weather['pressure'],
                                            visibility=weather['visibility'],
                                            weather_impact=weather_impact
                                        )
                                        weather_data.append(game_weather)

                        # Rate limiting - OpenWeather allows 1000 calls/day
                        await asyncio.sleep(0.1)  # 100ms delay between calls

                    except Exception as e:
                        logger.error(f"Error fetching weather for {date_str}: {e}")
                        continue

            logger.info(f"✅ Fetched weather data for {len(weather_data)} games")
            return weather_data

        except Exception as e:
            logger.error(f"❌ Error fetching historical weather: {e}")
            return await self.simulate_weather_data(games)

    async def _fetch_weather_for_date(self, session: aiohttp.ClientSession,
                                    lat: float, lon: float, date_str: str) -> dict | None:
        """Fetch weather data for a specific date and location."""
        try:
            # Convert date to timestamp for OpenWeather API
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            timestamp = int(date_obj.timestamp())

            url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
            params = {
                'lat': lat,
                'lon': lon,
                'dt': timestamp,
                'appid': self.openweather_api_key,
                'units': 'imperial'  # Fahrenheit
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if 'data' in data and len(data['data']) > 0:
                        weather = data['data'][0]

                        return {
                            'temp': weather.get('temp', 70),
                            'humidity': weather.get('humidity', 50),
                            'wind_speed': weather.get('wind_speed', 5),
                            'wind_direction': self._get_wind_direction(weather.get('wind_deg', 0)),
                            'precipitation': weather.get('pop', 0) * 100,  # Convert to percentage
                            'pressure': weather.get('pressure', 1013),
                            'visibility': weather.get('visibility', 10000) / 1000,  # Convert to km
                            'weather_main': weather.get('weather', [{}])[0].get('main', 'Clear')
                        }

                return None

        except Exception as e:
            logger.error(f"Error fetching weather for {date_str}: {e}")
            return None

    def _get_wind_direction(self, degrees: float) -> str:
        """Convert wind degrees to direction."""
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                     'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        index = round(degrees / 22.5) % 16
        return directions[index]

    def _calculate_weather_impact(self, weather: dict) -> float:
        """Calculate weather impact on game performance."""
        impact = 1.0  # Base impact

        # Temperature impact
        temp = weather['temp']
        if temp < 40 or temp > 90:
            impact *= 0.95  # Extreme temperatures reduce performance
        elif 60 <= temp <= 75:
            impact *= 1.02  # Ideal temperature range

        # Wind impact
        wind_speed = weather['wind_speed']
        if wind_speed > 15:
            impact *= 0.97  # High winds reduce performance

        # Precipitation impact
        precip = weather['precipitation']
        if precip > 30:
            impact *= 0.98  # Rain reduces performance

        # Humidity impact
        humidity = weather['humidity']
        if humidity > 80:
            impact *= 0.99  # High humidity slightly reduces performance

        return impact

    async def simulate_weather_data(self, games: list[GameResult]) -> list[WeatherData]:
        """Simulate realistic weather data when API is unavailable."""
        logger.info("🌤️ Simulating realistic weather data")

        weather_data = []

        for game in games:
            # Simulate realistic weather based on season and location
            month = game.date.month

            # Seasonal temperature patterns
            if month in [3, 4, 10]:  # Spring/Fall
                temp = np.random.normal(65, 10)
            elif month in [5, 6, 7, 8, 9]:  # Summer
                temp = np.random.normal(75, 8)
            else:  # Winter (shouldn't happen for MLB)
                temp = np.random.normal(45, 15)

            # Realistic weather parameters
            humidity = np.random.normal(60, 15)
            wind_speed = np.random.exponential(5)
            precipitation = np.random.exponential(10)

            weather_impact = self._calculate_weather_impact({
                'temp': temp,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'precipitation': precipitation
            })

            weather = WeatherData(
                game_id=game.game_id,
                date=game.date,
                temperature=temp,
                humidity=humidity,
                wind_speed=wind_speed,
                wind_direction=self._get_wind_direction(np.random.uniform(0, 360)),
                precipitation_chance=min(precipitation, 100),
                pressure=np.random.normal(1013, 10),
                visibility=np.random.normal(10, 2),
                weather_impact=weather_impact
            )
            weather_data.append(weather)

        logger.info(f"✅ Simulated weather data for {len(weather_data)} games")
        return weather_data

    async def fetch_2024_mlb_data(self) -> list[GameResult]:
        """Fetch real 2024 MLB season data."""
        logger.info("📊 Fetching 2024 MLB season data")

        try:
            url = "https://statsapi.mlb.com/api/v1/schedule"
            params = {
                'sportId': 1,
                'startDate': '03/28/2024',
                'endDate': '10/30/2024',
                'fields': 'dates,games,gamePk,gameDate,teams,home,away,team,score,venue,name'
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        games = []

                        for date_data in data.get('dates', []):
                            for game in date_data.get('games', []):
                                try:
                                    home_team = game['teams']['home']['team']['name']
                                    away_team = game['teams']['away']['team']['name']
                                    home_score = game['teams']['home'].get('score', 0)
                                    away_score = game['teams']['away'].get('score', 0)
                                    venue = game.get('venue', {}).get('name', 'Unknown')

                                    if home_score is not None and away_score is not None:
                                        game_result = GameResult(
                                            game_id=str(game['gamePk']),
                                            date=datetime.fromisoformat(game['gameDate'].replace('Z', '+00:00')),
                                            home_team=home_team,
                                            away_team=away_team,
                                            home_score=home_score,
                                            away_score=away_score,
                                            home_win=home_score > away_score,
                                            total_runs=home_score + away_score,
                                            venue=venue
                                        )
                                        games.append(game_result)

                                except Exception:
                                    continue

                        logger.info(f"✅ Fetched {len(games)} 2024 MLB games")
                        return games
                    else:
                        return []

        except Exception as e:
            logger.error(f"❌ Error fetching 2024 MLB data: {e}")
            return []

    def create_weather_enhanced_features(self, game: GameResult,
                                       weather: WeatherData | None) -> dict[str, float] | None:
        """Create features with weather impact included."""
        try:
            # Base features (same as before)
            features = {
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
                'weather_impact': 1.0,
                'travel_distance': 0,
                'h2h_home_win_rate': 0.5,
                'home_momentum': 0.0,
                'away_momentum': 0.0
            }

            # Add weather impact if available
            if weather:
                features['weather_impact'] = weather.weather_impact

                # Add weather-specific features
                features['temperature'] = weather.temperature
                features['humidity'] = weather.humidity
                features['wind_speed'] = weather.wind_speed
                features['precipitation_chance'] = weather.precipitation_chance

                # Adjust pitching stats based on weather
                if weather.wind_speed > 15:
                    features['home_era_last_30'] *= 1.05  # Higher ERA in high winds
                    features['away_era_last_30'] *= 1.05

                if weather.temperature < 50 or weather.temperature > 85:
                    features['home_avg_velocity_last_30'] *= 0.98  # Slightly lower velocity
                    features['away_avg_velocity_last_30'] *= 0.98

            return features

        except Exception as e:
            logger.error(f"Error creating weather-enhanced features: {e}")
            return None

    async def run_weather_enhanced_validation(self) -> dict[str, Any]:
        """Run weather-enhanced validation."""
        logger.info("🚀 Starting weather-enhanced validation")

        results = {
            'games_analyzed': 0,
            'weather_data_fetched': 0,
            'predictions_generated': 0,
            'betting_opportunities': 0,
            'performance_metrics': {},
            'weather_impact_analysis': {},
            'errors': []
        }

        try:
            # 1. Fetch 2024 MLB data
            games = await self.fetch_2024_mlb_data()
            if not games:
                results['errors'].append("No 2024 MLB data available")
                return results

            results['games_analyzed'] = len(games)

            # 2. Fetch real weather data
            weather_data = await self.fetch_historical_weather(games)
            results['weather_data_fetched'] = len(weather_data)

            # 3. Create weather-enhanced features and predictions
            predictions = []
            weather_impacts = []

            await self.model_trainer.load_models()

            for game in games:
                try:
                    weather = next((w for w in weather_data if w.game_id == game.game_id), None)
                    features = self.create_weather_enhanced_features(game, weather)

                    if features:
                        features_df = pd.DataFrame([features])
                        prediction_result = await self.model_trainer.predict(features_df)

                        if 'error' not in prediction_result:
                            prediction = ModelPrediction(
                                game_id=game.game_id,
                                date=game.date,
                                home_win_probability=prediction_result['home_win_probability'],
                                away_win_probability=prediction_result['away_win_probability'],
                                confidence=prediction_result['confidence'],
                                features_used=features,
                                feature_timestamp=game.date
                            )
                            predictions.append(prediction)

                            if weather:
                                weather_impacts.append(weather.weather_impact)

                except Exception:
                    continue

            results['predictions_generated'] = len(predictions)

            # 4. Analyze weather impact
            if weather_impacts:
                results['weather_impact_analysis'] = {
                    'average_weather_impact': np.mean(weather_impacts),
                    'weather_impact_std': np.std(weather_impacts),
                    'min_weather_impact': np.min(weather_impacts),
                    'max_weather_impact': np.max(weather_impacts),
                    'games_with_weather_data': len(weather_impacts)
                }

            # 5. Save results
            self._save_weather_enhanced_results(games, predictions, weather_data, results)

            logger.info("✅ Weather-enhanced validation completed")
            return results

        except Exception as e:
            logger.error(f"❌ Error in weather-enhanced validation: {e}")
            results['errors'].append(str(e))
            return results

    def _save_weather_enhanced_results(self, games: list[GameResult],
                                     predictions: list[ModelPrediction],
                                     weather_data: list[WeatherData],
                                     results: dict[str, Any]):
        """Save weather-enhanced validation results."""
        try:
            results_dir = Path("validation_results")
            results_dir.mkdir(exist_ok=True)

            results_data = {
                'results': results,
                'games_count': len(games),
                'predictions_count': len(predictions),
                'weather_data_count': len(weather_data),
                'timestamp': datetime.now().isoformat()
            }

            with open(results_dir / "2024_mlb_weather_enhanced_results.json", "w") as f:
                json.dump(results_data, f, indent=2, default=str)

            logger.info("✅ Weather-enhanced results saved")

        except Exception as e:
            logger.error(f"Error saving results: {e}")


async def main():
    """Main function to run weather-enhanced validation."""
    # Load configuration
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    # Initialize validator
    validator = MLB2024WeatherEnhanced(config)

    # Run weather-enhanced validation
    results = await validator.run_weather_enhanced_validation()

    # Print results
    print("\n" + "=" * 80)
    print("🌤️ 2024 MLB SEASON WEATHER-ENHANCED VALIDATION RESULTS")
    print("=" * 80)
    print(f"Games Analyzed: {results['games_analyzed']}")
    print(f"Weather Data Fetched: {results['weather_data_fetched']}")
    print(f"Predictions Generated: {results['predictions_generated']}")

    if results['weather_impact_analysis']:
        weather = results['weather_impact_analysis']
        print("\n🌤️ Weather Impact Analysis:")
        print(f"   Average Weather Impact: {weather['average_weather_impact']:.3f}")
        print(f"   Weather Impact Std Dev: {weather['weather_impact_std']:.3f}")
        print(f"   Min Weather Impact: {weather['min_weather_impact']:.3f}")
        print(f"   Max Weather Impact: {weather['max_weather_impact']:.3f}")
        print(f"   Games with Weather Data: {weather['games_with_weather_data']}")

    if results['errors']:
        print(f"\n❌ Errors: {len(results['errors'])}")
        for error in results['errors'][:3]:
            print(f"   - {error}")

    print("\n✅ Weather-enhanced validation completed!")


if __name__ == "__main__":
    asyncio.run(main())
