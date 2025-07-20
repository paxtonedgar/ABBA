"""Data processor for ABBA system."""

import structlog
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from ..analytics.interfaces import DataProcessor as DataProcessorInterface

logger = structlog.get_logger()


class DataProcessor(DataProcessorInterface):
    """Data processor for handling various data types."""
    
    def __init__(self, config: dict):
        self.config = config
        logger.info("DataProcessor initialized")
    
    async def process(self, data: dict) -> dict:
        """Process data according to type."""
        try:
            data_type = data.get("type", "unknown")
            
            if data_type == "mlb":
                return await self._process_mlb_data(data)
            elif data_type == "nhl":
                return await self._process_nhl_data(data)
            elif data_type == "odds":
                return await self._process_odds_data(data)
            elif data_type == "weather":
                return await self._process_weather_data(data)
            else:
                logger.warning(f"Unknown data type: {data_type}")
                return data
                
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return data
    
    async def _process_mlb_data(self, data: dict) -> dict:
        """Process MLB-specific data."""
        try:
            raw_data = data.get("raw_data", {})
            
            # Extract key MLB metrics
            processed_data = {
                "batting_average": raw_data.get("batting_average", 0.0),
                "era": raw_data.get("era", 0.0),
                "home_runs": raw_data.get("home_runs", 0),
                "rbis": raw_data.get("rbis", 0),
                "strikeouts": raw_data.get("strikeouts", 0),
                "walks": raw_data.get("walks", 0),
                "ops": raw_data.get("ops", 0.0),
                "whip": raw_data.get("whip", 0.0),
                "innings_pitched": raw_data.get("innings_pitched", 0.0),
                "saves": raw_data.get("saves", 0)
            }
            
            # Calculate derived metrics
            if processed_data["batting_average"] > 0:
                processed_data["on_base_percentage"] = processed_data.get("on_base_percentage", 0.0)
                processed_data["slugging_percentage"] = processed_data.get("slugging_percentage", 0.0)
            
            logger.info("MLB data processed successfully")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing MLB data: {e}")
            return data
    
    async def _process_nhl_data(self, data: dict) -> dict:
        """Process NHL-specific data."""
        try:
            raw_data = data.get("raw_data", {})
            
            # Extract key NHL metrics
            processed_data = {
                "goals": raw_data.get("goals", 0),
                "assists": raw_data.get("assists", 0),
                "points": raw_data.get("points", 0),
                "plus_minus": raw_data.get("plus_minus", 0),
                "penalty_minutes": raw_data.get("penalty_minutes", 0),
                "shots": raw_data.get("shots", 0),
                "shot_percentage": raw_data.get("shot_percentage", 0.0),
                "time_on_ice": raw_data.get("time_on_ice", 0.0),
                "power_play_goals": raw_data.get("power_play_goals", 0),
                "power_play_points": raw_data.get("power_play_points", 0)
            }
            
            # Calculate derived metrics
            if processed_data["shots"] > 0:
                processed_data["shot_percentage"] = (processed_data["goals"] / processed_data["shots"]) * 100
            
            logger.info("NHL data processed successfully")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing NHL data: {e}")
            return data
    
    async def _process_odds_data(self, data: dict) -> dict:
        """Process odds data."""
        try:
            raw_data = data.get("raw_data", {})
            
            # Extract and validate odds
            processed_data = {
                "home_odds": raw_data.get("home_odds", 0.0),
                "away_odds": raw_data.get("away_odds", 0.0),
                "draw_odds": raw_data.get("draw_odds", 0.0),
                "total_over": raw_data.get("total_over", 0.0),
                "total_under": raw_data.get("total_under", 0.0),
                "spread": raw_data.get("spread", 0.0),
                "spread_odds": raw_data.get("spread_odds", 0.0)
            }
            
            # Calculate implied probabilities
            if processed_data["home_odds"] > 0:
                processed_data["home_probability"] = 1 / processed_data["home_odds"]
            if processed_data["away_odds"] > 0:
                processed_data["away_probability"] = 1 / processed_data["away_odds"]
            if processed_data["draw_odds"] > 0:
                processed_data["draw_probability"] = 1 / processed_data["draw_odds"]
            
            logger.info("Odds data processed successfully")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing odds data: {e}")
            return data
    
    async def _process_weather_data(self, data: dict) -> dict:
        """Process weather data."""
        try:
            raw_data = data.get("raw_data", {})
            
            # Extract weather metrics
            processed_data = {
                "temperature": raw_data.get("temperature", 0.0),
                "humidity": raw_data.get("humidity", 0.0),
                "wind_speed": raw_data.get("wind_speed", 0.0),
                "wind_direction": raw_data.get("wind_direction", ""),
                "precipitation_chance": raw_data.get("precipitation_chance", 0.0),
                "visibility": raw_data.get("visibility", 0.0),
                "pressure": raw_data.get("pressure", 0.0)
            }
            
            # Calculate weather impact score
            impact_score = 0.0
            
            # Temperature impact (optimal range 60-80°F)
            temp = processed_data["temperature"]
            if 60 <= temp <= 80:
                impact_score += 0.2
            elif temp < 40 or temp > 90:
                impact_score += 0.8
            
            # Wind impact
            wind = processed_data["wind_speed"]
            if wind > 20:
                impact_score += 0.6
            elif wind > 15:
                impact_score += 0.4
            
            # Precipitation impact
            precip = processed_data["precipitation_chance"]
            if precip > 0.7:
                impact_score += 0.8
            elif precip > 0.5:
                impact_score += 0.5
            
            processed_data["weather_impact_score"] = min(impact_score, 1.0)
            
            logger.info("Weather data processed successfully")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing weather data: {e}")
            return data 