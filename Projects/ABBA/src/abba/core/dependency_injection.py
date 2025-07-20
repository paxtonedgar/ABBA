"""Dependency injection container for ABBA system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import structlog

from ..analytics.interfaces import (
    DataProcessor, 
    DatabaseInterface, 
    AgentInterface,
    PredictionModel
)

logger = structlog.get_logger()


class ServiceProvider(ABC):
    """Abstract service provider interface."""
    
    @abstractmethod
    def get_service(self, service_type: str) -> Any:
        """Get a service by type."""
        ...
    
    @abstractmethod
    def register_service(self, service_type: str, service: Any) -> None:
        """Register a service."""
        ...


class DatabaseService(DatabaseInterface):
    """Database service implementation."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._connection = None
    
    async def get_bets(self, **kwargs) -> list:
        """Get bets from database."""
        try:
            # Implementation would query database
            logger.info(f"Getting bets with filters: {kwargs}")
            return []
        except Exception as e:
            logger.error(f"Database error getting bets: {e}")
            raise
    
    async def save_bet(self, bet: dict) -> bool:
        """Save a bet to database."""
        try:
            # Implementation would save to database
            logger.info(f"Saving bet: {bet.get('id', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Database error saving bet: {e}")
            raise
    
    async def update_bet(self, bet_id: str, updates: dict) -> bool:
        """Update a bet in database."""
        try:
            # Implementation would update database
            logger.info(f"Updating bet {bet_id} with: {updates}")
            return True
        except Exception as e:
            logger.error(f"Database error updating bet: {e}")
            raise


class BiometricsProcessor(DataProcessor):
    """Biometrics processor implementation."""
    
    async def process(self, data: dict) -> dict:
        """Process biometric data."""
        try:
            logger.info("Processing biometric data")
            
            # Extract biometric data
            heart_rate = data.get("heart_rate", [])
            fatigue_metrics = data.get("fatigue_metrics", {})
            movement = data.get("movement", {})
            
            # Process heart rate
            hr_features = await self._process_heart_rate(heart_rate)
            
            # Process fatigue
            fatigue_level = await self._calculate_fatigue(fatigue_metrics)
            
            # Process movement
            movement_metrics = await self._process_movement(movement)
            
            # Calculate recovery status
            recovery_status = await self._calculate_recovery({
                "hr_features": hr_features,
                "fatigue_level": fatigue_level,
                "movement_metrics": movement_metrics
            })
            
            return {
                "heart_rate": hr_features,
                "fatigue_level": fatigue_level,
                "movement_metrics": movement_metrics,
                "recovery_status": recovery_status
            }
            
        except (ValueError, TypeError) as e:
            logger.error(f"Data validation error in biometric processing: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in biometric processing: {e}")
            raise
    
    async def _process_heart_rate(self, hr_data: list) -> dict:
        """Process heart rate data."""
        try:
            if not hr_data:
                return {"mean_hr": 0.0, "max_hr": 0.0, "min_hr": 0.0, "hr_variability": 0.0, "fatigue_indicator": 0.0}
            
            hr_array = [float(x) for x in hr_data if x is not None]
            if not hr_array:
                return {"mean_hr": 0.0, "max_hr": 0.0, "min_hr": 0.0, "hr_variability": 0.0, "fatigue_indicator": 0.0}
            
            return {
                "mean_hr": sum(hr_array) / len(hr_array),
                "max_hr": max(hr_array),
                "min_hr": min(hr_array),
                "hr_variability": max(hr_array) - min(hr_array),
                "fatigue_indicator": 0.0  # Would calculate based on patterns
            }
        except Exception as e:
            logger.error(f"Error processing heart rate: {e}")
            return {"mean_hr": 0.0, "max_hr": 0.0, "min_hr": 0.0, "hr_variability": 0.0, "fatigue_indicator": 0.0}
    
    async def _calculate_fatigue(self, fatigue_metrics: dict) -> float:
        """Calculate fatigue level."""
        try:
            sleep_quality = fatigue_metrics.get("sleep_quality", 0.5)
            stress_level = fatigue_metrics.get("stress_level", 0.5)
            
            # Simple fatigue calculation
            fatigue = (1.0 - sleep_quality) * 0.6 + stress_level * 0.4
            return min(max(fatigue, 0.0), 1.0)
        except Exception as e:
            logger.error(f"Error calculating fatigue: {e}")
            return 0.5
    
    async def _process_movement(self, movement: dict) -> dict:
        """Process movement data."""
        try:
            return {
                "total_distance": float(movement.get("total_distance", 0.0)),
                "avg_speed": float(movement.get("avg_speed", 0.0)),
                "max_speed": float(movement.get("max_speed", 0.0)),
                "acceleration_count": int(movement.get("acceleration_count", 0))
            }
        except (ValueError, TypeError) as e:
            logger.error(f"Error processing movement data: {e}")
            return {"total_distance": 0.0, "avg_speed": 0.0, "max_speed": 0.0, "acceleration_count": 0}
    
    async def _calculate_recovery(self, processed_data: dict) -> float:
        """Calculate recovery status."""
        try:
            hr_features = processed_data.get("hr_features", {})
            fatigue_level = processed_data.get("fatigue_level", 0.5)
            movement = processed_data.get("movement_metrics", {})
            
            # Simple recovery calculation
            hr_variability = hr_features.get("hr_variability", 0.0)
            movement_score = min(movement.get("total_distance", 0.0) / 10000.0, 1.0)
            
            recovery = (1.0 - fatigue_level) * 0.5 + movement_score * 0.3 + min(hr_variability / 50.0, 1.0) * 0.2
            return min(max(recovery, 0.0), 1.0)
        except Exception as e:
            logger.error(f"Error calculating recovery: {e}")
            return 0.5


class PersonalizationEngine(DataProcessor):
    """Personalization engine implementation."""
    
    async def process(self, data: dict) -> dict:
        """Process user data for personalization."""
        try:
            logger.info("Processing user data for personalization")
            
            user_history = data.get("history", [])
            if not user_history:
                return {"patterns": {}, "model": None}
            
            # Analyze patterns
            patterns = await self._analyze_patterns(user_history)
            
            # Create model based on patterns
            model = await self._create_model_from_patterns(patterns)
            
            return {
                "patterns": patterns,
                "model": model
            }
            
        except (ValueError, TypeError) as e:
            logger.error(f"Data validation error in personalization: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in personalization: {e}")
            raise
    
    async def _analyze_patterns(self, user_history: list) -> dict:
        """Analyze user betting patterns."""
        try:
            if not user_history:
                return {}
            
            # Simple pattern analysis
            total_bets = len(user_history)
            winning_bets = sum(1 for bet in user_history if bet.get("result") == "win")
            win_rate = winning_bets / total_bets if total_bets > 0 else 0.0
            
            avg_stake = sum(float(bet.get("stake", 0)) for bet in user_history) / total_bets if total_bets > 0 else 0.0
            
            return {
                "win_rate": win_rate,
                "avg_stake": avg_stake,
                "total_bets": total_bets,
                "risk_tolerance": 1.0 - win_rate  # Simple risk calculation
            }
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return {}
    
    async def _create_model_from_patterns(self, patterns: dict) -> Optional[Any]:
        """Create model based on user patterns."""
        try:
            # Implementation would create appropriate model based on patterns
            if patterns.get("total_bets", 0) > 10:
                return {"type": "personalized", "patterns": patterns}
            else:
                return None
        except Exception as e:
            logger.error(f"Error creating model from patterns: {e}")
            return None


class GraphAnalyzer(DataProcessor):
    """Graph analyzer implementation."""
    
    async def process(self, data: dict) -> dict:
        """Process team data for graph analysis."""
        try:
            logger.info("Processing team data for graph analysis")
            
            players = data.get("players", [])
            connections = data.get("connections", [])
            
            if not players:
                return {"players": [], "connections": []}
            
            # Process player data
            processed_players = await self._process_players(players)
            
            # Process connections
            processed_connections = await self._process_connections(connections)
            
            return {
                "players": processed_players,
                "connections": processed_connections
            }
            
        except (ValueError, TypeError) as e:
            logger.error(f"Data validation error in graph analysis: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in graph analysis: {e}")
            raise
    
    async def _process_players(self, players: list) -> list:
        """Process player data."""
        try:
            processed = []
            for player in players:
                if isinstance(player, dict):
                    processed.append({
                        "id": player.get("id", ""),
                        "name": player.get("name", ""),
                        "position": player.get("position", ""),
                        "stats": player.get("stats", {})
                    })
            return processed
        except Exception as e:
            logger.error(f"Error processing players: {e}")
            return []
    
    async def _process_connections(self, connections: list) -> list:
        """Process connection data."""
        try:
            processed = []
            for connection in connections:
                if isinstance(connection, dict):
                    processed.append({
                        "from_player": connection.get("from", ""),
                        "to_player": connection.get("to", ""),
                        "strength": float(connection.get("strength", 0.0))
                    })
            return processed
        except Exception as e:
            logger.error(f"Error processing connections: {e}")
            return []


class DependencyContainer:
    """Dependency injection container."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
    
    def register_service(self, service_type: str, service: Any) -> None:
        """Register a service."""
        self._services[service_type] = service
        logger.info(f"Registered service: {service_type}")
    
    def register_singleton(self, service_type: str, service: Any) -> None:
        """Register a singleton service."""
        self._singletons[service_type] = service
        logger.info(f"Registered singleton: {service_type}")
    
    def get_service(self, service_type: str) -> Any:
        """Get a service by type."""
        # Check singletons first
        if service_type in self._singletons:
            return self._singletons[service_type]
        
        # Check regular services
        if service_type in self._services:
            return self._services[service_type]
        
        raise ValueError(f"Service not found: {service_type}")
    
    def configure_default_services(self, config: dict) -> None:
        """Configure default services."""
        try:
            # Database service
            db_service = DatabaseService(config.get("database_url", "sqlite:///abba.db"))
            self.register_singleton("database", db_service)
            
            # Biometrics processor
            biometrics_processor = BiometricsProcessor()
            self.register_singleton("biometrics_processor", biometrics_processor)
            
            # Personalization engine
            personalization_engine = PersonalizationEngine()
            self.register_singleton("personalization_engine", personalization_engine)
            
            # Graph analyzer
            graph_analyzer = GraphAnalyzer()
            self.register_singleton("graph_analyzer", graph_analyzer)
            
            logger.info("Default services configured successfully")
            
        except Exception as e:
            logger.error(f"Error configuring default services: {e}")
            raise
    
    def get_database(self) -> DatabaseInterface:
        """Get database service."""
        return self.get_service("database")
    
    def get_biometrics_processor(self) -> DataProcessor:
        """Get biometrics processor."""
        return self.get_service("biometrics_processor")
    
    def get_personalization_engine(self) -> DataProcessor:
        """Get personalization engine."""
        return self.get_service("personalization_engine")
    
    def get_graph_analyzer(self) -> DataProcessor:
        """Get graph analyzer."""
        return self.get_service("graph_analyzer") 