"""Configuration management for ABBA."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Config(BaseModel):
    """Main configuration class for ABBA."""

    # API Configuration
    openai_api_key: str | None = Field(default=None, env="OPENAI_API_KEY")
    browserbase_api_key: str | None = Field(default=None, env="BROWSERBASE_API_KEY")

    # Database Configuration
    database_url: str = Field("sqlite:///abba.db", env="DATABASE_URL")
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")

    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Path | None = Field(default=None, env="LOG_FILE")

    # Sports Configuration
    supported_sports: list[str] = Field(["MLB", "NHL"], env="SUPPORTED_SPORTS")

    # Model Configuration
    model_cache_dir: Path = Field(Path("./models"), env="MODEL_CACHE_DIR")

    # Trading Configuration
    max_bet_amount: float = Field(100.0, env="MAX_BET_AMOUNT")
    risk_tolerance: float = Field(0.1, env="RISK_TOLERANCE")

    model_config = {"env_file": ".env", "case_sensitive": False}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_cache_dir.mkdir(exist_ok=True)

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.log_level.upper() == "PRODUCTION"

    def get_sport_config(self, sport: str) -> dict[str, Any]:
        """Get sport-specific configuration."""
        return {
            "MLB": {
                "season_length": 162,
                "playoff_teams": 12,
                "data_sources": ["pybaseball", "mlb_api"],
            },
            "NHL": {
                "season_length": 82,
                "playoff_teams": 16,
                "data_sources": ["nhl_api"],
            },
        }.get(sport.upper(), {})
