"""
Data models for ABMBA system using Pydantic for validation and type safety.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, validator


class SportType(str, Enum):
    """Supported sports types."""
    BASKETBALL_NBA = "basketball_nba"
    BASKETBALL_NCAAB = "basketball_ncaab"
    FOOTBALL_NFL = "football_nfl"
    FOOTBALL_NCAAF = "football_ncaaf"
    BASEBALL_MLB = "baseball_mlb"
    HOCKEY_NHL = "hockey_nhl"


class MarketType(str, Enum):
    """Supported betting markets."""
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTALS = "totals"
    PLAYER_PROPS = "player_props"
    TEAM_PROPS = "team_props"


class PlatformType(str, Enum):
    """Supported betting platforms."""
    FANDUEL = "fanduel"
    DRAFTKINGS = "draftkings"
    BETMGM = "betmgm"
    CAESARS = "caesars"


class BetStatus(str, Enum):
    """Bet status enumeration."""
    PENDING = "pending"
    PLACED = "placed"
    WON = "won"
    LOST = "lost"
    PUSH = "push"
    CANCELLED = "cancelled"


class EventStatus(str, Enum):
    """Event status enumeration."""
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"
    CANCELLED = "cancelled"


class Event(BaseModel):
    """Sports event model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sport: SportType
    home_team: str
    away_team: str
    event_date: datetime
    status: EventStatus = EventStatus.SCHEDULED
    home_score: int | None = None
    away_score: int | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }


class Odds(BaseModel):
    """Odds model for different markets."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str
    platform: PlatformType
    market_type: MarketType
    selection: str  # e.g., "home", "away", "over", "under"
    odds: Decimal
    line: Decimal | None = None  # For spreads and totals
    implied_probability: Decimal | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @validator('implied_probability', pre=True, always=True)
    def calculate_implied_probability(cls, v, values):
        """Calculate implied probability from odds if not provided."""
        if v is None and 'odds' in values:
            odds = values['odds']
            if odds > 0:
                return Decimal('100') / (odds + Decimal('100'))
            else:
                return abs(odds) / (abs(odds) + Decimal('100'))
        return v


class Bet(BaseModel):
    """Bet model representing a placed bet."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str
    platform: PlatformType
    market_type: MarketType
    selection: str
    odds: Decimal
    stake: Decimal
    potential_win: Decimal
    expected_value: Decimal
    kelly_fraction: Decimal
    status: BetStatus = BetStatus.PENDING
    placed_at: datetime | None = None
    settled_at: datetime | None = None
    result: str | None = None
    profit_loss: Decimal | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator('potential_win', pre=True, always=True)
    def calculate_potential_win(cls, v, values):
        """Calculate potential win from stake and odds."""
        if v is None and 'stake' in values and 'odds' in values:
            stake = values['stake']
            odds = values['odds']
            if odds > 0:
                return stake * (odds / Decimal('100'))
            else:
                return stake * (Decimal('100') / abs(odds))
        return v


class BankrollLog(BaseModel):
    """Bankroll tracking model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    balance: Decimal
    change: Decimal
    bet_id: str | None = None
    description: str
    source: str  # "bet", "deposit", "withdrawal", "adjustment"


class SimulationResult(BaseModel):
    """Monte Carlo simulation result."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str
    iterations: int
    win_probability: Decimal
    expected_value: Decimal
    variance: Decimal
    confidence_interval_lower: Decimal
    confidence_interval_upper: Decimal
    kelly_fraction: Decimal
    recommended_stake: Decimal
    risk_level: str  # "low", "medium", "high"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ModelPrediction(BaseModel):
    """ML model prediction result."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str
    model_name: str
    prediction: str
    confidence: Decimal
    features: dict[str, str | int | float | Decimal]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ArbitrageOpportunity(BaseModel):
    """Arbitrage opportunity model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str
    market_type: MarketType
    selections: list[dict[str, str | Decimal | PlatformType]]
    total_implied_probability: Decimal
    arbitrage_percentage: Decimal
    recommended_stakes: dict[str, Decimal]
    potential_profit: Decimal
    risk_level: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SystemMetrics(BaseModel):
    """System performance metrics."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_bets: int
    winning_bets: int
    losing_bets: int
    win_rate: Decimal
    total_profit_loss: Decimal
    roi_percentage: Decimal
    current_bankroll: Decimal
    max_drawdown: Decimal
    sharpe_ratio: Decimal | None = None
    var_95: Decimal | None = None


class Alert(BaseModel):
    """System alert model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    level: str  # "info", "warning", "error", "critical"
    category: str  # "bankroll", "detection", "api", "execution"
    message: str
    details: dict | None = None
    resolved: bool = False
    resolved_at: datetime | None = None


class Configuration(BaseModel):
    """System configuration model."""
    system: dict
    bankroll: dict
    apis: dict
    database: dict
    platforms: dict
    sports: list[dict]
    simulation: dict
    agents: dict
    monitoring: dict
    deployment: dict
    security: dict
