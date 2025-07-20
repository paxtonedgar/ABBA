#!/usr/bin/env python3
"""
Test database integration with bias detection system.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import structlog
from database import DatabaseManager
from simulations import BiasMitigator, MLPredictor

from models import (
    Event,
    EventStatus,
    MarketType,
    Odds,
    PlatformType,
    SportType,
)

logger = structlog.get_logger()


async def test_database_integration():
    """Test database integration with bias detection."""
    print("Testing Database Integration with Bias Detection...")

    # Initialize database manager
    db_manager = DatabaseManager("sqlite+aiosqlite:///abmba.db")

    try:
        # Test basic database operations
        print("1. Testing basic database operations...")

        # Get current bankroll
        current_bankroll = await db_manager.get_current_bankroll()
        print(f"   Current bankroll: ${current_bankroll}")

        # Get events
        events = await db_manager.get_events()
        print(f"   Total events: {len(events)}")

        # Test bias detection with database
        print("\n2. Testing bias detection with database...")

        # Create a bias mitigator
        bias_mitigator = BiasMitigator()

        # Simulate some player stats
        player_stats = np.array([0.300, 0.320, 0.280, 0.350, 0.290])
        print(f"   Original player stats: {player_stats}")

        # Apply park effects adjustment
        adjusted_stats = bias_mitigator.adjust_park_effects(
            player_stats, park_factor=1.15, park_name="Fenway Park"
        )
        print(f"   Adjusted stats (Fenway Park): {adjusted_stats}")

        # Test ML predictor with bias detection
        print("\n3. Testing ML predictor with bias detection...")

        # Create mock data
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 3))
        y = np.random.choice([0, 1], 100)
        groups = np.random.choice(["group_a", "group_b"], 100)

        # Create and train ML predictor
        predictor = MLPredictor("random_forest")

        # Convert to DataFrame (simplified)
        import pandas as pd

        X_df = pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])
        y_series = pd.Series(y)

        # Train model
        predictor.train(X_df, y_series)
        print("   Model trained successfully")

        # Audit for bias
        bias_audit = predictor.audit_bias(X_df, y_series, groups)
        print(f"   Bias audit results: {bias_audit}")

        # Test database storage of bias results
        print("\n4. Testing database storage...")

        # Create a test event
        test_event = Event(
            id="test_event_001",
            sport=SportType.BASEBALL_MLB,
            home_team="Red Sox",
            away_team="Yankees",
            event_date=datetime.now(timezone.utc),
            status=EventStatus.SCHEDULED,
        )

        # Save event to database
        event_id = await db_manager.save_event(test_event)
        print(f"   Saved test event: {event_id}")

        # Create test odds
        test_odds = Odds(
            id="test_odds_001",
            event_id="test_event_001",
            platform=PlatformType.FANDUEL,
            market_type=MarketType.MONEYLINE,
            selection="home",
            odds=Decimal("1.85"),
            timestamp=datetime.now(timezone.utc),
        )

        # Save odds to database
        odds_id = await db_manager.save_odds(test_odds)
        print(f"   Saved test odds: {odds_id}")

        # Get bias report
        bias_report = bias_mitigator.get_bias_report()
        print(f"   Bias report: {bias_report}")

        print("\n✅ Database integration test completed successfully!")

    except Exception as e:
        print(f"❌ Error in database integration test: {e}")
        raise
    finally:
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(test_database_integration())
