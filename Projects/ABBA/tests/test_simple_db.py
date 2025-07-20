#!/usr/bin/env python3
"""
Simple database test for ABBA project.
"""

import asyncio

import structlog
from database import DatabaseManager

logger = structlog.get_logger()


async def test_simple_db():
    """Test basic database functionality."""
    print("Testing Simple Database Operations...")

    # Initialize database manager
    db_manager = DatabaseManager("sqlite+aiosqlite:///abmba.db")

    try:
        # Test basic operations
        print("1. Testing basic database operations...")

        # Get current bankroll
        current_bankroll = await db_manager.get_current_bankroll()
        print(f"   ✅ Current bankroll: ${current_bankroll}")

        # Get events
        events = await db_manager.get_events()
        print(f"   ✅ Total events: {len(events)}")

        # Get bankroll history
        bankroll_history = await db_manager.get_bankroll_history(limit=10)
        print(f"   ✅ Bankroll history entries: {len(bankroll_history)}")

        print("\n✅ Database is working correctly!")
        print("📁 Database file: abmba.db")
        print("🔧 Ready for bias detection system integration")

    except Exception as e:
        print(f"❌ Error in database test: {e}")
        raise
    finally:
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(test_simple_db())
