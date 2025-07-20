#!/usr/bin/env python3
"""
Database initialization script for ABBA project.
"""

import asyncio

from database import DatabaseManager


async def init_db():
    """Initialize the database."""
    try:
        # Use async SQLite driver
        db_manager = DatabaseManager('sqlite+aiosqlite:///abmba.db')
        await db_manager.initialize()
        await db_manager.close()
        print('✅ Database initialized successfully!')
        print('📁 Database file: abmba.db')
    except Exception as e:
        print(f'❌ Error initializing database: {e}')
        raise

if __name__ == "__main__":
    asyncio.run(init_db())
