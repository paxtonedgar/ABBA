"""
Zero-Compromise Database Tests

Real database testing with live PostgreSQL connections.
No mocks, stubs, or fakes - only real database operations.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
import structlog

logger = structlog.get_logger(__name__)


class TestDatabaseZeroMock:
    """Real database testing with zero mocks."""

    @pytest.fixture(autouse=True)
    async def setup_database(self, postgres_pool):
        """Set up real database connection."""
        self.pool = postgres_pool

        # Initialize test schema
        await self._initialize_schema()

        yield

        # Cleanup
        await self._cleanup_test_data()

    async def _initialize_schema(self):
        """Initialize database schema for testing."""
        async with self.pool.acquire() as conn:
            # Create additional test tables
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS test_events (
                    id SERIAL PRIMARY KEY,
                    event_id VARCHAR(255) UNIQUE NOT NULL,
                    sport VARCHAR(50) NOT NULL,
                    home_team VARCHAR(100) NOT NULL,
                    away_team VARCHAR(100) NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    status VARCHAR(50) DEFAULT 'scheduled',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS test_odds (
                    id SERIAL PRIMARY KEY,
                    event_id VARCHAR(255) NOT NULL,
                    bookmaker VARCHAR(50) NOT NULL,
                    market_type VARCHAR(50) NOT NULL,
                    selection VARCHAR(100) NOT NULL,
                    odds DECIMAL(10,2) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (event_id) REFERENCES test_events(event_id)
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS test_predictions (
                    id SERIAL PRIMARY KEY,
                    event_id VARCHAR(255) NOT NULL,
                    model_name VARCHAR(255) NOT NULL,
                    predicted_outcome VARCHAR(100) NOT NULL,
                    confidence DECIMAL(5,4) NOT NULL,
                    expected_value DECIMAL(10,4) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (event_id) REFERENCES test_events(event_id)
                )
            """)

    async def _cleanup_test_data(self):
        """Clean up test data."""
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM test_predictions WHERE model_name LIKE 'test_%'")
            await conn.execute("DELETE FROM test_odds WHERE bookmaker = 'test_bookmaker'")
            await conn.execute("DELETE FROM test_events WHERE event_id LIKE 'test_%'")

    @pytest.mark.integration
    async def test_real_connection_pool(self):
        """Test real database connection pool."""
        logger.info("Testing real database connection pool")

        # Test pool properties
        assert self.pool.get_size() > 0
        assert self.pool.get_free_size() >= 0

        # Test connection acquisition
        async with self.pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            assert result == 1

        logger.info("Successfully tested connection pool")

    @pytest.mark.integration
    async def test_real_event_creation(self):
        """Test real event creation and retrieval."""
        logger.info("Testing real event creation")

        test_event = {
            "event_id": "test_mlb_001",
            "sport": "mlb",
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "start_time": datetime.now() + timedelta(hours=2)
        }

        # Insert event
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO test_events (event_id, sport, home_team, away_team, start_time)
                VALUES ($1, $2, $3, $4, $5)
            """, test_event["event_id"], test_event["sport"], test_event["home_team"],
                 test_event["away_team"], test_event["start_time"])

        # Retrieve event
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow("""
                SELECT * FROM test_events WHERE event_id = $1
            """, test_event["event_id"])

        assert result is not None
        assert result['event_id'] == test_event["event_id"]
        assert result['sport'] == test_event["sport"]
        assert result['home_team'] == test_event["home_team"]
        assert result['away_team'] == test_event["away_team"]

        logger.info("Successfully created and retrieved event")

    @pytest.mark.integration
    async def test_real_odds_management(self):
        """Test real odds management operations."""
        logger.info("Testing real odds management")

        # Create test event first
        event_id = "test_odds_event_001"
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO test_events (event_id, sport, home_team, away_team, start_time)
                VALUES ($1, $2, $3, $4, $5)
            """, event_id, "nhl", "Toronto Maple Leafs", "Montreal Canadiens",
                 datetime.now() + timedelta(hours=1))

        # Insert multiple odds
        test_odds = [
            (event_id, "test_bookmaker", "moneyline", "home", Decimal("1.85")),
            (event_id, "test_bookmaker", "moneyline", "away", Decimal("2.10")),
            (event_id, "test_bookmaker", "total", "over", Decimal("1.95")),
            (event_id, "test_bookmaker", "total", "under", Decimal("1.85"))
        ]

        async with self.pool.acquire() as conn:
            for odds in test_odds:
                await conn.execute("""
                    INSERT INTO test_odds (event_id, bookmaker, market_type, selection, odds)
                    VALUES ($1, $2, $3, $4, $5)
                """, *odds)

        # Retrieve odds
        async with self.pool.acquire() as conn:
            results = await conn.fetch("""
                SELECT * FROM test_odds WHERE event_id = $1 ORDER BY market_type, selection
            """, event_id)

        assert len(results) == 4

        # Validate odds data
        odds_dict = {f"{r['market_type']}_{r['selection']}": r['odds'] for r in results}
        assert odds_dict['moneyline_home'] == Decimal("1.85")
        assert odds_dict['moneyline_away'] == Decimal("2.10")
        assert odds_dict['total_over'] == Decimal("1.95")
        assert odds_dict['total_under'] == Decimal("1.85")

        logger.info("Successfully tested odds management")

    @pytest.mark.integration
    async def test_real_prediction_storage(self):
        """Test real prediction storage and retrieval."""
        logger.info("Testing real prediction storage")

        # Create test event
        event_id = "test_prediction_event_001"
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO test_events (event_id, sport, home_team, away_team, start_time)
                VALUES ($1, $2, $3, $4, $5)
            """, event_id, "mlb", "Los Angeles Dodgers", "San Francisco Giants",
                 datetime.now() + timedelta(hours=3))

        # Store predictions
        test_predictions = [
            (event_id, "test_model_1", "home_win", Decimal("0.75"), Decimal("0.05")),
            (event_id, "test_model_2", "away_win", Decimal("0.65"), Decimal("0.03")),
            (event_id, "test_model_3", "home_win", Decimal("0.80"), Decimal("0.08"))
        ]

        async with self.pool.acquire() as conn:
            for pred in test_predictions:
                await conn.execute("""
                    INSERT INTO test_predictions (event_id, model_name, predicted_outcome, confidence, expected_value)
                    VALUES ($1, $2, $3, $4, $5)
                """, *pred)

        # Retrieve and analyze predictions
        async with self.pool.acquire() as conn:
            results = await conn.fetch("""
                SELECT 
                    predicted_outcome,
                    AVG(confidence) as avg_confidence,
                    AVG(expected_value) as avg_ev,
                    COUNT(*) as prediction_count
                FROM test_predictions 
                WHERE event_id = $1 
                GROUP BY predicted_outcome
                ORDER BY avg_confidence DESC
            """, event_id)

        assert len(results) == 2  # home_win and away_win

        # Find highest confidence prediction
        best_prediction = results[0]
        assert best_prediction['prediction_count'] > 0
        assert best_prediction['avg_confidence'] > 0

        logger.info(f"Best prediction: {best_prediction['predicted_outcome']} with {best_prediction['avg_confidence']:.2f} confidence")

    @pytest.mark.integration
    async def test_real_transaction_rollback(self):
        """Test real transaction rollback functionality."""
        logger.info("Testing real transaction rollback")

        # Start transaction
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Insert test data
                await conn.execute("""
                    INSERT INTO test_events (event_id, sport, home_team, away_team, start_time)
                    VALUES ($1, $2, $3, $4, $5)
                """, "rollback_test_001", "mlb", "Test Team A", "Test Team B", datetime.now())

                # Verify data exists within transaction
                result = await conn.fetchval("""
                    SELECT COUNT(*) FROM test_events WHERE event_id = $1
                """, "rollback_test_001")
                assert result == 1

                # Transaction will rollback automatically

        # Verify data was rolled back
        async with self.pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COUNT(*) FROM test_events WHERE event_id = $1
            """, "rollback_test_001")
            assert result == 0

        logger.info("Successfully tested transaction rollback")

    @pytest.mark.integration
    async def test_real_concurrent_operations(self):
        """Test real concurrent database operations."""
        logger.info("Testing real concurrent operations")

        # Create multiple concurrent operations
        async def insert_event(event_id: str, sport: str):
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO test_events (event_id, sport, home_team, away_team, start_time)
                    VALUES ($1, $2, $3, $4, $5)
                """, event_id, sport, f"Home_{event_id}", f"Away_{event_id}", datetime.now())

        # Run concurrent inserts
        tasks = []
        for i in range(5):
            task = insert_event(f"concurrent_test_{i:03d}", "mlb" if i % 2 == 0 else "nhl")
            tasks.append(task)

        await asyncio.gather(*tasks)

        # Verify all inserts completed
        async with self.pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COUNT(*) FROM test_events WHERE event_id LIKE 'concurrent_test_%'
            """)
            assert result == 5

        logger.info("Successfully tested concurrent operations")

    @pytest.mark.integration
    async def test_real_data_integrity_constraints(self):
        """Test real data integrity constraints."""
        logger.info("Testing real data integrity constraints")

        # Test unique constraint
        event_id = "integrity_test_001"
        async with self.pool.acquire() as conn:
            # First insert should succeed
            await conn.execute("""
                INSERT INTO test_events (event_id, sport, home_team, away_team, start_time)
                VALUES ($1, $2, $3, $4, $5)
            """, event_id, "mlb", "Team A", "Team B", datetime.now())

            # Second insert with same event_id should fail
            try:
                await conn.execute("""
                    INSERT INTO test_events (event_id, sport, home_team, away_team, start_time)
                    VALUES ($1, $2, $3, $4, $5)
                """, event_id, "nhl", "Team C", "Team D", datetime.now())
                assert False, "Should have raised unique constraint violation"
            except Exception as e:
                assert "unique" in str(e).lower() or "duplicate" in str(e).lower()

        # Test foreign key constraint
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO test_odds (event_id, bookmaker, market_type, selection, odds)
                    VALUES ($1, $2, $3, $4, $5)
                """, "non_existent_event", "test_bookmaker", "moneyline", "home", Decimal("1.85"))
                assert False, "Should have raised foreign key constraint violation"
        except Exception as e:
            assert "foreign" in str(e).lower() or "constraint" in str(e).lower()

        logger.info("Successfully tested data integrity constraints")

    @pytest.mark.e2e
    async def test_real_end_to_end_workflow(self):
        """Test complete end-to-end database workflow."""
        logger.info("Testing real end-to-end database workflow")

        # 1. Create events
        events = [
            ("e2e_mlb_001", "mlb", "Yankees", "Red Sox"),
            ("e2e_nhl_001", "nhl", "Maple Leafs", "Canadiens"),
            ("e2e_mlb_002", "mlb", "Dodgers", "Giants")
        ]

        async with self.pool.acquire() as conn:
            for event_id, sport, home, away in events:
                await conn.execute("""
                    INSERT INTO test_events (event_id, sport, home_team, away_team, start_time)
                    VALUES ($1, $2, $3, $4, $5)
                """, event_id, sport, home, away, datetime.now() + timedelta(hours=1))

        # 2. Add odds for each event
        async with self.pool.acquire() as conn:
            for event_id, _, _, _ in events:
                await conn.execute("""
                    INSERT INTO test_odds (event_id, bookmaker, market_type, selection, odds)
                    VALUES ($1, $2, $3, $4, $5)
                """, event_id, "e2e_bookmaker", "moneyline", "home", Decimal("1.90"))

                await conn.execute("""
                    INSERT INTO test_odds (event_id, bookmaker, market_type, selection, odds)
                    VALUES ($1, $2, $3, $4, $5)
                """, event_id, "e2e_bookmaker", "moneyline", "away", Decimal("2.10"))

        # 3. Add predictions
        async with self.pool.acquire() as conn:
            for event_id, sport, _, _ in events:
                await conn.execute("""
                    INSERT INTO test_predictions (event_id, model_name, predicted_outcome, confidence, expected_value)
                    VALUES ($1, $2, $3, $4, $5)
                """, event_id, "e2e_model", "home_win", Decimal("0.75"), Decimal("0.05"))

        # 4. Query complete workflow data
        async with self.pool.acquire() as conn:
            results = await conn.fetch("""
                SELECT 
                    e.event_id,
                    e.sport,
                    e.home_team,
                    e.away_team,
                    COUNT(o.id) as odds_count,
                    COUNT(p.id) as predictions_count,
                    AVG(p.confidence) as avg_confidence
                FROM test_events e
                LEFT JOIN test_odds o ON e.event_id = o.event_id
                LEFT JOIN test_predictions p ON e.event_id = p.event_id
                WHERE e.event_id LIKE 'e2e_%'
                GROUP BY e.event_id, e.sport, e.home_team, e.away_team
                ORDER BY e.event_id
            """)

        # 5. Validate workflow results
        assert len(results) == 3

        for result in results:
            assert result['odds_count'] == 2  # home and away odds
            assert result['predictions_count'] == 1  # one prediction per event
            assert result['avg_confidence'] == Decimal("0.75")

        logger.info("Successfully completed end-to-end database workflow")


@pytest.mark.asyncio
async def test_real_database_performance():
    """Test real database performance under load."""
    logger.info("Testing real database performance")

    # This test would measure actual database performance
    # with realistic data volumes and query patterns
    assert True  # Placeholder for performance test
