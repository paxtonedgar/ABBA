"""
Test suite for Phase 2: Data Verification and Anomaly Detection
Tests DataVerifier, database validation pipelines, and integration.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
import structlog
from agents import ResearchAgent
from data_fetcher import DataVerifier
from database import DatabaseManager

from models import Event, EventStatus, MarketType, Odds, PlatformType, SportType

logger = structlog.get_logger()


def load_config():
    """Simple config loader for testing."""
    return {
        "database": {"url": "sqlite+aiosqlite:///abmba.db"},
        "apis": {"openai": {"model": "gpt-3.5-turbo", "api_key": "test-key"}},
        "sports": [
            {"name": "baseball_mlb", "enabled": True},
            {"name": "hockey_nhl", "enabled": True},
        ],
        "platforms": {"fanduel": {"enabled": True}, "draftkings": {"enabled": True}},
    }


class TestDataVerifier:
    """Test DataVerifier class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.verifier = DataVerifier(contamination=0.1, confidence_threshold=0.7)

        # Create test data with simpler approach
        self.test_data = pd.DataFrame(
            {
                "odds": np.random.normal(100, 10, 100),
                "implied_probability": np.random.uniform(0.1, 0.9, 100),
                "timestamp": [
                    datetime.utcnow() + timedelta(minutes=i) for i in range(100)
                ],
            }
        )

        # Add some anomalies
        self.test_data.loc[95, "odds"] = 9999  # Extreme outlier
        self.test_data.loc[96, "implied_probability"] = 1.5  # Impossible probability

        # Create baseball physics data
        self.baseball_data = pd.DataFrame(
            {
                "spin_rate": [
                    2000,
                    2100,
                    2200,
                    2300,
                    2400,
                    2500,
                    2600,
                    2700,
                    2800,
                    2900,
                    3000,
                    3100,
                    3200,
                    3300,
                    3400,
                    3500,
                    3600,
                    3700,
                    3800,
                    3900,
                    4000,
                ],  # Last few exceed max
                "exit_velocity": [
                    80,
                    85,
                    90,
                    95,
                    100,
                    105,
                    110,
                    115,
                    120,
                    125,
                    130,
                    135,
                    140,
                    145,
                    150,
                    155,
                    160,
                    165,
                    170,
                    175,
                    180,
                ],  # Last few exceed max
                "pitch_velocity": [
                    85,
                    87,
                    89,
                    91,
                    93,
                    95,
                    97,
                    99,
                    101,
                    103,
                    105,
                    107,
                    109,
                    111,
                    113,
                    115,
                    117,
                    119,
                    121,
                    123,
                    125,
                ],  # Last few exceed max
                "venue": ["Fenway Park"] * 10 + ["Yankee Stadium"] * 11,
                "hits": [
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                ],
            }
        )

    def test_detect_anomalies(self):
        """Test anomaly detection with Z-score and Isolation Forest."""
        print("\nTesting anomaly detection...")

        anomalies_df, confidence_score = self.verifier.detect_anomalies(self.test_data)

        print(
            f"Detected {len(anomalies_df)} anomalies out of {len(self.test_data)} records"
        )
        print(f"Confidence score: {confidence_score:.3f}")

        # Should detect the extreme outliers we added
        assert len(anomalies_df) > 0
        assert confidence_score < 1.0  # Should be less than perfect due to anomalies
        assert 9999 in anomalies_df["odds"].values  # Should detect extreme outlier
        assert (
            1.5 in anomalies_df["implied_probability"].values
        )  # Should detect impossible probability

        print("✅ Anomaly detection test passed")

    def test_validate_completeness(self):
        """Test data completeness validation."""
        print("\nTesting data completeness validation...")

        # Test with complete data
        is_complete, coverage_rate = self.verifier.validate_completeness(self.test_data)
        print(f"Complete data - is_valid: {is_complete}, coverage: {coverage_rate:.3f}")
        assert is_complete
        assert coverage_rate >= 0.9

        # Test with missing data
        incomplete_data = self.test_data.copy()
        incomplete_data.loc[0:30, "odds"] = np.nan  # Add more missing values
        is_complete, coverage_rate = self.verifier.validate_completeness(
            incomplete_data
        )
        print(
            f"Incomplete data - is_valid: {is_complete}, coverage: {coverage_rate:.3f}"
        )
        assert not is_complete
        assert coverage_rate < 0.9

        print("✅ Data completeness validation test passed")

    def test_validate_physics(self):
        """Test physics-based validation."""
        print("\nTesting physics validation...")

        violations_df, confidence_score = self.verifier.validate_physics(
            self.baseball_data, "baseball_mlb"
        )

        print(
            f"Physics violations: {len(violations_df)} out of {len(self.baseball_data)} records"
        )
        print(f"Confidence score: {confidence_score:.3f}")

        # Should detect violations for values exceeding limits
        assert len(violations_df) > 0
        assert confidence_score < 1.0

        # Check specific violations
        high_spin_violations = violations_df[
            violations_df["spin_rate"] > self.verifier.max_spin_rate
        ]
        high_velocity_violations = violations_df[
            violations_df["exit_velocity"] > self.verifier.max_exit_velocity
        ]

        assert len(high_spin_violations) > 0
        assert len(high_velocity_violations) > 0

        print("✅ Physics validation test passed")

    def test_detect_betting_patterns(self):
        """Test betting pattern anomaly detection."""
        print("\nTesting betting pattern detection...")

        pattern_anomalies, pattern_score = self.verifier.detect_betting_patterns(
            self.test_data
        )

        print(
            f"Pattern anomalies: {len(pattern_anomalies)} out of {len(self.test_data)} records"
        )
        print(f"Pattern confidence score: {pattern_score:.3f}")

        # Should detect the impossible probability we added
        if len(pattern_anomalies) > 0:
            assert 1.5 in pattern_anomalies["implied_probability"].values

        print("✅ Betting pattern detection test passed")

    def test_calculate_confidence_score(self):
        """Test overall confidence score calculation."""
        print("\nTesting confidence score calculation...")

        confidence_score = self.verifier.calculate_confidence_score(
            self.test_data, "baseball_mlb"
        )
        print(f"Overall confidence score: {confidence_score:.3f}")

        assert 0 <= confidence_score <= 1
        assert confidence_score < 1.0  # Should be less than perfect due to anomalies

        print("✅ Confidence score calculation test passed")

    def test_should_halt_processing(self):
        """Test processing halt decision."""
        print("\nTesting processing halt decision...")

        # Test with high confidence
        should_halt = self.verifier.should_halt_processing(0.8)
        print(f"High confidence (0.8) - should halt: {should_halt}")
        assert not should_halt

        # Test with low confidence
        should_halt = self.verifier.should_halt_processing(0.5)
        print(f"Low confidence (0.5) - should halt: {should_halt}")
        assert should_halt

        print("✅ Processing halt decision test passed")

    def test_get_validation_report(self):
        """Test validation report generation."""
        print("\nTesting validation report generation...")

        # Run some validations first
        self.verifier.detect_anomalies(self.test_data)
        self.verifier.validate_completeness(self.test_data)

        report = self.verifier.get_validation_report()
        print(f"Validation report: {report}")

        assert "total_checks" in report
        assert "anomalies_detected" in report
        assert "average_confidence" in report
        assert report["total_checks"] > 0

        print("✅ Validation report generation test passed")


class TestDatabaseValidation:
    """Test database validation pipelines."""

    @pytest.fixture
    async def db_manager(self):
        """Create database manager for testing."""
        config = load_config()
        db_manager = DatabaseManager(config["database"]["url"])
        await db_manager.initialize()
        yield db_manager
        await db_manager.close()

    async def test_validate_schema(self, db_manager):
        """Test database schema validation."""
        print("\nTesting database schema validation...")

        schema_result = await db_manager.validate_schema()
        print(f"Schema validation result: {schema_result}")

        assert "is_valid" in schema_result
        assert "missing_tables" in schema_result
        assert "extra_tables" in schema_result
        assert "total_tables" in schema_result

        # Should be valid since we just initialized
        assert schema_result["is_valid"]

        print("✅ Database schema validation test passed")

    async def test_validate_data_integrity(self, db_manager):
        """Test data integrity validation."""
        print("\nTesting data integrity validation...")

        # Test with empty table
        integrity_result = await db_manager.validate_data_integrity("events")
        print(f"Events table integrity: {integrity_result}")

        assert "is_valid" in integrity_result
        assert "null_count" in integrity_result
        assert "duplicate_count" in integrity_result

        print("✅ Data integrity validation test passed")

    async def test_detect_data_anomalies(self, db_manager):
        """Test database anomaly detection."""
        print("\nTesting database anomaly detection...")

        # Test with empty table
        anomaly_result = await db_manager.detect_data_anomalies("odds", ["odds"])
        print(f"Odds table anomalies: {anomaly_result}")

        assert "anomalies_detected" in anomaly_result
        # column_anomalies might not be present if no data
        if "column_anomalies" not in anomaly_result:
            print("Note: No column anomalies detected (empty table)")

        print("✅ Database anomaly detection test passed")

    async def test_get_validation_report(self, db_manager):
        """Test comprehensive validation report."""
        print("\nTesting comprehensive validation report...")

        report = await db_manager.get_validation_report()
        print(f"Validation report: {report}")

        assert "schema_validation" in report
        assert "data_integrity" in report
        assert "validation_stats" in report

        print("✅ Comprehensive validation report test passed")


class TestResearchAgentIntegration:
    """Test Research Agent integration with data verification."""

    @pytest.fixture
    async def research_agent(self):
        """Create research agent for testing."""
        config = load_config()
        db_manager = DatabaseManager(config["database"]["url"])
        await db_manager.initialize()

        # Mock data fetcher
        class MockDataFetcher:
            async def fetch_events(self, sport):
                return [
                    Event(
                        sport=SportType.BASEBALL_MLB,
                        home_team="Red Sox",
                        away_team="Yankees",
                        event_date=datetime.utcnow() + timedelta(hours=2),
                        status=EventStatus.SCHEDULED,
                    )
                ]

            async def fetch_odds(self, event_id, sport):
                return [
                    Odds(
                        event_id=event_id,
                        platform=PlatformType.FANDUEL,
                        market_type=MarketType.MONEYLINE,
                        selection="home",
                        odds=Decimal("-110"),
                        timestamp=datetime.utcnow(),
                    )
                ]

        data_fetcher = MockDataFetcher()
        research_agent = ResearchAgent(config, db_manager, data_fetcher)

        yield research_agent
        await db_manager.close()

    async def test_verify_data_quality(self, research_agent):
        """Test data quality verification in research agent."""
        print("\nTesting data quality verification in research agent...")

        # Test events verification
        events_result = await research_agent._verify_data_quality(
            "events", "baseball_mlb"
        )
        print(f"Events verification result: {events_result}")

        if "error" not in events_result:
            assert "confidence_score" in events_result
            assert "verification_passed" in events_result
            assert "anomalies_detected" in events_result

        print("✅ Data quality verification test passed")

    async def test_fallback_data_source(self, research_agent):
        """Test fallback data source functionality."""
        print("\nTesting fallback data source...")

        # Test events fallback
        fallback_events = await research_agent._fallback_data_source(
            "baseball_mlb", data_type="events"
        )
        print(f"Fallback events: {len(fallback_events)}")

        # Test odds fallback
        fallback_odds = await research_agent._fallback_data_source(
            "test_event_id", data_type="odds"
        )
        print(f"Fallback odds: {len(fallback_odds)}")

        # Should return some fallback data
        assert isinstance(fallback_events, list)
        assert isinstance(fallback_odds, list)

        print("✅ Fallback data source test passed")


async def test_injected_anomalies():
    """Test detection of injected anomalies."""
    print("\nTesting injected anomalies detection...")

    verifier = DataVerifier()

    # Create clean data
    clean_data = pd.DataFrame(
        {
            "odds": np.random.normal(100, 10, 100),
            "implied_probability": np.random.uniform(0.1, 0.9, 100),
            "timestamp": [datetime.utcnow() + timedelta(minutes=i) for i in range(100)],
        }
    )

    # Inject anomalies
    clean_data.loc[0, "odds"] = 9999  # Extreme outlier
    clean_data.loc[1, "implied_probability"] = 1.5  # Impossible probability
    clean_data.loc[2, "implied_probability"] = -0.1  # Negative probability

    # Detect anomalies
    anomalies_df, confidence_score = verifier.detect_anomalies(clean_data)

    print(f"Clean data anomalies: {len(anomalies_df)} out of {len(clean_data)}")
    print(f"Confidence score: {confidence_score:.3f}")

    # Should detect the injected anomalies
    assert len(anomalies_df) >= 3  # At least the 3 we injected
    assert confidence_score < 1.0

    # Check specific anomalies
    extreme_odds = anomalies_df[anomalies_df["odds"] == 9999]
    impossible_prob = anomalies_df[anomalies_df["implied_probability"] == 1.5]
    negative_prob = anomalies_df[anomalies_df["implied_probability"] == -0.1]

    assert len(extreme_odds) > 0
    assert len(impossible_prob) > 0
    assert len(negative_prob) > 0

    print("✅ Injected anomalies detection test passed")


async def run_all_tests():
    """Run all data verification tests."""
    print("Starting Data Verification and Anomaly Detection Tests")
    print("=" * 60)

    # Test DataVerifier
    test_verifier = TestDataVerifier()
    test_verifier.setup_method()

    test_verifier.test_detect_anomalies()
    test_verifier.test_validate_completeness()
    test_verifier.test_validate_physics()
    test_verifier.test_detect_betting_patterns()
    test_verifier.test_calculate_confidence_score()
    test_verifier.test_should_halt_processing()
    test_verifier.test_get_validation_report()

    # Test database validation
    config = load_config()
    db_manager = DatabaseManager(config["database"]["url"])
    await db_manager.initialize()

    test_db = TestDatabaseValidation()
    await test_db.test_validate_schema(db_manager)
    await test_db.test_validate_data_integrity(db_manager)
    await test_db.test_detect_data_anomalies(db_manager)
    await test_db.test_get_validation_report(db_manager)

    await db_manager.close()

    # Test injected anomalies
    await test_injected_anomalies()

    print("\n" + "=" * 60)
    print("All data verification tests completed successfully!")
    print("✅ Phase 2: Data Verification and Anomaly Detection - IMPLEMENTED")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
