"""
Test suite for ABMBA system.
Includes unit tests, integration tests, and simulation tests.
"""

import os
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from data_fetcher import DataValidator
from database import DatabaseManager
from simulations import (
    KellyCriterion,
    MLPredictor,
    MonteCarloSimulator,
    SimulationManager,
    StatisticalAnalyzer,
)
from utils import ConfigManager, calculate_roi, format_currency, format_percentage

from models import (
    BankrollLog,
    Bet,
    Event,
    MarketType,
    Odds,
    PlatformType,
    SportType,
    SystemMetrics,
)


class TestModels:
    """Test data models."""

    def test_event_creation(self):
        """Test Event model creation."""
        event = Event(
            sport=SportType.BASKETBALL_NBA,
            home_team="Lakers",
            away_team="Warriors",
            event_date=datetime.utcnow() + timedelta(days=1)
        )

        assert event.sport == SportType.BASKETBALL_NBA
        assert event.home_team == "Lakers"
        assert event.away_team == "Warriors"
        assert event.status.value == "scheduled"

    def test_odds_creation(self):
        """Test Odds model creation."""
        odds = Odds(
            event_id="test_event_1",
            platform=PlatformType.FANDUEL,
            market_type=MarketType.MONEYLINE,
            selection="home",
            odds=Decimal("-110")
        )

        assert odds.event_id == "test_event_1"
        assert odds.platform == PlatformType.FANDUEL
        assert odds.market_type == MarketType.MONEYLINE
        assert odds.selection == "home"
        assert odds.odds == Decimal("-110")
        assert odds.implied_probability is not None

    def test_bet_creation(self):
        """Test Bet model creation."""
        bet = Bet(
            event_id="test_event_1",
            platform=PlatformType.FANDUEL,
            market_type=MarketType.MONEYLINE,
            selection="home",
            odds=Decimal("-110"),
            stake=Decimal("10.00"),
            expected_value=Decimal("0.05"),
            kelly_fraction=Decimal("0.02")
        )

        assert bet.event_id == "test_event_1"
        assert bet.stake == Decimal("10.00")
        assert bet.potential_win > 0
        assert bet.status.value == "pending"


class TestMonteCarloSimulator:
    """Test Monte Carlo simulation."""

    def setup_method(self):
        """Setup test method."""
        self.simulator = MonteCarloSimulator(iterations=1000)

    def test_moneyline_simulation(self):
        """Test moneyline simulation."""
        result = self.simulator.simulate_moneyline(-110, -110)

        assert 'win_probability' in result
        assert 'expected_value' in result
        assert 'variance' in result
        assert 0 <= result['win_probability'] <= 1
        assert isinstance(result['expected_value'], float)
        assert isinstance(result['variance'], float)

    def test_spread_simulation(self):
        """Test spread simulation."""
        result = self.simulator.simulate_spread(-3.5, -110, -110)

        assert 'win_probability' in result
        assert 'expected_value' in result
        assert 'variance' in result
        assert 0 <= result['win_probability'] <= 1

    def test_totals_simulation(self):
        """Test totals simulation."""
        result = self.simulator.simulate_totals(220.5, -110, -110)

        assert 'win_probability' in result
        assert 'expected_value' in result
        assert 'variance' in result
        assert 0 <= result['win_probability'] <= 1


class TestKellyCriterion:
    """Test Kelly Criterion calculations."""

    def test_kelly_fraction_calculation(self):
        """Test Kelly fraction calculation."""
        # Test with positive expected value
        kelly_fraction = KellyCriterion.calculate_kelly_fraction(
            win_prob=0.6, odds=-110, fraction=0.5
        )

        assert kelly_fraction > 0
        assert kelly_fraction <= 0.5  # Should be half-Kelly

    def test_kelly_fraction_negative_ev(self):
        """Test Kelly fraction with negative expected value."""
        kelly_fraction = KellyCriterion.calculate_kelly_fraction(
            win_prob=0.4, odds=-110, fraction=0.5
        )

        assert kelly_fraction == 0  # Should be 0 for negative EV

    def test_optimal_stake_calculation(self):
        """Test optimal stake calculation."""
        stake = KellyCriterion.calculate_optimal_stake(
            bankroll=1000, win_prob=0.6, odds=-110, max_risk_percent=0.02
        )

        assert stake > 0
        assert stake <= 20  # Should not exceed 2% of bankroll


class TestStatisticalAnalyzer:
    """Test statistical analysis functions."""

    def test_volatility_calculation(self):
        """Test volatility calculation."""
        returns = [0.01, -0.02, 0.03, -0.01, 0.02]
        volatility = StatisticalAnalyzer.calculate_volatility(returns)

        assert isinstance(volatility, float)
        assert volatility >= 0

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        returns = [0.01, -0.02, 0.03, -0.01, 0.02]
        sharpe = StatisticalAnalyzer.calculate_sharpe_ratio(returns)

        assert isinstance(sharpe, float)

    def test_var_calculation(self):
        """Test Value at Risk calculation."""
        returns = [0.01, -0.02, 0.03, -0.01, 0.02]
        var = StatisticalAnalyzer.calculate_var(returns, confidence_level=0.95)

        assert isinstance(var, float)

    def test_stationarity_test(self):
        """Test stationarity test."""
        time_series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = StatisticalAnalyzer.test_stationarity(time_series)

        assert 'is_stationary' in result
        assert 'p_value' in result
        assert isinstance(result['is_stationary'], bool)


class TestDataValidator:
    """Test data validation functions."""

    def test_event_validation(self):
        """Test event validation."""
        valid_event = Event(
            sport=SportType.BASKETBALL_NBA,
            home_team="Lakers",
            away_team="Warriors",
            event_date=datetime.utcnow() + timedelta(days=1)
        )

        assert DataValidator.validate_event(valid_event) == True

    def test_event_validation_invalid(self):
        """Test event validation with invalid data."""
        invalid_event = Event(
            sport=SportType.BASKETBALL_NBA,
            home_team="",  # Empty team name
            away_team="Warriors",
            event_date=datetime.utcnow() + timedelta(days=1)
        )

        assert DataValidator.validate_event(invalid_event) == False

    def test_odds_validation(self):
        """Test odds validation."""
        valid_odds = Odds(
            event_id="test_event_1",
            platform=PlatformType.FANDUEL,
            market_type=MarketType.MONEYLINE,
            selection="home",
            odds=Decimal("-110")
        )

        assert DataValidator.validate_odds(valid_odds) == True

    def test_odds_validation_invalid(self):
        """Test odds validation with invalid data."""
        invalid_odds = Odds(
            event_id="",  # Empty event ID
            platform=PlatformType.FANDUEL,
            market_type=MarketType.MONEYLINE,
            selection="home",
            odds=Decimal("-110")
        )

        assert DataValidator.validate_odds(invalid_odds) == False

    def test_team_name_cleaning(self):
        """Test team name cleaning."""
        assert DataValidator.clean_team_name("LA Lakers") == "Los Angeles Lakers"
        assert DataValidator.clean_team_name("  Lakers  ") == "Lakers"
        assert DataValidator.clean_team_name("Lakers") == "Lakers"


class TestUtils:
    """Test utility functions."""

    def test_format_currency(self):
        """Test currency formatting."""
        assert format_currency(100.50) == "$100.50"
        assert format_currency(0) == "$0.00"
        assert format_currency(-50.25) == "$-50.25"

    def test_format_percentage(self):
        """Test percentage formatting."""
        assert format_percentage(0.05) == "5.00%"
        assert format_percentage(0.1234) == "12.34%"
        assert format_percentage(0) == "0.00%"

    def test_calculate_roi(self):
        """Test ROI calculation."""
        assert calculate_roi(100, 110) == 10.0
        assert calculate_roi(100, 90) == -10.0
        assert calculate_roi(0, 100) == 0.0


@pytest.mark.asyncio
class TestDatabaseManager:
    """Test database manager."""

    async def setup_method(self):
        """Setup test method."""
        # Use in-memory SQLite for testing
        self.db_manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
        await self.db_manager.initialize()

    async def test_save_and_get_event(self):
        """Test saving and retrieving events."""
        event = Event(
            sport=SportType.BASKETBALL_NBA,
            home_team="Lakers",
            away_team="Warriors",
            event_date=datetime.utcnow() + timedelta(days=1)
        )

        # Save event
        event_id = await self.db_manager.save_event(event)
        assert event_id == event.id

        # Get events
        events = await self.db_manager.get_events()
        assert len(events) == 1
        assert events[0].home_team == "Lakers"
        assert events[0].away_team == "Warriors"

    async def test_save_and_get_odds(self):
        """Test saving and retrieving odds."""
        odds = Odds(
            event_id="test_event_1",
            platform=PlatformType.FANDUEL,
            market_type=MarketType.MONEYLINE,
            selection="home",
            odds=Decimal("-110")
        )

        # Save odds
        odds_id = await self.db_manager.save_odds(odds)
        assert odds_id == odds.id

        # Get odds
        odds_list = await self.db_manager.get_latest_odds("test_event_1")
        assert len(odds_list) == 1
        assert odds_list[0].odds == Decimal("-110")

    async def test_bankroll_operations(self):
        """Test bankroll operations."""
        # Initial bankroll should be 0
        initial_bankroll = await self.db_manager.get_current_bankroll()
        assert initial_bankroll == Decimal('0')

        # Add bankroll log
        log = BankrollLog(
            balance=Decimal('100.00'),
            change=Decimal('100.00'),
            description="Initial deposit",
            source="deposit"
        )

        await self.db_manager.save_bankroll_log(log)

        # Check current bankroll
        current_bankroll = await self.db_manager.get_current_bankroll()
        assert current_bankroll == Decimal('100.00')

    async def test_system_metrics(self):
        """Test system metrics calculation."""
        metrics = await self.db_manager.get_system_metrics()

        assert isinstance(metrics, SystemMetrics)
        assert metrics.total_bets == 0
        assert metrics.winning_bets == 0
        assert metrics.losing_bets == 0
        assert metrics.win_rate == Decimal('0')
        assert metrics.total_profit_loss == Decimal('0')
        assert metrics.roi_percentage == Decimal('0')


@pytest.mark.asyncio
class TestSimulationManager:
    """Test simulation manager."""

    async def setup_method(self):
        """Setup test method."""
        self.db_manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
        await self.db_manager.initialize()

        self.config = {
            'simulation': {
                'monte_carlo_iterations': 1000
            },
            'bankroll': {
                'kelly_fraction': 0.5
            }
        }

        self.simulation_manager = SimulationManager(self.db_manager, self.config)

    async def test_event_simulation(self):
        """Test event simulation."""
        # Create test event
        event = Event(
            sport=SportType.BASKETBALL_NBA,
            home_team="Lakers",
            away_team="Warriors",
            event_date=datetime.utcnow() + timedelta(days=1)
        )

        # Create test odds
        odds_list = [
            Odds(
                event_id=event.id,
                platform=PlatformType.FANDUEL,
                market_type=MarketType.MONEYLINE,
                selection="home",
                odds=Decimal("-110")
            ),
            Odds(
                event_id=event.id,
                platform=PlatformType.FANDUEL,
                market_type=MarketType.MONEYLINE,
                selection="away",
                odds=Decimal("-110")
            )
        ]

        # Run simulation
        result = await self.simulation_manager.run_event_simulation(event, odds_list)

        assert result is not None
        assert result.event_id == event.id
        assert result.iterations == 1000
        assert 0 <= float(result.win_probability) <= 1
        assert isinstance(result.expected_value, Decimal)
        assert isinstance(result.kelly_fraction, Decimal)
        assert result.risk_level in ['low', 'medium', 'high']


class TestMLPredictor:
    """Test machine learning predictor."""

    def setup_method(self):
        """Setup test method."""
        self.predictor = MLPredictor('random_forest')

    def test_model_initialization(self):
        """Test model initialization."""
        assert self.predictor.model is not None
        assert self.predictor.model_type == 'random_forest'
        assert not self.predictor.is_trained

    def test_feature_preparation(self):
        """Test feature preparation."""
        # Create test events and odds
        events = [
            Event(
                sport=SportType.BASKETBALL_NBA,
                home_team="Lakers",
                away_team="Warriors",
                event_date=datetime.utcnow() + timedelta(days=1)
            )
        ]

        odds_data = [
            Odds(
                event_id=events[0].id,
                platform=PlatformType.FANDUEL,
                market_type=MarketType.MONEYLINE,
                selection="home",
                odds=Decimal("-110")
            )
        ]

        features = self.predictor.prepare_features(events, odds_data)

        assert len(features) == 1
        assert 'event_id' in features.columns
        assert 'sport' in features.columns
        assert 'home_team' in features.columns
        assert 'away_team' in features.columns


class TestConfigManager:
    """Test configuration manager."""

    def setup_method(self):
        """Setup test method."""
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")

        with open(self.config_path, 'w') as f:
            f.write("""
system:
  mode: "simulation"
  log_level: "INFO"
bankroll:
  initial_amount: 100.00
  max_risk_per_bet: 2.0
""")

    def teardown_method(self):
        """Teardown test method."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_config_loading(self):
        """Test configuration loading."""
        config_manager = ConfigManager(self.config_path)

        assert config_manager.get('system.mode') == 'simulation'
        assert config_manager.get('system.log_level') == 'INFO'
        assert config_manager.get('bankroll.initial_amount') == 100.00
        assert config_manager.get('bankroll.max_risk_per_bet') == 2.0

    def test_config_get_default(self):
        """Test configuration get with default value."""
        config_manager = ConfigManager(self.config_path)

        # Test non-existent key with default
        value = config_manager.get('non.existent.key', 'default_value')
        assert value == 'default_value'

    def test_config_set(self):
        """Test configuration setting."""
        config_manager = ConfigManager(self.config_path)

        config_manager.set('test.key', 'test_value')
        assert config_manager.get('test.key') == 'test_value'


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests."""

    async def setup_method(self):
        """Setup test method."""
        self.db_manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
        await self.db_manager.initialize()

        self.config = {
            'simulation': {
                'monte_carlo_iterations': 1000
            },
            'bankroll': {
                'kelly_fraction': 0.5,
                'initial_amount': 100.00
            }
        }

    async def test_full_betting_cycle(self):
        """Test a full betting cycle."""
        # 1. Create event
        event = Event(
            sport=SportType.BASKETBALL_NBA,
            home_team="Lakers",
            away_team="Warriors",
            event_date=datetime.utcnow() + timedelta(days=1)
        )
        await self.db_manager.save_event(event)

        # 2. Add odds
        odds_list = [
            Odds(
                event_id=event.id,
                platform=PlatformType.FANDUEL,
                market_type=MarketType.MONEYLINE,
                selection="home",
                odds=Decimal("-110")
            ),
            Odds(
                event_id=event.id,
                platform=PlatformType.FANDUEL,
                market_type=MarketType.MONEYLINE,
                selection="away",
                odds=Decimal("-110")
            )
        ]

        for odds in odds_list:
            await self.db_manager.save_odds(odds)

        # 3. Run simulation
        simulation_manager = SimulationManager(self.db_manager, self.config)
        simulation_result = await simulation_manager.run_event_simulation(event, odds_list)

        assert simulation_result is not None

        # 4. Create bet (if simulation shows positive EV)
        if float(simulation_result.expected_value) > 0.05:
            bet = Bet(
                event_id=event.id,
                platform=PlatformType.FANDUEL,
                market_type=MarketType.MONEYLINE,
                selection="home",
                odds=Decimal("-110"),
                stake=Decimal("10.00"),
                expected_value=simulation_result.expected_value,
                kelly_fraction=simulation_result.kelly_fraction
            )

            await self.db_manager.save_bet(bet)

            # 5. Update bankroll
            log = BankrollLog(
                balance=Decimal('90.00'),
                change=Decimal('-10.00'),
                bet_id=bet.id,
                description="Bet placed",
                source="bet"
            )
            await self.db_manager.save_bankroll_log(log)

            # 6. Check final state
            current_bankroll = await self.db_manager.get_current_bankroll()
            assert current_bankroll == Decimal('90.00')

    async def test_system_metrics_integration(self):
        """Test system metrics integration."""
        # Add some test data
        event = Event(
            sport=SportType.BASKETBALL_NBA,
            home_team="Lakers",
            away_team="Warriors",
            event_date=datetime.utcnow() + timedelta(days=1)
        )
        await self.db_manager.save_event(event)

        # Add a bet
        bet = Bet(
            event_id=event.id,
            platform=PlatformType.FANDUEL,
            market_type=MarketType.MONEYLINE,
            selection="home",
            odds=Decimal("-110"),
            stake=Decimal("10.00"),
            expected_value=Decimal("0.05"),
            kelly_fraction=Decimal("0.02"),
            status="won",
            profit_loss=Decimal("9.09")
        )
        await self.db_manager.save_bet(bet)

        # Add bankroll logs
        logs = [
            BankrollLog(
                balance=Decimal('100.00'),
                change=Decimal('100.00'),
                description="Initial deposit",
                source="deposit"
            ),
            BankrollLog(
                balance=Decimal('90.00'),
                change=Decimal('-10.00'),
                bet_id=bet.id,
                description="Bet placed",
                source="bet"
            ),
            BankrollLog(
                balance=Decimal('99.09'),
                change=Decimal('9.09'),
                bet_id=bet.id,
                description="Bet won",
                source="bet"
            )
        ]

        for log in logs:
            await self.db_manager.save_bankroll_log(log)

        # Get metrics
        metrics = await self.db_manager.get_system_metrics()

        assert metrics.total_bets == 1
        assert metrics.winning_bets == 1
        assert metrics.losing_bets == 0
        assert metrics.win_rate == Decimal('1.0')
        assert metrics.total_profit_loss == Decimal('9.09')
        assert metrics.current_bankroll == Decimal('99.09')


def run_performance_tests():
    """Run performance tests."""
    import time

    print("Running performance tests...")

    # Test Monte Carlo simulation performance
    start_time = time.time()
    simulator = MonteCarloSimulator(iterations=10000)
    result = simulator.simulate_moneyline(-110, -110)
    end_time = time.time()

    print(f"Monte Carlo simulation (10k iterations): {end_time - start_time:.3f}s")
    assert end_time - start_time < 5.0  # Should complete within 5 seconds

    # Test Kelly Criterion performance
    start_time = time.time()
    for _ in range(1000):
        KellyCriterion.calculate_kelly_fraction(0.6, -110, 0.5)
    end_time = time.time()

    print(f"Kelly Criterion (1000 calculations): {end_time - start_time:.3f}s")
    assert end_time - start_time < 1.0  # Should complete within 1 second


if __name__ == "__main__":
    # Run performance tests
    run_performance_tests()

    # Run pytest
    pytest.main([__file__, "-v"])
