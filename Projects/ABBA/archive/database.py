"""
Database layer for ABMBA system using SQLAlchemy for data persistence.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy.stats import zscore
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Integer,
    Numeric,
    String,
    Text,
    create_engine,
    inspect,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import text

from models import Alert, BankrollLog, Bet, Event, Odds, SystemMetrics

logger = structlog.get_logger()
Base = declarative_base()


class EventTable(Base):
    """SQLAlchemy model for events table."""
    __tablename__ = "events"

    id = Column(String, primary_key=True)
    sport = Column(String, nullable=False)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    event_date = Column(DateTime, nullable=False)
    status = Column(String, default="scheduled")
    home_score = Column(Integer)
    away_score = Column(Integer)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class OddsTable(Base):
    """SQLAlchemy model for odds table."""
    __tablename__ = "odds"

    id = Column(String, primary_key=True)
    event_id = Column(String, nullable=False)
    platform = Column(String, nullable=False)
    market_type = Column(String, nullable=False)
    selection = Column(String, nullable=False)
    odds = Column(Numeric(10, 2), nullable=False)
    line = Column(Numeric(10, 2))
    implied_probability = Column(Numeric(10, 4))
    timestamp = Column(DateTime, nullable=False)


class BetTable(Base):
    """SQLAlchemy model for bets table."""
    __tablename__ = "bets"

    id = Column(String, primary_key=True)
    event_id = Column(String, nullable=False)
    platform = Column(String, nullable=False)
    market_type = Column(String, nullable=False)
    selection = Column(String, nullable=False)
    odds = Column(Numeric(10, 2), nullable=False)
    stake = Column(Numeric(10, 2), nullable=False)
    potential_win = Column(Numeric(10, 2), nullable=False)
    expected_value = Column(Numeric(10, 4), nullable=False)
    kelly_fraction = Column(Numeric(10, 4), nullable=False)
    status = Column(String, default="pending")
    placed_at = Column(DateTime)
    settled_at = Column(DateTime)
    result = Column(String)
    profit_loss = Column(Numeric(10, 2))
    created_at = Column(DateTime, nullable=False)


class BankrollLogTable(Base):
    """SQLAlchemy model for bankroll logs table."""
    __tablename__ = "bankroll_logs"

    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    balance = Column(Numeric(10, 2), nullable=False)
    change = Column(Numeric(10, 2), nullable=False)
    bet_id = Column(String)
    description = Column(Text, nullable=False)
    source = Column(String, nullable=False)


class SimulationResultTable(Base):
    """SQLAlchemy model for simulation results table."""
    __tablename__ = "simulation_results"

    id = Column(String, primary_key=True)
    event_id = Column(String, nullable=False)
    iterations = Column(Integer, nullable=False)
    win_probability = Column(Numeric(10, 4), nullable=False)
    expected_value = Column(Numeric(10, 4), nullable=False)
    variance = Column(Numeric(10, 6), nullable=False)
    confidence_interval_lower = Column(Numeric(10, 4), nullable=False)
    confidence_interval_upper = Column(Numeric(10, 4), nullable=False)
    kelly_fraction = Column(Numeric(10, 4), nullable=False)
    recommended_stake = Column(Numeric(10, 2), nullable=False)
    risk_level = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)


class ModelPredictionTable(Base):
    """SQLAlchemy model for model predictions table."""
    __tablename__ = "model_predictions"

    id = Column(String, primary_key=True)
    event_id = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    prediction = Column(String, nullable=False)
    confidence = Column(Numeric(10, 4), nullable=False)
    features = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False)


class ArbitrageOpportunityTable(Base):
    """SQLAlchemy model for arbitrage opportunities table."""
    __tablename__ = "arbitrage_opportunities"

    id = Column(String, primary_key=True)
    event_id = Column(String, nullable=False)
    market_type = Column(String, nullable=False)
    selections = Column(JSON, nullable=False)
    total_implied_probability = Column(Numeric(10, 4), nullable=False)
    arbitrage_percentage = Column(Numeric(10, 4), nullable=False)
    recommended_stakes = Column(JSON, nullable=False)
    potential_profit = Column(Numeric(10, 2), nullable=False)
    risk_level = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)


class SystemMetricsTable(Base):
    """SQLAlchemy model for system metrics table."""
    __tablename__ = "system_metrics"

    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    total_bets = Column(Integer, nullable=False)
    winning_bets = Column(Integer, nullable=False)
    losing_bets = Column(Integer, nullable=False)
    win_rate = Column(Numeric(10, 4), nullable=False)
    total_profit_loss = Column(Numeric(10, 2), nullable=False)
    roi_percentage = Column(Numeric(10, 4), nullable=False)
    current_bankroll = Column(Numeric(10, 2), nullable=False)
    max_drawdown = Column(Numeric(10, 2), nullable=False)
    sharpe_ratio = Column(Numeric(10, 4))
    var_95 = Column(Numeric(10, 2))
    created_at = Column(DateTime, nullable=False)


class AlertTable(Base):
    """SQLAlchemy model for alerts table."""
    __tablename__ = "alerts"

    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    level = Column(String, nullable=False)
    category = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    details = Column(JSON)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)


class DatabaseManager:
    """Database manager for ABMBA system with validation pipelines."""

    def __init__(self, database_url: str, is_async: bool = True):
        self.database_url = database_url
        self.is_async = is_async

        if is_async:
            self.engine = create_async_engine(database_url, echo=False)
            self.session_factory = async_sessionmaker(self.engine, class_=AsyncSession)
        else:
            self.engine = create_engine(database_url, echo=False)
            self.session_factory = sessionmaker(self.engine)

        # Validation statistics
        self.validation_stats = {
            'schema_checks': 0,
            'data_integrity_checks': 0,
            'missing_data_corrections': 0,
            'validation_errors': 0
        }

    async def initialize(self):
        """Initialize database tables with validation."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Run initial schema validation
        await self.validate_schema()
        logger.info("Database initialized successfully with validation")

    async def validate_schema(self) -> dict[str, Any]:
        """
        Validate database schema integrity.
        
        Returns:
            Dictionary with schema validation results
        """
        try:
            async with self.engine.begin() as conn:
                # Use run_sync for inspection with async engine
                inspector = await conn.run_sync(lambda sync_conn: inspect(sync_conn))
                tables = await conn.run_sync(lambda sync_conn: inspector.get_table_names())

                expected_tables = [
                    'events', 'odds', 'bets', 'bankroll_logs',
                    'simulation_results', 'model_predictions',
                    'arbitrage_opportunities', 'system_metrics', 'alerts'
                ]

                missing_tables = set(expected_tables) - set(tables)
                extra_tables = set(tables) - set(expected_tables)

                # Check column structure for key tables
                schema_issues = []
                for table in expected_tables:
                    if table in tables:
                        columns = await conn.run_sync(lambda sync_conn: inspector.get_columns(table))
                        column_names = [col['name'] for col in columns]

                        # Basic column checks
                        if table == 'events' and 'home_team' not in column_names:
                            schema_issues.append(f"Missing 'home_team' column in {table}")
                        if table == 'odds' and 'odds' not in column_names:
                            schema_issues.append(f"Missing 'odds' column in {table}")
                        if table == 'bets' and 'stake' not in column_names:
                            schema_issues.append(f"Missing 'stake' column in {table}")

            self.validation_stats['schema_checks'] += 1

            validation_result = {
                'is_valid': len(missing_tables) == 0 and len(schema_issues) == 0,
                'missing_tables': list(missing_tables),
                'extra_tables': list(extra_tables),
                'schema_issues': schema_issues,
                'total_tables': len(tables),
                'validation_timestamp': datetime.utcnow().isoformat()
            }

            if not validation_result['is_valid']:
                logger.warning(f"Schema validation issues: {validation_result}")
                self.validation_stats['validation_errors'] += 1
            else:
                logger.info("Schema validation passed")

            return validation_result

        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            self.validation_stats['validation_errors'] += 1
            return {'is_valid': False, 'error': str(e)}

    async def validate_data_integrity(self, table_name: str) -> dict[str, Any]:
        """
        Validate data integrity for a specific table.
        
        Args:
            table_name: Name of the table to validate
            
        Returns:
            Dictionary with integrity validation results
        """
        try:
            async with self.get_session() as session:
                # Check for null values in required columns
                null_check_query = text(f"""
                    SELECT COUNT(*) as null_count 
                    FROM {table_name} 
                    WHERE id IS NULL OR created_at IS NULL
                """)
                result = await session.execute(null_check_query)
                null_count = result.scalar()

                # Check for duplicate IDs
                duplicate_check_query = text(f"""
                    SELECT COUNT(*) as duplicate_count 
                    FROM (
                        SELECT id, COUNT(*) as cnt 
                        FROM {table_name} 
                        GROUP BY id 
                        HAVING COUNT(*) > 1
                    ) duplicates
                """)
                result = await session.execute(duplicate_check_query)
                duplicate_count = result.scalar()

                # Check for data type violations
                type_violations = 0
                if table_name == 'odds':
                    # Check for invalid odds values
                    type_check_query = text(f"""
                        SELECT COUNT(*) as invalid_odds 
                        FROM {table_name} 
                        WHERE odds <= 0 OR odds IS NULL
                    """)
                    result = await session.execute(type_check_query)
                    type_violations = result.scalar()

                self.validation_stats['data_integrity_checks'] += 1

                validation_result = {
                    'table_name': table_name,
                    'is_valid': null_count == 0 and duplicate_count == 0 and type_violations == 0,
                    'null_count': null_count,
                    'duplicate_count': duplicate_count,
                    'type_violations': type_violations,
                    'validation_timestamp': datetime.utcnow().isoformat()
                }

                if not validation_result['is_valid']:
                    logger.warning(f"Data integrity issues in {table_name}: {validation_result}")
                    self.validation_stats['validation_errors'] += 1
                else:
                    logger.info(f"Data integrity validation passed for {table_name}")

                return validation_result

        except Exception as e:
            logger.error(f"Data integrity validation failed for {table_name}: {e}")
            self.validation_stats['validation_errors'] += 1
            return {'table_name': table_name, 'is_valid': False, 'error': str(e)}

    async def apply_inverse_probability_weighting(self, table_name: str, missing_columns: list[str]) -> dict[str, Any]:
        """
        Apply inverse probability weighting for missing data.
        
        Args:
            table_name: Name of the table to process
            missing_columns: Columns with missing data
            
        Returns:
            Dictionary with weighting results
        """
        try:
            async with self.get_session() as session:
                # Get data with missing values
                query = text(f"SELECT * FROM {table_name}")
                result = await session.execute(query)
                rows = result.fetchall()

                if not rows:
                    return {'table_name': table_name, 'rows_processed': 0, 'corrections_applied': 0}

                # Convert to DataFrame for analysis
                df = pd.DataFrame(rows, columns=[desc[0] for desc in result.description])

                corrections_applied = 0

                for column in missing_columns:
                    if column in df.columns:
                        # Calculate missing rate
                        missing_rate = df[column].isnull().mean()

                        if missing_rate > 0:
                            # Apply inverse probability weighting
                            # For each non-missing value, weight by 1/(1-missing_rate)
                            weight = 1 / (1 - missing_rate)

                            # Update non-missing values with weights
                            non_missing_mask = df[column].notna()
                            df.loc[non_missing_mask, f'{column}_weight'] = weight

                            corrections_applied += non_missing_mask.sum()

                            logger.info(f"Applied IPW to {column} in {table_name}: weight={weight:.3f}, missing_rate={missing_rate:.3f}")

                self.validation_stats['missing_data_corrections'] += corrections_applied

                return {
                    'table_name': table_name,
                    'rows_processed': len(df),
                    'corrections_applied': corrections_applied,
                    'missing_columns_processed': len(missing_columns),
                    'processing_timestamp': datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error(f"Inverse probability weighting failed for {table_name}: {e}")
            return {'table_name': table_name, 'error': str(e)}

    async def detect_data_anomalies(self, table_name: str, numeric_columns: list[str]) -> dict[str, Any]:
        """
        Detect anomalies in numeric columns using statistical methods.
        
        Args:
            table_name: Name of the table to analyze
            numeric_columns: Numeric columns to check for anomalies
            
        Returns:
            Dictionary with anomaly detection results
        """
        try:
            async with self.get_session() as session:
                # Get numeric data
                columns_str = ', '.join(numeric_columns)
                query = text(f"SELECT {columns_str} FROM {table_name}")
                result = await session.execute(query)
                rows = result.fetchall()

                if not rows:
                    return {'table_name': table_name, 'anomalies_detected': 0}

                # Convert to DataFrame
                df = pd.DataFrame(rows, columns=numeric_columns)

                anomalies = {}
                total_anomalies = 0

                for column in numeric_columns:
                    if column in df.columns:
                        # Z-score method
                        z_scores = np.abs(zscore(df[column].dropna()))
                        z_anomalies = z_scores > 3

                        # IQR method
                        Q1 = df[column].quantile(0.25)
                        Q3 = df[column].quantile(0.75)
                        IQR = Q3 - Q1
                        iqr_anomalies = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))

                        # Combine methods
                        combined_anomalies = z_anomalies | iqr_anomalies
                        anomaly_count = combined_anomalies.sum()

                        anomalies[column] = {
                            'z_score_anomalies': z_anomalies.sum(),
                            'iqr_anomalies': iqr_anomalies.sum(),
                            'total_anomalies': anomaly_count,
                            'anomaly_rate': anomaly_count / len(df)
                        }

                        total_anomalies += anomaly_count

                return {
                    'table_name': table_name,
                    'anomalies_detected': total_anomalies,
                    'column_anomalies': anomalies,
                    'analysis_timestamp': datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error(f"Anomaly detection failed for {table_name}: {e}")
            return {'table_name': table_name, 'error': str(e)}

    async def get_validation_report(self) -> dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Returns:
            Dictionary with validation statistics
        """
        # Run schema validation
        schema_result = await self.validate_schema()

        # Run data integrity checks on key tables
        integrity_results = {}
        key_tables = ['events', 'odds', 'bets']

        for table in key_tables:
            integrity_results[table] = await self.validate_data_integrity(table)

        return {
            'schema_validation': schema_result,
            'data_integrity': integrity_results,
            'validation_stats': self.validation_stats,
            'report_timestamp': datetime.utcnow().isoformat()
        }

    @asynccontextmanager
    async def get_session(self):
        """Get database session context manager."""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error("Database session error", error=str(e))
                raise

    async def save_event(self, event: Event) -> str:
        """Save event to database."""
        async with self.get_session() as session:
            event_dict = event.dict()
            event_dict['created_at'] = event_dict['created_at'].isoformat()
            event_dict['updated_at'] = event_dict['updated_at'].isoformat()
            event_dict['event_date'] = event_dict['event_date'].isoformat()

            stmt = text("""
                INSERT INTO events (id, sport, home_team, away_team, event_date, status, 
                                  home_score, away_score, created_at, updated_at)
                VALUES (:id, :sport, :home_team, :away_team, :event_date, :status,
                       :home_score, :away_score, :created_at, :updated_at)
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    home_score = EXCLUDED.home_score,
                    away_score = EXCLUDED.away_score,
                    updated_at = EXCLUDED.updated_at
            """)

            await session.execute(stmt, event_dict)
            return event.id

    async def save_odds(self, odds: Odds) -> str:
        """Save odds to database."""
        async with self.get_session() as session:
            odds_dict = odds.dict()
            odds_dict['timestamp'] = odds_dict['timestamp'].isoformat()

            # Convert Decimal to float for SQLite compatibility
            if 'odds' in odds_dict and odds_dict['odds'] is not None:
                odds_dict['odds'] = float(odds_dict['odds'])
            if 'line' in odds_dict and odds_dict['line'] is not None:
                odds_dict['line'] = float(odds_dict['line'])
            if 'implied_probability' in odds_dict and odds_dict['implied_probability'] is not None:
                odds_dict['implied_probability'] = float(odds_dict['implied_probability'])

            stmt = text("""
                INSERT INTO odds (id, event_id, platform, market_type, selection,
                                odds, line, implied_probability, timestamp)
                VALUES (:id, :event_id, :platform, :market_type, :selection,
                       :odds, :line, :implied_probability, :timestamp)
                ON CONFLICT (id) DO UPDATE SET
                    odds = EXCLUDED.odds,
                    line = EXCLUDED.line,
                    implied_probability = EXCLUDED.implied_probability,
                    timestamp = EXCLUDED.timestamp
            """)

            await session.execute(stmt, odds_dict)
            return odds.id

    async def save_bet(self, bet: Bet) -> str:
        """Save bet to database."""
        async with self.get_session() as session:
            bet_dict = bet.dict()
            bet_dict['created_at'] = bet_dict['created_at'].isoformat()
            if bet_dict['placed_at']:
                bet_dict['placed_at'] = bet_dict['placed_at'].isoformat()
            if bet_dict['settled_at']:
                bet_dict['settled_at'] = bet_dict['settled_at'].isoformat()

            # Convert Decimal to float for SQLite compatibility
            if 'odds' in bet_dict and bet_dict['odds'] is not None:
                bet_dict['odds'] = float(bet_dict['odds'])
            if 'stake' in bet_dict and bet_dict['stake'] is not None:
                bet_dict['stake'] = float(bet_dict['stake'])
            if 'potential_win' in bet_dict and bet_dict['potential_win'] is not None:
                bet_dict['potential_win'] = float(bet_dict['potential_win'])
            if 'expected_value' in bet_dict and bet_dict['expected_value'] is not None:
                bet_dict['expected_value'] = float(bet_dict['expected_value'])
            if 'kelly_fraction' in bet_dict and bet_dict['kelly_fraction'] is not None:
                bet_dict['kelly_fraction'] = float(bet_dict['kelly_fraction'])
            if 'profit_loss' in bet_dict and bet_dict['profit_loss'] is not None:
                bet_dict['profit_loss'] = float(bet_dict['profit_loss'])

            stmt = text("""
                INSERT INTO bets (id, event_id, platform, market_type, selection,
                                odds, stake, potential_win, expected_value, kelly_fraction,
                                status, placed_at, settled_at, result, profit_loss, created_at)
                VALUES (:id, :event_id, :platform, :market_type, :selection,
                       :odds, :stake, :potential_win, :expected_value, :kelly_fraction,
                       :status, :placed_at, :settled_at, :result, :profit_loss, :created_at)
            """)

            await session.execute(stmt, bet_dict)
            return bet.id

    async def get_events(self, sport: str | None = None, status: str | None = None) -> list[Event]:
        """Get events from database."""
        async with self.get_session() as session:
            query = "SELECT * FROM events WHERE 1=1"
            params = {}

            if sport:
                query += " AND sport = :sport"
                params['sport'] = sport

            if status:
                query += " AND status = :status"
                params['status'] = status

            query += " ORDER BY event_date DESC"

            result = await session.execute(text(query), params)
            rows = result.fetchall()

            events = []
            for row in rows:
                event_dict = dict(row._mapping)
                # Convert string dates back to datetime
                event_dict['event_date'] = datetime.fromisoformat(event_dict['event_date'])
                event_dict['created_at'] = datetime.fromisoformat(event_dict['created_at'])
                event_dict['updated_at'] = datetime.fromisoformat(event_dict['updated_at'])
                events.append(Event(**event_dict))

            return events

    async def get_latest_odds(self, event_id: str, platform: str | None = None) -> list[Odds]:
        """Get latest odds for an event."""
        async with self.get_session() as session:
            query = """
                SELECT o.* FROM odds o
                INNER JOIN (
                    SELECT event_id, platform, market_type, selection, MAX(timestamp) as max_timestamp
                    FROM odds
                    WHERE event_id = :event_id
                    GROUP BY event_id, platform, market_type, selection
                ) latest ON o.event_id = latest.event_id 
                    AND o.platform = latest.platform 
                    AND o.market_type = latest.market_type 
                    AND o.selection = latest.selection 
                    AND o.timestamp = latest.max_timestamp
            """
            params = {'event_id': event_id}

            if platform:
                query += " AND o.platform = :platform"
                params['platform'] = platform

            result = await session.execute(text(query), params)
            rows = result.fetchall()

            odds_list = []
            for row in rows:
                odds_dict = dict(row._mapping)
                odds_dict['timestamp'] = datetime.fromisoformat(odds_dict['timestamp'])
                odds_list.append(Odds(**odds_dict))

            return odds_list

    async def get_bankroll_history(self, limit: int = 100) -> list[BankrollLog]:
        """Get bankroll history."""
        async with self.get_session() as session:
            query = "SELECT * FROM bankroll_logs ORDER BY timestamp DESC LIMIT :limit"
            result = await session.execute(text(query), {'limit': limit})
            rows = result.fetchall()

            logs = []
            for row in rows:
                log_dict = dict(row._mapping)
                log_dict['timestamp'] = datetime.fromisoformat(log_dict['timestamp'])
                logs.append(BankrollLog(**log_dict))

            return logs

    async def get_current_bankroll(self) -> Decimal:
        """Get current bankroll balance."""
        async with self.get_session() as session:
            query = "SELECT balance FROM bankroll_logs ORDER BY timestamp DESC LIMIT 1"
            result = await session.execute(text(query))
            row = result.fetchone()

            if row:
                return Decimal(str(row[0]))
            return Decimal('0')

    async def save_bankroll_log(self, log: BankrollLog) -> str:
        """Save bankroll log entry."""
        async with self.get_session() as session:
            log_dict = log.dict()
            log_dict['timestamp'] = log_dict['timestamp'].isoformat()

            stmt = text("""
                INSERT INTO bankroll_logs (id, timestamp, balance, change, bet_id, description, source)
                VALUES (:id, :timestamp, :balance, :change, :bet_id, :description, :source)
            """)

            await session.execute(stmt, log_dict)
            return log.id

    async def get_system_metrics(self, days: int = 30) -> SystemMetrics:
        """Calculate and return system metrics."""
        async with self.get_session() as session:
            # Get bets from last N days
            query = """
                SELECT 
                    COUNT(*) as total_bets,
                    COUNT(CASE WHEN status = 'won' THEN 1 END) as winning_bets,
                    COUNT(CASE WHEN status = 'lost' THEN 1 END) as losing_bets,
                    COALESCE(SUM(CASE WHEN status = 'won' THEN profit_loss ELSE 0 END), 0) as total_profit,
                    COALESCE(SUM(CASE WHEN status = 'lost' THEN profit_loss ELSE 0 END), 0) as total_loss
                FROM bets 
                WHERE created_at >= NOW() - INTERVAL ':days days'
            """

            result = await session.execute(text(query), {'days': days})
            row = result.fetchone()

            if row:
                total_bets = row[0] or 0
                winning_bets = row[1] or 0
                losing_bets = row[2] or 0
                total_profit = Decimal(str(row[3] or 0))
                total_loss = Decimal(str(row[4] or 0))

                win_rate = Decimal(winning_bets) / Decimal(total_bets) if total_bets > 0 else Decimal('0')
                total_profit_loss = total_profit + total_loss
                current_bankroll = await self.get_current_bankroll()

                # Calculate ROI
                initial_bankroll = Decimal('100')  # From config
                roi_percentage = ((current_bankroll - initial_bankroll) / initial_bankroll) * 100

                # Calculate max drawdown
                drawdown_query = """
                    SELECT MIN(balance) as min_balance
                    FROM bankroll_logs
                    WHERE timestamp >= NOW() - INTERVAL ':days days'
                """
                drawdown_result = await session.execute(text(drawdown_query), {'days': days})
                drawdown_row = drawdown_result.fetchone()
                max_drawdown = Decimal(str(drawdown_row[0] or 0)) if drawdown_row else Decimal('0')

                return SystemMetrics(
                    total_bets=total_bets,
                    winning_bets=winning_bets,
                    losing_bets=losing_bets,
                    win_rate=win_rate,
                    total_profit_loss=total_profit_loss,
                    roi_percentage=roi_percentage,
                    current_bankroll=current_bankroll,
                    max_drawdown=max_drawdown
                )

            return SystemMetrics(
                total_bets=0,
                winning_bets=0,
                losing_bets=0,
                win_rate=Decimal('0'),
                total_profit_loss=Decimal('0'),
                roi_percentage=Decimal('0'),
                current_bankroll=Decimal('0'),
                max_drawdown=Decimal('0')
            )

    async def save_alert(self, alert: Alert) -> str:
        """Save alert to database."""
        async with self.get_session() as session:
            alert_dict = alert.dict()
            alert_dict['timestamp'] = alert_dict['timestamp'].isoformat()
            if alert_dict['resolved_at']:
                alert_dict['resolved_at'] = alert_dict['resolved_at'].isoformat()

            stmt = text("""
                INSERT INTO alerts (id, timestamp, level, category, message, details, resolved, resolved_at)
                VALUES (:id, :timestamp, :level, :category, :message, :details, :resolved, :resolved_at)
            """)

            await session.execute(stmt, alert_dict)
            return alert.id

    async def close(self):
        """Close database connections."""
        await self.engine.dispose()
        logger.info("Database connections closed")


# Utility functions for data transformation
def event_to_dict(event: Event) -> dict[str, Any]:
    """Convert Event model to dictionary for database storage."""
    return {
        'id': event.id,
        'sport': event.sport.value,
        'home_team': event.home_team,
        'away_team': event.away_team,
        'event_date': event.event_date.isoformat(),
        'status': event.status.value,
        'home_score': event.home_score,
        'away_score': event.away_score,
        'created_at': event.created_at.isoformat(),
        'updated_at': event.updated_at.isoformat()
    }


def odds_to_dict(odds: Odds) -> dict[str, Any]:
    """Convert Odds model to dictionary for database storage."""
    return {
        'id': odds.id,
        'event_id': odds.event_id,
        'platform': odds.platform.value,
        'market_type': odds.market_type.value,
        'selection': odds.selection,
        'odds': float(odds.odds),
        'line': float(odds.line) if odds.line else None,
        'implied_probability': float(odds.implied_probability) if odds.implied_probability else None,
        'timestamp': odds.timestamp.isoformat()
    }
