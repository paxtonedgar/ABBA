"""
Algo Trading Manager for ABMBA system.
Handles systematic backtesting and execution.
"""

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import structlog
from database import DatabaseManager

from models import Bet

logger = structlog.get_logger()


@dataclass
class BacktestResult:
    """Container for backtest results."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_bets: int
    profit_factor: float
    avg_bet_size: float
    volatility: float
    calmar_ratio: float
    trades: list[dict]


@dataclass
class ExecutionSignal:
    """Container for execution signals."""
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    stake_size: float
    odds: float
    expected_value: float
    timestamp: datetime
    metadata: dict[str, Any]


class AlgoTradingManager:
    """Manages algorithmic trading strategies and execution."""

    def __init__(self, config: dict, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.backtest_results_dir = "results/backtests"
        os.makedirs(self.backtest_results_dir, exist_ok=True)

        # Trading parameters
        self.risk_management = RiskManager(config)
        self.position_sizer = PositionSizer(config)
        self.signal_generator = SignalGenerator(config)

        # Performance tracking
        self.performance_metrics = PerformanceMetrics()

        logger.info("Algo Trading Manager initialized")

    async def systematic_backtesting(self, strategy_config: dict,
                                   start_date: datetime,
                                   end_date: datetime) -> BacktestResult:
        """Run systematic backtesting of trading strategies."""
        try:
            logger.info(f"Starting systematic backtesting from {start_date} to {end_date}")

            # Initialize backtest environment
            backtest_env = await self._initialize_backtest_environment(
                strategy_config, start_date, end_date
            )

            # Run backtest
            trades = []
            current_bankroll = strategy_config.get('initial_bankroll', 10000)
            max_bankroll = current_bankroll

            # Get historical data
            historical_data = await self._get_historical_data(start_date, end_date)

            for timestamp, data_point in historical_data.iterrows():
                # Generate signals
                signals = await self.signal_generator.generate_signals(data_point, strategy_config)

                # Apply risk management
                filtered_signals = await self.risk_management.filter_signals(signals, current_bankroll)

                # Execute trades
                for signal in filtered_signals:
                    trade_result = await self._execute_backtest_trade(
                        signal, current_bankroll, timestamp
                    )

                    if trade_result:
                        trades.append(trade_result)
                        current_bankroll = trade_result['ending_bankroll']
                        max_bankroll = max(max_bankroll, current_bankroll)

            # Calculate performance metrics
            backtest_result = await self._calculate_backtest_metrics(
                trades, strategy_config.get('initial_bankroll', 10000)
            )

            # Save results
            await self._save_backtest_results(backtest_result, strategy_config)

            logger.info(f"Backtesting completed. Total return: {backtest_result.total_return:.2%}")

            return backtest_result

        except Exception as e:
            logger.error(f"Error in systematic backtesting: {e}")
            return BacktestResult(
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_bets=0,
                profit_factor=0.0,
                avg_bet_size=0.0,
                volatility=0.0,
                calmar_ratio=0.0,
                trades=[]
            )

    async def _initialize_backtest_environment(self, strategy_config: dict,
                                            start_date: datetime,
                                            end_date: datetime) -> dict:
        """Initialize backtest environment."""
        try:
            environment = {
                'strategy_config': strategy_config,
                'start_date': start_date,
                'end_date': end_date,
                'current_date': start_date,
                'bankroll': strategy_config.get('initial_bankroll', 10000),
                'open_positions': [],
                'trade_history': [],
                'risk_limits': strategy_config.get('risk_limits', {}),
                'position_limits': strategy_config.get('position_limits', {})
            }

            return environment

        except Exception as e:
            logger.error(f"Error initializing backtest environment: {e}")
            return {}

    async def _get_historical_data(self, start_date: datetime,
                                 end_date: datetime) -> pd.DataFrame:
        """Get historical data for backtesting."""
        try:
            # Get historical bets and events
            historical_bets = await self.db_manager.get_bets(
                start_date=start_date,
                end_date=end_date
            )

            # Convert to DataFrame
            data = []
            for bet in historical_bets:
                data.append({
                    'timestamp': bet.placed_at,
                    'event_id': bet.event_id,
                    'odds': float(bet.odds or 0),
                    'stake': float(bet.stake or 0),
                    'expected_value': float(bet.expected_value or 0),
                    'result': bet.result,
                    'return_amount': float(bet.return_amount or 0),
                    'sport': getattr(bet, 'sport', 'unknown'),
                    'market_type': getattr(bet, 'market_type', 'unknown')
                })

            df = pd.DataFrame(data)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')

            return df

        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()

    async def _execute_backtest_trade(self, signal: ExecutionSignal,
                                    current_bankroll: float,
                                    timestamp: datetime) -> dict | None:
        """Execute a trade in backtest environment."""
        try:
            if signal.signal_type != 'buy':
                return None

            # Calculate position size
            stake_size = await self.position_sizer.calculate_stake_size(
                signal, current_bankroll
            )

            if stake_size <= 0:
                return None

            # Simulate trade outcome
            trade_outcome = await self._simulate_trade_outcome(signal, stake_size)

            # Calculate new bankroll
            if trade_outcome['result'] == 'win':
                ending_bankroll = current_bankroll - stake_size + trade_outcome['return_amount']
            else:
                ending_bankroll = current_bankroll - stake_size

            trade_result = {
                'timestamp': timestamp,
                'signal': signal,
                'stake_size': stake_size,
                'starting_bankroll': current_bankroll,
                'ending_bankroll': ending_bankroll,
                'result': trade_outcome['result'],
                'return_amount': trade_outcome['return_amount'],
                'profit_loss': ending_bankroll - current_bankroll
            }

            return trade_result

        except Exception as e:
            logger.error(f"Error executing backtest trade: {e}")
            return None

    async def _simulate_trade_outcome(self, signal: ExecutionSignal,
                                    stake_size: float) -> dict[str, Any]:
        """Simulate the outcome of a trade."""
        try:
            # Use expected value to determine win probability
            win_probability = signal.expected_value / (signal.odds - 1) if signal.odds > 1 else 0.5

            # Add some randomness to make it realistic
            win_probability = min(max(win_probability, 0.1), 0.9)

            # Simulate outcome
            if np.random.random() < win_probability:
                result = 'win'
                return_amount = stake_size * signal.odds
            else:
                result = 'loss'
                return_amount = 0.0

            return {
                'result': result,
                'return_amount': return_amount,
                'win_probability': win_probability
            }

        except Exception as e:
            logger.error(f"Error simulating trade outcome: {e}")
            return {'result': 'loss', 'return_amount': 0.0, 'win_probability': 0.5}

    async def _calculate_backtest_metrics(self, trades: list[dict],
                                        initial_bankroll: float) -> BacktestResult:
        """Calculate comprehensive backtest metrics."""
        try:
            if not trades:
                return BacktestResult(
                    total_return=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    win_rate=0.0,
                    total_bets=len(trades),
                    profit_factor=0.0,
                    avg_bet_size=0.0,
                    volatility=0.0,
                    calmar_ratio=0.0,
                    trades=trades
                )

            # Basic metrics
            total_bets = len(trades)
            winning_trades = [t for t in trades if t['result'] == 'win']
            losing_trades = [t for t in trades if t['result'] == 'loss']

            win_rate = len(winning_trades) / total_bets if total_bets > 0 else 0

            # Returns calculation
            final_bankroll = trades[-1]['ending_bankroll'] if trades else initial_bankroll
            total_return = (final_bankroll - initial_bankroll) / initial_bankroll

            # Profit factor
            gross_profit = sum(t['return_amount'] for t in winning_trades)
            gross_loss = sum(t['stake_size'] for t in losing_trades)
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            # Average bet size
            avg_bet_size = np.mean([t['stake_size'] for t in trades])

            # Calculate returns series for advanced metrics
            returns_series = []
            running_bankroll = initial_bankroll

            for trade in trades:
                trade_return = (trade['ending_bankroll'] - running_bankroll) / running_bankroll
                returns_series.append(trade_return)
                running_bankroll = trade['ending_bankroll']

            # Advanced metrics
            volatility = np.std(returns_series) if returns_series else 0
            sharpe_ratio = np.mean(returns_series) / volatility if volatility > 0 else 0

            # Maximum drawdown
            max_drawdown = await self._calculate_max_drawdown(trades, initial_bankroll)

            # Calmar ratio
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0

            return BacktestResult(
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_bets=total_bets,
                profit_factor=profit_factor,
                avg_bet_size=avg_bet_size,
                volatility=volatility,
                calmar_ratio=calmar_ratio,
                trades=trades
            )

        except Exception as e:
            logger.error(f"Error calculating backtest metrics: {e}")
            return BacktestResult(
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_bets=0,
                profit_factor=0.0,
                avg_bet_size=0.0,
                volatility=0.0,
                calmar_ratio=0.0,
                trades=[]
            )

    async def _calculate_max_drawdown(self, trades: list[dict],
                                    initial_bankroll: float) -> float:
        """Calculate maximum drawdown."""
        try:
            if not trades:
                return 0.0

            peak = initial_bankroll
            max_dd = 0.0

            for trade in trades:
                current_bankroll = trade['ending_bankroll']

                if current_bankroll > peak:
                    peak = current_bankroll

                drawdown = (peak - current_bankroll) / peak
                max_dd = max(max_dd, drawdown)

            return max_dd

        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0

    async def _save_backtest_results(self, result: BacktestResult,
                                   strategy_config: dict):
        """Save backtest results to file."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{timestamp}.json"
            filepath = os.path.join(self.backtest_results_dir, filename)

            # Convert to serializable format
            data = {
                'timestamp': timestamp,
                'strategy_config': strategy_config,
                'results': {
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate,
                    'total_bets': result.total_bets,
                    'profit_factor': result.profit_factor,
                    'avg_bet_size': result.avg_bet_size,
                    'volatility': result.volatility,
                    'calmar_ratio': result.calmar_ratio
                },
                'trades_summary': [
                    {
                        'timestamp': t['timestamp'].isoformat(),
                        'stake_size': t['stake_size'],
                        'result': t['result'],
                        'profit_loss': t['profit_loss']
                    }
                    for t in result.trades
                ]
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Backtest results saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")

    async def real_time_execution(self, signals: list[ExecutionSignal]) -> list[dict]:
        """Execute trades in real-time."""
        try:
            logger.info(f"Executing {len(signals)} real-time signals")

            execution_results = []

            for signal in signals:
                # Apply risk management
                if not await self.risk_management.validate_signal(signal):
                    logger.warning(f"Signal rejected by risk management: {signal}")
                    continue

                # Calculate position size
                stake_size = await self.position_sizer.calculate_stake_size(signal)

                if stake_size <= 0:
                    logger.warning(f"Invalid stake size for signal: {signal}")
                    continue

                # Execute trade
                execution_result = await self._execute_real_time_trade(signal, stake_size)

                if execution_result:
                    execution_results.append(execution_result)

                    # Update performance metrics
                    await self.performance_metrics.update_metrics(execution_result)

            logger.info(f"Real-time execution completed. {len(execution_results)} trades executed")

            return execution_results

        except Exception as e:
            logger.error(f"Error in real-time execution: {e}")
            return []

    async def _execute_real_time_trade(self, signal: ExecutionSignal,
                                     stake_size: float) -> dict | None:
        """Execute a real-time trade."""
        try:
            # Create bet record
            bet_data = {
                'event_id': signal.metadata.get('event_id'),
                'odds': Decimal(str(signal.odds)),
                'stake': Decimal(str(stake_size)),
                'expected_value': Decimal(str(signal.expected_value)),
                'placed_at': signal.timestamp,
                'status': 'pending',
                'sport': signal.metadata.get('sport', 'unknown'),
                'market_type': signal.metadata.get('market_type', 'unknown')
            }

            # Save to database
            bet_id = await self.db_manager.create_bet(bet_data)

            if not bet_id:
                logger.error("Failed to create bet record")
                return None

            # Execute through execution agent (simulated)
            execution_success = await self._simulate_execution(bet_id, signal)

            if execution_success:
                # Update bet status
                await self.db_manager.update_bet_status(bet_id, 'placed')

                execution_result = {
                    'bet_id': bet_id,
                    'signal': signal,
                    'stake_size': stake_size,
                    'execution_time': datetime.utcnow(),
                    'status': 'executed',
                    'execution_price': signal.odds
                }

                logger.info(f"Trade executed successfully: {bet_id}")
                return execution_result
            else:
                # Update bet status to failed
                await self.db_manager.update_bet_status(bet_id, 'failed')
                logger.error(f"Trade execution failed: {bet_id}")
                return None

        except Exception as e:
            logger.error(f"Error executing real-time trade: {e}")
            return None

    async def _simulate_execution(self, bet_id: str, signal: ExecutionSignal) -> bool:
        """Simulate trade execution (would integrate with real execution system)."""
        try:
            # Simulate execution delay
            await asyncio.sleep(0.1)

            # Simulate execution success (90% success rate)
            success = np.random.random() > 0.1

            return success

        except Exception as e:
            logger.error(f"Error simulating execution: {e}")
            return False

    async def optimize_strategy_parameters(self, strategy_config: dict,
                                         historical_data: pd.DataFrame) -> dict:
        """Optimize strategy parameters using historical data."""
        try:
            logger.info("Starting strategy parameter optimization")

            # Define parameter ranges to test
            param_ranges = {
                'ev_threshold': [0.01, 0.02, 0.05, 0.10, 0.15],
                'max_stake_percent': [0.01, 0.02, 0.05, 0.10],
                'max_drawdown': [0.05, 0.10, 0.15, 0.20],
                'stop_loss': [0.05, 0.10, 0.15, 0.20]
            }

            best_params = {}
            best_sharpe = -float('inf')

            # Grid search optimization
            for ev_threshold in param_ranges['ev_threshold']:
                for max_stake in param_ranges['max_stake_percent']:
                    for max_dd in param_ranges['max_drawdown']:
                        for stop_loss in param_ranges['stop_loss']:

                            # Create test config
                            test_config = strategy_config.copy()
                            test_config.update({
                                'ev_threshold': ev_threshold,
                                'max_stake_percent': max_stake,
                                'max_drawdown': max_dd,
                                'stop_loss': stop_loss
                            })

                            # Run backtest with these parameters
                            start_date = historical_data['timestamp'].min()
                            end_date = historical_data['timestamp'].max()

                            result = await self.systematic_backtesting(
                                test_config, start_date, end_date
                            )

                            # Check if this is the best result
                            if result.sharpe_ratio > best_sharpe:
                                best_sharpe = result.sharpe_ratio
                                best_params = {
                                    'ev_threshold': ev_threshold,
                                    'max_stake_percent': max_stake,
                                    'max_drawdown': max_dd,
                                    'stop_loss': stop_loss,
                                    'sharpe_ratio': best_sharpe
                                }

            logger.info(f"Parameter optimization completed. Best Sharpe: {best_sharpe:.3f}")

            return best_params

        except Exception as e:
            logger.error(f"Error optimizing strategy parameters: {e}")
            return {}

    async def get_performance_report(self, days_back: int = 30) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)

            # Get recent trades
            recent_bets = await self.db_manager.get_bets(
                start_date=start_date,
                end_date=end_date
            )

            if not recent_bets:
                return {"message": f"No trades in the last {days_back} days"}

            # Calculate metrics
            total_bets = len(recent_bets)
            winning_bets = [b for b in recent_bets if b.result == "win"]
            losing_bets = [b for b in recent_bets if b.result == "loss"]

            win_rate = len(winning_bets) / total_bets if total_bets > 0 else 0

            # Calculate returns
            total_stake = sum([float(b.stake or 0) for b in recent_bets])
            total_return = sum([float(b.return_amount or 0) for b in recent_bets])
            net_profit = total_return - total_stake

            # Calculate ROI
            roi = net_profit / total_stake if total_stake > 0 else 0

            # Calculate average EV
            avg_ev = np.mean([float(b.expected_value or 0) for b in recent_bets])

            # Calculate profit factor
            gross_profit = sum([float(b.return_amount or 0) for b in winning_bets])
            gross_loss = sum([float(b.stake or 0) for b in losing_bets])
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            # Calculate average odds
            avg_odds = np.mean([float(b.odds or 0) for b in recent_bets])

            # Calculate average stake
            avg_stake = np.mean([float(b.stake or 0) for b in recent_bets])

            # Calculate daily returns
            daily_returns = await self._calculate_daily_returns(recent_bets)

            # Calculate Sharpe ratio
            if daily_returns:
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0
            else:
                sharpe_ratio = 0

            report = {
                'period_days': days_back,
                'total_bets': total_bets,
                'winning_bets': len(winning_bets),
                'losing_bets': len(losing_bets),
                'win_rate': win_rate,
                'total_stake': total_stake,
                'total_return': total_return,
                'net_profit': net_profit,
                'roi': roi,
                'avg_ev': avg_ev,
                'profit_factor': profit_factor,
                'avg_odds': avg_odds,
                'avg_stake': avg_stake,
                'sharpe_ratio': sharpe_ratio,
                'daily_returns': daily_returns
            }

            return report

        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"error": str(e)}

    async def _calculate_daily_returns(self, bets: list[Bet]) -> list[float]:
        """Calculate daily returns from bets."""
        try:
            # Group bets by date
            daily_bets = {}
            for bet in bets:
                date = bet.placed_at.date() if bet.placed_at else datetime.utcnow().date()
                if date not in daily_bets:
                    daily_bets[date] = []
                daily_bets[date].append(bet)

            # Calculate daily returns
            daily_returns = []
            for date, day_bets in daily_bets.items():
                daily_stake = sum([float(b.stake or 0) for b in day_bets])
                daily_return = sum([float(b.return_amount or 0) for b in day_bets])

                if daily_stake > 0:
                    daily_return_rate = (daily_return - daily_stake) / daily_stake
                    daily_returns.append(daily_return_rate)

            return daily_returns

        except Exception as e:
            logger.error(f"Error calculating daily returns: {e}")
            return []


class RiskManager:
    """Manages risk for algorithmic trading."""

    def __init__(self, config: dict):
        self.config = config
        self.risk_limits = config.get('risk_management', {})

        # Default risk limits
        self.max_position_size = self.risk_limits.get('max_position_size', 0.05)  # 5% of bankroll
        self.max_daily_loss = self.risk_limits.get('max_daily_loss', 0.10)  # 10% of bankroll
        self.max_drawdown = self.risk_limits.get('max_drawdown', 0.20)  # 20% drawdown
        self.min_ev_threshold = self.risk_limits.get('min_ev_threshold', 0.02)  # 2% minimum EV

    async def filter_signals(self, signals: list[ExecutionSignal],
                           current_bankroll: float) -> list[ExecutionSignal]:
        """Filter signals based on risk management rules."""
        try:
            filtered_signals = []

            for signal in signals:
                if await self.validate_signal(signal, current_bankroll):
                    filtered_signals.append(signal)
                else:
                    logger.debug(f"Signal filtered by risk management: {signal}")

            return filtered_signals

        except Exception as e:
            logger.error(f"Error filtering signals: {e}")
            return []

    async def validate_signal(self, signal: ExecutionSignal,
                            current_bankroll: float = None) -> bool:
        """Validate a single signal against risk management rules."""
        try:
            # Check EV threshold
            if signal.expected_value < self.min_ev_threshold:
                logger.debug(f"Signal rejected: EV {signal.expected_value} below threshold {self.min_ev_threshold}")
                return False

            # Check odds sanity
            if signal.odds < 1.01 or signal.odds > 100:
                logger.debug(f"Signal rejected: Odds {signal.odds} outside valid range")
                return False

            # Check confidence
            if signal.confidence < 0.5:
                logger.debug(f"Signal rejected: Confidence {signal.confidence} too low")
                return False

            # Check position size if bankroll provided
            if current_bankroll:
                max_stake = current_bankroll * self.max_position_size
                if signal.stake_size > max_stake:
                    logger.debug(f"Signal rejected: Stake {signal.stake_size} exceeds max {max_stake}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False

    async def check_daily_loss_limit(self, daily_pnl: float,
                                   bankroll: float) -> bool:
        """Check if daily loss limit has been reached."""
        try:
            daily_loss_rate = abs(min(daily_pnl, 0)) / bankroll
            return daily_loss_rate < self.max_daily_loss

        except Exception as e:
            logger.error(f"Error checking daily loss limit: {e}")
            return False

    async def check_drawdown_limit(self, current_bankroll: float,
                                 peak_bankroll: float) -> bool:
        """Check if maximum drawdown limit has been reached."""
        try:
            if peak_bankroll <= 0:
                return True

            drawdown = (peak_bankroll - current_bankroll) / peak_bankroll
            return drawdown < self.max_drawdown

        except Exception as e:
            logger.error(f"Error checking drawdown limit: {e}")
            return False


class PositionSizer:
    """Calculates optimal position sizes for trades."""

    def __init__(self, config: dict):
        self.config = config
        self.sizing_config = config.get('position_sizing', {})

        # Default sizing parameters
        self.kelly_fraction = self.sizing_config.get('kelly_fraction', 0.25)  # Conservative Kelly
        self.max_position_size = self.sizing_config.get('max_position_size', 0.05)  # 5% max
        self.min_position_size = self.sizing_config.get('min_position_size', 0.01)  # 1% min

    async def calculate_stake_size(self, signal: ExecutionSignal,
                                 bankroll: float = None) -> float:
        """Calculate optimal stake size for a signal."""
        try:
            # Use Kelly Criterion
            kelly_stake = await self._kelly_criterion(signal)

            # Apply position size limits
            if bankroll:
                max_stake = bankroll * self.max_position_size
                min_stake = bankroll * self.min_position_size

                kelly_stake = max(min_stake, min(kelly_stake, max_stake))

            return kelly_stake

        except Exception as e:
            logger.error(f"Error calculating stake size: {e}")
            return 0.0

    async def _kelly_criterion(self, signal: ExecutionSignal) -> float:
        """Calculate Kelly Criterion stake size."""
        try:
            # Kelly formula: f = (bp - q) / b
            # where: b = odds - 1, p = win probability, q = loss probability

            b = signal.odds - 1
            p = signal.expected_value / b if b > 0 else 0.5  # Win probability
            q = 1 - p  # Loss probability

            kelly_fraction = (b * p - q) / b if b > 0 else 0

            # Apply conservative Kelly (fraction of full Kelly)
            conservative_kelly = kelly_fraction * self.kelly_fraction

            # Ensure positive stake
            return max(conservative_kelly, 0)

        except Exception as e:
            logger.error(f"Error calculating Kelly Criterion: {e}")
            return 0.0


class SignalGenerator:
    """Generates trading signals based on various strategies."""

    def __init__(self, config: dict):
        self.config = config
        self.strategy_config = config.get('signal_generation', {})

    async def generate_signals(self, data_point: pd.Series,
                             strategy_config: dict) -> list[ExecutionSignal]:
        """Generate trading signals from data point."""
        try:
            signals = []

            # Check if this is a valid betting opportunity
            if not await self._is_valid_opportunity(data_point):
                return signals

            # Generate signals based on different strategies
            ev_signal = await self._generate_ev_signal(data_point, strategy_config)
            if ev_signal:
                signals.append(ev_signal)

            momentum_signal = await self._generate_momentum_signal(data_point, strategy_config)
            if momentum_signal:
                signals.append(momentum_signal)

            arbitrage_signal = await self._generate_arbitrage_signal(data_point, strategy_config)
            if arbitrage_signal:
                signals.append(arbitrage_signal)

            return signals

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []

    async def _is_valid_opportunity(self, data_point: pd.Series) -> bool:
        """Check if data point represents a valid betting opportunity."""
        try:
            # Check for required fields
            required_fields = ['odds', 'expected_value']
            for field in required_fields:
                if field not in data_point or pd.isna(data_point[field]):
                    return False

            # Check for reasonable values
            odds = data_point['odds']
            ev = data_point['expected_value']

            if odds < 1.01 or odds > 100:
                return False

            if ev < 0 or ev > 1:
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking valid opportunity: {e}")
            return False

    async def _generate_ev_signal(self, data_point: pd.Series,
                                strategy_config: dict) -> ExecutionSignal | None:
        """Generate signal based on expected value."""
        try:
            ev_threshold = strategy_config.get('ev_threshold', 0.02)
            ev = data_point['expected_value']

            if ev >= ev_threshold:
                confidence = min(ev / 0.1, 1.0)  # Scale confidence based on EV

                return ExecutionSignal(
                    signal_type='buy',
                    confidence=confidence,
                    stake_size=0.0,  # Will be calculated later
                    odds=data_point['odds'],
                    expected_value=ev,
                    timestamp=data_point['timestamp'],
                    metadata={
                        'strategy': 'ev_based',
                        'event_id': data_point.get('event_id'),
                        'sport': data_point.get('sport', 'unknown'),
                        'market_type': data_point.get('market_type', 'unknown')
                    }
                )

            return None

        except Exception as e:
            logger.error(f"Error generating EV signal: {e}")
            return None

    async def _generate_momentum_signal(self, data_point: pd.Series,
                                      strategy_config: dict) -> ExecutionSignal | None:
        """Generate signal based on momentum indicators."""
        try:
            # This would use momentum indicators like moving averages, RSI, etc.
            # For now, return None as placeholder
            return None

        except Exception as e:
            logger.error(f"Error generating momentum signal: {e}")
            return None

    async def _generate_arbitrage_signal(self, data_point: pd.Series,
                                       strategy_config: dict) -> ExecutionSignal | None:
        """Generate signal based on arbitrage opportunities."""
        try:
            # This would detect arbitrage opportunities across different books
            # For now, return None as placeholder
            return None

        except Exception as e:
            logger.error(f"Error generating arbitrage signal: {e}")
            return None


class PerformanceMetrics:
    """Tracks and calculates performance metrics."""

    def __init__(self):
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'win_streak': 0,
            'loss_streak': 0,
            'current_streak': 0
        }

    async def update_metrics(self, execution_result: dict):
        """Update metrics with new execution result."""
        try:
            self.metrics['total_trades'] += 1

            profit_loss = execution_result.get('profit_loss', 0)

            if profit_loss > 0:
                self.metrics['winning_trades'] += 1
                self.metrics['total_profit'] += profit_loss
                self.metrics['largest_win'] = max(self.metrics['largest_win'], profit_loss)

                # Update streaks
                if self.metrics['current_streak'] >= 0:
                    self.metrics['current_streak'] += 1
                else:
                    self.metrics['current_streak'] = 1

                self.metrics['win_streak'] = max(self.metrics['win_streak'], self.metrics['current_streak'])

            else:
                self.metrics['losing_trades'] += 1
                self.metrics['total_loss'] += abs(profit_loss)
                self.metrics['largest_loss'] = max(self.metrics['largest_loss'], abs(profit_loss))

                # Update streaks
                if self.metrics['current_streak'] <= 0:
                    self.metrics['current_streak'] -= 1
                else:
                    self.metrics['current_streak'] = -1

                self.metrics['loss_streak'] = max(self.metrics['loss_streak'], abs(self.metrics['current_streak']))

            # Update averages
            if self.metrics['winning_trades'] > 0:
                self.metrics['avg_win'] = self.metrics['total_profit'] / self.metrics['winning_trades']

            if self.metrics['losing_trades'] > 0:
                self.metrics['avg_loss'] = self.metrics['total_loss'] / self.metrics['losing_trades']

        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    def get_metrics(self) -> dict[str, Any]:
        """Get current performance metrics."""
        try:
            metrics = self.metrics.copy()

            # Calculate derived metrics
            if metrics['total_trades'] > 0:
                metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
            else:
                metrics['win_rate'] = 0.0

            metrics['net_profit'] = metrics['total_profit'] - metrics['total_loss']

            if metrics['total_loss'] > 0:
                metrics['profit_factor'] = metrics['total_profit'] / metrics['total_loss']
            else:
                metrics['profit_factor'] = float('inf')

            return metrics

        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}
