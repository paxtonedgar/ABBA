"""
Main entry point for ABMBA system.
Autonomous Bankroll Management and Betting Agent
"""

import argparse
import asyncio
import signal
import sys

import structlog
from agents import ABMBACrew
from data_fetcher import DataFetcher
from database import DatabaseManager
from execution import BettingExecutor
from simulations import SimulationManager
from utils import AlertManager, ConfigManager, HealthChecker, PerformanceMonitor

logger = structlog.get_logger()


class ABMBASystem:
    """Main ABMBA system orchestrator."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.running = False
        self.crew = None
        self.alert_manager = None
        self.health_checker = None
        self.performance_monitor = PerformanceMonitor()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    async def initialize(self):
        """Initialize the ABMBA system."""
        try:
            logger.info("Initializing ABMBA system...")

            # Initialize database
            self.db_manager = DatabaseManager(self.config['database']['url'])
            await self.db_manager.initialize()

            # Initialize components
            self.data_fetcher = DataFetcher(self.config)
            self.simulation_manager = SimulationManager(self.db_manager, self.config)
            self.executor = BettingExecutor(self.config)

            # Initialize managers
            self.alert_manager = AlertManager(self.config, self.db_manager)
            self.health_checker = HealthChecker(self.config, self.db_manager)

            # Initialize crew
            self.crew = ABMBACrew(self.config_manager.config_path)
            await self.crew.initialize()

            # Initialize executor
            await self.executor.initialize()

            # Send startup alert
            await self.alert_manager.send_alert(
                'info', 'system', 'ABMBA system initialized successfully'
            )

            logger.info("ABMBA system initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing ABMBA system: {e}")
            if self.alert_manager:
                await self.alert_manager.send_alert(
                    'error', 'system', f'Failed to initialize ABMBA system: {e}'
                )
            raise

    async def run_simulation_mode(self):
        """Run the system in simulation mode."""
        try:
            logger.info("Starting ABMBA system in simulation mode")

            # Run health check
            health_status = await self.health_checker.check_system_health()
            if health_status['overall'] != 'healthy':
                logger.warning(f"System health check failed: {health_status}")

            # Run full cycle
            self.performance_monitor.start_timer('full_cycle')
            result = await self.crew.run_full_cycle()
            self.performance_monitor.end_timer('full_cycle')

            # Log results
            logger.info("Simulation cycle completed", result=result)

            # Get system metrics
            metrics = await self.db_manager.get_system_metrics()
            logger.info("System metrics", metrics=metrics.dict())

            return result

        except Exception as e:
            logger.error(f"Error in simulation mode: {e}")
            await self.alert_manager.send_alert(
                'error', 'simulation', f'Simulation error: {e}'
            )
            raise

    async def run_live_mode(self):
        """Run the system in live mode."""
        try:
            logger.info("Starting ABMBA system in live mode")

            # Run health check
            health_status = await self.health_checker.check_system_health()
            if health_status['overall'] != 'healthy':
                logger.error(f"System health check failed: {health_status}")
                await self.alert_manager.send_alert(
                    'critical', 'system', f'System health check failed: {health_status}'
                )
                return

            # Run full cycle
            self.performance_monitor.start_timer('full_cycle')
            result = await self.crew.run_full_cycle()
            self.performance_monitor.end_timer('full_cycle')

            # Log results
            logger.info("Live cycle completed", result=result)

            # Get system metrics
            metrics = await self.db_manager.get_system_metrics()
            logger.info("System metrics", metrics=metrics.dict())

            # Check for critical conditions
            current_bankroll = await self.db_manager.get_current_bankroll()
            initial_bankroll = self.config['bankroll']['initial_amount']
            min_bankroll = initial_bankroll * (self.config['bankroll']['min_bankroll_percentage'] / 100)

            if current_bankroll < min_bankroll:
                await self.alert_manager.send_alert(
                    'critical', 'bankroll',
                    f'Bankroll below minimum threshold: ${current_bankroll} < ${min_bankroll}'
                )

            return result

        except Exception as e:
            logger.error(f"Error in live mode: {e}")
            await self.alert_manager.send_alert(
                'error', 'live', f'Live mode error: {e}'
            )
            raise

    async def run_backtest_mode(self, start_date: str, end_date: str):
        """Run the system in backtest mode."""
        try:
            logger.info(f"Starting ABMBA system in backtest mode: {start_date} to {end_date}")

            # This would implement backtesting with historical data
            # For now, run simulation mode
            return await self.run_simulation_mode()

        except Exception as e:
            logger.error(f"Error in backtest mode: {e}")
            await self.alert_manager.send_alert(
                'error', 'backtest', f'Backtest error: {e}'
            )
            raise

    async def run_continuous(self, interval_minutes: int = 60):
        """Run the system continuously with specified interval."""
        try:
            logger.info(f"Starting ABMBA system in continuous mode (interval: {interval_minutes} minutes)")

            self.running = True

            while self.running:
                try:
                    # Check system health
                    health_status = await self.health_checker.check_system_health()

                    if health_status['overall'] == 'healthy':
                        # Run appropriate mode
                        if self.config['system']['mode'] == 'simulation':
                            await self.run_simulation_mode()
                        elif self.config['system']['mode'] == 'live':
                            await self.run_live_mode()
                        else:
                            logger.warning(f"Unknown mode: {self.config['system']['mode']}")

                    else:
                        logger.error(f"System health check failed: {health_status}")
                        await self.alert_manager.send_alert(
                            'critical', 'system', f'System health check failed: {health_status}'
                        )

                    # Wait for next cycle
                    await asyncio.sleep(interval_minutes * 60)

                except Exception as e:
                    logger.error(f"Error in continuous mode cycle: {e}")
                    await self.alert_manager.send_alert(
                        'error', 'continuous', f'Continuous mode error: {e}'
                    )
                    await asyncio.sleep(60)  # Wait 1 minute before retrying

        except Exception as e:
            logger.error(f"Error in continuous mode: {e}")
            await self.alert_manager.send_alert(
                'error', 'continuous', f'Continuous mode error: {e}'
            )
            raise

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    async def shutdown(self):
        """Shutdown the ABMBA system."""
        try:
            logger.info("Shutting down ABMBA system...")

            # Stop continuous mode
            self.running = False

            # Close components
            if hasattr(self, 'executor'):
                await self.executor.close()

            if hasattr(self, 'crew'):
                await self.crew.close()

            if hasattr(self, 'db_manager'):
                await self.db_manager.close()

            # Send shutdown alert
            if self.alert_manager:
                await self.alert_manager.send_alert(
                    'info', 'system', 'ABMBA system shut down successfully'
                )

            logger.info("ABMBA system shut down successfully")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='ABMBA - Autonomous Bankroll Management and Betting Agent')
    parser.add_argument('--config', '-c', default='config.yaml', help='Configuration file path')
    parser.add_argument('--mode', '-m', choices=['simulation', 'live', 'backtest', 'continuous'],
                       help='Operation mode')
    parser.add_argument('--interval', '-i', type=int, default=60,
                       help='Interval in minutes for continuous mode')
    parser.add_argument('--start-date', help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--health-check', action='store_true', help='Run health check only')
    parser.add_argument('--export', action='store_true', help='Export data and exit')

    args = parser.parse_args()

    try:
        # Initialize system
        system = ABMBASystem(args.config)
        await system.initialize()

        # Health check only
        if args.health_check:
            health_status = await system.health_checker.check_system_health()
            print(f"System Health: {health_status}")
            return

        # Export data only
        if args.export:
            from utils import DataExporter
            exporter = DataExporter(system.db_manager)
            await exporter.export_betting_history()
            await exporter.export_system_metrics()
            return

        # Determine mode
        mode = args.mode or system.config['system']['mode']

        # Run appropriate mode
        if mode == 'simulation':
            await system.run_simulation_mode()
        elif mode == 'live':
            await system.run_live_mode()
        elif mode == 'backtest':
            if not args.start_date or not args.end_date:
                print("Error: Start date and end date required for backtest mode")
                return
            await system.run_backtest_mode(args.start_date, args.end_date)
        elif mode == 'continuous':
            await system.run_continuous(args.interval)
        else:
            print(f"Unknown mode: {mode}")
            return

        # Shutdown
        await system.shutdown()

    except KeyboardInterrupt:
        print("\nShutting down...")
        if 'system' in locals():
            await system.shutdown()
    except Exception as e:
        print(f"Error: {e}")
        if 'system' in locals():
            await system.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
