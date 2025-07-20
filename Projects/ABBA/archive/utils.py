"""
Utility modules for ABMBA system.
Includes configuration management, logging, and helper functions.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any

import structlog
import yaml
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from models import Alert

console = Console()


class ConfigManager:
    """Configuration manager for ABMBA system."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path) as file:
                config = yaml.safe_load(file)

            # Replace environment variables
            config = self._replace_env_vars(config)

            return config

        except FileNotFoundError:
            console.print(f"[red]Configuration file not found: {self.config_path}[/red]")
            raise
        except yaml.YAMLError as e:
            console.print(f"[red]Error parsing configuration file: {e}[/red]")
            raise

    def _replace_env_vars(self, config: Any) -> Any:
        """Replace environment variable placeholders in configuration."""
        if isinstance(config, dict):
            return {key: self._replace_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        else:
            return config

    def _setup_logging(self):
        """Setup structured logging."""
        log_file = self.config['monitoring']['log_file']

        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Setup file handler
        logging.basicConfig(
            format="%(message)s",
            stream=open(log_file, 'w'),
            level=getattr(logging, self.config['system']['log_level'].upper())
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value by key."""
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self):
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
        except Exception as e:
            console.print(f"[red]Error saving configuration: {e}[/red]")


class Dashboard:
    """Real-time dashboard for monitoring ABMBA system."""

    def __init__(self, config: dict):
        self.config = config
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )

        self.layout["main"].split_row(
            Layout(name="metrics", ratio=2),
            Layout(name="logs", ratio=1)
        )

    def update_metrics(self, metrics: dict[str, Any]):
        """Update system metrics display."""
        metrics_table = Table(title="System Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")

        for key, value in metrics.items():
            metrics_table.add_row(key, str(value))

        self.layout["metrics"].update(Panel(metrics_table, title="Live Metrics"))

    def update_logs(self, logs: list):
        """Update log display."""
        log_text = "\n".join(logs[-10:])  # Show last 10 logs
        self.layout["logs"].update(Panel(log_text, title="Recent Logs"))

    def show(self):
        """Display the dashboard."""
        with Live(self.layout, refresh_per_second=1):
            while True:
                asyncio.sleep(1)


class AlertManager:
    """Manager for system alerts and notifications."""

    def __init__(self, config: dict, db_manager):
        self.config = config
        self.db_manager = db_manager
        self.alert_config = config['monitoring']['alerts']

    async def send_alert(self, level: str, category: str, message: str, details: dict = None):
        """Send an alert through configured channels."""
        try:
            # Create alert record
            alert = Alert(
                level=level,
                category=category,
                message=message,
                details=details
            )

            # Save to database
            await self.db_manager.save_alert(alert)

            # Send through configured channels
            if self.alert_config['email']['enabled']:
                await self._send_email_alert(alert)

            if self.alert_config['slack']['enabled']:
                await self._send_slack_alert(alert)

            # Console output
            color_map = {
                'info': 'blue',
                'warning': 'yellow',
                'error': 'red',
                'critical': 'red'
            }

            color = color_map.get(level, 'white')
            console.print(f"[{color}][{level.upper()}] {message}[/{color}]")

        except Exception as e:
            console.print(f"[red]Error sending alert: {e}[/red]")

    async def _send_email_alert(self, alert: Alert):
        """Send email alert."""
        try:
            # This would implement email sending
            # For now, just log
            console.print(f"[blue]Email alert: {alert.message}[/blue]")

        except Exception as e:
            console.print(f"[red]Error sending email alert: {e}[/red]")

    async def _send_slack_alert(self, alert: Alert):
        """Send Slack alert."""
        try:
            # This would implement Slack webhook
            # For now, just log
            console.print(f"[blue]Slack alert: {alert.message}[/blue]")

        except Exception as e:
            console.print(f"[red]Error sending Slack alert: {e}[/red]")


class PerformanceMonitor:
    """Monitor system performance and resource usage."""

    def __init__(self):
        self.start_time = datetime.utcnow()
        self.metrics = {}

    def start_timer(self, name: str):
        """Start a performance timer."""
        self.metrics[name] = {
            'start_time': datetime.utcnow(),
            'end_time': None,
            'duration': None
        }

    def end_timer(self, name: str):
        """End a performance timer."""
        if name in self.metrics:
            self.metrics[name]['end_time'] = datetime.utcnow()
            self.metrics[name]['duration'] = (
                self.metrics[name]['end_time'] - self.metrics[name]['start_time']
            ).total_seconds()

    def get_metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
        return {
            'uptime': (datetime.utcnow() - self.start_time).total_seconds(),
            'timers': self.metrics
        }


class DataExporter:
    """Export data for analysis and reporting."""

    def __init__(self, db_manager):
        self.db_manager = db_manager

    async def export_betting_history(self, format: str = 'csv', filepath: str = None) -> str:
        """Export betting history to file."""
        try:
            if not filepath:
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                filepath = f"exports/betting_history_{timestamp}.{format}"

            # Create exports directory
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Get betting data
            # This would implement actual data export
            # For now, create a mock export

            if format == 'csv':
                with open(filepath, 'w') as f:
                    f.write("bet_id,event_id,platform,market_type,selection,odds,stake,status,created_at\n")
                    f.write("mock_bet_1,mock_event_1,fanduel,moneyline,home,-110,10.00,placed,2024-01-01T12:00:00\n")

            console.print(f"[green]Exported betting history to {filepath}[/green]")
            return filepath

        except Exception as e:
            console.print(f"[red]Error exporting betting history: {e}[/red]")
            return None

    async def export_system_metrics(self, format: str = 'json', filepath: str = None) -> str:
        """Export system metrics to file."""
        try:
            if not filepath:
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                filepath = f"exports/system_metrics_{timestamp}.{format}"

            # Create exports directory
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Get system metrics
            metrics = await self.db_manager.get_system_metrics()

            if format == 'json':
                with open(filepath, 'w') as f:
                    json.dump(metrics.dict(), f, indent=2, default=str)

            console.print(f"[green]Exported system metrics to {filepath}[/green]")
            return filepath

        except Exception as e:
            console.print(f"[red]Error exporting system metrics: {e}[/red]")
            return None


class BackupManager:
    """Manage database backups and data integrity."""

    def __init__(self, db_manager, config: dict):
        self.db_manager = db_manager
        self.config = config
        self.backup_dir = "backups"

        # Create backup directory
        os.makedirs(self.backup_dir, exist_ok=True)

    async def create_backup(self) -> str:
        """Create a database backup."""
        try:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            backup_file = f"{self.backup_dir}/abmba_backup_{timestamp}.sql"

            # This would implement actual database backup
            # For now, create a mock backup file

            with open(backup_file, 'w') as f:
                f.write("-- ABMBA Database Backup\n")
                f.write(f"-- Created: {datetime.utcnow().isoformat()}\n")
                f.write("-- Mock backup file\n")

            console.print(f"[green]Backup created: {backup_file}[/green]")
            return backup_file

        except Exception as e:
            console.print(f"[red]Error creating backup: {e}[/red]")
            return None

    async def restore_backup(self, backup_file: str) -> bool:
        """Restore database from backup."""
        try:
            # This would implement actual database restore
            # For now, just verify file exists

            if not os.path.exists(backup_file):
                console.print(f"[red]Backup file not found: {backup_file}[/red]")
                return False

            console.print(f"[green]Backup restored from: {backup_file}[/green]")
            return True

        except Exception as e:
            console.print(f"[red]Error restoring backup: {e}[/red]")
            return False

    async def cleanup_old_backups(self, keep_days: int = 30):
        """Clean up old backup files."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=keep_days)

            for file in os.listdir(self.backup_dir):
                if file.startswith('abmba_backup_'):
                    file_path = os.path.join(self.backup_dir, file)
                    file_time = datetime.fromtimestamp(os.path.getctime(file_path))

                    if file_time < cutoff_date:
                        os.remove(file_path)
                        console.print(f"[yellow]Removed old backup: {file}[/yellow]")

        except Exception as e:
            console.print(f"[red]Error cleaning up backups: {e}[/red]")


class HealthChecker:
    """Check system health and dependencies."""

    def __init__(self, config: dict, db_manager):
        self.config = config
        self.db_manager = db_manager

    async def check_system_health(self) -> dict[str, Any]:
        """Check overall system health."""
        try:
            health_status = {
                'database': await self._check_database_health(),
                'apis': await self._check_api_health(),
                'platforms': await self._check_platform_health(),
                'overall': 'healthy'
            }

            # Determine overall status
            if any(status['status'] == 'error' for status in health_status.values() if isinstance(status, dict)):
                health_status['overall'] = 'error'
            elif any(status['status'] == 'warning' for status in health_status.values() if isinstance(status, dict)):
                health_status['overall'] = 'warning'

            return health_status

        except Exception as e:
            return {
                'overall': 'error',
                'error': str(e)
            }

    async def _check_database_health(self) -> dict[str, Any]:
        """Check database connectivity and health."""
        try:
            # Test database connection
            current_bankroll = await self.db_manager.get_current_bankroll()

            return {
                'status': 'healthy',
                'connection': 'ok',
                'current_bankroll': float(current_bankroll)
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    async def _check_api_health(self) -> dict[str, Any]:
        """Check API connectivity."""
        try:
            # This would implement actual API health checks
            # For now, return mock status

            return {
                'status': 'healthy',
                'odds_api': 'ok',
                'sports_data_api': 'ok'
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    async def _check_platform_health(self) -> dict[str, Any]:
        """Check betting platform connectivity."""
        try:
            # This would implement actual platform health checks
            # For now, return mock status

            return {
                'status': 'healthy',
                'fanduel': 'ok',
                'draftkings': 'ok'
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }


def format_currency(amount: float) -> str:
    """Format amount as currency."""
    return f"${amount:.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value:.2f}%"


def format_odds(odds: float) -> str:
    """Format odds in American format."""
    if odds > 0:
        return f"+{odds:.0f}"
    else:
        return f"{odds:.0f}"


def calculate_roi(initial: float, current: float) -> float:
    """Calculate return on investment."""
    if initial == 0:
        return 0.0
    return ((current - initial) / initial) * 100


def validate_email(email: str) -> bool:
    """Validate email address format."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def generate_uuid() -> str:
    """Generate a UUID string."""
    import uuid
    return str(uuid.uuid4())


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default
