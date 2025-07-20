"""
Fund Management Integration for ABMBA System
Comprehensive integration of balance monitoring, fund management, and human-agent interaction
"""

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import structlog
import yaml
from balance_monitor import BalanceInfo, BalanceMonitor, BalanceStatus, FundManager
from browserbase_executor import BrowserBaseConfig
from browserbase_integration import BrowserBaseBettingIntegration
from human_agent_interface import (
    FundManagementAgent,
    FundRequestStatus,
    HumanAgentInterface,
)

logger = structlog.get_logger()


class FundManagementMode(Enum):
    """Fund management mode enumeration."""
    MANUAL = "manual"
    AUTO = "auto"
    HYBRID = "hybrid"


@dataclass
class FundManagementConfig:
    """Configuration for fund management."""
    mode: FundManagementMode = FundManagementMode.HYBRID
    auto_request_threshold: Decimal = Decimal("50.00")
    request_amount: Decimal = Decimal("200.00")
    minimum_balance: Decimal = Decimal("100.00")
    check_interval: int = 300  # 5 minutes
    notification_channels: list[str] = None
    platforms: list[str] = None

    def __post_init__(self):
        if self.notification_channels is None:
            self.notification_channels = ["console", "email"]
        if self.platforms is None:
            self.platforms = ["fanduel", "draftkings"]


class ABMBAFundManagementSystem:
    """Complete fund management system for ABMBA."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.browserbase_config = self._create_browserbase_config()
        self.fund_config = self._create_fund_config()

        # Initialize components
        self.balance_monitor = None
        self.fund_manager = None
        self.human_interface = None
        self.fund_agent = None
        self.betting_integration = None

        # State management
        self.is_running = False
        self.monitoring_task = None
        self.last_balance_check = {}
        self.balance_cache = {}

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path) as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def _create_browserbase_config(self) -> BrowserBaseConfig:
        """Create BrowserBase configuration."""
        return BrowserBaseConfig(
            api_key=os.getenv("BROWSERBASE_API_KEY"),
            project_id=os.getenv("BROWSERBASE_PROJECT_ID"),
            stealth_mode=self.config.get('security', {}).get('browserbase', {}).get('enabled', True),
            proxy_enabled=self.config.get('security', {}).get('proxy_enabled', False),
            proxy_url=os.getenv("BROWSERBASE_PROXY"),
            viewport_width=self.config.get('security', {}).get('viewport_width', 1920),
            viewport_height=self.config.get('security', {}).get('viewport_height', 1080)
        )

    def _create_fund_config(self) -> FundManagementConfig:
        """Create fund management configuration."""
        fund_config = self.config.get('fund_management', {})

        return FundManagementConfig(
            mode=FundManagementMode(fund_config.get('mode', 'hybrid')),
            auto_request_threshold=Decimal(str(fund_config.get('auto_request_threshold', 50.00))),
            request_amount=Decimal(str(fund_config.get('request_amount', 200.00))),
            minimum_balance=Decimal(str(fund_config.get('minimum_balance', 100.00))),
            check_interval=fund_config.get('check_interval', 300),
            notification_channels=fund_config.get('notification_channels', ['console', 'email']),
            platforms=fund_config.get('platforms', ['fanduel', 'draftkings'])
        )

    async def initialize(self):
        """Initialize the fund management system."""
        try:
            logger.info("Initializing ABMBA Fund Management System")

            # Initialize balance monitor
            self.balance_monitor = BalanceMonitor(self.browserbase_config)
            await self.balance_monitor.__aenter__()

            # Initialize fund manager
            self.fund_manager = FundManager(self.balance_monitor)

            # Initialize human interface
            self.human_interface = HumanAgentInterface(self.fund_manager)

            # Initialize fund agent
            self.fund_agent = FundManagementAgent(self.human_interface)
            self.fund_agent.auto_request_threshold = self.fund_config.auto_request_threshold
            self.fund_agent.request_amount = self.fund_config.request_amount
            self.fund_agent.minimum_balance = self.fund_config.minimum_balance

            # Initialize betting integration
            self.betting_integration = BrowserBaseBettingIntegration()
            await self.betting_integration.initialize()

            # Register notification handlers
            self._register_notification_handlers()

            logger.info("ABMBA Fund Management System initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing fund management system: {e}")
            raise

    async def close(self):
        """Close the fund management system."""
        try:
            if self.is_running:
                await self.stop_monitoring()

            if self.balance_monitor:
                await self.balance_monitor.__aexit__(None, None, None)

            if self.betting_integration:
                await self.betting_integration.close()

            logger.info("ABMBA Fund Management System closed")

        except Exception as e:
            logger.error(f"Error closing fund management system: {e}")

    def _register_notification_handlers(self):
        """Register notification handlers based on configuration."""
        try:
            for channel in self.fund_config.notification_channels:
                if channel == "email":
                    self.human_interface.register_notification_callback(self._email_notification)
                elif channel == "slack":
                    self.human_interface.register_notification_callback(self._slack_notification)
                elif channel == "sms":
                    self.human_interface.register_notification_callback(self._sms_notification)
                elif channel == "webhook":
                    self.human_interface.register_notification_callback(self._webhook_notification)
                elif channel == "console":
                    self.human_interface.register_notification_callback(self._console_notification)

        except Exception as e:
            logger.error(f"Error registering notification handlers: {e}")

    async def start_monitoring(self):
        """Start continuous balance monitoring."""
        try:
            if self.is_running:
                logger.warning("Monitoring is already running")
                return

            self.is_running = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

            logger.info("Started continuous balance monitoring")

        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
            raise

    async def stop_monitoring(self):
        """Stop continuous balance monitoring."""
        try:
            if not self.is_running:
                logger.warning("Monitoring is not running")
                return

            self.is_running = False

            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass

            logger.info("Stopped continuous balance monitoring")

        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while self.is_running:
                try:
                    # Check balances
                    await self._check_all_balances()

                    # Handle fund requests based on mode
                    if self.fund_config.mode in [FundManagementMode.AUTO, FundManagementMode.HYBRID]:
                        await self._handle_auto_fund_requests()

                    # Wait for next check
                    await asyncio.sleep(self.fund_config.check_interval)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute before retrying

        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")

    async def _check_all_balances(self):
        """Check balances on all platforms."""
        try:
            credentials = self._get_all_platform_credentials()

            for platform, creds in credentials.items():
                if platform not in self.fund_config.platforms:
                    continue

                try:
                    balance_info = await self.balance_monitor.check_balance(platform, creds)
                    if balance_info:
                        self.balance_cache[platform] = balance_info
                        self.last_balance_check[platform] = datetime.utcnow()

                        # Check for fund needs
                        if balance_info.status in [BalanceStatus.LOW, BalanceStatus.CRITICAL, BalanceStatus.DEPLETED]:
                            await self._handle_low_balance(platform, balance_info)

                except Exception as e:
                    logger.error(f"Error checking balance for {platform}: {e}")

        except Exception as e:
            logger.error(f"Error checking all balances: {e}")

    async def _handle_low_balance(self, platform: str, balance_info: BalanceInfo):
        """Handle low balance situation."""
        try:
            if self.fund_config.mode == FundManagementMode.MANUAL:
                # Only notify, don't auto-request
                await self._notify_low_balance(platform, balance_info)
            elif self.fund_config.mode == FundManagementMode.AUTO:
                # Auto-request funds
                await self._auto_request_funds(platform, balance_info)
            elif self.fund_config.mode == FundManagementMode.HYBRID:
                # Notify and auto-request if no pending requests
                await self._notify_low_balance(platform, balance_info)
                await self._auto_request_funds(platform, balance_info)

        except Exception as e:
            logger.error(f"Error handling low balance for {platform}: {e}")

    async def _notify_low_balance(self, platform: str, balance_info: BalanceInfo):
        """Send notifications about low balance."""
        try:
            message = f"LOW BALANCE ALERT: {platform.upper()} balance is ${balance_info.current_balance} ({balance_info.status.value})"

            for callback in self.human_interface.notification_callbacks:
                try:
                    await callback(platform, balance_info)
                except Exception as e:
                    logger.error(f"Error in notification callback: {e}")

        except Exception as e:
            logger.error(f"Error notifying low balance: {e}")

    async def _auto_request_funds(self, platform: str, balance_info: BalanceInfo):
        """Automatically request funds for low balance."""
        try:
            # Check if we already have a pending request
            pending_requests = await self.human_interface.get_fund_requests(FundRequestStatus.PENDING)
            platform_requests = [r for r in pending_requests if r["platform"] == platform]

            if not platform_requests:
                # Create new fund request
                request_id = await self.human_interface.request_funds(
                    platform=platform,
                    amount=self.fund_config.request_amount,
                    reason=f"Auto-request: Balance ${balance_info.current_balance} below threshold ${self.fund_config.auto_request_threshold}"
                )
                logger.info(f"Auto-requested funds for {platform}: ${self.fund_config.request_amount} (ID: {request_id})")
            else:
                logger.info(f"Fund request already pending for {platform}")

        except Exception as e:
            logger.error(f"Error auto-requesting funds for {platform}: {e}")

    async def _handle_auto_fund_requests(self):
        """Handle automatic fund requests."""
        try:
            request_ids = await self.fund_agent.monitor_and_request_funds()

            if request_ids:
                logger.info(f"Created {len(request_ids)} auto fund requests")

        except Exception as e:
            logger.error(f"Error handling auto fund requests: {e}")

    async def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        try:
            status = {
                "timestamp": datetime.utcnow().isoformat(),
                "is_running": self.is_running,
                "mode": self.fund_config.mode.value,
                "platforms": {},
                "pending_requests": [],
                "recent_deposits": {},
                "configuration": {
                    "auto_request_threshold": float(self.fund_config.auto_request_threshold),
                    "request_amount": float(self.fund_config.request_amount),
                    "minimum_balance": float(self.fund_config.minimum_balance),
                    "check_interval": self.fund_config.check_interval,
                    "notification_channels": self.fund_config.notification_channels
                }
            }

            # Add platform balances
            for platform, balance_info in self.balance_cache.items():
                status["platforms"][platform] = {
                    "balance": float(balance_info.current_balance),
                    "status": balance_info.status.value,
                    "last_updated": balance_info.last_updated.isoformat(),
                    "warnings": balance_info.threshold_warnings
                }

            # Add pending requests
            pending_requests = await self.human_interface.get_fund_requests(FundRequestStatus.PENDING)
            status["pending_requests"] = pending_requests

            # Add recent deposits
            for platform in self.fund_config.platforms:
                if platform in self.human_interface.deposit_history:
                    status["recent_deposits"][platform] = self.human_interface.deposit_history[platform][-5:]

            return status

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}

    async def request_funds_manual(self, platform: str, amount: Decimal, reason: str = "Manual request") -> str:
        """Manually request funds."""
        try:
            request_id = await self.human_interface.request_funds(platform, amount, reason)
            logger.info(f"Manual fund request created: {request_id} for {platform} - ${amount}")
            return request_id

        except Exception as e:
            logger.error(f"Error creating manual fund request: {e}")
            raise

    async def confirm_deposit(self, platform: str, amount: Decimal, confirmation_id: str = None) -> bool:
        """Confirm a deposit made by the human."""
        try:
            success = await self.human_interface.confirm_deposit(platform, amount, confirmation_id)

            if success:
                logger.info(f"Deposit confirmed for {platform}: ${amount}")

                # Update balance cache
                credentials = self._get_platform_credentials(platform)
                if credentials:
                    balance_info = await self.balance_monitor.check_balance(platform, credentials)
                    if balance_info:
                        self.balance_cache[platform] = balance_info

            return success

        except Exception as e:
            logger.error(f"Error confirming deposit: {e}")
            return False

    async def get_balance_summary(self, platform: str = None) -> dict[str, Any]:
        """Get balance summary."""
        try:
            return await self.human_interface.get_balance_summary(platform)

        except Exception as e:
            logger.error(f"Error getting balance summary: {e}")
            return {"error": str(e)}

    async def get_fund_requests(self, status: FundRequestStatus = None) -> list[dict[str, Any]]:
        """Get fund requests."""
        try:
            return await self.human_interface.get_fund_requests(status)

        except Exception as e:
            logger.error(f"Error getting fund requests: {e}")
            return []

    def _get_platform_credentials(self, platform: str) -> dict[str, str] | None:
        """Get platform credentials."""
        platform_upper = platform.upper()

        username = os.getenv(f"{platform_upper}_USERNAME")
        password = os.getenv(f"{platform_upper}_PASSWORD")

        if username and password:
            return {
                "username": username,
                "password": password
            }

        return None

    def _get_all_platform_credentials(self) -> dict[str, dict[str, str]]:
        """Get credentials for all platforms."""
        credentials = {}

        for platform in self.fund_config.platforms:
            creds = self._get_platform_credentials(platform)
            if creds:
                credentials[platform] = creds

        return credentials

    # Notification handlers
    async def _console_notification(self, platform: str, balance_info: BalanceInfo):
        """Console notification handler."""
        print(f"\n🚨 FUND ALERT: {platform.upper()}")
        print(f"   Current Balance: ${balance_info.current_balance}")
        print(f"   Status: {balance_info.status.value}")
        print(f"   Warnings: {balance_info.threshold_warnings}")
        print(f"   Time: {balance_info.last_updated.strftime('%Y-%m-%d %H:%M:%S')}\n")

    async def _email_notification(self, platform: str, balance_info: BalanceInfo):
        """Email notification handler."""
        # Implementation would use your email service
        logger.info(f"EMAIL: {platform} needs funds - balance: ${balance_info.current_balance}")

    async def _slack_notification(self, platform: str, balance_info: BalanceInfo):
        """Slack notification handler."""
        # Implementation would use your Slack webhook
        logger.info(f"SLACK: {platform} needs funds - balance: ${balance_info.current_balance}")

    async def _sms_notification(self, platform: str, balance_info: BalanceInfo):
        """SMS notification handler."""
        # Implementation would use your SMS service
        logger.info(f"SMS: {platform} needs funds - balance: ${balance_info.current_balance}")

    async def _webhook_notification(self, platform: str, balance_info: BalanceInfo):
        """Webhook notification handler."""
        # Implementation would use your webhook service
        logger.info(f"WEBHOOK: {platform} needs funds - balance: ${balance_info.current_balance}")


# CLI interface for the complete system
class ABMBAFundManagementCLI:
    """Command-line interface for the complete fund management system."""

    def __init__(self, system: ABMBAFundManagementSystem):
        self.system = system

    async def run_interactive(self):
        """Run interactive CLI."""
        print("💰 ABMBA Fund Management System")
        print("=" * 50)

        while True:
            print("\nOptions:")
            print("1. Start monitoring")
            print("2. Stop monitoring")
            print("3. Check all balances")
            print("4. Check specific platform balance")
            print("5. Request funds manually")
            print("6. Confirm deposit")
            print("7. View pending requests")
            print("8. Get system status")
            print("9. Get fund management report")
            print("10. Exit")

            choice = input("\nEnter your choice (1-10): ").strip()

            if choice == "1":
                await self._start_monitoring()
            elif choice == "2":
                await self._stop_monitoring()
            elif choice == "3":
                await self._check_all_balances()
            elif choice == "4":
                await self._check_platform_balance()
            elif choice == "5":
                await self._request_funds()
            elif choice == "6":
                await self._confirm_deposit()
            elif choice == "7":
                await self._view_pending_requests()
            elif choice == "8":
                await self._get_system_status()
            elif choice == "9":
                await self._get_report()
            elif choice == "10":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

    async def _start_monitoring(self):
        """Start monitoring."""
        try:
            await self.system.start_monitoring()
            print("✅ Monitoring started successfully")
        except Exception as e:
            print(f"❌ Error starting monitoring: {e}")

    async def _stop_monitoring(self):
        """Stop monitoring."""
        try:
            await self.system.stop_monitoring()
            print("✅ Monitoring stopped successfully")
        except Exception as e:
            print(f"❌ Error stopping monitoring: {e}")

    async def _check_all_balances(self):
        """Check all balances."""
        print("\nChecking all balances...")
        summary = await self.system.get_balance_summary()

        if "error" in summary:
            print(f"Error: {summary['error']}")
            return

        print(f"\nTotal Balance: ${summary['total_balance']}")
        print("\nPlatform Balances:")
        for platform, balance_info in summary["platforms"].items():
            status = balance_info["status"]
            balance = balance_info["current_balance"]
            print(f"  {platform.upper()}: ${balance} ({status})")

    async def _check_platform_balance(self):
        """Check specific platform balance."""
        platform = input("Enter platform (fanduel/draftkings): ").strip().lower()

        if platform not in ["fanduel", "draftkings"]:
            print("Invalid platform. Please enter 'fanduel' or 'draftkings'.")
            return

        print(f"\nChecking {platform} balance...")
        summary = await self.system.get_balance_summary(platform)

        if "error" in summary:
            print(f"Error: {summary['error']}")
            return

        balance_info = summary["balance"]
        print(f"\n{platform.upper()} Balance: ${balance_info['current_balance']} ({balance_info['status']})")
        print(f"Last Updated: {balance_info['last_updated']}")

    async def _request_funds(self):
        """Request funds manually."""
        platform = input("Enter platform (fanduel/draftkings): ").strip().lower()
        amount_str = input("Enter amount: $").strip()
        reason = input("Enter reason (optional): ").strip() or "Manual request"

        try:
            amount = Decimal(amount_str)
            request_id = await self.system.request_funds_manual(platform, amount, reason)
            print(f"\n✅ Fund request created: {request_id}")
        except Exception as e:
            print(f"Error: {e}")

    async def _confirm_deposit(self):
        """Confirm a deposit."""
        platform = input("Enter platform (fanduel/draftkings): ").strip().lower()
        amount_str = input("Enter deposit amount: $").strip()
        confirmation_id = input("Enter confirmation ID (optional): ").strip() or None

        try:
            amount = Decimal(amount_str)
            success = await self.system.confirm_deposit(platform, amount, confirmation_id)

            if success:
                print(f"\n✅ Deposit confirmed for {platform}: ${amount}")
            else:
                print(f"\n❌ Failed to confirm deposit for {platform}")
        except Exception as e:
            print(f"Error: {e}")

    async def _view_pending_requests(self):
        """View pending requests."""
        requests = await self.system.get_fund_requests(FundRequestStatus.PENDING)

        if not requests:
            print("\nNo pending fund requests.")
            return

        print(f"\nPending Fund Requests ({len(requests)}):")
        for request in requests:
            print(f"  ID: {request['request_id']}")
            print(f"  Platform: {request['platform'].upper()}")
            print(f"  Amount: ${request['amount']}")
            print(f"  Reason: {request['reason']}")
            print(f"  Requested: {request['requested_at']}")
            print()

    async def _get_system_status(self):
        """Get system status."""
        print("\nGetting system status...")
        status = await self.system.get_system_status()

        if "error" in status:
            print(f"Error: {status['error']}")
            return

        print("\n📊 System Status")
        print(f"Running: {status['is_running']}")
        print(f"Mode: {status['mode']}")
        print(f"Total Balance: ${status['platforms']}")
        print(f"Pending Requests: {len(status['pending_requests'])}")

    async def _get_report(self):
        """Get fund management report."""
        print("\nGenerating fund management report...")
        report = await self.system.fund_agent.get_fund_management_report()

        if "error" in report:
            print(f"Error: {report['error']}")
            return

        print("\n📊 Fund Management Report")
        print(f"Generated: {report['timestamp']}")
        print(f"Total Balance: ${report['balance_summary']['total_balance']}")
        print(f"Pending Requests: {len(report['pending_requests'])}")

        # Save report to file
        filename = f"abmba_fund_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Report saved to: {filename}")


# Example usage
async def main():
    """Example usage of the complete fund management system."""
    system = ABMBAFundManagementSystem()

    try:
        await system.initialize()

        cli = ABMBAFundManagementCLI(system)
        await cli.run_interactive()

    except Exception as e:
        logger.error(f"Error in main: {e}")

    finally:
        await system.close()


if __name__ == "__main__":
    asyncio.run(main())
