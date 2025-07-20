"""
Human-Agent Interface for Fund Management
Manages communication between human and agent for fund synchronization
"""

import asyncio
import json
import os
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import structlog
from balance_monitor import BalanceInfo, BalanceMonitor, BalanceStatus, FundManager
from browserbase_executor import BrowserBaseConfig

logger = structlog.get_logger()


class NotificationType(Enum):
    """Notification type enumeration."""
    EMAIL = "email"
    SLACK = "slack"
    CONSOLE = "console"
    SMS = "sms"
    WEBHOOK = "webhook"


class FundRequestStatus(Enum):
    """Fund request status enumeration."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class FundRequest:
    """Fund request structure."""
    platform: str
    amount: Decimal
    reason: str
    status: FundRequestStatus
    requested_at: datetime
    confirmed_at: datetime | None = None
    confirmation_id: str | None = None
    notes: str | None = None


class HumanAgentInterface:
    """Interface for human-agent fund management communication."""

    def __init__(self, fund_manager: FundManager):
        self.fund_manager = fund_manager
        self.pending_requests: dict[str, FundRequest] = {}
        self.notification_handlers: dict[NotificationType, Callable] = {}
        self.balance_history: dict[str, list[BalanceInfo]] = {}
        self.deposit_history: dict[str, list[dict]] = {}

        # Register default notification handlers
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default notification handlers."""
        self.notification_handlers[NotificationType.CONSOLE] = self._console_notification
        self.notification_handlers[NotificationType.EMAIL] = self._email_notification
        self.notification_handlers[NotificationType.SLACK] = self._slack_notification
        self.notification_handlers[NotificationType.SMS] = self._sms_notification
        self.notification_handlers[NotificationType.WEBHOOK] = self._webhook_notification

    async def request_funds(self, platform: str, amount: Decimal, reason: str = "Low balance") -> str:
        """Request funds from human."""
        try:
            # Create fund request
            request = FundRequest(
                platform=platform,
                amount=amount,
                reason=reason,
                status=FundRequestStatus.PENDING,
                requested_at=datetime.utcnow()
            )

            # Generate unique request ID
            request_id = f"{platform}_{int(time.time())}_{random.randint(1000, 9999)}"

            # Store request
            self.pending_requests[request_id] = request

            # Send notifications
            await self._send_fund_request_notifications(request_id, request)

            logger.info(f"Fund request created: {request_id} for {platform} - ${amount}")
            return request_id

        except Exception as e:
            logger.error(f"Error creating fund request: {e}")
            raise

    async def confirm_deposit(self, platform: str, amount: Decimal, confirmation_id: str = None) -> bool:
        """Confirm a deposit made by the human."""
        try:
            # Confirm with fund manager
            success = await self.fund_manager.confirm_deposit(platform, amount, confirmation_id)

            if success:
                # Update any pending requests for this platform
                await self._update_pending_requests(platform, amount)

                # Store in deposit history
                if platform not in self.deposit_history:
                    self.deposit_history[platform] = []

                self.deposit_history[platform].append({
                    "amount": amount,
                    "timestamp": datetime.utcnow(),
                    "confirmation_id": confirmation_id
                })

                # Send confirmation notifications
                await self._send_deposit_confirmation_notifications(platform, amount)

                logger.info(f"Deposit confirmed for {platform}: ${amount}")
                return True
            else:
                logger.error(f"Failed to confirm deposit for {platform}: ${amount}")
                return False

        except Exception as e:
            logger.error(f"Error confirming deposit: {e}")
            return False

    async def get_balance_summary(self, platform: str = None) -> dict[str, Any]:
        """Get balance summary for all platforms or specific platform."""
        try:
            if platform:
                # Get specific platform balance
                credentials = self._get_platform_credentials(platform)
                if not credentials:
                    return {"error": f"No credentials found for {platform}"}

                balance_info = await self.fund_manager.balance_monitor.check_balance(platform, credentials)
                if not balance_info:
                    return {"error": f"Could not get balance for {platform}"}

                return {
                    "platform": platform,
                    "balance": asdict(balance_info),
                    "pending_requests": self._get_pending_requests_for_platform(platform),
                    "recent_deposits": self.deposit_history.get(platform, [])
                }
            else:
                # Get all platform balances
                credentials = self._get_all_platform_credentials()
                results = await self.fund_manager.monitor_all_platforms(credentials)

                summary = {
                    "platforms": {},
                    "total_balance": Decimal("0.00"),
                    "low_balance_platforms": [],
                    "critical_platforms": []
                }

                for platform, balance_info in results.items():
                    summary["platforms"][platform] = asdict(balance_info)
                    summary["total_balance"] += balance_info.current_balance

                    if balance_info.status == BalanceStatus.LOW:
                        summary["low_balance_platforms"].append(platform)
                    elif balance_info.status in [BalanceStatus.CRITICAL, BalanceStatus.DEPLETED]:
                        summary["critical_platforms"].append(platform)

                return summary

        except Exception as e:
            logger.error(f"Error getting balance summary: {e}")
            return {"error": str(e)}

    async def get_fund_requests(self, status: FundRequestStatus = None) -> list[dict[str, Any]]:
        """Get fund requests, optionally filtered by status."""
        try:
            requests = []

            for request_id, request in self.pending_requests.items():
                if status is None or request.status == status:
                    request_dict = asdict(request)
                    request_dict["request_id"] = request_id
                    requests.append(request_dict)

            return requests

        except Exception as e:
            logger.error(f"Error getting fund requests: {e}")
            return []

    async def update_request_status(self, request_id: str, status: FundRequestStatus, notes: str = None) -> bool:
        """Update the status of a fund request."""
        try:
            if request_id not in self.pending_requests:
                logger.error(f"Request ID not found: {request_id}")
                return False

            request = self.pending_requests[request_id]
            request.status = status
            request.notes = notes

            if status == FundRequestStatus.CONFIRMED:
                request.confirmed_at = datetime.utcnow()

            logger.info(f"Updated request {request_id} status to {status.value}")
            return True

        except Exception as e:
            logger.error(f"Error updating request status: {e}")
            return False

    async def _send_fund_request_notifications(self, request_id: str, request: FundRequest):
        """Send notifications about fund requests."""
        try:
            message = f"FUND REQUEST: {request.platform.upper()} needs ${request.amount} - {request.reason}"

            # Send to all notification handlers
            for notification_type, handler in self.notification_handlers.items():
                try:
                    await handler(
                        notification_type=notification_type,
                        platform=request.platform,
                        amount=request.amount,
                        reason=request.reason,
                        request_id=request_id,
                        message=message
                    )
                except Exception as e:
                    logger.error(f"Error sending {notification_type.value} notification: {e}")

        except Exception as e:
            logger.error(f"Error sending fund request notifications: {e}")

    async def _send_deposit_confirmation_notifications(self, platform: str, amount: Decimal):
        """Send notifications about deposit confirmations."""
        try:
            message = f"DEPOSIT CONFIRMED: {platform.upper()} received ${amount}"

            for notification_type, handler in self.notification_handlers.items():
                try:
                    await handler(
                        notification_type=notification_type,
                        platform=platform,
                        amount=amount,
                        message=message,
                        event_type="deposit_confirmed"
                    )
                except Exception as e:
                    logger.error(f"Error sending {notification_type.value} notification: {e}")

        except Exception as e:
            logger.error(f"Error sending deposit confirmation notifications: {e}")

    async def _update_pending_requests(self, platform: str, amount: Decimal):
        """Update pending requests after deposit confirmation."""
        try:
            for request_id, request in self.pending_requests.items():
                if (request.platform == platform and
                    request.status == FundRequestStatus.PENDING and
                    request.amount <= amount):

                    await self.update_request_status(
                        request_id,
                        FundRequestStatus.CONFIRMED,
                        f"Deposit of ${amount} confirmed"
                    )

        except Exception as e:
            logger.error(f"Error updating pending requests: {e}")

    def _get_pending_requests_for_platform(self, platform: str) -> list[dict[str, Any]]:
        """Get pending requests for a specific platform."""
        requests = []
        for request_id, request in self.pending_requests.items():
            if request.platform == platform and request.status == FundRequestStatus.PENDING:
                request_dict = asdict(request)
                request_dict["request_id"] = request_id
                requests.append(request_dict)
        return requests

    def _get_platform_credentials(self, platform: str) -> dict[str, str] | None:
        """Get platform credentials from environment variables."""
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

        platforms = ["fanduel", "draftkings"]
        for platform in platforms:
            creds = self._get_platform_credentials(platform)
            if creds:
                credentials[platform] = creds

        return credentials

    # Notification handlers
    async def _console_notification(self, **kwargs):
        """Console notification handler."""
        message = kwargs.get("message", "Notification")
        print(f"\n🔔 {message}")
        if "request_id" in kwargs:
            print(f"   Request ID: {kwargs['request_id']}")
        if "amount" in kwargs:
            print(f"   Amount: ${kwargs['amount']}")
        print()

    async def _email_notification(self, **kwargs):
        """Email notification handler."""
        # Implementation would use your email service
        logger.info(f"EMAIL: {kwargs.get('message', 'Notification')}")

    async def _slack_notification(self, **kwargs):
        """Slack notification handler."""
        # Implementation would use your Slack webhook
        logger.info(f"SLACK: {kwargs.get('message', 'Notification')}")

    async def _sms_notification(self, **kwargs):
        """SMS notification handler."""
        # Implementation would use your SMS service
        logger.info(f"SMS: {kwargs.get('message', 'Notification')}")

    async def _webhook_notification(self, **kwargs):
        """Webhook notification handler."""
        # Implementation would use your webhook service
        logger.info(f"WEBHOOK: {kwargs.get('message', 'Notification')}")


class FundManagementAgent:
    """Agent for automated fund management."""

    def __init__(self, interface: HumanAgentInterface):
        self.interface = interface
        self.minimum_balance = Decimal("100.00")  # $100 minimum
        self.auto_request_threshold = Decimal("50.00")  # Request funds when below $50
        self.request_amount = Decimal("200.00")  # Request $200 at a time

    async def monitor_and_request_funds(self) -> list[str]:
        """Monitor balances and automatically request funds when needed."""
        try:
            # Get balance summary
            summary = await self.interface.get_balance_summary()

            if "error" in summary:
                logger.error(f"Error getting balance summary: {summary['error']}")
                return []

            request_ids = []

            # Check each platform
            for platform, balance_info in summary["platforms"].items():
                current_balance = Decimal(str(balance_info["current_balance"]))

                if current_balance <= self.auto_request_threshold:
                    # Check if we already have a pending request
                    pending_requests = await self.interface.get_fund_requests(FundRequestStatus.PENDING)
                    platform_requests = [r for r in pending_requests if r["platform"] == platform]

                    if not platform_requests:
                        # Create new fund request
                        request_id = await self.interface.request_funds(
                            platform=platform,
                            amount=self.request_amount,
                            reason=f"Auto-request: Balance ${current_balance} below threshold ${self.auto_request_threshold}"
                        )
                        request_ids.append(request_id)
                        logger.info(f"Auto-requested funds for {platform}: ${self.request_amount}")

            return request_ids

        except Exception as e:
            logger.error(f"Error in monitor_and_request_funds: {e}")
            return []

    async def get_fund_management_report(self) -> dict[str, Any]:
        """Get comprehensive fund management report."""
        try:
            # Get balance summary
            summary = await self.interface.get_balance_summary()

            # Get pending requests
            pending_requests = await self.interface.get_fund_requests(FundRequestStatus.PENDING)

            # Get recent deposits
            recent_deposits = {}
            for platform in ["fanduel", "draftkings"]:
                if platform in self.interface.deposit_history:
                    recent_deposits[platform] = self.interface.deposit_history[platform][-5:]  # Last 5 deposits

            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "balance_summary": summary,
                "pending_requests": pending_requests,
                "recent_deposits": recent_deposits,
                "auto_request_settings": {
                    "minimum_balance": float(self.minimum_balance),
                    "auto_request_threshold": float(self.auto_request_threshold),
                    "request_amount": float(self.request_amount)
                }
            }

            return report

        except Exception as e:
            logger.error(f"Error getting fund management report: {e}")
            return {"error": str(e)}


# Example usage and CLI interface
class FundManagementCLI:
    """Command-line interface for fund management."""

    def __init__(self, agent: FundManagementAgent):
        self.agent = agent

    async def run_interactive(self):
        """Run interactive CLI."""
        print("💰 ABMBA Fund Management Interface")
        print("=" * 50)

        while True:
            print("\nOptions:")
            print("1. Check all balances")
            print("2. Check specific platform balance")
            print("3. Request funds")
            print("4. Confirm deposit")
            print("5. View pending requests")
            print("6. Get fund management report")
            print("7. Auto-monitor and request funds")
            print("8. Exit")

            choice = input("\nEnter your choice (1-8): ").strip()

            if choice == "1":
                await self._check_all_balances()
            elif choice == "2":
                await self._check_platform_balance()
            elif choice == "3":
                await self._request_funds()
            elif choice == "4":
                await self._confirm_deposit()
            elif choice == "5":
                await self._view_pending_requests()
            elif choice == "6":
                await self._get_report()
            elif choice == "7":
                await self._auto_monitor()
            elif choice == "8":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

    async def _check_all_balances(self):
        """Check all platform balances."""
        print("\nChecking all balances...")
        summary = await self.agent.interface.get_balance_summary()

        if "error" in summary:
            print(f"Error: {summary['error']}")
            return

        print(f"\nTotal Balance: ${summary['total_balance']}")
        print("\nPlatform Balances:")
        for platform, balance_info in summary["platforms"].items():
            status = balance_info["status"]
            balance = balance_info["current_balance"]
            print(f"  {platform.upper()}: ${balance} ({status})")

        if summary["low_balance_platforms"]:
            print(f"\n⚠️  Low Balance Platforms: {', '.join(summary['low_balance_platforms'])}")

        if summary["critical_platforms"]:
            print(f"🚨 Critical Platforms: {', '.join(summary['critical_platforms'])}")

    async def _check_platform_balance(self):
        """Check specific platform balance."""
        platform = input("Enter platform (fanduel/draftkings): ").strip().lower()

        if platform not in ["fanduel", "draftkings"]:
            print("Invalid platform. Please enter 'fanduel' or 'draftkings'.")
            return

        print(f"\nChecking {platform} balance...")
        summary = await self.agent.interface.get_balance_summary(platform)

        if "error" in summary:
            print(f"Error: {summary['error']}")
            return

        balance_info = summary["balance"]
        print(f"\n{platform.upper()} Balance: ${balance_info['current_balance']} ({balance_info['status']})")
        print(f"Last Updated: {balance_info['last_updated']}")

        if balance_info["threshold_warnings"]:
            print("\nWarnings:")
            for warning in balance_info["threshold_warnings"]:
                print(f"  ⚠️  {warning}")

    async def _request_funds(self):
        """Request funds manually."""
        platform = input("Enter platform (fanduel/draftkings): ").strip().lower()
        amount_str = input("Enter amount: $").strip()
        reason = input("Enter reason (optional): ").strip() or "Manual request"

        try:
            amount = Decimal(amount_str)
            request_id = await self.agent.interface.request_funds(platform, amount, reason)
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
            success = await self.agent.interface.confirm_deposit(platform, amount, confirmation_id)

            if success:
                print(f"\n✅ Deposit confirmed for {platform}: ${amount}")
            else:
                print(f"\n❌ Failed to confirm deposit for {platform}")
        except Exception as e:
            print(f"Error: {e}")

    async def _view_pending_requests(self):
        """View pending fund requests."""
        requests = await self.agent.interface.get_fund_requests(FundRequestStatus.PENDING)

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

    async def _get_report(self):
        """Get fund management report."""
        print("\nGenerating fund management report...")
        report = await self.agent.get_fund_management_report()

        if "error" in report:
            print(f"Error: {report['error']}")
            return

        print("\n📊 Fund Management Report")
        print(f"Generated: {report['timestamp']}")
        print(f"Total Balance: ${report['balance_summary']['total_balance']}")
        print(f"Pending Requests: {len(report['pending_requests'])}")

        # Save report to file
        filename = f"fund_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Report saved to: {filename}")

    async def _auto_monitor(self):
        """Run auto-monitoring and fund requests."""
        print("\nRunning auto-monitoring...")
        request_ids = await self.agent.monitor_and_request_funds()

        if request_ids:
            print(f"Created {len(request_ids)} fund requests:")
            for request_id in request_ids:
                print(f"  {request_id}")
        else:
            print("No fund requests needed.")


# Example usage
async def main():
    """Example usage of human-agent interface."""
    config = BrowserBaseConfig(
        api_key=os.getenv("BROWSERBASE_API_KEY"),
        project_id=os.getenv("BROWSERBASE_PROJECT_ID"),
        stealth_mode=True
    )

    async with BalanceMonitor(config) as monitor:
        fund_manager = FundManager(monitor)
        interface = HumanAgentInterface(fund_manager)
        agent = FundManagementAgent(interface)
        cli = FundManagementCLI(agent)

        await cli.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
