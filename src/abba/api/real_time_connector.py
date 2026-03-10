"""
Real-Time API Connector for ABMBA system.
Handles webhook subscriptions and live data feeds.
"""

import asyncio
import json
import ssl
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import aiohttp
import certifi
import structlog
import websockets
from database import DatabaseManager

from models import Bet

logger = structlog.get_logger()


@dataclass
class WebhookEvent:
    """Container for webhook events."""

    event_type: str
    data: dict[str, Any]
    timestamp: datetime
    source: str
    signature: str | None = None


@dataclass
class LiveDataFeed:
    """Container for live data feeds."""

    feed_type: str
    data: dict[str, Any]
    timestamp: datetime
    source: str
    sequence_number: int


class RealTimeAPIConnector:
    """Manages real-time API connections and data feeds."""

    def __init__(self, config: dict, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.api_config = config.get("apis", {})

        # Connection management
        self.active_connections = {}
        self.webhook_subscriptions = {}
        self.data_feeds = {}
        self.connection_status = {}

        # Event handlers
        self.event_handlers = {}
        self.data_handlers = {}

        # Rate limiting
        self.rate_limits = {}
        self.last_request_time = {}

        # SSL context
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

        logger.info("Real-Time API Connector initialized")

    async def setup_webhook_subscriptions(self, webhook_configs: list[dict]) -> bool:
        """Setup webhook subscriptions for real-time data."""
        try:
            logger.info(f"Setting up {len(webhook_configs)} webhook subscriptions")

            for config in webhook_configs:
                success = await self._setup_single_webhook(config)
                if success:
                    logger.info(
                        f"Webhook subscription setup successful: {config.get('name', 'unknown')}"
                    )
                else:
                    logger.error(
                        f"Webhook subscription setup failed: {config.get('name', 'unknown')}"
                    )

            return True

        except Exception as e:
            logger.error(f"Error setting up webhook subscriptions: {e}")
            return False

    async def _setup_single_webhook(self, config: dict) -> bool:
        """Setup a single webhook subscription."""
        try:
            webhook_name = config.get("name", "default")
            endpoint = config.get("endpoint")
            events = config.get("events", [])
            api_key = config.get("api_key")

            if not endpoint or not events:
                logger.error(f"Invalid webhook config: {config}")
                return False

            # Create webhook subscription request
            subscription_data = {
                "url": config.get("webhook_url", "http://localhost:8000/webhook"),
                "events": events,
                "description": f"ABMBA webhook for {webhook_name}",
            }

            # Add authentication headers
            headers = await self._get_auth_headers(api_key, config)

            # Make subscription request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=subscription_data,
                    headers=headers,
                    ssl=self.ssl_context,
                ) as response:

                    if response.status == 200 or response.status == 201:
                        # Store subscription info
                        self.webhook_subscriptions[webhook_name] = {
                            "config": config,
                            "subscription_id": await response.text(),
                            "status": "active",
                            "created_at": datetime.utcnow(),
                        }

                        logger.info(f"Webhook subscription created: {webhook_name}")
                        return True
                    else:
                        logger.error(
                            f"Webhook subscription failed: {response.status} - {await response.text()}"
                        )
                        return False

        except Exception as e:
            logger.error(f"Error setting up webhook: {e}")
            return False

    async def _get_auth_headers(self, api_key: str, config: dict) -> dict[str, str]:
        """Generate authentication headers for API requests."""
        try:
            headers = {"Content-Type": "application/json", "User-Agent": "ABMBA/1.0"}

            if api_key:
                auth_type = config.get("auth_type", "bearer")
                if auth_type == "bearer":
                    headers["Authorization"] = f"Bearer {api_key}"
                elif auth_type == "api_key":
                    headers["X-API-Key"] = api_key
                elif auth_type == "custom":
                    headers[config.get("custom_header", "X-Auth")] = api_key

            return headers

        except Exception as e:
            logger.error(f"Error generating auth headers: {e}")
            return {}

    async def start_live_data_feeds(self, feed_configs: list[dict]) -> bool:
        """Start live data feeds for real-time information."""
        try:
            logger.info(f"Starting {len(feed_configs)} live data feeds")

            for config in feed_configs:
                feed_name = config.get("name", "default")

                # Start feed in background
                asyncio.create_task(self._run_data_feed(config))

                self.data_feeds[feed_name] = {
                    "config": config,
                    "status": "starting",
                    "started_at": datetime.utcnow(),
                    "last_update": None,
                }

                logger.info(f"Live data feed started: {feed_name}")

            return True

        except Exception as e:
            logger.error(f"Error starting live data feeds: {e}")
            return False

    async def _run_data_feed(self, config: dict):
        """Run a single data feed."""
        try:
            _feed_name = config.get("name", "default")
            feed_type = config.get("type", "websocket")

            if feed_type == "websocket":
                await self._run_websocket_feed(config)
            elif feed_type == "rest_polling":
                await self._run_rest_polling_feed(config)
            elif feed_type == "sse":
                await self._run_sse_feed(config)
            else:
                logger.error(f"Unknown feed type: {feed_type}")

        except Exception as e:
            logger.error(
                f"Error running data feed {config.get('name', 'unknown')}: {e}"
            )
            self.data_feeds[config.get("name", "default")]["status"] = "error"

    async def _run_websocket_feed(self, config: dict):
        """Run a WebSocket data feed."""
        try:
            feed_name = config.get("name", "default")
            ws_url = config.get("url")
            api_key = config.get("api_key")

            if not ws_url:
                logger.error(f"No WebSocket URL provided for feed: {feed_name}")
                return

            # Prepare connection headers
            headers = await self._get_auth_headers(api_key, config)

            # Connect to WebSocket
            async with websockets.connect(
                ws_url, extra_headers=headers, ssl=self.ssl_context
            ) as websocket:

                self.data_feeds[feed_name]["status"] = "connected"
                logger.info(f"WebSocket connected for feed: {feed_name}")

                # Subscribe to channels if needed
                channels = config.get("channels", [])
                if channels:
                    subscribe_message = {"type": "subscribe", "channels": channels}
                    await websocket.send(json.dumps(subscribe_message))

                # Listen for messages
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self._process_feed_data(feed_name, data)

                        # Update last update time
                        self.data_feeds[feed_name]["last_update"] = datetime.utcnow()

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON received from feed {feed_name}")
                    except Exception as e:
                        logger.error(
                            f"Error processing message from feed {feed_name}: {e}"
                        )

        except Exception as e:
            logger.error(f"WebSocket feed error for {feed_name}: {e}")
            self.data_feeds[feed_name]["status"] = "disconnected"

    async def _run_rest_polling_feed(self, config: dict):
        """Run a REST polling data feed."""
        try:
            feed_name = config.get("name", "default")
            endpoint = config.get("endpoint")
            api_key = config.get("api_key")
            poll_interval = config.get("poll_interval", 30)  # seconds

            if not endpoint:
                logger.error(f"No endpoint provided for feed: {feed_name}")
                return

            headers = await self._get_auth_headers(api_key, config)

            self.data_feeds[feed_name]["status"] = "polling"
            logger.info(f"REST polling started for feed: {feed_name}")

            while self.data_feeds[feed_name]["status"] != "stopped":
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            endpoint, headers=headers, ssl=self.ssl_context
                        ) as response:

                            if response.status == 200:
                                data = await response.json()
                                await self._process_feed_data(feed_name, data)

                                # Update last update time
                                self.data_feeds[feed_name][
                                    "last_update"
                                ] = datetime.utcnow()
                            else:
                                logger.warning(
                                    f"REST polling failed for {feed_name}: {response.status}"
                                )

                    # Wait before next poll
                    await asyncio.sleep(poll_interval)

                except Exception as e:
                    logger.error(f"Error in REST polling for {feed_name}: {e}")
                    await asyncio.sleep(poll_interval)

        except Exception as e:
            logger.error(f"REST polling feed error for {feed_name}: {e}")
            self.data_feeds[feed_name]["status"] = "error"

    async def _run_sse_feed(self, config: dict):
        """Run a Server-Sent Events data feed."""
        try:
            feed_name = config.get("name", "default")
            endpoint = config.get("endpoint")
            api_key = config.get("api_key")

            if not endpoint:
                logger.error(f"No endpoint provided for SSE feed: {feed_name}")
                return

            headers = await self._get_auth_headers(api_key, config)
            headers["Accept"] = "text/event-stream"

            self.data_feeds[feed_name]["status"] = "connected"
            logger.info(f"SSE feed started: {feed_name}")

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    endpoint, headers=headers, ssl=self.ssl_context
                ) as response:

                    if response.status == 200:
                        async for line in response.content:
                            line = line.decode("utf-8").strip()

                            if line.startswith("data: "):
                                try:
                                    data = json.loads(
                                        line[6:]
                                    )  # Remove 'data: ' prefix
                                    await self._process_feed_data(feed_name, data)

                                    # Update last update time
                                    self.data_feeds[feed_name][
                                        "last_update"
                                    ] = datetime.utcnow()

                                except json.JSONDecodeError:
                                    logger.warning(
                                        f"Invalid JSON in SSE feed {feed_name}"
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Error processing SSE data for {feed_name}: {e}"
                                    )
                    else:
                        logger.error(
                            f"SSE connection failed for {feed_name}: {response.status}"
                        )

        except Exception as e:
            logger.error(f"SSE feed error for {feed_name}: {e}")
            self.data_feeds[feed_name]["status"] = "error"

    async def _process_feed_data(self, feed_name: str, data: dict):
        """Process data received from a feed."""
        try:
            # Create feed event
            feed_event = LiveDataFeed(
                feed_type=feed_name,
                data=data,
                timestamp=datetime.utcnow(),
                source=feed_name,
                sequence_number=len(
                    self.data_feeds.get(feed_name, {}).get("events", [])
                ),
            )

            # Store in feed history
            if "events" not in self.data_feeds[feed_name]:
                self.data_feeds[feed_name]["events"] = []

            self.data_feeds[feed_name]["events"].append(feed_event)

            # Limit history size
            if len(self.data_feeds[feed_name]["events"]) > 1000:
                self.data_feeds[feed_name]["events"] = self.data_feeds[feed_name][
                    "events"
                ][-1000:]

            # Process based on data type
            await self._process_data_by_type(feed_name, data)

            # Call registered handlers
            if feed_name in self.data_handlers:
                for handler in self.data_handlers[feed_name]:
                    try:
                        await handler(feed_event)
                    except Exception as e:
                        logger.error(f"Error in data handler for {feed_name}: {e}")

        except Exception as e:
            logger.error(f"Error processing feed data for {feed_name}: {e}")

    async def _process_data_by_type(self, feed_name: str, data: dict):
        """Process data based on its type."""
        try:
            data_type = data.get("type", "unknown")

            if data_type == "odds_update":
                await self._process_odds_update(data)
            elif data_type == "event_update":
                await self._process_event_update(data)
            elif data_type == "market_update":
                await self._process_market_update(data)
            elif data_type == "result_update":
                await self._process_result_update(data)
            else:
                logger.debug(f"Unknown data type: {data_type} from feed {feed_name}")

        except Exception as e:
            logger.error(f"Error processing data by type: {e}")

    async def _process_odds_update(self, data: dict):
        """Process odds update data."""
        try:
            event_id = data.get("event_id")
            market_id = data.get("market_id")
            new_odds = data.get("odds")

            if event_id and market_id and new_odds:
                # Update odds in database
                await self.db_manager.update_odds(event_id, market_id, new_odds)

                # Check for value betting opportunities
                await self._check_value_opportunities(event_id, new_odds)

        except Exception as e:
            logger.error(f"Error processing odds update: {e}")

    async def _process_event_update(self, data: dict):
        """Process event update data."""
        try:
            event_id = data.get("event_id")
            event_data = data.get("event_data", {})

            if event_id and event_data:
                # Update event in database
                await self.db_manager.update_event(event_id, event_data)

        except Exception as e:
            logger.error(f"Error processing event update: {e}")

    async def _process_market_update(self, data: dict):
        """Process market update data."""
        try:
            market_id = data.get("market_id")
            market_data = data.get("market_data", {})

            if market_id and market_data:
                # Update market in database
                await self.db_manager.update_market(market_id, market_data)

        except Exception as e:
            logger.error(f"Error processing market update: {e}")

    async def _process_result_update(self, data: dict):
        """Process result update data."""
        try:
            event_id = data.get("event_id")
            result_data = data.get("result_data", {})

            if event_id and result_data:
                # Update result in database
                await self.db_manager.update_event_result(event_id, result_data)

                # Settle related bets
                await self._settle_related_bets(event_id, result_data)

        except Exception as e:
            logger.error(f"Error processing result update: {e}")

    async def _check_value_opportunities(self, event_id: str, new_odds: dict):
        """Check for value betting opportunities after odds update."""
        try:
            # This would integrate with the simulation agent to check for value
            # For now, just log the opportunity
            logger.info(
                f"Checking value opportunities for event {event_id} with odds {new_odds}"
            )

        except Exception as e:
            logger.error(f"Error checking value opportunities: {e}")

    async def _settle_related_bets(self, event_id: str, result_data: dict):
        """Settle bets related to an event result."""
        try:
            # Get all bets for this event
            bets = await self.db_manager.get_bets(event_id=event_id, status="placed")

            for bet in bets:
                # Determine bet result based on event result
                bet_result = await self._determine_bet_result(bet, result_data)

                # Update bet status
                await self.db_manager.update_bet_result(bet.id, bet_result)

        except Exception as e:
            logger.error(f"Error settling related bets: {e}")

    async def _determine_bet_result(self, bet: Bet, result_data: dict) -> str:
        """Determine the result of a bet based on event result."""
        try:
            # This would implement logic to determine if bet won or lost
            # For now, return a placeholder
            return "pending"

        except Exception as e:
            logger.error(f"Error determining bet result: {e}")
            return "pending"

    async def register_data_handler(self, feed_name: str, handler: Callable):
        """Register a handler for data from a specific feed."""
        try:
            if feed_name not in self.data_handlers:
                self.data_handlers[feed_name] = []

            self.data_handlers[feed_name].append(handler)
            logger.info(f"Registered data handler for feed: {feed_name}")

        except Exception as e:
            logger.error(f"Error registering data handler: {e}")

    async def register_event_handler(self, event_type: str, handler: Callable):
        """Register a handler for specific event types."""
        try:
            if event_type not in self.event_handlers:
                self.event_handlers[event_type] = []

            self.event_handlers[event_type].append(handler)
            logger.info(f"Registered event handler for type: {event_type}")

        except Exception as e:
            logger.error(f"Error registering event handler: {e}")

    async def handle_webhook_event(
        self, event_data: dict, signature: str = None
    ) -> bool:
        """Handle incoming webhook events."""
        try:
            # Verify signature if provided
            if signature and not await self._verify_webhook_signature(
                event_data, signature
            ):
                logger.warning("Webhook signature verification failed")
                return False

            # Parse event
            event_type = event_data.get("type", "unknown")
            data = event_data.get("data", {})
            timestamp = datetime.fromisoformat(
                event_data.get("timestamp", datetime.utcnow().isoformat())
            )
            source = event_data.get("source", "unknown")

            # Create webhook event
            webhook_event = WebhookEvent(
                event_type=event_type,
                data=data,
                timestamp=timestamp,
                source=source,
                signature=signature,
            )

            # Process event
            await self._process_webhook_event(webhook_event)

            # Call registered handlers
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    try:
                        await handler(webhook_event)
                    except Exception as e:
                        logger.error(f"Error in event handler for {event_type}: {e}")

            logger.info(f"Webhook event processed: {event_type} from {source}")
            return True

        except Exception as e:
            logger.error(f"Error handling webhook event: {e}")
            return False

    async def _verify_webhook_signature(self, event_data: dict, signature: str) -> bool:
        """Verify webhook signature for security."""
        try:
            # This would implement signature verification based on the webhook provider
            # For now, return True as placeholder
            return True

        except Exception as e:
            logger.error(f"Error verifying webhook signature: {e}")
            return False

    async def _process_webhook_event(self, event: WebhookEvent):
        """Process webhook event based on type."""
        try:
            if event.event_type == "odds_update":
                await self._process_odds_update(event.data)
            elif event.event_type == "event_update":
                await self._process_event_update(event.data)
            elif event.event_type == "market_update":
                await self._process_market_update(event.data)
            elif event.event_type == "result_update":
                await self._process_result_update(event.data)
            else:
                logger.debug(f"Unknown webhook event type: {event.event_type}")

        except Exception as e:
            logger.error(f"Error processing webhook event: {e}")

    async def get_connection_status(self) -> dict[str, Any]:
        """Get status of all connections."""
        try:
            status = {"webhooks": {}, "data_feeds": {}, "overall_status": "healthy"}

            # Webhook status
            for name, info in self.webhook_subscriptions.items():
                status["webhooks"][name] = {
                    "status": info.get("status", "unknown"),
                    "created_at": (
                        info.get("created_at", "").isoformat()
                        if info.get("created_at")
                        else None
                    ),
                }

            # Data feed status
            for name, info in self.data_feeds.items():
                status["data_feeds"][name] = {
                    "status": info.get("status", "unknown"),
                    "started_at": (
                        info.get("started_at", "").isoformat()
                        if info.get("started_at")
                        else None
                    ),
                    "last_update": (
                        info.get("last_update", "").isoformat()
                        if info.get("last_update")
                        else None
                    ),
                    "event_count": len(info.get("events", [])),
                }

            # Overall status
            webhook_statuses = [s["status"] for s in status["webhooks"].values()]
            feed_statuses = [s["status"] for s in status["data_feeds"].values()]

            if "error" in webhook_statuses or "error" in feed_statuses:
                status["overall_status"] = "error"
            elif "disconnected" in feed_statuses:
                status["overall_status"] = "degraded"

            return status

        except Exception as e:
            logger.error(f"Error getting connection status: {e}")
            return {"error": str(e)}

    async def stop_all_feeds(self):
        """Stop all active data feeds."""
        try:
            logger.info("Stopping all data feeds")

            for feed_name in self.data_feeds:
                self.data_feeds[feed_name]["status"] = "stopped"

            logger.info("All data feeds stopped")

        except Exception as e:
            logger.error(f"Error stopping feeds: {e}")

    async def cleanup_webhook_subscriptions(self):
        """Clean up webhook subscriptions."""
        try:
            logger.info("Cleaning up webhook subscriptions")

            for name, info in self.webhook_subscriptions.items():
                config = info.get("config", {})
                endpoint = config.get("cleanup_endpoint")
                api_key = config.get("api_key")

                if endpoint and api_key:
                    headers = await self._get_auth_headers(api_key, config)

                    async with aiohttp.ClientSession() as session:
                        async with session.delete(
                            endpoint, headers=headers, ssl=self.ssl_context
                        ) as response:

                            if response.status == 200:
                                logger.info(f"Webhook subscription cleaned up: {name}")
                            else:
                                logger.warning(f"Failed to cleanup webhook: {name}")

            logger.info("Webhook cleanup completed")

        except Exception as e:
            logger.error(f"Error cleaning up webhooks: {e}")

    async def get_feed_history(
        self, feed_name: str, limit: int = 100
    ) -> list[LiveDataFeed]:
        """Get history of events from a specific feed."""
        try:
            if feed_name not in self.data_feeds:
                return []

            events = self.data_feeds[feed_name].get("events", [])
            return events[-limit:] if events else []

        except Exception as e:
            logger.error(f"Error getting feed history: {e}")
            return []

    async def get_webhook_history(
        self, webhook_name: str, limit: int = 100
    ) -> list[WebhookEvent]:
        """Get history of webhook events."""
        try:
            # This would retrieve webhook event history from storage
            # For now, return empty list as placeholder
            return []

        except Exception as e:
            logger.error(f"Error getting webhook history: {e}")
            return []
