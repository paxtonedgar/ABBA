"""
Data fetcher module for ABMBA system.
Handles API calls and web scraping for sports data.
"""

import asyncio
import random
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import aiohttp
import numpy as np
import pandas as pd
import structlog
from bs4 import BeautifulSoup
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

from models import Event, EventStatus, MarketType, Odds, PlatformType, SportType

logger = structlog.get_logger()


class DataVerifier:
    """Data verification and anomaly detection for sports betting data."""

    def __init__(self, contamination: float = 0.1, confidence_threshold: float = 0.7):
        """
        Initialize data verifier.
        
        Args:
            contamination: Expected fraction of anomalies (default 0.1 = 10%)
            confidence_threshold: Minimum confidence score to accept data (default 0.7)
        """
        self.iso_forest = IsolationForest(contamination=contamination, random_state=42)
        self.confidence_threshold = confidence_threshold
        self.anomaly_history = []
        self.validation_stats = {
            'total_checks': 0,
            'anomalies_detected': 0,
            'confidence_scores': []
        }

        # Physics constants for validation
        self.max_spin_rate = 3500  # RPM (baseball)
        self.max_exit_velocity = 120  # mph (baseball)
        self.max_pitch_velocity = 105  # mph (baseball)
        self.max_hockey_speed = 110  # mph (hockey)

    def detect_anomalies(self, data_df: pd.DataFrame, columns: list[str] = None) -> tuple[pd.DataFrame, float]:
        """
        Flag outliers in odds/stats using multiple detection methods.
        
        Args:
            data_df: DataFrame containing data to check
            columns: Columns to check for anomalies (default: all numeric)
            
        Returns:
            Tuple of (anomalies_df, confidence_score)
        """
        if columns is None:
            columns = data_df.select_dtypes(include=[np.number]).columns.tolist()

        anomalies_mask = pd.Series([False] * len(data_df), index=data_df.index)

        # Z-score method
        for col in columns:
            if col in data_df.columns:
                z_scores = np.abs(zscore(data_df[col].dropna()))
                anomalies_z = z_scores > 3
                anomalies_mask |= anomalies_z

        # Isolation Forest method
        try:
            numeric_data = data_df[columns].fillna(data_df[columns].median())
            self.iso_forest.fit(numeric_data)
            anomalies_iso = self.iso_forest.predict(numeric_data) == -1
            anomalies_mask |= anomalies_iso
        except Exception as e:
            logger.warning(f"Isolation Forest failed: {e}")

        # Calculate confidence score
        anomaly_rate = anomalies_mask.mean()
        confidence_score = max(0, 1 - anomaly_rate)

        # Log detection results
        self.validation_stats['total_checks'] += 1
        self.validation_stats['anomalies_detected'] += anomalies_mask.sum()
        self.validation_stats['confidence_scores'].append(confidence_score)

        anomalies_df = data_df[anomalies_mask].copy()
        anomalies_df['anomaly_score'] = confidence_score

        logger.info(f"Detected {anomalies_mask.sum()} anomalies out of {len(data_df)} records (confidence: {confidence_score:.3f})")

        return anomalies_df, confidence_score

    def validate_completeness(self, data: pd.DataFrame, expected_coverage: float = 0.9) -> tuple[bool, float]:
        """
        Check for missing data bias (e.g., untracked balls).
        
        Args:
            data: DataFrame to check
            expected_coverage: Minimum expected data coverage (default 0.9 = 90%)
            
        Returns:
            Tuple of (is_valid, coverage_rate)
        """
        missing_rate = data.isnull().mean()
        coverage_rate = 1 - missing_rate.mean()

        is_valid = coverage_rate >= expected_coverage

        if not is_valid:
            logger.warning(f"Data incompleteness exceeds threshold: {coverage_rate:.3f} < {expected_coverage}")
        else:
            logger.info(f"Data completeness check passed: {coverage_rate:.3f} >= {expected_coverage}")

        return is_valid, coverage_rate

    def validate_physics(self, data: pd.DataFrame, sport: str) -> tuple[pd.DataFrame, float]:
        """
        Physics-based validation using Newtonian mechanics.
        
        Args:
            data: DataFrame with physics measurements
            sport: Sport type for validation rules
            
        Returns:
            Tuple of (violations_df, confidence_score)
        """
        violations_mask = pd.Series([False] * len(data), index=data.index)

        if sport == 'baseball_mlb':
            # Baseball physics validation
            if 'spin_rate' in data.columns:
                violations_mask |= data['spin_rate'] > self.max_spin_rate

            if 'exit_velocity' in data.columns:
                violations_mask |= data['exit_velocity'] > self.max_exit_velocity

            if 'pitch_velocity' in data.columns:
                violations_mask |= data['pitch_velocity'] > self.max_pitch_velocity

        elif sport == 'hockey_nhl':
            # Hockey physics validation
            if 'puck_speed' in data.columns:
                violations_mask |= data['puck_speed'] > self.max_hockey_speed

        # Cross-venue calibration check (16.6 more hits in certain arenas)
        if 'venue' in data.columns and 'hits' in data.columns:
            venue_hits = data.groupby('venue')['hits'].mean()
            overall_mean = data['hits'].mean()
            venue_bias = venue_hits - overall_mean

            # Flag venues with >10% bias
            biased_venues = venue_bias[abs(venue_bias) > overall_mean * 0.1]
            if not biased_venues.empty:
                logger.warning(f"Venue bias detected: {biased_venues.to_dict()}")

        confidence_score = 1 - violations_mask.mean()
        violations_df = data[violations_mask].copy()

        logger.info(f"Physics validation: {violations_mask.sum()} violations, confidence: {confidence_score:.3f}")

        return violations_df, confidence_score

    def detect_betting_patterns(self, odds_data: pd.DataFrame, window: int = 10) -> tuple[pd.DataFrame, float]:
        """
        ML for betting pattern anomalies (e.g., sudden odds shifts).
        
        Args:
            odds_data: DataFrame with odds over time
            window: Rolling window for pattern detection
            
        Returns:
            Tuple of (anomalies_df, confidence_score)
        """
        anomalies_mask = pd.Series([False] * len(odds_data), index=odds_data.index)

        if 'odds' in odds_data.columns and 'timestamp' in odds_data.columns:
            # Sort by timestamp
            odds_data = odds_data.sort_values('timestamp')

            # Calculate odds movement
            odds_data['odds_change'] = odds_data['odds'].diff()
            odds_data['odds_change_pct'] = odds_data['odds_change'] / odds_data['odds'].shift(1)

            # Detect sudden shifts (>5% in single update)
            sudden_shifts = abs(odds_data['odds_change_pct']) > 0.05
            anomalies_mask |= sudden_shifts

            # Rolling volatility check
            rolling_std = odds_data['odds_change_pct'].rolling(window=window).std()
            high_volatility = rolling_std > rolling_std.quantile(0.95)
            anomalies_mask |= high_volatility

            # Impossible probabilities check
            if 'implied_probability' in odds_data.columns:
                impossible_probs = (odds_data['implied_probability'] < 0) | (odds_data['implied_probability'] > 1)
                anomalies_mask |= impossible_probs

        confidence_score = 1 - anomalies_mask.mean()
        anomalies_df = odds_data[anomalies_mask].copy()

        logger.info(f"Betting pattern analysis: {anomalies_mask.sum()} anomalies, confidence: {confidence_score:.3f}")

        return anomalies_df, confidence_score

    def calculate_confidence_score(self, data: pd.DataFrame, sport: str = None) -> float:
        """
        Calculate overall confidence score (0-1 scale).
        
        Args:
            data: DataFrame to evaluate
            sport: Sport type for specific validations
            
        Returns:
            Confidence score between 0 and 1
        """
        scores = []

        # Completeness check
        _, coverage_score = self.validate_completeness(data)
        scores.append(coverage_score)

        # Anomaly detection
        if not data.empty:
            _, anomaly_score = self.detect_anomalies(data)
            scores.append(anomaly_score)

        # Physics validation
        if sport:
            _, physics_score = self.validate_physics(data, sport)
            scores.append(physics_score)

        # Betting pattern analysis
        if 'odds' in data.columns:
            _, pattern_score = self.detect_betting_patterns(data)
            scores.append(pattern_score)

        # Calculate weighted average
        if scores:
            confidence_score = np.mean(scores)
        else:
            confidence_score = 0.5  # Default neutral score

        return confidence_score

    def should_halt_processing(self, confidence_score: float) -> bool:
        """
        Check if processing should halt due to low confidence.
        
        Args:
            confidence_score: Calculated confidence score
            
        Returns:
            True if processing should halt
        """
        should_halt = confidence_score < self.confidence_threshold

        if should_halt:
            logger.error(f"Confidence score {confidence_score:.3f} below threshold {self.confidence_threshold}, halting processing")
        else:
            logger.info(f"Confidence score {confidence_score:.3f} above threshold {self.confidence_threshold}, proceeding")

        return should_halt

    def get_validation_report(self) -> dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Returns:
            Dictionary with validation statistics
        """
        avg_confidence = np.mean(self.validation_stats['confidence_scores']) if self.validation_stats['confidence_scores'] else 0

        return {
            'total_checks': self.validation_stats['total_checks'],
            'anomalies_detected': self.validation_stats['anomalies_detected'],
            'average_confidence': avg_confidence,
            'confidence_threshold': self.confidence_threshold,
            'anomaly_rate': self.validation_stats['anomalies_detected'] / max(1, self.validation_stats['total_checks']),
            'validation_timestamp': datetime.utcnow().isoformat()
        }


class DataFetcher:
    """Data fetcher for sports betting information with verification and caching."""

    def __init__(self, config: dict):
        self.config = config
        self.session = None
        self.verifier = DataVerifier()
        self.user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        ]

        # Initialize caching system
        from cache_manager import APIOptimizer, CacheManager, DataPersistenceManager
        self.cache_manager = CacheManager(config)
        self.api_optimizer = APIOptimizer(self.cache_manager, config)
        self.persistence_manager = DataPersistenceManager(config)

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            headers={'User-Agent': random.choice(self.user_agents)}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def fetch_events(self, sport: str, force_refresh: bool = False) -> list[Event]:
        """
        Fetch events for a specific sport with verification and caching.
        
        Args:
            sport: Sport type (e.g., 'basketball_nba')
            force_refresh: Force refresh from API instead of using cache
        
        Returns:
            List of Event objects
        """
        try:
            # Generate cache key
            cache_key = f"events:{sport}"

            # Try to get from cache first
            if not force_refresh:
                cached_events = await self.cache_manager.get(cache_key, 'events')
                if cached_events:
                    logger.info(f"Using cached events for {sport} ({len(cached_events)} events)")
                    return cached_events

            # Fetch fresh data
            async def fetch_fresh_events():
                # Try The Odds API first
                events = await self._fetch_odds_api_events(sport)

                if not events:
                    # Fallback to scraping
                    events = await self._scrape_events(sport)

                return events

            # Use get_or_fetch for optimized API calls
            events = await self.cache_manager.get_or_fetch(
                cache_key,
                fetch_fresh_events,
                'events',
                force_refresh=force_refresh
            )

            # Verify fetched data
            if events:
                events_df = pd.DataFrame([{
                    'home_team': e.home_team,
                    'away_team': e.away_team,
                    'event_date': e.event_date,
                    'sport': e.sport.value
                } for e in events])

                confidence_score = self.verifier.calculate_confidence_score(events_df, sport)

                if self.verifier.should_halt_processing(confidence_score):
                    logger.error(f"Event data verification failed for {sport}, confidence: {confidence_score:.3f}")
                    return []

                logger.info(f"Fetched {len(events)} events for {sport} (confidence: {confidence_score:.3f})")

            return events

        except Exception as e:
            logger.error(f"Error fetching events for {sport}: {e}")
            return []

    async def fetch_odds(self, event_id: str, sport: str, force_refresh: bool = False) -> list[Odds]:
        """
        Fetch odds for a specific event with verification and caching.
        
        Args:
            event_id: Event identifier
            sport: Sport type
            force_refresh: Force refresh from API instead of using cache
        
        Returns:
            List of Odds objects
        """
        try:
            # Generate cache key
            cache_key = f"odds:{event_id}:{sport}"

            # Try to get from cache first
            if not force_refresh:
                cached_odds = await self.cache_manager.get(cache_key, 'odds')
                if cached_odds:
                    logger.info(f"Using cached odds for event {event_id} ({len(cached_odds)} odds)")
                    return cached_odds

            # Fetch fresh data
            async def fetch_fresh_odds():
                odds_list = []

                # Fetch from multiple platforms
                platforms = ['fanduel', 'draftkings']

                for platform in platforms:
                    if self.config['platforms'][platform]['enabled']:
                        platform_odds = await self._fetch_platform_odds(event_id, sport, platform)
                        odds_list.extend(platform_odds)

                return odds_list

            # Use get_or_fetch for optimized API calls
            odds_list = await self.cache_manager.get_or_fetch(
                cache_key,
                fetch_fresh_odds,
                'odds',
                force_refresh=force_refresh
            )

            # Verify odds data
            if odds_list:
                odds_df = pd.DataFrame([{
                    'odds': float(o.odds),
                    'implied_probability': float(o.implied_probability) if o.implied_probability else None,
                    'platform': o.platform.value,
                    'market_type': o.market_type.value,
                    'timestamp': o.timestamp
                } for o in odds_list])

                confidence_score = self.verifier.calculate_confidence_score(odds_df, sport)

                if self.verifier.should_halt_processing(confidence_score):
                    logger.error(f"Odds data verification failed for event {event_id}, confidence: {confidence_score:.3f}")
                    return []

                logger.info(f"Fetched {len(odds_list)} odds for event {event_id} (confidence: {confidence_score:.3f})")

            return odds_list

        except Exception as e:
            logger.error(f"Error fetching odds for event {event_id}: {e}")
            return []

    async def _fetch_odds_api_events(self, sport: str) -> list[Event]:
        """Fetch events from The Odds API."""
        try:
            api_key = self.config['apis']['odds_api']['key']
            base_url = self.config['apis']['odds_api']['base_url']

            # Map sport names to API format
            sport_mapping = {
                'basketball_nba': 'basketball_nba',
                'basketball_ncaab': 'basketball_ncaab',
                'football_nfl': 'americanfootball_nfl',
                'football_ncaaf': 'americanfootball_ncaaf',
                'baseball_mlb': 'baseball_mlb',
                'hockey_nhl': 'icehockey_nhl'
            }

            api_sport = sport_mapping.get(sport, sport)

            url = f"{base_url}/sports/{api_sport}/odds"
            params = {
                'apiKey': api_key,
                'regions': 'us',
                'markets': 'h2h,spreads,totals',
                'oddsFormat': 'american'
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    events = []
                    for game in data:
                        # Parse event date
                        event_date = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))

                        # Create Event object
                        event = Event(
                            sport=SportType(sport),
                            home_team=game['home_team'],
                            away_team=game['away_team'],
                            event_date=event_date,
                            status=EventStatus.SCHEDULED
                        )

                        events.append(event)

                    return events
                else:
                    logger.warning(f"Odds API returned status {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Error fetching from Odds API: {e}")
            return []

    async def _fetch_platform_odds(self, event_id: str, sport: str, platform: str) -> list[Odds]:
        """Fetch odds from a specific platform."""
        try:
            if platform == 'fanduel':
                return await self._fetch_fanduel_odds(event_id, sport)
            elif platform == 'draftkings':
                return await self._fetch_draftkings_odds(event_id, sport)
            else:
                logger.warning(f"Unsupported platform: {platform}")
                return []

        except Exception as e:
            logger.error(f"Error fetching odds from {platform}: {e}")
            return []

    async def _fetch_fanduel_odds(self, event_id: str, sport: str) -> list[Odds]:
        """Fetch odds from FanDuel (mock implementation)."""
        # This would be a real implementation using FanDuel's API or web scraping
        # For now, return mock data

        odds_list = []

        # Mock moneyline odds
        home_odds = Odds(
            event_id=event_id,
            platform=PlatformType.FANDUEL,
            market_type=MarketType.MONEYLINE,
            selection='home',
            odds=Decimal('-110'),
            timestamp=datetime.utcnow()
        )
        odds_list.append(home_odds)

        away_odds = Odds(
            event_id=event_id,
            platform=PlatformType.FANDUEL,
            market_type=MarketType.MONEYLINE,
            selection='away',
            odds=Decimal('-110'),
            timestamp=datetime.utcnow()
        )
        odds_list.append(away_odds)

        # Mock spread odds
        home_spread = Odds(
            event_id=event_id,
            platform=PlatformType.FANDUEL,
            market_type=MarketType.SPREAD,
            selection='home',
            odds=Decimal('-110'),
            line=Decimal('-3.5'),
            timestamp=datetime.utcnow()
        )
        odds_list.append(home_spread)

        away_spread = Odds(
            event_id=event_id,
            platform=PlatformType.FANDUEL,
            market_type=MarketType.SPREAD,
            selection='away',
            odds=Decimal('-110'),
            line=Decimal('3.5'),
            timestamp=datetime.utcnow()
        )
        odds_list.append(away_spread)

        return odds_list

    async def _fetch_draftkings_odds(self, event_id: str, sport: str) -> list[Odds]:
        """Fetch odds from DraftKings (mock implementation)."""
        # Similar mock implementation for DraftKings

        odds_list = []

        # Mock moneyline odds
        home_odds = Odds(
            event_id=event_id,
            platform=PlatformType.DRAFTKINGS,
            market_type=MarketType.MONEYLINE,
            selection='home',
            odds=Decimal('-105'),
            timestamp=datetime.utcnow()
        )
        odds_list.append(home_odds)

        away_odds = Odds(
            event_id=event_id,
            platform=PlatformType.DRAFTKINGS,
            market_type=MarketType.MONEYLINE,
            selection='away',
            odds=Decimal('-115'),
            timestamp=datetime.utcnow()
        )
        odds_list.append(away_odds)

        return odds_list

    async def _scrape_events(self, sport: str) -> list[Event]:
        """Scrape events from betting websites (fallback method)."""
        try:
            # This would implement web scraping as a fallback
            # For now, return empty list
            logger.info(f"Scraping not implemented for {sport}")
            return []

        except Exception as e:
            logger.error(f"Error scraping events: {e}")
            return []

    async def fetch_historical_data(self, sport: str, days: int = 30) -> list[dict]:
        """Fetch historical data for model training."""
        try:
            # This would fetch historical results and odds
            # For now, return mock data

            historical_data = []
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)

            # Generate mock historical events
            current_date = start_date
            while current_date <= end_date:
                if current_date.weekday() < 5:  # Weekdays only
                    historical_data.append({
                        'date': current_date.isoformat(),
                        'sport': sport,
                        'home_team': 'Team A',
                        'away_team': 'Team B',
                        'home_score': random.randint(80, 120),
                        'away_score': random.randint(80, 120),
                        'home_odds': random.choice([-110, -120, -130, -140]),
                        'away_odds': random.choice([-110, -120, -130, -140]),
                        'spread': random.choice([-3.5, -2.5, -1.5, 1.5, 2.5, 3.5]),
                        'total': random.choice([200, 210, 220, 230, 240])
                    })

                current_date += timedelta(days=1)

            logger.info(f"Generated {len(historical_data)} historical records for {sport}")
            return historical_data

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return []

    async def check_api_limits(self) -> dict[str, Any]:
        """Check API rate limits and usage."""
        try:
            api_key = self.config['apis']['odds_api']['key']
            base_url = self.config['apis']['odds_api']['base_url']

            url = f"{base_url}/sports"
            params = {'apiKey': api_key}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    # Check remaining requests from headers
                    remaining = response.headers.get('x-requests-remaining', 'unknown')
                    used = response.headers.get('x-requests-used', 'unknown')

                    return {
                        'remaining_requests': remaining,
                        'used_requests': used,
                        'status': 'healthy'
                    }
                else:
                    return {
                        'status': 'error',
                        'error': f"API returned status {response.status}"
                    }

        except Exception as e:
            logger.error(f"Error checking API limits: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }


class ScrapingManager:
    """Manager for web scraping operations."""

    def __init__(self, config: dict):
        self.config = config
        self.session = None
        self.delay_range = (
            self.config['agents']['execution']['min_delay'],
            self.config['agents']['execution']['max_delay']
        )

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def scrape_fanduel_odds(self, sport: str) -> list[dict]:
        """Scrape odds from FanDuel website."""
        try:
            # This would implement actual web scraping
            # For now, return mock data

            await self._random_delay()

            # Mock scraped data
            scraped_odds = [
                {
                    'sport': sport,
                    'home_team': 'Lakers',
                    'away_team': 'Warriors',
                    'moneyline_home': -150,
                    'moneyline_away': +130,
                    'spread_home': -3.5,
                    'spread_away': +3.5,
                    'total_over': 220.5,
                    'total_under': 220.5,
                    'timestamp': datetime.utcnow().isoformat()
                }
            ]

            return scraped_odds

        except Exception as e:
            logger.error(f"Error scraping FanDuel odds: {e}")
            return []

    async def scrape_draftkings_odds(self, sport: str) -> list[dict]:
        """Scrape odds from DraftKings website."""
        try:
            await self._random_delay()

            # Mock scraped data
            scraped_odds = [
                {
                    'sport': sport,
                    'home_team': 'Lakers',
                    'away_team': 'Warriors',
                    'moneyline_home': -145,
                    'moneyline_away': +125,
                    'spread_home': -3.0,
                    'spread_away': +3.0,
                    'total_over': 219.5,
                    'total_under': 219.5,
                    'timestamp': datetime.utcnow().isoformat()
                }
            ]

            return scraped_odds

        except Exception as e:
            logger.error(f"Error scraping DraftKings odds: {e}")
            return []

    async def _random_delay(self):
        """Add random delay to avoid detection."""
        delay = random.uniform(*self.delay_range)
        await asyncio.sleep(delay)

    def _parse_odds_from_html(self, html: str) -> list[dict]:
        """Parse odds from HTML content."""
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # This would implement actual HTML parsing
            # For now, return empty list
            return []

        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
            return []


class DataValidator:
    """Validate and clean fetched data."""

    @staticmethod
    def validate_event(event: Event) -> bool:
        """Validate event data."""
        try:
            # Check required fields
            if not event.home_team or not event.away_team:
                return False

            if not event.event_date:
                return False

            # Check date is in the future
            if event.event_date <= datetime.utcnow():
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating event: {e}")
            return False

    @staticmethod
    def validate_odds(odds: Odds) -> bool:
        """Validate odds data."""
        try:
            # Check required fields
            if not odds.event_id or not odds.platform or not odds.market_type:
                return False

            if not odds.selection or not odds.odds:
                return False

            # Check odds are reasonable
            if float(odds.odds) < -1000 or float(odds.odds) > 1000:
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating odds: {e}")
            return False

    @staticmethod
    def clean_team_name(name: str) -> str:
        """Clean and standardize team names."""
        try:
            # Remove common prefixes/suffixes
            name = name.strip()
            name = name.replace('  ', ' ')  # Remove double spaces

            # Standardize common variations
            name_mapping = {
                'LA Lakers': 'Los Angeles Lakers',
                'LA Clippers': 'Los Angeles Clippers',
                'NY Knicks': 'New York Knicks',
                'NY Nets': 'Brooklyn Nets'
            }

            return name_mapping.get(name, name)

        except Exception as e:
            logger.error(f"Error cleaning team name: {e}")
            return name
