#!/usr/bin/env python3
"""
Advanced Caching Manager for ABBA System
Reduces API overhead through intelligent caching and data persistence.
"""

import gzip
import hashlib
import json
import pickle
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os
import structlog

logger = structlog.get_logger()


class CacheManager:
    """Advanced caching manager for reducing API overhead and improving performance."""

    def __init__(self, config: dict):
        self.config = config
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)

        # Cache configuration
        self.cache_config = {
            'odds': {
                'ttl': 300,  # 5 minutes for odds data
                'max_size': 10000,
                'compression': True
            },
            'events': {
                'ttl': 3600,  # 1 hour for events
                'max_size': 5000,
                'compression': True
            },
            'historical': {
                'ttl': 86400,  # 24 hours for historical data
                'max_size': 50000,
                'compression': True
            },
            'ml_models': {
                'ttl': 604800,  # 1 week for ML models
                'max_size': 1000,
                'compression': True
            },
            'api_responses': {
                'ttl': 1800,  # 30 minutes for API responses
                'max_size': 20000,
                'compression': True
            }
        }

        # Initialize cache databases
        self._init_cache_db()

        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }

        logger.info("Cache Manager initialized")

    def _init_cache_db(self):
        """Initialize SQLite cache database."""
        self.cache_db_path = self.cache_dir / "cache.db"

        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    data BLOB,
                    cache_type TEXT,
                    created_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at 
                ON cache_entries(expires_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_type 
                ON cache_entries(cache_type)
            """)

    async def get(self, key: str, cache_type: str = 'api_responses') -> Any | None:
        """Get data from cache."""
        try:
            cache_key = self._generate_key(key, cache_type)

            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute("""
                    SELECT data, expires_at FROM cache_entries 
                    WHERE key = ? AND expires_at > ?
                """, (cache_key, datetime.utcnow().isoformat()))

                row = cursor.fetchone()

                if row:
                    # Update access statistics
                    conn.execute("""
                        UPDATE cache_entries 
                        SET access_count = access_count + 1, 
                            last_accessed = ? 
                        WHERE key = ?
                    """, (datetime.utcnow().isoformat(), cache_key))

                    # Decompress and deserialize data
                    data_blob = row[0]
                    data = self._deserialize_data(data_blob)

                    self.stats['hits'] += 1
                    logger.debug(f"Cache hit for key: {key[:50]}...")
                    return data
                else:
                    self.stats['misses'] += 1
                    logger.debug(f"Cache miss for key: {key[:50]}...")
                    return None

        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None

    async def set(self, key: str, data: Any, cache_type: str = 'api_responses',
                  ttl: int | None = None) -> bool:
        """Store data in cache."""
        try:
            cache_key = self._generate_key(key, cache_type)

            # Get TTL from config or use default
            if ttl is None:
                ttl = self.cache_config[cache_type]['ttl']

            expires_at = datetime.utcnow() + timedelta(seconds=ttl)

            # Serialize and compress data
            data_blob = self._serialize_data(data)

            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, data, cache_type, created_at, expires_at, access_count, last_accessed)
                    VALUES (?, ?, ?, ?, ?, 0, ?)
                """, (
                    cache_key, data_blob, cache_type,
                    datetime.utcnow().isoformat(),
                    expires_at.isoformat(),
                    datetime.utcnow().isoformat()
                ))

            # Cleanup old entries
            await self._cleanup_expired()

            logger.debug(f"Cached data for key: {key[:50]}... (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False

    async def get_or_fetch(self, key: str, fetch_func, cache_type: str = 'api_responses',
                          ttl: int | None = None, force_refresh: bool = False) -> Any:
        """Get from cache or fetch if not available."""
        if not force_refresh:
            cached_data = await self.get(key, cache_type)
            if cached_data is not None:
                return cached_data

        # Fetch fresh data
        try:
            data = await fetch_func()
            await self.set(key, data, cache_type, ttl)
            return data
        except Exception as e:
            logger.error(f"Error in get_or_fetch: {e}")
            # Return cached data even if expired as fallback
            return await self.get(key, cache_type)

    async def batch_get(self, keys: list[str], cache_type: str = 'api_responses') -> dict[str, Any]:
        """Get multiple items from cache."""
        results = {}

        for key in keys:
            data = await self.get(key, cache_type)
            if data is not None:
                results[key] = data

        return results

    async def batch_set(self, data_dict: dict[str, Any], cache_type: str = 'api_responses',
                       ttl: int | None = None) -> bool:
        """Store multiple items in cache."""
        try:
            for key, data in data_dict.items():
                await self.set(key, data, cache_type, ttl)
            return True
        except Exception as e:
            logger.error(f"Error in batch_set: {e}")
            return False

    async def invalidate(self, pattern: str, cache_type: str = None) -> int:
        """Invalidate cache entries matching pattern."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                if cache_type:
                    cursor = conn.execute("""
                        DELETE FROM cache_entries 
                        WHERE key LIKE ? AND cache_type = ?
                    """, (f"%{pattern}%", cache_type))
                else:
                    cursor = conn.execute("""
                        DELETE FROM cache_entries 
                        WHERE key LIKE ?
                    """, (f"%{pattern}%",))

                deleted_count = cursor.rowcount
                logger.info(f"Invalidated {deleted_count} cache entries matching pattern: {pattern}")
                return deleted_count

        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return 0

    async def _cleanup_expired(self):
        """Remove expired cache entries."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM cache_entries 
                    WHERE expires_at <= ?
                """, (datetime.utcnow().isoformat(),))

                evicted_count = cursor.rowcount
                if evicted_count > 0:
                    self.stats['evictions'] += evicted_count
                    logger.debug(f"Evicted {evicted_count} expired cache entries")

        except Exception as e:
            logger.error(f"Error cleaning up expired entries: {e}")

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                # Get total entries
                total_entries = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]

                # Get entries by type
                type_counts = {}
                for cache_type in self.cache_config.keys():
                    count = conn.execute("""
                        SELECT COUNT(*) FROM cache_entries WHERE cache_type = ?
                    """, (cache_type,)).fetchone()[0]
                    type_counts[cache_type] = count

                # Get hit rate
                total_requests = self.stats['hits'] + self.stats['misses']
                hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0

                return {
                    'total_entries': total_entries,
                    'entries_by_type': type_counts,
                    'hits': self.stats['hits'],
                    'misses': self.stats['misses'],
                    'evictions': self.stats['evictions'],
                    'hit_rate': hit_rate,
                    'cache_size_mb': self._get_cache_size_mb()
                }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}

    def _generate_key(self, key: str, cache_type: str) -> str:
        """Generate cache key with type prefix."""
        return f"{cache_type}:{hashlib.md5(key.encode()).hexdigest()}"

    def _serialize_data(self, data: Any) -> bytes:
        """Serialize and compress data."""
        try:
            # Serialize to pickle
            serialized = pickle.dumps(data)

            # Compress if enabled
            if self.cache_config.get('compression', True):
                compressed = gzip.compress(serialized)
                return compressed
            else:
                return serialized

        except Exception as e:
            logger.error(f"Error serializing data: {e}")
            return pickle.dumps(data)

    def _deserialize_data(self, data_blob: bytes) -> Any:
        """Deserialize and decompress data."""
        try:
            # Try to decompress first
            try:
                decompressed = gzip.decompress(data_blob)
                return pickle.loads(decompressed)
            except:
                # If decompression fails, try direct deserialization
                return pickle.loads(data_blob)

        except Exception as e:
            logger.error(f"Error deserializing data: {e}")
            return None

    def _get_cache_size_mb(self) -> float:
        """Get cache database size in MB."""
        try:
            size_bytes = self.cache_db_path.stat().st_size
            return size_bytes / (1024 * 1024)
        except:
            return 0.0


class APIOptimizer:
    """Optimizes API calls to reduce overhead."""

    def __init__(self, cache_manager: CacheManager, config: dict):
        self.cache_manager = cache_manager
        self.config = config
        self.rate_limits = {}
        self.last_request_time = {}
        self.request_counts = {}

        # API rate limit configuration
        self.api_limits = {
            'odds_api': {
                'requests_per_month': 500,
                'requests_per_minute': 10,
                'burst_limit': 5
            },
            'sports_data_io': {
                'requests_per_minute': 60,
                'burst_limit': 10
            }
        }

    async def optimized_api_call(self, api_name: str, endpoint: str, params: dict = None,
                               cache_type: str = 'api_responses', ttl: int = None) -> dict:
        """Make optimized API call with caching and rate limiting."""

        # Generate cache key
        cache_key = f"{api_name}:{endpoint}:{hashlib.md5(str(params).encode()).hexdigest()}"

        # Check rate limits
        if not await self._check_rate_limit(api_name):
            logger.warning(f"Rate limit exceeded for {api_name}, using cached data")
            cached_data = await self.cache_manager.get(cache_key, cache_type)
            if cached_data:
                return cached_data
            else:
                raise Exception(f"Rate limit exceeded and no cached data available for {api_name}")

        # Try to get from cache first
        cached_data = await self.cache_manager.get(cache_key, cache_type)
        if cached_data:
            return cached_data

        # Make API call
        try:
            # This would be the actual API call implementation
            # For now, return mock data
            data = await self._make_api_call(api_name, endpoint, params)

            # Cache the response
            await self.cache_manager.set(cache_key, data, cache_type, ttl)

            # Update rate limit tracking
            await self._update_rate_limit(api_name)

            return data

        except Exception as e:
            logger.error(f"API call failed for {api_name}: {e}")
            # Return cached data as fallback
            return cached_data or {}

    async def batch_api_calls(self, api_name: str, endpoints: list[str],
                            params_list: list[dict] = None) -> list[dict]:
        """Make multiple API calls efficiently."""
        results = []

        for i, endpoint in enumerate(endpoints):
            params = params_list[i] if params_list else {}

            # Check if we can make the call
            if await self._check_rate_limit(api_name):
                result = await self.optimized_api_call(api_name, endpoint, params)
                results.append(result)
            else:
                logger.warning(f"Rate limit reached, skipping remaining calls for {api_name}")
                break

        return results

    async def _check_rate_limit(self, api_name: str) -> bool:
        """Check if API call is within rate limits."""
        if api_name not in self.api_limits:
            return True

        limit_config = self.api_limits[api_name]
        current_time = datetime.utcnow()

        # Initialize tracking if needed
        if api_name not in self.request_counts:
            self.request_counts[api_name] = []
            self.last_request_time[api_name] = current_time

        # Clean old requests (older than 1 minute)
        minute_ago = current_time - timedelta(minutes=1)
        self.request_counts[api_name] = [
            req_time for req_time in self.request_counts[api_name]
            if req_time > minute_ago
        ]

        # Check minute limit
        if len(self.request_counts[api_name]) >= limit_config.get('requests_per_minute', 60):
            return False

        # Check burst limit
        last_requests = [
            req_time for req_time in self.request_counts[api_name]
            if req_time > current_time - timedelta(seconds=10)
        ]

        if len(last_requests) >= limit_config.get('burst_limit', 5):
            return False

        return True

    async def _update_rate_limit(self, api_name: str):
        """Update rate limit tracking."""
        if api_name not in self.request_counts:
            self.request_counts[api_name] = []

        self.request_counts[api_name].append(datetime.utcnow())
        self.last_request_time[api_name] = datetime.utcnow()

    async def _make_api_call(self, api_name: str, endpoint: str, params: dict = None) -> dict:
        """Make actual API call (placeholder implementation)."""
        # This would be the real API call implementation
        # For now, return mock data
        return {
            'api_name': api_name,
            'endpoint': endpoint,
            'params': params,
            'timestamp': datetime.utcnow().isoformat(),
            'data': f"Mock data for {api_name}:{endpoint}"
        }


class DataPersistenceManager:
    """Manages data persistence and archiving."""

    def __init__(self, config: dict):
        self.config = config
        self.archive_dir = Path("data_archive")
        self.archive_dir.mkdir(exist_ok=True)

        # Archive configuration
        self.archive_config = {
            'odds_data': {
                'retention_days': 365,
                'compression': True,
                'batch_size': 1000
            },
            'events_data': {
                'retention_days': 730,
                'compression': True,
                'batch_size': 500
            },
            'ml_predictions': {
                'retention_days': 180,
                'compression': True,
                'batch_size': 5000
            }
        }

    async def archive_data(self, data_type: str, data: list[dict],
                          timestamp: datetime = None) -> bool:
        """Archive data for long-term storage."""
        try:
            if timestamp is None:
                timestamp = datetime.utcnow()

            # Create archive file
            date_str = timestamp.strftime('%Y-%m-%d')
            filename = f"{data_type}_{date_str}.json.gz"
            filepath = self.archive_dir / filename

            # Compress and save data
            async with aiofiles.open(filepath, 'wb') as f:
                json_str = json.dumps(data, default=str)
                compressed = gzip.compress(json_str.encode())
                await f.write(compressed)

            logger.info(f"Archived {len(data)} {data_type} records to {filename}")
            return True

        except Exception as e:
            logger.error(f"Error archiving data: {e}")
            return False

    async def load_archived_data(self, data_type: str, start_date: datetime,
                               end_date: datetime) -> list[dict]:
        """Load archived data for a date range."""
        try:
            all_data = []

            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                filename = f"{data_type}_{date_str}.json.gz"
                filepath = self.archive_dir / filename

                if filepath.exists():
                    async with aiofiles.open(filepath, 'rb') as f:
                        compressed_data = await f.read()
                        json_str = gzip.decompress(compressed_data).decode()
                        data = json.loads(json_str)
                        all_data.extend(data)

                current_date += timedelta(days=1)

            logger.info(f"Loaded {len(all_data)} archived {data_type} records")
            return all_data

        except Exception as e:
            logger.error(f"Error loading archived data: {e}")
            return []

    async def cleanup_old_archives(self) -> int:
        """Remove old archived data based on retention policies."""
        try:
            deleted_count = 0
            current_time = datetime.utcnow()

            for data_type, config in self.archive_config.items():
                retention_days = config['retention_days']
                cutoff_date = current_time - timedelta(days=retention_days)

                # Find old files
                pattern = f"{data_type}_*.json.gz"
                for filepath in self.archive_dir.glob(pattern):
                    # Extract date from filename
                    date_str = filepath.stem.split('_')[1]
                    file_date = datetime.strptime(date_str, '%Y-%m-%d')

                    if file_date < cutoff_date:
                        await aiofiles.os.remove(filepath)
                        deleted_count += 1
                        logger.info(f"Deleted old archive: {filepath.name}")

            return deleted_count

        except Exception as e:
            logger.error(f"Error cleaning up old archives: {e}")
            return 0
