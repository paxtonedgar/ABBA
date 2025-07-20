#!/usr/bin/env python3
"""
Test script for ABBA caching system.
Demonstrates how caching reduces API overhead and improves performance.
"""

import asyncio
import time
from datetime import datetime, timedelta

import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import core components
from cache_manager import APIOptimizer, CacheManager, DataPersistenceManager
from data_fetcher import DataFetcher


async def test_caching_performance():
    """Test caching system performance and API overhead reduction."""
    print("🧪 Testing ABBA Caching System Performance")
    print("=" * 60)

    try:
        # Load configuration
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        # Initialize components
        cache_manager = CacheManager(config)
        api_optimizer = APIOptimizer(cache_manager, config)
        data_fetcher = DataFetcher(config)

        print("\n1. Testing Cache Manager...")

        # Test basic caching
        test_data = {
            "events": [
                {"id": "1", "home": "Lakers", "away": "Warriors", "date": "2024-01-15"},
                {"id": "2", "home": "Celtics", "away": "Heat", "date": "2024-01-16"},
            ],
            "odds": [
                {"event_id": "1", "platform": "fanduel", "odds": -110},
                {"event_id": "1", "platform": "draftkings", "odds": -105},
            ],
        }

        # Store data in cache
        await cache_manager.set("test_events", test_data["events"], "events")
        await cache_manager.set("test_odds", test_data["odds"], "odds")

        # Retrieve from cache
        cached_events = await cache_manager.get("test_events", "events")
        cached_odds = await cache_manager.get("test_odds", "odds")

        print(f"✅ Cached events: {len(cached_events)} events retrieved")
        print(f"✅ Cached odds: {len(cached_odds)} odds retrieved")

        # Test cache statistics
        stats = await cache_manager.get_cache_stats()
        print(
            f"✅ Cache stats: {stats['total_entries']} entries, {stats['hit_rate']:.2%} hit rate"
        )

        print("\n2. Testing API Optimization...")

        # Simulate API calls with caching
        api_calls = [
            ("odds_api", "/sports/basketball_nba/odds", {"regions": "us"}),
            ("odds_api", "/sports/football_nfl/odds", {"regions": "us"}),
            ("sports_data_io", "/nba/stats", {"season": "2024"}),
        ]

        start_time = time.time()

        # First call - should hit API
        print("   Making first API call (should hit API)...")
        result1 = await api_optimizer.optimized_api_call(*api_calls[0])

        # Second call - should hit cache
        print("   Making second API call (should hit cache)...")
        result2 = await api_optimizer.optimized_api_call(*api_calls[0])

        # Third call - different endpoint
        print("   Making third API call (different endpoint)...")
        result3 = await api_optimizer.optimized_api_call(*api_calls[1])

        end_time = time.time()
        total_time = end_time - start_time

        print(f"✅ API calls completed in {total_time:.2f} seconds")

        # Test batch operations
        print("\n3. Testing Batch Operations...")

        batch_keys = ["batch_test_1", "batch_test_2", "batch_test_3"]
        batch_data = {
            "batch_test_1": {"data": "test1", "timestamp": datetime.utcnow()},
            "batch_test_2": {"data": "test2", "timestamp": datetime.utcnow()},
            "batch_test_3": {"data": "test3", "timestamp": datetime.utcnow()},
        }

        # Batch set
        await cache_manager.batch_set(batch_data, "api_responses")

        # Batch get
        batch_results = await cache_manager.batch_get(batch_keys, "api_responses")
        print(f"✅ Batch operations: {len(batch_results)} items retrieved")

        print("\n4. Testing Data Persistence...")

        persistence_manager = DataPersistenceManager(config)

        # Test data archiving
        test_archive_data = [
            {"event_id": "1", "odds": -110, "timestamp": datetime.utcnow()},
            {"event_id": "2", "odds": -105, "timestamp": datetime.utcnow()},
            {"event_id": "3", "odds": +150, "timestamp": datetime.utcnow()},
        ]

        await persistence_manager.archive_data("odds_data", test_archive_data)

        # Test loading archived data
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow() + timedelta(days=1)

        archived_data = await persistence_manager.load_archived_data(
            "odds_data", start_date, end_date
        )
        print(f"✅ Data persistence: {len(archived_data)} records archived and loaded")

        print("\n5. Testing Cache Performance Comparison...")

        # Simulate repeated API calls without caching
        print("   Testing without caching (simulated)...")
        start_time = time.time()
        for i in range(10):
            # Simulate API call delay
            await asyncio.sleep(0.1)
        no_cache_time = time.time() - start_time

        # Simulate repeated API calls with caching
        print("   Testing with caching...")
        start_time = time.time()
        for i in range(10):
            await cache_manager.get_or_fetch(
                f"performance_test_{i}",
                lambda: asyncio.sleep(0.1) or {"data": f"test_{i}"},
                "api_responses",
            )
        cache_time = time.time() - start_time

        print("✅ Performance comparison:")
        print(f"   Without caching: {no_cache_time:.2f} seconds")
        print(f"   With caching: {cache_time:.2f} seconds")
        print(
            f"   Speed improvement: {((no_cache_time - cache_time) / no_cache_time * 100):.1f}%"
        )

        print("\n6. Testing Cache Invalidation...")

        # Invalidate cache entries
        invalidated_count = await cache_manager.invalidate("test_", "events")
        print(f"✅ Cache invalidation: {invalidated_count} entries removed")

        # Final cache statistics
        final_stats = await cache_manager.get_cache_stats()
        print(
            f"✅ Final cache stats: {final_stats['total_entries']} entries, {final_stats['hit_rate']:.2%} hit rate"
        )

        print("\n7. Testing Rate Limiting...")

        # Test rate limiting
        rate_limit_tests = []
        for i in range(15):  # Try to exceed rate limit
            try:
                result = await api_optimizer.optimized_api_call(
                    "odds_api", f"/test/{i}", {}
                )
                rate_limit_tests.append(result)
            except Exception as e:
                if "Rate limit exceeded" in str(e):
                    print(f"   Rate limit hit at call {i+1} (expected)")
                    break

        print(f"✅ Rate limiting: {len(rate_limit_tests)} calls made before limit")

        print("\n🎉 Caching System Test Completed Successfully!")

        # Summary
        print("\n" + "=" * 60)
        print("📊 CACHING SYSTEM SUMMARY:")
        print("=" * 60)
        print(f"• Cache hit rate: {final_stats['hit_rate']:.2%}")
        print(f"• Total cache entries: {final_stats['total_entries']}")
        print(f"• Cache size: {final_stats['cache_size_mb']:.2f} MB")
        print(
            f"• Performance improvement: {((no_cache_time - cache_time) / no_cache_time * 100):.1f}%"
        )
        print(
            f"• API calls saved: {final_stats['hits']} hits vs {final_stats['misses']} misses"
        )
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error in caching test: {e}")
        import traceback

        traceback.print_exc()


async def test_real_api_integration():
    """Test caching with real API integration."""
    print("\n🔗 Testing Real API Integration with Caching")
    print("=" * 60)

    try:
        # Load configuration
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        # Initialize data fetcher with caching
        async with DataFetcher(config) as fetcher:
            print("\n1. Testing Events Fetching with Cache...")

            # First fetch (should hit API)
            start_time = time.time()
            events1 = await fetcher.fetch_events("basketball_nba")
            first_fetch_time = time.time() - start_time
            print(f"   First fetch: {len(events1)} events in {first_fetch_time:.2f}s")

            # Second fetch (should hit cache)
            start_time = time.time()
            events2 = await fetcher.fetch_events("basketball_nba")
            second_fetch_time = time.time() - start_time
            print(f"   Second fetch: {len(events2)} events in {second_fetch_time:.2f}s")

            # Force refresh
            start_time = time.time()
            events3 = await fetcher.fetch_events("basketball_nba", force_refresh=True)
            refresh_time = time.time() - start_time
            print(f"   Force refresh: {len(events3)} events in {refresh_time:.2f}s")

            print(
                f"✅ Cache performance: {((first_fetch_time - second_fetch_time) / first_fetch_time * 100):.1f}% improvement"
            )

            print("\n2. Testing API Limits...")

            # Check API limits
            limits = await fetcher.check_api_limits()
            print(f"   API Status: {limits.get('status', 'unknown')}")
            if "remaining_requests" in limits:
                print(f"   Remaining requests: {limits['remaining_requests']}")

            print("\n3. Testing Cache Statistics...")

            # Get cache statistics
            cache_stats = await fetcher.cache_manager.get_cache_stats()
            print(f"   Cache hit rate: {cache_stats.get('hit_rate', 0):.2%}")
            print(f"   Total entries: {cache_stats.get('total_entries', 0)}")
            print(f"   Cache size: {cache_stats.get('cache_size_mb', 0):.2f} MB")

        print("\n✅ Real API Integration Test Completed!")

    except Exception as e:
        print(f"❌ Error in real API test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_caching_performance())
    asyncio.run(test_real_api_integration())
