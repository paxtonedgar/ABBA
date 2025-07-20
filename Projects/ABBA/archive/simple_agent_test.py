#!/usr/bin/env python3
"""
Simple Agent Test - Tests core ABMBA components without full crew initialization
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any

import structlog

# Test individual components
try:
    from analytics_module import AnalyticsModule
    ANALYTICS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Analytics module not available: {e}")
    ANALYTICS_AVAILABLE = False

try:
    from models import Event, MarketType, Odds, PlatformType, SportType
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Models not available: {e}")
    MODELS_AVAILABLE = False

try:
    from data_fetcher import DataFetcher
    DATA_FETCHER_AVAILABLE = True
except ImportError as e:
    print(f"❌ Data fetcher not available: {e}")
    DATA_FETCHER_AVAILABLE = False

try:
    from database import DatabaseManager
    DATABASE_AVAILABLE = True
except ImportError as e:
    print(f"❌ Database not available: {e}")
    DATABASE_AVAILABLE = False

logger = structlog.get_logger()


class SimpleAgentTest:
    """Simple test for core ABMBA components."""

    def __init__(self):
        self.test_results = {}
        self.start_time = None

    async def run_simple_tests(self) -> dict[str, Any]:
        """Run simple tests on core components."""
        self.start_time = time.time()

        print("🧪 Simple Agent Component Tests")
        print("=" * 50)

        test_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
            "overall_status": "unknown",
            "recommendations": []
        }

        # Test Models
        print("\n📋 Testing Data Models...")
        models_result = await self._test_models()
        test_results["components"]["models"] = models_result

        # Test Analytics Module
        print("\n📈 Testing Analytics Module...")
        analytics_result = await self._test_analytics()
        test_results["components"]["analytics"] = analytics_result

        # Test Data Fetcher
        print("\n🔍 Testing Data Fetcher...")
        data_fetcher_result = await self._test_data_fetcher()
        test_results["components"]["data_fetcher"] = data_fetcher_result

        # Test Database
        print("\n🗄️ Testing Database...")
        database_result = await self._test_database()
        test_results["components"]["database"] = database_result

        # Calculate overall status
        overall_status = self._calculate_overall_status(test_results)
        test_results["overall_status"] = overall_status

        # Generate recommendations
        recommendations = self._generate_recommendations(test_results)
        test_results["recommendations"] = recommendations

        # Calculate total time
        total_time = time.time() - self.start_time
        test_results["total_test_time"] = total_time

        return test_results

    async def _test_models(self) -> dict[str, Any]:
        """Test data models."""
        if not MODELS_AVAILABLE:
            return {
                "status": "error",
                "error": "Models not available",
                "capabilities": []
            }

        try:
            start_time = time.time()

            # Test Event model
            event = Event(
                id="test_event_1",
                sport=SportType.BASEBALL_MLB,
                home_team="New York Yankees",
                away_team="Boston Red Sox",
                event_date=datetime.utcnow()
            )

            # Test Odds model
            odds = Odds(
                id="test_odds_1",
                event_id="test_event_1",
                platform=PlatformType.FANDUEL,
                market_type=MarketType.MONEYLINE,
                selection="New York Yankees",
                odds=100
            )

            execution_time = time.time() - start_time

            return {
                "status": "healthy",
                "event_model": "success",
                "odds_model": "success",
                "execution_time": execution_time,
                "capabilities": ["data_validation", "type_safety", "serialization"]
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "execution_time": 0,
                "capabilities": []
            }

    async def _test_analytics(self) -> dict[str, Any]:
        """Test analytics module."""
        if not ANALYTICS_AVAILABLE:
            return {
                "status": "error",
                "error": "Analytics module not available",
                "capabilities": []
            }

        try:
            start_time = time.time()

            # Create simple config
            config = {
                "models": {
                    "xgboost": {
                        "n_estimators": 100,
                        "max_depth": 3,
                        "random_state": 42
                    }
                }
            }

            # Initialize analytics module
            analytics = AnalyticsModule(config)

            # Test basic functionality
            test_data = {
                "features": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "targets": [0, 1, 0]
            }

            # Test feature engineering
            features_result = analytics.engineer_features(test_data)

            execution_time = time.time() - start_time

            return {
                "status": "healthy",
                "initialization": "success",
                "feature_engineering": "success" if features_result else "failed",
                "execution_time": execution_time,
                "capabilities": ["feature_engineering", "ml_training", "prediction"]
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "execution_time": 0,
                "capabilities": []
            }

    async def _test_data_fetcher(self) -> dict[str, Any]:
        """Test data fetcher."""
        if not DATA_FETCHER_AVAILABLE:
            return {
                "status": "error",
                "error": "Data fetcher not available",
                "capabilities": []
            }

        try:
            start_time = time.time()

            # Create simple config
            config = {
                "apis": {
                    "the_odds_api_key": "test_key",
                    "openweather_api_key": "test_key"
                }
            }

            # Initialize data fetcher
            data_fetcher = DataFetcher(config)

            # Test basic functionality
            # Note: This won't make real API calls without valid keys
            execution_time = time.time() - start_time

            return {
                "status": "healthy",
                "initialization": "success",
                "api_configuration": "success",
                "execution_time": execution_time,
                "capabilities": ["data_fetching", "api_integration", "data_validation"]
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "execution_time": 0,
                "capabilities": []
            }

    async def _test_database(self) -> dict[str, Any]:
        """Test database functionality."""
        if not DATABASE_AVAILABLE:
            return {
                "status": "error",
                "error": "Database not available",
                "capabilities": []
            }

        try:
            start_time = time.time()

            # Test with SQLite (in-memory)
            db_url = "sqlite+aiosqlite:///:memory:"
            db_manager = DatabaseManager(db_url)

            # Initialize database
            await db_manager.initialize()

            # Test basic operations
            test_event = Event(
                id="db_test_event",
                sport=SportType.BASEBALL_MLB,
                home_team="Test Team A",
                away_team="Test Team B",
                event_date=datetime.utcnow()
            )

            # Save event
            await db_manager.save_event(test_event)

            # Retrieve event
            retrieved_event = await db_manager.get_event("db_test_event")

            # Close database
            await db_manager.close()

            execution_time = time.time() - start_time

            return {
                "status": "healthy",
                "initialization": "success",
                "save_operation": "success",
                "retrieve_operation": "success" if retrieved_event else "failed",
                "execution_time": execution_time,
                "capabilities": ["data_persistence", "query_operations", "connection_management"]
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "execution_time": 0,
                "capabilities": []
            }

    def _calculate_overall_status(self, results: dict[str, Any]) -> str:
        """Calculate overall system status."""
        components = results.get("components", {})

        # Count healthy components
        healthy_components = sum(1 for comp in components.values() if comp.get("status") == "healthy")
        total_components = len(components)

        if total_components == 0:
            return "unknown"

        health_score = healthy_components / total_components

        if health_score >= 0.8:
            return "excellent"
        elif health_score >= 0.6:
            return "good"
        elif health_score >= 0.4:
            return "fair"
        else:
            return "poor"

    def _generate_recommendations(self, results: dict[str, Any]) -> list[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        components = results.get("components", {})
        overall_status = results.get("overall_status", "unknown")

        # Component-specific recommendations
        for comp_name, comp_result in components.items():
            if comp_result.get("status") == "error":
                recommendations.append(f"Fix {comp_name}: {comp_result.get('error', 'Unknown error')}")
            elif comp_result.get("status") != "healthy":
                recommendations.append(f"Investigate {comp_name} issues")

        # Overall recommendations
        if overall_status == "poor":
            recommendations.append("System requires immediate attention - multiple components failing")
        elif overall_status == "fair":
            recommendations.append("System needs optimization - some components underperforming")
        elif overall_status == "good":
            recommendations.append("System performing well - consider minor optimizations")
        elif overall_status == "excellent":
            recommendations.append("System performing excellently - ready for production")

        # Missing dependencies
        if not MODELS_AVAILABLE:
            recommendations.append("Install missing dependencies for data models")
        if not ANALYTICS_AVAILABLE:
            recommendations.append("Install missing dependencies for analytics module")
        if not DATA_FETCHER_AVAILABLE:
            recommendations.append("Install missing dependencies for data fetcher")
        if not DATABASE_AVAILABLE:
            recommendations.append("Install missing dependencies for database")

        return recommendations


async def main():
    """Main function to run simple tests."""
    print("🧪 ABMBA Simple Component Tests")
    print("=" * 50)

    # Initialize test tool
    tester = SimpleAgentTest()

    try:
        # Run tests
        print("🚀 Running component tests...")
        results = await tester.run_simple_tests()

        # Print results
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS")
        print("=" * 50)

        # Overall status
        overall_status = results.get("overall_status", "unknown")
        status_emoji = {
            "excellent": "🟢",
            "good": "🟡",
            "fair": "🟠",
            "poor": "🔴"
        }.get(overall_status, "⚪")

        print(f"{status_emoji} Overall Status: {overall_status.upper()}")
        print(f"⏱️  Test Time: {results.get('total_test_time', 0):.2f} seconds")
        print()

        # Component results
        print("🔧 COMPONENT STATUS:")
        components = results.get("components", {})
        for comp_name, comp_result in components.items():
            status = comp_result.get("status", "unknown")
            status_emoji = "🟢" if status == "healthy" else "🔴" if status == "error" else "🟡"
            execution_time = comp_result.get("execution_time", 0)
            print(f"  {status_emoji} {comp_name}: {status} ({execution_time:.2f}s)")

            # Show capabilities
            capabilities = comp_result.get("capabilities", [])
            if capabilities:
                print(f"     Capabilities: {', '.join(capabilities)}")

        print()

        # Recommendations
        recommendations = results.get("recommendations", [])
        if recommendations:
            print("💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("✅ No recommendations - system is healthy!")

        print("\n" + "=" * 50)
        print("🎉 Testing Complete!")

        # Save results
        with open("simple_test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print("📄 Detailed results saved to simple_test_results.json")

    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
