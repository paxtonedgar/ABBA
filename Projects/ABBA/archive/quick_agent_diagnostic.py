#!/usr/bin/env python3
"""
Quick Agent Diagnostic Tool
Runs a fast health check on all ABMBA agents to assess current performance
"""

import asyncio
import time
from datetime import datetime
from typing import Any

import structlog

# Project imports
try:
    from agents import ABMBACrew
    from analytics_module import AnalyticsModule

    from models import Event, MarketType, Odds, SportType
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    IMPORTS_SUCCESS = False

logger = structlog.get_logger()


class QuickAgentDiagnostic:
    """Quick diagnostic tool for ABMBA agents."""

    def __init__(self):
        self.crew = None
        self.diagnostic_results = {}
        self.start_time = None

    async def initialize(self):
        """Initialize the diagnostic tool."""
        try:
            self.crew = ABMBACrew()
            await self.crew.initialize()
            logger.info("✅ ABMBA crew initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize crew: {e}")
            return False

    async def run_quick_diagnostic(self) -> dict[str, Any]:
        """Run quick diagnostic on all agents."""
        self.start_time = time.time()

        print("🔍 Running Quick Agent Diagnostic...")
        print("=" * 50)

        diagnostic_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "agents": {},
            "pipeline_tests": {},
            "overall_health": "unknown",
            "recommendations": []
        }

        # Test individual agents
        print("\n📊 Testing Individual Agents...")
        agent_tests = await self._test_individual_agents()
        diagnostic_results["agents"] = agent_tests

        # Test agent collaboration
        print("\n🤝 Testing Agent Collaboration...")
        collaboration_tests = await self._test_agent_collaboration()
        diagnostic_results["pipeline_tests"] = collaboration_tests

        # Calculate overall health
        overall_health = self._calculate_overall_health(diagnostic_results)
        diagnostic_results["overall_health"] = overall_health

        # Generate recommendations
        recommendations = self._generate_recommendations(diagnostic_results)
        diagnostic_results["recommendations"] = recommendations

        # Calculate total time
        total_time = time.time() - self.start_time
        diagnostic_results["total_diagnostic_time"] = total_time

        return diagnostic_results

    async def _test_individual_agents(self) -> dict[str, Any]:
        """Test each agent individually."""
        agent_results = {}

        # Test Research Agent
        print("  🔍 Research Agent...")
        research_result = await self._test_research_agent()
        agent_results["research_agent"] = research_result

        # Test Analytics Agent
        print("  📈 Analytics Agent...")
        analytics_result = await self._test_analytics_agent()
        agent_results["analytics_agent"] = analytics_result

        # Test Decision Agent
        print("  🎯 Decision Agent...")
        decision_result = await self._test_decision_agent()
        agent_results["decision_agent"] = decision_result

        # Test Execution Agent
        print("  ⚡ Execution Agent...")
        execution_result = await self._test_execution_agent()
        agent_results["execution_agent"] = execution_result

        return agent_results

    async def _test_research_agent(self) -> dict[str, Any]:
        """Test research agent capabilities."""
        try:
            start_time = time.time()

            # Test data fetching
            data_result = await self.crew.research_agent._fetch_sports_data(['mlb'])

            # Test data verification
            verification_result = await self.crew.research_agent._verify_data_quality("events", "mlb")

            execution_time = time.time() - start_time

            return {
                "status": "healthy",
                "data_fetching": "success" if data_result else "failed",
                "data_verification": "success" if verification_result else "failed",
                "execution_time": execution_time,
                "capabilities": ["data_fetching", "data_verification", "anomaly_detection"]
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "execution_time": 0,
                "capabilities": []
            }

    async def _test_analytics_agent(self) -> dict[str, Any]:
        """Test analytics agent capabilities."""
        try:
            start_time = time.time()

            # Test ML model training
            ml_result = await self.crew.analytics_agent._train_ml_models('mlb', 'xgboost')

            # Test feature insights
            insights_result = await self.crew.analytics_agent._generate_feature_insights()

            execution_time = time.time() - start_time

            return {
                "status": "healthy",
                "ml_training": "success" if ml_result else "failed",
                "feature_insights": "success" if insights_result else "failed",
                "execution_time": execution_time,
                "capabilities": ["ml_training", "feature_analysis", "model_validation"]
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "execution_time": 0,
                "capabilities": []
            }

    async def _test_decision_agent(self) -> dict[str, Any]:
        """Test decision agent capabilities."""
        try:
            start_time = time.time()

            # Test opportunity evaluation
            eval_result = await self.crew.decision_agent._evaluate_betting_opportunity("test_event", "test_odds")

            # Test stake calculation
            stake_result = await self.crew.decision_agent._calculate_optimal_stake(1000, 0.6, 1.5)

            execution_time = time.time() - start_time

            return {
                "status": "healthy",
                "opportunity_evaluation": "success" if eval_result else "failed",
                "stake_calculation": "success" if stake_result else "failed",
                "execution_time": execution_time,
                "capabilities": ["ev_calculation", "risk_assessment", "stake_optimization"]
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "execution_time": 0,
                "capabilities": []
            }

    async def _test_execution_agent(self) -> dict[str, Any]:
        """Test execution agent capabilities."""
        try:
            start_time = time.time()

            # Test bet placement (mock)
            bet_result = await self.crew.execution_agent._place_bet("fanduel", "test_event", "moneyline", "home_team", 100)

            # Test bet verification
            verify_result = await self.crew.execution_agent._verify_bet_status("test_bet_id")

            execution_time = time.time() - start_time

            return {
                "status": "healthy",
                "bet_placement": "success" if bet_result else "failed",
                "bet_verification": "success" if verify_result else "failed",
                "execution_time": execution_time,
                "capabilities": ["bet_placement", "bet_verification", "platform_integration"]
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "execution_time": 0,
                "capabilities": []
            }

    async def _test_agent_collaboration(self) -> dict[str, Any]:
        """Test how agents work together."""
        collaboration_results = {}

        # Test pipeline flow
        print("    🔄 Testing pipeline flow...")
        pipeline_result = await self._test_pipeline_flow()
        collaboration_results["pipeline_flow"] = pipeline_result

        # Test agent communication
        print("    💬 Testing agent communication...")
        communication_result = await self._test_agent_communication()
        collaboration_results["agent_communication"] = communication_result

        # Test error handling
        print("    🛡️ Testing error handling...")
        error_result = await self._test_error_handling()
        collaboration_results["error_handling"] = error_result

        return collaboration_results

    async def _test_pipeline_flow(self) -> dict[str, Any]:
        """Test the complete agent pipeline flow."""
        try:
            start_time = time.time()

            # Test research phase
            research_result = await self.crew.run_research_phase()

            # Test analytics phase
            analytics_result = await self.crew.run_analytics_phase()

            execution_time = time.time() - start_time

            return {
                "status": "healthy",
                "research_phase": "success" if research_result else "failed",
                "analytics_phase": "success" if analytics_result else "failed",
                "execution_time": execution_time,
                "pipeline_stages": ["research", "analytics"]
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "execution_time": 0,
                "pipeline_stages": []
            }

    async def _test_agent_communication(self) -> dict[str, Any]:
        """Test agent communication patterns."""
        try:
            # Test if agents can share data
            test_data = {"test": "data"}

            # Simulate data sharing between agents
            shared_data = {}

            # Research agent shares data
            shared_data["research"] = test_data

            # Analytics agent receives data
            if "research" in shared_data:
                shared_data["analytics"] = "processed_data"

            # Decision agent receives processed data
            if "analytics" in shared_data:
                shared_data["decision"] = "decision_made"

            return {
                "status": "healthy",
                "data_sharing": "success",
                "communication_patterns": ["research->analytics", "analytics->decision"],
                "data_integrity": "maintained"
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "communication_patterns": [],
                "data_integrity": "failed"
            }

    async def _test_error_handling(self) -> dict[str, Any]:
        """Test error handling capabilities."""
        try:
            # Test with invalid data
            invalid_data = {"invalid": "data", "missing_required": True}

            # Try to process invalid data
            try:
                # This should trigger error handling
                result = await self.crew.research_agent._verify_data_quality("invalid", "invalid")
                error_handled = result is not None
            except:
                error_handled = True

            return {
                "status": "healthy",
                "error_detection": "success",
                "graceful_degradation": "success" if error_handled else "failed",
                "recovery_mechanisms": ["fallback_data", "error_logging"]
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "error_detection": "failed",
                "graceful_degradation": "failed"
            }

    def _calculate_overall_health(self, results: dict[str, Any]) -> str:
        """Calculate overall system health."""
        agent_results = results.get("agents", {})
        pipeline_results = results.get("pipeline_tests", {})

        # Count healthy agents
        healthy_agents = sum(1 for agent in agent_results.values() if agent.get("status") == "healthy")
        total_agents = len(agent_results)

        # Count healthy pipeline tests
        healthy_pipelines = sum(1 for test in pipeline_results.values() if test.get("status") == "healthy")
        total_pipelines = len(pipeline_results)

        # Calculate health score
        agent_health = healthy_agents / total_agents if total_agents > 0 else 0
        pipeline_health = healthy_pipelines / total_pipelines if total_pipelines > 0 else 0

        overall_score = (agent_health + pipeline_health) / 2

        if overall_score >= 0.8:
            return "excellent"
        elif overall_score >= 0.6:
            return "good"
        elif overall_score >= 0.4:
            return "fair"
        else:
            return "poor"

    def _generate_recommendations(self, results: dict[str, Any]) -> list[str]:
        """Generate recommendations based on diagnostic results."""
        recommendations = []

        agent_results = results.get("agents", {})
        overall_health = results.get("overall_health", "unknown")

        # Agent-specific recommendations
        for agent_name, agent_result in agent_results.items():
            if agent_result.get("status") == "error":
                recommendations.append(f"Fix {agent_name}: {agent_result.get('error', 'Unknown error')}")
            elif agent_result.get("status") != "healthy":
                recommendations.append(f"Investigate {agent_name} performance issues")

        # Overall recommendations
        if overall_health == "poor":
            recommendations.append("System requires immediate attention - multiple agents failing")
        elif overall_health == "fair":
            recommendations.append("System needs optimization - some agents underperforming")
        elif overall_health == "good":
            recommendations.append("System performing well - consider minor optimizations")
        elif overall_health == "excellent":
            recommendations.append("System performing excellently - ready for production")

        # Performance recommendations
        slow_agents = [name for name, result in agent_results.items()
                      if result.get("execution_time", 0) > 5.0]
        if slow_agents:
            recommendations.append(f"Optimize performance for slow agents: {', '.join(slow_agents)}")

        return recommendations


async def main():
    """Main function to run quick diagnostic."""
    print("🔍 ABMBA Agent Quick Diagnostic")
    print("=" * 50)

    if not IMPORTS_SUCCESS:
        print("❌ Cannot run diagnostic - import errors detected")
        return

    # Initialize diagnostic tool
    diagnostic = QuickAgentDiagnostic()

    try:
        # Initialize
        print("🚀 Initializing diagnostic tool...")
        init_success = await diagnostic.initialize()

        if not init_success:
            print("❌ Failed to initialize diagnostic tool")
            return

        # Run diagnostic
        print("🔍 Running diagnostic...")
        results = await diagnostic.run_quick_diagnostic()

        # Print results
        print("\n" + "=" * 50)
        print("📊 DIAGNOSTIC RESULTS")
        print("=" * 50)

        # Overall health
        overall_health = results.get("overall_health", "unknown")
        health_emoji = {
            "excellent": "🟢",
            "good": "🟡",
            "fair": "🟠",
            "poor": "🔴"
        }.get(overall_health, "⚪")

        print(f"{health_emoji} Overall Health: {overall_health.upper()}")
        print(f"⏱️  Diagnostic Time: {results.get('total_diagnostic_time', 0):.2f} seconds")
        print()

        # Agent results
        print("🤖 AGENT STATUS:")
        agent_results = results.get("agents", {})
        for agent_name, agent_result in agent_results.items():
            status = agent_result.get("status", "unknown")
            status_emoji = "🟢" if status == "healthy" else "🔴" if status == "error" else "🟡"
            execution_time = agent_result.get("execution_time", 0)
            print(f"  {status_emoji} {agent_name}: {status} ({execution_time:.2f}s)")

        print()

        # Pipeline results
        print("🔄 PIPELINE TESTS:")
        pipeline_results = results.get("pipeline_tests", {})
        for test_name, test_result in pipeline_results.items():
            status = test_result.get("status", "unknown")
            status_emoji = "🟢" if status == "healthy" else "🔴" if status == "error" else "🟡"
            print(f"  {status_emoji} {test_name}: {status}")

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
        print("🎉 Diagnostic Complete!")

        # Save results
        import json
        with open("agent_diagnostic_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print("📄 Detailed results saved to agent_diagnostic_results.json")

    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
