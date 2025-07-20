#!/usr/bin/env python3
"""
MLB Test Runner
Executes comprehensive MLB prediction system tests with detailed reporting.
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime

# Import testing modules
import pandas as pd
import structlog
from mlb_season_testing_system import MLBSeasonTester

logger = structlog.get_logger()


class MLBTestRunner:
    """Comprehensive test runner for MLB prediction system."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the test runner."""
        self.config_path = config_path
        self.tester = MLBSeasonTester(config_path)
        self.results = {
            'test_start_time': datetime.now().isoformat(),
            'test_end_time': None,
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'test_results': {},
            'performance_metrics': {},
            'summary': {}
        }

    async def run_comprehensive_tests(self, season: int = 2024) -> dict:
        """Run comprehensive testing suite."""
        logger.info(f"🚀 Starting comprehensive MLB {season} testing...")

        start_time = time.time()

        try:
            # Step 1: Run season testing system
            logger.info("Step 1: Running season testing system...")
            season_results = await self.tester.run_comprehensive_season_test(season)
            self.results['test_results']['season_testing'] = season_results

            # Step 2: Run prediction test suite
            logger.info("Step 2: Running prediction test suite...")
            test_suite_results = await self._run_test_suite()
            self.results['test_results']['prediction_tests'] = test_suite_results

            # Step 3: Run performance benchmarks
            logger.info("Step 3: Running performance benchmarks...")
            benchmark_results = await self._run_performance_benchmarks()
            self.results['performance_metrics'] = benchmark_results

            # Step 4: Generate comprehensive summary
            logger.info("Step 4: Generating comprehensive summary...")
            self.results['summary'] = self._generate_comprehensive_summary()

            # Step 5: Save results
            logger.info("Step 5: Saving results...")
            self._save_comprehensive_results()

            end_time = time.time()
            self.results['test_end_time'] = datetime.now().isoformat()
            self.results['total_duration'] = end_time - start_time

            logger.info(f"✅ Comprehensive testing completed in {self.results['total_duration']:.2f} seconds")
            return self.results

        except Exception as e:
            logger.error(f"❌ Comprehensive testing failed: {e}")
            self.results['error'] = str(e)
            return self.results

    async def _run_test_suite(self) -> dict:
        """Run the prediction test suite."""
        try:
            # Run pytest tests
            import subprocess

            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "mlb_prediction_test_suite.py",
                "-v",
                "--tb=short",
                "--asyncio-mode=auto"
            ], capture_output=True, text=True)

            # Parse test results
            test_results = {
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }

            # Parse test output to extract counts
            stdout_lines = result.stdout.split('\n')
            total_tests = 0
            passed_tests = 0
            failed_tests = 0
            skipped_tests = 0

            for line in stdout_lines:
                if 'passed' in line and 'failed' in line and 'skipped' in line:
                    # Extract numbers from summary line
                    import re
                    numbers = re.findall(r'\d+', line)
                    if len(numbers) >= 3:
                        passed_tests = int(numbers[0])
                        failed_tests = int(numbers[1])
                        skipped_tests = int(numbers[2])
                        total_tests = passed_tests + failed_tests + skipped_tests
                    break

            self.results['total_tests'] = total_tests
            self.results['passed_tests'] = passed_tests
            self.results['failed_tests'] = failed_tests
            self.results['skipped_tests'] = skipped_tests

            return test_results

        except Exception as e:
            logger.error(f"Error running test suite: {e}")
            return {'error': str(e), 'success': False}

    async def _run_performance_benchmarks(self) -> dict:
        """Run performance benchmarks."""
        logger.info("Running performance benchmarks...")

        benchmarks = {
            'data_fetching': {},
            'feature_engineering': {},
            'prediction_speed': {},
            'model_training': {}
        }

        try:
            # Benchmark data fetching
            start_time = time.time()
            season_data = await self.tester.fetch_mlb_season_data(2024)
            fetch_time = time.time() - start_time

            benchmarks['data_fetching'] = {
                'duration': fetch_time,
                'records_fetched': len(season_data),
                'records_per_second': len(season_data) / fetch_time if fetch_time > 0 else 0
            }

            # Benchmark feature engineering
            if not season_data.empty:
                start_time = time.time()
                engineered_data = self.tester.engineer_test_features(season_data, pd.DataFrame())
                engineering_time = time.time() - start_time

                benchmarks['feature_engineering'] = {
                    'duration': engineering_time,
                    'original_features': len(season_data.columns),
                    'engineered_features': len(engineered_data.columns),
                    'features_per_second': len(engineered_data.columns) / engineering_time if engineering_time > 0 else 0
                }

                # Benchmark prediction speed
                if len(engineered_data) > 0:
                    game_features = self.tester._prepare_game_features(engineered_data.iloc[0])
                    if game_features is not None:
                        start_time = time.time()
                        prediction = await self.tester.predictor.predict_mlb_outcome(game_features)
                        prediction_time = time.time() - start_time

                        benchmarks['prediction_speed'] = {
                            'duration': prediction_time,
                            'prediction_success': prediction is not None
                        }

            logger.info("✅ Performance benchmarks completed")
            return benchmarks

        except Exception as e:
            logger.error(f"Error running performance benchmarks: {e}")
            return {'error': str(e)}

    def _generate_comprehensive_summary(self) -> dict:
        """Generate comprehensive test summary."""
        summary = {
            'overall_status': 'unknown',
            'test_coverage': 0.0,
            'performance_score': 0.0,
            'data_quality_score': 0.0,
            'recommendations': []
        }

        try:
            # Calculate test coverage
            total_tests = self.results['total_tests']
            passed_tests = self.results['passed_tests']

            if total_tests > 0:
                summary['test_coverage'] = passed_tests / total_tests

            # Get data quality score
            season_results = self.results['test_results'].get('season_testing', {})
            data_quality = season_results.get('data_quality', {})
            summary['data_quality_score'] = data_quality.get('overall_score', 0.0)

            # Calculate performance score
            performance_metrics = self.results['performance_metrics']
            if performance_metrics and 'error' not in performance_metrics:
                # Score based on various performance metrics
                scores = []

                # Data fetching performance
                data_fetching = performance_metrics.get('data_fetching', {})
                if data_fetching.get('records_per_second', 0) > 10:
                    scores.append(1.0)
                elif data_fetching.get('records_per_second', 0) > 5:
                    scores.append(0.7)
                else:
                    scores.append(0.3)

                # Feature engineering performance
                feature_eng = performance_metrics.get('feature_engineering', {})
                if feature_eng.get('duration', float('inf')) < 5:
                    scores.append(1.0)
                elif feature_eng.get('duration', float('inf')) < 10:
                    scores.append(0.7)
                else:
                    scores.append(0.3)

                # Prediction speed
                prediction_speed = performance_metrics.get('prediction_speed', {})
                if prediction_speed.get('duration', float('inf')) < 1:
                    scores.append(1.0)
                elif prediction_speed.get('duration', float('inf')) < 5:
                    scores.append(0.7)
                else:
                    scores.append(0.3)

                if scores:
                    summary['performance_score'] = sum(scores) / len(scores)

            # Determine overall status
            if summary['test_coverage'] >= 0.8 and summary['data_quality_score'] >= 0.7:
                summary['overall_status'] = 'excellent'
            elif summary['test_coverage'] >= 0.6 and summary['data_quality_score'] >= 0.5:
                summary['overall_status'] = 'good'
            elif summary['test_coverage'] >= 0.4:
                summary['overall_status'] = 'fair'
            else:
                summary['overall_status'] = 'poor'

            # Generate recommendations
            recommendations = []

            if summary['test_coverage'] < 0.8:
                recommendations.append("Increase test coverage to at least 80%")

            if summary['data_quality_score'] < 0.7:
                recommendations.append("Improve data quality validation and cleaning")

            if summary['performance_score'] < 0.7:
                recommendations.append("Optimize performance bottlenecks in data processing")

            if not recommendations:
                recommendations.append("System is performing well - continue monitoring")

            summary['recommendations'] = recommendations

            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return summary

    def _save_comprehensive_results(self):
        """Save comprehensive test results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_mlb_test_results_{timestamp}.json"
        filepath = self.tester.results_dir / filename

        try:
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj

            serializable_results = convert_numpy(self.results)

            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)

            logger.info(f"Comprehensive results saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving comprehensive results: {e}")

    def print_summary(self):
        """Print a formatted summary of test results."""
        print("\n" + "="*60)
        print("📊 MLB PREDICTION SYSTEM TEST SUMMARY")
        print("="*60)

        summary = self.results['summary']

        print(f"Overall Status: {summary['overall_status'].upper()}")
        print(f"Test Coverage: {summary['test_coverage']:.1%}")
        print(f"Data Quality Score: {summary['data_quality_score']:.3f}")
        print(f"Performance Score: {summary['performance_score']:.3f}")

        print("\nTest Results:")
        print(f"  Total Tests: {self.results['total_tests']}")
        print(f"  Passed: {self.results['passed_tests']}")
        print(f"  Failed: {self.results['failed_tests']}")
        print(f"  Skipped: {self.results['skipped_tests']}")

        if 'total_duration' in self.results:
            print(f"\nTotal Duration: {self.results['total_duration']:.2f} seconds")

        print("\nRecommendations:")
        for rec in summary.get('recommendations', []):
            print(f"  • {rec}")

        print("="*60)


async def main():
    """Main function to run the test runner."""
    parser = argparse.ArgumentParser(description='MLB Prediction System Test Runner')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--season', type=int, default=2024, help='MLB season to test')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
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

    print("🧪 MLB Prediction System Test Runner")
    print(f"Testing MLB {args.season} season")
    print(f"Configuration: {args.config}")
    print(f"Quick mode: {args.quick}")

    # Initialize and run tests
    runner = MLBTestRunner(args.config)

    try:
        if args.quick:
            # Run quick tests
            print("\n🏃 Running quick tests...")
            results = await runner._run_test_suite()
        else:
            # Run comprehensive tests
            results = await runner.run_comprehensive_tests(args.season)

        # Print summary
        runner.print_summary()

        # Exit with appropriate code
        if results.get('summary', {}).get('overall_status') in ['excellent', 'good']:
            print("\n✅ Tests completed successfully!")
            exit(0)
        else:
            print("\n⚠️ Tests completed with issues - check results for details")
            exit(1)

    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        exit(130)
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
