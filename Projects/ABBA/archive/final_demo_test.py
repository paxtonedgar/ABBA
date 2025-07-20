#!/usr/bin/env python3
"""
Final Working Demo Test for ABMBA System
Complete end-to-end pipeline demonstration with proper error handling
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import structlog

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import Event, EventStatus, MarketType, Odds, PlatformType, SportType

# Configure logging
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

logger = structlog.get_logger()


class KellyCriterion:
    """Simple Kelly Criterion calculator."""

    def calculate_kelly_stake(self, win_prob: float, odds_decimal: float, fractional_kelly: float = 0.5) -> float:
        """Calculate Kelly stake with fractional Kelly."""
        if win_prob <= 0 or win_prob >= 1:
            return 0.0

        # Standard Kelly formula
        kelly = (win_prob * odds_decimal - 1) / (odds_decimal - 1)

        # Apply fractional Kelly
        kelly = kelly * fractional_kelly

        # Ensure non-negative
        return max(0.0, kelly)


class FinalDemoTest:
    """Final working demo test for ABMBA system."""

    def __init__(self):
        self.results = {
            'start_time': datetime.utcnow(),
            'phases': {},
            'final_recommendations': [],
            'errors': [],
            'warnings': []
        }

        # Initialize components
        self.kelly_calculator = KellyCriterion()

    def generate_mock_events(self) -> list[Event]:
        """Generate mock events for demo."""
        events = []

        # MLB Events
        mlb_teams = [
            ("New York Yankees", "Boston Red Sox"),
            ("Los Angeles Dodgers", "San Francisco Giants"),
            ("Chicago Cubs", "St. Louis Cardinals"),
            ("Houston Astros", "Texas Rangers"),
            ("Atlanta Braves", "Philadelphia Phillies")
        ]

        for i, (home, away) in enumerate(mlb_teams):
            event = Event(
                id=f"mlb_event_{i+1}",
                sport=SportType.BASEBALL_MLB,
                home_team=home,
                away_team=away,
                event_date=datetime.utcnow() + timedelta(hours=i+1),
                status=EventStatus.SCHEDULED
            )
            events.append(event)

        # NHL Events
        nhl_teams = [
            ("Toronto Maple Leafs", "Montreal Canadiens"),
            ("Boston Bruins", "New York Rangers"),
            ("Chicago Blackhawks", "Detroit Red Wings"),
            ("Edmonton Oilers", "Calgary Flames"),
            ("Vancouver Canucks", "Seattle Kraken")
        ]

        for i, (home, away) in enumerate(nhl_teams):
            event = Event(
                id=f"nhl_event_{i+1}",
                sport=SportType.HOCKEY_NHL,
                home_team=home,
                away_team=away,
                event_date=datetime.utcnow() + timedelta(hours=i+1),
                status=EventStatus.SCHEDULED
            )
            events.append(event)

        return events

    def generate_mock_odds(self, events: list[Event]) -> list[Odds]:
        """Generate mock odds for events."""
        odds = []

        for event in events:
            # Generate realistic odds
            if event.sport == SportType.BASEBALL_MLB:
                # MLB moneyline odds (typically -150 to +150)
                home_odds = np.random.choice([-140, -120, -110, -105, +105, +110, +120, +140])
                away_odds = -home_odds if home_odds > 0 else abs(home_odds)

                # Home team odds
                odds.append(Odds(
                    id=f"odds_{event.id}_home",
                    event_id=event.id,
                    platform=PlatformType.FANDUEL,
                    market_type=MarketType.MONEYLINE,
                    selection=event.home_team,
                    odds=Decimal(str(home_odds)),
                    timestamp=datetime.utcnow()
                ))

                # Away team odds
                odds.append(Odds(
                    id=f"odds_{event.id}_away",
                    event_id=event.id,
                    platform=PlatformType.DRAFTKINGS,
                    market_type=MarketType.MONEYLINE,
                    selection=event.away_team,
                    odds=Decimal(str(away_odds)),
                    timestamp=datetime.utcnow()
                ))

            elif event.sport == SportType.HOCKEY_NHL:
                # NHL moneyline odds (typically -200 to +200)
                home_odds = np.random.choice([-180, -150, -130, -110, +110, +130, +150, +180])
                away_odds = -home_odds if home_odds > 0 else abs(home_odds)

                # Home team odds
                odds.append(Odds(
                    id=f"odds_{event.id}_home",
                    event_id=event.id,
                    platform=PlatformType.FANDUEL,
                    market_type=MarketType.MONEYLINE,
                    selection=event.home_team,
                    odds=Decimal(str(home_odds)),
                    timestamp=datetime.utcnow()
                ))

                # Away team odds
                odds.append(Odds(
                    id=f"odds_{event.id}_away",
                    event_id=event.id,
                    platform=PlatformType.DRAFTKINGS,
                    market_type=MarketType.MONEYLINE,
                    selection=event.away_team,
                    odds=Decimal(str(away_odds)),
                    timestamp=datetime.utcnow()
                ))

        return odds

    async def phase_1_generate_mock_data(self) -> dict[str, Any]:
        """Phase 1: Generate mock events and odds."""
        logger.info("📊 Phase 1: Generating Mock Data")

        phase_results = {
            'start_time': datetime.utcnow(),
            'events_generated': 0,
            'odds_generated': 0,
            'sports_covered': [],
            'errors': []
        }

        try:
            # Generate mock events
            events = self.generate_mock_events()
            phase_results['events_generated'] = len(events)

            # Generate mock odds
            odds = self.generate_mock_odds(events)
            phase_results['odds_generated'] = len(odds)

            # Track sports covered
            sports = set(event.sport.value for event in events)
            phase_results['sports_covered'] = list(sports)

            # Store results
            phase_results['end_time'] = datetime.utcnow()
            phase_results['duration'] = (phase_results['end_time'] - phase_results['start_time']).total_seconds()
            phase_results['events'] = events
            phase_results['odds'] = odds

            self.results['phases']['generate_mock_data'] = phase_results

            logger.info(f"✅ Phase 1 completed: {phase_results['events_generated']} events, {phase_results['odds_generated']} odds")

            return phase_results

        except Exception as e:
            logger.error(f"❌ Phase 1 failed: {e}")
            phase_results['errors'].append(f"Phase 1 error: {e}")
            self.results['phases']['generate_mock_data'] = phase_results
            return phase_results

    async def phase_2_statistical_analysis(self, events: list[Event]) -> dict[str, Any]:
        """Phase 2: Perform statistical analysis on mock data."""
        logger.info("📈 Phase 2: Statistical Analysis")

        phase_results = {
            'start_time': datetime.utcnow(),
            'events_analyzed': 0,
            'mlb_insights': {},
            'nhl_insights': {},
            'feature_importance': {},
            'errors': []
        }

        try:
            # Separate events by sport
            mlb_events = [e for e in events if e.sport == SportType.BASEBALL_MLB]
            nhl_events = [e for e in events if e.sport == SportType.HOCKEY_NHL]

            logger.info(f"⚾ MLB events to analyze: {len(mlb_events)}")
            logger.info(f"🏒 NHL events to analyze: {len(nhl_events)}")

            # Generate mock statistical insights
            if mlb_events:
                logger.info("🔬 Analyzing MLB statistics...")

                # Mock MLB insights
                phase_results['mlb_insights'] = {
                    'avg_velocity': 92.5,
                    'avg_exit_velocity': 88.2,
                    'barrel_percentage': 6.8,
                    'hard_hit_percentage': 35.2,
                    'avg_launch_angle': 12.4,
                    'strike_zone_accuracy': 0.67,
                    'pitch_quality_score': 0.72,
                    'velocity_consistency': 0.85
                }

                # Mock feature importance
                phase_results['feature_importance']['mlb'] = {
                    'pitching_velocity': 0.25,
                    'exit_velocity': 0.20,
                    'barrel_percentage': 0.18,
                    'strike_zone_accuracy': 0.15,
                    'launch_angle': 0.12,
                    'rest_days': 0.10
                }

                logger.info("✅ MLB analysis completed")

            if nhl_events:
                logger.info("🔬 Analyzing NHL statistics...")

                # Mock NHL insights
                phase_results['nhl_insights'] = {
                    'avg_shot_distance': 18.5,
                    'high_danger_percentage': 28.3,
                    'avg_save_percentage': 0.912,
                    'powerplay_percentage': 0.205,
                    'penalty_kill_percentage': 0.815,
                    'shot_quality_score': 0.68,
                    'scoring_chance_percentage': 0.32
                }

                # Mock feature importance
                phase_results['feature_importance']['nhl'] = {
                    'save_percentage': 0.30,
                    'shot_distance': 0.22,
                    'high_danger_percentage': 0.20,
                    'powerplay_percentage': 0.15,
                    'penalty_kill_percentage': 0.13
                }

                logger.info("✅ NHL analysis completed")

            phase_results['events_analyzed'] = len(events)

            # Store results
            phase_results['end_time'] = datetime.utcnow()
            phase_results['duration'] = (phase_results['end_time'] - phase_results['start_time']).total_seconds()

            self.results['phases']['statistical_analysis'] = phase_results

            logger.info(f"✅ Phase 2 completed: {phase_results['events_analyzed']} events analyzed")

            return phase_results

        except Exception as e:
            logger.error(f"❌ Phase 2 failed: {e}")
            phase_results['errors'].append(f"Phase 2 error: {e}")
            self.results['phases']['statistical_analysis'] = phase_results
            return phase_results

    async def phase_3_generate_predictions(self, events: list[Event]) -> dict[str, Any]:
        """Phase 3: Generate predictions for events."""
        logger.info("🔮 Phase 3: Generating Predictions")

        phase_results = {
            'start_time': datetime.utcnow(),
            'predictions_generated': 0,
            'predictions': [],
            'confidence_scores': [],
            'errors': []
        }

        try:
            for event in events:
                try:
                    logger.info(f"🎯 Generating prediction for {event.home_team} vs {event.away_team}")

                    # Generate realistic predictions based on sport
                    if event.sport == SportType.BASEBALL_MLB:
                        # MLB predictions (slightly favor home team)
                        home_win_prob = np.random.normal(0.52, 0.08)
                        home_win_prob = max(0.35, min(0.65, home_win_prob))  # Clamp between 35-65%

                        predicted_winner = event.home_team if home_win_prob > 0.5 else event.away_team
                        confidence_score = np.random.uniform(0.6, 0.85)

                    elif event.sport == SportType.HOCKEY_NHL:
                        # NHL predictions (more balanced)
                        home_win_prob = np.random.normal(0.51, 0.06)
                        home_win_prob = max(0.40, min(0.60, home_win_prob))  # Clamp between 40-60%

                        predicted_winner = event.home_team if home_win_prob > 0.5 else event.away_team
                        confidence_score = np.random.uniform(0.55, 0.80)

                    else:
                        continue

                    # Create prediction result
                    prediction_result = {
                        'event_id': event.id,
                        'home_team': event.home_team,
                        'away_team': event.away_team,
                        'sport': event.sport.value,
                        'predicted_winner': predicted_winner,
                        'win_probability': home_win_prob if predicted_winner == event.home_team else 1 - home_win_prob,
                        'confidence_score': confidence_score,
                        'model_used': 'ensemble',
                        'key_factors': [
                            'recent_form',
                            'head_to_head_history',
                            'home_away_performance',
                            'rest_days',
                            'weather_conditions'
                        ],
                        'timestamp': datetime.utcnow().isoformat()
                    }

                    phase_results['predictions'].append(prediction_result)
                    phase_results['confidence_scores'].append(prediction_result['confidence_score'])
                    phase_results['predictions_generated'] += 1

                    logger.info(f"📊 Prediction: {prediction_result['predicted_winner']} wins ({prediction_result['win_probability']:.3f} probability)")

                except Exception as e:
                    logger.warning(f"⚠️ Failed to generate prediction for event {event.id}: {e}")
                    phase_results['errors'].append(f"Prediction error for {event.id}: {e}")

            # Store results
            phase_results['end_time'] = datetime.utcnow()
            phase_results['duration'] = (phase_results['end_time'] - phase_results['start_time']).total_seconds()

            self.results['phases']['generate_predictions'] = phase_results

            logger.info(f"✅ Phase 3 completed: {phase_results['predictions_generated']} predictions generated")

            return phase_results

        except Exception as e:
            logger.error(f"❌ Phase 3 failed: {e}")
            phase_results['errors'].append(f"Phase 3 error: {e}")
            self.results['phases']['generate_predictions'] = phase_results
            return phase_results

    async def phase_4_bet_selection(self, predictions: list[dict], odds: list[Odds]) -> dict[str, Any]:
        """Phase 4: Select optimal bets based on predictions and odds."""
        logger.info("💰 Phase 4: Bet Selection")

        phase_results = {
            'start_time': datetime.utcnow(),
            'opportunities_analyzed': 0,
            'bets_selected': 0,
            'selected_bets': [],
            'kelly_calculations': [],
            'risk_metrics': {},
            'errors': []
        }

        try:
            # Match predictions with odds
            for prediction in predictions:
                try:
                    event_id = prediction['event_id']
                    matching_odds = [o for o in odds if o.event_id == event_id]

                    if not matching_odds:
                        continue

                    phase_results['opportunities_analyzed'] += 1

                    # Find best odds for this event
                    best_odds = max(matching_odds, key=lambda x: float(x.odds) if float(x.odds) > 0 else 0)

                    # Calculate expected value
                    win_prob = prediction['win_probability']
                    odds_decimal = float(best_odds.odds) / 100 + 1 if float(best_odds.odds) > 0 else 100 / abs(float(best_odds.odds)) + 1

                    expected_value = (win_prob * (odds_decimal - 1)) - ((1 - win_prob) * 1)

                    # Calculate Kelly stake
                    kelly_stake = self.kelly_calculator.calculate_kelly_stake(
                        win_prob, odds_decimal, fractional_kelly=0.5
                    )

                    # Risk assessment
                    risk_score = self._calculate_risk_score(prediction, best_odds)

                    # Determine if this is a good bet
                    is_good_bet = (
                        expected_value > 0.05 and  # 5% minimum EV
                        kelly_stake > 0.01 and     # 1% minimum stake
                        prediction['confidence_score'] > 0.6 and  # 60% minimum confidence
                        risk_score < 0.7           # Low risk
                    )

                    bet_opportunity = {
                        'event_id': event_id,
                        'home_team': prediction['home_team'],
                        'away_team': prediction['away_team'],
                        'sport': prediction['sport'],
                        'predicted_winner': prediction['predicted_winner'],
                        'win_probability': win_prob,
                        'odds': float(best_odds.odds),
                        'odds_decimal': odds_decimal,
                        'expected_value': expected_value,
                        'kelly_stake': kelly_stake,
                        'confidence_score': prediction['confidence_score'],
                        'risk_score': risk_score,
                        'is_good_bet': is_good_bet,
                        'platform': best_odds.platform.value,
                        'market_type': best_odds.market_type.value,
                        'timestamp': datetime.utcnow().isoformat()
                    }

                    if is_good_bet:
                        phase_results['selected_bets'].append(bet_opportunity)
                        phase_results['bets_selected'] += 1

                        logger.info(f"🎯 Selected bet: {bet_opportunity['home_team']} vs {bet_opportunity['away_team']}")
                        logger.info(f"   EV: {expected_value:.3f}, Kelly: {kelly_stake:.3f}, Risk: {risk_score:.3f}")

                    phase_results['kelly_calculations'].append({
                        'event_id': event_id,
                        'kelly_stake': kelly_stake,
                        'expected_value': expected_value
                    })

                except Exception as e:
                    logger.warning(f"⚠️ Failed to analyze opportunity for event {event_id}: {e}")
                    phase_results['errors'].append(f"Opportunity analysis error for {event_id}: {e}")

            # Calculate risk metrics
            if phase_results['kelly_calculations']:
                kelly_stakes = [k['kelly_stake'] for k in phase_results['kelly_calculations']]
                expected_values = [k['expected_value'] for k in phase_results['kelly_calculations']]

                phase_results['risk_metrics'] = {
                    'total_kelly_stake': sum(kelly_stakes),
                    'avg_kelly_stake': np.mean(kelly_stakes),
                    'max_kelly_stake': max(kelly_stakes),
                    'avg_expected_value': np.mean(expected_values),
                    'positive_ev_opportunities': sum(1 for ev in expected_values if ev > 0),
                    'total_opportunities': len(expected_values)
                }

            # Store results
            phase_results['end_time'] = datetime.utcnow()
            phase_results['duration'] = (phase_results['end_time'] - phase_results['start_time']).total_seconds()

            self.results['phases']['bet_selection'] = phase_results

            logger.info(f"✅ Phase 4 completed: {phase_results['bets_selected']} bets selected from {phase_results['opportunities_analyzed']} opportunities")

            return phase_results

        except Exception as e:
            logger.error(f"❌ Phase 4 failed: {e}")
            phase_results['errors'].append(f"Phase 4 error: {e}")
            self.results['phases']['bet_selection'] = phase_results
            return phase_results

    def _calculate_risk_score(self, prediction: dict, odds: Odds) -> float:
        """Calculate risk score for a betting opportunity."""
        # Base risk factors
        confidence_risk = 1 - prediction['confidence_score']
        odds_risk = 1 / (1 + abs(float(odds.odds)) / 100)  # Higher odds = higher risk

        # Sport-specific risk
        sport_risk = 0.1 if prediction['sport'] == 'baseball_mlb' else 0.15  # NHL slightly riskier

        # Market type risk
        market_risk = {
            'moneyline': 0.1,
            'spread': 0.2,
            'totals': 0.15
        }.get(odds.market_type.value, 0.2)

        # Combine risk factors
        total_risk = (confidence_risk * 0.4 +
                     odds_risk * 0.3 +
                     sport_risk * 0.2 +
                     market_risk * 0.1)

        return min(total_risk, 1.0)

    async def generate_final_report(self) -> dict[str, Any]:
        """Generate comprehensive final report."""
        logger.info("📋 Generating Final Report")

        # Calculate overall metrics
        total_duration = (datetime.utcnow() - self.results['start_time']).total_seconds()

        # Extract key metrics from phases
        events_generated = self.results['phases'].get('generate_mock_data', {}).get('events_generated', 0)
        predictions_generated = self.results['phases'].get('generate_predictions', {}).get('predictions_generated', 0)
        bets_selected = self.results['phases'].get('bet_selection', {}).get('bets_selected', 0)

        # Calculate success rates
        prediction_success_rate = predictions_generated / events_generated if events_generated > 0 else 0
        bet_selection_rate = bets_selected / predictions_generated if predictions_generated > 0 else 0

        # Generate recommendations
        recommendations = []

        if bets_selected > 0:
            selected_bets = self.results['phases'].get('bet_selection', {}).get('selected_bets', [])

            # Sort by expected value
            selected_bets.sort(key=lambda x: x['expected_value'], reverse=True)

            for i, bet in enumerate(selected_bets[:3]):  # Top 3 recommendations
                recommendations.append({
                    'rank': i + 1,
                    'sport': bet['sport'],
                    'match': f"{bet['home_team']} vs {bet['away_team']}",
                    'prediction': bet['predicted_winner'],
                    'odds': bet['odds'],
                    'expected_value': f"{bet['expected_value']:.3f}",
                    'kelly_stake': f"{bet['kelly_stake']:.3f}",
                    'confidence': f"{bet['confidence_score']:.3f}",
                    'platform': bet['platform'],
                    'reasoning': f"High EV ({bet['expected_value']:.3f}) with good confidence ({bet['confidence_score']:.3f})"
                })

        # Generate insights
        insights = []

        # Data insights
        insights.append("✅ Mock data generated successfully for demonstration")

        # Model performance insights
        avg_confidence = np.mean(self.results['phases'].get('generate_predictions', {}).get('confidence_scores', [0]))
        if avg_confidence > 0.7:
            insights.append("🎯 High model confidence - strong predictions")
        elif avg_confidence > 0.5:
            insights.append("📊 Moderate model confidence - mixed predictions")
        else:
            insights.append("🤔 Low model confidence - consider manual review")

        # Risk insights
        if bets_selected > 0:
            risk_metrics = self.results['phases'].get('bet_selection', {}).get('risk_metrics', {})
            total_kelly = risk_metrics.get('total_kelly_stake', 0)
            if total_kelly > 0.1:
                insights.append("⚠️ High total exposure - consider reducing stakes")
            elif total_kelly > 0.05:
                insights.append("📈 Moderate exposure - reasonable risk level")
            else:
                insights.append("🛡️ Low exposure - conservative approach")

        # Add sport-specific insights
        mlb_insights = self.results['phases'].get('statistical_analysis', {}).get('mlb_insights', {})
        nhl_insights = self.results['phases'].get('statistical_analysis', {}).get('nhl_insights', {})

        if mlb_insights:
            insights.append(f"⚾ MLB: Avg velocity {mlb_insights.get('avg_velocity', 0):.1f} mph, Barrel rate {mlb_insights.get('barrel_percentage', 0):.1f}%")

        if nhl_insights:
            insights.append(f"🏒 NHL: Avg save % {nhl_insights.get('avg_save_percentage', 0):.3f}, PP % {nhl_insights.get('powerplay_percentage', 0):.1%}")

        final_report = {
            'demo_summary': {
                'total_duration_seconds': total_duration,
                'events_generated': events_generated,
                'predictions_generated': predictions_generated,
                'bets_selected': bets_selected,
                'prediction_success_rate': prediction_success_rate,
                'bet_selection_rate': bet_selection_rate,
                'total_errors': len(self.results['errors']),
                'total_warnings': len(self.results['warnings'])
            },
            'top_recommendations': recommendations,
            'key_insights': insights,
            'phase_performance': {
                phase: {
                    'duration': results.get('duration', 0),
                    'success': len(results.get('errors', [])) == 0
                }
                for phase, results in self.results['phases'].items()
            },
            'system_health': {
                'overall_status': 'healthy' if len(self.results['errors']) == 0 else 'degraded',
                'data_quality_score': 0.95,  # Mock data is high quality
                'model_confidence': avg_confidence,
                'risk_level': 'low' if bets_selected == 0 else 'moderate'
            }
        }

        self.results['final_report'] = final_report

        return final_report

    async def run_complete_demo(self) -> dict[str, Any]:
        """Run the complete final demo test."""
        logger.info("🎬 Starting ABMBA Final Demo Test")

        try:
            # Phase 1: Generate mock data
            phase1_results = await self.phase_1_generate_mock_data()
            if not phase1_results.get('events'):
                logger.error("❌ No events generated - cannot continue demo")
                return self.results

            # Phase 2: Statistical analysis
            phase2_results = await self.phase_2_statistical_analysis(phase1_results['events'])

            # Phase 3: Generate predictions
            phase3_results = await self.phase_3_generate_predictions(phase1_results['events'])

            # Phase 4: Bet selection
            phase4_results = await self.phase_4_bet_selection(
                phase3_results['predictions'],
                phase1_results['odds']
            )

            # Generate final report
            final_report = await self.generate_final_report()

            # Log completion
            self.results['end_time'] = datetime.utcnow()
            total_duration = (self.results['end_time'] - self.results['start_time']).total_seconds()

            logger.info(f"🎉 Final Demo Test completed in {total_duration:.2f} seconds")
            logger.info(f"📊 Summary: {final_report['demo_summary']['events_generated']} events, "
                       f"{final_report['demo_summary']['bets_selected']} bets selected")

            return self.results

        except Exception as e:
            logger.error(f"❌ Final demo test failed: {e}")
            self.results['errors'].append(f"Demo failure: {e}")
            return self.results


async def main():
    """Main function to run the final demo test."""
    print("🎬 ABMBA Final Demo Test")
    print("=" * 50)
    print("Running complete end-to-end pipeline demonstration...")
    print()

    # Create and run demo
    demo = FinalDemoTest()

    try:
        results = await demo.run_complete_demo()

        # Print results
        print("\n📋 DEMO RESULTS")
        print("=" * 50)

        # Summary
        summary = results.get('final_report', {}).get('demo_summary', {})
        print(f"⏱️  Total Duration: {summary.get('total_duration_seconds', 0):.2f} seconds")
        print(f"📊 Events Generated: {summary.get('events_generated', 0)}")
        print(f"🔮 Predictions Generated: {summary.get('predictions_generated', 0)}")
        print(f"💰 Bets Selected: {summary.get('bets_selected', 0)}")
        print(f"📈 Prediction Success Rate: {summary.get('prediction_success_rate', 0):.2%}")
        print(f"🎯 Bet Selection Rate: {summary.get('bet_selection_rate', 0):.2%}")

        # Top recommendations
        recommendations = results.get('final_report', {}).get('top_recommendations', [])
        if recommendations:
            print(f"\n🏆 TOP {len(recommendations)} RECOMMENDATIONS")
            print("=" * 50)
            for rec in recommendations:
                print(f"#{rec['rank']} {rec['sport'].upper()}: {rec['match']}")
                print(f"   Prediction: {rec['prediction']} (Odds: {rec['odds']})")
                print(f"   EV: {rec['expected_value']}, Kelly: {rec['kelly_stake']}, Confidence: {rec['confidence']}")
                print(f"   Platform: {rec['platform']}")
                print(f"   Reasoning: {rec['reasoning']}")
                print()
        else:
            print("\n⚠️  NO BETTING RECOMMENDATIONS")
            print("=" * 50)
            print("No bets met the conservative criteria (5% EV, 1% Kelly, 60% confidence, low risk)")
            print("This demonstrates the system's risk management approach.")
            print()

        # Key insights
        insights = results.get('final_report', {}).get('key_insights', [])
        if insights:
            print("💡 KEY INSIGHTS")
            print("=" * 50)
            for insight in insights:
                print(f"• {insight}")
            print()

        # System health
        health = results.get('final_report', {}).get('system_health', {})
        print("🏥 SYSTEM HEALTH")
        print("=" * 50)
        print(f"Status: {health.get('overall_status', 'unknown')}")
        print(f"Data Quality: {health.get('data_quality_score', 0):.3f}")
        print(f"Model Confidence: {health.get('model_confidence', 0):.3f}")
        print(f"Risk Level: {health.get('risk_level', 'unknown')}")

        # Phase performance
        phase_performance = results.get('final_report', {}).get('phase_performance', {})
        if phase_performance:
            print("\n⚙️  PHASE PERFORMANCE")
            print("=" * 50)
            for phase, perf in phase_performance.items():
                status = "✅" if perf['success'] else "❌"
                print(f"{status} {phase.replace('_', ' ').title()}: {perf['duration']:.2f}s")

        # Errors and warnings
        if results.get('errors'):
            print(f"\n❌ ERRORS ({len(results['errors'])}):")
            for error in results['errors']:
                print(f"• {error}")

        if results.get('warnings'):
            print(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
            for warning in results['warnings']:
                print(f"• {warning}")

        print("\n" + "=" * 50)
        print("🎉 Final Demo Test Complete!")
        print("\nThis demonstration shows the complete ABMBA pipeline:")
        print("1. 📊 Data Generation/Fetching")
        print("2. 📈 Statistical Analysis")
        print("3. 🔮 Prediction Generation")
        print("4. 💰 Bet Selection & Risk Management")
        print("\nThe system successfully processed MLB and NHL data with")
        print("realistic odds, generated predictions, and applied")
        print("conservative betting criteria for sustainable returns.")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
