"""
Integrated NHL Pipeline Demonstration
Shows the complete pipeline in action with multiple games and detailed analysis.
"""

import asyncio
import json

from Integrated_NHL_Pipeline import IntegratedNHLPipeline


async def demonstrate_pipeline():
    """Demonstrate the complete integrated NHL pipeline."""

    print("🏒 INTEGRATED NHL PIPELINE DEMONSTRATION")
    print("=" * 60)

    # Initialize pipeline with $100,000 bankroll
    pipeline = IntegratedNHLPipeline(bankroll=100000)

    # Sample games with different scenarios
    games = [
        {
            'game_id': 'BOS_TOR_2024_01_15',
            'home_team': 'Bruins',
            'away_team': 'Maple Leafs',
            'odds': {'moneyline': {'home': -140, 'away': +120}},
            'scenario': 'Elite goalie vs weak goalie'
        },
        {
            'game_id': 'EDM_CGY_2024_01_16',
            'home_team': 'Oilers',
            'away_team': 'Flames',
            'odds': {'moneyline': {'home': -110, 'away': -110}},
            'scenario': 'Even matchup'
        },
        {
            'game_id': 'COL_ARI_2024_01_17',
            'home_team': 'Avalanche',
            'away_team': 'Coyotes',
            'odds': {'moneyline': {'home': -200, 'away': +170}},
            'scenario': 'Heavy favorite'
        },
        {
            'game_id': 'VGK_LAK_2024_01_18',
            'home_team': 'Golden Knights',
            'away_team': 'Kings',
            'odds': {'moneyline': {'home': +130, 'away': -150}},
            'scenario': 'Road favorite'
        }
    ]

    results = []

    for game in games:
        print(f"\n🎯 Processing Game: {game['home_team']} vs {game['away_team']}")
        print(f"Scenario: {game['scenario']}")
        print(f"Odds: {game['odds']['moneyline']}")

        # Process game through pipeline
        result = await pipeline.process_game(game['game_id'], game['odds'])
        results.append(result)

        # Display results
        print("📊 Results:")
        print(f"   Prediction: {result['prediction']['prediction']:.1%}")
        print(f"   Confidence: {result['prediction']['confidence']:.1%}")
        print(f"   Model Agreement: {result['prediction']['model_agreement']:.1%}")
        print(f"   Edge: {result['risk_assessment']['edge_analysis'].get('adjusted_edge', 0):.1%}")
        print(f"   Edge Quality: {result['risk_assessment']['edge_analysis'].get('edge_quality', 'low')}")
        print(f"   Recommendation: {result['recommendation']['action'].upper()}")
        print(f"   Reason: {result['recommendation']['reason']}")
        print(f"   Priority: {result['recommendation']['priority']}")

        if result['recommendation']['action'] == 'bet':
            stake = result['risk_assessment']['stake']
            stake_amount = result['risk_assessment']['stake_amount']
            print(f"   💰 Stake: {stake:.1%} (${stake_amount:,.0f})")
            print(f"   Expected Value: ${result['risk_assessment']['risk_metrics']['expected_value']:,.0f}")
        else:
            print("   ❌ No bet recommended")

        print("-" * 40)

    # Performance Analysis
    print("\n📈 PIPELINE PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Calculate summary statistics
    total_games = len(results)
    bet_recommendations = [r for r in results if r['recommendation']['action'] == 'bet']
    pass_recommendations = [r for r in results if r['recommendation']['action'] == 'pass']

    print(f"Total Games Analyzed: {total_games}")
    print(f"Bet Recommendations: {len(bet_recommendations)} ({len(bet_recommendations)/total_games:.1%})")
    print(f"Pass Recommendations: {len(pass_recommendations)} ({len(pass_recommendations)/total_games:.1%})")

    if bet_recommendations:
        avg_edge = sum(r['risk_assessment']['edge_analysis'].get('adjusted_edge', 0) for r in bet_recommendations) / len(bet_recommendations)
        avg_confidence = sum(r['prediction']['confidence'] for r in bet_recommendations) / len(bet_recommendations)
        avg_stake = sum(r['risk_assessment']['stake'] for r in bet_recommendations) / len(bet_recommendations)
        total_stake_amount = sum(r['risk_assessment']['stake_amount'] for r in bet_recommendations)

        print("\n💰 BETTING ANALYSIS:")
        print(f"Average Edge: {avg_edge:.1%}")
        print(f"Average Confidence: {avg_confidence:.1%}")
        print(f"Average Stake: {avg_stake:.1%}")
        print(f"Total Stake Amount: ${total_stake_amount:,.0f}")
        print(f"Portfolio Exposure: {total_stake_amount/pipeline.risk_pipeline.bankroll:.1%}")

    # Feature Analysis
    print("\n🔍 FEATURE ANALYSIS")
    print("=" * 60)

    # Show feature importance for first game
    if results:
        first_game = results[0]
        features = first_game['features']

        print(f"Key Features for {first_game['game_data'].home_team} vs {first_game['game_data'].away_team}:")

        # Sort features by absolute value
        sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)

        for feature_name, value in sorted_features[:10]:
            print(f"   {feature_name}: {value:.3f}")

    # Risk Management Analysis
    print("\n🛡️ RISK MANAGEMENT ANALYSIS")
    print("=" * 60)

    print(f"Bankroll: ${pipeline.risk_pipeline.bankroll:,.0f}")
    print(f"Max Single Bet: {pipeline.risk_pipeline.risk_manager.max_single_bet:.1%}")
    print(f"Max Daily Risk: {pipeline.risk_pipeline.risk_manager.max_daily_risk:.1%}")
    print(f"Max Weekly Risk: {pipeline.risk_pipeline.risk_manager.max_weekly_risk:.1%}")
    print(f"Fractional Kelly: {pipeline.risk_pipeline.risk_manager.fractional_kelly:.1%}")

    # Model Performance
    print("\n🤖 MODEL PERFORMANCE")
    print("=" * 60)

    if results:
        first_result = results[0]
        base_predictions = first_result['prediction']['base_predictions']
        model_weights = first_result['prediction']['model_weights']

        print("Model Predictions:")
        for model_name, prediction in base_predictions.items():
            weight = model_weights.get(model_name, 0)
            print(f"   {model_name}: {prediction:.1%} (weight: {weight:.1%})")

        print(f"\nEnsemble Prediction: {first_result['prediction']['prediction']:.1%}")
        print(f"Model Agreement: {first_result['prediction']['model_agreement']:.1%}")

    # Pipeline Performance Summary
    print("\n📊 PIPELINE PERFORMANCE SUMMARY")
    print("=" * 60)

    summary = pipeline.get_performance_summary()
    print(f"Total Predictions: {summary['total_predictions']}")
    print(f"Average Confidence: {summary['avg_confidence']:.1%}")
    print(f"Average Edge: {summary['avg_edge']:.1%}")
    print(f"Bet Rate: {summary['bet_rate']:.1%}")

    # Save results to file
    save_results(results, 'pipeline_demo_results.json')

    print("\n✅ DEMONSTRATION COMPLETE")
    print("Results saved to 'pipeline_demo_results.json'")

def save_results(results, filename):
    """Save results to JSON file."""

    # Convert results to serializable format
    serializable_results = []

    for result in results:
        serializable_result = {
            'game_id': result['game_id'],
            'prediction': {
                'prediction': result['prediction']['prediction'],
                'confidence': result['prediction']['confidence'],
                'uncertainty': result['prediction']['uncertainty'],
                'model_agreement': result['prediction']['model_agreement']
            },
            'risk_assessment': {
                'stake': result['risk_assessment']['stake'],
                'stake_amount': result['risk_assessment']['stake_amount'],
                'edge_analysis': result['risk_assessment']['edge_analysis'],
                'risk_metrics': result['risk_assessment']['risk_metrics']
            },
            'recommendation': result['recommendation'],
            'timestamp': result['timestamp']
        }

        serializable_results.append(serializable_result)

    # Save to file
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"Results saved to {filename}")

async def demonstrate_real_time_processing():
    """Demonstrate real-time processing capabilities."""

    print("\n⚡ REAL-TIME PROCESSING DEMONSTRATION")
    print("=" * 60)

    pipeline = IntegratedNHLPipeline(bankroll=100000)

    # Simulate real-time game updates
    games = [
        {'game_id': 'real_time_1', 'odds': {'moneyline': {'home': -150, 'away': +130}}},
        {'game_id': 'real_time_2', 'odds': {'moneyline': {'home': -120, 'away': +100}}},
        {'game_id': 'real_time_3', 'odds': {'moneyline': {'home': -180, 'away': +160}}}
    ]

    print("Processing games in real-time...")

    for i, game in enumerate(games, 1):
        print(f"\n🔄 Processing Game {i}...")

        # Simulate processing time
        await asyncio.sleep(0.5)

        result = await pipeline.process_game(game['game_id'], game['odds'])

        print(f"✅ Game {i} processed in real-time")
        print(f"   Recommendation: {result['recommendation']['action']}")
        print("   Processing Time: ~0.5 seconds")

    print("\n🎯 Real-time processing demonstration complete!")

if __name__ == "__main__":
    # Run main demonstration
    asyncio.run(demonstrate_pipeline())

    # Run real-time demonstration
    asyncio.run(demonstrate_real_time_processing())
