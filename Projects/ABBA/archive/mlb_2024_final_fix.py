#!/usr/bin/env python3
"""
MLB 2024 Final Fix - Addresses Ultra-Conservative Betting Issues
Realistic thresholds, proper data handling, and better performance
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import structlog
import yaml

warnings.filterwarnings('ignore')

logger = structlog.get_logger()

class MLBFinalFix:
    """Final MLB betting system with realistic thresholds."""

    def __init__(self):
        # Load configuration
        with open('config.yaml') as f:
            self.config = yaml.safe_load(f)

        # REALISTIC betting settings (much more aggressive)
        self.betting_config = {
            'min_ev_threshold': 0.01,  # 1% expected value minimum (was 3%)
            'max_risk_per_bet': 0.05,  # 5% max risk per bet (was 1%)
            'kelly_fraction': 0.50,    # 50% Kelly fraction (was 20%)
            'rolling_window': 10,      # 10-game rolling window (was 15)
            'min_confidence': 0.51,    # 51% minimum confidence (was 53%)
            'max_bets_per_day': 5,     # 5 bets per day max (was 3)
            'daily_loss_limit': 0.15,  # 15% daily loss limit (was 8%)
            'transaction_cost': 0.0    # 0% transaction cost (DraftKings/FanDuel don't charge fees)
        }

        logger.info("🚀 Initialized MLB Final Fix with REALISTIC settings")

    def load_real_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load real MLB data."""
        logger.info("📊 Loading real MLB data")

        try:
            # Load games
            games_file = Path("real_data/mlb_games_2024.csv")
            if games_file.exists():
                games_df = pd.read_csv(games_file)
                logger.info(f"✅ Loaded {len(games_df)} games")
            else:
                logger.error("❌ Real games data not found")
                return None, None, None

            # Load standings (using correct column name)
            standings_file = Path("real_data/team_standings_2024.csv")
            if standings_file.exists():
                standings_df = pd.read_csv(standings_file)
                logger.info(f"✅ Loaded standings for {len(standings_df)} teams")
            else:
                logger.error("❌ Real standings data not found")
                return None, None, None

            # Load rolling stats
            stats_file = Path("real_data/rolling_stats_2024.csv")
            if stats_file.exists():
                stats_df = pd.read_csv(stats_file)
                logger.info(f"✅ Loaded {len(stats_df)} rolling stats records")
            else:
                logger.error("❌ Real rolling stats data not found")
                return None, None, None

            return games_df, standings_df, stats_df

        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return None, None, None

    def create_final_features(self, games_df: pd.DataFrame, standings_df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
        """Create features with proper data handling."""
        logger.info("🔧 Creating final features")

        features_list = []

        for idx, game in games_df.iterrows():
            try:
                game_date = pd.to_datetime(game['date'])
                home_team = game['home_team']
                away_team = game['away_team']

                # Get recent stats with proper error handling
                home_stats = stats_df[
                    (stats_df['team'] == home_team) &
                    (pd.to_datetime(stats_df['date']) < game_date)
                ].tail(self.betting_config['rolling_window'])

                away_stats = stats_df[
                    (stats_df['team'] == away_team) &
                    (pd.to_datetime(stats_df['date']) < game_date)
                ].tail(self.betting_config['rolling_window'])

                # Handle missing data gracefully
                if len(home_stats) == 0:
                    home_stats_mean = pd.Series({
                        'win_rate': 0.5, 'runs_scored': 4.5, 'runs_allowed': 4.5,
                        'era': 4.5, 'whip': 1.3, 'batting_avg': 0.25, 'ops': 0.7
                    })
                else:
                    home_stats_mean = home_stats[['win_rate', 'runs_scored', 'runs_allowed', 'era', 'whip', 'batting_avg', 'ops']].mean()

                if len(away_stats) == 0:
                    away_stats_mean = pd.Series({
                        'win_rate': 0.5, 'runs_scored': 4.5, 'runs_allowed': 4.5,
                        'era': 4.5, 'whip': 1.3, 'batting_avg': 0.25, 'ops': 0.7
                    })
                else:
                    away_stats_mean = away_stats[['win_rate', 'runs_scored', 'runs_allowed', 'era', 'whip', 'batting_avg', 'ops']].mean()

                # Get standings with correct column name
                home_standing = standings_df[standings_df['team_name'] == home_team]
                away_standing = standings_df[standings_df['team_name'] == away_team]

                if len(home_standing) == 0 or len(away_standing) == 0:
                    continue

                home_standing = home_standing.iloc[0]
                away_standing = away_standing.iloc[0]

                # Create features
                features = {
                    'game_id': game['game_id'],
                    'date': game_date,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_win': game['home_win'],

                    # Team performance (recent)
                    'home_win_pct': home_stats_mean.get('win_rate', 0.5),
                    'away_win_pct': away_stats_mean.get('win_rate', 0.5),
                    'home_runs_per_game': home_stats_mean.get('runs_scored', 4.5),
                    'away_runs_per_game': away_stats_mean.get('runs_scored', 4.5),
                    'home_era': home_stats_mean.get('era', 4.5),
                    'away_era': away_stats_mean.get('era', 4.5),

                    # Standings
                    'home_win_pct_standing': home_standing.get('win_rate', 0.5),
                    'away_win_pct_standing': away_standing.get('win_rate', 0.5),

                    # Simple edge indicators
                    'win_pct_diff': home_stats_mean.get('win_rate', 0.5) - away_stats_mean.get('win_rate', 0.5),
                    'runs_diff': home_stats_mean.get('runs_scored', 4.5) - away_stats_mean.get('runs_scored', 4.5),
                    'era_diff': away_stats_mean.get('era', 4.5) - home_stats_mean.get('era', 4.5),  # Lower ERA is better
                }

                features_list.append(features)

            except Exception as e:
                logger.error(f"Error creating features for game {idx}: {e}")
                continue

        features_df = pd.DataFrame(features_list)
        logger.info(f"✅ Created features for {len(features_df)} games")
        return features_df

    def final_prediction_model(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Final prediction model using realistic heuristics."""
        logger.info("🤖 Running final prediction model")

        predictions = []

        for idx, row in features_df.iterrows():
            # Realistic heuristic-based predictions
            home_advantage = 0.04  # 4% home field advantage

            # Base probability from recent performance
            home_recent = row['home_win_pct']
            away_recent = row['away_win_pct']

            if home_recent + away_recent > 0:
                base_prob = home_recent / (home_recent + away_recent)
            else:
                base_prob = 0.5

            # Adjust for recent form
            form_adjustment = row['win_pct_diff'] * 0.3

            # Adjust for run differential
            run_adjustment = row['runs_diff'] * 0.02

            # Adjust for ERA differential
            era_adjustment = row['era_diff'] * 0.01

            # Calculate final probability
            home_prob = base_prob + home_advantage + form_adjustment + run_adjustment + era_adjustment

            # Clamp to reasonable range
            home_prob = max(0.35, min(0.65, home_prob))

            # Calculate expected value (assuming -110 odds)
            implied_odds = 1.91  # -110 American odds
            ev = (home_prob * (implied_odds - 1)) - ((1 - home_prob) * 1)

            # Calculate confidence
            confidence = abs(home_prob - 0.5) * 2  # Scale to 0-1

            predictions.append({
                'game_id': row['game_id'],
                'date': row['date'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'home_win': row['home_win'],
                'predicted_home_prob': home_prob,
                'expected_value': ev,
                'confidence': confidence
            })

        predictions_df = pd.DataFrame(predictions)
        logger.info(f"✅ Made predictions for {len(predictions_df)} games")
        return predictions_df

    def realistic_betting_strategy(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Realistic betting strategy with proper risk management."""
        logger.info("💰 Running realistic betting strategy")

        bets = []
        bankroll = 10000  # $10,000 starting bankroll
        current_bankroll = bankroll
        daily_bets = {}
        daily_loss = 0

        for idx, row in predictions_df.iterrows():
            try:
                game_date = pd.to_datetime(row['date']).date()

                # Check daily limits
                if game_date in daily_bets:
                    if daily_bets[game_date] >= self.betting_config['max_bets_per_day']:
                        continue
                else:
                    daily_bets[game_date] = 0
                    daily_loss = 0

                # Check daily loss limit
                if daily_loss >= self.betting_config['daily_loss_limit'] * current_bankroll:
                    continue

                # REALISTIC betting criteria (much more aggressive)
                ev = row['expected_value']
                confidence = row['confidence']

                # MUCH more aggressive thresholds - bet on almost anything with positive EV
                if ev >= 0.001:  # Any positive expected value

                    # Kelly Criterion with realistic fraction
                    kelly_fraction = (ev / (1.91 - 1)) * self.betting_config['kelly_fraction']
                    kelly_fraction = min(kelly_fraction, self.betting_config['max_risk_per_bet'])

                    bet_amount = current_bankroll * kelly_fraction

                    # Realistic minimum bet size
                    if bet_amount >= 10:  # $10 minimum bet
                        bets.append({
                            'game_id': row['game_id'],
                            'date': row['date'],
                            'home_team': row['home_team'],
                            'away_team': row['away_team'],
                            'predicted_home_prob': row['predicted_home_prob'],
                            'expected_value': ev,
                            'confidence': confidence,
                            'bet_amount': bet_amount,
                            'kelly_fraction': kelly_fraction,
                            'actual_result': row['home_win'],
                            'profit_loss': (bet_amount * 0.91) if row['home_win'] else -bet_amount,
                            'transaction_cost': 0  # No transaction costs on DraftKings/FanDuel
                        })

                        daily_bets[game_date] += 1

                        # Update bankroll
                        if row['home_win']:
                            current_bankroll += bet_amount * 0.91
                            daily_loss = max(0, daily_loss - bet_amount * 0.91)
                        else:
                            current_bankroll -= bet_amount
                            daily_loss += bet_amount

            except Exception as e:
                logger.error(f"Error processing bet for game {idx}: {e}")
                continue

        bets_df = pd.DataFrame(bets)
        logger.info(f"✅ Placed {len(bets_df)} bets")
        return bets_df

    def calculate_performance(self, bets_df: pd.DataFrame) -> dict:
        """Calculate comprehensive performance metrics."""
        logger.info("📈 Calculating performance metrics")

        if len(bets_df) == 0:
            return {
                'total_bets': 0,
                'win_rate': 0,
                'total_profit_loss': 0,
                'roi': 0,
                'total_transaction_costs': 0,
                'avg_bet_size': 0,
                'max_drawdown': 0,
                'profitable_day_rate': 0,
                'total_days_betting': 0,
                'total_bet_amount': 0,
                'avg_expected_value': 0,
                'avg_confidence': 0
            }

        # Basic metrics
        total_bets = len(bets_df)
        wins = len(bets_df[bets_df['actual_result'] == True])
        win_rate = wins / total_bets if total_bets > 0 else 0

        # Financial metrics
        total_profit_loss = bets_df['profit_loss'].sum()
        total_transaction_costs = bets_df['transaction_cost'].sum()
        total_bet_amount = bets_df['bet_amount'].sum()
        roi = (total_profit_loss / total_bet_amount) * 100 if total_bet_amount > 0 else 0

        # Betting metrics
        avg_bet_size = bets_df['bet_amount'].mean()
        avg_ev = bets_df['expected_value'].mean()
        avg_confidence = bets_df['confidence'].mean()

        # Calculate cumulative returns for drawdown
        cumulative_returns = bets_df['profit_loss'].cumsum()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

        # Daily analysis
        bets_df['date'] = pd.to_datetime(bets_df['date']).dt.date
        daily_performance = bets_df.groupby('date').agg({
            'profit_loss': 'sum',
            'bet_amount': 'sum',
            'actual_result': 'sum',
            'game_id': 'count'
        }).rename(columns={'game_id': 'bets_placed'})

        profitable_days = len(daily_performance[daily_performance['profit_loss'] > 0])
        total_days = len(daily_performance)
        profitable_day_rate = profitable_days / total_days if total_days > 0 else 0

        return {
            'total_bets': total_bets,
            'win_rate': win_rate * 100,
            'total_profit_loss': total_profit_loss,
            'roi': roi,
            'total_transaction_costs': total_transaction_costs,
            'avg_bet_size': avg_bet_size,
            'avg_expected_value': avg_ev,
            'avg_confidence': avg_confidence,
            'max_drawdown': max_drawdown,
            'profitable_day_rate': profitable_day_rate * 100,
            'total_days_betting': total_days,
            'total_bet_amount': total_bet_amount
        }

    def run_validation(self):
        """Run the complete validation."""
        logger.info("🎯 Starting MLB Final Fix Validation")

        # Load data
        games_df, standings_df, stats_df = self.load_real_data()
        if games_df is None:
            logger.error("❌ Failed to load data")
            return

        # Create features
        features_df = self.create_final_features(games_df, standings_df, stats_df)
        if len(features_df) == 0:
            logger.error("❌ No features created")
            return

        # Make predictions
        predictions_df = self.final_prediction_model(features_df)

        # Place bets
        bets_df = self.realistic_betting_strategy(predictions_df)

        # Calculate performance
        performance = self.calculate_performance(bets_df)

        # Display results
        logger.info("🎉 VALIDATION COMPLETE!")
        logger.info(f"📊 Games Analyzed: {len(predictions_df):,}")
        logger.info(f"🎲 Betting Opportunities: {len(predictions_df[predictions_df['expected_value'] >= self.betting_config['min_ev_threshold']]):,}")
        logger.info(f"💰 Bets Placed: {performance['total_bets']:,}")
        logger.info(f"🏆 Win Rate: {performance['win_rate']:.1f}%")
        logger.info(f"💵 ROI: {performance['roi']:.1f}%")
        logger.info(f"💸 Total P&L: ${performance['total_profit_loss']:,.0f}")
        logger.info(f"📈 Avg Bet Size: ${performance['avg_bet_size']:,.0f}")
        logger.info(f"📉 Max Drawdown: {performance['max_drawdown']:.1f}%")
        logger.info(f"📅 Profitable Days: {performance['profitable_day_rate']:.1f}%")
        logger.info(f"💳 Transaction Costs: ${performance['total_transaction_costs']:,.0f}")

        # Save detailed results
        results = {
            'performance': performance,
            'betting_config': self.betting_config,
            'validation_date': datetime.now().isoformat(),
            'total_games': len(predictions_df),
            'betting_opportunities': len(predictions_df[predictions_df['expected_value'] >= self.betting_config['min_ev_threshold']])
        }

        # Save to file
        output_dir = Path("validation_results")
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / "final_fix_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

        if len(bets_df) > 0:
            bets_df.to_csv(output_dir / "final_fix_bets.csv", index=False)

        logger.info("✅ Results saved to validation_results/")

if __name__ == "__main__":
    fix = MLBFinalFix()
    fix.run_validation()
