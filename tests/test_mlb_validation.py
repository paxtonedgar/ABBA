#!/usr/bin/env python3
"""
Comprehensive MLB Validation Testing Framework
Tests: Drawdown protection, ML model, risk management, out-of-sample validation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import structlog
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
warnings.filterwarnings('ignore')

logger = structlog.get_logger()

class MLBValidationTester:
    """Comprehensive MLB validation testing framework."""
    
    def __init__(self):
        # Load configuration
        config_path = Path(__file__).parent.parent / 'config.yaml'
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # IMPROVED settings with aggressive drawdown protection
        self.betting_config = {
            'min_ev_threshold': 0.008,  # 0.8% expected value minimum
            'max_risk_per_bet': 0.015,  # 1.5% max risk per bet (reduced)
            'kelly_fraction': 0.20,     # 20% Kelly fraction (reduced)
            'rolling_window': 15,       # 15-game rolling window
            'min_confidence': 0.02,     # 2% minimum confidence
            'max_bets_per_day': 3,      # 3 bets per day max
            'daily_loss_limit': 0.05,   # 5% daily loss limit (reduced)
            'max_drawdown_limit': 0.15, # 15% max drawdown limit (aggressive)
            'transaction_cost': 0.0,    # No transaction costs
            'stop_loss_drawdown': 0.10, # Stop betting at 10% drawdown
            'recovery_threshold': 0.05  # Resume betting at 5% drawdown
        }
        
        # ML model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        # Test results storage
        self.test_results = {}
        
        logger.info("🚀 Initialized MLB Validation Tester")
    
    def generate_synthetic_data(self, num_games: int = 2500) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate synthetic MLB data for testing."""
        logger.info(f"📊 Generating synthetic data for {num_games} games")
        
        # Team names
        teams = [
            'New York Yankees', 'Boston Red Sox', 'Toronto Blue Jays', 'Baltimore Orioles', 'Tampa Bay Rays',
            'Cleveland Guardians', 'Minnesota Twins', 'Detroit Tigers', 'Chicago White Sox', 'Kansas City Royals',
            'Houston Astros', 'Texas Rangers', 'Seattle Mariners', 'Los Angeles Angels', 'Oakland Athletics',
            'Atlanta Braves', 'New York Mets', 'Philadelphia Phillies', 'Washington Nationals', 'Miami Marlins',
            'Milwaukee Brewers', 'Chicago Cubs', 'St. Louis Cardinals', 'Pittsburgh Pirates', 'Cincinnati Reds',
            'Los Angeles Dodgers', 'San Francisco Giants', 'San Diego Padres', 'Colorado Rockies', 'Arizona Diamondbacks'
        ]
        
        # Generate games
        games_data = []
        start_date = datetime(2024, 3, 28)
        
        for i in range(num_games):
            game_date = start_date + timedelta(days=i % 180)  # 6-month season
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            # Realistic win probability based on team strength
            home_strength = hash(home_team) % 100 / 100.0
            away_strength = hash(away_team) % 100 / 100.0
            home_advantage = 0.035
            
            home_win_prob = (home_strength + home_advantage) / (home_strength + away_strength + home_advantage)
            home_win = np.random.random() < home_win_prob
            
            games_data.append({
                'game_id': f'game_{i:04d}',
                'date': game_date.strftime('%Y-%m-%d'),
                'home_team': home_team,
                'away_team': away_team,
                'home_win': home_win
            })
        
        games_df = pd.DataFrame(games_data)
        
        # Generate standings
        standings_data = []
        for team in teams:
            team_strength = hash(team) % 100 / 100.0
            wins = int(162 * team_strength * 0.6)  # 60% of max wins
            losses = 162 - wins
            win_rate = wins / 162
            
            standings_data.append({
                'team_name': team,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate
            })
        
        standings_df = pd.DataFrame(standings_data)
        
        # Generate rolling stats
        stats_data = []
        for team in teams:
            team_strength = hash(team) % 100 / 100.0
            
            for i in range(50):  # 50 stat records per team
                game_date = start_date + timedelta(days=i * 3)
                
                # Realistic stats based on team strength
                win_rate = team_strength + np.random.normal(0, 0.1)
                win_rate = max(0.3, min(0.7, win_rate))
                
                runs_scored = 4.0 + team_strength * 2 + np.random.normal(0, 0.5)
                runs_allowed = 4.5 - team_strength * 1.5 + np.random.normal(0, 0.5)
                era = 4.5 - team_strength * 1.5 + np.random.normal(0, 0.3)
                whip = 1.4 - team_strength * 0.3 + np.random.normal(0, 0.1)
                batting_avg = 0.24 + team_strength * 0.06 + np.random.normal(0, 0.02)
                ops = 0.68 + team_strength * 0.12 + np.random.normal(0, 0.03)
                
                stats_data.append({
                    'team': team,
                    'date': game_date.strftime('%Y-%m-%d'),
                    'win_rate': win_rate,
                    'runs_scored': runs_scored,
                    'runs_allowed': runs_allowed,
                    'era': era,
                    'whip': whip,
                    'batting_avg': batting_avg,
                    'ops': ops
                })
        
        stats_df = pd.DataFrame(stats_data)
        
        logger.info(f"✅ Generated synthetic data: {len(games_df)} games, {len(standings_df)} teams, {len(stats_df)} stat records")
        return games_df, standings_df, stats_df
    
    def create_enhanced_features(self, games_df: pd.DataFrame, standings_df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced features with sophisticated engineering."""
        logger.info("🔧 Creating enhanced features")
        
        features_list = []
        
        for idx, game in games_df.iterrows():
            try:
                game_date = pd.to_datetime(game['date'])
                home_team = game['home_team']
                away_team = game['away_team']
                
                # Get recent stats with different windows
                home_stats_15 = stats_df[
                    (stats_df['team'] == home_team) & 
                    (pd.to_datetime(stats_df['date']) < game_date)
                ].tail(15)
                
                home_stats_7 = stats_df[
                    (stats_df['team'] == home_team) & 
                    (pd.to_datetime(stats_df['date']) < game_date)
                ].tail(7)
                
                away_stats_15 = stats_df[
                    (stats_df['team'] == away_team) & 
                    (pd.to_datetime(stats_df['date']) < game_date)
                ].tail(15)
                
                away_stats_7 = stats_df[
                    (stats_df['team'] == away_team) & 
                    (pd.to_datetime(stats_df['date']) < game_date)
                ].tail(7)
                
                # Handle missing data with better defaults
                if len(home_stats_15) == 0:
                    home_15_mean = pd.Series({
                        'win_rate': 0.5, 'runs_scored': 4.5, 'runs_allowed': 4.5, 
                        'era': 4.5, 'whip': 1.3, 'batting_avg': 0.25, 'ops': 0.7
                    })
                else:
                    home_15_mean = home_stats_15[['win_rate', 'runs_scored', 'runs_allowed', 'era', 'whip', 'batting_avg', 'ops']].mean()
                
                if len(home_stats_7) == 0:
                    home_7_mean = home_15_mean
                else:
                    home_7_mean = home_stats_7[['win_rate', 'runs_scored', 'runs_allowed', 'era', 'whip', 'batting_avg', 'ops']].mean()
                
                if len(away_stats_15) == 0:
                    away_15_mean = pd.Series({
                        'win_rate': 0.5, 'runs_scored': 4.5, 'runs_allowed': 4.5, 
                        'era': 4.5, 'whip': 1.3, 'batting_avg': 0.25, 'ops': 0.7
                    })
                else:
                    away_15_mean = away_stats_15[['win_rate', 'runs_scored', 'runs_allowed', 'era', 'whip', 'batting_avg', 'ops']].mean()
                
                if len(away_stats_7) == 0:
                    away_7_mean = away_15_mean
                else:
                    away_7_mean = away_stats_7[['win_rate', 'runs_scored', 'runs_allowed', 'era', 'whip', 'batting_avg', 'ops']].mean()
                
                # Get standings
                home_standing = standings_df[standings_df['team_name'] == home_team]
                away_standing = standings_df[standings_df['team_name'] == away_team]
                
                if len(home_standing) == 0 or len(away_standing) == 0:
                    continue
                
                home_standing = home_standing.iloc[0]
                away_standing = away_standing.iloc[0]
                
                # Enhanced features
                features = {
                    'game_id': game['game_id'],
                    'date': game_date,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_win': game['home_win'],
                    
                    # Recent performance (15-game window)
                    'home_win_pct_15': home_15_mean.get('win_rate', 0.5),
                    'away_win_pct_15': away_15_mean.get('win_rate', 0.5),
                    'home_runs_per_game_15': home_15_mean.get('runs_scored', 4.5),
                    'away_runs_per_game_15': away_15_mean.get('runs_scored', 4.5),
                    'home_era_15': home_15_mean.get('era', 4.5),
                    'away_era_15': away_15_mean.get('era', 4.5),
                    'home_whip_15': home_15_mean.get('whip', 1.3),
                    'away_whip_15': away_15_mean.get('whip', 1.3),
                    'home_batting_avg_15': home_15_mean.get('batting_avg', 0.25),
                    'away_batting_avg_15': away_15_mean.get('batting_avg', 0.25),
                    'home_ops_15': home_15_mean.get('ops', 0.7),
                    'away_ops_15': away_15_mean.get('ops', 0.7),
                    
                    # Recent form (7-game window)
                    'home_win_pct_7': home_7_mean.get('win_rate', 0.5),
                    'away_win_pct_7': away_7_mean.get('win_rate', 0.5),
                    'home_runs_per_game_7': home_7_mean.get('runs_scored', 4.5),
                    'away_runs_per_game_7': away_7_mean.get('runs_scored', 4.5),
                    'home_era_7': home_7_mean.get('era', 4.5),
                    'away_era_7': away_7_mean.get('era', 4.5),
                    
                    # Standings
                    'home_win_pct_standing': home_standing.get('win_rate', 0.5),
                    'away_win_pct_standing': away_standing.get('win_rate', 0.5),
                    
                    # Derived features
                    'win_pct_diff_15': home_15_mean.get('win_rate', 0.5) - away_15_mean.get('win_rate', 0.5),
                    'win_pct_diff_7': home_7_mean.get('win_rate', 0.5) - away_7_mean.get('win_rate', 0.5),
                    'runs_diff_15': home_15_mean.get('runs_scored', 4.5) - away_15_mean.get('runs_scored', 4.5),
                    'runs_diff_7': home_7_mean.get('runs_scored', 4.5) - away_7_mean.get('runs_scored', 4.5),
                    'era_diff_15': away_15_mean.get('era', 4.5) - home_15_mean.get('era', 4.5),
                    'era_diff_7': away_7_mean.get('era', 4.5) - home_7_mean.get('era', 4.5),
                    'whip_diff_15': away_15_mean.get('whip', 1.3) - home_15_mean.get('whip', 1.3),
                    'batting_avg_diff_15': home_15_mean.get('batting_avg', 0.25) - away_15_mean.get('batting_avg', 0.25),
                    'ops_diff_15': home_15_mean.get('ops', 0.7) - away_15_mean.get('ops', 0.7),
                    
                    # Form momentum
                    'home_form_momentum': home_7_mean.get('win_rate', 0.5) - home_15_mean.get('win_rate', 0.5),
                    'away_form_momentum': away_7_mean.get('win_rate', 0.5) - away_15_mean.get('win_rate', 0.5),
                    
                    # Season-long performance
                    'home_season_advantage': home_standing.get('win_rate', 0.5) - away_standing.get('win_rate', 0.5),
                }
                
                features_list.append(features)
                
            except Exception as e:
                logger.error(f"Error creating features for game {idx}: {e}")
                continue
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"✅ Created enhanced features for {len(features_df)} games")
        return features_df
    
    def train_ml_model(self, features_df: pd.DataFrame) -> None:
        """Train ML model on historical data."""
        logger.info("🤖 Training ML model")
        
        # Prepare features for ML
        feature_columns = [
            'home_win_pct_15', 'away_win_pct_15', 'home_runs_per_game_15', 'away_runs_per_game_15',
            'home_era_15', 'away_era_15', 'home_whip_15', 'away_whip_15', 'home_batting_avg_15', 'away_batting_avg_15',
            'home_ops_15', 'away_ops_15', 'home_win_pct_7', 'away_win_pct_7', 'home_runs_per_game_7', 'away_runs_per_game_7',
            'home_era_7', 'away_era_7', 'home_win_pct_standing', 'away_win_pct_standing',
            'win_pct_diff_15', 'win_pct_diff_7', 'runs_diff_15', 'runs_diff_7', 'era_diff_15', 'era_diff_7',
            'whip_diff_15', 'batting_avg_diff_15', 'ops_diff_15', 'home_form_momentum', 'away_form_momentum',
            'home_season_advantage'
        ]
        
        self.feature_columns = feature_columns
        
        # Prepare data
        X = features_df[feature_columns].fillna(0)
        y = features_df['home_win'].astype(int)
        
        # Split data (use first 70% for training)
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"✅ ML Model trained - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
        logger.info(f"🔝 Top 5 features: {feature_importance.head()['feature'].tolist()}")
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importance': feature_importance.to_dict('records')
        }
    
    def enhanced_prediction_model(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced prediction model combining ML and heuristics."""
        logger.info("🤖 Running enhanced prediction model")
        
        # Train ML model on first 70% of data
        split_idx = int(len(features_df) * 0.7)
        train_features = features_df[:split_idx]
        test_features = features_df[split_idx:]
        
        model_metrics = self.train_ml_model(train_features)
        
        predictions = []
        
        for idx, row in features_df.iterrows():
            try:
                # ML prediction
                if self.model is not None and idx >= split_idx:
                    X = row[self.feature_columns].fillna(0).values.reshape(1, -1)
                    X_scaled = self.scaler.transform(X)
                    ml_prob = self.model.predict_proba(X_scaled)[0][1]
                else:
                    ml_prob = 0.5  # Default for training period
                
                # Heuristic prediction
                home_advantage = 0.035  # 3.5% home field advantage
                
                # Base probability from recent performance
                home_recent = row['home_win_pct_15']
                away_recent = row['away_win_pct_15']
                
                if home_recent + away_recent > 0:
                    base_prob = home_recent / (home_recent + away_recent)
                else:
                    base_prob = 0.5
                
                # Enhanced adjustments
                form_adjustment = row['win_pct_diff_15'] * 0.25
                run_adjustment = row['runs_diff_15'] * 0.015
                era_adjustment = row['era_diff_15'] * 0.008
                momentum_adjustment = row['home_form_momentum'] * 0.3
                season_adjustment = row['home_season_advantage'] * 0.2
                
                # Calculate heuristic probability
                heuristic_prob = base_prob + home_advantage + form_adjustment + run_adjustment + era_adjustment + momentum_adjustment + season_adjustment
                heuristic_prob = max(0.35, min(0.65, heuristic_prob))
                
                # Combine ML and heuristic (weighted average)
                if idx >= split_idx:
                    home_prob = 0.6 * ml_prob + 0.4 * heuristic_prob
                else:
                    home_prob = heuristic_prob
                
                home_prob = max(0.35, min(0.65, home_prob))
                
                # Calculate expected value
                implied_odds = 1.91  # -110 American odds
                ev = (home_prob * (implied_odds - 1)) - ((1 - home_prob) * 1)
                
                # Calculate confidence
                confidence = abs(home_prob - 0.5) * 2
                
                predictions.append({
                    'game_id': row['game_id'],
                    'date': row['date'],
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'home_win': row['home_win'],
                    'predicted_home_prob': home_prob,
                    'ml_prob': ml_prob if idx >= split_idx else None,
                    'heuristic_prob': heuristic_prob,
                    'expected_value': ev,
                    'confidence': confidence,
                    'is_ml_prediction': idx >= split_idx
                })
                
            except Exception as e:
                logger.error(f"Error making prediction for game {idx}: {e}")
                continue
        
        predictions_df = pd.DataFrame(predictions)
        logger.info(f"✅ Made enhanced predictions for {len(predictions_df)} games")
        
        # Store model metrics
        self.test_results['model_metrics'] = model_metrics
        
        return predictions_df
    
    def aggressive_drawdown_protection_strategy(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Aggressive drawdown protection strategy."""
        logger.info("💰 Running aggressive drawdown protection strategy")
        
        bets = []
        bankroll = 10000  # $10,000 starting bankroll
        current_bankroll = bankroll
        initial_bankroll = bankroll
        daily_bets = {}
        daily_loss = 0
        max_drawdown = 0
        running_max = initial_bankroll
        betting_suspended = False
        suspension_start = None
        suspension_count = 0
        
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
                if daily_loss >= self.betting_config['daily_loss_limit'] * initial_bankroll:
                    continue
                
                # AGGRESSIVE drawdown protection
                if len(bets) > 0:
                    current_drawdown = (running_max - current_bankroll) / running_max
                    
                    # Suspend betting if drawdown exceeds stop loss
                    if current_drawdown >= self.betting_config['stop_loss_drawdown']:
                        if not betting_suspended:
                            betting_suspended = True
                            suspension_start = current_drawdown
                            suspension_count += 1
                            logger.info(f"🚫 Betting suspended at {current_drawdown:.1%} drawdown")
                        continue
                    
                    # Resume betting only if drawdown recovers significantly
                    if betting_suspended:
                        if current_drawdown <= self.betting_config['recovery_threshold']:
                            betting_suspended = False
                            logger.info(f"✅ Betting resumed at {current_drawdown:.1%} drawdown")
                        else:
                            continue
                    
                    # Reduce bet size based on drawdown level
                    drawdown_factor = 1.0
                    if current_drawdown > 0.05:  # 5% drawdown
                        drawdown_factor = 0.7
                    if current_drawdown > 0.08:  # 8% drawdown
                        drawdown_factor = 0.5
                    if current_drawdown > 0.10:  # 10% drawdown
                        drawdown_factor = 0.3
                
                # Betting criteria
                ev = row['expected_value']
                confidence = row['confidence']
                
                if (ev >= self.betting_config['min_ev_threshold'] and 
                    confidence >= self.betting_config['min_confidence']):
                    
                    # Kelly Criterion with drawdown adjustment
                    kelly_fraction = (ev / (1.91 - 1)) * self.betting_config['kelly_fraction']
                    kelly_fraction = min(kelly_fraction, self.betting_config['max_risk_per_bet'])
                    
                    # Apply drawdown factor
                    if len(bets) > 0:
                        kelly_fraction *= drawdown_factor
                    
                    bet_amount = current_bankroll * kelly_fraction
                    
                    # Minimum bet size
                    if bet_amount >= 25:  # $25 minimum bet
                        bets.append({
                            'game_id': row['game_id'],
                            'date': row['date'],
                            'home_team': row['home_team'],
                            'away_team': row['away_team'],
                            'predicted_home_prob': row['predicted_home_prob'],
                            'ml_prob': row['ml_prob'],
                            'heuristic_prob': row['heuristic_prob'],
                            'expected_value': ev,
                            'confidence': confidence,
                            'bet_amount': bet_amount,
                            'kelly_fraction': kelly_fraction,
                            'drawdown_factor': drawdown_factor if len(bets) > 0 else 1.0,
                            'actual_result': row['home_win'],
                            'profit_loss': (bet_amount * 0.91) if row['home_win'] else -bet_amount,
                            'transaction_cost': 0,
                            'current_bankroll': current_bankroll,
                            'current_drawdown': (running_max - current_bankroll) / running_max if len(bets) > 0 else 0,
                            'betting_suspended': betting_suspended
                        })
                        
                        daily_bets[game_date] += 1
                        
                        # Update bankroll and tracking
                        if row['home_win']:
                            current_bankroll += bet_amount * 0.91
                            daily_loss = max(0, daily_loss - bet_amount * 0.91)
                        else:
                            current_bankroll -= bet_amount
                            daily_loss += bet_amount
                        
                        # Update running max and drawdown
                        if current_bankroll > running_max:
                            running_max = current_bankroll
                        
                        max_drawdown = max(max_drawdown, (running_max - current_bankroll) / running_max)
                
            except Exception as e:
                logger.error(f"Error processing bet for game {idx}: {e}")
                continue
        
        bets_df = pd.DataFrame(bets)
        logger.info(f"✅ Placed {len(bets_df)} bets with aggressive drawdown protection")
        logger.info(f"🔄 Betting suspended {suspension_count} times due to drawdown")
        
        return bets_df
    
    def calculate_performance(self, bets_df: pd.DataFrame) -> Dict:
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
                'avg_confidence': 0,
                'ml_accuracy': 0,
                'heuristic_accuracy': 0,
                'final_bankroll': 10000,
                'total_return': 0
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
        
        # Final bankroll
        final_bankroll = 10000 + total_profit_loss
        total_return = (final_bankroll - 10000) / 10000 * 100
        
        # Betting metrics
        avg_bet_size = bets_df['bet_amount'].mean()
        avg_ev = bets_df['expected_value'].mean()
        avg_confidence = bets_df['confidence'].mean()
        
        # Calculate drawdown properly
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
        
        # Model accuracy
        ml_bets = bets_df[bets_df['ml_prob'].notna()]
        heuristic_bets = bets_df[bets_df['heuristic_prob'].notna()]
        
        ml_accuracy = 0
        if len(ml_bets) > 0:
            ml_correct = len(ml_bets[ml_bets['actual_result'] == (ml_bets['ml_prob'] > 0.5)])
            ml_accuracy = ml_correct / len(ml_bets) * 100
        
        heuristic_accuracy = 0
        if len(heuristic_bets) > 0:
            heuristic_correct = len(heuristic_bets[heuristic_bets['actual_result'] == (heuristic_bets['heuristic_prob'] > 0.5)])
            heuristic_accuracy = heuristic_correct / len(heuristic_bets) * 100
        
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
            'total_bet_amount': total_bet_amount,
            'ml_accuracy': ml_accuracy,
            'heuristic_accuracy': heuristic_accuracy,
            'final_bankroll': final_bankroll,
            'total_return': total_return
        }
    
    def run_out_of_sample_test(self, features_df: pd.DataFrame) -> Dict:
        """Run out-of-sample validation."""
        logger.info("🔬 Running out-of-sample validation")
        
        # Split data into multiple seasons
        total_games = len(features_df)
        season_size = total_games // 3  # 3 seasons
        
        results = {}
        
        for season in range(3):
            start_idx = season * season_size
            end_idx = start_idx + season_size if season < 2 else total_games
            
            season_data = features_df.iloc[start_idx:end_idx].copy()
            
            # Train on previous seasons
            if season == 0:
                # First season - use heuristic only
                train_data = None
                test_data = season_data
            else:
                train_data = features_df.iloc[:start_idx]
                test_data = season_data
            
            # Make predictions
            if train_data is not None:
                # Train model on previous data
                self.train_ml_model(train_data)
                
                # Predict on test data
                predictions = []
                for idx, row in test_data.iterrows():
                    try:
                        X = row[self.feature_columns].fillna(0).values.reshape(1, -1)
                        X_scaled = self.scaler.transform(X)
                        ml_prob = self.model.predict_proba(X_scaled)[0][1]
                        
                        # Heuristic prediction
                        home_advantage = 0.035
                        home_recent = row['home_win_pct_15']
                        away_recent = row['away_win_pct_15']
                        
                        if home_recent + away_recent > 0:
                            base_prob = home_recent / (home_recent + away_recent)
                        else:
                            base_prob = 0.5
                        
                        form_adjustment = row['win_pct_diff_15'] * 0.25
                        run_adjustment = row['runs_diff_15'] * 0.015
                        era_adjustment = row['era_diff_15'] * 0.008
                        momentum_adjustment = row['home_form_momentum'] * 0.3
                        season_adjustment = row['home_season_advantage'] * 0.2
                        
                        heuristic_prob = base_prob + home_advantage + form_adjustment + run_adjustment + era_adjustment + momentum_adjustment + season_adjustment
                        heuristic_prob = max(0.35, min(0.65, heuristic_prob))
                        
                        # Combined prediction
                        home_prob = 0.6 * ml_prob + 0.4 * heuristic_prob
                        home_prob = max(0.35, min(0.65, home_prob))
                        
                        # Expected value
                        implied_odds = 1.91
                        ev = (home_prob * (implied_odds - 1)) - ((1 - home_prob) * 1)
                        confidence = abs(home_prob - 0.5) * 2
                        
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
                    except Exception as e:
                        continue
                
                predictions_df = pd.DataFrame(predictions)
            else:
                # Use heuristic only for first season
                predictions = []
                for idx, row in test_data.iterrows():
                    try:
                        home_advantage = 0.035
                        home_recent = row['home_win_pct_15']
                        away_recent = row['away_win_pct_15']
                        
                        if home_recent + away_recent > 0:
                            base_prob = home_recent / (home_recent + away_recent)
                        else:
                            base_prob = 0.5
                        
                        form_adjustment = row['win_pct_diff_15'] * 0.25
                        run_adjustment = row['runs_diff_15'] * 0.015
                        era_adjustment = row['era_diff_15'] * 0.008
                        momentum_adjustment = row['home_form_momentum'] * 0.3
                        season_adjustment = row['home_season_advantage'] * 0.2
                        
                        home_prob = base_prob + home_advantage + form_adjustment + run_adjustment + era_adjustment + momentum_adjustment + season_adjustment
                        home_prob = max(0.35, min(0.65, home_prob))
                        
                        implied_odds = 1.91
                        ev = (home_prob * (implied_odds - 1)) - ((1 - home_prob) * 1)
                        confidence = abs(home_prob - 0.5) * 2
                        
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
                    except Exception as e:
                        continue
                
                predictions_df = pd.DataFrame(predictions)
            
            # Place bets
            bets_df = self.aggressive_drawdown_protection_strategy(predictions_df)
            
            # Calculate performance
            performance = self.calculate_performance(bets_df)
            
            results[f'season_{season+1}'] = {
                'games': len(test_data),
                'bets': performance['total_bets'],
                'win_rate': performance['win_rate'],
                'roi': performance['roi'],
                'max_drawdown': performance['max_drawdown'],
                'total_return': performance['total_return']
            }
        
        logger.info("✅ Out-of-sample validation complete")
        return results
    
    def run_comprehensive_test(self):
        """Run comprehensive testing suite."""
        logger.info("🎯 Starting Comprehensive MLB Validation Testing")
        
        # Generate synthetic data
        games_df, standings_df, stats_df = self.generate_synthetic_data(2500)
        
        # Create features
        features_df = self.create_enhanced_features(games_df, standings_df, stats_df)
        if len(features_df) == 0:
            logger.error("❌ No features created")
            return
        
        # Run main validation
        predictions_df = self.enhanced_prediction_model(features_df)
        bets_df = self.aggressive_drawdown_protection_strategy(predictions_df)
        performance = self.calculate_performance(bets_df)
        
        # Run out-of-sample test
        oos_results = self.run_out_of_sample_test(features_df)
        
        # Store all results
        self.test_results.update({
            'main_performance': performance,
            'out_of_sample_results': oos_results,
            'betting_config': self.betting_config,
            'test_date': datetime.now().isoformat(),
            'total_games': len(predictions_df),
            'betting_opportunities': len(predictions_df[predictions_df['expected_value'] >= self.betting_config['min_ev_threshold']])
        })
        
        # Display results
        logger.info("🎉 COMPREHENSIVE TESTING COMPLETE!")
        logger.info(f"📊 Games Analyzed: {len(predictions_df):,}")
        logger.info(f"💰 Bets Placed: {performance['total_bets']:,}")
        logger.info(f"🏆 Win Rate: {performance['win_rate']:.1f}%")
        logger.info(f"💵 ROI: {performance['roi']:.1f}%")
        logger.info(f"💸 Total P&L: ${performance['total_profit_loss']:,.0f}")
        logger.info(f"📈 Total Return: {performance['total_return']:.1f}%")
        logger.info(f"📉 Max Drawdown: {performance['max_drawdown']:.1f}%")
        logger.info(f"🤖 ML Accuracy: {performance['ml_accuracy']:.1f}%")
        logger.info(f"🧠 Heuristic Accuracy: {performance['heuristic_accuracy']:.1f}%")
        
        # Out-of-sample results
        logger.info("🔬 OUT-OF-SAMPLE RESULTS:")
        for season, results in oos_results.items():
            logger.info(f"  {season}: {results['bets']} bets, {results['win_rate']:.1f}% win rate, {results['roi']:.1f}% ROI, {results['max_drawdown']:.1f}% drawdown")
        
        # Calculate viability score
        viability_score = self.calculate_viability_score(performance, oos_results)
        logger.info(f"🎯 VIABILITY SCORE: {viability_score}/10")
        
        # Save results
        self.save_test_results()
        
        return self.test_results
    
    def calculate_viability_score(self, performance: Dict, oos_results: Dict) -> float:
        """Calculate overall viability score."""
        score = 0.0
        
        # Main performance metrics
        if performance['win_rate'] > 52:
            score += 1.5
        elif performance['win_rate'] > 50:
            score += 1.0
        
        if performance['roi'] > 2:
            score += 2.0
        elif performance['roi'] > 0:
            score += 1.0
        
        if performance['max_drawdown'] < 20:
            score += 2.0
        elif performance['max_drawdown'] < 50:
            score += 1.0
        
        if performance['total_bets'] > 500:
            score += 0.5
        
        if performance['ml_accuracy'] > 55:
            score += 1.0
        
        # Out-of-sample consistency
        consistent_seasons = 0
        for season, results in oos_results.items():
            if results['roi'] > 0 and results['max_drawdown'] < 50:
                consistent_seasons += 1
        
        if consistent_seasons >= 2:
            score += 2.0
        elif consistent_seasons >= 1:
            score += 1.0
        
        return min(10.0, score)
    
    def save_test_results(self):
        """Save comprehensive test results."""
        output_dir = Path("tests/results")
        output_dir.mkdir(exist_ok=True)
        
        # Save JSON results
        with open(output_dir / "comprehensive_test_results.json", 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # Save bets CSV if any
        if 'main_performance' in self.test_results and self.test_results['main_performance']['total_bets'] > 0:
            # This would need to be implemented if we want to save individual bets
            pass
        
        logger.info("✅ Test results saved to tests/results/")

if __name__ == "__main__":
    tester = MLBValidationTester()
    results = tester.run_comprehensive_test() 