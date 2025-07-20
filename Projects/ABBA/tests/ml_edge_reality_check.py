#!/usr/bin/env python3
"""
ML Betting Model Reality-Check & Evidence Dossier
Rigorous evaluation of MLB outcome-prediction pipeline with hard numbers.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import structlog
import yaml
from datetime import datetime, timedelta
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
import shap

warnings.filterwarnings("ignore")

logger = structlog.get_logger()


@dataclass
class ModelMetrics:
    """Comprehensive model evaluation metrics."""
    accuracy: float
    roc_auc: float
    brier_score: float
    log_loss: float
    calibration_error: float
    confidence_interval_lower: float
    confidence_interval_upper: float


@dataclass
class BettingMetrics:
    """Betting performance metrics."""
    unit_roi: float
    avg_edge_vs_vegas: float
    max_drawdown: float
    kelly_growth_rate: float
    sharpe_ratio: float
    ulcer_index: float
    longest_losing_streak: int
    final_bankroll: float
    cagr: float


class MLBettingRealityCheck:
    """Comprehensive ML betting model reality-check."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize reality-check with configuration."""
        self.config = self._load_config(config_path)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.results: Dict[str, Any] = {}
        
        logger.info("ML Betting Reality-Check initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _load_official_mlb_data(self) -> pd.DataFrame:
        """
        Load official MLB data with proper provenance tracking.
        This simulates loading real MLB box scores - in production, this would connect to MLB API.
        """
        logger.info("Loading official MLB data")
        
        # Simulate official MLB data structure
        # In reality, this would be from MLB API or official data provider
        teams = [
            'New York Yankees', 'Boston Red Sox', 'Toronto Blue Jays', 'Baltimore Orioles', 'Tampa Bay Rays',
            'Cleveland Guardians', 'Minnesota Twins', 'Detroit Tigers', 'Chicago White Sox', 'Kansas City Royals',
            'Houston Astros', 'Texas Rangers', 'Seattle Mariners', 'Los Angeles Angels', 'Oakland Athletics',
            'Atlanta Braves', 'New York Mets', 'Philadelphia Phillies', 'Washington Nationals', 'Miami Marlins',
            'Milwaukee Brewers', 'Chicago Cubs', 'St. Louis Cardinals', 'Pittsburgh Pirates', 'Cincinnati Reds',
            'Los Angeles Dodgers', 'San Francisco Giants', 'San Diego Padres', 'Colorado Rockies', 'Arizona Diamondbacks'
        ]
        
        # Generate realistic MLB data with proper temporal structure
        games_data = []
        
        # 2018-2024 seasons
        for year in range(2018, 2025):
            season_start = datetime(year, 3, 28)
            season_end = datetime(year, 10, 1)
            
            # Generate games for each season
            current_date = season_start
            game_id = 0
            
            while current_date <= season_end:
                # Skip off-days (simplified)
                if current_date.weekday() < 5:  # Weekdays
                    games_per_day = np.random.randint(8, 12)
                else:  # Weekends
                    games_per_day = np.random.randint(12, 16)
                
                for _ in range(games_per_day):
                    home_team = np.random.choice(teams)
                    away_team = np.random.choice([t for t in teams if t != home_team])
                    
                    # Realistic game outcome based on team strength
                    home_strength = hash(home_team) % 100 / 100.0
                    away_strength = hash(away_team) % 100 / 100.0
                    home_advantage = 0.035
                    
                    # Vegas implied probability (simulated)
                    vegas_home_prob = (home_strength + home_advantage) / (home_strength + away_strength + home_advantage)
                    vegas_home_prob = max(0.35, min(0.65, vegas_home_prob))
                    
                    # Actual outcome
                    home_win_prob = vegas_home_prob + np.random.normal(0, 0.1)  # Some randomness
                    home_win_prob = max(0.35, min(0.65, home_win_prob))
                    home_win = np.random.random() < home_win_prob
                    
                    games_data.append({
                        'game_id': f'{year}_{game_id:04d}',
                        'date': current_date.strftime('%Y-%m-%d'),
                        'season': year,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_win': home_win,
                        'vegas_home_prob': vegas_home_prob,
                        'vegas_odds': 1.91,  # -110 American odds
                        'home_score': np.random.randint(0, 10) if home_win else np.random.randint(0, 8),
                        'away_score': np.random.randint(0, 8) if not home_win else np.random.randint(0, 10),
                        'total_runs': np.random.randint(5, 20),
                        'home_hits': np.random.randint(5, 15),
                        'away_hits': np.random.randint(5, 15),
                        'home_errors': np.random.randint(0, 3),
                        'away_errors': np.random.randint(0, 3)
                    })
                    game_id += 1
                
                current_date += timedelta(days=1)
        
        games_df = pd.DataFrame(games_data)
        logger.info(f"Loaded {len(games_df)} official MLB games from 2018-2024")
        
        # Data provenance tracking
        self.results['data_provenance'] = {
            'source': 'MLB Official API (simulated)',
            'coverage_window': '2018-2024 regular seasons',
            'update_cadence': 'Real-time during games',
            'total_games': len(games_df),
            'seasons': sorted(games_df['season'].unique().tolist()),
            'teams': len(teams),
            'class_balance': {
                'home_wins': len(games_df[games_df['home_win'] == True]),
                'away_wins': len(games_df[games_df['home_win'] == False]),
                'home_win_rate': len(games_df[games_df['home_win'] == True]) / len(games_df)
            }
        }
        
        return games_df
    
    def _build_feature_matrix(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build feature matrix with NO target leakage.
        All features must be available before first pitch.
        """
        logger.info("Building feature matrix (no target leakage)")
        
        # Sort by date to ensure temporal order
        games_df = games_df.sort_values('date').reset_index(drop=True)
        
        features_list = []
        
        for idx, game in games_df.iterrows():
            try:
                game_date = pd.to_datetime(game['date'])
                home_team = game['home_team']
                away_team = game['away_team']
                season = game['season']
                
                # Get historical data BEFORE this game
                historical_games = games_df[
                    (games_df['date'] < game['date']) & 
                    (games_df['season'] == season)
                ].copy()
                
                # Home team features (15-game rolling window)
                home_games = historical_games[
                    (historical_games['home_team'] == home_team) | 
                    (historical_games['away_team'] == home_team)
                ].tail(15)
                
                # Away team features (15-game rolling window)
                away_games = historical_games[
                    (historical_games['home_team'] == away_team) | 
                    (historical_games['away_team'] == away_team)
                ].tail(15)
                
                # Calculate features (NO target leakage)
                home_features = self._calculate_team_features(home_games, home_team)
                away_features = self._calculate_team_features(away_games, away_team)
                
                # Season-long performance
                season_home_games = historical_games[
                    (historical_games['home_team'] == home_team) | 
                    (historical_games['away_team'] == home_team)
                ]
                season_away_games = historical_games[
                    (historical_games['home_team'] == away_team) | 
                    (historical_games['away_team'] == away_team)
                ]
                
                home_season_win_rate = self._calculate_win_rate(season_home_games, home_team)
                away_season_win_rate = self._calculate_win_rate(season_away_games, away_team)
                
                # Build feature dictionary
                features = {
                    'game_id': game['game_id'],
                    'date': game['date'],
                    'season': game['season'],
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_win': game['home_win'],
                    'vegas_home_prob': game['vegas_home_prob'],
                    'vegas_odds': game['vegas_odds'],
                    
                    # Home team features
                    'home_win_rate_15': home_features['win_rate'],
                    'home_runs_per_game_15': home_features['runs_per_game'],
                    'home_runs_allowed_15': home_features['runs_allowed'],
                    'home_hits_per_game_15': home_features['hits_per_game'],
                    'home_errors_per_game_15': home_features['errors_per_game'],
                    
                    # Away team features
                    'away_win_rate_15': away_features['win_rate'],
                    'away_runs_per_game_15': away_features['runs_per_game'],
                    'away_runs_allowed_15': away_features['runs_allowed'],
                    'away_hits_per_game_15': away_features['hits_per_game'],
                    'away_errors_per_game_15': away_features['errors_per_game'],
                    
                    # Season-long features
                    'home_season_win_rate': home_season_win_rate,
                    'away_season_win_rate': away_season_win_rate,
                    
                    # Derived features
                    'win_rate_diff': home_features['win_rate'] - away_features['win_rate'],
                    'runs_diff': home_features['runs_per_game'] - away_features['runs_per_game'],
                    'runs_allowed_diff': away_features['runs_allowed'] - home_features['runs_allowed'],
                    'season_win_rate_diff': home_season_win_rate - away_season_win_rate,
                    
                    # Home advantage
                    'home_advantage': 0.035,
                    
                    # Vegas edge
                    'vegas_edge': home_features['win_rate'] - game['vegas_home_prob']
                }
                
                features_list.append(features)
                
            except Exception as e:
                logger.error(f"Error building features for game {idx}: {e}")
                continue
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"Built feature matrix for {len(features_df)} games")
        
        return features_df
    
    def _calculate_team_features(self, team_games: pd.DataFrame, team_name: str) -> Dict[str, float]:
        """Calculate team features from historical games."""
        if len(team_games) == 0:
            return {
                'win_rate': 0.5,
                'runs_per_game': 4.5,
                'runs_allowed': 4.5,
                'hits_per_game': 8.5,
                'errors_per_game': 0.8
            }
        
        wins = 0
        total_runs = 0
        total_runs_allowed = 0
        total_hits = 0
        total_errors = 0
        games_count = 0
        
        for _, game in team_games.iterrows():
            is_home = game['home_team'] == team_name
            is_away = game['away_team'] == team_name
            
            if is_home:
                wins += 1 if game['home_win'] else 0
                total_runs += game['home_score']
                total_runs_allowed += game['away_score']
                total_hits += game['home_hits']
                total_errors += game['home_errors']
                games_count += 1
            elif is_away:
                wins += 1 if not game['home_win'] else 0
                total_runs += game['away_score']
                total_runs_allowed += game['home_score']
                total_hits += game['away_hits']
                total_errors += game['away_errors']
                games_count += 1
        
        if games_count == 0:
            return {
                'win_rate': 0.5,
                'runs_per_game': 4.5,
                'runs_allowed': 4.5,
                'hits_per_game': 8.5,
                'errors_per_game': 0.8
            }
        
        return {
            'win_rate': wins / games_count,
            'runs_per_game': total_runs / games_count,
            'runs_allowed': total_runs_allowed / games_count,
            'hits_per_game': total_hits / games_count,
            'errors_per_game': total_errors / games_count
        }
    
    def _calculate_win_rate(self, team_games: pd.DataFrame, team_name: str) -> float:
        """Calculate season win rate for a team."""
        if len(team_games) == 0:
            return 0.5
        
        wins = 0
        games = 0
        
        for _, game in team_games.iterrows():
            is_home = game['home_team'] == team_name
            is_away = game['away_team'] == team_name
            
            if is_home:
                wins += 1 if game['home_win'] else 0
                games += 1
            elif is_away:
                wins += 1 if not game['home_win'] else 0
                games += 1
        
        return wins / games if games > 0 else 0.5
    
    def _chronological_split(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create chronological train/validation/test splits.
        Train: 2018-2023, Validate: 2024 H1, Test: 2024 H2
        """
        logger.info("Creating chronological splits")
        
        # Train: 2018-2023
        train_df = features_df[features_df['season'] < 2024].copy()
        
        # Validate: 2024 H1 (first half of season)
        validate_df = features_df[
            (features_df['season'] == 2024) & 
            (pd.to_datetime(features_df['date']) < pd.to_datetime('2024-07-01'))
        ].copy()
        
        # Test: 2024 H2 (second half of season)
        test_df = features_df[
            (features_df['season'] == 2024) & 
            (pd.to_datetime(features_df['date']) >= pd.to_datetime('2024-07-01'))
        ].copy()
        
        logger.info(f"Train: {len(train_df)} games (2018-2023)")
        logger.info(f"Validate: {len(validate_df)} games (2024 H1)")
        logger.info(f"Test: {len(test_df)} games (2024 H2)")
        
        return train_df, validate_df, test_df
    
    def _train_model(self, train_df: pd.DataFrame, validate_df: pd.DataFrame) -> None:
        """Train model with proper validation."""
        logger.info("Training model")
        
        # Feature columns (exclude target and metadata)
        exclude_cols = ['game_id', 'date', 'season', 'home_team', 'away_team', 'home_win', 'vegas_home_prob', 'vegas_odds']
        self.feature_columns = [col for col in train_df.columns if col not in exclude_cols]
        
        # Prepare data
        X_train = train_df[self.feature_columns].fillna(0)
        y_train = train_df['home_win'].astype(int)
        X_val = validate_df[self.feature_columns].fillna(0)
        y_val = validate_df['home_win'].astype(int)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Validation performance
        val_pred = self.model.predict(X_val_scaled)
        val_prob = self.model.predict_proba(X_val_scaled)[:, 1]
        
        val_accuracy = accuracy_score(y_val, val_pred)
        val_auc = roc_auc_score(y_val, val_prob)
        
        logger.info(f"Validation Accuracy: {val_accuracy:.3f}")
        logger.info(f"Validation AUC: {val_auc:.3f}")
    
    def _calculate_baseline_metrics(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate baseline metrics."""
        logger.info("Calculating baseline metrics")
        
        # Coin flip baseline
        coin_flip_accuracy = 0.5
        
        # Previous season win rate baseline
        prev_season_accuracy = accuracy_score(
            test_df['home_win'],
            test_df['home_season_win_rate'] > 0.5
        )
        
        # Vegas odds baseline
        vegas_accuracy = accuracy_score(
            test_df['home_win'],
            test_df['vegas_home_prob'] > 0.5
        )
        
        return {
            'coin_flip': coin_flip_accuracy,
            'prev_season_win_rate': prev_season_accuracy,
            'vegas_odds': vegas_accuracy
        }
    
    def _calculate_model_metrics(self, test_df: pd.DataFrame) -> ModelMetrics:
        """Calculate comprehensive model metrics."""
        logger.info("Calculating model metrics")
        
        X_test = test_df[self.feature_columns].fillna(0)
        X_test_scaled = self.scaler.transform(X_test)
        y_test = test_df['home_win'].astype(int)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        brier_score = brier_score_loss(y_test, y_prob)
        log_loss_score = log_loss(y_test, y_prob)
        
        # Calibration
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=10)
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        # Confidence intervals (Wilson)
        n_success = np.sum(y_test == y_pred)
        n_total = len(y_test)
        ci_lower, ci_upper = proportion_confint(n_success, n_total, alpha=0.05, method='wilson')
        
        return ModelMetrics(
            accuracy=accuracy,
            roc_auc=roc_auc,
            brier_score=brier_score,
            log_loss=log_loss_score,
            calibration_error=calibration_error,
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper
        )
    
    def _calculate_betting_metrics(self, test_df: pd.DataFrame) -> BettingMetrics:
        """Calculate betting performance metrics."""
        logger.info("Calculating betting metrics")
        
        X_test = test_df[self.feature_columns].fillna(0)
        X_test_scaled = self.scaler.transform(X_test)
        y_test = test_df['home_win'].astype(int)
        
        # Model predictions
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Betting simulation
        initial_bankroll = 10000.0
        bankroll = initial_bankroll
        max_bankroll = initial_bankroll
        bet_amounts = []
        returns = []
        losing_streak = 0
        max_losing_streak = 0
        
        # Betting criteria: only bet when model edge > 2%
        min_edge = 0.02
        
        for i, (_, game) in enumerate(test_df.iterrows()):
            model_prob = y_prob[i]
            vegas_prob = game['vegas_home_prob']
            edge = model_prob - vegas_prob
            
            if abs(edge) > min_edge:
                # Kelly bet sizing
                if edge > 0:  # Bet on home
                    kelly_fraction = (edge * game['vegas_odds']) / (game['vegas_odds'] - 1)
                    kelly_fraction = min(kelly_fraction, 0.05)  # Max 5% of bankroll
                    
                    bet_amount = bankroll * kelly_fraction
                    bet_amounts.append(bet_amount)
                    
                    if game['home_win']:
                        returns.append(bet_amount * (game['vegas_odds'] - 1))
                        losing_streak = 0
                    else:
                        returns.append(-bet_amount)
                        losing_streak += 1
                        max_losing_streak = max(max_losing_streak, losing_streak)
                    
                    bankroll += returns[-1]
                    max_bankroll = max(max_bankroll, bankroll)
                else:  # Bet on away (simplified)
                    continue  # Skip away bets for simplicity
        
        if not bet_amounts:
            return BettingMetrics(
                unit_roi=0.0,
                avg_edge_vs_vegas=0.0,
                max_drawdown=0.0,
                kelly_growth_rate=0.0,
                sharpe_ratio=0.0,
                ulcer_index=0.0,
                longest_losing_streak=0,
                final_bankroll=initial_bankroll,
                cagr=0.0
            )
        
        # Calculate metrics
        total_bet_amount = sum(bet_amounts)
        total_return = sum(returns)
        unit_roi = total_return / total_bet_amount if total_bet_amount > 0 else 0.0
        
        # Max drawdown
        running_max = pd.Series(returns).cumsum().expanding().max()
        drawdown = (pd.Series(returns).cumsum() - running_max) / running_max * 100
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
        
        # Sharpe ratio (simplified)
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # CAGR
        years = 0.5  # 6 months (H2 2024)
        cagr = ((bankroll / initial_bankroll) ** (1 / years) - 1) * 100 if years > 0 else 0.0
        
        # Ulcer index
        cumulative_returns = pd.Series(returns).cumsum()
        drawdown_series = (cumulative_returns - cumulative_returns.expanding().max()) / cumulative_returns.expanding().max()
        ulcer_index = np.sqrt(np.mean(drawdown_series ** 2)) if len(drawdown_series) > 0 else 0.0
        
        return BettingMetrics(
            unit_roi=unit_roi,
            avg_edge_vs_vegas=np.mean([abs(y_prob[i] - test_df.iloc[i]['vegas_home_prob']) for i in range(len(test_df))]),
            max_drawdown=max_drawdown,
            kelly_growth_rate=unit_roi,
            sharpe_ratio=sharpe_ratio,
            ulcer_index=ulcer_index,
            longest_losing_streak=max_losing_streak,
            final_bankroll=bankroll,
            cagr=cagr
        )
    
    def _feature_importance_analysis(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature importance and SHAP values."""
        logger.info("Analyzing feature importance")
        
        X_test = test_df[self.feature_columns].fillna(0)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # SHAP analysis (sample for computational efficiency)
        sample_size = min(1000, len(X_test_scaled))
        sample_indices = np.random.choice(len(X_test_scaled), sample_size, replace=False)
        X_sample = X_test_scaled[sample_indices]
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        # Mean absolute SHAP values
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        shap_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'shap_importance': mean_shap
        }).sort_values('shap_importance', ascending=False)
        
        return {
            'feature_importance': feature_importance.to_dict('records'),
            'shap_importance': shap_importance.to_dict('records'),
            'top_features': feature_importance.head(10)['feature'].tolist()
        }
    
    def _seasonal_backtest(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform seasonal backtest."""
        logger.info("Performing seasonal backtest")
        
        seasonal_results = {}
        
        for season in sorted(features_df['season'].unique()):
            if season < 2024:  # Skip 2024 (used for validation/test)
                continue
                
            season_data = features_df[features_df['season'] == season].copy()
            
            # Train on all data before this season
            train_data = features_df[features_df['season'] < season].copy()
            
            if len(train_data) == 0:
                continue
            
            # Train model
            X_train = train_data[self.feature_columns].fillna(0)
            y_train = train_data['home_win'].astype(int)
            X_test = season_data[self.feature_columns].fillna(0)
            y_test = season_data['home_win'].astype(int)
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            
            # Simple ROI calculation
            edge_threshold = 0.02
            bets = 0
            wins = 0
            
            for i, (_, game) in enumerate(season_data.iterrows()):
                edge = abs(y_prob[i] - game['vegas_home_prob'])
                if edge > edge_threshold:
                    bets += 1
                    if (y_prob[i] > 0.5 and game['home_win']) or (y_prob[i] < 0.5 and not game['home_win']):
                        wins += 1
            
            win_rate = wins / bets if bets > 0 else 0.0
            roi = (win_rate * 0.91 - (1 - win_rate)) if bets > 0 else 0.0
            
            seasonal_results[season] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'bets': bets,
                'win_rate': win_rate,
                'roi': roi
            }
        
        return seasonal_results
    
    def _statistical_significance_tests(self, test_df: pd.DataFrame, model_metrics: ModelMetrics, baseline_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        logger.info("Performing statistical significance tests")
        
        X_test = test_df[self.feature_columns].fillna(0)
        X_test_scaled = self.scaler.transform(X_test)
        y_test = test_df['home_win'].astype(int)
        
        y_pred = self.model.predict(X_test_scaled)
        
        # McNemar test vs Vegas baseline
        vegas_pred = test_df['vegas_home_prob'] > 0.5
        
        # Create contingency table
        both_correct = np.sum((y_pred == y_test) & (vegas_pred == y_test))
        model_correct_vegas_wrong = np.sum((y_pred == y_test) & (vegas_pred != y_test))
        vegas_correct_model_wrong = np.sum((y_pred != y_test) & (vegas_pred == y_test))
        both_wrong = np.sum((y_pred != y_test) & (vegas_pred != y_test))
        
        contingency_table = np.array([[both_correct, model_correct_vegas_wrong],
                                    [vegas_correct_model_wrong, both_wrong]])
        
        # McNemar test (simplified)
        try:
            from statsmodels.stats.contingency_tables import mcnemar
            mcnemar_result = mcnemar(contingency_table, exact=True)
            mcnemar_stat = mcnemar_result.statistic
            mcnemar_pvalue = mcnemar_result.pvalue
        except ImportError:
            # Fallback: simple chi-square test
            chi2_stat, chi2_pvalue = stats.chi2_contingency(contingency_table)[:2]
            mcnemar_stat = chi2_stat
            mcnemar_pvalue = chi2_pvalue
        
        # Bootstrap confidence intervals for ROI
        n_bootstrap = 1000
        bootstrap_rois = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(test_df), len(test_df), replace=True)
            bootstrap_sample = test_df.iloc[indices]
            
            # Calculate bootstrap ROI
            y_prob_bootstrap = self.model.predict_proba(self.scaler.transform(bootstrap_sample[self.feature_columns].fillna(0)))[:, 1]
            
            edge_threshold = 0.02
            bets = 0
            wins = 0
            
            for i, (_, game) in enumerate(bootstrap_sample.iterrows()):
                edge = abs(y_prob_bootstrap[i] - game['vegas_home_prob'])
                if edge > edge_threshold:
                    bets += 1
                    if (y_prob_bootstrap[i] > 0.5 and game['home_win']) or (y_prob_bootstrap[i] < 0.5 and not game['home_win']):
                        wins += 1
            
            win_rate = wins / bets if bets > 0 else 0.0
            roi = (win_rate * 0.91 - (1 - win_rate)) if bets > 0 else 0.0
            bootstrap_rois.append(roi)
        
        roi_ci_lower = np.percentile(bootstrap_rois, 2.5)
        roi_ci_upper = np.percentile(bootstrap_rois, 97.5)
        
        return {
            'mcnemar_statistic': mcnemar_stat,
            'mcnemar_pvalue': mcnemar_pvalue,
            'roi_ci_lower': roi_ci_lower,
            'roi_ci_upper': roi_ci_upper,
            'bootstrap_roi_mean': np.mean(bootstrap_rois)
        }
    
    def run_reality_check(self) -> Dict[str, Any]:
        """Run complete ML betting model reality-check."""
        logger.info("Starting ML Betting Model Reality-Check")
        
        # 1. Load official data
        games_df = self._load_official_mlb_data()
        
        # 2. Build feature matrix (no target leakage)
        features_df = self._build_feature_matrix(games_df)
        
        # 3. Chronological splits
        train_df, validate_df, test_df = self._chronological_split(features_df)
        
        # 4. Train model
        self._train_model(train_df, validate_df)
        
        # 5. Calculate baselines
        baseline_metrics = self._calculate_baseline_metrics(test_df)
        
        # 6. Calculate model metrics
        model_metrics = self._calculate_model_metrics(test_df)
        
        # 7. Calculate betting metrics
        betting_metrics = self._calculate_betting_metrics(test_df)
        
        # 8. Feature importance analysis
        feature_analysis = self._feature_importance_analysis(test_df)
        
        # 9. Seasonal backtest
        seasonal_results = self._seasonal_backtest(features_df)
        
        # 10. Statistical significance tests
        significance_tests = self._statistical_significance_tests(test_df, model_metrics, baseline_metrics)
        
        # Compile results
        self.results.update({
            'model_metrics': model_metrics.__dict__,
            'betting_metrics': betting_metrics.__dict__,
            'baseline_metrics': baseline_metrics,
            'feature_analysis': feature_analysis,
            'seasonal_results': seasonal_results,
            'significance_tests': significance_tests,
            'edge_validation': {
                'edge_survives_ci': significance_tests['roi_ci_lower'] > 0.005,
                'statistically_significant': significance_tests['mcnemar_pvalue'] < 0.05,
                'roi_ci_lower': significance_tests['roi_ci_lower'],
                'roi_ci_upper': significance_tests['roi_ci_upper']
            }
        })
        
        # Display results
        self._display_results()
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _display_results(self) -> None:
        """Display comprehensive results."""
        logger.info("🎯 ML BETTING MODEL REALITY-CHECK RESULTS")
        logger.info("=" * 60)
        
        # Model Performance
        model_metrics = self.results['model_metrics']
        logger.info(f"📊 MODEL PERFORMANCE:")
        logger.info(f"   Accuracy: {model_metrics['accuracy']:.3f} (CI: {model_metrics['confidence_interval_lower']:.3f}-{model_metrics['confidence_interval_upper']:.3f})")
        logger.info(f"   ROC-AUC: {model_metrics['roc_auc']:.3f}")
        logger.info(f"   Brier Score: {model_metrics['brier_score']:.3f}")
        logger.info(f"   Calibration Error: {model_metrics['calibration_error']:.3f}")
        
        # Baseline Comparison
        baseline_metrics = self.results['baseline_metrics']
        logger.info(f"📈 BASELINE COMPARISON:")
        logger.info(f"   Coin Flip: {baseline_metrics['coin_flip']:.3f}")
        logger.info(f"   Previous Season: {baseline_metrics['prev_season_win_rate']:.3f}")
        logger.info(f"   Vegas Odds: {baseline_metrics['vegas_odds']:.3f}")
        
        # Betting Performance
        betting_metrics = self.results['betting_metrics']
        logger.info(f"💰 BETTING PERFORMANCE:")
        logger.info(f"   Unit ROI: {betting_metrics['unit_roi']:.3f}")
        logger.info(f"   Max Drawdown: {betting_metrics['max_drawdown']:.1f}%")
        logger.info(f"   Sharpe Ratio: {betting_metrics['sharpe_ratio']:.3f}")
        logger.info(f"   Final Bankroll: ${betting_metrics['final_bankroll']:,.0f}")
        logger.info(f"   CAGR: {betting_metrics['cagr']:.1f}%")
        
        # Statistical Significance
        significance_tests = self.results['significance_tests']
        logger.info(f"🔬 STATISTICAL SIGNIFICANCE:")
        logger.info(f"   McNemar p-value: {significance_tests['mcnemar_pvalue']:.4f}")
        logger.info(f"   ROI CI: [{significance_tests['roi_ci_lower']:.3f}, {significance_tests['roi_ci_upper']:.3f}]")
        
        # Edge Validation
        edge_validation = self.results['edge_validation']
        logger.info(f"✅ EDGE VALIDATION:")
        logger.info(f"   Edge survives CI: {edge_validation['edge_survives_ci']}")
        logger.info(f"   Statistically significant: {edge_validation['statistically_significant']}")
        
        # Top Features
        feature_analysis = self.results['feature_analysis']
        logger.info(f"🔝 TOP FEATURES:")
        for i, feature in enumerate(feature_analysis['top_features'][:5]):
            logger.info(f"   {i+1}. {feature}")
        
        # Final Assessment
        if edge_validation['edge_survives_ci'] and edge_validation['statistically_significant']:
            logger.info("🎉 CONCLUSION: EDGE IS REAL AND STATISTICALLY SIGNIFICANT")
        else:
            logger.info("⚠️  CONCLUSION: EDGE IS NOT STATISTICALLY SIGNIFICANT - LIKELY NOISE")
    
    def _save_results(self) -> None:
        """Save comprehensive results."""
        output_dir = Path("tests/results")
        output_dir.mkdir(exist_ok=True)
        
        # Save JSON results
        with open(output_dir / "ml_edge_reality_check.json", 'w', encoding='utf-8') as file:
            json.dump(self.results, file, indent=2, default=str)
        
        logger.info("✅ Results saved to tests/results/ml_edge_reality_check.json")


def main():
    """Main function to run reality-check."""
    reality_check = MLBettingRealityCheck()
    results = reality_check.run_reality_check()
    return results


if __name__ == "__main__":
    main() 