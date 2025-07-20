"""
Simulation and prediction module for ABMBA system.
Includes Monte Carlo simulations, ML models, and statistical analysis.
"""

import os
from datetime import datetime
from decimal import Decimal
from typing import Any

import joblib
import numpy as np
import pandas as pd
import structlog
from scipy import stats
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# from statsmodels.tsa.arch.arch_model import arch_model  # Commented out for compatibility
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

# Advanced 2025 imports (optional)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Neural network features will be disabled.")

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False
    print("Warning: NetworkX not available. Graph features will be disabled.")

try:
    import arviz as az
    import pymc as pm
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("Warning: PyMC not available. Bayesian features will be disabled.")

try:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. NLP features will be disabled.")

try:
    import dgl
    import dgl.nn as dglnn
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    print("Warning: DGL not available. Graph neural network features will be disabled.")

try:
    import heartpy as hp
    import neurokit2 as nk
    from biosppy.signals import ecg, eeg, emg
    BIOSIGNAL_AVAILABLE = True
except ImportError:
    BIOSIGNAL_AVAILABLE = False
    print("Warning: Biosignal processing libraries not available. Biometric features will be disabled.")

from models import Event, MarketType, Odds, SimulationResult

logger = structlog.get_logger()


class MonteCarloSimulator:
    """Monte Carlo simulation for betting outcomes."""

    def __init__(self, iterations: int = 10000):
        self.iterations = iterations
        self.rng = np.random.default_rng()

    def simulate_moneyline(self, home_odds: float, away_odds: float,
                          home_win_prob: float = None) -> dict[str, float]:
        """
        Simulate moneyline betting outcomes.
        
        Args:
            home_odds: Home team odds (American format)
            away_odds: Away team odds (American format)
            home_win_prob: Optional home team win probability (if None, calculated from odds)
        
        Returns:
            Dictionary with simulation results
        """
        # Convert American odds to implied probabilities
        if home_odds > 0:
            home_implied_prob = 100 / (home_odds + 100)
        else:
            home_implied_prob = abs(home_odds) / (abs(home_odds) + 100)

        if away_odds > 0:
            away_implied_prob = 100 / (away_odds + 100)
        else:
            away_implied_prob = abs(away_odds) / (abs(away_odds) + 100)

        # Normalize probabilities
        total_prob = home_implied_prob + away_implied_prob
        home_implied_prob /= total_prob
        away_implied_prob /= total_prob

        # Use provided probability or implied probability
        if home_win_prob is not None:
            home_win_prob = home_win_prob
        else:
            home_win_prob = home_implied_prob

        # Simulate outcomes
        outcomes = self.rng.choice([1, 0], size=self.iterations, p=[home_win_prob, 1-home_win_prob])

        # Calculate payouts
        home_payout = home_odds if home_odds > 0 else 100
        away_payout = away_odds if away_odds > 0 else 100

        payouts = np.where(outcomes == 1, home_payout, -100)

        # Calculate statistics
        win_rate = np.mean(outcomes)
        expected_value = np.mean(payouts) / 100
        variance = np.var(payouts) / 10000
        std_dev = np.sqrt(variance)

        # Calculate confidence intervals
        confidence_interval = stats.t.interval(0.95, len(payouts)-1,
                                             loc=np.mean(payouts),
                                             scale=stats.sem(payouts))

        return {
            'win_probability': win_rate,
            'expected_value': expected_value,
            'variance': variance,
            'std_dev': std_dev,
            'confidence_interval_lower': confidence_interval[0] / 100,
            'confidence_interval_upper': confidence_interval[1] / 100,
            'home_implied_prob': home_implied_prob,
            'away_implied_prob': away_implied_prob
        }

    def simulate_spread(self, spread: float, home_odds: float, away_odds: float,
                       home_win_prob: float = None) -> dict[str, float]:
        """
        Simulate spread betting outcomes.
        
        Args:
            spread: Point spread (positive means home team favored)
            home_odds: Home team odds
            away_odds: Away team odds
            home_win_prob: Optional home team win probability
        
        Returns:
            Dictionary with simulation results
        """
        # Convert spread to probability using normal distribution
        # Assuming spread follows normal distribution with mean=0, std=spread/2
        spread_std = abs(spread) / 2

        if home_win_prob is None:
            # Calculate probability from spread
            home_win_prob = 1 - norm.cdf(spread, 0, spread_std)

        # Simulate margin of victory
        margins = self.rng.normal(0, spread_std, self.iterations)

        # Determine outcomes (home team covers if margin > -spread)
        outcomes = (margins > -spread).astype(int)

        # Calculate payouts
        home_payout = home_odds if home_odds > 0 else 100
        payouts = np.where(outcomes == 1, home_payout, -100)

        # Calculate statistics
        win_rate = np.mean(outcomes)
        expected_value = np.mean(payouts) / 100
        variance = np.var(payouts) / 10000

        # Confidence intervals
        confidence_interval = stats.t.interval(0.95, len(payouts)-1,
                                             loc=np.mean(payouts),
                                             scale=stats.sem(payouts))

        return {
            'win_probability': win_rate,
            'expected_value': expected_value,
            'variance': variance,
            'confidence_interval_lower': confidence_interval[0] / 100,
            'confidence_interval_upper': confidence_interval[1] / 100,
            'avg_margin': np.mean(margins),
            'margin_std': np.std(margins)
        }

    def simulate_totals(self, total: float, over_odds: float, under_odds: float,
                       avg_score: float = None) -> dict[str, float]:
        """
        Simulate totals (over/under) betting outcomes.
        
        Args:
            total: Total points line
            over_odds: Over odds
            under_odds: Under odds
            avg_score: Optional average score (if None, uses total/2)
        
        Returns:
            Dictionary with simulation results
        """
        if avg_score is None:
            avg_score = total / 2

        # Simulate total scores using normal distribution
        # Assuming scores follow normal distribution around avg_score
        score_std = total * 0.15  # 15% of total as standard deviation
        scores = self.rng.normal(avg_score, score_std, self.iterations)

        # Determine outcomes (over if score > total)
        outcomes = (scores > total).astype(int)

        # Calculate payouts
        over_payout = over_odds if over_odds > 0 else 100
        payouts = np.where(outcomes == 1, over_payout, -100)

        # Calculate statistics
        win_rate = np.mean(outcomes)
        expected_value = np.mean(payouts) / 100
        variance = np.var(payouts) / 10000

        # Confidence intervals
        confidence_interval = stats.t.interval(0.95, len(payouts)-1,
                                             loc=np.mean(payouts),
                                             scale=stats.sem(payouts))

        return {
            'win_probability': win_rate,
            'expected_value': expected_value,
            'variance': variance,
            'confidence_interval_lower': confidence_interval[0] / 100,
            'confidence_interval_upper': confidence_interval[1] / 100,
            'avg_score': np.mean(scores),
            'score_std': np.std(scores)
        }


class KellyCriterion:
    """Kelly Criterion implementation for optimal bet sizing."""

    @staticmethod
    def calculate_kelly_fraction(win_prob: float, odds: float,
                               fraction: float = 0.5) -> float:
        """
        Calculate Kelly Criterion fraction for bet sizing.
        
        Args:
            win_prob: Probability of winning
            odds: American odds
            fraction: Fraction of Kelly to use (default 0.5 for half-Kelly)
        
        Returns:
            Recommended fraction of bankroll to bet
        """
        # Convert American odds to decimal odds
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1

        # Kelly formula: f = (bp - q) / b
        # where b = decimal odds - 1, p = win probability, q = loss probability
        b = decimal_odds - 1
        p = win_prob
        q = 1 - p

        kelly_fraction = (b * p - q) / b

        # Apply fraction and ensure non-negative
        kelly_fraction = max(0, kelly_fraction * fraction)

        return kelly_fraction

    @staticmethod
    def calculate_optimal_stake(bankroll: float, win_prob: float, odds: float,
                              max_risk_percent: float = 0.02,
                              kelly_fraction: float = 0.5) -> float:
        """
        Calculate optimal stake using Kelly Criterion with constraints.
        
        Args:
            bankroll: Current bankroll
            win_prob: Probability of winning
            odds: American odds
            max_risk_percent: Maximum percentage of bankroll to risk
            kelly_fraction: Fraction of Kelly to use
        
        Returns:
            Recommended stake amount
        """
        kelly_fraction = KellyCriterion.calculate_kelly_fraction(
            win_prob, odds, kelly_fraction
        )

        # Calculate stake based on Kelly
        kelly_stake = bankroll * kelly_fraction

        # Apply maximum risk constraint
        max_stake = bankroll * max_risk_percent

        return min(kelly_stake, max_stake)

    @staticmethod
    def optimized_kelly(p_win: float, odds: float, bankroll: float,
                       sharpe_target: float = 1.0, initial_bankroll: float = None) -> float:
        """
        Optimized Kelly with Sharpe ratio optimization for risk-adjusted returns.
        
        Args:
            p_win: Probability of winning
            odds: Decimal odds
            bankroll: Current bankroll
            sharpe_target: Target Sharpe ratio
            initial_bankroll: Initial bankroll for drawdown protection
        
        Returns:
            Optimized stake amount
        """
        from scipy.optimize import minimize

        # Convert American odds to decimal if needed
        if odds > 100 or odds < -100:  # American odds
            if odds > 0:
                decimal_odds = (odds / 100) + 1
            else:
                decimal_odds = (100 / abs(odds)) + 1
        else:
            decimal_odds = odds

        # Basic Kelly: f = (p*(odds-1) - (1-p)) / (odds-1)
        kelly_f = (p_win * (decimal_odds - 1) - (1 - p_win)) / (decimal_odds - 1)
        half_kelly = kelly_f / 2  # Conservative starting point

        # Optimize with Sharpe: Maximize (expected return) / (std dev)
        def neg_sharpe(f):
            ret = f * (p_win * decimal_odds - 1)  # Expected return
            vol = f * np.sqrt(p_win * (1 - p_win)) * decimal_odds  # Volatility approx
            return - (ret / vol)  # Negative for minimization

        try:
            res = minimize(neg_sharpe, half_kelly, bounds=[(0, 1)])
            stake_fraction = res.x[0]
        except:
            stake_fraction = half_kelly  # Fallback to half-Kelly

        # Enforce risk management rules
        stake = min(stake_fraction * bankroll, 0.02 * bankroll)  # <2% risk

        # Drawdown protection
        if initial_bankroll and bankroll < 0.2 * initial_bankroll:
            stake = 0  # Halt if bankroll <20% initial

        return stake

    @staticmethod
    def calculate_portfolio_kelly(bets: list[dict], bankroll: float,
                                correlation_matrix: np.ndarray = None) -> list[float]:
        """
        Calculate Kelly fractions for a portfolio of bets considering correlations.
        
        Args:
            bets: List of bet dictionaries with 'win_prob', 'odds', 'stake'
            bankroll: Total bankroll
            correlation_matrix: Correlation matrix between bets
        
        Returns:
            List of optimal stakes for each bet
        """
        if not correlation_matrix:
            # Assume no correlation if not provided
            correlation_matrix = np.eye(len(bets))

        # Portfolio optimization using quadratic programming
        from scipy.optimize import minimize

        def portfolio_variance(stakes):
            # Calculate portfolio variance considering correlations
            stakes_array = np.array(stakes)
            return stakes_array.T @ correlation_matrix @ stakes_array

        def portfolio_return(stakes):
            # Calculate expected portfolio return
            return sum(stakes[i] * (bets[i]['win_prob'] * bets[i]['odds'] - 1)
                      for i in range(len(bets)))

        def objective(stakes):
            # Maximize Sharpe ratio: return / sqrt(variance)
            ret = portfolio_return(stakes)
            var = portfolio_variance(stakes)
            return -ret / np.sqrt(var) if var > 0 else -ret

        # Constraints: stakes >= 0, sum(stakes) <= bankroll
        constraints = [
            {'type': 'ineq', 'fun': lambda x: bankroll - sum(x)},  # Budget constraint
        ]

        bounds = [(0, bankroll * 0.02) for _ in bets]  # Individual bet limits

        # Initial guess: equal allocation
        x0 = [bankroll / len(bets) * 0.01] * len(bets)

        try:
            result = minimize(objective, x0, bounds=bounds, constraints=constraints)
            return result.x.tolist()
        except:
            # Fallback to individual Kelly
            return [KellyCriterion.calculate_optimal_stake(bankroll, bet['win_prob'], bet['odds'])
                    for bet in bets]


class SimulationManager:
    """Enhanced simulation manager with ensemble prediction capabilities."""

    def __init__(self, db_manager, config: dict):
        self.db_manager = db_manager
        self.config = config
        self.monte_carlo = MonteCarloSimulator(
            iterations=config['simulation']['monte_carlo_iterations']
        )
        self.ml_predictor = MLPredictor('random_forest')
        self.stats_analyzer = StatisticalAnalyzer()
        self.bias_mitigator = BiasMitigator()

        # Initialize ensemble predictor if available
        if TORCH_AVAILABLE and PYMC_AVAILABLE and DGL_AVAILABLE:
            self.ensemble_predictor = EnsemblePredictor(config)
        else:
            self.ensemble_predictor = None
            logger.warning("Ensemble predictor not available due to missing dependencies")

    async def run_event_simulation(self, event: Event, odds_list: list[Odds],
                                 biometric_data: dict = None, sentiment_data: list[str] = None) -> SimulationResult:
        """
        Run comprehensive simulation for a single event with ensemble prediction.
        
        Args:
            event: Event to simulate
            odds_list: List of odds for the event
            biometric_data: Real-time biometric data
            sentiment_data: Social media/news sentiment data
        
        Returns:
            SimulationResult object
        """
        logger.info(f"Running enhanced simulation for event: {event.id}")

        # Use ensemble prediction if available
        if self.ensemble_predictor:
            event_data = self._prepare_event_data(event, odds_list)
            ensemble_prob, uncertainty = self.ensemble_predictor.ensemble_prediction(
                event_data, biometric_data, sentiment_data
            )

            # Use ensemble prediction for simulation
            best_ev = 0.0
            best_odds = None
            best_simulation = None

            for odds in odds_list:
                if odds.market_type == MarketType.MONEYLINE:
                    simulation = self.monte_carlo.simulate_moneyline(
                        float(odds.odds),
                        float(next((o.odds for o in odds_list if o.selection != odds.selection), odds.odds)),
                        ensemble_prob if odds.selection == 'home' else 1 - ensemble_prob
                    )

                    if simulation['expected_value'] > best_ev:
                        best_ev = simulation['expected_value']
                        best_odds = odds
                        best_simulation = simulation

            if best_simulation:
                # Calculate Kelly fraction with enhanced optimization
                kelly_fraction = KellyCriterion.optimized_kelly(
                    best_simulation['win_probability'],
                    float(best_odds.odds),
                    self.config['bankroll']['initial_amount'],
                    sharpe_target=1.0,
                    initial_bankroll=self.config['bankroll']['initial_amount']
                )

                # Determine risk level based on uncertainty
                if uncertainty < 0.1:
                    risk_level = 'low'
                elif uncertainty < 0.2:
                    risk_level = 'medium'
                else:
                    risk_level = 'high'

                result = SimulationResult(
                    event_id=event.id,
                    iterations=self.config['simulation']['monte_carlo_iterations'],
                    win_probability=Decimal(str(ensemble_prob)),
                    expected_value=Decimal(str(best_ev)),
                    variance=Decimal(str(best_simulation['variance'])),
                    confidence_interval_lower=Decimal(str(ensemble_prob - 2 * uncertainty)),
                    confidence_interval_upper=Decimal(str(ensemble_prob + 2 * uncertainty)),
                    kelly_fraction=Decimal(str(kelly_fraction)),
                    recommended_stake=Decimal('0'),  # Will be calculated later
                    risk_level=risk_level
                )

                logger.info(f"Ensemble simulation completed for {event.id}: EV={best_ev:.4f}, Risk={risk_level}")
                return result

        # Fallback to traditional simulation
        return await self._run_traditional_simulation(event, odds_list)

    def _prepare_event_data(self, event: Event, odds_list: list[Odds]) -> dict:
        """Prepare event data for ensemble prediction."""
        event_data = {
            'home_team': event.home_team,
            'away_team': event.away_team,
            'sport': event.sport.value,
            'event_date': event.event_date.isoformat(),
            'home_rank': 10,  # Default values - would be populated from external data
            'away_rank': 10,
            'home_form': 0.5,
            'away_form': 0.5,
            'home_injuries': 0,
            'away_injuries': 0,
            'home_rest_days': 3,
            'away_rest_days': 3,
            'home_advantage': 0.05,
            'head_to_head_home_wins': 0.5
        }

        # Add odds information
        for odds in odds_list:
            if odds.market_type == MarketType.MONEYLINE:
                if odds.selection == 'home':
                    event_data['home_moneyline'] = float(odds.odds)
                elif odds.selection == 'away':
                    event_data['away_moneyline'] = float(odds.odds)

        return event_data

    async def _run_traditional_simulation(self, event: Event, odds_list: list[Odds]) -> SimulationResult:
        """Run traditional simulation as fallback."""
        # Group odds by market type
        moneyline_odds = [o for o in odds_list if o.market_type == MarketType.MONEYLINE]
        spread_odds = [o for o in odds_list if o.market_type == MarketType.SPREAD]
        totals_odds = [o for o in odds_list if o.market_type == MarketType.TOTALS]

        best_ev = 0.0
        best_odds = None
        best_simulation = None

        # Simulate moneyline markets
        for odds in moneyline_odds:
            if odds.selection == 'home':
                away_odds = next((o for o in moneyline_odds if o.selection == 'away'), None)
                if away_odds:
                    simulation = self.monte_carlo.simulate_moneyline(
                        float(odds.odds), float(away_odds.odds)
                    )

                    if simulation['expected_value'] > best_ev:
                        best_ev = simulation['expected_value']
                        best_odds = odds
                        best_simulation = simulation

        if not best_simulation:
            logger.warning(f"No valid simulation found for event: {event.id}")
            return None

        # Calculate Kelly fraction
        kelly_fraction = KellyCriterion.calculate_kelly_fraction(
            best_simulation['win_probability'],
            float(best_odds.odds),
            self.config['bankroll']['kelly_fraction']
        )

        # Determine risk level
        if best_simulation['variance'] < 0.01:
            risk_level = 'low'
        elif best_simulation['variance'] < 0.05:
            risk_level = 'medium'
        else:
            risk_level = 'high'

        # Create simulation result
        result = SimulationResult(
            event_id=event.id,
            iterations=self.config['simulation']['monte_carlo_iterations'],
            win_probability=Decimal(str(best_simulation['win_probability'])),
            expected_value=Decimal(str(best_ev)),
            variance=Decimal(str(best_simulation['variance'])),
            confidence_interval_lower=Decimal(str(best_simulation['confidence_interval_lower'])),
            confidence_interval_upper=Decimal(str(best_simulation['confidence_interval_upper'])),
            kelly_fraction=Decimal(str(kelly_fraction)),
            recommended_stake=Decimal('0'),  # Will be calculated later
            risk_level=risk_level
        )

        logger.info(f"Traditional simulation completed for {event.id}: EV={best_ev:.4f}, Risk={risk_level}")
        return result

    async def run_portfolio_simulation(self, events: list[Event],
                                     odds_data: list[Odds]) -> list[SimulationResult]:
        """
        Run simulations for multiple events (portfolio view).
        
        Args:
            events: List of events to simulate
            odds_data: List of odds data
        
        Returns:
            List of SimulationResult objects
        """
        logger.info(f"Running portfolio simulation for {len(events)} events")

        results = []
        for event in events:
            event_odds = [o for o in odds_data if o.event_id == event.id]
            if event_odds:
                result = await self.run_event_simulation(event, event_odds)
                if result:
                    results.append(result)

        logger.info(f"Portfolio simulation completed: {len(results)} results")
        return results


class MLPredictor:
    """Machine learning predictor for sports outcomes."""

    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.bias_mitigator = BiasMitigator()
        self.bias_audit_results = {}

        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(random_state=42)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def prepare_features(self, events: list[Event], odds_data: list[Odds]) -> pd.DataFrame:
        """
        Prepare features for ML model training.
        
        Args:
            events: List of events
            odds_data: List of odds data
        
        Returns:
            DataFrame with features
        """
        features = []

        for event in events:
            event_odds = [o for o in odds_data if o.event_id == event.id]

            if not event_odds:
                continue

            # Basic features
            feature_dict = {
                'event_id': event.id,
                'sport': event.sport.value,
                'home_team': event.home_team,
                'away_team': event.away_team,
                'days_until_event': (event.event_date - datetime.utcnow()).days
            }

            # Odds features
            for odds in event_odds:
                if odds.market_type == MarketType.MONEYLINE:
                    if odds.selection == 'home':
                        feature_dict['home_moneyline'] = float(odds.odds)
                        feature_dict['home_implied_prob'] = float(odds.implied_probability or 0)
                    elif odds.selection == 'away':
                        feature_dict['away_moneyline'] = float(odds.odds)
                        feature_dict['away_implied_prob'] = float(odds.implied_probability or 0)

                elif odds.market_type == MarketType.SPREAD:
                    if odds.selection == 'home':
                        feature_dict['home_spread'] = float(odds.line or 0)
                        feature_dict['home_spread_odds'] = float(odds.odds)
                    elif odds.selection == 'away':
                        feature_dict['away_spread'] = float(odds.line or 0)
                        feature_dict['away_spread_odds'] = float(odds.odds)

                elif odds.market_type == MarketType.TOTALS:
                    if odds.selection == 'over':
                        feature_dict['total_over'] = float(odds.line or 0)
                        feature_dict['over_odds'] = float(odds.odds)
                    elif odds.selection == 'under':
                        feature_dict['total_under'] = float(odds.line or 0)
                        feature_dict['under_odds'] = float(odds.odds)

            features.append(feature_dict)

        df = pd.DataFrame(features)

        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        return df

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the ML model.
        
        Args:
            X: Feature matrix
            y: Target variable
        """
        # Store feature names
        self.feature_names = X.columns.tolist()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        logger.info(f"Model trained with CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using trained model.
        
        Args:
            X: Feature matrix
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Ensure same features as training
        X = X[self.feature_names]

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        return predictions, probabilities

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}

        return dict(zip(self.feature_names, self.model.feature_importances_, strict=False))

    def save_model(self, filepath: str):
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("No trained model to save")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = True

        logger.info(f"Model loaded from {filepath}")

    def audit_bias(self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray = None) -> dict[str, Any]:
        """
        Audit model for bias across different groups.
        
        Args:
            X: Feature matrix
            y: Target variable
            groups: Group labels for bias analysis
            
        Returns:
            Dictionary with bias audit results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before bias audit")

        # Make predictions
        predictions, _ = self.predict(X)

        # Detect bias
        bias_metrics = self.bias_mitigator.detect_bias(predictions, y.values, groups)

        # Store audit results
        self.bias_audit_results = {
            'bias_metrics': bias_metrics,
            'audit_timestamp': datetime.utcnow().isoformat(),
            'model_type': self.model_type
        }

        # Check for high bias disparity
        if 'disparity' in bias_metrics and bias_metrics['disparity'] > 0.10:
            logger.warning(f"High bias disparity detected: {bias_metrics['disparity']:.3f}")

        return self.bias_audit_results

    def apply_fairness_corrections(self, X: pd.DataFrame, groups: np.ndarray = None) -> np.ndarray:
        """
        Apply fairness corrections to predictions.
        
        Args:
            X: Feature matrix
            groups: Group labels
            
        Returns:
            Fairness-corrected predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before applying fairness corrections")

        # Make original predictions
        predictions, _ = self.predict(X)

        # Apply fairness corrections if groups provided
        if groups is not None:
            corrected_predictions = self.bias_mitigator.apply_fairness_corrections(
                predictions, groups, 'demographic_parity'
            )
            logger.info("Applied fairness corrections to predictions")
            return corrected_predictions

        return predictions

    def get_bias_report(self) -> dict[str, Any]:
        """
        Get comprehensive bias report.
        
        Returns:
            Dictionary with bias analysis results
        """
        return {
            'bias_audit_results': self.bias_audit_results,
            'bias_mitigator_report': self.bias_mitigator.get_bias_report(),
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }


class StatisticalAnalyzer:
    """Statistical analysis for betting data."""

    @staticmethod
    def calculate_volatility(returns: list[float], window: int = 30) -> float:
        """
        Calculate volatility using rolling standard deviation.
        
        Args:
            returns: List of return values
            window: Rolling window size
        
        Returns:
            Volatility measure
        """
        if len(returns) < window:
            return np.std(returns)

        returns_array = np.array(returns)
        rolling_std = pd.Series(returns_array).rolling(window=window).std()
        return float(rolling_std.iloc[-1])

    @staticmethod
    def calculate_sharpe_ratio(returns: list[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: List of return values
            risk_free_rate: Risk-free rate (annual)
        
        Returns:
            Sharpe ratio
        """
        if not returns:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate

        if np.std(excess_returns) == 0:
            return 0.0

        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))

    @staticmethod
    def calculate_var(returns: list[float], confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: List of return values
            confidence_level: Confidence level for VaR
        
        Returns:
            VaR value
        """
        if not returns:
            return 0.0

        returns_array = np.array(returns)
        return float(np.percentile(returns_array, (1 - confidence_level) * 100))

    @staticmethod
    def test_stationarity(time_series: list[float]) -> dict[str, float]:
        """
        Test time series stationarity using Augmented Dickey-Fuller test.
        
        Args:
            time_series: Time series data
        
        Returns:
            Dictionary with test results
        """
        if len(time_series) < 2:
            return {'is_stationary': False, 'p_value': 1.0, 'test_statistic': 0.0}

        result = adfuller(time_series)

        return {
            'is_stationary': result[1] < 0.05,
            'p_value': result[1],
            'test_statistic': result[0],
            'critical_values': result[4]
        }


class EnsemblePredictor:
    """Cutting-edge ensemble prediction system combining neural networks, Bayesian networks, and GNNs."""

    def __init__(self, config: dict):
        self.config = config

        # Initialize device and models based on availability
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.nn_model = None
            self.gnn_model = None
        else:
            self.device = None
            self.nn_model = None
            self.gnn_model = None

        if PYMC_AVAILABLE:
            self.bayes_model = None
        else:
            self.bayes_model = None

        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_model = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                self.sentiment_model = None
        else:
            self.sentiment_model = None

        self.ensemble_weights = {'nn': 0.4, 'bayes': 0.3, 'gnn': 0.3}

    def build_neural_network(self, input_dim: int, hidden_dims: list[int] = [128, 64, 32]):
        """Build neural network for pattern recognition."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, neural network building disabled")
            return None

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def build_graph_neural_network(self, node_dim: int, hidden_dim: int = 64):
        """Build graph neural network for player/team interconnections."""
        if not TORCH_AVAILABLE or not DGL_AVAILABLE:
            logger.warning("PyTorch or DGL not available, GNN building disabled")
            return None

        class GNNModel(nn.Module):
            def __init__(self, node_dim, hidden_dim):
                super().__init__()
                self.conv1 = dglnn.GraphConv(node_dim, hidden_dim)
                self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
                self.conv3 = dglnn.GraphConv(hidden_dim, 1)
                self.dropout = nn.Dropout(0.2)

            def forward(self, g, features):
                h = F.relu(self.conv1(g, features))
                h = self.dropout(h)
                h = F.relu(self.conv2(g, h))
                h = self.dropout(h)
                h = self.conv3(g, h)
                return torch.sigmoid(h)

        return GNNModel(node_dim, hidden_dim)

    def create_player_graph(self, event_data: dict, biometric_data: dict = None):
        """Create graph representation of players and teams."""
        if not DGL_AVAILABLE:
            logger.warning("DGL not available, graph creation disabled")
            return None

        G = dgl.DGLGraph()

        # Add nodes for players and teams
        players = event_data.get('players', [])
        teams = event_data.get('teams', [])

        # Create player nodes
        for i, player in enumerate(players):
            G.add_nodes(1, {'features': torch.tensor([
                player.get('age', 25),
                player.get('experience', 5),
                player.get('avg_performance', 0.5),
                player.get('injury_risk', 0.1),
                biometric_data.get(f"player_{player['id']}_heart_rate", 70) if biometric_data else 70,
                biometric_data.get(f"player_{player['id']}_fatigue", 0.3) if biometric_data else 0.3
            ], dtype=torch.float32)})

        # Create team nodes
        for i, team in enumerate(teams):
            G.add_nodes(1, {'features': torch.tensor([
                team.get('rank', 10),
                team.get('home_advantage', 0.05),
                team.get('recent_form', 0.5),
                team.get('injury_count', 0),
                team.get('rest_days', 3)
            ], dtype=torch.float32)})

        # Add edges (player-team relationships, player-player interactions)
        # This is a simplified version - in practice, you'd have more complex relationships
        for i, player in enumerate(players):
            team_id = player.get('team_id', 0)
            G.add_edge(i, len(players) + team_id)

        return G

    def bayesian_uncertainty(self, event_data: dict) -> tuple[float, float]:
        """Bayesian network for uncertainty quantification."""
        if not PYMC_AVAILABLE:
            logger.warning("PyMC not available, using simplified uncertainty estimation")
            # Fallback to simple uncertainty estimation
            home_form = event_data.get('home_form', 0.5)
            away_form = event_data.get('away_form', 0.5)
            home_injuries = event_data.get('home_injuries', 0)
            away_injuries = event_data.get('away_injuries', 0)

            # Simple probability calculation
            win_prob = 0.5 + 0.1 * (home_form - away_form) - 0.05 * (home_injuries - away_injuries)
            uncertainty = 0.1 + 0.05 * abs(home_form - away_form)

            return float(np.clip(win_prob, 0.1, 0.9)), float(uncertainty)

        with pm.Model() as model:
            # Prior distributions
            home_advantage = pm.Normal('home_advantage', mu=0.05, sigma=0.02)
            form_weight = pm.Normal('form_weight', mu=0.3, sigma=0.1)
            injury_penalty = pm.Normal('injury_penalty', mu=-0.1, sigma=0.05)

            # Likelihood
            home_form = event_data.get('home_form', 0.5)
            away_form = event_data.get('away_form', 0.5)
            home_injuries = event_data.get('home_injuries', 0)
            away_injuries = event_data.get('away_injuries', 0)

            win_prob = pm.Deterministic('win_prob',
                pm.math.sigmoid(
                    home_advantage +
                    form_weight * (home_form - away_form) +
                    injury_penalty * (home_injuries - away_injuries)
                )
            )

            # Sample from posterior
            trace = pm.sample(1000, return_inferencedata=False)

            # Return mean and uncertainty
            return float(np.mean(trace['win_prob'])), float(np.std(trace['win_prob']))

    def process_biometric_data(self, biometric_data: dict) -> dict[str, float]:
        """Process real-time biometric data for fatigue and performance metrics."""
        if not BIOSIGNAL_AVAILABLE:
            logger.warning("Biosignal processing libraries not available, using simplified processing")
            processed_data = {}

            for player_id, data in biometric_data.items():
                if 'heart_rate' in data:
                    hr_signal = np.array(data['heart_rate'])
                    processed_data[f"{player_id}_hr_mean"] = float(np.mean(hr_signal))
                    processed_data[f"{player_id}_hr_std"] = float(np.std(hr_signal))
                    processed_data[f"{player_id}_fatigue"] = self._calculate_fatigue_simple(hr_signal)

                if 'movement' in data:
                    movement_signal = np.array(data['movement'])
                    processed_data[f"{player_id}_activity"] = float(np.mean(np.abs(movement_signal)))
                    processed_data[f"{player_id}_intensity"] = float(np.std(movement_signal))

            return processed_data

        processed_data = {}

        for player_id, data in biometric_data.items():
            if 'heart_rate' in data:
                # Process heart rate data
                hr_signal = np.array(data['heart_rate'])
                hr_processed = hp.process(hr_signal, sample_rate=100)

                processed_data[f"{player_id}_hr_mean"] = hr_processed['hr_mean']
                processed_data[f"{player_id}_hr_std"] = hr_processed['hr_std']
                processed_data[f"{player_id}_fatigue"] = self._calculate_fatigue(hr_processed)

            if 'movement' in data:
                # Process movement/acceleration data
                movement_signal = np.array(data['movement'])
                processed_data[f"{player_id}_activity"] = np.mean(np.abs(movement_signal))
                processed_data[f"{player_id}_intensity"] = np.std(movement_signal)

        return processed_data

    def _calculate_fatigue(self, hr_data: dict) -> float:
        """Calculate fatigue level from heart rate data."""
        # Simplified fatigue calculation based on HR variability
        if 'hr_mean' in hr_data and 'hr_std' in hr_data:
            hr_mean = hr_data['hr_mean']
            hr_std = hr_data['hr_std']

            # Higher HR with low variability suggests fatigue
            if hr_mean > 160 and hr_std < 5:
                return 0.8
            elif hr_mean > 140 and hr_std < 8:
                return 0.6
            elif hr_mean > 120:
                return 0.4
            else:
                return 0.2

        return 0.3  # Default moderate fatigue

    def _calculate_fatigue_simple(self, hr_signal: np.ndarray) -> float:
        """Simplified fatigue calculation based on HR variability."""
        if len(hr_signal) > 0:
            hr_mean = np.mean(hr_signal)
            hr_std = np.std(hr_signal)

            # Higher HR with low variability suggests fatigue
            if hr_mean > 160 and hr_std < 5:
                return 0.8
            elif hr_mean > 140 and hr_std < 8:
                return 0.6
            elif hr_mean > 120:
                return 0.4
            else:
                return 0.2

        return 0.3  # Default moderate fatigue

    def analyze_sentiment(self, text_data: list[str]) -> float:
        """Analyze sentiment from social media/news feeds."""
        if not text_data:
            return 0.5  # Neutral sentiment

        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, using simple sentiment analysis")
            # Simple keyword-based sentiment analysis
            positive_words = ['win', 'victory', 'great', 'amazing', 'excellent', 'strong']
            negative_words = ['loss', 'defeat', 'terrible', 'awful', 'weak', 'poor']

            sentiment_scores = []
            for text in text_data:
                text_lower = text.lower()
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)

                if positive_count > negative_count:
                    sentiment_scores.append(0.7)
                elif negative_count > positive_count:
                    sentiment_scores.append(0.3)
                else:
                    sentiment_scores.append(0.5)

            return np.mean(sentiment_scores)

        embeddings = self.sentiment_model.encode(text_data)
        sentiment_scores = []

        for text in text_data:
            # Use VADER sentiment analysis for sports-specific sentiment
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)
            sentiment_scores.append(scores['compound'])

        return np.mean(sentiment_scores)

    def ensemble_prediction(self, event_data: dict, biometric_data: dict = None,
                          sentiment_data: list[str] = None) -> tuple[float, float]:
        """
        Ensemble prediction combining neural networks, Bayesian networks, and GNNs.
        
        Returns:
            Tuple of (predicted_probability, uncertainty)
        """
        try:
            # 1. Neural Network Prediction
            nn_features = self._extract_nn_features(event_data, biometric_data)
            if self.nn_model is None:
                self.nn_model = self.build_neural_network(len(nn_features))

            nn_input = torch.tensor(nn_features, dtype=torch.float32).unsqueeze(0)
            nn_prob = self.nn_model(nn_input).item()

            # 2. Bayesian Network Prediction
            bayes_prob, bayes_uncertainty = self.bayesian_uncertainty(event_data)

            # 3. Graph Neural Network Prediction
            if biometric_data:
                G = self.create_player_graph(event_data, biometric_data)
                if self.gnn_model is None:
                    node_dim = G.ndata['features'].shape[1]
                    self.gnn_model = self.build_graph_neural_network(node_dim)

                gnn_output = self.gnn_model(G, G.ndata['features'])
                gnn_prob = torch.mean(gnn_output).item()
            else:
                gnn_prob = 0.5  # Default if no biometric data

            # 4. Sentiment Analysis
            sentiment_score = 0.5  # Neutral default
            if sentiment_data:
                sentiment_score = self.analyze_sentiment(sentiment_data)
                # Adjust probabilities based on sentiment
                sentiment_adjustment = (sentiment_score - 0.5) * 0.1
                nn_prob += sentiment_adjustment
                bayes_prob += sentiment_adjustment
                gnn_prob += sentiment_adjustment

            # 5. Ensemble Combination
            ensemble_prob = (
                self.ensemble_weights['nn'] * nn_prob +
                self.ensemble_weights['bayes'] * bayes_prob +
                self.ensemble_weights['gnn'] * gnn_prob
            )

            # 6. Uncertainty Quantification
            uncertainties = [bayes_uncertainty, abs(nn_prob - ensemble_prob), abs(gnn_prob - ensemble_prob)]
            ensemble_uncertainty = np.mean(uncertainties)

            # Ensure probabilities are in valid range
            ensemble_prob = np.clip(ensemble_prob, 0.01, 0.99)

            logger.info(f"Ensemble prediction: {ensemble_prob:.3f} ± {ensemble_uncertainty:.3f}")

            return ensemble_prob, ensemble_uncertainty

        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return 0.5, 0.2  # Default fallback

    def _extract_nn_features(self, event_data: dict, biometric_data: dict = None) -> list[float]:
        """Extract features for neural network."""
        features = []

        # Basic event features
        features.extend([
            event_data.get('home_rank', 10),
            event_data.get('away_rank', 10),
            event_data.get('home_form', 0.5),
            event_data.get('away_form', 0.5),
            event_data.get('home_injuries', 0),
            event_data.get('away_injuries', 0),
            event_data.get('home_rest_days', 3),
            event_data.get('away_rest_days', 3),
            event_data.get('home_advantage', 0.05),
            event_data.get('head_to_head_home_wins', 0.5)
        ])

        # Biometric features
        if biometric_data:
            home_fatigue = np.mean([v for k, v in biometric_data.items() if 'home' in k and 'fatigue' in k])
            away_fatigue = np.mean([v for k, v in biometric_data.items() if 'away' in k and 'fatigue' in k])
            features.extend([home_fatigue, away_fatigue])
        else:
            features.extend([0.3, 0.3])  # Default fatigue

        return features

    def uncertainty_error_bars(self, prediction: float, uncertainty: float) -> dict[str, float]:
        """Calculate error bars for prediction uncertainty."""
        return {
            'lower_bound': max(0.01, prediction - 2 * uncertainty),
            'upper_bound': min(0.99, prediction + 2 * uncertainty),
            'confidence_68': uncertainty,
            'confidence_95': 2 * uncertainty
        }


class BiasMitigator:
    """Bias detection and mitigation for sports betting predictions."""

    def __init__(self, n_components: int = 5):
        """
        Initialize bias mitigator.
        
        Args:
            n_components: Number of components for Gaussian Mixture Model
        """
        self.gmm = GaussianMixture(n_components=n_components, random_state=42)
        self.park_factors = {}
        self.position_adjustments = {}
        self.survival_rates = {}
        self.bias_scores = {}

    def adjust_park_effects(self, player_stats: np.ndarray, park_factor: float = 1.15,
                          park_name: str = None) -> np.ndarray:
        """
        Adjust for environmental bias (e.g., Fenway inflating right-handed stats).
        
        Args:
            player_stats: Player statistics array
            park_factor: Park adjustment factor (default 1.15 for 15% inflation)
            park_name: Name of the park for logging
            
        Returns:
            Adjusted player statistics
        """
        if park_name:
            self.park_factors[park_name] = park_factor
            logger.info(f"Applied park factor {park_factor} for {park_name}")

        # Simple normalization; extend with hierarchical Bayesian model
        adjusted_stats = player_stats * (1 / park_factor)

        return adjusted_stats

    def generate_synthetic_data(self, historical_data: np.ndarray, n_samples: int = 1000) -> np.ndarray:
        """
        Mitigate historical biases with synthetic scenarios.
        
        Args:
            historical_data: Historical performance data
            n_samples: Number of synthetic samples to generate
            
        Returns:
            Synthetic data array
        """
        try:
            # Fit GMM to historical data
            self.gmm.fit(historical_data)

            # Generate synthetic samples
            synthetic, _ = self.gmm.sample(n_samples)

            logger.info(f"Generated {n_samples} synthetic samples using GMM")
            return synthetic

        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            return historical_data

    def correct_survivorship_bias(self, aging_curve: np.ndarray, survival_rate: float = 0.9,
                                 age_group: str = None) -> np.ndarray:
        """
        Adjust for age bias in projections.
        
        Args:
            aging_curve: Aging curve data
            survival_rate: Survival rate for the age group (default 0.9)
            age_group: Age group identifier for logging
            
        Returns:
            Corrected aging curve
        """
        if age_group:
            self.survival_rates[age_group] = survival_rate

        # Incorporate inverse probability weighting
        corrected_curve = aging_curve / survival_rate

        logger.info(f"Applied survivorship bias correction with rate {survival_rate}")
        return corrected_curve

    def adjust_position_metrics(self, player_stats: np.ndarray, position: str,
                              adjustment_factor: float = 1.25) -> np.ndarray:
        """
        Adjust metrics for undervalued positions.
        
        Args:
            player_stats: Player statistics
            position: Player position
            adjustment_factor: Adjustment factor (default 1.25 for 25% undervaluation)
            
        Returns:
            Position-adjusted statistics
        """
        self.position_adjustments[position] = adjustment_factor

        adjusted_stats = player_stats * adjustment_factor

        logger.info(f"Applied position adjustment {adjustment_factor} for {position}")
        return adjusted_stats

    def detect_bias(self, predictions: np.ndarray, actuals: np.ndarray,
                   groups: np.ndarray = None) -> dict[str, float]:
        """
        Detect bias in predictions across different groups.
        
        Args:
            predictions: Model predictions
            actuals: Actual outcomes
            groups: Group labels for bias analysis
            
        Returns:
            Dictionary with bias metrics
        """
        bias_metrics = {}

        # Overall prediction bias
        overall_bias = np.mean(predictions - actuals)
        bias_metrics['overall_bias'] = float(overall_bias)

        # Group-specific bias if groups provided
        if groups is not None:
            unique_groups = np.unique(groups)
            group_biases = {}

            for group in unique_groups:
                group_mask = groups == group
                group_bias = np.mean(predictions[group_mask] - actuals[group_mask])
                group_biases[str(group)] = float(group_bias)

            bias_metrics['group_biases'] = group_biases

            # Calculate disparity
            if len(unique_groups) > 1:
                max_bias = max(abs(bias) for bias in group_biases.values())
                min_bias = min(abs(bias) for bias in group_biases.values())
                disparity = max_bias - min_bias
                bias_metrics['disparity'] = float(disparity)

                # Flag if disparity > 10%
                if disparity > 0.10:
                    logger.warning(f"High bias disparity detected: {disparity:.3f}")

        self.bias_scores = bias_metrics
        return bias_metrics

    def apply_fairness_corrections(self, predictions: np.ndarray, groups: np.ndarray,
                                 target_fairness: str = 'demographic_parity') -> np.ndarray:
        """
        Apply fairness corrections to predictions.
        
        Args:
            predictions: Original predictions
            groups: Group labels
            target_fairness: Target fairness metric
            
        Returns:
            Fairness-corrected predictions
        """
        try:
            # Simple post-processing correction
            unique_groups = np.unique(groups)
            corrected_predictions = predictions.astype(float).copy()

            # Calculate group means
            group_means = {}
            for group in unique_groups:
                group_mask = groups == group
                group_means[group] = np.mean(predictions[group_mask])

            # Apply correction to equalize group means
            overall_mean = np.mean(predictions)
            for group in unique_groups:
                group_mask = groups == group
                group_mean = group_means[group]
                correction = overall_mean - group_mean
                corrected_predictions[group_mask] += correction

            logger.info(f"Applied fairness correction for {target_fairness}")
            return corrected_predictions

        except Exception as e:
            logger.error(f"Error applying fairness corrections: {e}")
            return predictions

    def get_bias_report(self) -> dict[str, Any]:
        """
        Generate comprehensive bias report.
        
        Returns:
            Dictionary with bias analysis results
        """
        return {
            'park_factors': self.park_factors,
            'position_adjustments': self.position_adjustments,
            'survival_rates': self.survival_rates,
            'bias_scores': self.bias_scores,
            'total_adjustments': len(self.park_factors) + len(self.position_adjustments) + len(self.survival_rates)
        }
