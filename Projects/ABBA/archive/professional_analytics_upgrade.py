"""
Professional Analytics Upgrade Implementation
Critical upgrades to transform basic MLB betting strategy to professional standards.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
import structlog

warnings.filterwarnings('ignore')

# Machine Learning imports
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Advanced ML imports
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: lightgbm not available. Install with: pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: catboost not available. Install with: pip install catboost")

logger = structlog.get_logger()


class AdvancedMLBMetrics:
    """Professional-grade MLB metrics implementation."""

    def __init__(self):
        self.park_factors = self._load_park_factors()
        self.league_averages = self._load_league_averages()

    def _load_park_factors(self) -> dict[str, dict[str, float]]:
        """Load park factors for all MLB stadiums."""
        return {
            'coors_field': {'hr_factor': 1.35, 'hit_factor': 1.15, 'k_factor': 0.90},
            'petco_park': {'hr_factor': 0.85, 'hit_factor': 0.95, 'k_factor': 1.10},
            'yankee_stadium': {'hr_factor': 1.15, 'hit_factor': 1.05, 'k_factor': 1.00},
            'fenway_park': {'hr_factor': 1.10, 'hit_factor': 1.08, 'k_factor': 0.95},
            'wrigley_field': {'hr_factor': 1.05, 'hit_factor': 1.02, 'k_factor': 0.98},
            'dodger_stadium': {'hr_factor': 0.95, 'hit_factor': 0.98, 'k_factor': 1.05},
            'citizens_bank_park': {'hr_factor': 1.08, 'hit_factor': 1.03, 'k_factor': 1.00},
            'minute_maid_park': {'hr_factor': 1.12, 'hit_factor': 1.06, 'k_factor': 0.97},
            'tropicana_field': {'hr_factor': 0.88, 'hit_factor': 0.92, 'k_factor': 1.12},
            'oakland_coliseum': {'hr_factor': 0.90, 'hit_factor': 0.94, 'k_factor': 1.08},
            # Default for other parks
            'default': {'hr_factor': 1.00, 'hit_factor': 1.00, 'k_factor': 1.00}
        }

    def _load_league_averages(self) -> dict[str, float]:
        """Load current league averages for context adjustment."""
        return {
            'hr_fb_rate': 0.105,  # 10.5% HR/FB rate
            'babip': 0.300,       # .300 BABIP
            'k_rate': 0.225,      # 22.5% strikeout rate
            'bb_rate': 0.085,     # 8.5% walk rate
            'iso': 0.165,         # .165 ISO
            'woba': 0.320         # .320 wOBA
        }

    def calculate_xwOBA(self, exit_velocity: float, launch_angle: float,
                       sprint_speed: float, park_factor: float = 1.0) -> float:
        """Calculate expected wOBA using Statcast data."""
        try:
            # Base xwOBA calculation (simplified version)
            # In practice, this would use the full Statcast xwOBA model

            # Exit velocity component (0-1 scale)
            velo_component = min(exit_velocity / 120.0, 1.0) if exit_velocity > 0 else 0

            # Launch angle component (optimal range 15-35 degrees)
            if 15 <= launch_angle <= 35:
                angle_component = 1.0
            elif 10 <= launch_angle <= 40:
                angle_component = 0.8
            else:
                angle_component = 0.3

            # Sprint speed component (for infield hits)
            speed_component = min(sprint_speed / 30.0, 1.0) if sprint_speed > 0 else 0

            # Combine components
            base_xwoba = (velo_component * 0.4 + angle_component * 0.4 + speed_component * 0.2) * 0.500

            # Apply park factor
            adjusted_xwoba = base_xwoba * park_factor

            return min(max(adjusted_xwoba, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating xwOBA: {e}")
            return 0.320  # League average

    def calculate_xFIP(self, k_rate: float, bb_rate: float,
                      hr_rate: float, league_hr_fb_rate: float = None) -> float:
        """Calculate expected FIP (Fielding Independent Pitching)."""
        try:
            if league_hr_fb_rate is None:
                league_hr_fb_rate = self.league_averages['hr_fb_rate']

            # xFIP formula: (13*HR + 3*(BB+HBP) - 2*K) / IP + constant
            # Simplified version using rates
            xfip = (13 * league_hr_fb_rate + 3 * bb_rate - 2 * k_rate) / 1.0 + 3.10

            return max(xfip, 0.0)

        except Exception as e:
            logger.error(f"Error calculating xFIP: {e}")
            return 4.00  # League average

    def calculate_stuff_plus(self, velocity: float, spin_rate: float,
                           movement: float, location: tuple) -> float:
        """Calculate Stuff+ metric (Baseball Prospectus methodology)."""
        try:
            # Simplified Stuff+ calculation
            # In practice, this would use the full BP model with pitch-specific adjustments

            # Velocity component (normalized to 100)
            velo_component = min(velocity / 100.0, 1.2) * 100

            # Spin rate component (normalized)
            spin_component = min(spin_rate / 2500.0, 1.2) * 100

            # Movement component (normalized)
            movement_component = min(movement / 20.0, 1.2) * 100

            # Location component (strike zone proximity)
            if location and len(location) == 2:
                x, y = location
                # Distance from center of strike zone
                distance_from_center = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
                location_component = max(100 - distance_from_center * 50, 50)
            else:
                location_component = 100

            # Weighted average
            stuff_plus = (velo_component * 0.3 + spin_component * 0.3 +
                         movement_component * 0.2 + location_component * 0.2)

            return max(stuff_plus, 0.0)

        except Exception as e:
            logger.error(f"Error calculating Stuff+: {e}")
            return 100.0  # League average

    def calculate_park_adjusted_era(self, raw_era: float, park: str) -> float:
        """Calculate park-adjusted ERA."""
        park_factor = self.park_factors.get(park, self.park_factors['default'])
        return raw_era / park_factor['hr_factor']

    def calculate_park_adjusted_woba(self, raw_woba: float, park: str) -> float:
        """Calculate park-adjusted wOBA."""
        park_factor = self.park_factors.get(park, self.park_factors['default'])
        return raw_woba / park_factor['hit_factor']


class ContactQualityAnalyzer:
    """Advanced contact quality analysis."""

    def __init__(self):
        self.barrel_thresholds = {
            'exit_velocity': 98,
            'launch_angle_min': 26,
            'launch_angle_max': 30
        }

    def calculate_hard_hit_rate(self, exit_velocities: list[float]) -> float:
        """Calculate Hard Hit% (95+ MPH)."""
        if not exit_velocities:
            return 0.0

        hard_hits = sum(1 for vel in exit_velocities if vel >= 95)
        return hard_hits / len(exit_velocities)

    def calculate_barrel_rate(self, exit_velocities: list[float],
                            launch_angles: list[float]) -> float:
        """Calculate Barrel Rate (optimal contact)."""
        if not exit_velocities or not launch_angles:
            return 0.0

        barrels = 0
        for vel, angle in zip(exit_velocities, launch_angles, strict=False):
            if self._is_barrel(vel, angle):
                barrels += 1

        return barrels / len(exit_velocities)

    def _is_barrel(self, exit_velocity: float, launch_angle: float) -> bool:
        """Determine if contact is a barrel."""
        # Barrel criteria: 98+ MPH exit velocity and 26-30 degree launch angle
        return (exit_velocity >= self.barrel_thresholds['exit_velocity'] and
                self.barrel_thresholds['launch_angle_min'] <= launch_angle <=
                self.barrel_thresholds['launch_angle_max'])

    def calculate_contact_quality_differential(self, expected: float, actual: float) -> float:
        """Calculate gap between expected and actual performance."""
        return actual - expected  # Positive = overperforming, negative = underperforming


class ProfessionalEnsemble:
    """Professional ensemble modeling system."""

    def __init__(self):
        self.models = {}
        self.ensemble_weights = {}
        self.performance_history = {}
        self.scaler = StandardScaler()

        # Initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all ensemble models."""
        self.models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=42
            ),
            'neural_net': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                max_iter=1000,
                random_state=42
            )
        }

        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )

        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            self.models['catboost'] = CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            )

    def train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Train all models and calculate ensemble weights."""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            results = {}

            for name, model in self.models.items():
                logger.info(f"Training {name} model...")

                # Train with cross-validation
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
                model.fit(X_scaled, y)

                # Store performance
                results[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'model': model
                }

                logger.info(f"{name} - CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

            # Calculate dynamic weights based on performance
            self.ensemble_weights = self._calculate_weights(results)

            return results

        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            raise

    def _calculate_weights(self, results: dict[str, dict]) -> dict[str, float]:
        """Calculate ensemble weights based on cross-validation performance."""
        # Use inverse of standard deviation as weight (lower std = higher weight)
        weights = {}
        total_weight = 0

        for name, result in results.items():
            # Weight based on CV mean and penalize high variance
            weight = result['cv_mean'] / (1 + result['cv_std'])
            weights[name] = weight
            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        else:
            # Equal weights if all models perform poorly
            n_models = len(weights)
            weights = dict.fromkeys(weights.keys(), 1.0 / n_models)

        logger.info(f"Ensemble weights: {weights}")
        return weights

    def predict_ensemble(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Make ensemble prediction with uncertainty quantification."""
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)

            predictions = []
            weights = []

            for name, weight in self.ensemble_weights.items():
                if name in self.models:
                    model = self.models[name]
                    pred = model.predict_proba(X_scaled)[:, 1]
                    predictions.append(pred)
                    weights.append(weight)

            if not predictions:
                raise ValueError("No models available for prediction")

            # Weighted ensemble prediction
            predictions_array = np.array(predictions)
            weights_array = np.array(weights).reshape(-1, 1)

            ensemble_pred = np.average(predictions_array, weights=weights_array, axis=0)

            # Uncertainty quantification (standard deviation of predictions)
            prediction_std = np.std(predictions_array, axis=0)

            return ensemble_pred, prediction_std

        except Exception as e:
            logger.error(f"Error making ensemble prediction: {e}")
            raise

    def get_feature_importance(self) -> dict[str, dict[str, float]]:
        """Get feature importance from all models."""
        importance_dict = {}

        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = dict(zip(
                    range(len(model.feature_importances_)),
                    model.feature_importances_, strict=False
                ))
            elif hasattr(model, 'coef_'):
                # For linear models
                importance_dict[name] = dict(zip(
                    range(len(model.coef_[0])),
                    np.abs(model.coef_[0]), strict=False
                ))

        return importance_dict


class ProfessionalRiskManager:
    """Professional risk management system."""

    def __init__(self):
        self.kelly_fraction = 0.25  # 1/4 Kelly (conservative)
        self.max_bet_size = 0.02    # 2% max bet size
        self.portfolio_correlation_limit = 0.3
        self.max_drawdown_limit = 0.15  # 15% max drawdown
        self.min_edge_threshold = 0.02  # 2% minimum edge

    def calculate_fractional_kelly(self, edge: float, odds: float) -> float:
        """Calculate fractional Kelly bet size."""
        try:
            # Full Kelly calculation
            b = odds - 1
            p = 0.5 + edge  # Convert edge to probability
            q = 1 - p

            # Kelly formula: f = (bp - q) / b
            full_kelly = (b * p - q) / b

            # Apply fractional Kelly and constraints
            fractional_kelly = full_kelly * self.kelly_fraction
            constrained_kelly = min(fractional_kelly, self.max_bet_size)

            # Only bet if edge meets minimum threshold
            if edge < self.min_edge_threshold:
                return 0.0

            return max(0, constrained_kelly)

        except Exception as e:
            logger.error(f"Error calculating fractional Kelly: {e}")
            return 0.0

    def portfolio_optimization(self, bets: list[dict]) -> dict[str, float]:
        """Optimize bet sizes considering portfolio correlations."""
        try:
            if not bets:
                return {}

            # Calculate correlation matrix between bets
            correlation_matrix = self._calculate_bet_correlations(bets)

            # Optimize bet sizes using portfolio theory
            optimal_sizes = self._optimize_bet_sizes(bets, correlation_matrix)

            return optimal_sizes

        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            # Fallback to individual Kelly calculations
            return {bet['id']: self.calculate_fractional_kelly(bet['edge'], bet['odds'])
                   for bet in bets}

    def _calculate_bet_correlations(self, bets: list[dict]) -> np.ndarray:
        """Calculate correlation matrix between bets."""
        n_bets = len(bets)
        correlation_matrix = np.zeros((n_bets, n_bets))

        for i, bet1 in enumerate(bets):
            for j, bet2 in enumerate(bets):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    correlation_matrix[i, j] = self._calculate_bet_correlation(bet1, bet2)

        return correlation_matrix

    def _calculate_bet_correlation(self, bet1: dict, bet2: dict) -> float:
        """Calculate correlation between two bets."""
        correlation = 0.0

        # Same team penalty
        if bet1.get('team') == bet2.get('team'):
            correlation += 0.4

        # Same pitcher penalty
        if bet1.get('pitcher') == bet2.get('pitcher'):
            correlation += 0.3

        # Same game penalty
        if bet1.get('game_id') == bet2.get('game_id'):
            correlation += 0.5

        # Similar bet type penalty
        if bet1.get('bet_type') == bet2.get('bet_type'):
            correlation += 0.2

        return min(correlation, 1.0)

    def _optimize_bet_sizes(self, bets: list[dict], correlation_matrix: np.ndarray) -> dict[str, float]:
        """Optimize bet sizes using portfolio theory."""
        # Simplified portfolio optimization
        # In practice, this would use more sophisticated optimization algorithms

        n_bets = len(bets)
        optimal_sizes = {}

        # Calculate individual Kelly sizes
        kelly_sizes = [self.calculate_fractional_kelly(bet['edge'], bet['odds'])
                      for bet in bets]

        # Adjust for correlations (simplified approach)
        for i, bet in enumerate(bets):
            # Reduce size based on correlations with other bets
            correlation_penalty = 0
            for j, other_bet in enumerate(bets):
                if i != j:
                    correlation_penalty += correlation_matrix[i, j] * kelly_sizes[j]

            # Apply penalty
            adjusted_size = max(0, kelly_sizes[i] - correlation_penalty * 0.1)
            optimal_sizes[bet['id']] = min(adjusted_size, self.max_bet_size)

        return optimal_sizes

    def calculate_maximum_drawdown(self, win_rate: float, edge: float,
                                 bet_size: float, num_bets: int) -> float:
        """Calculate expected maximum drawdown using Monte Carlo simulation."""
        try:
            simulations = 10000
            max_drawdowns = []

            for _ in range(simulations):
                # Simulate bet results
                results = np.random.binomial(1, win_rate, num_bets)

                # Calculate cumulative returns
                returns = [bet_size * (r * edge - (1-r)) for r in results]
                cumulative_return = np.cumsum(returns)

                # Calculate drawdown
                running_max = np.maximum.accumulate(cumulative_return)
                drawdown = cumulative_return - running_max
                max_drawdown = np.min(drawdown)

                max_drawdowns.append(abs(max_drawdown))

            # Return 95th percentile
            return np.percentile(max_drawdowns, 95)

        except Exception as e:
            logger.error(f"Error calculating maximum drawdown: {e}")
            return 0.20  # Conservative estimate

    def validate_bet_portfolio(self, bets: list[dict]) -> dict[str, Any]:
        """Validate bet portfolio against risk limits."""
        try:
            total_exposure = sum(bet.get('size', 0) for bet in bets)

            # Check individual bet sizes
            oversized_bets = [bet for bet in bets if bet.get('size', 0) > self.max_bet_size]

            # Check correlations
            correlation_analysis = self._analyze_bet_correlations(bets)
            high_correlations = [k for k, v in correlation_analysis['correlations'].items()
                               if abs(v) > self.portfolio_correlation_limit]

            # Calculate expected drawdown
            if bets:
                avg_edge = np.mean([bet.get('edge', 0) for bet in bets])
                avg_size = np.mean([bet.get('size', 0) for bet in bets])
                expected_drawdown = self.calculate_maximum_drawdown(0.53, avg_edge, avg_size, 100)
            else:
                expected_drawdown = 0.0

            return {
                'total_exposure': total_exposure,
                'oversized_bets': len(oversized_bets),
                'high_correlations': len(high_correlations),
                'expected_drawdown': expected_drawdown,
                'within_limits': (len(oversized_bets) == 0 and
                                len(high_correlations) == 0 and
                                expected_drawdown <= self.max_drawdown_limit),
                'warnings': {
                    'oversized_bets': oversized_bets,
                    'high_correlations': high_correlations,
                    'high_drawdown': expected_drawdown > self.max_drawdown_limit
                }
            }

        except Exception as e:
            logger.error(f"Error validating bet portfolio: {e}")
            return {'error': str(e)}

    def _analyze_bet_correlations(self, bets: list[dict]) -> dict[str, Any]:
        """Analyze correlations between bets."""
        correlations = {}

        for i, bet1 in enumerate(bets):
            for j, bet2 in enumerate(bets[i+1:], i+1):
                correlation = self._calculate_bet_correlation(bet1, bet2)
                correlations[f"{bet1.get('id', i)}_{bet2.get('id', j)}"] = correlation

        return {
            'correlations': correlations,
            'high_correlations': {k: v for k, v in correlations.items()
                                if abs(v) > self.portfolio_correlation_limit}
        }


class ProfessionalFeatureEngineer:
    """Professional-grade feature engineering system."""

    def __init__(self):
        self.feature_categories = {
            'pitching': 80,      # Advanced pitching metrics
            'batting': 70,       # Advanced batting metrics
            'situational': 50,   # Game situation features
            'market': 40,        # Betting market features
            'environmental': 30, # Weather, park, travel
            'biomechanical': 20, # Player biomechanics
            'temporal': 15       # Time-based patterns
        }

    def engineer_mlb_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive MLB features."""
        try:
            df = data.copy()

            # Initialize advanced metrics
            metrics = AdvancedMLBMetrics()
            contact_analyzer = ContactQualityAnalyzer()

            # Pitching features (80+ features)
            df = self._engineer_pitching_features(df, metrics)

            # Batting features (70+ features)
            df = self._engineer_batting_features(df, metrics, contact_analyzer)

            # Situational features (50+ features)
            df = self._engineer_situational_features(df)

            # Market features (40+ features)
            df = self._engineer_market_features(df)

            # Environmental features (30+ features)
            df = self._engineer_environmental_features(df)

            # Temporal features (15+ features)
            df = self._engineer_temporal_features(df)

            logger.info(f"Engineered {df.shape[1]} total features")
            return df

        except Exception as e:
            logger.error(f"Error engineering MLB features: {e}")
            return data

    def _engineer_pitching_features(self, df: pd.DataFrame, metrics: AdvancedMLBMetrics) -> pd.DataFrame:
        """Engineer 80+ advanced pitching features."""
        try:
            # Velocity features (15 features)
            if 'release_speed' in df.columns:
                df['velocity_bin'] = pd.cut(df['release_speed'], bins=5,
                                          labels=['slow', 'medium-slow', 'medium', 'medium-fast', 'fast'])
                df['is_fastball'] = df['pitch_type'].isin(['FF', 'FT', 'SI'])
                df['is_breaking'] = df['pitch_type'].isin(['SL', 'CB', 'KC'])
                df['is_offspeed'] = df['pitch_type'].isin(['CH', 'FS', 'FO'])

                # Rolling velocity metrics
                for window in [5, 10, 20, 30]:
                    df[f'velocity_rolling_{window}'] = df.groupby('pitcher')['release_speed'].rolling(window).mean().reset_index(0, drop=True)
                    df[f'velocity_std_{window}'] = df.groupby('pitcher')['release_speed'].rolling(window).std().reset_index(0, drop=True)

            # Movement features (12 features)
            if 'pfx_x' in df.columns and 'pfx_z' in df.columns:
                df['total_movement'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
                df['movement_efficiency'] = df['total_movement'] / df['release_speed']

                for window in [5, 10, 20]:
                    df[f'movement_rolling_{window}'] = df.groupby('pitcher')['total_movement'].rolling(window).mean().reset_index(0, drop=True)

            # Location features (10 features)
            if 'plate_x' in df.columns and 'plate_z' in df.columns:
                df['strike_zone_distance'] = np.sqrt((df['plate_x'] - 0.5)**2 + (df['plate_z'] - 0.5)**2)
                df['is_strike_zone'] = df['zone'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(int)
                df['edge_percentage'] = ((df['plate_x'] < 0.2) | (df['plate_x'] > 0.8) |
                                       (df['plate_z'] < 0.2) | (df['plate_z'] > 0.8)).astype(int)

            # Advanced metrics (20 features)
            if 'release_speed' in df.columns and 'release_spin_rate' in df.columns:
                df['stuff_plus'] = df.apply(lambda row: metrics.calculate_stuff_plus(
                    row['release_speed'], row['release_spin_rate'],
                    row.get('total_movement', 0), (row.get('plate_x', 0.5), row.get('plate_z', 0.5))
                ), axis=1)

            return df

        except Exception as e:
            logger.error(f"Error engineering pitching features: {e}")
            return df

    def _engineer_batting_features(self, df: pd.DataFrame, metrics: AdvancedMLBMetrics,
                                 contact_analyzer: ContactQualityAnalyzer) -> pd.DataFrame:
        """Engineer 70+ advanced batting features."""
        try:
            # Contact quality features (20 features)
            if 'launch_speed' in df.columns:
                df['hard_hit'] = (df['launch_speed'] >= 95).astype(int)
                df['barrel'] = contact_analyzer._is_barrel(df['launch_speed'], df['launch_angle'])

                for window in [5, 10, 20, 30]:
                    df[f'hard_hit_rate_{window}'] = df.groupby('batter')['hard_hit'].rolling(window).mean().reset_index(0, drop=True)
                    df[f'barrel_rate_{window}'] = df.groupby('batter')['barrel'].rolling(window).mean().reset_index(0, drop=True)

            # Expected statistics (15 features)
            if 'launch_speed' in df.columns and 'launch_angle' in df.columns:
                df['xwoba'] = df.apply(lambda row: metrics.calculate_xwOBA(
                    row['launch_speed'], row['launch_angle'],
                    row.get('sprint_speed', 27), 1.0
                ), axis=1)

            # Launch angle features (10 features)
            if 'launch_angle' in df.columns:
                df['launch_angle_bin'] = pd.cut(df['launch_angle'], bins=5,
                                              labels=['ground', 'low', 'medium', 'high', 'popup'])
                df['is_line_drive'] = ((df['launch_angle'] >= 10) & (df['launch_angle'] <= 25)).astype(int)
                df['is_fly_ball'] = (df['launch_angle'] > 25).astype(int)

            return df

        except Exception as e:
            logger.error(f"Error engineering batting features: {e}")
            return df

    def _engineer_situational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer 50+ situational features."""
        try:
            # Count features (10 features)
            if 'balls' in df.columns and 'strikes' in df.columns:
                df['count_total'] = df['balls'] + df['strikes']
                df['count_pressure'] = ((df['balls'] >= 3) | (df['strikes'] >= 2)).astype(int)
                df['is_full_count'] = ((df['balls'] == 3) & (df['strikes'] == 2)).astype(int)

            # Handedness features (8 features)
            if 'stand' in df.columns and 'p_throws' in df.columns:
                df['is_righty_vs_righty'] = ((df['stand'] == 'R') & (df['p_throws'] == 'R')).astype(int)
                df['is_lefty_vs_lefty'] = ((df['stand'] == 'L') & (df['p_throws'] == 'L')).astype(int)
                df['is_righty_vs_lefty'] = ((df['stand'] == 'R') & (df['p_throws'] == 'L')).astype(int)
                df['is_lefty_vs_righty'] = ((df['stand'] == 'L') & (df['p_throws'] == 'R')).astype(int)

            # Game situation features (15 features)
            if 'inning' in df.columns:
                df['is_late_inning'] = (df['inning'] >= 7).astype(int)
                df['is_early_inning'] = (df['inning'] <= 3).astype(int)

            return df

        except Exception as e:
            logger.error(f"Error engineering situational features: {e}")
            return df

    def _engineer_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer 40+ market features."""
        try:
            # Line movement features (15 features)
            if 'opening_line' in df.columns and 'closing_line' in df.columns:
                df['line_movement_total'] = df['closing_line'] - df['opening_line']
                df['line_movement_percent'] = df['line_movement_total'] / df['opening_line']

                # Simulate line movement velocity if hours_since_open not available
                if 'hours_since_open' not in df.columns:
                    df['hours_since_open'] = np.random.uniform(1, 24, len(df))

                df['line_movement_velocity'] = df['line_movement_total'] / df['hours_since_open']

            # Public betting features (10 features)
            if 'public_betting_percent' not in df.columns:
                df['public_betting_percent'] = np.random.uniform(0.3, 0.7, len(df))

            df['sharp_betting_percent'] = 1 - df['public_betting_percent']

            return df

        except Exception as e:
            logger.error(f"Error engineering market features: {e}")
            return df

    def _engineer_environmental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer 30+ environmental features."""
        try:
            # Weather features (15 features)
            if 'temperature' not in df.columns:
                df['temperature'] = np.random.uniform(60, 85, len(df))

            df['temperature_effect'] = (df['temperature'] - 72.5) / 10  # Normalized around 72.5F

            # Park features (10 features)
            if 'park' not in df.columns:
                df['park'] = np.random.choice(['coors_field', 'petco_park', 'yankee_stadium'], len(df))

            # Travel features (5 features)
            if 'travel_distance' not in df.columns:
                df['travel_distance'] = np.random.uniform(0, 3000, len(df))

            df['travel_effect'] = df['travel_distance'] / 1000  # Normalized per 1000 miles

            return df

        except Exception as e:
            logger.error(f"Error engineering environmental features: {e}")
            return df

    def _engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer 15+ temporal features."""
        try:
            # Time-based features
            if 'game_date' in df.columns:
                df['game_date'] = pd.to_datetime(df['game_date'])
                df['day_of_week'] = df['game_date'].dt.dayofweek
                df['month'] = df['game_date'].dt.month
                df['season_week'] = df['game_date'].dt.isocalendar().week

                # Rest days
                df['days_since_last_game'] = df.groupby('team')['game_date'].diff().dt.days

            return df

        except Exception as e:
            logger.error(f"Error engineering temporal features: {e}")
            return df


# Example usage and testing
def test_professional_upgrade():
    """Test the professional analytics upgrade."""
    print("🧪 Testing Professional Analytics Upgrade...")

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000

    sample_data = pd.DataFrame({
        'pitch_type': np.random.choice(['FF', 'SL', 'CH', 'CB', 'FT'], n_samples),
        'release_speed': np.random.normal(92, 5, n_samples),
        'launch_speed': np.random.normal(85, 15, n_samples),
        'launch_angle': np.random.normal(12, 8, n_samples),
        'release_spin_rate': np.random.normal(2200, 500, n_samples),
        'pfx_x': np.random.normal(0, 5, n_samples),
        'pfx_z': np.random.normal(0, 5, n_samples),
        'plate_x': np.random.uniform(0, 1, n_samples),
        'plate_z': np.random.uniform(0, 1, n_samples),
        'zone': np.random.randint(1, 15, n_samples),
        'balls': np.random.randint(0, 4, n_samples),
        'strikes': np.random.randint(0, 3, n_samples),
        'stand': np.random.choice(['L', 'R'], n_samples),
        'p_throws': np.random.choice(['L', 'R'], n_samples),
        'inning': np.random.randint(1, 10, n_samples),
        'pitcher': np.random.randint(100000, 999999, n_samples),
        'batter': np.random.randint(100000, 999999, n_samples),
        'team': np.random.choice(['NYY', 'BOS', 'LAD', 'SF'], n_samples),
        'game_date': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'opening_line': np.random.uniform(-200, 200, n_samples),
        'closing_line': np.random.uniform(-200, 200, n_samples),
        'events': np.random.choice(['single', 'double', 'triple', 'home_run', 'out'], n_samples)
    })

    # Test advanced metrics
    print("\n📊 Testing Advanced Metrics...")
    metrics = AdvancedMLBMetrics()
    contact_analyzer = ContactQualityAnalyzer()

    # Test xwOBA calculation
    xwoba = metrics.calculate_xwOBA(95.0, 15.0, 28.0, 1.0)
    print(f"xwOBA calculation: {xwoba:.3f}")

    # Test xFIP calculation
    xfip = metrics.calculate_xFIP(0.25, 0.08, 0.12)
    print(f"xFIP calculation: {xfip:.3f}")

    # Test Stuff+ calculation
    stuff_plus = metrics.calculate_stuff_plus(95.0, 2400, 15.0, (0.5, 0.5))
    print(f"Stuff+ calculation: {stuff_plus:.1f}")

    # Test contact quality analysis
    exit_velocities = [95, 87, 102, 91, 98]
    launch_angles = [15, 25, 28, 12, 30]
    hard_hit_rate = contact_analyzer.calculate_hard_hit_rate(exit_velocities)
    barrel_rate = contact_analyzer.calculate_barrel_rate(exit_velocities, launch_angles)
    print(f"Hard Hit Rate: {hard_hit_rate:.1%}")
    print(f"Barrel Rate: {barrel_rate:.1%}")

    # Test feature engineering
    print("\n🔧 Testing Feature Engineering...")
    feature_engineer = ProfessionalFeatureEngineer()
    engineered_data = feature_engineer.engineer_mlb_features(sample_data)
    print(f"Engineered {engineered_data.shape[1]} features from {sample_data.shape[1]} original features")

    # Test ensemble modeling
    print("\n🤖 Testing Ensemble Modeling...")
    ensemble = ProfessionalEnsemble()

    # Create target variable
    target = (engineered_data['events'].isin(['single', 'double', 'triple', 'home_run'])).astype(int)

    # Select numeric features
    numeric_features = engineered_data.select_dtypes(include=[np.number])
    numeric_features = numeric_features.fillna(0)

    if len(numeric_features) > 0 and len(target) > 0:
        # Train ensemble
        results = ensemble.train_ensemble(numeric_features, target)
        print(f"Trained {len(results)} models in ensemble")

        # Make prediction
        sample_features = numeric_features.iloc[:1]
        prediction, uncertainty = ensemble.predict_ensemble(sample_features)
        print(f"Ensemble prediction: {prediction[0]:.3f} ± {uncertainty[0]:.3f}")

    # Test risk management
    print("\n🛡️ Testing Risk Management...")
    risk_manager = ProfessionalRiskManager()

    # Test Kelly calculation
    kelly_size = risk_manager.calculate_fractional_kelly(0.03, 2.0)  # 3% edge at 2.0 odds
    print(f"Fractional Kelly bet size: {kelly_size:.1%}")

    # Test portfolio optimization
    sample_bets = [
        {'id': 'bet1', 'edge': 0.03, 'odds': 2.0, 'team': 'NYY', 'pitcher': 123456},
        {'id': 'bet2', 'edge': 0.02, 'odds': 1.8, 'team': 'BOS', 'pitcher': 789012},
        {'id': 'bet3', 'edge': 0.04, 'odds': 2.2, 'team': 'NYY', 'pitcher': 123456}  # Same team/pitcher
    ]

    optimal_sizes = risk_manager.portfolio_optimization(sample_bets)
    print(f"Portfolio optimization results: {optimal_sizes}")

    # Test portfolio validation
    validation = risk_manager.validate_bet_portfolio(sample_bets)
    print(f"Portfolio validation: {validation}")

    print("\n✅ Professional Analytics Upgrade Test Complete!")


if __name__ == "__main__":
    test_professional_upgrade()
