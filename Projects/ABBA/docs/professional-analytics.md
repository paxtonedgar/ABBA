# Professional Analytics Guide

**Status**: ✅ **PRODUCTION READY**  
**Last Updated**: 2025-01-20

## Overview

This guide covers the professional analytics capabilities of the ABBA system, including advanced statistical analysis, machine learning models, and performance optimization techniques for MLB and NHL betting strategies.

## Analytics Architecture

### 1. Multi-Layer Analytics Stack

#### Data Layer
```python
class DataAnalyticsLayer:
    def __init__(self):
        self.data_sources = {
            'mlb': ['baseball_savant', 'mlb_stats_api', 'fangraphs'],
            'nhl': ['sportlogiq', 'natural_stat_trick', 'moneypuck']
        }
        self.data_validator = DataValidator()
        self.data_processor = DataProcessor()
```

#### Feature Engineering Layer
```python
class FeatureEngineeringLayer:
    def __init__(self):
        self.mlb_features = MLBFeatureEngineer()
        self.nhl_features = NHLFeatureEngineer()
        self.feature_cache = LRUCache(1000)
        self.feature_validator = FeatureValidator()
```

#### Model Layer
```python
class ModelAnalyticsLayer:
    def __init__(self):
        self.models = {
            'xgboost': XGBoostModel(),
            'random_forest': RandomForestModel(),
            'neural_network': NeuralNetworkModel(),
            'ensemble': EnsembleModel()
        }
        self.model_registry = ModelRegistry()
        self.performance_tracker = PerformanceTracker()
```

#### Analytics Layer
```python
class AnalyticsLayer:
    def __init__(self):
        self.statistical_analyzer = StatisticalAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.market_analyzer = MarketAnalyzer()
```

### 2. Advanced Statistical Analysis

#### MLB Statistical Analysis
```python
class MLBStatisticalAnalyzer:
    def analyze_pitching_metrics(self, pitcher_data):
        """Advanced pitching analysis using Statcast data."""
        metrics = {
            'velocity_analysis': self._analyze_velocity(pitcher_data),
            'movement_analysis': self._analyze_movement(pitcher_data),
            'command_analysis': self._analyze_command(pitcher_data),
            'pitch_mix_analysis': self._analyze_pitch_mix(pitcher_data),
            'situational_analysis': self._analyze_situational(pitcher_data)
        }
        return metrics
    
    def analyze_batting_metrics(self, batter_data):
        """Advanced batting analysis using Statcast data."""
        metrics = {
            'exit_velocity_analysis': self._analyze_exit_velocity(batter_data),
            'launch_angle_analysis': self._analyze_launch_angle(batter_data),
            'barrel_analysis': self._analyze_barrel_rate(batter_data),
            'plate_discipline': self._analyze_plate_discipline(batter_data),
            'clutch_performance': self._analyze_clutch_performance(batter_data)
        }
        return metrics
    
    def _analyze_velocity(self, data):
        """Analyze pitch velocity patterns."""
        return {
            'average_velocity': np.mean(data['release_speed']),
            'velocity_consistency': np.std(data['release_speed']),
            'velocity_trend': self._calculate_trend(data['release_speed']),
            'velocity_by_pitch_type': data.groupby('pitch_type')['release_speed'].agg(['mean', 'std'])
        }
    
    def _analyze_movement(self, data):
        """Analyze pitch movement patterns."""
        return {
            'horizontal_movement': np.mean(data['pfx_x']),
            'vertical_movement': np.mean(data['pfx_z']),
            'movement_efficiency': self._calculate_movement_efficiency(data),
            'movement_by_pitch_type': data.groupby('pitch_type')[['pfx_x', 'pfx_z']].mean()
        }
```

#### NHL Statistical Analysis
```python
class NHLStatisticalAnalyzer:
    def analyze_goalie_metrics(self, goalie_data):
        """Advanced goalie analysis using modern metrics."""
        metrics = {
            'gsaa_analysis': self._analyze_gsaa(goalie_data),
            'save_percentage_analysis': self._analyze_save_percentage(goalie_data),
            'high_danger_analysis': self._analyze_high_danger_saves(goalie_data),
            'quality_start_analysis': self._analyze_quality_starts(goalie_data),
            'recent_form_analysis': self._analyze_recent_form(goalie_data)
        }
        return metrics
    
    def analyze_team_metrics(self, team_data):
        """Advanced team analysis using possession metrics."""
        metrics = {
            'possession_analysis': self._analyze_possession(team_data),
            'expected_goals_analysis': self._analyze_expected_goals(team_data),
            'special_teams_analysis': self._analyze_special_teams(team_data),
            'scoring_chance_analysis': self._analyze_scoring_chances(team_data),
            'momentum_analysis': self._analyze_momentum(team_data)
        }
        return metrics
    
    def _analyze_gsaa(self, data):
        """Analyze Goals Saved Above Average."""
        return {
            'gsaa_total': np.sum(data['gsaa']),
            'gsaa_per_game': np.mean(data['gsaa']),
            'gsaa_trend': self._calculate_trend(data['gsaa']),
            'gsaa_percentile': self._calculate_percentile(data['gsaa'])
        }
    
    def _analyze_possession(self, data):
        """Analyze possession metrics (Corsi, Fenwick, xGF)."""
        return {
            'corsi_for_percentage': np.mean(data['corsi_for_percentage']),
            'fenwick_for_percentage': np.mean(data['fenwick_for_percentage']),
            'xgf_percentage': np.mean(data['xgf_percentage']),
            'possession_trend': self._calculate_trend(data['corsi_for_percentage'])
        }
```

### 3. Machine Learning Models

#### Advanced Model Architecture
```python
class AdvancedMLModels:
    def __init__(self):
        self.models = {
            'xgboost': XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        }
        self.ensemble = VotingClassifier(
            estimators=[
                ('xgboost', self.models['xgboost']),
                ('random_forest', self.models['random_forest']),
                ('neural_network', self.models['neural_network']),
                ('gradient_boosting', self.models['gradient_boosting'])
            ],
            voting='soft'
        )
```

#### Model Training and Optimization
```python
class ModelOptimizer:
    def __init__(self):
        self.optimizer = OptunaOptimizer()
        self.cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.feature_selector = SelectKBest(score_func=f_classif, k=50)
    
    def optimize_hyperparameters(self, model_name, X, y):
        """Optimize model hyperparameters using Optuna."""
        def objective(trial):
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 5, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
                }
            
            model = self.models[model_name].set_params(**params)
            scores = cross_val_score(model, X, y, cv=self.cross_validator, scoring='accuracy')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        
        return study.best_params
    
    def feature_selection(self, X, y):
        """Perform feature selection using statistical tests."""
        X_selected = self.feature_selector.fit_transform(X, y)
        selected_features = X.columns[self.feature_selector.get_support()]
        
        return X_selected, selected_features
```

### 4. Performance Analytics

#### Model Performance Tracking
```python
class PerformanceAnalytics:
    def __init__(self):
        self.metrics_tracker = MetricsTracker()
        self.performance_history = {}
        self.alert_system = AlertSystem()
    
    def track_model_performance(self, model_name, predictions, actuals, metadata):
        """Track comprehensive model performance metrics."""
        metrics = {
            'accuracy': accuracy_score(actuals, predictions),
            'precision': precision_score(actuals, predictions, average='weighted'),
            'recall': recall_score(actuals, predictions, average='weighted'),
            'f1_score': f1_score(actuals, predictions, average='weighted'),
            'roc_auc': roc_auc_score(actuals, predictions),
            'log_loss': log_loss(actuals, predictions),
            'calibration_error': self._calculate_calibration_error(predictions, actuals),
            'confidence_interval': self._calculate_confidence_interval(predictions, actuals)
        }
        
        # Store performance history
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        self.performance_history[model_name].append({
            'timestamp': datetime.now(),
            'metrics': metrics,
            'metadata': metadata
        })
        
        # Check for performance degradation
        self._check_performance_degradation(model_name, metrics)
        
        return metrics
    
    def _check_performance_degradation(self, model_name, current_metrics):
        """Check for significant performance degradation."""
        if len(self.performance_history[model_name]) < 5:
            return
        
        # Calculate rolling average
        recent_metrics = self.performance_history[model_name][-5:]
        avg_accuracy = np.mean([m['metrics']['accuracy'] for m in recent_metrics])
        
        # Alert if performance drops by more than 5%
        if current_metrics['accuracy'] < avg_accuracy * 0.95:
            self.alert_system.send_alert(
                f"Model {model_name} performance degradation detected: "
                f"Current: {current_metrics['accuracy']:.3f}, "
                f"Average: {avg_accuracy:.3f}"
            )
```

#### Risk Analytics
```python
class RiskAnalytics:
    def __init__(self):
        self.risk_calculator = RiskCalculator()
        self.portfolio_analyzer = PortfolioAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
    
    def calculate_bet_risk(self, prediction, odds, bankroll, metadata):
        """Calculate comprehensive risk metrics for a bet."""
        risk_metrics = {
            'kelly_fraction': self._calculate_kelly_fraction(prediction, odds),
            'expected_value': self._calculate_expected_value(prediction, odds),
            'var_95': self._calculate_value_at_risk(prediction, odds, 0.95),
            'max_loss': self._calculate_max_loss(odds, bankroll),
            'risk_reward_ratio': self._calculate_risk_reward_ratio(prediction, odds),
            'confidence_interval': self._calculate_confidence_interval(prediction),
            'correlation_risk': self._calculate_correlation_risk(metadata)
        }
        
        return risk_metrics
    
    def analyze_portfolio_risk(self, active_bets, bankroll):
        """Analyze overall portfolio risk."""
        portfolio_metrics = {
            'total_exposure': sum(bet['stake'] for bet in active_bets),
            'exposure_percentage': sum(bet['stake'] for bet in active_bets) / bankroll,
            'diversification_score': self._calculate_diversification_score(active_bets),
            'correlation_matrix': self._calculate_correlation_matrix(active_bets),
            'portfolio_var': self._calculate_portfolio_var(active_bets, bankroll),
            'max_drawdown_risk': self._calculate_max_drawdown_risk(active_bets)
        }
        
        return portfolio_metrics
    
    def _calculate_kelly_fraction(self, prediction, odds):
        """Calculate Kelly Criterion fraction."""
        win_prob = prediction['win_probability']
        decimal_odds = self._convert_to_decimal(odds)
        
        kelly = (win_prob * decimal_odds - 1) / (decimal_odds - 1)
        return max(0, min(kelly, 0.25))  # Cap at 25% for safety
```

### 5. Market Analytics

#### Market Efficiency Analysis
```python
class MarketAnalytics:
    def __init__(self):
        self.efficiency_analyzer = EfficiencyAnalyzer()
        self.line_movement_analyzer = LineMovementAnalyzer()
        self.sharp_action_detector = SharpActionDetector()
    
    def analyze_market_efficiency(self, odds_data, predictions):
        """Analyze betting market efficiency."""
        efficiency_metrics = {
            'market_bias': self._calculate_market_bias(odds_data),
            'efficiency_score': self._calculate_efficiency_score(odds_data, predictions),
            'arbitrage_opportunities': self._detect_arbitrage(odds_data),
            'line_movement_patterns': self._analyze_line_movements(odds_data),
            'sharp_action_indicators': self._detect_sharp_action(odds_data)
        }
        
        return efficiency_metrics
    
    def analyze_line_movements(self, odds_history):
        """Analyze betting line movements."""
        movements = {
            'opening_to_current': self._calculate_line_movement(odds_history, 'opening', 'current'),
            'movement_velocity': self._calculate_movement_velocity(odds_history),
            'movement_direction': self._analyze_movement_direction(odds_history),
            'movement_magnitude': self._calculate_movement_magnitude(odds_history),
            'movement_timing': self._analyze_movement_timing(odds_history)
        }
        
        return movements
    
    def detect_sharp_action(self, odds_data, volume_data):
        """Detect sharp money movement patterns."""
        sharp_indicators = {
            'volume_spikes': self._detect_volume_spikes(volume_data),
            'line_movement_without_news': self._detect_unexplained_movements(odds_data),
            'reverse_line_movement': self._detect_reverse_movements(odds_data, volume_data),
            'sharp_consensus': self._detect_sharp_consensus(odds_data),
            'money_percentage': self._calculate_money_percentage(volume_data)
        }
        
        return sharp_indicators
```

## Analytics Dashboard

### 1. Real-Time Monitoring

#### Performance Dashboard
```python
class AnalyticsDashboard:
    def __init__(self):
        self.dashboard = Dashboard()
        self.real_time_updater = RealTimeUpdater()
        self.alert_manager = AlertManager()
    
    def create_performance_dashboard(self):
        """Create comprehensive performance dashboard."""
        dashboard = {
            'model_performance': self._create_model_performance_widget(),
            'risk_metrics': self._create_risk_metrics_widget(),
            'market_analysis': self._create_market_analysis_widget(),
            'portfolio_overview': self._create_portfolio_widget(),
            'system_health': self._create_system_health_widget()
        }
        
        return dashboard
    
    def _create_model_performance_widget(self):
        """Create model performance monitoring widget."""
        return {
            'type': 'performance_chart',
            'data': self._get_model_performance_data(),
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
            'time_range': 'last_30_days',
            'alerts': self._get_performance_alerts()
        }
```

### 2. Automated Reporting

#### Performance Reports
```python
class AutomatedReporting:
    def __init__(self):
        self.report_generator = ReportGenerator()
        self.scheduler = APScheduler()
    
    def generate_daily_report(self):
        """Generate comprehensive daily performance report."""
        report = {
            'executive_summary': self._generate_executive_summary(),
            'model_performance': self._generate_model_performance_report(),
            'risk_analysis': self._generate_risk_analysis_report(),
            'market_analysis': self._generate_market_analysis_report(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_executive_summary(self):
        """Generate executive summary of daily performance."""
        return {
            'total_bets': self._get_total_bets(),
            'win_rate': self._calculate_daily_win_rate(),
            'profit_loss': self._calculate_daily_pnl(),
            'risk_metrics': self._get_daily_risk_metrics(),
            'key_insights': self._get_key_insights()
        }
```

## Implementation

### 1. Configuration

#### Analytics Configuration
```python
# Analytics configuration
ANALYTICS_CONFIG = {
    'model_parameters': {
        'xgboost': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 8
        }
    },
    'performance_thresholds': {
        'min_accuracy': 0.54,
        'min_precision': 0.50,
        'max_drawdown': 0.12
    },
    'risk_parameters': {
        'max_bet_size': 0.05,
        'max_daily_risk': 0.08,
        'kelly_fraction': 0.25
    }
}
```

### 2. Monitoring

#### Analytics Monitoring
- **Model performance tracking**
- **Risk metrics calculation**
- **Market efficiency analysis**
- **Portfolio health monitoring**
- **System performance metrics**

---

**Status**: ✅ **PRODUCTION READY** - Advanced analytics capabilities
**Performance**: Comprehensive statistical analysis and ML modeling
**Features**: Real-time monitoring, automated reporting, risk analytics 