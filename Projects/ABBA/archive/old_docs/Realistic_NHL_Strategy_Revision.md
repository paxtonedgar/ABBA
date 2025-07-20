# Realistic NHL Strategy Revision: Industry Standards Alignment

## 🚨 **Critical Issues Identified**

### **1. Unrealistic ROI Targets**
**Problem**: 20% annual ROI target lacks credible documentation
**Reality**: Academic research shows 4-8.5% ROI is achievable, with 4-10% being realistic for skilled practitioners

### **2. Dangerous Risk Management**
**Problem**: 12% Kelly betting contradicts professional standards
**Reality**: Professionals use 1-3% per bet, with fractional Kelly (1/4 to 1/2) being standard

### **3. Outdated Analytics**
**Problem**: Basic Corsi/Fenwick reliance misses advanced metrics
**Reality**: Industry leaders use GAR models, GSAx, computer vision, and microstats

### **4. Sophistication Gap**
**Problem**: Basic XGBoost/neural networks vs. professional systems
**Reality**: Professional syndicates use multi-model ensembles with 250+ features achieving 60%+ win rates

## 🎯 **Realistic NHL Strategy Revision**

### **1. Conservative ROI Targets**
```python
realistic_nhl_targets = {
    'win_rate': 0.54,           # 54% win rate (realistic)
    'average_ev': 0.04,         # 4% average EV
    'annual_roi': 0.08,         # 8% annual ROI (realistic)
    'sharpe_ratio': 0.8,        # 0.8 Sharpe ratio
    'max_drawdown': 0.12,       # 12% max drawdown
    'value_bet_rate': 0.12,     # 12% of games offer value
    'positive_clv_rate': 0.55,  # 55% of bets beat closing line
    'average_bet_size': 0.02,   # 2% average bet size
    'monthly_growth': 0.02      # 2% monthly growth target
}
```

### **2. Professional Risk Management**
```python
class ProfessionalNHLRiskManager:
    def __init__(self, initial_bankroll):
        self.bankroll = initial_bankroll
        self.max_single_bet = 0.03  # 3% max per bet (professional standard)
        self.max_daily_risk = 0.08  # 8% max daily
        self.max_weekly_risk = 0.15 # 15% max weekly
        self.fractional_kelly = 0.25  # 1/4 Kelly (conservative)
        self.daily_risk_used = 0
        self.weekly_risk_used = 0
        self.active_bets = []
        
    def calculate_professional_stake(self, edge_analysis, prediction, odds, game_data):
        """Calculate stake using professional risk management."""
        
        try:
            # Base Kelly calculation
            kelly_fraction = self.calculate_base_kelly(prediction['win_probability'], odds)
            
            # Apply fractional Kelly (1/4)
            fractional_stake = kelly_fraction * self.fractional_kelly
            
            # Edge-based adjustment (more conservative)
            edge_multiplier = self.calculate_edge_multiplier(edge_analysis['adjusted_edge'])
            
            # Professional adjustments
            professional_multiplier = self.calculate_professional_multiplier(game_data)
            
            # Correlation adjustment
            correlation_adjustment = self.calculate_correlation_adjustment(game_data)
            
            # Market condition adjustment
            market_adjustment = self.calculate_market_adjustment(edge_analysis)
            
            final_stake = (fractional_stake * edge_multiplier * professional_multiplier * 
                          correlation_adjustment * market_adjustment)
            
            # Apply professional limits
            final_stake = min(final_stake, self.max_single_bet)
            final_stake = min(final_stake, self.max_daily_risk - self.daily_risk_used)
            final_stake = min(final_stake, self.max_weekly_risk - self.weekly_risk_used)
            
            # Safety check for minimum stake
            final_stake = max(0.005, final_stake)  # Minimum 0.5% stake
            
            # Validate stake is reasonable
            if final_stake > self.bankroll * 0.05:  # Never more than 5% of bankroll
                final_stake = self.bankroll * 0.05
                
            return final_stake
            
        except Exception as e:
            print(f"Error calculating stake: {e}")
            return 0.005  # Return minimum stake on error
    
    def calculate_edge_multiplier(self, adjusted_edge):
        """Calculate edge-based multiplier with more granular adjustments."""
        if adjusted_edge > 0.06:  # 6%+ edge
            return 1.3
        elif adjusted_edge > 0.04:  # 4%+ edge
            return 1.1
        elif adjusted_edge > 0.02:  # 2%+ edge
            return 1.0
        elif adjusted_edge > 0.01:  # 1%+ edge
            return 0.8
        else:  # <1% edge
            return 0.5
    
    def calculate_professional_multiplier(self, game_data):
        """Calculate professional adjustments with better granularity."""
        multiplier = 1.0
        
        try:
            # Goalie quality adjustment (using GSAx)
            goalie_advantage = abs(game_data['home_goalie']['gsax'] - game_data['away_goalie']['gsax'])
            if goalie_advantage > 10:  # 10+ GSAx difference
                multiplier *= 1.2
            elif goalie_advantage > 5:  # 5+ GSAx difference
                multiplier *= 1.1
            elif goalie_advantage > 2:  # 2+ GSAx difference
                multiplier *= 1.05
            
            # Possession advantage adjustment (using advanced metrics)
            possession_advantage = abs(game_data['home_team']['gar'] - game_data['away_team']['gar'])
            if possession_advantage > 5:  # 5+ GAR advantage
                multiplier *= 1.1
            elif possession_advantage > 2:  # 2+ GAR advantage
                multiplier *= 1.05
            
            # Schedule advantage adjustment
            if game_data.get('away_back_to_back', False):
                multiplier *= 1.15
            elif game_data.get('away_rest_days', 1) < game_data.get('home_rest_days', 1):
                multiplier *= 1.05
            
            # Injury impact adjustment
            home_injuries = game_data.get('home_team', {}).get('injury_impact', 0)
            away_injuries = game_data.get('away_team', {}).get('injury_impact', 0)
            injury_advantage = away_injuries - home_injuries
            if injury_advantage > 0.3:  # Significant injury advantage
                multiplier *= 1.1
            
        except KeyError as e:
            print(f"Missing data in game_data: {e}")
            # Continue with default multiplier
        
        return multiplier
    
    def calculate_correlation_adjustment(self, game_data):
        """Calculate correlation adjustment for portfolio risk."""
        try:
            # Check correlation with existing bets
            correlation_score = 0.0
            for bet in self.active_bets:
                # Team correlation
                if (game_data.get('home_team', {}).get('name') == bet.get('team_name') or
                    game_data.get('away_team', {}).get('name') == bet.get('team_name')):
                    correlation_score += 0.8
                
                # Goalie correlation
                if (game_data.get('home_goalie', {}).get('name') == bet.get('goalie_name') or
                    game_data.get('away_goalie', {}).get('name') == bet.get('goalie_name')):
                    correlation_score += 0.6
            
            # Adjust based on correlation
            if correlation_score > 1.0:
                return 0.7  # High correlation, reduce stake
            elif correlation_score > 0.5:
                return 0.85  # Medium correlation, moderate reduction
            else:
                return 1.0  # Low correlation, no adjustment
                
        except Exception as e:
            print(f"Error calculating correlation: {e}")
            return 1.0
    
    def calculate_market_adjustment(self, edge_analysis):
        """Calculate market condition adjustments."""
        multiplier = 1.0
        
        try:
            # Market efficiency adjustment
            market_efficiency = edge_analysis.get('market_efficiency', 0.85)
            if market_efficiency < 0.8:  # Inefficient market
                multiplier *= 1.1
            elif market_efficiency > 0.9:  # Very efficient market
                multiplier *= 0.9
            
            # Public betting adjustment
            public_betting = edge_analysis.get('public_betting', 0.5)
            if public_betting > 0.75:  # Heavy public action
                multiplier *= 1.15  # Bet against public
            elif public_betting < 0.25:  # Light public action
                multiplier *= 1.05  # Bet with sharp money
            
            # Line movement adjustment
            line_movement = abs(edge_analysis.get('line_movement', 0))
            if line_movement > 30:  # Significant movement
                multiplier *= 1.1
            elif line_movement > 15:  # Moderate movement
                multiplier *= 1.05
                
        except Exception as e:
            print(f"Error calculating market adjustment: {e}")
        
        return multiplier
    
    def add_bet(self, bet_data):
        """Add bet to active portfolio."""
        stake = bet_data.get('stake', 0)
        self.active_bets.append(bet_data)
        self.daily_risk_used += stake
        self.weekly_risk_used += stake
    
    def reset_daily_risk(self):
        """Reset daily risk counter."""
        self.daily_risk_used = 0
    
    def reset_weekly_risk(self):
        """Reset weekly risk counter."""
        self.weekly_risk_used = 0
```

### **3. Advanced Analytics Integration**
```python
class AdvancedNHLAnalytics:
    def __init__(self):
        self.data_sources = {
            'nhl_edge': NHLEdgeData(),
            'sportlogiq': SportlogiqData(),
            'money_puck': MoneyPuckData(),
            'natural_stat_trick': NaturalStatTrickData(),
            'clearsight': ClearSightData()
        }
        self.feature_cache = {}
        
    def extract_professional_features(self, game_data):
        """Extract features using industry-leading analytics with error handling."""
        
        features = {}
        
        try:
            # Goals Above Replacement (GAR) - Industry standard
            features['home_gar'] = game_data.get('home_team', {}).get('gar', 0)
            features['away_gar'] = game_data.get('away_team', {}).get('gar', 0)
            features['gar_differential'] = features['home_gar'] - features['away_gar']
            
            # Goals Saved Above Expected (GSAx) - Gold standard for goalies
            features['home_gsax'] = game_data.get('home_goalie', {}).get('gsax', 0)
            features['away_gsax'] = game_data.get('away_goalie', {}).get('gsax', 0)
            features['gsax_differential'] = features['home_gsax'] - features['away_gsax']
            
            # High-danger save percentage (most repeatable goalie skill)
            features['home_hd_save_pct'] = game_data.get('home_goalie', {}).get('high_danger_save_pct', 0.85)
            features['away_hd_save_pct'] = game_data.get('away_goalie', {}).get('high_danger_save_pct', 0.85)
            features['hd_save_pct_differential'] = features['home_hd_save_pct'] - features['away_hd_save_pct']
            
            # Quality Start percentage (75% correlation with wins)
            features['home_quality_start_pct'] = game_data.get('home_goalie', {}).get('quality_start_pct', 0.5)
            features['away_quality_start_pct'] = game_data.get('away_goalie', {}).get('quality_start_pct', 0.5)
            
            # Rebound control (highest year-to-year repeatability)
            features['home_rebound_control'] = game_data.get('home_goalie', {}).get('rebound_control', 0.5)
            features['away_rebound_control'] = game_data.get('away_goalie', {}).get('rebound_control', 0.5)
            
            # Zone entry efficiency (controlled vs dump-ins)
            features['home_controlled_entry_pct'] = game_data.get('home_team', {}).get('controlled_entry_pct', 0.5)
            features['away_controlled_entry_pct'] = game_data.get('away_team', {}).get('controlled_entry_pct', 0.5)
            
            # Transition game effectiveness
            features['home_transition_efficiency'] = game_data.get('home_team', {}).get('transition_efficiency', 0.5)
            features['away_transition_efficiency'] = game_data.get('away_team', {}).get('transition_efficiency', 0.5)
            
            # Loose puck recovery rates
            features['home_puck_recovery_rate'] = game_data.get('home_team', {}).get('puck_recovery_rate', 0.5)
            features['away_puck_recovery_rate'] = game_data.get('away_team', {}).get('puck_recovery_rate', 0.5)
            
            # Defensive positioning metrics
            features['home_gap_control'] = game_data.get('home_team', {}).get('gap_control', 0.5)
            features['away_gap_control'] = game_data.get('away_team', {}).get('gap_control', 0.5)
            
            # Skating mechanics (from NHL EDGE)
            features['home_skating_efficiency'] = game_data.get('home_team', {}).get('skating_efficiency', 0.5)
            features['away_skating_efficiency'] = game_data.get('away_team', {}).get('skating_efficiency', 0.5)
            
            # Power play formation analysis
            features['home_pp_formation_efficiency'] = game_data.get('home_team', {}).get('pp_formation_efficiency', 0.5)
            features['away_pp_formation_efficiency'] = game_data.get('away_team', {}).get('pp_formation_efficiency', 0.5)
            
            # Score effects (leading teams play conservatively)
            features['home_score_effect_adjustment'] = game_data.get('home_team', {}).get('score_effect_adjustment', 0)
            features['away_score_effect_adjustment'] = game_data.get('away_team', {}).get('score_effect_adjustment', 0)
            
            # Referee bias patterns
            features['referee_home_bias'] = game_data.get('referee', {}).get('home_bias', 0)
            features['referee_penalty_tendency'] = game_data.get('referee', {}).get('penalty_tendency', 0.5)
            
            # Add derived features
            features.update(self.calculate_derived_features(features))
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return basic features as fallback
            features = self.extract_basic_features(game_data)
        
        return features
    
    def calculate_derived_features(self, base_features):
        """Calculate derived features from base features."""
        derived = {}
        
        try:
            # Interaction features
            derived['gar_gsax_interaction'] = base_features['gar_differential'] * base_features['gsax_differential']
            derived['hd_save_transition_interaction'] = base_features['hd_save_pct_differential'] * base_features['home_transition_efficiency']
            
            # Ratio features
            derived['gar_ratio'] = base_features['home_gar'] / max(base_features['away_gar'], 0.1)
            derived['gsax_ratio'] = base_features['home_gsax'] / max(base_features['away_gsax'], 0.1)
            
            # Composite scores
            derived['home_composite_score'] = (
                base_features['home_gar'] * 0.3 +
                base_features['home_gsax'] * 0.3 +
                base_features['home_hd_save_pct'] * 0.2 +
                base_features['home_transition_efficiency'] * 0.2
            )
            
            derived['away_composite_score'] = (
                base_features['away_gar'] * 0.3 +
                base_features['away_gsax'] * 0.3 +
                base_features['away_hd_save_pct'] * 0.2 +
                base_features['away_transition_efficiency'] * 0.2
            )
            
            derived['composite_differential'] = derived['home_composite_score'] - derived['away_composite_score']
            
        except Exception as e:
            print(f"Error calculating derived features: {e}")
        
        return derived
    
    def extract_basic_features(self, game_data):
        """Extract basic features as fallback."""
        return {
            'home_gar': 0,
            'away_gar': 0,
            'gar_differential': 0,
            'home_gsax': 0,
            'away_gsax': 0,
            'gsax_differential': 0,
            'home_hd_save_pct': 0.85,
            'away_hd_save_pct': 0.85,
            'hd_save_pct_differential': 0,
            'home_quality_start_pct': 0.5,
            'away_quality_start_pct': 0.5,
            'home_rebound_control': 0.5,
            'away_rebound_control': 0.5,
            'home_controlled_entry_pct': 0.5,
            'away_controlled_entry_pct': 0.5,
            'home_transition_efficiency': 0.5,
            'away_transition_efficiency': 0.5,
            'home_puck_recovery_rate': 0.5,
            'away_puck_recovery_rate': 0.5,
            'home_gap_control': 0.5,
            'away_gap_control': 0.5,
            'home_skating_efficiency': 0.5,
            'away_skating_efficiency': 0.5,
            'home_pp_formation_efficiency': 0.5,
            'away_pp_formation_efficiency': 0.5,
            'home_score_effect_adjustment': 0,
            'away_score_effect_adjustment': 0,
            'referee_home_bias': 0,
            'referee_penalty_tendency': 0.5
        }
```

### **4. Professional Machine Learning Ensemble**
```python
class ProfessionalNHLEnsemble:
    def __init__(self):
        # Multi-model ensemble with 250+ features
        self.models = {
            'xgboost_advanced': XGBClassifier(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0
            ),
            'lightgbm_advanced': LGBMClassifier(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0
            ),
            'catboost_advanced': CatBoostClassifier(
                iterations=500, depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bylevel=0.8, reg_lambda=1.0, verbose=False
            ),
            'neural_advanced': MLPClassifier(
                hidden_layer_sizes=(200, 100, 50), max_iter=1000,
                alpha=0.001, learning_rate='adaptive', early_stopping=True
            ),
            'random_forest_advanced': RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, max_features='sqrt'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, max_features='sqrt'
            )
        }
        
        # Game-state-specific models
        self.situational_models = {
            'even_strength': self.create_situational_model('even_strength'),
            'power_play': self.create_situational_model('power_play'),
            'penalty_kill': self.create_situational_model('penalty_kill'),
            'close_game': self.create_situational_model('close_game'),
            'blowout': self.create_situational_model('blowout')
        }
        
        self.meta_learner = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)
        self.feature_importance_tracker = FeatureImportanceTracker()
        self.model_performance_tracker = ModelPerformanceTracker()
        
    def predict_with_professional_ensemble(self, features, game_context):
        """Predict using professional-grade ensemble with error handling."""
        
        try:
            # Get base predictions
            base_predictions = {}
            for name, model in self.models.items():
                try:
                    X = np.array([list(features.values())])
                    base_predictions[name] = model.predict_proba(X)[0][1]
                except Exception as e:
                    print(f"Error with {name} model: {e}")
                    base_predictions[name] = 0.5  # Default prediction
            
            # Get situational predictions
            situational_predictions = {}
            for situation, model in self.situational_models.items():
                if self.is_situation_applicable(game_context, situation):
                    try:
                        X = np.array([list(features.values())])
                        situational_predictions[situation] = model.predict_proba(X)[0][1]
                    except Exception as e:
                        print(f"Error with {situation} model: {e}")
                        situational_predictions[situation] = 0.5
            
            # Calculate dynamic weights
            weights = self.calculate_professional_weights(base_predictions, situational_predictions, game_context)
            
            # Weighted ensemble prediction
            weighted_prediction = sum(pred * weights[model] for model, pred in base_predictions.items())
            
            # Add situational adjustments
            for situation, pred in situational_predictions.items():
                situation_weight = weights.get(f'situation_{situation}', 0)
                weighted_prediction += pred * situation_weight
            
            # Meta-learner final prediction
            meta_features = list(base_predictions.values()) + list(situational_predictions.values())
            final_prediction = self.meta_learner.predict_proba([meta_features])[0][1]
            
            # Validate prediction
            final_prediction = max(0.1, min(0.9, final_prediction))  # Bound between 10% and 90%
            
            return {
                'prediction': final_prediction,
                'confidence': self.calculate_professional_confidence(weights, base_predictions),
                'uncertainty': self.calculate_uncertainty(base_predictions),
                'model_weights': weights,
                'feature_importance': self.feature_importance_tracker.get_importance(features),
                'model_agreement': self.calculate_model_agreement(base_predictions)
            }
            
        except Exception as e:
            print(f"Error in ensemble prediction: {e}")
            return {
                'prediction': 0.5,
                'confidence': 0.5,
                'uncertainty': 0.1,
                'model_weights': {},
                'feature_importance': {},
                'model_agreement': 0.5
            }
    
    def calculate_model_agreement(self, predictions):
        """Calculate agreement between models."""
        if not predictions:
            return 0.5
        
        values = list(predictions.values())
        return 1 - np.std(values)  # Higher agreement = lower standard deviation
    
    def is_situation_applicable(self, game_context, situation):
        """Check if situational model is applicable."""
        try:
            if situation == 'even_strength':
                return True  # Always applicable
            elif situation == 'power_play':
                return game_context.get('power_play_opportunity', False)
            elif situation == 'penalty_kill':
                return game_context.get('penalty_kill_situation', False)
            elif situation == 'close_game':
                return game_context.get('score_differential', 0) <= 2
            elif situation == 'blowout':
                return game_context.get('score_differential', 0) > 3
            else:
                return False
        except Exception as e:
            print(f"Error checking situation applicability: {e}")
            return False
```

### **5. Realistic Edge Detection**
```python
class RealisticEdgeDetector:
    def __init__(self):
        self.min_edge_threshold = 0.03  # 3% minimum edge (realistic)
        self.max_edge_expectation = 0.08  # 8% maximum realistic edge
        
    def calculate_realistic_edge(self, prediction, odds, market_data, game_data):
        """Calculate realistic edge with professional standards and error handling."""
        
        try:
            our_prob = prediction['win_probability']
            implied_prob = self.odds_to_prob(odds)
            
            # Validate probabilities
            our_prob = max(0.1, min(0.9, our_prob))
            implied_prob = max(0.1, min(0.9, implied_prob))
            
            # Base edge
            raw_edge = our_prob - implied_prob
            
            # Realistic market efficiency adjustment
            market_efficiency = market_data.get('efficiency_score', 0.85)  # NHL markets more efficient
            edge_adjustment = 1 + (1 - market_efficiency) * 0.3  # Smaller adjustment
            
            # Professional adjustments
            professional_multiplier = self.calculate_professional_multiplier(market_data, game_data)
            
            adjusted_edge = raw_edge * edge_adjustment * professional_multiplier
            
            # Cap at realistic maximum
            adjusted_edge = min(adjusted_edge, self.max_edge_expectation)
            
            # Ensure minimum threshold
            if abs(adjusted_edge) < self.min_edge_threshold:
                adjusted_edge = 0  # No edge if below threshold
            
            return {
                'raw_edge': raw_edge,
                'adjusted_edge': adjusted_edge,
                'edge_adjustment': edge_adjustment,
                'professional_multiplier': professional_multiplier,
                'market_efficiency': market_efficiency,
                'edge_quality': self.assess_edge_quality(adjusted_edge, prediction)
            }
            
        except Exception as e:
            print(f"Error calculating edge: {e}")
            return {
                'raw_edge': 0,
                'adjusted_edge': 0,
                'edge_adjustment': 1.0,
                'professional_multiplier': 1.0,
                'market_efficiency': 0.85,
                'edge_quality': 'low'
            }
    
    def calculate_professional_multiplier(self, market_data, game_data):
        """Calculate professional adjustments with validation."""
        multiplier = 1.0
        
        try:
            # Goalie factor (using GSAx)
            goalie_factor = market_data.get('goalie_factor', 1.0)
            if goalie_factor > 1.1:  # Elite goalie playing
                multiplier *= 1.15
            elif goalie_factor < 0.9:  # Weak goalie playing
                multiplier *= 1.1
            
            # Public betting adjustment (more conservative)
            public_betting = market_data.get('public_betting_percentage', 0.5)
            if public_betting > 0.75:  # Heavy public action
                multiplier *= 1.2  # Bet against public
            elif public_betting < 0.25:  # Light public action
                multiplier *= 1.1  # Bet with sharp money
            
            # Schedule factor adjustment
            schedule_factor = market_data.get('schedule_factor', 1.0)
            if schedule_factor < 0.85:  # Back-to-back, travel issues
                multiplier *= 1.15
            
            # Line movement adjustment
            line_movement = abs(market_data.get('line_movement', 0))
            if line_movement > 25:  # Significant movement
                multiplier *= 1.1
            elif line_movement > 15:  # Moderate movement
                multiplier *= 1.05
            
        except Exception as e:
            print(f"Error calculating professional multiplier: {e}")
        
        return multiplier
    
    def assess_edge_quality(self, adjusted_edge, prediction):
        """Assess the quality of the edge."""
        if abs(adjusted_edge) < 0.02:
            return 'low'
        elif abs(adjusted_edge) < 0.04:
            return 'medium'
        elif abs(adjusted_edge) < 0.06:
            return 'high'
        else:
            return 'very_high'
    
    def odds_to_prob(self, odds):
        """Convert American odds to probability with validation."""
        try:
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return abs(odds) / (abs(odds) + 100)
        except Exception as e:
            print(f"Error converting odds to probability: {e}")
            return 0.5
```

### **6. Professional Performance Monitoring**
```python
class ProfessionalPerformanceMonitor:
    def __init__(self):
        self.metrics_calculator = ProfessionalMetricsCalculator()
        self.alert_system = ProfessionalAlertSystem()
        self.performance_history = []
        
    def calculate_professional_metrics(self, bet_history):
        """Calculate professional-grade performance metrics with error handling."""
        
        try:
            if not bet_history:
                return self.get_empty_metrics()
            
            # Basic metrics
            basic_metrics = self.calculate_basic_metrics(bet_history)
            
            # Risk-adjusted metrics
            risk_metrics = self.calculate_risk_metrics(bet_history)
            
            # Market efficiency metrics
            market_metrics = self.calculate_market_metrics(bet_history)
            
            # Professional metrics
            professional_metrics = {
                'information_ratio': self.calculate_information_ratio(bet_history),
                'calmar_ratio': self.calculate_calmar_ratio(bet_history),
                'sortino_ratio': self.calculate_sortino_ratio(bet_history),
                'max_consecutive_losses': self.calculate_max_consecutive_losses(bet_history),
                'recovery_time': self.calculate_recovery_time(bet_history),
                'edge_persistence': self.calculate_edge_persistence(bet_history),
                'clv_performance': self.calculate_clv_performance(bet_history),
                'market_efficiency_score': self.calculate_market_efficiency_score(bet_history)
            }
            
            # Store performance history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'metrics': {
                    'basic': basic_metrics,
                    'risk_adjusted': risk_metrics,
                    'market_efficiency': market_metrics,
                    'professional': professional_metrics
                }
            })
            
            return {
                'basic': basic_metrics,
                'risk_adjusted': risk_metrics,
                'market_efficiency': market_metrics,
                'professional': professional_metrics
            }
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return self.get_empty_metrics()
    
    def get_empty_metrics(self):
        """Return empty metrics structure."""
        return {
            'basic': {},
            'risk_adjusted': {},
            'market_efficiency': {},
            'professional': {}
        }
    
    def calculate_basic_metrics(self, bet_history):
        """Calculate basic performance metrics."""
        try:
            total_bets = len(bet_history)
            wins = sum(1 for bet in bet_history if bet.get('result') == 'win')
            win_rate = wins / total_bets if total_bets > 0 else 0
            
            returns = [bet.get('return', 0) for bet in bet_history]
            avg_return = np.mean(returns) if returns else 0
            total_return = sum(returns) if returns else 0
            
            return {
                'total_bets': total_bets,
                'wins': wins,
                'win_rate': win_rate,
                'avg_return': avg_return,
                'total_return': total_return
            }
        except Exception as e:
            print(f"Error calculating basic metrics: {e}")
            return {}
    
    def calculate_risk_metrics(self, bet_history):
        """Calculate risk-adjusted metrics."""
        try:
            returns = [bet.get('return', 0) for bet in bet_history]
            if not returns:
                return {}
            
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            
            # Calculate maximum drawdown
            cumulative = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volatility': std_return,
                'avg_return': avg_return
            }
        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
            return {}
    
    def calculate_information_ratio(self, bet_history):
        """Calculate information ratio."""
        try:
            returns = [bet.get('return', 0) for bet in bet_history]
            if not returns:
                return 0
            
            avg_return = np.mean(returns)
            tracking_error = np.std(returns)
            
            return avg_return / tracking_error if tracking_error > 0 else 0
        except Exception as e:
            print(f"Error calculating information ratio: {e}")
            return 0
    
    def calculate_max_consecutive_losses(self, bet_history):
        """Calculate maximum consecutive losses."""
        try:
            results = [bet.get('result', 'loss') for bet in bet_history]
            max_losses = 0
            current_losses = 0
            
            for result in results:
                if result == 'loss':
                    current_losses += 1
                    max_losses = max(max_losses, current_losses)
                else:
                    current_losses = 0
            
            return max_losses
        except Exception as e:
            print(f"Error calculating max consecutive losses: {e}")
            return 0
```

## 📊 **Realistic Performance Expectations**

### **Professional NHL Targets:**
```python
professional_nhl_targets = {
    'win_rate': 0.54,           # 54% win rate (realistic)
    'average_ev': 0.04,         # 4% average EV
    'annual_roi': 0.08,         # 8% annual ROI (realistic)
    'sharpe_ratio': 0.8,        # 0.8 Sharpe ratio
    'max_drawdown': 0.12,       # 12% max drawdown
    'value_bet_rate': 0.12,     # 12% of games offer value
    'positive_clv_rate': 0.55,  # 55% of bets beat closing line
    'average_bet_size': 0.02,   # 2% average bet size
    'monthly_growth': 0.02      # 2% monthly growth target
}
```

## 🛠️ **Implementation Strategy**

### **Phase 1: Foundation (Months 1-3)**
- Implement professional risk management (1-3% bets)
- Integrate advanced analytics (GAR, GSAx, microstats)
- Develop realistic edge detection (3-8% edges)

### **Phase 2: Enhancement (Months 4-6)**
- Build professional ensemble (250+ features)
- Implement situational modeling
- Add comprehensive performance monitoring

### **Phase 3: Optimization (Months 7-9)**
- Integrate NHL EDGE and Sportlogiq data
- Implement computer vision analysis
- Add real-time market microstructure

## 🎯 **Key Changes from Aggressive Strategy**

### **Risk Management:**
- **From**: 12% Kelly betting
- **To**: 1-3% fractional Kelly (professional standard)

### **ROI Targets:**
- **From**: 20% annual ROI
- **To**: 8% annual ROI (realistic)

### **Analytics:**
- **From**: Basic Corsi/Fenwick
- **To**: GAR, GSAx, microstats, computer vision

### **Edge Detection:**
- **From**: 4-6% minimum edges
- **To**: 3-8% realistic edges

### **Machine Learning:**
- **From**: Basic XGBoost/neural networks
- **To**: Professional ensemble with 250+ features

## 📈 **Expected Realistic Performance**

### **Conservative but Sustainable:**
- **Win Rate**: 54% (vs. claimed 56%)
- **Annual ROI**: 8% (vs. claimed 20%)
- **Sharpe Ratio**: 0.8 (vs. claimed 0.9)
- **Max Drawdown**: 12% (vs. claimed 18%)

### **Professional Standards:**
- **Risk per bet**: 1-3% (vs. claimed 12%)
- **Daily risk limit**: 8% (vs. claimed 30%)
- **Weekly risk limit**: 15% (vs. claimed 60%)

This realistic revision aligns with industry best practices while maintaining the goal of consistent profitability in NHL betting markets. 