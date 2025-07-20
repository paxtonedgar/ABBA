# Ultra-Conservative MLB Betting Strategy: Addressing Remaining Issues

## Executive Summary

The "realistic" approach still contains optimistic assumptions. This ultra-conservative version addresses the remaining issues with model complexity, CLV calculation, and line movement analysis.

## Core Philosophy: Simplicity Over Sophistication

### 1. **Minimalist Model Approach**
```python
# Start with ONE simple model only
class SimpleMLBModel:
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.features = [
            'home_pitcher_era_last_30',
            'away_pitcher_era_last_30', 
            'home_team_woba_last_30',
            'away_team_woba_last_30',
            'park_factor',
            'rest_advantage'  # 1 if home has more rest, -1 if away, 0 if even
        ]
    
    def train(self, X, y):
        """Train on simple features only."""
        X_simple = X[self.features]
        self.model.fit(X_simple, y)
    
    def predict(self, X):
        """Predict win probability."""
        X_simple = X[self.features]
        return self.model.predict_proba(X_simple)[:, 1]
    
    def get_feature_importance(self):
        """Get feature importance for transparency."""
        return dict(zip(self.features, self.model.coef_[0]))
```

### 2. **Conservative Performance Expectations**
```python
ultra_conservative_targets = {
    'win_rate': 0.52,           # 52% win rate
    'average_ev': 0.008,        # 0.8% average EV
    'annual_roi': 0.03,         # 3% annual ROI
    'sharpe_ratio': 0.3,        # 0.3 Sharpe ratio
    'max_drawdown': 0.20,       # 20% max drawdown
    'value_bet_rate': 0.05,     # 5% of games offer value
    'positive_clv_rate': 0.52   # 52% of bets beat closing line
}
```

## Fixed CLV Implementation

### 1. **Correct CLV Calculation**
```python
def calculate_clv_correct(bet_odds, closing_odds):
    """
    Calculate Closing Line Value correctly.
    Positive CLV = closing odds moved in your favor
    """
    bet_prob = odds_to_probability(bet_odds)
    closing_prob = odds_to_probability(closing_odds)
    
    # CLV is positive when closing line moves in your favor
    clv = closing_prob - bet_prob
    
    return clv

def odds_to_probability(odds):
    """Convert American odds to probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

# Example:
# Bet on team at +150 (40% implied probability)
# Closing line moves to +120 (45.5% implied probability)
# CLV = 0.455 - 0.400 = +0.055 (5.5% positive CLV)
```

### 2. **CLV-Based Performance Tracking**
```python
class CLVTracker:
    def __init__(self):
        self.bets = []
    
    def add_bet(self, bet_odds, closing_odds, bet_result, stake):
        """Track a completed bet."""
        clv = calculate_clv_correct(bet_odds, closing_odds)
        
        bet_record = {
            'bet_odds': bet_odds,
            'closing_odds': closing_odds,
            'bet_prob': odds_to_probability(bet_odds),
            'closing_prob': odds_to_probability(closing_odds),
            'clv': clv,
            'result': bet_result,  # 'win', 'loss', 'push'
            'stake': stake,
            'timestamp': datetime.now()
        }
        
        self.bets.append(bet_record)
    
    def get_performance_metrics(self):
        """Calculate CLV-based performance metrics."""
        if not self.bets:
            return {}
        
        clv_values = [bet['clv'] for bet in self.bets]
        positive_clv_bets = [bet for bet in self.bets if bet['clv'] > 0]
        
        metrics = {
            'total_bets': len(self.bets),
            'average_clv': np.mean(clv_values),
            'clv_std': np.std(clv_values),
            'positive_clv_rate': len(positive_clv_bets) / len(self.bets),
            'average_positive_clv': np.mean([bet['clv'] for bet in positive_clv_bets]) if positive_clv_bets else 0,
            'average_negative_clv': np.mean([bet['clv'] for bet in self.bets if bet['clv'] < 0]) if any(bet['clv'] < 0 for bet in self.bets) else 0
        }
        
        return metrics
```

## Simplified Line Movement Analysis

### 1. **Basic Line Movement Tracking**
```python
class SimpleLineTracker:
    def __init__(self):
        self.line_history = {}
    
    def track_line(self, game_id, bet_type, odds, timestamp):
        """Track basic line movement without assumptions about sharp action."""
        if game_id not in self.line_history:
            self.line_history[game_id] = {}
        
        if bet_type not in self.line_history[game_id]:
            self.line_history[game_id][bet_type] = []
        
        self.line_history[game_id][bet_type].append({
            'odds': odds,
            'timestamp': timestamp,
            'implied_prob': odds_to_probability(odds)
        })
    
    def get_line_movement(self, game_id, bet_type):
        """Get basic line movement without sharp action assumptions."""
        if game_id not in self.line_history or bet_type not in self.line_history[game_id]:
            return None
        
        history = self.line_history[game_id][bet_type]
        if len(history) < 2:
            return None
        
        opening = history[0]
        current = history[-1]
        
        movement = {
            'opening_odds': opening['odds'],
            'current_odds': current['odds'],
            'opening_prob': opening['implied_prob'],
            'current_prob': current['implied_prob'],
            'prob_movement': current['implied_prob'] - opening['implied_prob'],
            'time_span': current['timestamp'] - opening['timestamp']
        }
        
        return movement
    
    def should_bet_now(self, game_id, bet_type, our_prob, current_odds):
        """Simple decision: bet if our probability is higher than current implied probability."""
        current_prob = odds_to_probability(current_odds)
        edge = our_prob - current_prob
        
        # Only bet if we have a clear edge (>1%)
        return edge > 0.01
```

### 2. **No Sharp Action Detection**
```python
# REMOVED: All sharp action detection
# REMOVED: Steam move detection  
# REMOVED: Reverse line movement analysis
# REMOVED: Public vs. sharp money assumptions

# Instead: Simple edge detection
def calculate_simple_edge(our_prob, current_odds):
    """Calculate simple edge without complex assumptions."""
    current_prob = odds_to_probability(current_odds)
    edge = our_prob - current_prob
    
    # Apply conservative adjustment for uncertainty
    adjusted_edge = edge * 0.7  # Reduce edge by 30% for uncertainty
    
    return adjusted_edge
```

## Ultra-Conservative Risk Management

### 1. **Minimal Position Sizing**
```python
def calculate_ultra_conservative_stake(bankroll, edge, confidence=0.5):
    """Calculate ultra-conservative stake size."""
    # Base stake: 0.5% of bankroll per 1% edge
    base_stake = edge * 0.5
    
    # Apply confidence adjustment
    confidence_stake = base_stake * confidence
    
    # Maximum stake: 1% of bankroll
    max_stake = min(confidence_stake, 0.01)
    
    # Minimum stake: 0.1% of bankroll
    final_stake = max(max_stake, 0.001)
    
    return final_stake

# Example:
# $10,000 bankroll, 2% edge, 70% confidence
# Base stake = 0.02 * 0.5 = 1% of bankroll
# Confidence stake = 1% * 0.7 = 0.7% of bankroll
# Final stake = 0.7% = $70
```

### 2. **Strict Portfolio Limits**
```python
ultra_conservative_limits = {
    'max_single_bet': 0.01,      # 1% max per bet
    'max_single_game': 0.02,     # 2% max per game
    'max_daily_risk': 0.05,      # 5% max daily
    'max_weekly_risk': 0.10,     # 10% max weekly
    'min_edge_threshold': 0.015, # 1.5% minimum edge
    'min_confidence': 0.6,       # 60% minimum confidence
    'max_concurrent_bets': 3     # Maximum 3 bets at once
}
```

## Simplified Decision Process

### 1. **Single Model Prediction**
```python
def analyze_game_ultra_simple(game_data):
    """Ultra-simple game analysis."""
    
    # Extract basic features
    features = {
        'home_pitcher_era': game_data['home_pitcher']['last_30_era'],
        'away_pitcher_era': game_data['away_pitcher']['last_30_era'],
        'home_team_woba': np.mean([p['woba_last_30'] for p in game_data['home_lineup']]),
        'away_team_woba': np.mean([p['woba_last_30'] for p in game_data['away_lineup']]),
        'park_factor': game_data['park_factor'],
        'rest_advantage': 1 if game_data['home_rest_days'] > game_data['away_rest_days'] else -1 if game_data['away_rest_days'] > game_data['home_rest_days'] else 0
    }
    
    # Simple probability calculation
    base_prob = 0.5
    
    # Pitching adjustment
    pitching_edge = (features['away_pitcher_era'] - features['home_pitcher_era']) / 10
    base_prob += pitching_edge
    
    # Batting adjustment
    batting_edge = (features['home_team_woba'] - features['away_team_woba']) * 2
    base_prob += batting_edge
    
    # Park adjustment
    park_adjustment = (features['park_factor'] - 1.0) * 0.1
    base_prob += park_adjustment
    
    # Rest adjustment
    rest_adjustment = features['rest_advantage'] * 0.02
    base_prob += rest_adjustment
    
    # Bound probability
    home_win_prob = max(0.35, min(0.65, base_prob))
    
    return {
        'home_win_probability': home_win_prob,
        'away_win_probability': 1 - home_win_prob,
        'confidence': 0.6,  # Conservative confidence
        'features': features
    }
```

### 2. **Simple Bet Evaluation**
```python
def evaluate_bet_simple(prediction, odds):
    """Simple bet evaluation without complex calculations."""
    
    our_prob = prediction['home_win_probability']
    implied_prob = odds_to_probability(odds)
    
    # Calculate edge
    edge = our_prob - implied_prob
    
    # Apply conservative adjustment
    adjusted_edge = edge * 0.7
    
    # Calculate simple EV
    if odds > 0:
        ev = (our_prob * odds / 100) - (1 - our_prob)
    else:
        ev = (our_prob * 100 / abs(odds)) - (1 - our_prob)
    
    # Apply conservative adjustment to EV
    adjusted_ev = ev * 0.7
    
    return {
        'edge': adjusted_edge,
        'ev': adjusted_ev,
        'our_prob': our_prob,
        'implied_prob': implied_prob,
        'odds': odds,
        'recommendation': 'bet' if adjusted_edge > 0.015 else 'pass'
    }
```

## Practical Implementation

### 1. **Minimal Data Requirements**
```python
required_data = {
    'pitching': ['last_30_era', 'k_per_9_last_30', 'bb_per_9_last_30'],
    'batting': ['woba_last_30'],
    'situational': ['park_factor', 'rest_days'],
    'odds': ['current_odds', 'opening_odds']
}

# Total features: ~10 (vs. 100+ in original)
```

### 2. **Simple Execution Process**
```python
def execute_bet_simple(game_data, odds):
    """Simple bet execution process."""
    
    # 1. Analyze game
    prediction = analyze_game_ultra_simple(game_data)
    
    # 2. Evaluate bet
    evaluation = evaluate_bet_simple(prediction, odds)
    
    # 3. Check if we should bet
    if evaluation['recommendation'] == 'pass':
        return {'action': 'pass', 'reason': 'insufficient_edge'}
    
    # 4. Calculate stake
    stake = calculate_ultra_conservative_stake(
        bankroll=10000,
        edge=evaluation['edge'],
        confidence=prediction['confidence']
    )
    
    # 5. Check limits
    if stake > ultra_conservative_limits['max_single_bet']:
        stake = ultra_conservative_limits['max_single_bet']
    
    return {
        'action': 'bet',
        'stake': stake,
        'stake_amount': stake * 10000,
        'edge': evaluation['edge'],
        'ev': evaluation['ev'],
        'confidence': prediction['confidence']
    }
```

## Expected Ultra-Conservative Performance

### 1. **Realistic Projections**
```python
ultra_conservative_performance = {
    'win_rate': 0.52,           # 52% win rate
    'average_ev': 0.008,        # 0.8% average EV
    'annual_roi': 0.03,         # 3% annual ROI
    'sharpe_ratio': 0.3,        # 0.3 Sharpe ratio
    'max_drawdown': 0.20,       # 20% max drawdown
    'recovery_time': '6-12 months',
    'value_bet_rate': 0.05,     # 5% of games offer value
    'positive_clv_rate': 0.52,  # 52% of bets beat closing line
    'average_bet_size': 0.005   # 0.5% average bet size
}
```

### 2. **Risk-Adjusted Metrics**
```python
def calculate_ultra_conservative_metrics(returns):
    """Calculate ultra-conservative performance metrics."""
    
    # Basic metrics
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Risk-adjusted metrics
    sharpe = avg_return / std_return if std_return > 0 else 0
    
    # Maximum drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Calmar ratio
    calmar = avg_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'avg_return': avg_return,
        'std_return': std_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar,
        'win_rate': sum(1 for r in returns if r > 0) / len(returns)
    }
```

## Implementation Timeline

### Phase 1: Foundation (Months 1-3)
- Implement single logistic regression model
- Set up basic CLV tracking
- Establish ultra-conservative limits

### Phase 2: Paper Trading (Months 4-6)
- Track 100+ games with paper money
- Monitor CLV performance
- Refine edge thresholds

### Phase 3: Small Bankroll (Months 7-9)
- Start with $1,000 bankroll
- Maximum $10 per bet (1%)
- Focus on learning, not profit

### Phase 4: Scale Up (Months 10+)
- Increase bankroll only if profitable
- Maintain same percentage limits
- Continue monitoring CLV

## Conclusion

This ultra-conservative approach addresses all remaining issues:

1. **Single Model**: Eliminates ensemble complexity and overfitting risk
2. **Correct CLV**: Fixed calculation and proper tracking
3. **No Sharp Action**: Removes unverifiable assumptions about market microstructure
4. **Minimal Features**: 10 features vs. 100+ in original
5. **Ultra-Conservative Limits**: 1% max per bet, 5% daily limit
6. **Realistic Expectations**: 3% annual ROI, 52% win rate

The strategy prioritizes learning and sustainability over profit maximization, recognizing that most games offer no value and that small, consistent edges are more valuable than large, uncertain ones. 