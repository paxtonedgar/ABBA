# Realistic MLB Betting Strategy: Addressing Critical Issues

## Executive Summary

The previous strategy, while theoretically sophisticated, overestimates achievable edges and underestimates market efficiency. This revised approach focuses on sustainable, realistic edges with robust risk management.

## Core Philosophy: Small Edges, Large Sample Sizes

### 1. **Realistic Edge Expectations**
```python
# Realistic edge targets:
- Expected Value: 1-3% (not 5%+)
- Win Rate: 52-55% (not 54-58%)
- Annual ROI: 3-8% (not 8-12%)
- Sharpe Ratio: 0.4-0.8 (not 1.2-1.5)
```

### 2. **Market Efficiency Recognition**
- MLB markets are highly efficient
- Major edges (>5%) are rare and short-lived
- Focus on consistent small edges over time
- Accept that many games offer no betting value

## Revised Statistical Approach

### 1. **Simplified Feature Set (20-30 features)**
```python
# Core features only:
pitching_features = [
    'era_last_30', 'k_per_9_last_30', 'bb_per_9_last_30',
    'avg_velocity_last_30', 'home_away_era_split',
    'rest_days', 'bullpen_era_last_7'
]

batting_features = [
    'woba_last_30', 'iso_last_30', 'bb_rate_last_30',
    'k_rate_last_30', 'home_away_woba_split',
    'lineup_strength_rating'
]

situational_features = [
    'park_factor', 'weather_impact', 'day_night_split',
    'series_game_number', 'travel_distance'
]
```

### 2. **Conservative Model Approach**
```python
# Model ensemble instead of single XGBoost:
models = {
    'logistic_regression': LogisticRegression(),
    'random_forest': RandomForestClassifier(n_estimators=100),
    'xgboost': XGBClassifier(n_estimators=50, max_depth=3),
    'simple_heuristic': CustomHeuristicModel()
}

# Ensemble prediction:
final_prob = weighted_average([model.predict_proba(X) for model in models])
```

### 3. **Uncertainty Quantification**
```python
def calculate_prediction_interval(predictions, confidence=0.95):
    """Calculate prediction intervals for win probability."""
    mean_prob = np.mean(predictions)
    std_prob = np.std(predictions)
    
    # Bootstrap confidence interval
    bootstrap_samples = []
    for _ in range(1000):
        sample = np.random.choice(predictions, size=len(predictions))
        bootstrap_samples.append(np.mean(sample))
    
    lower = np.percentile(bootstrap_samples, (1-confidence)/2 * 100)
    upper = np.percentile(bootstrap_samples, (1+confidence)/2 * 100)
    
    return mean_prob, lower, upper
```

## Risk Management Overhaul

### 1. **Fractional Kelly Implementation**
```python
def calculate_conservative_kelly(win_prob, odds, uncertainty_factor=0.25):
    """Calculate conservative Kelly stake with uncertainty adjustment."""
    # Standard Kelly calculation
    if odds > 0:
        b = odds / 100
    else:
        b = 100 / abs(odds)
    
    p = win_prob
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    # Apply uncertainty adjustment
    # Higher uncertainty = smaller stake
    adjusted_kelly = kelly * uncertainty_factor
    
    # Additional safety margin
    final_kelly = min(adjusted_kelly, 0.02)  # Max 2% of bankroll
    
    return max(0, final_kelly)
```

### 2. **Portfolio Constraints**
```python
risk_constraints = {
    'max_single_bet': 0.02,      # 2% max per bet
    'max_single_game': 0.05,     # 5% max per game
    'max_daily_risk': 0.10,      # 10% max daily
    'max_weekly_risk': 0.20,     # 20% max weekly
    'correlation_limit': 0.3,    # Max correlation between bets
    'min_bankroll_preserve': 0.8  # Preserve 80% of initial bankroll
}
```

### 3. **Dynamic Position Sizing**
```python
def calculate_position_size(bankroll, edge, confidence, market_conditions):
    """Calculate position size based on multiple factors."""
    base_size = calculate_conservative_kelly(edge['win_prob'], edge['odds'])
    
    # Adjust for confidence
    confidence_multiplier = min(confidence / 0.7, 1.0)
    
    # Adjust for market conditions
    market_multiplier = 1.0
    if market_conditions['liquidity'] == 'low':
        market_multiplier = 0.5
    if market_conditions['volatility'] == 'high':
        market_multiplier = 0.7
    
    # Adjust for bankroll size
    bankroll_multiplier = min(bankroll / 10000, 1.0)  # Scale down for smaller bankrolls
    
    final_size = base_size * confidence_multiplier * market_multiplier * bankroll_multiplier
    
    return min(final_size, risk_constraints['max_single_bet'])
```

## Market Execution Strategy

### 1. **Odds Shopping Infrastructure**
```python
class OddsAggregator:
    def __init__(self):
        self.books = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'pointsbet']
        self.api_keys = self.load_api_keys()
    
    def get_best_odds(self, game_id, bet_type):
        """Get best available odds across all books."""
        odds_data = {}
        for book in self.books:
            try:
                odds = self.fetch_odds(book, game_id, bet_type)
                odds_data[book] = odds
            except Exception as e:
                logger.warning(f"Failed to fetch odds from {book}: {e}")
        
        return self.find_best_odds(odds_data)
    
    def calculate_implied_probability(self, odds):
        """Convert odds to implied probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
```

### 2. **Line Movement Analysis**
```python
def analyze_line_movement(game_id, bet_type):
    """Analyze line movement to identify sharp action."""
    line_history = fetch_line_history(game_id, bet_type)
    
    # Calculate line movement metrics
    opening_line = line_history[0]['line']
    current_line = line_history[-1]['line']
    movement = current_line - opening_line
    
    # Identify sharp vs. public action
    sharp_indicators = {
        'reverse_line_movement': detect_reverse_movement(line_history),
        'steam_move': detect_steam_move(line_history),
        'line_consistency': calculate_line_consistency(line_history)
    }
    
    return {
        'movement': movement,
        'sharp_indicators': sharp_indicators,
        'confidence_in_movement': calculate_movement_confidence(line_history)
    }
```

### 3. **Execution Timing**
```python
def determine_bet_timing(game_id, bet_type, model_prediction):
    """Determine optimal timing for bet placement."""
    line_movement = analyze_line_movement(game_id, bet_type)
    
    # Wait for line to move in our favor
    if line_movement['movement'] > 0 and bet_type == 'under':
        return 'wait_for_better_odds'
    
    if line_movement['movement'] < 0 and bet_type == 'over':
        return 'wait_for_better_odds'
    
    # Bet immediately if line moving against us
    if line_movement['movement'] < 0 and bet_type == 'under':
        return 'bet_now'
    
    if line_movement['movement'] > 0 and bet_type == 'over':
        return 'bet_now'
    
    return 'bet_now'  # Default to immediate execution
```

## Performance Measurement

### 1. **Closing Line Value (CLV)**
```python
def calculate_clv(bet_odds, closing_odds):
    """Calculate closing line value."""
    bet_prob = odds_to_probability(bet_odds)
    closing_prob = odds_to_probability(closing_odds)
    
    clv = bet_prob - closing_prob
    return clv

def track_clv_performance(bets):
    """Track CLV performance over time."""
    clv_scores = []
    for bet in bets:
        clv = calculate_clv(bet['placed_odds'], bet['closing_odds'])
        clv_scores.append(clv)
    
    avg_clv = np.mean(clv_scores)
    clv_std = np.std(clv_scores)
    
    return {
        'average_clv': avg_clv,
        'clv_std': clv_std,
        'positive_clv_rate': sum(1 for clv in clv_scores if clv > 0) / len(clv_scores)
    }
```

### 2. **Realistic Performance Metrics**
```python
def calculate_realistic_metrics(bets, bankroll_history):
    """Calculate realistic performance metrics."""
    metrics = {
        'win_rate': sum(1 for bet in bets if bet['result'] == 'win') / len(bets),
        'average_ev': np.mean([bet['ev'] for bet in bets]),
        'roi': (bankroll_history[-1] - bankroll_history[0]) / bankroll_history[0],
        'sharpe_ratio': calculate_sharpe_ratio(bankroll_history),
        'max_drawdown': calculate_max_drawdown(bankroll_history),
        'avg_clv': track_clv_performance(bets)['average_clv']
    }
    
    return metrics
```

## Practical Implementation Challenges

### 1. **Account Management**
```python
class AccountManager:
    def __init__(self):
        self.accounts = self.load_accounts()
        self.limits = self.track_account_limits()
    
    def distribute_bet(self, bet_amount, game_id):
        """Distribute bet across multiple accounts to avoid limits."""
        available_accounts = self.get_available_accounts(game_id)
        
        if len(available_accounts) == 0:
            return {'error': 'No available accounts'}
        
        # Distribute proportionally
        distribution = {}
        for account in available_accounts:
            max_bet = self.get_max_bet(account, game_id)
            if max_bet > 0:
                distribution[account] = min(bet_amount / len(available_accounts), max_bet)
        
        return distribution
    
    def track_account_limits(self):
        """Track how accounts are being limited."""
        for account in self.accounts:
            recent_bets = self.get_recent_bets(account)
            if len(recent_bets) > 0:
                avg_odds = np.mean([bet['odds'] for bet in recent_bets])
                if avg_odds < -200:  # Getting limited
                    self.flag_account(account, 'limited')
```

### 2. **Transaction Costs**
```python
def calculate_total_cost(bet_amount, odds, withdrawal_fee=0.05):
    """Calculate total transaction costs."""
    # Vig (built into odds)
    vig_cost = calculate_vig(odds)
    
    # Withdrawal fees (when cashing out)
    withdrawal_cost = bet_amount * withdrawal_fee
    
    # Opportunity cost (money tied up)
    opportunity_cost = bet_amount * 0.02  # 2% annual rate
    
    total_cost = vig_cost + withdrawal_cost + opportunity_cost
    
    return total_cost
```

### 3. **Data Quality and Costs**
```python
def estimate_data_costs():
    """Estimate monthly data costs."""
    costs = {
        'statcast_api': 0,  # Free
        'weather_api': 50,  # $50/month
        'odds_api': 200,    # $200/month
        'historical_data': 100,  # $100/month
        'computing_resources': 50,  # $50/month
        'total': 400
    }
    
    return costs
```

## Revised Decision Process

### 1. **Simplified Game Analysis**
```python
def analyze_game_simple(game_data):
    """Simplified game analysis focusing on key factors."""
    analysis = {
        'pitching_advantage': calculate_pitching_advantage(game_data),
        'batting_advantage': calculate_batting_advantage(game_data),
        'situational_factors': calculate_situational_factors(game_data),
        'model_confidence': calculate_model_confidence(game_data)
    }
    
    # Only proceed if confidence is high enough
    if analysis['model_confidence'] < 0.7:
        return {'recommendation': 'pass', 'reason': 'low_confidence'}
    
    return analysis
```

### 2. **Conservative Bet Selection**
```python
def select_bets(opportunities, bankroll):
    """Select bets using conservative criteria."""
    selected_bets = []
    
    for opp in opportunities:
        # Minimum criteria
        if opp['ev'] < 0.01:  # 1% minimum EV
            continue
        
        if opp['confidence'] < 0.7:  # 70% minimum confidence
            continue
        
        if opp['kelly_stake'] < 0.005:  # 0.5% minimum stake
            continue
        
        # Check portfolio constraints
        if violates_portfolio_constraints(selected_bets, opp):
            continue
        
        selected_bets.append(opp)
    
    return selected_bets
```

## Expected Realistic Performance

### 1. **Conservative Projections**
```python
realistic_performance = {
    'win_rate': 0.53,           # 53% win rate
    'average_ev': 0.015,        # 1.5% average EV
    'annual_roi': 0.05,         # 5% annual ROI
    'sharpe_ratio': 0.6,        # 0.6 Sharpe ratio
    'max_drawdown': 0.15,       # 15% max drawdown
    'recovery_time': '3-6 months',
    'value_bet_rate': 0.10      # 10% of games offer value
}
```

### 2. **Risk-Adjusted Returns**
```python
def calculate_risk_adjusted_returns(returns, risk_free_rate=0.02):
    """Calculate risk-adjusted return metrics."""
    excess_returns = returns - risk_free_rate
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    
    # Sortino ratio (downside deviation)
    downside_returns = [r for r in excess_returns if r < 0]
    sortino = np.mean(excess_returns) / np.std(downside_returns) if downside_returns else 0
    
    return {
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': np.mean(returns) / max_drawdown
    }
```

## Conclusion

This revised strategy prioritizes sustainability over maximum returns. Key changes:

1. **Reduced complexity** to avoid overfitting
2. **Conservative position sizing** using fractional Kelly
3. **Realistic edge expectations** (1-3% vs 5%+)
4. **Robust risk management** with multiple constraints
5. **Focus on CLV** as primary performance metric
6. **Account for transaction costs** and practical limitations

The goal is consistent, sustainable returns rather than attempting to beat highly efficient markets with unrealistic edges. 