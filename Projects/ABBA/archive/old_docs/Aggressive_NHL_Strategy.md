# Aggressive NHL Betting Strategy: Advanced Hockey Analytics

## Core Philosophy: Hockey-Specific Edge Detection

NHL betting requires specialized analysis due to:
- **High variance**: Low-scoring games with significant randomness
- **Goalie impact**: Single player can dominate outcomes
- **Special teams**: Power play/penalty kill efficiency crucial
- **Possession metrics**: Corsi, Fenwick, xGF more predictive than basic stats
- **Schedule effects**: Back-to-backs, travel, rest days critical

## Advanced NHL Predictive Model

### 1. **Hockey-Specific Feature Engineering**
```python
class AggressiveNHLPredictor:
    def __init__(self):
        self.models = {
            'primary': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1),
            'ensemble': RandomForestClassifier(n_estimators=100, max_depth=8),
            'neural': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
        }
        self.feature_importance = {}
        
    def extract_advanced_hockey_features(self, game_data):
        """Extract 60+ hockey-specific predictive features."""
        features = {}
        
        # Goaltending dominance features
        features['goalie_save_percentage_diff'] = (
            game_data['home_goalie']['save_percentage'] - 
            game_data['away_goalie']['save_percentage']
        )
        features['goalie_gsaa_diff'] = (
            game_data['home_goalie']['gsaa'] - 
            game_data['away_goalie']['gsaa']
        )  # Goals Saved Above Average
        features['goalie_high_danger_save_pct_diff'] = (
            game_data['home_goalie']['high_danger_save_pct'] - 
            game_data['away_goalie']['high_danger_save_pct']
        )
        features['goalie_recent_form_diff'] = (
            game_data['home_goalie']['last_5_games_save_pct'] - 
            game_data['away_goalie']['last_5_games_save_pct']
        )
        
        # Possession metrics (most predictive in hockey)
        features['corsi_for_percentage_diff'] = (
            game_data['home_team']['corsi_for_percentage'] - 
            game_data['away_team']['corsi_for_percentage']
        )
        features['fenwick_for_percentage_diff'] = (
            game_data['home_team']['fenwick_for_percentage'] - 
            game_data['away_team']['fenwick_for_percentage']
        )
        features['expected_goals_for_percentage_diff'] = (
            game_data['home_team']['xgf_percentage'] - 
            game_data['away_team']['xgf_percentage']
        )
        features['scoring_chances_for_percentage_diff'] = (
            game_data['home_team']['scf_percentage'] - 
            game_data['away_team']['scf_percentage']
        )
        features['high_danger_chances_for_percentage_diff'] = (
            game_data['home_team']['hdcf_percentage'] - 
            game_data['away_team']['hdcf_percentage']
        )
        
        # Special teams efficiency
        features['power_play_percentage_diff'] = (
            game_data['home_team']['power_play_percentage'] - 
            game_data['away_team']['power_play_percentage']
        )
        features['penalty_kill_percentage_diff'] = (
            game_data['home_team']['penalty_kill_percentage'] - 
            game_data['away_team']['penalty_kill_percentage']
        )
        features['power_play_opportunities_diff'] = (
            game_data['home_team']['power_play_opportunities_per_game'] - 
            game_data['away_team']['power_play_opportunities_per_game']
        )
        
        # Recent form features (last 5, 10, 20 games)
        for period in [5, 10, 20]:
            features[f'home_team_win_percentage_{period}g'] = game_data['home_team'][f'win_percentage_last_{period}']
            features[f'away_team_win_percentage_{period}g'] = game_data['away_team'][f'win_percentage_last_{period}']
            features[f'home_team_goals_for_per_game_{period}g'] = game_data['home_team'][f'gf_per_game_last_{period}']
            features[f'away_team_goals_for_per_game_{period}g'] = game_data['away_team'][f'gf_per_game_last_{period}']
            features[f'home_team_goals_against_per_game_{period}g'] = game_data['home_team'][f'ga_per_game_last_{period}']
            features[f'away_team_goals_against_per_game_{period}g'] = game_data['away_team'][f'ga_per_game_last_{period}']
            features[f'home_team_corsi_percentage_{period}g'] = game_data['home_team'][f'corsi_percentage_last_{period}']
            features[f'away_team_corsi_percentage_{period}g'] = game_data['away_team'][f'corsi_percentage_last_{period}']
        
        # Situational features
        features['home_advantage'] = 1.0  # Home teams win ~55% of games
        features['rest_advantage'] = game_data['home_rest_days'] - game_data['away_rest_days']
        features['back_to_back_penalty'] = 1 if game_data['away_back_to_back'] else 0
        features['travel_distance_penalty'] = game_data['away_travel_distance'] / 1000
        
        # Arena-specific features
        features['home_arena_altitude'] = game_data['home_arena']['altitude']
        features['home_arena_ice_quality'] = game_data['home_arena']['ice_quality_rating']
        features['home_arena_crowd_factor'] = game_data['home_arena']['crowd_factor']
        
        # Weather impact (affects travel, ice conditions)
        features['weather_temperature'] = game_data['weather']['temperature']
        features['weather_humidity'] = game_data['weather']['humidity']
        features['weather_pressure'] = game_data['weather']['pressure']
        
        # Team depth and injuries
        features['home_team_injury_impact'] = self._calculate_injury_impact(game_data['home_team']['injuries'])
        features['away_team_injury_impact'] = self._calculate_injury_impact(game_data['away_team']['injuries'])
        features['home_team_depth_score'] = self._calculate_team_depth(game_data['home_team']['roster'])
        features['away_team_depth_score'] = self._calculate_team_depth(game_data['away_team']['roster'])
        
        # Historical matchup data
        features['h2h_home_advantage'] = game_data['h2h_stats']['home_wins'] / game_data['h2h_stats']['total_games']
        features['h2h_goals_per_game'] = game_data['h2h_stats']['total_goals'] / game_data['h2h_stats']['total_games']
        features['h2h_power_play_efficiency'] = game_data['h2h_stats']['home_power_play_goals'] / game_data['h2h_stats']['home_power_play_opportunities']
        
        # Momentum and streak features
        features['home_team_current_streak'] = game_data['home_team']['current_streak']
        features['away_team_current_streak'] = game_data['away_team']['current_streak']
        features['home_team_momentum_score'] = self._calculate_momentum_score(game_data['home_team']['last_10_games'])
        features['away_team_momentum_score'] = self._calculate_momentum_score(game_data['away_team']['last_10_games'])
        
        return features
    
    def _calculate_injury_impact(self, injuries):
        """Calculate impact of injuries on team performance."""
        if not injuries:
            return 0.0
        
        total_impact = 0.0
        for injury in injuries:
            # Weight by player importance and injury severity
            player_importance = injury['player_importance']  # 0-1 scale
            injury_severity = injury['injury_severity']  # 0-1 scale
            total_impact += player_importance * injury_severity
        
        return total_impact / len(injuries)
    
    def _calculate_team_depth(self, roster):
        """Calculate team depth score based on roster quality."""
        forward_depth = np.mean([p['forward_rating'] for p in roster['forwards']])
        defense_depth = np.mean([p['defense_rating'] for p in roster['defensemen']])
        goalie_depth = np.mean([p['goalie_rating'] for p in roster['goalies']])
        
        # Weight by position importance
        return (forward_depth * 0.4 + defense_depth * 0.35 + goalie_depth * 0.25)
    
    def _calculate_momentum_score(self, last_10_games):
        """Calculate momentum score based on recent performance."""
        if not last_10_games:
            return 0.0
        
        # Weight recent games more heavily
        weights = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        results = np.array([1 if game['result'] == 'W' else 0 for game in last_10_games])
        
        return np.average(results, weights=weights[:len(results)])
```

### 2. **Advanced Hockey Edge Detection**
```python
def calculate_aggressive_hockey_edge(prediction, odds, market_data, game_data):
    """Calculate edge with hockey-specific market analysis."""
    
    our_prob = prediction['home_win_probability']
    implied_prob = odds_to_probability(odds)
    
    # Base edge
    raw_edge = our_prob - implied_prob
    
    # Hockey-specific market efficiency adjustment
    market_efficiency = market_data.get('efficiency_score', 0.7)  # NHL markets less efficient
    edge_adjustment = 1 + (1 - market_efficiency) * 0.7  # Bigger adjustment for NHL
    
    # Goalie factor adjustment
    goalie_factor = market_data.get('goalie_factor', 1.0)
    if goalie_factor > 1.2:  # Elite goalie playing
        edge_adjustment *= 1.3
    elif goalie_factor < 0.8:  # Weak goalie playing
        edge_adjustment *= 1.2
    
    # Public betting adjustment (NHL has strong public biases)
    public_betting = market_data.get('public_betting_percentage', 0.5)
    if public_betting > 0.7:  # Heavy public action
        edge_adjustment *= 1.4  # Bet against public
    elif public_betting < 0.3:  # Light public action
        edge_adjustment *= 1.2  # Bet with sharp money
    
    # Schedule factor adjustment
    schedule_factor = market_data.get('schedule_factor', 1.0)
    if schedule_factor < 0.8:  # Back-to-back, travel issues
        edge_adjustment *= 1.3
    
    # Line movement analysis
    line_movement = market_data.get('line_movement', 0)
    if abs(line_movement) > 20:  # Significant line movement
        edge_adjustment *= 1.2
    
    adjusted_edge = raw_edge * edge_adjustment
    
    return {
        'raw_edge': raw_edge,
        'adjusted_edge': adjusted_edge,
        'edge_adjustment': edge_adjustment,
        'market_efficiency': market_efficiency,
        'goalie_factor': goalie_factor,
        'public_betting': public_betting,
        'schedule_factor': schedule_factor,
        'line_movement': line_movement
    }
```

## Aggressive NHL Kelly Criterion

### 1. **Hockey-Specific Position Sizing**
```python
def calculate_aggressive_hockey_kelly(win_prob, odds, confidence, bankroll, game_data):
    """Calculate aggressive Kelly stake with hockey-specific adjustments."""
    
    # Standard Kelly calculation
    if odds > 0:
        b = odds / 100
    else:
        b = 100 / abs(odds)
    
    p = win_prob
    q = 1 - p
    
    kelly_fraction = (b * p - q) / b
    
    # Confidence adjustment (higher confidence = bigger stake)
    confidence_multiplier = 0.6 + (confidence * 0.4)  # 60% to 100%
    
    # Bankroll adjustment (larger bankroll = bigger stakes)
    bankroll_multiplier = min(bankroll / 10000, 2.5)  # Scale up to 2.5x for larger bankrolls
    
    # Hockey-specific adjustments
    hockey_multiplier = 1.0
    
    # Goalie quality adjustment
    goalie_advantage = abs(game_data['home_goalie']['save_percentage'] - game_data['away_goalie']['save_percentage'])
    if goalie_advantage > 0.02:  # 2%+ save percentage difference
        hockey_multiplier *= 1.3
    
    # Possession advantage adjustment
    possession_advantage = abs(game_data['home_team']['corsi_for_percentage'] - game_data['away_team']['corsi_for_percentage'])
    if possession_advantage > 5:  # 5%+ possession advantage
        hockey_multiplier *= 1.2
    
    # Special teams advantage adjustment
    pp_advantage = abs(game_data['home_team']['power_play_percentage'] - game_data['away_team']['power_play_percentage'])
    pk_advantage = abs(game_data['home_team']['penalty_kill_percentage'] - game_data['away_team']['penalty_kill_percentage'])
    if pp_advantage > 5 or pk_advantage > 5:  # 5%+ special teams advantage
        hockey_multiplier *= 1.15
    
    # Schedule advantage adjustment
    if game_data['away_back_to_back']:
        hockey_multiplier *= 1.25
    
    # Market opportunity adjustment
    market_opportunity = 1.0
    if kelly_fraction > 0.15:  # Very big edge
        market_opportunity = 1.8
    elif kelly_fraction > 0.08:  # Big edge
        market_opportunity = 1.4
    elif kelly_fraction > 0.04:  # Medium edge
        market_opportunity = 1.2
    
    final_kelly = kelly_fraction * confidence_multiplier * bankroll_multiplier * hockey_multiplier * market_opportunity
    
    # Cap at 12% of bankroll for very large edges (higher than MLB due to higher variance)
    final_kelly = min(final_kelly, 0.12)
    
    return max(0, final_kelly)
```

### 2. **Hockey-Specific Risk Management**
```python
class AggressiveNHLRiskManager:
    def __init__(self, initial_bankroll):
        self.bankroll = initial_bankroll
        self.max_single_bet = 0.12  # 12% max per bet (higher than MLB)
        self.max_daily_risk = 0.30  # 30% max daily
        self.max_weekly_risk = 0.60  # 60% max weekly
        self.daily_risk_used = 0
        self.weekly_risk_used = 0
        
        # Hockey-specific risk factors
        self.goalie_risk_factor = 1.0
        self.schedule_risk_factor = 1.0
        self.possesssion_risk_factor = 1.0
        
    def calculate_optimal_stake(self, edge_analysis, prediction, odds, game_data):
        """Calculate optimal stake for maximum growth in NHL."""
        
        # Base Kelly stake
        kelly_stake = calculate_aggressive_hockey_kelly(
            prediction['home_win_probability'],
            odds,
            prediction['confidence'],
            self.bankroll,
            game_data
        )
        
        # Edge-based adjustment
        edge_multiplier = 1.0
        if edge_analysis['adjusted_edge'] > 0.12:  # 12%+ edge
            edge_multiplier = 1.8
        elif edge_analysis['adjusted_edge'] > 0.08:  # 8%+ edge
            edge_multiplier = 1.4
        elif edge_analysis['adjusted_edge'] > 0.04:  # 4%+ edge
            edge_multiplier = 1.2
        elif edge_analysis['adjusted_edge'] < 0.02:  # <2% edge
            edge_multiplier = 0.5
        
        # Hockey-specific adjustments
        hockey_multiplier = 1.0
        
        # Goalie quality adjustment
        goalie_advantage = abs(game_data['home_goalie']['save_percentage'] - game_data['away_goalie']['save_percentage'])
        if goalie_advantage > 0.03:  # 3%+ save percentage difference
            hockey_multiplier *= 1.4
        
        # Possession advantage adjustment
        possession_advantage = abs(game_data['home_team']['corsi_for_percentage'] - game_data['away_team']['corsi_for_percentage'])
        if possession_advantage > 7:  # 7%+ possession advantage
            hockey_multiplier *= 1.3
        
        # Schedule advantage adjustment
        if game_data['away_back_to_back']:
            hockey_multiplier *= 1.3
        
        # Public betting adjustment
        if edge_analysis['public_betting'] > 0.7:  # Heavy public action
            hockey_multiplier *= 1.2  # Bet against public
        
        final_stake = kelly_stake * edge_multiplier * hockey_multiplier
        
        # Apply risk limits
        final_stake = min(final_stake, self.max_single_bet)
        final_stake = min(final_stake, self.max_daily_risk - self.daily_risk_used)
        final_stake = min(final_stake, self.max_weekly_risk - self.weekly_risk_used)
        
        return max(0.015, final_stake)  # Minimum 1.5% stake
```

## Advanced NHL Market Analysis

### 1. **Hockey-Specific Odds Monitoring**
```python
class AggressiveNHLOddsMonitor:
    def __init__(self):
        self.odds_history = {}
        self.books = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'pointsbet']
        
    def monitor_hockey_odds_movement(self, game_id, bet_type):
        """Monitor odds movement with hockey-specific analysis."""
        
        current_odds = self.get_current_odds(game_id, bet_type)
        opening_odds = self.get_opening_odds(game_id, bet_type)
        
        # Calculate line movement
        line_movement = current_odds - opening_odds
        
        # Hockey-specific sharp action detection
        sharp_indicators = self.detect_hockey_sharp_action(game_id, bet_type)
        
        # Goalie confirmation analysis
        goalie_analysis = self.analyze_goalie_impact(game_id)
        
        # Public betting analysis
        public_analysis = self.analyze_public_betting_patterns(game_id, bet_type)
        
        return {
            'current_odds': current_odds,
            'opening_odds': opening_odds,
            'line_movement': line_movement,
            'sharp_indicators': sharp_indicators,
            'goalie_analysis': goalie_analysis,
            'public_analysis': public_analysis,
            'market_efficiency': self.calculate_hockey_market_efficiency(game_id, bet_type)
        }
    
    def detect_hockey_sharp_action(self, game_id, bet_type):
        """Detect sharp money movement in hockey markets."""
        
        line_history = self.get_line_history(game_id, bet_type)
        
        sharp_indicators = {
            'reverse_line_movement': False,
            'steam_move': False,
            'goalie_confirmation': False,
            'possession_confirmation': False
        }
        
        # Detect reverse line movement
        if len(line_history) > 3:
            public_betting = self.get_public_betting_percentages(game_id, bet_type)
            line_direction = np.sign(line_history[-1] - line_history[0])
            public_direction = np.sign(public_betting['home'] - 0.5)
            
            if line_direction != public_direction:
                sharp_indicators['reverse_line_movement'] = True
        
        # Detect steam moves (more common in hockey)
        if len(line_history) > 2:
            recent_movement = abs(line_history[-1] - line_history[-2])
            if recent_movement > 25:  # 25+ point move in hockey
                sharp_indicators['steam_move'] = True
        
        return sharp_indicators
    
    def analyze_goalie_impact(self, game_id):
        """Analyze goalie impact on betting patterns."""
        
        goalie_data = self.get_goalie_data(game_id)
        
        return {
            'home_goalie_quality': goalie_data['home']['save_percentage'],
            'away_goalie_quality': goalie_data['away']['save_percentage'],
            'goalie_advantage': goalie_data['home']['save_percentage'] - goalie_data['away']['save_percentage'],
            'goalie_recent_form': goalie_data['home']['last_5_games_save_pct'] - goalie_data['away']['last_5_games_save_pct']
        }
```

### 2. **Hockey-Specific Arbitrage Detection**
```python
def detect_hockey_arbitrage_opportunities(game_id, bet_type):
    """Detect arbitrage opportunities in hockey markets."""
    
    odds_by_book = {}
    for book in ['draftkings', 'fanduel', 'betmgm', 'caesars', 'pointsbet']:
        try:
            odds = fetch_hockey_odds(book, game_id, bet_type)
            odds_by_book[book] = odds
        except:
            continue
    
    # Find best odds for each side
    home_odds = [(book, odds['home']) for book, odds in odds_by_book.items()]
    away_odds = [(book, odds['away']) for book, odds in odds_by_book.items()]
    
    best_home = max(home_odds, key=lambda x: x[1])
    best_away = max(away_odds, key=lambda x: x[1])
    
    # Calculate arbitrage percentage
    home_prob = odds_to_probability(best_home[1])
    away_prob = odds_to_probability(best_away[1])
    
    total_prob = home_prob + away_prob
    arbitrage_percentage = (1 - total_prob) / total_prob
    
    if arbitrage_percentage > 0.015:  # 1.5%+ arbitrage (higher threshold for hockey)
        return {
            'arbitrage_percentage': arbitrage_percentage,
            'home_bet': {'book': best_home[0], 'odds': best_home[1]},
            'away_bet': {'book': best_away[0], 'odds': best_away[1]},
            'optimal_stakes': calculate_arbitrage_stakes(home_prob, away_prob)
        }
    
    return None
```

## Aggressive NHL Decision Framework

### 1. **High-Confidence Hockey Betting Criteria**
```python
def evaluate_aggressive_hockey_bet(prediction, edge_analysis, market_data, game_data):
    """Evaluate betting opportunity with hockey-specific criteria."""
    
    # Minimum thresholds for aggressive hockey betting
    min_edge = 0.04  # 4% minimum edge (higher than MLB due to variance)
    min_confidence = 0.80  # 80% minimum confidence
    min_model_agreement = 0.75  # 75% model agreement
    min_goalie_advantage = 0.02  # 2% minimum goalie advantage
    
    # Check if we meet aggressive criteria
    meets_criteria = (
        edge_analysis['adjusted_edge'] >= min_edge and
        prediction['confidence'] >= min_confidence and
        prediction['model_agreement'] >= min_model_agreement and
        abs(game_data['home_goalie']['save_percentage'] - game_data['away_goalie']['save_percentage']) >= min_goalie_advantage
    )
    
    if not meets_criteria:
        return {'recommendation': 'pass', 'reason': 'insufficient_aggressive_criteria'}
    
    # Calculate optimal stake
    risk_manager = AggressiveNHLRiskManager(bankroll=100000)
    optimal_stake = risk_manager.calculate_optimal_stake(
        edge_analysis, prediction, market_data['current_odds'], game_data
    )
    
    # Calculate expected value
    ev = calculate_expected_value(
        prediction['home_win_probability'],
        market_data['current_odds']
    )
    
    return {
        'recommendation': 'bet',
        'stake': optimal_stake,
        'stake_amount': optimal_stake * 100000,
        'edge': edge_analysis['adjusted_edge'],
        'ev': ev,
        'confidence': prediction['confidence'],
        'model_agreement': prediction['model_agreement'],
        'goalie_advantage': abs(game_data['home_goalie']['save_percentage'] - game_data['away_goalie']['save_percentage'])
    }
```

### 2. **Hockey-Specific Portfolio Optimization**
```python
class AggressiveNHLPortfolioManager:
    def __init__(self, bankroll):
        self.bankroll = bankroll
        self.active_bets = []
        self.max_correlation = 0.4  # Higher correlation limit for hockey
        self.max_goalie_exposure = 0.20  # Max 20% exposure to single goalie
        
    def add_hockey_bet(self, bet):
        """Add hockey bet to portfolio with specialized criteria."""
        
        # Check correlation with existing bets
        correlation = self.calculate_hockey_correlation(bet, self.active_bets)
        
        if correlation > self.max_correlation:
            return {'action': 'reject', 'reason': f'correlation {correlation:.2f} too high'}
        
        # Check goalie exposure
        goalie_exposure = self.calculate_goalie_exposure(bet, self.active_bets)
        if goalie_exposure > self.max_goalie_exposure:
            return {'action': 'reject', 'reason': f'goalie exposure {goalie_exposure:.2f} too high'}
        
        # Check if bet improves portfolio EV
        portfolio_ev = self.calculate_portfolio_ev()
        new_portfolio_ev = self.calculate_portfolio_ev_with_bet(bet)
        
        if new_portfolio_ev <= portfolio_ev:
            return {'action': 'reject', 'reason': 'does not improve portfolio EV'}
        
        self.active_bets.append(bet)
        return {'action': 'accept', 'new_portfolio_ev': new_portfolio_ev}
    
    def calculate_hockey_correlation(self, new_bet, existing_bets):
        """Calculate correlation between hockey bets."""
        
        if not existing_bets:
            return 0.0
        
        correlations = []
        for bet in existing_bets:
            # Team correlation
            if new_bet['team'] == bet['team']:
                correlations.append(1.0)
            elif new_bet['team'] in bet['opponents']:
                correlations.append(-0.6)  # Higher negative correlation in hockey
            else:
                correlations.append(0.0)
            
            # Goalie correlation
            if new_bet['goalie'] == bet['goalie']:
                correlations.append(0.8)
        
        return np.mean(correlations) if correlations else 0.0
    
    def calculate_goalie_exposure(self, new_bet, existing_bets):
        """Calculate exposure to specific goalies."""
        
        total_exposure = new_bet['stake']
        for bet in existing_bets:
            if bet['goalie'] == new_bet['goalie']:
                total_exposure += bet['stake']
        
        return total_exposure / self.bankroll
```

## Expected Aggressive NHL Performance

### 1. **Hockey-Specific Growth Targets**
```python
aggressive_nhl_performance_targets = {
    'win_rate': 0.56,           # 56% win rate (lower than MLB due to variance)
    'average_ev': 0.06,         # 6% average EV
    'annual_roi': 0.20,         # 20% annual ROI
    'sharpe_ratio': 0.9,        # 0.9 Sharpe ratio
    'max_drawdown': 0.18,       # 18% max drawdown
    'value_bet_rate': 0.15,     # 15% of games offer value
    'positive_clv_rate': 0.58,  # 58% of bets beat closing line
    'average_bet_size': 0.06,   # 6% average bet size
    'monthly_growth': 0.06      # 6% monthly growth target
}
```

### 2. **Hockey-Specific Risk Metrics**
```python
def calculate_hockey_growth_metrics(returns):
    """Calculate hockey-specific performance metrics."""
    
    # Basic metrics
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Growth metrics
    cumulative_return = np.prod(1 + returns) - 1
    geometric_mean = (cumulative_return + 1) ** (1/len(returns)) - 1
    
    # Risk-adjusted metrics
    sharpe = avg_return / std_return if std_return > 0 else 0
    sortino = avg_return / np.std([r for r in returns if r < 0]) if any(r < 0 for r in returns) else float('inf')
    
    # Maximum drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Hockey-specific metrics
    win_streak = max_consecutive_wins(returns)
    loss_streak = max_consecutive_losses(returns)
    
    return {
        'cumulative_return': cumulative_return,
        'geometric_mean': geometric_mean,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'win_streak': win_streak,
        'loss_streak': loss_streak,
        'win_rate': sum(1 for r in returns if r > 0) / len(returns)
    }
```

## Implementation Strategy

### Phase 1: Hockey Model Development (Months 1-2)
- Develop hockey-specific predictive models
- Backtest on historical NHL data
- Optimize possession metrics and goalie analysis

### Phase 2: Paper Trading (Months 3-4)
- Test aggressive strategy with paper money
- Refine goalie impact algorithms
- Optimize position sizing for hockey variance

### Phase 3: Small Bankroll (Months 5-6)
- Start with $10,000 bankroll
- Maximum $1,200 per bet (12%)
- Focus on high-confidence goalie matchups

### Phase 4: Scale Up (Months 7+)
- Increase bankroll based on performance
- Maintain aggressive position sizing
- Expand to multiple hockey markets

## Conclusion

This aggressive NHL strategy prioritizes:

1. **Hockey-Specific Accuracy**: Advanced possession metrics, goalie analysis, special teams
2. **Edge Maximization**: Market inefficiency exploitation with hockey-specific adjustments
3. **Growth Optimization**: Full Kelly Criterion with hockey variance adjustments
4. **Speed**: Real-time goalie confirmation and market analysis

The strategy recognizes hockey's unique characteristics and leverages advanced analytics to maximize prediction accuracy and bankroll growth in the NHL market. 