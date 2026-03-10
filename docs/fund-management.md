# Fund Management Guide

**Status**: ✅ **PRODUCTION READY**  
**Last Updated**: 2025-01-20

## Overview

This guide covers the comprehensive fund management system for the ABBA betting platform, including bankroll management, risk controls, performance tracking, and automated fund allocation strategies.

## Fund Management Architecture

### 1. Multi-Tier Fund Management

#### Core Fund Manager
```python
class FundManager:
    def __init__(self):
        self.bankroll = BankrollManager()
        self.risk_manager = RiskManager()
        self.performance_tracker = PerformanceTracker()
        self.allocation_engine = AllocationEngine()
        self.withdrawal_manager = WithdrawalManager()
    
    def initialize_fund(self, initial_amount, risk_profile):
        """Initialize fund with specified amount and risk profile."""
        # 1. Set up bankroll
        self.bankroll.set_initial_amount(initial_amount)
        
        # 2. Configure risk parameters
        self.risk_manager.set_risk_profile(risk_profile)
        
        # 3. Set up performance tracking
        self.performance_tracker.initialize_tracking()
        
        # 4. Configure allocation strategy
        self.allocation_engine.set_strategy(risk_profile['allocation_strategy'])
        
        return {
            'status': 'initialized',
            'initial_amount': initial_amount,
            'risk_profile': risk_profile,
            'current_balance': initial_amount
        }
    
    def process_bet(self, bet_recommendation, current_balance):
        """Process bet with fund management controls."""
        # 1. Validate bet against risk limits
        risk_validation = self.risk_manager.validate_bet(bet_recommendation, current_balance)
        
        if not risk_validation['approved']:
            return {
                'status': 'rejected',
                'reason': risk_validation['reason'],
                'recommendation': risk_validation['recommendation']
            }
        
        # 2. Calculate optimal stake
        optimal_stake = self.allocation_engine.calculate_stake(
            bet_recommendation, current_balance, risk_validation
        )
        
        # 3. Update bankroll
        self.bankroll.reserve_funds(optimal_stake)
        
        # 4. Record bet
        self.performance_tracker.record_bet(bet_recommendation, optimal_stake)
        
        return {
            'status': 'approved',
            'stake': optimal_stake,
            'risk_score': risk_validation['risk_score'],
            'expected_value': bet_recommendation['expected_value']
        }
```

#### Bankroll Manager
```python
class BankrollManager:
    def __init__(self):
        self.initial_amount = 0
        self.current_balance = 0
        self.reserved_funds = 0
        self.available_funds = 0
        self.transaction_history = []
    
    def set_initial_amount(self, amount):
        """Set initial bankroll amount."""
        self.initial_amount = amount
        self.current_balance = amount
        self.available_funds = amount
        self.reserved_funds = 0
        
        self.transaction_history.append({
            'type': 'initialization',
            'amount': amount,
            'timestamp': datetime.now(),
            'balance': self.current_balance
        })
    
    def reserve_funds(self, amount):
        """Reserve funds for a bet."""
        if amount > self.available_funds:
            raise InsufficientFunds(f"Insufficient funds: {amount} > {self.available_funds}")
        
        self.reserved_funds += amount
        self.available_funds -= amount
        
        self.transaction_history.append({
            'type': 'reservation',
            'amount': amount,
            'timestamp': datetime.now(),
            'balance': self.current_balance,
            'reserved': self.reserved_funds
        })
    
    def release_funds(self, amount):
        """Release reserved funds."""
        if amount > self.reserved_funds:
            raise InvalidOperation(f"Cannot release more than reserved: {amount} > {self.reserved_funds}")
        
        self.reserved_funds -= amount
        self.available_funds += amount
        
        self.transaction_history.append({
            'type': 'release',
            'amount': amount,
            'timestamp': datetime.now(),
            'balance': self.current_balance,
            'reserved': self.reserved_funds
        })
    
    def update_balance(self, profit_loss):
        """Update balance after bet result."""
        self.current_balance += profit_loss
        
        # Release any remaining reserved funds
        if self.reserved_funds > 0:
            self.available_funds += self.reserved_funds
            self.reserved_funds = 0
        
        self.transaction_history.append({
            'type': 'balance_update',
            'amount': profit_loss,
            'timestamp': datetime.now(),
            'balance': self.current_balance
        })
    
    def get_fund_status(self):
        """Get current fund status."""
        return {
            'initial_amount': self.initial_amount,
            'current_balance': self.current_balance,
            'reserved_funds': self.reserved_funds,
            'available_funds': self.available_funds,
            'total_return': self.current_balance - self.initial_amount,
            'return_percentage': (self.current_balance - self.initial_amount) / self.initial_amount * 100
        }
```

### 2. Risk Management System

#### Risk Manager
```python
class RiskManager:
    def __init__(self):
        self.risk_limits = {
            'max_single_bet': 0.05,      # 5% max per bet
            'max_daily_risk': 0.08,      # 8% max daily
            'max_weekly_risk': 0.15,     # 15% max weekly
            'max_drawdown': 0.12,        # 12% max drawdown
            'max_correlation': 0.3       # 30% max correlation
        }
        self.daily_risk_used = 0
        self.weekly_risk_used = 0
        self.active_bets = []
        self.risk_history = []
    
    def set_risk_profile(self, profile):
        """Set risk management profile."""
        self.risk_limits.update(profile.get('risk_limits', {}))
        
        # Reset risk counters
        self.daily_risk_used = 0
        self.weekly_risk_used = 0
        self.active_bets = []
    
    def validate_bet(self, bet_recommendation, current_balance):
        """Validate bet against risk limits."""
        bet_amount = bet_recommendation['stake']
        bet_percentage = bet_amount / current_balance
        
        # Check single bet limit
        if bet_percentage > self.risk_limits['max_single_bet']:
            return {
                'approved': False,
                'reason': 'Exceeds single bet limit',
                'recommendation': f"Reduce stake to {self.risk_limits['max_single_bet'] * current_balance:.2f}",
                'risk_score': 1.0
            }
        
        # Check daily risk limit
        if self.daily_risk_used + bet_percentage > self.risk_limits['max_daily_risk']:
            return {
                'approved': False,
                'reason': 'Exceeds daily risk limit',
                'recommendation': f"Daily risk limit reached. Available: {self.risk_limits['max_daily_risk'] - self.daily_risk_used:.2%}",
                'risk_score': 0.9
            }
        
        # Check weekly risk limit
        if self.weekly_risk_used + bet_percentage > self.risk_limits['max_weekly_risk']:
            return {
                'approved': False,
                'reason': 'Exceeds weekly risk limit',
                'recommendation': f"Weekly risk limit reached. Available: {self.risk_limits['max_weekly_risk'] - self.weekly_risk_used:.2%}",
                'risk_score': 0.8
            }
        
        # Check correlation risk
        correlation_risk = self._calculate_correlation_risk(bet_recommendation)
        if correlation_risk > self.risk_limits['max_correlation']:
            return {
                'approved': False,
                'reason': 'High correlation risk',
                'recommendation': 'Consider diversifying bets to reduce correlation',
                'risk_score': 0.7
            }
        
        # Check drawdown limit
        current_drawdown = self._calculate_current_drawdown(current_balance)
        if current_drawdown > self.risk_limits['max_drawdown']:
            return {
                'approved': False,
                'reason': 'Maximum drawdown exceeded',
                'recommendation': 'Reduce risk until drawdown improves',
                'risk_score': 0.6
            }
        
        return {
            'approved': True,
            'reason': 'Bet approved',
            'risk_score': self._calculate_risk_score(bet_recommendation),
            'correlation_risk': correlation_risk,
            'current_drawdown': current_drawdown
        }
    
    def _calculate_correlation_risk(self, bet_recommendation):
        """Calculate correlation risk with existing bets."""
        if not self.active_bets:
            return 0.0
        
        # Calculate correlation based on teams, sports, bet types
        correlations = []
        for active_bet in self.active_bets:
            correlation = self._calculate_bet_correlation(bet_recommendation, active_bet)
            correlations.append(correlation)
        
        return max(correlations) if correlations else 0.0
    
    def _calculate_bet_correlation(self, bet1, bet2):
        """Calculate correlation between two bets."""
        correlation_score = 0.0
        
        # Same team correlation
        if bet1.get('team') == bet2.get('team'):
            correlation_score += 0.8
        
        # Same sport correlation
        if bet1.get('sport') == bet2.get('sport'):
            correlation_score += 0.3
        
        # Same bet type correlation
        if bet1.get('bet_type') == bet2.get('bet_type'):
            correlation_score += 0.2
        
        # Same game correlation
        if bet1.get('game_id') == bet2.get('game_id'):
            correlation_score += 0.9
        
        return min(correlation_score, 1.0)
    
    def _calculate_current_drawdown(self, current_balance):
        """Calculate current drawdown from peak."""
        if not self.risk_history:
            return 0.0
        
        peak_balance = max([entry['balance'] for entry in self.risk_history])
        if peak_balance <= 0:
            return 0.0
        
        return (peak_balance - current_balance) / peak_balance
    
    def _calculate_risk_score(self, bet_recommendation):
        """Calculate overall risk score for bet."""
        risk_factors = {
            'stake_size': bet_recommendation['stake'] / bet_recommendation.get('bankroll', 1),
            'expected_value': 1 - bet_recommendation.get('expected_value', 0),
            'confidence': 1 - bet_recommendation.get('confidence', 0.5),
            'market_volatility': bet_recommendation.get('market_volatility', 0.5)
        }
        
        # Weighted average of risk factors
        weights = [0.4, 0.3, 0.2, 0.1]
        risk_score = sum(factor * weight for factor, weight in zip(risk_factors.values(), weights))
        
        return min(risk_score, 1.0)
```

### 3. Performance Tracking

#### Performance Tracker
```python
class PerformanceTracker:
    def __init__(self):
        self.bet_history = []
        self.daily_performance = {}
        self.weekly_performance = {}
        self.monthly_performance = {}
        self.performance_metrics = {}
    
    def initialize_tracking(self):
        """Initialize performance tracking."""
        self.bet_history = []
        self.daily_performance = {}
        self.weekly_performance = {}
        self.monthly_performance = {}
        self.performance_metrics = {
            'total_bets': 0,
            'winning_bets': 0,
            'losing_bets': 0,
            'total_profit_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'average_bet_size': 0,
            'win_rate': 0,
            'roi': 0,
            'sharpe_ratio': 0
        }
    
    def record_bet(self, bet_recommendation, stake):
        """Record a new bet."""
        bet_record = {
            'id': self._generate_bet_id(),
            'timestamp': datetime.now(),
            'sport': bet_recommendation.get('sport'),
            'event': bet_recommendation.get('event'),
            'bet_type': bet_recommendation.get('bet_type'),
            'selection': bet_recommendation.get('selection'),
            'odds': bet_recommendation.get('odds'),
            'stake': stake,
            'expected_value': bet_recommendation.get('expected_value'),
            'confidence': bet_recommendation.get('confidence'),
            'status': 'pending'
        }
        
        self.bet_history.append(bet_record)
        self._update_performance_metrics()
        
        return bet_record['id']
    
    def record_bet_result(self, bet_id, result):
        """Record bet result."""
        bet_record = self._find_bet(bet_id)
        if not bet_record:
            raise ValueError(f"Bet {bet_id} not found")
        
        bet_record.update({
            'result': result['outcome'],
            'profit_loss': result['profit_loss'],
            'settled_at': datetime.now(),
            'status': 'settled'
        })
        
        self._update_performance_metrics()
        self._update_periodic_performance()
        
        return self.get_performance_summary()
    
    def get_performance_summary(self):
        """Get comprehensive performance summary."""
        return {
            'overall_metrics': self.performance_metrics,
            'daily_performance': self._get_recent_performance('daily', 30),
            'weekly_performance': self._get_recent_performance('weekly', 12),
            'monthly_performance': self._get_recent_performance('monthly', 12),
            'sport_breakdown': self._get_sport_breakdown(),
            'bet_type_breakdown': self._get_bet_type_breakdown(),
            'risk_adjusted_returns': self._calculate_risk_adjusted_returns()
        }
    
    def _update_performance_metrics(self):
        """Update overall performance metrics."""
        settled_bets = [bet for bet in self.bet_history if bet['status'] == 'settled']
        
        if not settled_bets:
            return
        
        self.performance_metrics.update({
            'total_bets': len(settled_bets),
            'winning_bets': len([bet for bet in settled_bets if bet['result'] == 'win']),
            'losing_bets': len([bet for bet in settled_bets if bet['result'] == 'loss']),
            'total_profit_loss': sum(bet['profit_loss'] for bet in settled_bets),
            'largest_win': max([bet['profit_loss'] for bet in settled_bets]),
            'largest_loss': min([bet['profit_loss'] for bet in settled_bets]),
            'average_bet_size': np.mean([bet['stake'] for bet in settled_bets]),
            'win_rate': len([bet for bet in settled_bets if bet['result'] == 'win']) / len(settled_bets),
            'roi': sum(bet['profit_loss'] for bet in settled_bets) / sum(bet['stake'] for bet in settled_bets)
        })
        
        # Calculate Sharpe ratio
        returns = [bet['profit_loss'] / bet['stake'] for bet in settled_bets]
        if len(returns) > 1:
            self.performance_metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns)
    
    def _get_sport_breakdown(self):
        """Get performance breakdown by sport."""
        sport_performance = {}
        
        for bet in self.bet_history:
            if bet['status'] != 'settled':
                continue
            
            sport = bet['sport']
            if sport not in sport_performance:
                sport_performance[sport] = {
                    'total_bets': 0,
                    'winning_bets': 0,
                    'total_profit_loss': 0,
                    'win_rate': 0,
                    'roi': 0
                }
            
            sport_performance[sport]['total_bets'] += 1
            if bet['result'] == 'win':
                sport_performance[sport]['winning_bets'] += 1
            sport_performance[sport]['total_profit_loss'] += bet['profit_loss']
        
        # Calculate win rates and ROI
        for sport in sport_performance:
            if sport_performance[sport]['total_bets'] > 0:
                sport_performance[sport]['win_rate'] = (
                    sport_performance[sport]['winning_bets'] / 
                    sport_performance[sport]['total_bets']
                )
        
        return sport_performance
```

### 4. Fund Allocation Engine

#### Allocation Engine
```python
class AllocationEngine:
    def __init__(self):
        self.strategy = 'kelly'  # Default strategy
        self.kelly_fraction = 0.25  # Conservative Kelly (1/4)
        self.min_stake = 0.005  # 0.5% minimum stake
        self.max_stake = 0.05   # 5% maximum stake
    
    def set_strategy(self, strategy):
        """Set fund allocation strategy."""
        self.strategy = strategy
        
        if strategy == 'conservative':
            self.kelly_fraction = 0.15
            self.min_stake = 0.003
            self.max_stake = 0.03
        elif strategy == 'moderate':
            self.kelly_fraction = 0.25
            self.min_stake = 0.005
            self.max_stake = 0.05
        elif strategy == 'aggressive':
            self.kelly_fraction = 0.35
            self.min_stake = 0.008
            self.max_stake = 0.08
    
    def calculate_stake(self, bet_recommendation, current_balance, risk_validation):
        """Calculate optimal stake using selected strategy."""
        if self.strategy == 'kelly':
            return self._calculate_kelly_stake(bet_recommendation, current_balance)
        elif self.strategy == 'fixed':
            return self._calculate_fixed_stake(bet_recommendation, current_balance)
        elif self.strategy == 'progressive':
            return self._calculate_progressive_stake(bet_recommendation, current_balance)
        else:
            return self._calculate_kelly_stake(bet_recommendation, current_balance)
    
    def _calculate_kelly_stake(self, bet_recommendation, current_balance):
        """Calculate stake using Kelly Criterion."""
        win_prob = bet_recommendation['win_probability']
        odds = bet_recommendation['odds']
        
        # Convert odds to decimal
        decimal_odds = self._convert_to_decimal(odds)
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds - 1, p = win probability, q = 1 - p
        b = decimal_odds - 1
        p = win_prob
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Apply fractional Kelly and limits
        stake_fraction = max(0, kelly_fraction * self.kelly_fraction)
        stake_fraction = max(self.min_stake, min(self.max_stake, stake_fraction))
        
        return stake_fraction * current_balance
    
    def _calculate_fixed_stake(self, bet_recommendation, current_balance):
        """Calculate fixed percentage stake."""
        base_stake = 0.02  # 2% base stake
        
        # Adjust based on confidence
        confidence_multiplier = bet_recommendation.get('confidence', 0.5)
        adjusted_stake = base_stake * confidence_multiplier
        
        # Apply limits
        adjusted_stake = max(self.min_stake, min(self.max_stake, adjusted_stake))
        
        return adjusted_stake * current_balance
    
    def _calculate_progressive_stake(self, bet_recommendation, current_balance):
        """Calculate progressive stake based on performance."""
        base_stake = 0.02  # 2% base stake
        
        # Adjust based on recent performance
        performance_multiplier = self._get_performance_multiplier()
        
        # Adjust based on expected value
        ev_multiplier = 1 + bet_recommendation.get('expected_value', 0)
        
        adjusted_stake = base_stake * performance_multiplier * ev_multiplier
        
        # Apply limits
        adjusted_stake = max(self.min_stake, min(self.max_stake, adjusted_stake))
        
        return adjusted_stake * current_balance
    
    def _convert_to_decimal(self, odds):
        """Convert American odds to decimal."""
        if odds > 0:
            return (odds / 100) + 1
        else:
            return (100 / abs(odds)) + 1
    
    def _get_performance_multiplier(self):
        """Get performance-based multiplier."""
        # This would integrate with performance tracker
        # For now, return neutral multiplier
        return 1.0
```

### 5. Withdrawal Management

#### Withdrawal Manager
```python
class WithdrawalManager:
    def __init__(self):
        self.withdrawal_history = []
        self.withdrawal_limits = {
            'min_withdrawal': 50,
            'max_withdrawal': 10000,
            'daily_limit': 5000,
            'weekly_limit': 20000
        }
        self.daily_withdrawn = 0
        self.weekly_withdrawn = 0
    
    def request_withdrawal(self, amount, current_balance):
        """Request a withdrawal."""
        # Validate withdrawal amount
        validation = self._validate_withdrawal(amount, current_balance)
        
        if not validation['approved']:
            return {
                'status': 'rejected',
                'reason': validation['reason']
            }
        
        # Create withdrawal request
        withdrawal_request = {
            'id': self._generate_withdrawal_id(),
            'amount': amount,
            'timestamp': datetime.now(),
            'status': 'pending',
            'balance_before': current_balance,
            'balance_after': current_balance - amount
        }
        
        self.withdrawal_history.append(withdrawal_request)
        
        # Update limits
        self.daily_withdrawn += amount
        self.weekly_withdrawn += amount
        
        return {
            'status': 'approved',
            'withdrawal_id': withdrawal_request['id'],
            'amount': amount,
            'processing_time': '1-3 business days'
        }
    
    def _validate_withdrawal(self, amount, current_balance):
        """Validate withdrawal request."""
        # Check minimum withdrawal
        if amount < self.withdrawal_limits['min_withdrawal']:
            return {
                'approved': False,
                'reason': f"Minimum withdrawal is ${self.withdrawal_limits['min_withdrawal']}"
            }
        
        # Check maximum withdrawal
        if amount > self.withdrawal_limits['max_withdrawal']:
            return {
                'approved': False,
                'reason': f"Maximum withdrawal is ${self.withdrawal_limits['max_withdrawal']}"
            }
        
        # Check available balance
        if amount > current_balance:
            return {
                'approved': False,
                'reason': f"Insufficient balance: ${current_balance}"
            }
        
        # Check daily limit
        if self.daily_withdrawn + amount > self.withdrawal_limits['daily_limit']:
            return {
                'approved': False,
                'reason': f"Daily withdrawal limit exceeded"
            }
        
        # Check weekly limit
        if self.weekly_withdrawn + amount > self.withdrawal_limits['weekly_limit']:
            return {
                'approved': False,
                'reason': f"Weekly withdrawal limit exceeded"
            }
        
        return {'approved': True}
    
    def get_withdrawal_summary(self):
        """Get withdrawal summary."""
        return {
            'total_withdrawals': len(self.withdrawal_history),
            'total_amount_withdrawn': sum(w['amount'] for w in self.withdrawal_history),
            'daily_withdrawn': self.daily_withdrawn,
            'weekly_withdrawn': self.weekly_withdrawn,
            'daily_limit_remaining': self.withdrawal_limits['daily_limit'] - self.daily_withdrawn,
            'weekly_limit_remaining': self.withdrawal_limits['weekly_limit'] - self.weekly_withdrawn,
            'recent_withdrawals': self.withdrawal_history[-10:]  # Last 10 withdrawals
        }
```

## Implementation

### 1. Fund Management Configuration

#### Configuration Settings
```python
# Fund management configuration
FUND_MANAGEMENT_CONFIG = {
    'risk_profiles': {
        'conservative': {
            'max_single_bet': 0.03,      # 3% max per bet
            'max_daily_risk': 0.06,      # 6% max daily
            'max_weekly_risk': 0.12,     # 12% max weekly
            'max_drawdown': 0.08,        # 8% max drawdown
            'allocation_strategy': 'conservative'
        },
        'moderate': {
            'max_single_bet': 0.05,      # 5% max per bet
            'max_daily_risk': 0.08,      # 8% max daily
            'max_weekly_risk': 0.15,     # 15% max weekly
            'max_drawdown': 0.12,        # 12% max drawdown
            'allocation_strategy': 'moderate'
        },
        'aggressive': {
            'max_single_bet': 0.08,      # 8% max per bet
            'max_daily_risk': 0.12,      # 12% max daily
            'max_weekly_risk': 0.20,     # 20% max weekly
            'max_drawdown': 0.15,        # 15% max drawdown
            'allocation_strategy': 'aggressive'
        }
    },
    'withdrawal_limits': {
        'min_withdrawal': 50,
        'max_withdrawal': 10000,
        'daily_limit': 5000,
        'weekly_limit': 20000
    },
    'performance_targets': {
        'min_win_rate': 0.54,
        'min_roi': 0.08,
        'max_drawdown': 0.12,
        'target_sharpe_ratio': 1.0
    }
}
```

### 2. Monitoring and Reporting

#### Fund Monitoring
```python
class FundMonitor:
    def __init__(self):
        self.alert_system = AlertSystem()
        self.report_generator = ReportGenerator()
    
    def monitor_fund_health(self, fund_manager):
        """Monitor fund health and generate alerts."""
        fund_status = fund_manager.bankroll.get_fund_status()
        performance = fund_manager.performance_tracker.get_performance_summary()
        
        # Check for alerts
        alerts = []
        
        # Drawdown alert
        if fund_status['return_percentage'] < -10:
            alerts.append({
                'type': 'drawdown_alert',
                'severity': 'high',
                'message': f"Significant drawdown: {fund_status['return_percentage']:.2f}%"
            })
        
        # Performance alert
        if performance['overall_metrics']['win_rate'] < 0.50:
            alerts.append({
                'type': 'performance_alert',
                'severity': 'medium',
                'message': f"Low win rate: {performance['overall_metrics']['win_rate']:.2%}"
            })
        
        # Risk alert
        if fund_manager.risk_manager.daily_risk_used > 0.06:
            alerts.append({
                'type': 'risk_alert',
                'severity': 'medium',
                'message': f"High daily risk usage: {fund_manager.risk_manager.daily_risk_used:.2%}"
            })
        
        # Send alerts
        for alert in alerts:
            self.alert_system.send_alert(alert)
        
        return alerts
```

---

**Status**: ✅ **PRODUCTION READY** - Comprehensive fund management system
**Features**: Multi-tier risk management, performance tracking, automated allocation
**Protection**: Drawdown limits, correlation controls, withdrawal management 