"""Algorithmic Trading Manager for systematic backtesting and execution."""

import structlog
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..analytics.interfaces import PredictionModel
from ..core.dependency_injection import DependencyContainer

logger = structlog.get_logger()


class AlgoTradingManager:
    """Manager for algorithmic trading operations."""
    
    def __init__(self, config: dict, db_manager: Any, analytics_module: Any):
        self.config = config
        self.db_manager = db_manager
        self.analytics_module = analytics_module
        self.container = DependencyContainer()
        self.container.configure_default_services(config)
        
        # Trading parameters
        self.risk_threshold = config.get("risk_threshold", 0.05)
        self.max_position_size = config.get("max_position_size", 0.1)
        self.stop_loss = config.get("stop_loss", 0.02)
        
        logger.info("AlgoTradingManager initialized")
    
    async def backtest_strategy(self, strategy_name: str, historical_data: pd.DataFrame, 
                               start_date: str, end_date: str) -> Dict[str, Any]:
        """Backtest a trading strategy."""
        try:
            logger.info(f"Backtesting strategy: {strategy_name}")
            
            # Filter data by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            filtered_data = historical_data[
                (historical_data['date'] >= start_dt) & 
                (historical_data['date'] <= end_dt)
            ]
            
            if filtered_data.empty:
                return {
                    "strategy_name": strategy_name,
                    "total_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "total_trades": 0,
                    "profitable_trades": 0
                }
            
            # Simulate trading
            initial_capital = 10000
            current_capital = initial_capital
            trades = []
            positions = []
            
            for _, row in filtered_data.iterrows():
                # Simple strategy: bet on home team if win probability > 0.6
                if row.get('home_win_probability', 0) > 0.6:
                    bet_amount = current_capital * self.max_position_size
                    if row.get('home_team_won', False):
                        current_capital += bet_amount * (row.get('odds', 2.0) - 1)
                        trades.append({
                            'date': row['date'],
                            'type': 'win',
                            'amount': bet_amount * (row.get('odds', 2.0) - 1)
                        })
                    else:
                        current_capital -= bet_amount
                        trades.append({
                            'date': row['date'],
                            'type': 'loss',
                            'amount': -bet_amount
                        })
            
            # Calculate metrics
            total_return = (current_capital - initial_capital) / initial_capital
            profitable_trades = len([t for t in trades if t['amount'] > 0])
            win_rate = profitable_trades / len(trades) if trades else 0.0
            
            # Calculate Sharpe ratio (simplified)
            returns = [t['amount'] / initial_capital for t in trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0.0
            
            # Calculate max drawdown
            capital_history = [initial_capital]
            for trade in trades:
                capital_history.append(capital_history[-1] + trade['amount'])
            
            peak = initial_capital
            max_drawdown = 0.0
            for capital in capital_history:
                if capital > peak:
                    peak = capital
                drawdown = (peak - capital) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            result = {
                "strategy_name": strategy_name,
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "total_trades": len(trades),
                "profitable_trades": profitable_trades,
                "final_capital": current_capital,
                "initial_capital": initial_capital
            }
            
            logger.info(f"Backtest completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return {}
    
    async def execute_trade(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade."""
        try:
            logger.info(f"Executing trade: {trade_data}")
            
            # Validate trade
            validation = await self._validate_trade(trade_data)
            if not validation['valid']:
                return {
                    "success": False,
                    "error": validation['error'],
                    "trade_id": None
                }
            
            # Calculate position size
            position_size = await self._calculate_position_size(trade_data)
            
            # Execute trade (mock implementation)
            trade_id = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save to database
            await self.db_manager.save_bet({
                "id": trade_id,
                "sport": trade_data.get("sport"),
                "event_id": trade_data.get("event_id"),
                "selection": trade_data.get("selection"),
                "odds": trade_data.get("odds"),
                "stake": position_size,
                "timestamp": datetime.now(),
                "status": "pending"
            })
            
            result = {
                "success": True,
                "trade_id": trade_id,
                "position_size": position_size,
                "execution_time": datetime.now()
            }
            
            logger.info(f"Trade executed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {
                "success": False,
                "error": str(e),
                "trade_id": None
            }
    
    async def _validate_trade(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trade parameters."""
        try:
            required_fields = ["sport", "event_id", "selection", "odds"]
            for field in required_fields:
                if field not in trade_data:
                    return {"valid": False, "error": f"Missing required field: {field}"}
            
            # Check odds validity
            odds = trade_data.get("odds", 0)
            if odds <= 1.0:
                return {"valid": False, "error": "Invalid odds (must be > 1.0)"}
            
            # Check risk threshold
            if trade_data.get("risk_score", 0) > self.risk_threshold:
                return {"valid": False, "error": "Risk exceeds threshold"}
            
            return {"valid": True, "error": None}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    async def _calculate_position_size(self, trade_data: Dict[str, Any]) -> float:
        """Calculate position size based on Kelly Criterion."""
        try:
            # Kelly Criterion: f = (bp - q) / b
            # where b = odds - 1, p = win probability, q = 1 - p
            
            odds = trade_data.get("odds", 2.0)
            win_probability = trade_data.get("win_probability", 0.5)
            
            b = odds - 1
            p = win_probability
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b if b > 0 else 0
            
            # Apply constraints
            kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
            
            # Convert to dollar amount
            available_capital = trade_data.get("available_capital", 10000)
            position_size = available_capital * kelly_fraction
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    async def get_performance_metrics(self, timeframe: str = "month") -> Dict[str, Any]:
        """Get trading performance metrics."""
        try:
            logger.info(f"Getting performance metrics for {timeframe}")
            
            # Implementation would fetch from database
            # For now, return mock metrics
            return {
                "timeframe": timeframe,
                "total_return": 0.125,
                "sharpe_ratio": 1.85,
                "max_drawdown": 0.08,
                "win_rate": 0.62,
                "total_trades": 45,
                "profitable_trades": 28,
                "average_trade_return": 0.025,
                "volatility": 0.15
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    async def manage_risk(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Manage portfolio risk."""
        try:
            logger.info("Managing portfolio risk")
            
            # Calculate current risk metrics
            total_exposure = portfolio.get("total_exposure", 0)
            max_exposure = portfolio.get("max_exposure", 10000)
            current_risk = total_exposure / max_exposure if max_exposure > 0 else 0
            
            # Risk management actions
            actions = []
            if current_risk > 0.8:
                actions.append("Reduce position sizes")
            if current_risk > 0.9:
                actions.append("Stop new trades")
            if current_risk > 0.95:
                actions.append("Close some positions")
            
            return {
                "current_risk": current_risk,
                "total_exposure": total_exposure,
                "max_exposure": max_exposure,
                "risk_level": "high" if current_risk > 0.8 else "medium" if current_risk > 0.5 else "low",
                "recommended_actions": actions
            }
            
        except Exception as e:
            logger.error(f"Error managing risk: {e}")
            return {} 