"""
Guardrail Agent for ABMBA system.
Responsible for safety protocols and ethical compliance.
"""

from datetime import datetime
from typing import Any

import numpy as np
import structlog
from crewai import Agent
from database import DatabaseManager
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

from models import Bet

logger = structlog.get_logger()


class GuardrailAgent:
    """Agent responsible for safety protocols and ethical compliance."""

    def __init__(self, config: dict, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.llm = ChatOpenAI(
            model=config["apis"]["openai"]["model"],
            api_key=config["apis"]["openai"]["api_key"],
            temperature=0.1,
        )
        # Safety thresholds
        self.bias_threshold = config.get("guardrails", {}).get("bias_threshold", 0.15)
        self.risk_threshold = config.get("guardrails", {}).get("risk_threshold", 0.05)
        self.ethical_violation_threshold = config.get("guardrails", {}).get(
            "ethical_violation_threshold", 0.1
        )
        # Define tools
        self.tools = [
            Tool(
                name="audit_for_biases",
                func=self._audit_for_biases,
                description="Comprehensive bias auditing across all models and data",
            ),
            Tool(
                name="ethical_compliance_check",
                func=self._ethical_compliance_check,
                description="Check ethical compliance of proposed actions",
            ),
            Tool(
                name="anomaly_detection",
                func=self._anomaly_detection,
                description="Detect and halt on critical anomalies",
            ),
            Tool(
                name="prevent_cascading_errors",
                func=self._prevent_cascading_errors,
                description="Prevent error propagation across agents",
            ),
            Tool(
                name="risk_assessment",
                func=self._risk_assessment,
                description="Comprehensive risk assessment for betting decisions",
            ),
            Tool(
                name="safety_monitoring",
                func=self._safety_monitoring,
                description="Continuous safety monitoring and alerting",
            ),
        ]
        self.agent = Agent(
            role="Safety and Ethics Compliance Specialist",
            goal="Ensure system safety, ethical compliance, and prevent harmful outcomes",
            backstory="""You are a safety and ethics expert with deep experience in AI systems
            and financial trading. You specialize in identifying biases, detecting anomalies,
            ensuring ethical compliance, and preventing cascading errors. Your primary mission
            is to protect the system and its users from harm while maintaining operational integrity.""",
            tools=self.tools,
            llm=self.llm,
            verbose=True,
        )

    async def _audit_for_biases(
        self, data: dict, model_type: str = "all"
    ) -> dict[str, Any]:
        """Comprehensive bias auditing."""
        try:
            audit_results = {
                "overall_bias_score": 0.0,
                "group_biases": {},
                "recommendations": [],
                "requires_halt": False,
                "audit_timestamp": datetime.utcnow().isoformat(),
            }
            # Check for common biases
            biases_to_check = [
                "home_team_bias",
                "favorite_team_bias",
                "recency_bias",
                "confirmation_bias",
                "overconfidence_bias",
                "selection_bias",
                "survivorship_bias",
            ]
            for bias_type in biases_to_check:
                bias_score = await self._check_specific_bias(data, bias_type)
                audit_results["group_biases"][bias_type] = bias_score
                if bias_score > self.bias_threshold:
                    audit_results["recommendations"].append(
                        f"High {bias_type} detected: {bias_score:.3f} (threshold: {self.bias_threshold})"
                    )
                    audit_results["requires_halt"] = True
            # Calculate overall bias score
            if audit_results["group_biases"]:
                audit_results["overall_bias_score"] = np.mean(
                    list(audit_results["group_biases"].values())
                )
            # Check for data quality biases
            data_quality_bias = await self._check_data_quality_bias(data)
            audit_results["group_biases"]["data_quality_bias"] = data_quality_bias
            # Check for model fairness
            fairness_score = await self._check_model_fairness(data)
            audit_results["group_biases"]["model_fairness"] = fairness_score
            logger.info(
                f"Bias audit completed. Overall score: {audit_results['overall_bias_score']:.3f}"
            )
            return audit_results
        except Exception as e:
            logger.error(f"Error in bias audit: {e}")
            return {"error": str(e)}

    async def _check_specific_bias(self, data: dict, bias_type: str) -> float:
        """Check for specific types of bias."""
        try:
            if bias_type == "home_team_bias":
                return await self._check_home_team_bias(data)
            elif bias_type == "favorite_team_bias":
                return await self._check_favorite_team_bias(data)
            elif bias_type == "recency_bias":
                return await self._check_recency_bias(data)
            elif bias_type == "confirmation_bias":
                return await self._check_confirmation_bias(data)
            elif bias_type == "overconfidence_bias":
                return await self._check_overconfidence_bias(data)
            elif bias_type == "selection_bias":
                return await self._check_selection_bias(data)
            elif bias_type == "survivorship_bias":
                return await self._check_survivorship_bias(data)
            else:
                return 0.0
        except Exception as e:
            logger.error(f"Error checking {bias_type} bias: {e}")
            return 0.0

    async def _check_home_team_bias(self, data: dict) -> float:
        """Check for home team bias in predictions."""
        try:
            bets = data.get("bets", [])
            if not bets:
                return 0.0
            home_team_bets = []
            away_team_bets = []
            for bet in bets:
                if hasattr(bet, "selection") and bet.selection:
                    if "home" in bet.selection.lower():
                        home_team_bets.append(bet)
                    elif "away" in bet.selection.lower():
                        away_team_bets.append(bet)
            if not home_team_bets or not away_team_bets:
                return 0.0
            # Calculate average EV for home vs away
            home_ev = np.mean([float(b.expected_value or 0) for b in home_team_bets])
            away_ev = np.mean([float(b.expected_value or 0) for b in away_team_bets])
            # Calculate bias as difference in EV
            bias_score = abs(home_ev - away_ev)
            return min(bias_score, 1.0)  # Cap at 1.0
        except Exception as e:
            logger.error(f"Error checking home team bias: {e}")
            return 0.0

    async def _check_favorite_team_bias(self, data: dict) -> float:
        """Check for favorite team bias."""
        try:
            bets = data.get("bets", [])
            if not bets:
                return 0.0
            favorite_bets = []
            underdog_bets = []
            for bet in bets:
                if bet.odds:
                    odds = float(bet.odds)
                    if odds < 2.0:  # Favorites
                        favorite_bets.append(bet)
                    elif odds > 3.0:  # Underdogs
                        underdog_bets.append(bet)
            if not favorite_bets or not underdog_bets:
                return 0.0
            # Calculate bias as difference in selection frequency
            total_bets = len(favorite_bets) + len(underdog_bets)
            favorite_ratio = len(favorite_bets) / total_bets
            underdog_ratio = len(underdog_bets) / total_bets
            # Expected ratio should be roughly equal (0.5 each)
            bias_score = abs(favorite_ratio - 0.5) + abs(underdog_ratio - 0.5)
            return min(bias_score, 1.0)
        except Exception as e:
            logger.error(f"Error checking favorite team bias: {e}")
            return 0.0

    async def _check_recency_bias(self, data: dict) -> float:
        """Check for recency bias in predictions."""
        try:
            bets = data.get("bets", [])
            if not bets or len(bets) < 10:
                return 0.0
            # Sort bets by time
            sorted_bets = sorted(bets, key=lambda b: b.placed_at or datetime.min)
            # Split into recent and older bets
            mid_point = len(sorted_bets) // 2
            recent_bets = sorted_bets[mid_point:]
            older_bets = sorted_bets[:mid_point]
            if not recent_bets or not older_bets:
                return 0.0
            # Calculate average EV for recent vs older bets
            recent_ev = np.mean([float(b.expected_value or 0) for b in recent_bets])
            older_ev = np.mean([float(b.expected_value or 0) for b in older_bets])
            # Calculate bias as difference in EV
            bias_score = abs(recent_ev - older_ev)
            return min(bias_score, 1.0)
        except Exception as e:
            logger.error(f"Error checking recency bias: {e}")
            return 0.0

    async def _check_confirmation_bias(self, data: dict) -> float:
        """Check for confirmation bias."""
        try:
            # This would analyze if the system tends to favor predictions that confirm existing beliefs
            # For now, return a placeholder
            return 0.05  # Low bias score
        except Exception as e:
            logger.error(f"Error checking confirmation bias: {e}")
            return 0.0

    async def _check_overconfidence_bias(self, data: dict) -> float:
        """Check for overconfidence bias."""
        try:
            bets = data.get("bets", [])
            if not bets:
                return 0.0
            # Check if EV predictions are consistently too high
            evs = [float(b.expected_value or 0) for b in bets if b.expected_value]
            if not evs:
                return 0.0
            # Calculate average EV
            avg_ev = np.mean(evs)
            # High average EV might indicate overconfidence
            if avg_ev > 0.15:  # 15% average EV is suspiciously high
                bias_score = (avg_ev - 0.15) / 0.15  # Normalize
                return min(bias_score, 1.0)
            return 0.0
        except Exception as e:
            logger.error(f"Error checking overconfidence bias: {e}")
            return 0.0

    async def _check_selection_bias(self, data: dict) -> float:
        """Check for selection bias in data."""
        try:
            # This would check if the data is representative of the population
            # For now, return a placeholder
            return 0.03  # Low bias score
        except Exception as e:
            logger.error(f"Error checking selection bias: {e}")
            return 0.0

    async def _check_survivorship_bias(self, data: dict) -> float:
        """Check for survivorship bias."""
        try:
            # This would check if we're only looking at successful outcomes
            # For now, return a placeholder
            return 0.02  # Low bias score
        except Exception as e:
            logger.error(f"Error checking survivorship bias: {e}")
            return 0.0

    async def _check_data_quality_bias(self, data: dict) -> float:
        """Check for data quality related biases."""
        try:
            # Check for missing data patterns
            missing_data_score = 0.0
            # Check for data completeness
            if "events" in data:
                events = data["events"]
                if events:
                    # Check if we have complete data for all events
                    complete_events = [e for e in events if self._is_event_complete(e)]
                    completeness_ratio = len(complete_events) / len(events)
                    missing_data_score = 1.0 - completeness_ratio
            return missing_data_score
        except Exception as e:
            logger.error(f"Error checking data quality bias: {e}")
            return 0.0

    def _is_event_complete(self, event: dict) -> bool:
        """Check if an event has complete data."""
        required_fields = ["home_team", "away_team", "event_date", "sport"]
        return all(field in event and event[field] for field in required_fields)

    async def _check_model_fairness(self, data: dict) -> float:
        """Check for model fairness across different groups."""
        try:
            # This would check if the model treats different groups fairly
            # For now, return a placeholder
            return 0.95  # High fairness score
        except Exception as e:
            logger.error(f"Error checking model fairness: {e}")
            return 0.5

    async def _ethical_compliance_check(self, action: str, context: dict) -> bool:
        """Check ethical compliance of proposed actions."""
        try:
            # Define ethical guidelines
            ethical_guidelines = [
                "No manipulation of betting markets",
                "No exploitation of vulnerable users",
                "No violation of platform terms of service",
                "No excessive risk-taking",
                "No discriminatory practices",
                "No insider trading",
                "No market manipulation",
                "No fraudulent activities",
            ]
            # Check action against guidelines
            prompt = f"""
            Check if this action complies with ethical guidelines:
            Action: {action}
            Context: {context}
            Guidelines:
            {chr(10).join(ethical_guidelines)}
            Respond with 'COMPLIANT' or 'NON_COMPLIANT' and brief explanation.
            Focus on potential ethical violations.
            """
            response = await self.llm.ainvoke(prompt)
            is_compliant = "COMPLIANT" in response.content.upper()
            if not is_compliant:
                logger.warning(f"Ethical compliance check failed: {response.content}")
                await self._log_ethical_violation(action, context, response.content)
            return is_compliant
        except Exception as e:
            logger.error(f"Error in ethical compliance check: {e}")
            return False  # Fail safe - don't proceed if check fails

    async def _log_ethical_violation(
        self, action: str, context: dict, explanation: str
    ):
        """Log ethical violations for review."""
        violation_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "context": context,
            "explanation": explanation,
            "severity": "high",
        }
        logger.error(f"Ethical violation detected: {violation_log}")
        # In a real implementation, this would be stored in a database

    async def _anomaly_detection(self, metrics: dict) -> bool:
        """Detect and halt on critical anomalies."""
        try:
            anomalies = []
            # Check for unusual betting patterns
            if "bets" in metrics:
                bet_anomalies = await self._detect_betting_anomalies(metrics["bets"])
                anomalies.extend(bet_anomalies)
            # Check for unusual performance metrics
            if "performance" in metrics:
                performance_anomalies = await self._detect_performance_anomalies(
                    metrics["performance"]
                )
                anomalies.extend(performance_anomalies)
            # Check for unusual risk metrics
            if "risk" in metrics:
                risk_anomalies = await self._detect_risk_anomalies(metrics["risk"])
                anomalies.extend(risk_anomalies)
            # Check for data anomalies
            if "data" in metrics:
                data_anomalies = await self._detect_data_anomalies(metrics["data"])
                anomalies.extend(data_anomalies)
            # If critical anomalies found, halt system
            critical_anomalies = [
                a for a in anomalies if a.get("severity") == "critical"
            ]
            if critical_anomalies:
                logger.error(f"Critical anomalies detected: {critical_anomalies}")
                await self._halt_system(critical_anomalies)
                return True
            # Log non-critical anomalies
            if anomalies:
                logger.warning(f"Non-critical anomalies detected: {anomalies}")
            return False
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return True  # Halt on error

    async def _detect_betting_anomalies(self, bets: list[Bet]) -> list[dict]:
        """Detect anomalies in betting patterns."""
        anomalies = []
        try:
            if not bets:
                return anomalies
            # Check for unusual stake sizes
            stakes = [float(b.stake or 0) for b in bets if b.stake]
            if stakes:
                avg_stake = np.mean(stakes)
                std_stake = np.std(stakes)
                for bet in bets:
                    if bet.stake:
                        stake = float(bet.stake)
                        z_score = (
                            abs(stake - avg_stake) / std_stake if std_stake > 0 else 0
                        )
                        if z_score > 3:  # More than 3 standard deviations
                            anomalies.append(
                                {
                                    "type": "unusual_stake_size",
                                    "bet_id": bet.id,
                                    "stake": stake,
                                    "z_score": z_score,
                                    "severity": "high" if z_score > 5 else "medium",
                                }
                            )
            # Check for unusual EV values
            evs = [float(b.expected_value or 0) for b in bets if b.expected_value]
            if evs:
                # Calculate average EV for analysis
                _avg_ev = np.mean(evs)
                for bet in bets:
                    if bet.expected_value:
                        ev = float(bet.expected_value)
                        if ev > 0.5:  # Suspiciously high EV
                            anomalies.append(
                                {
                                    "type": "suspicious_ev",
                                    "bet_id": bet.id,
                                    "ev": ev,
                                    "severity": "critical",
                                }
                            )
            # Check for rapid betting (potential automation detection)
            if len(bets) > 1:
                sorted_bets = sorted(bets, key=lambda b: b.placed_at or datetime.min)
                for i in range(1, len(sorted_bets)):
                    time_diff = (
                        sorted_bets[i].placed_at - sorted_bets[i - 1].placed_at
                    ).total_seconds()
                    if time_diff < 30:  # Less than 30 seconds between bets
                        anomalies.append(
                            {
                                "type": "rapid_betting",
                                "bet_ids": [sorted_bets[i - 1].id, sorted_bets[i].id],
                                "time_diff": time_diff,
                                "severity": "high",
                            }
                        )
            return anomalies
        except Exception as e:
            logger.error(f"Error detecting betting anomalies: {e}")
            return anomalies

    async def _detect_performance_anomalies(self, performance: dict) -> list[dict]:
        """Detect anomalies in performance metrics."""
        anomalies = []
        try:
            # Check for unusual win rates
            win_rate = performance.get("win_rate", 0)
            if win_rate > 0.8:  # Suspiciously high win rate
                anomalies.append(
                    {
                        "type": "suspicious_win_rate",
                        "win_rate": win_rate,
                        "severity": "high",
                    }
                )
            elif win_rate < 0.2:  # Suspiciously low win rate
                anomalies.append(
                    {
                        "type": "very_low_win_rate",
                        "win_rate": win_rate,
                        "severity": "medium",
                    }
                )
            # Check for unusual ROI
            roi = performance.get("roi", 0)
            if roi > 0.5:  # Suspiciously high ROI
                anomalies.append(
                    {"type": "suspicious_roi", "roi": roi, "severity": "critical"}
                )
            elif roi < -0.3:  # Very negative ROI
                anomalies.append(
                    {"type": "very_negative_roi", "roi": roi, "severity": "high"}
                )
            return anomalies
        except Exception as e:
            logger.error(f"Error detecting performance anomalies: {e}")
            return anomalies

    async def _detect_risk_anomalies(self, risk: dict) -> list[dict]:
        """Detect anomalies in risk metrics."""
        anomalies = []
        try:
            # Check for excessive drawdown
            drawdown = risk.get("drawdown", 0)
            if drawdown > 0.3:  # More than 30% drawdown
                anomalies.append(
                    {
                        "type": "excessive_drawdown",
                        "drawdown": drawdown,
                        "severity": "critical",
                    }
                )
            # Check for excessive VaR
            var = risk.get("var", 0)
            if var > 0.1:  # More than 10% VaR
                anomalies.append(
                    {"type": "excessive_var", "var": var, "severity": "high"}
                )
            # Check for unusual volatility
            volatility = risk.get("volatility", 0)
            if volatility > 0.5:  # Very high volatility
                anomalies.append(
                    {
                        "type": "high_volatility",
                        "volatility": volatility,
                        "severity": "medium",
                    }
                )
            return anomalies
        except Exception as e:
            logger.error(f"Error detecting risk anomalies: {e}")
            return anomalies

    async def _detect_data_anomalies(self, data: dict) -> list[dict]:
        """Detect anomalies in data."""
        anomalies = []
        try:
            # Check for data quality issues
            if "quality_score" in data:
                quality_score = data["quality_score"]
                if quality_score < 0.7:  # Low data quality
                    anomalies.append(
                        {
                            "type": "low_data_quality",
                            "quality_score": quality_score,
                            "severity": "high",
                        }
                    )
            # Check for missing data
            if "missing_data_ratio" in data:
                missing_ratio = data["missing_data_ratio"]
                if missing_ratio > 0.2:  # More than 20% missing data
                    anomalies.append(
                        {
                            "type": "high_missing_data",
                            "missing_ratio": missing_ratio,
                            "severity": "medium",
                        }
                    )
            # Check for data staleness
            if "last_update" in data:
                last_update = data["last_update"]
                if isinstance(last_update, str):
                    last_update = datetime.fromisoformat(
                        last_update.replace("Z", "+00:00")
                    )
                time_diff = datetime.utcnow() - last_update
                if time_diff.total_seconds() > 3600:  # More than 1 hour old
                    anomalies.append(
                        {
                            "type": "stale_data",
                            "age_hours": time_diff.total_seconds() / 3600,
                            "severity": "medium",
                        }
                    )
            return anomalies
        except Exception as e:
            logger.error(f"Error detecting data anomalies: {e}")
            return anomalies

    async def _halt_system(self, anomalies: list[dict]):
        """Halt the system due to critical anomalies."""
        try:
            logger.critical("SYSTEM HALTED due to critical anomalies")
            # Log the halt
            halt_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "reason": "critical_anomalies",
                "anomalies": anomalies,
                "system_state": "halted",
            }
            logger.critical(f"System halt details: {halt_log}")
            # In a real implementation, this would:
            # 1. Stop all betting activities
            # 2. Send emergency alerts
            # 3. Initiate recovery procedures
            # 4. Notify administrators
        except Exception as e:
            logger.error(f"Error halting system: {e}")

    async def _prevent_cascading_errors(self, error: Exception) -> bool:
        """Prevent error propagation across agents."""
        try:
            error_info = {
                "timestamp": datetime.utcnow().isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "severity": self._assess_error_severity(error),
            }
            logger.error(f"Error detected: {error_info}")
            # Check if error should trigger system halt
            if error_info["severity"] == "critical":
                await self._halt_system(
                    [
                        {
                            "type": "critical_error",
                            "error": error_info,
                            "severity": "critical",
                        }
                    ]
                )
                return True
            # For non-critical errors, implement circuit breaker
            if error_info["severity"] == "high":
                await self._activate_circuit_breaker(error_info)
                return True
            # Log and continue for low severity errors
            return False
        except Exception as e:
            logger.error(f"Error in error prevention: {e}")
            return True  # Halt on error in error handling

    def _assess_error_severity(self, error: Exception) -> str:
        """Assess the severity of an error."""
        error_message = str(error).lower()
        # Critical errors
        critical_keywords = [
            "database",
            "authentication",
            "api_key",
            "security",
            "corruption",
        ]
        if any(keyword in error_message for keyword in critical_keywords):
            return "critical"
        # High severity errors
        high_keywords = ["connection", "timeout", "rate_limit", "validation"]
        if any(keyword in error_message for keyword in high_keywords):
            return "high"
        # Medium severity errors
        medium_keywords = ["parsing", "format", "missing"]
        if any(keyword in error_message for keyword in medium_keywords):
            return "medium"
        return "low"

    async def _activate_circuit_breaker(self, error_info: dict):
        """Activate circuit breaker for high severity errors."""
        try:
            logger.warning("Circuit breaker activated")
            # In a real implementation, this would:
            # 1. Stop processing new requests
            # 2. Wait for a cooldown period
            # 3. Gradually resume operations
            # 4. Monitor for repeated errors
            circuit_breaker_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "trigger": error_info,
                "status": "activated",
                "cooldown_period": 300,  # 5 minutes
            }
            logger.warning(f"Circuit breaker details: {circuit_breaker_log}")
        except Exception as e:
            logger.error(f"Error activating circuit breaker: {e}")

    async def _risk_assessment(self, betting_decision: dict) -> dict[str, Any]:
        """Comprehensive risk assessment for betting decisions."""
        try:
            risk_assessment = {
                "overall_risk_score": 0.0,
                "risk_factors": [],
                "recommendations": [],
                "requires_halt": False,
                "assessment_timestamp": datetime.utcnow().isoformat(),
            }
            # Check stake size risk
            stake = betting_decision.get("stake", 0)
            bankroll = betting_decision.get("bankroll", 1)
            if bankroll > 0:
                stake_ratio = stake / bankroll
                if stake_ratio > 0.05:  # More than 5% of bankroll
                    risk_assessment["risk_factors"].append(
                        {
                            "factor": "high_stake_ratio",
                            "value": stake_ratio,
                            "severity": "high" if stake_ratio > 0.1 else "medium",
                        }
                    )
            # Check EV risk
            ev = betting_decision.get("expected_value", 0)
            if ev < 0:  # Negative EV
                risk_assessment["risk_factors"].append(
                    {"factor": "negative_ev", "value": ev, "severity": "critical"}
                )
            elif ev < 0.02:  # Very low EV
                risk_assessment["risk_factors"].append(
                    {"factor": "very_low_ev", "value": ev, "severity": "medium"}
                )
            # Check odds risk
            odds = betting_decision.get("odds", 0)
            if odds > 10:  # Very high odds
                risk_assessment["risk_factors"].append(
                    {"factor": "very_high_odds", "value": odds, "severity": "medium"}
                )
            # Check concentration risk
            sport = betting_decision.get("sport", "")
            if sport:
                # Check if too much exposure to one sport
                sport_exposure = await self._check_sport_exposure(sport)
                if sport_exposure > 0.3:  # More than 30% in one sport
                    risk_assessment["risk_factors"].append(
                        {
                            "factor": "high_sport_concentration",
                            "sport": sport,
                            "exposure": sport_exposure,
                            "severity": "medium",
                        }
                    )
            # Calculate overall risk score
            if risk_assessment["risk_factors"]:
                severity_scores = {
                    "low": 0.1,
                    "medium": 0.3,
                    "high": 0.6,
                    "critical": 1.0,
                }
                risk_scores = [
                    severity_scores.get(factor.get("severity", "low"), 0.1)
                    for factor in risk_assessment["risk_factors"]
                ]
                risk_assessment["overall_risk_score"] = np.mean(risk_scores)
            # Generate recommendations
            risk_assessment["recommendations"] = (
                await self._generate_risk_recommendations(
                    risk_assessment["risk_factors"]
                )
            )
            # Check if halt is required
            if risk_assessment["overall_risk_score"] > 0.7:  # High risk threshold
                risk_assessment["requires_halt"] = True
            return risk_assessment
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return {"error": str(e)}

    async def _check_sport_exposure(self, sport: str) -> float:
        """Check exposure to a specific sport."""
        try:
            # Get recent bets for this sport
            recent_bets = await self.db_manager.get_bets(sport=sport, days_back=7)
            if not recent_bets:
                return 0.0
            # Calculate total stake for this sport
            sport_stake = sum([float(b.stake or 0) for b in recent_bets])
            # Get total stake across all sports
            all_bets = await self.db_manager.get_bets(days_back=7)
            total_stake = sum([float(b.stake or 0) for b in all_bets])
            if total_stake > 0:
                return sport_stake / total_stake
            return 0.0
        except Exception as e:
            logger.error(f"Error checking sport exposure: {e}")
            return 0.0

    async def _generate_risk_recommendations(
        self, risk_factors: list[dict]
    ) -> list[str]:
        """Generate recommendations based on risk factors."""
        recommendations = []
        for factor in risk_factors:
            factor_type = factor.get("factor", "")
            if factor_type == "high_stake_ratio":
                recommendations.append(
                    "Consider reducing stake size to below 5% of bankroll"
                )
            elif factor_type == "negative_ev":
                recommendations.append("Avoid bets with negative expected value")
            elif factor_type == "very_low_ev":
                recommendations.append(
                    "Consider higher EV threshold for better risk-adjusted returns"
                )
            elif factor_type == "very_high_odds":
                recommendations.append(
                    "High odds bets carry higher variance - consider position sizing"
                )
            elif factor_type == "high_sport_concentration":
                recommendations.append(
                    "Diversify across multiple sports to reduce concentration risk"
                )
        return recommendations

    async def _safety_monitoring(self) -> dict[str, Any]:
        """Continuous safety monitoring and alerting."""
        try:
            monitoring_results = {
                "system_health": "healthy",
                "alerts": [],
                "metrics": {},
                "timestamp": datetime.utcnow().isoformat(),
            }
            # Check system health
            health_status = await self._check_system_health()
            monitoring_results["system_health"] = health_status["status"]
            if health_status["status"] != "healthy":
                monitoring_results["alerts"].append(
                    {
                        "type": "system_health",
                        "message": f"System health: {health_status['status']}",
                        "severity": (
                            "high"
                            if health_status["status"] == "critical"
                            else "medium"
                        ),
                    }
                )
            # Check performance metrics
            performance_metrics = await self._get_performance_metrics()
            monitoring_results["metrics"]["performance"] = performance_metrics
            # Check for performance anomalies
            if performance_metrics.get("win_rate", 0) < 0.4:
                monitoring_results["alerts"].append(
                    {
                        "type": "low_performance",
                        "message": f"Low win rate: {performance_metrics.get('win_rate', 0):.3f}",
                        "severity": "medium",
                    }
                )
            # Check risk metrics
            risk_metrics = await self._get_risk_metrics()
            monitoring_results["metrics"]["risk"] = risk_metrics
            if risk_metrics.get("drawdown", 0) > 0.2:
                monitoring_results["alerts"].append(
                    {
                        "type": "high_drawdown",
                        "message": f"High drawdown: {risk_metrics.get('drawdown', 0):.3f}",
                        "severity": "high",
                    }
                )
            return monitoring_results
        except Exception as e:
            logger.error(f"Error in safety monitoring: {e}")
            return {"error": str(e)}

    async def _check_system_health(self) -> dict[str, Any]:
        """Check overall system health."""
        try:
            health_status = {
                "status": "healthy",
                "checks": {},
                "timestamp": datetime.utcnow().isoformat(),
            }
            # Check database connectivity
            try:
                await self.db_manager.get_current_bankroll()
                health_status["checks"]["database"] = "healthy"
            except Exception:
                health_status["checks"]["database"] = "unhealthy"
                health_status["status"] = "critical"
            # Check API connectivity
            try:
                # This would check API connectivity
                health_status["checks"]["api"] = "healthy"
            except Exception:
                health_status["checks"]["api"] = "unhealthy"
                health_status["status"] = "critical"
            # Check agent status
            health_status["checks"]["agents"] = "healthy"
            return health_status
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {"status": "unknown", "error": str(e)}

    async def _get_performance_metrics(self) -> dict[str, Any]:
        """Get current performance metrics."""
        try:
            # Get recent bets
            recent_bets = await self.db_manager.get_bets(days_back=7)
            if not recent_bets:
                return {"win_rate": 0, "roi": 0, "total_bets": 0}
            # Calculate metrics
            total_bets = len(recent_bets)
            winning_bets = [b for b in recent_bets if b.result == "win"]
            win_rate = len(winning_bets) / total_bets if total_bets > 0 else 0
            total_stake = sum([float(b.stake or 0) for b in recent_bets])
            total_return = sum([float(b.return_amount or 0) for b in recent_bets])
            roi = (total_return - total_stake) / total_stake if total_stake > 0 else 0
            return {
                "win_rate": win_rate,
                "roi": roi,
                "total_bets": total_bets,
                "winning_bets": len(winning_bets),
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"win_rate": 0, "roi": 0, "total_bets": 0}

    async def _get_risk_metrics(self) -> dict[str, Any]:
        """Get current risk metrics."""
        try:
            # Get recent bets for risk calculation
            recent_bets = await self.db_manager.get_bets(days_back=30)
            if not recent_bets:
                return {"drawdown": 0, "var": 0, "volatility": 0}
            # Calculate returns
            returns = []
            for bet in recent_bets:
                if bet.stake and bet.return_amount:
                    stake = float(bet.stake)
                    return_amount = float(bet.return_amount)
                    if stake > 0:
                        returns.append((return_amount - stake) / stake)
            if not returns:
                return {"drawdown": 0, "var": 0, "volatility": 0}
            # Calculate risk metrics
            returns_array = np.array(returns)
            # Drawdown (simplified)
            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = np.max((running_max - cumulative_returns) / running_max)
            # Value at Risk (95% confidence)
            var = np.percentile(returns_array, 5)
            # Volatility
            volatility = np.std(returns_array)
            return {
                "drawdown": float(drawdown),
                "var": float(var),
                "volatility": float(volatility),
            }
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {"drawdown": 0, "var": 0, "volatility": 0}
