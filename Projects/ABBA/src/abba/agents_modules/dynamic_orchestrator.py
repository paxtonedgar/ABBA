"""
Dynamic Agent Orchestrator for ABMBA system.
Manages real-time agent collaboration and debates.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import structlog
from crewai import Agent as CrewAgent
from langchain_openai import ChatOpenAI

logger = structlog.get_logger()


class DynamicAgentOrchestrator:
    """Orchestrates dynamic agent collaboration and real-time interactions."""

    def __init__(self, config: dict, agents: dict[str, Any]):
        self.config = config
        self.agents = agents
        self.debate_history = []
        self.confidence_threshold = config.get("dynamic_collaboration", {}).get(
            "confidence_threshold", 0.8
        )
        self.debate_timeout = config.get("dynamic_collaboration", {}).get(
            "debate_timeout", 300
        )
        self.max_sub_agents = config.get("dynamic_collaboration", {}).get(
            "max_sub_agents", 5
        )
        self.llm = ChatOpenAI(
            model=config["apis"]["openai"]["model"],
            api_key=config["apis"]["openai"]["api_key"],
            temperature=0.1,
        )
        # Track active sub-agents
        self.active_sub_agents = {}
        self.agent_communication_log = []

    async def debate_results(self, agents: list[str], results: dict) -> dict[str, Any]:
        """Facilitate agent debates and cross-validation."""
        try:
            debate_results = {
                "participants": agents,
                "initial_results": results,
                "debate_outcome": {},
                "consensus_reached": False,
                "confidence_score": 0.0,
                "debate_timestamp": datetime.utcnow().isoformat(),
            }
            # Collect opinions from each agent
            opinions = {}
            for agent_name in agents:
                if agent_name in self.agents:
                    agent_opinion = await self._get_agent_opinion(
                        self.agents[agent_name], results
                    )
                    opinions[agent_name] = agent_opinion
                    # Log communication
                    self.agent_communication_log.append(
                        {
                            "timestamp": datetime.utcnow().isoformat(),
                            "agent": agent_name,
                            "action": "opinion_requested",
                            "result": "success",
                        }
                    )
            # Facilitate debate
            debate_outcome = await self._facilitate_debate(opinions, results)
            # Check for consensus
            consensus_reached = await self._check_consensus(debate_outcome)
            # Calculate confidence score
            confidence_score = await self._calculate_confidence(debate_outcome)
            debate_results.update(
                {
                    "debate_outcome": debate_outcome,
                    "consensus_reached": consensus_reached,
                    "confidence_score": confidence_score,
                    "opinions": opinions,
                }
            )
            # Store debate history
            self.debate_history.append(debate_results)
            logger.info(
                f"Debate completed. Consensus: {consensus_reached}, Confidence: {confidence_score:.3f}"
            )
            return debate_results
        except Exception as e:
            logger.error(f"Error in agent debate: {e}")
            return {"error": str(e)}

    async def _get_agent_opinion(self, agent: Any, results: dict) -> dict[str, Any]:
        """Get opinion from a specific agent."""
        try:
            # Create a prompt for the agent to analyze results
            prompt = f"""
            Analyze these betting results and provide your expert opinion:
            Results: {json.dumps(results, indent=2)}
            Please provide:
            1. Your assessment of the results (1-10 scale)
            2. Key concerns or positive aspects
            3. Recommendations for improvement
            4. Confidence in your assessment (0-1)
            Format your response as JSON with keys: assessment, concerns, positives, recommendations, confidence
            """
            # Use the agent's LLM to generate opinion
            if hasattr(agent, "llm"):
                response = await agent.llm.ainvoke(prompt)
                # Try to parse JSON response
                try:
                    opinion = json.loads(response.content)
                except json.JSONDecodeError:
                    # Fallback to text parsing
                    opinion = {
                        "assessment": 5,
                        "concerns": ["Unable to parse structured response"],
                        "positives": ["Agent provided response"],
                        "recommendations": ["Improve response format"],
                        "confidence": 0.5,
                        "raw_response": response.content,
                    }
            else:
                # Fallback for agents without direct LLM access
                opinion = {
                    "assessment": 5,
                    "concerns": ["Agent LLM not accessible"],
                    "positives": ["Agent available for debate"],
                    "recommendations": ["Enable direct LLM access"],
                    "confidence": 0.3,
                }
            return opinion
        except Exception as e:
            logger.error(f"Error getting agent opinion: {e}")
            return {
                "assessment": 1,
                "concerns": [f"Error: {str(e)}"],
                "positives": [],
                "recommendations": ["Fix agent communication"],
                "confidence": 0.0,
            }

    async def _facilitate_debate(
        self, opinions: dict[str, dict], results: dict
    ) -> dict[str, Any]:
        """Facilitate debate between agents."""
        try:
            debate_outcome = {
                "debate_rounds": [],
                "final_consensus": {},
                "disagreements": [],
                "agreements": [],
            }
            # First round: Share opinions
            round_1 = {
                "round": 1,
                "type": "opinion_sharing",
                "participants": list(opinions.keys()),
                "opinions": opinions,
            }
            debate_outcome["debate_rounds"].append(round_1)
            # Second round: Identify disagreements
            disagreements = await self._identify_disagreements(opinions)
            agreements = await self._identify_agreements(opinions)
            round_2 = {
                "round": 2,
                "type": "disagreement_identification",
                "disagreements": disagreements,
                "agreements": agreements,
            }
            debate_outcome["debate_rounds"].append(round_2)
            # Third round: Resolve disagreements
            if disagreements:
                resolution = await self._resolve_disagreements(disagreements, opinions)
                round_3 = {
                    "round": 3,
                    "type": "disagreement_resolution",
                    "resolution": resolution,
                }
                debate_outcome["debate_rounds"].append(round_3)
            # Final consensus
            final_consensus = await self._reach_final_consensus(opinions, disagreements)
            debate_outcome["final_consensus"] = final_consensus
            debate_outcome["disagreements"] = disagreements
            debate_outcome["agreements"] = agreements
            return debate_outcome
        except Exception as e:
            logger.error(f"Error facilitating debate: {e}")
            return {"error": str(e)}

    async def _identify_disagreements(self, opinions: dict[str, dict]) -> list[dict]:
        """Identify areas of disagreement between agents."""
        try:
            disagreements = []
            # Compare assessments
            assessments = [op.get("assessment", 5) for op in opinions.values()]
            if len(assessments) > 1:
                assessment_std = np.std(assessments)
                if assessment_std > 2:  # Significant disagreement
                    disagreements.append(
                        {
                            "type": "assessment_disagreement",
                            "agents": list(opinions.keys()),
                            "assessments": assessments,
                            "std": assessment_std,
                            "severity": "high" if assessment_std > 3 else "medium",
                        }
                    )
            # Compare confidence levels
            confidences = [op.get("confidence", 0.5) for op in opinions.values()]
            if len(confidences) > 1:
                confidence_std = np.std(confidences)
                if confidence_std > 0.3:  # Significant confidence disagreement
                    disagreements.append(
                        {
                            "type": "confidence_disagreement",
                            "agents": list(opinions.keys()),
                            "confidences": confidences,
                            "std": confidence_std,
                            "severity": "high" if confidence_std > 0.5 else "medium",
                        }
                    )
            # Compare concerns
            all_concerns = []
            for agent, opinion in opinions.items():
                concerns = opinion.get("concerns", [])
                for concern in concerns:
                    all_concerns.append({"agent": agent, "concern": concern})
            # Find conflicting concerns
            concern_groups = {}
            for concern_item in all_concerns:
                concern_text = concern_item["concern"].lower()
                if concern_text not in concern_groups:
                    concern_groups[concern_text] = []
                concern_groups[concern_text].append(concern_item["agent"])
            # Identify concerns mentioned by only some agents
            for concern, agents in concern_groups.items():
                if len(agents) < len(opinions):
                    disagreements.append(
                        {
                            "type": "concern_disagreement",
                            "concern": concern,
                            "agents_with_concern": agents,
                            "agents_without_concern": [
                                a for a in opinions.keys() if a not in agents
                            ],
                            "severity": "medium",
                        }
                    )
            return disagreements
        except Exception as e:
            logger.error(f"Error identifying disagreements: {e}")
            return []

    async def _identify_agreements(self, opinions: dict[str, dict]) -> list[dict]:
        """Identify areas of agreement between agents."""
        try:
            agreements = []
            # Find common concerns
            all_concerns = []
            for _agent, opinion in opinions.items():
                concerns = opinion.get("concerns", [])
                for concern in concerns:
                    all_concerns.append(concern.lower())
            # Count concern frequency
            concern_counts = {}
            for concern in all_concerns:
                concern_counts[concern] = concern_counts.get(concern, 0) + 1
            # Find concerns mentioned by multiple agents
            for concern, count in concern_counts.items():
                if count > 1:
                    agreements.append(
                        {
                            "type": "shared_concern",
                            "concern": concern,
                            "agent_count": count,
                            "total_agents": len(opinions),
                        }
                    )
            # Find common recommendations
            all_recommendations = []
            for _agent, opinion in opinions.items():
                recommendations = opinion.get("recommendations", [])
                for rec in recommendations:
                    all_recommendations.append(rec.lower())
            # Count recommendation frequency
            rec_counts = {}
            for rec in all_recommendations:
                rec_counts[rec] = rec_counts.get(rec, 0) + 1
            # Find recommendations shared by multiple agents
            for rec, count in rec_counts.items():
                if count > 1:
                    agreements.append(
                        {
                            "type": "shared_recommendation",
                            "recommendation": rec,
                            "agent_count": count,
                            "total_agents": len(opinions),
                        }
                    )
            return agreements
        except Exception as e:
            logger.error(f"Error identifying agreements: {e}")
            return []

    async def _resolve_disagreements(
        self, disagreements: list[dict], opinions: dict[str, dict]
    ) -> dict[str, Any]:
        """Resolve disagreements between agents."""
        try:
            resolution = {
                "resolved_disagreements": [],
                "unresolved_disagreements": [],
                "resolution_method": "consensus_building",
            }
            for disagreement in disagreements:
                if disagreement["type"] == "assessment_disagreement":
                    # Use weighted average based on confidence
                    assessments = disagreement["assessments"]
                    confidences = [
                        opinions[agent].get("confidence", 0.5)
                        for agent in disagreement["agents"]
                    ]
                    # Weighted average
                    weighted_assessment = sum(
                        a * c for a, c in zip(assessments, confidences, strict=False)
                    ) / sum(confidences)
                    resolution["resolved_disagreements"].append(
                        {
                            "disagreement": disagreement,
                            "resolution": weighted_assessment,
                            "method": "weighted_average",
                        }
                    )
                elif disagreement["type"] == "confidence_disagreement":
                    # Use highest confidence as resolution
                    confidences = disagreement["confidences"]
                    max_confidence = max(confidences)
                    max_confidence_agent = disagreement["agents"][
                        confidences.index(max_confidence)
                    ]
                    resolution["resolved_disagreements"].append(
                        {
                            "disagreement": disagreement,
                            "resolution": max_confidence,
                            "method": "highest_confidence",
                            "agent": max_confidence_agent,
                        }
                    )
                elif disagreement["type"] == "concern_disagreement":
                    # Include concern in final assessment
                    resolution["resolved_disagreements"].append(
                        {
                            "disagreement": disagreement,
                            "resolution": "include_concern",
                            "method": "consensus_inclusion",
                        }
                    )
                else:
                    # Unresolved disagreement
                    resolution["unresolved_disagreements"].append(disagreement)
            return resolution
        except Exception as e:
            logger.error(f"Error resolving disagreements: {e}")
            return {"error": str(e)}

    async def _reach_final_consensus(
        self, opinions: dict[str, dict], disagreements: list[dict]
    ) -> dict[str, Any]:
        """Reach final consensus after resolving disagreements."""
        try:
            consensus = {
                "final_assessment": 0.0,
                "final_confidence": 0.0,
                "consolidated_concerns": [],
                "consolidated_recommendations": [],
                "consensus_method": "weighted_average",
            }
            # Calculate weighted final assessment
            assessments = [op.get("assessment", 5) for op in opinions.values()]
            confidences = [op.get("confidence", 0.5) for op in opinions.values()]
            if confidences and sum(confidences) > 0:
                consensus["final_assessment"] = sum(
                    a * c for a, c in zip(assessments, confidences, strict=False)
                ) / sum(confidences)
                consensus["final_confidence"] = np.mean(confidences)
            # Consolidate concerns
            all_concerns = []
            for opinion in opinions.values():
                concerns = opinion.get("concerns", [])
                all_concerns.extend(concerns)
            # Remove duplicates and rank by frequency
            concern_counts = {}
            for concern in all_concerns:
                concern_counts[concern] = concern_counts.get(concern, 0) + 1
            consensus["consolidated_concerns"] = sorted(
                concern_counts.items(), key=lambda x: x[1], reverse=True
            )[
                :5
            ]  # Top 5 concerns
            # Consolidate recommendations
            all_recommendations = []
            for opinion in opinions.values():
                recommendations = opinion.get("recommendations", [])
                all_recommendations.extend(recommendations)
            # Remove duplicates and rank by frequency
            rec_counts = {}
            for rec in all_recommendations:
                rec_counts[rec] = rec_counts.get(rec, 0) + 1
            consensus["consolidated_recommendations"] = sorted(
                rec_counts.items(), key=lambda x: x[1], reverse=True
            )[
                :5
            ]  # Top 5 recommendations
            return consensus
        except Exception as e:
            logger.error(f"Error reaching final consensus: {e}")
            return {"error": str(e)}

    async def _check_consensus(self, debate_outcome: dict) -> bool:
        """Check if consensus was reached."""
        try:
            final_consensus = debate_outcome.get("final_consensus", {})
            # Check if we have a valid consensus
            if "error" in final_consensus:
                return False
            # Check confidence level
            final_confidence = final_consensus.get("final_confidence", 0)
            if final_confidence < self.confidence_threshold:
                return False
            # Check if there are too many unresolved disagreements
            unresolved = debate_outcome.get("unresolved_disagreements", [])
            if len(unresolved) > 2:  # Too many unresolved issues
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking consensus: {e}")
            return False

    async def _calculate_confidence(self, debate_outcome: dict) -> float:
        """Calculate confidence score from debate outcome."""
        try:
            final_consensus = debate_outcome.get("final_consensus", {})
            if "error" in final_consensus:
                return 0.0
            # Base confidence from final consensus
            base_confidence = final_consensus.get("final_confidence", 0.0)
            # Adjust based on agreement level
            agreements = debate_outcome.get("agreements", [])
            disagreements = debate_outcome.get("disagreements", [])
            total_issues = len(agreements) + len(disagreements)
            if total_issues > 0:
                agreement_ratio = len(agreements) / total_issues
                # Boost confidence for high agreement
                agreement_boost = agreement_ratio * 0.2
                base_confidence += agreement_boost
            # Penalize for unresolved disagreements
            unresolved = debate_outcome.get("unresolved_disagreements", [])
            if unresolved:
                penalty = len(unresolved) * 0.1
                base_confidence -= penalty
            return max(0.0, min(1.0, base_confidence))  # Clamp between 0 and 1
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0

    async def adaptive_confidence_loop(self, confidence: float) -> bool:
        """Retrain if confidence <80%."""
        try:
            if confidence < self.confidence_threshold:
                logger.warning(
                    f"Low confidence detected: {confidence:.3f} < {self.confidence_threshold}"
                )
                # Trigger model retraining
                retraining_triggered = await self._trigger_model_retraining(confidence)
                if retraining_triggered:
                    # Re-run analysis with updated models
                    await self._rerun_analysis()
                    return True  # Retraining triggered
                else:
                    logger.error("Failed to trigger model retraining")
                    return False
            return False  # No retraining needed
        except Exception as e:
            logger.error(f"Error in confidence loop: {e}")
            return False

    async def _trigger_model_retraining(self, confidence: float) -> bool:
        """Trigger model retraining due to low confidence."""
        try:
            logger.info(
                f"Triggering model retraining due to low confidence: {confidence:.3f}"
            )
            # In a real implementation, this would:
            # 1. Notify the simulation agent to retrain models
            # 2. Collect additional training data
            # 3. Update model parameters
            # 4. Validate new models
            # For now, simulate the process
            await asyncio.sleep(1)  # Simulate retraining time
            logger.info("Model retraining completed")
            return True
        except Exception as e:
            logger.error(f"Error triggering model retraining: {e}")
            return False

    async def _rerun_analysis(self) -> bool:
        """Re-run analysis with updated models."""
        try:
            logger.info("Re-running analysis with updated models")
            # In a real implementation, this would:
            # 1. Re-run simulations with new models
            # 2. Re-evaluate betting opportunities
            # 3. Update confidence scores
            # For now, simulate the process
            await asyncio.sleep(1)  # Simulate analysis time
            logger.info("Analysis re-run completed")
            return True
        except Exception as e:
            logger.error(f"Error re-running analysis: {e}")
            return False

    async def spawn_sub_agent(self, context: str, task_type: str) -> Any | None:
        """Create temporary agents for specific tasks."""
        try:
            # Check if we've reached the maximum number of sub-agents
            if len(self.active_sub_agents) >= self.max_sub_agents:
                logger.warning(f"Maximum sub-agents ({self.max_sub_agents}) reached")
                return None
            # Define sub-agent configurations
            sub_agent_configs = {
                "arbitrage_verification": {
                    "role": "Arbitrage Verification Specialist",
                    "goal": "Verify arbitrage opportunities across multiple books",
                    "tools": ["verify_arbitrage", "cross_check_odds", "calculate_ev"],
                    "backstory": "Expert in identifying and verifying arbitrage opportunities across different betting platforms.",
                },
                "volatility_analysis": {
                    "role": "Volatility Analysis Specialist",
                    "goal": "Analyze market volatility and adjust risk parameters",
                    "tools": ["calculate_volatility", "adjust_risk", "monitor_market"],
                    "backstory": "Specialist in market volatility analysis and dynamic risk adjustment.",
                },
                "anomaly_investigation": {
                    "role": "Anomaly Investigation Specialist",
                    "goal": "Investigate and resolve data anomalies",
                    "tools": ["investigate_anomaly", "validate_data", "correct_errors"],
                    "backstory": "Expert in detecting and resolving data anomalies and quality issues.",
                },
                "market_microstructure": {
                    "role": "Market Microstructure Analyst",
                    "goal": "Analyze market microstructure and liquidity patterns",
                    "tools": ["analyze_liquidity", "detect_patterns", "assess_impact"],
                    "backstory": "Specialist in market microstructure analysis and liquidity pattern detection.",
                },
                "sentiment_analysis": {
                    "role": "Market Sentiment Analyst",
                    "goal": "Analyze market sentiment and social media signals",
                    "tools": ["analyze_sentiment", "monitor_social", "assess_impact"],
                    "backstory": "Expert in sentiment analysis and social media signal processing.",
                },
            }
            if task_type not in sub_agent_configs:
                logger.error(f"Unknown task type: {task_type}")
                return None
            config = sub_agent_configs[task_type]
            # Create temporary agent
            sub_agent = CrewAgent(
                role=config["role"],
                goal=config["goal"],
                backstory=f"{config['backstory']} Temporary agent created for {task_type} in context: {context}",
                tools=config["tools"],
                llm=self.llm,
                verbose=True,
            )
            # Generate unique ID for sub-agent
            sub_agent_id = f"{task_type}_{len(self.active_sub_agents)}_{datetime.utcnow().timestamp()}"
            # Store sub-agent
            self.active_sub_agents[sub_agent_id] = {
                "agent": sub_agent,
                "task_type": task_type,
                "context": context,
                "created_at": datetime.utcnow(),
                "status": "active",
            }
            logger.info(f"Spawned sub-agent {sub_agent_id} for {task_type}")
            # Log communication
            self.agent_communication_log.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": "sub_agent_spawned",
                    "sub_agent_id": sub_agent_id,
                    "task_type": task_type,
                    "context": context,
                }
            )
            return sub_agent
        except Exception as e:
            logger.error(f"Error spawning sub-agent: {e}")
            return None

    async def terminate_sub_agent(self, sub_agent_id: str) -> bool:
        """Terminate a sub-agent."""
        try:
            if sub_agent_id in self.active_sub_agents:
                sub_agent_info = self.active_sub_agents[sub_agent_id]
                sub_agent_info["status"] = "terminated"
                sub_agent_info["terminated_at"] = datetime.utcnow()
                # Remove from active agents
                del self.active_sub_agents[sub_agent_id]
                logger.info(f"Terminated sub-agent {sub_agent_id}")
                # Log communication
                self.agent_communication_log.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "action": "sub_agent_terminated",
                        "sub_agent_id": sub_agent_id,
                        "task_type": sub_agent_info["task_type"],
                    }
                )
                return True
            return False
        except Exception as e:
            logger.error(f"Error terminating sub-agent: {e}")
            return False

    async def get_agent_status(self) -> dict[str, Any]:
        """Get status of all agents and orchestrator."""
        try:
            status = {
                "orchestrator_status": "active",
                "confidence_threshold": self.confidence_threshold,
                "active_sub_agents": len(self.active_sub_agents),
                "max_sub_agents": self.max_sub_agents,
                "debate_history_count": len(self.debate_history),
                "communication_log_count": len(self.agent_communication_log),
                "sub_agents": {},
            }
            # Add sub-agent details
            for sub_agent_id, info in self.active_sub_agents.items():
                status["sub_agents"][sub_agent_id] = {
                    "task_type": info["task_type"],
                    "context": info["context"],
                    "status": info["status"],
                    "created_at": info["created_at"].isoformat(),
                    "age_seconds": (
                        datetime.utcnow() - info["created_at"]
                    ).total_seconds(),
                }
            return status
        except Exception as e:
            logger.error(f"Error getting agent status: {e}")
            return {"error": str(e)}

    async def cleanup_old_sub_agents(self, max_age_hours: int = 24) -> int:
        """Clean up old sub-agents."""
        try:
            current_time = datetime.utcnow()
            terminated_count = 0
            for sub_agent_id, info in list(self.active_sub_agents.items()):
                age_hours = (current_time - info["created_at"]).total_seconds() / 3600
                if age_hours > max_age_hours:
                    await self.terminate_sub_agent(sub_agent_id)
                    terminated_count += 1
            if terminated_count > 0:
                logger.info(f"Cleaned up {terminated_count} old sub-agents")
            return terminated_count
        except Exception as e:
            logger.error(f"Error cleaning up sub-agents: {e}")
            return 0

    async def get_debate_summary(self, days_back: int = 7) -> dict[str, Any]:
        """Get summary of recent debates."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days_back)
            recent_debates = [
                debate
                for debate in self.debate_history
                if datetime.fromisoformat(debate["debate_timestamp"]) > cutoff_time
            ]
            if not recent_debates:
                return {"message": f"No debates in the last {days_back} days"}
            # Calculate statistics
            consensus_rate = sum(
                1 for d in recent_debates if d["consensus_reached"]
            ) / len(recent_debates)
            avg_confidence = np.mean([d["confidence_score"] for d in recent_debates])
            # Most common participants
            all_participants = []
            for debate in recent_debates:
                all_participants.extend(debate["participants"])
            participant_counts = {}
            for participant in all_participants:
                participant_counts[participant] = (
                    participant_counts.get(participant, 0) + 1
                )
            most_active_participants = sorted(
                participant_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]
            return {
                "total_debates": len(recent_debates),
                "consensus_rate": consensus_rate,
                "average_confidence": avg_confidence,
                "most_active_participants": most_active_participants,
                "period_days": days_back,
            }
        except Exception as e:
            logger.error(f"Error getting debate summary: {e}")
            return {"error": str(e)}
