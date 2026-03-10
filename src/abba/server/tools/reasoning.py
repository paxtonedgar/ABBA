"""Reasoning observability tools — think-aloud and session replay.

These tools let the calling LLM externalize its reasoning chain into
DuckDB so users can inspect *why* the agent made the decisions it did,
what it was uncertain about, and where it sees gaps in the data pipeline.
"""

from __future__ import annotations

import time
from typing import Any


class ReasoningToolsMixin:
    """Mixin providing think and session_replay tools."""

    def think(
        self,
        phase: str,
        plan: str | None = None,
        uncertainty: list[str] | None = None,
        data_trust: list[dict[str, Any]] | None = None,
        workflow_gaps: list[str] | None = None,
        want_to_verify: list[str] | None = None,
        raw_thought: str | None = None,
    ) -> dict[str, Any]:
        """Record structured reasoning between tool calls.

        The LLM should call this BEFORE analytical tools to articulate:
        - What it plans to do and why (plan)
        - What it's uncertain about despite the numbers (uncertainty)
        - Whether it trusts specific data sources and why (data_trust)
        - What gaps or shortcomings it sees in the workflow (workflow_gaps)
        - What it would cross-reference if it could (want_to_verify)

        Parameters
        ----------
        phase : str
            Where in the workflow this thought occurs.
            One of: "planning", "pre_analysis", "post_analysis",
            "synthesis", "caveat_check", "recommendation".
        plan : str, optional
            What the agent intends to do next and why.
        uncertainty : list[str], optional
            Specific uncertainties the agent has despite the numbers.
            e.g. ["Goalie stats are season-wide, not recent form",
                   "No way to know if lineup tonight differs from roster data"]
        data_trust : list[dict], optional
            Per-source trust assessments. Each dict should have:
            {"source": str, "trust_level": "high"|"medium"|"low"|"unknown",
             "reason": str}
        workflow_gaps : list[str], optional
            Shortcomings in the current analytical workflow.
            e.g. ["No back-to-back fatigue adjustment",
                   "Elo doesn't account for roster turnover mid-season"]
        want_to_verify : list[str], optional
            Things the agent would want to check but can't.
            e.g. ["Is the starting goalie confirmed for tonight?",
                   "Are these odds from before or after the injury news?"]
        raw_thought : str, optional
            Free-form reasoning text for anything that doesn't fit above.
        """
        start = time.time()

        # Build a context snapshot of what the agent has seen so far
        session = self.storage.get_session(self._session_id)
        context_snapshot = {
            "tool_calls_so_far": session.get("tool_calls", 0) if session else 0,
            "data_freshness_ts": self._last_refresh_ts,
            "phase": phase,
        }

        self.storage.log_reasoning(
            session_id=self._session_id,
            phase=phase,
            plan=plan,
            uncertainty=uncertainty,
            data_trust=data_trust,
            workflow_gaps=workflow_gaps,
            want_to_verify=want_to_verify,
            raw_thought=raw_thought,
            context_snapshot=context_snapshot,
        )

        elapsed = (time.time() - start) * 1000

        return {
            "status": "recorded",
            "phase": phase,
            "prompt": self._get_reflection_prompt(phase),
            "_meta": {
                "tool": "think",
                "latency_ms": round(elapsed, 1),
            },
        }

    def session_replay(self, limit: int = 200) -> dict[str, Any]:
        """Retrieve the full reasoning + tool call chain for the current session.

        Returns chronologically ordered entries showing both tool calls
        and reasoning entries, so a user can see the agent's full
        decision-making process.
        """
        start = time.time()

        entries = self.storage.query_session_replay(
            session_id=self._session_id, limit=limit,
        )

        # Summarize
        reasoning_count = sum(1 for e in entries if e.get("entry_type") == "reasoning")
        tool_count = sum(1 for e in entries if e.get("entry_type") == "tool_call")

        # Extract all uncertainties and gaps across the session
        all_uncertainties = []
        all_gaps = []
        all_trust_issues = []
        for e in entries:
            if e.get("entry_type") == "reasoning":
                if e.get("uncertainty"):
                    all_uncertainties.extend(e["uncertainty"])
                if e.get("workflow_gaps"):
                    all_gaps.extend(e["workflow_gaps"])
                if e.get("data_trust"):
                    low_trust = [
                        d for d in e["data_trust"]
                        if d.get("trust_level") in ("low", "unknown")
                    ]
                    all_trust_issues.extend(low_trust)

        elapsed = (time.time() - start) * 1000

        return {
            "session_id": self._session_id,
            "total_entries": len(entries),
            "reasoning_entries": reasoning_count,
            "tool_call_entries": tool_count,
            "timeline": entries,
            "session_summary": {
                "unique_uncertainties": list(dict.fromkeys(all_uncertainties)),
                "unique_workflow_gaps": list(dict.fromkeys(all_gaps)),
                "low_trust_sources": all_trust_issues,
            },
            "_meta": {
                "tool": "session_replay",
                "latency_ms": round(elapsed, 1),
            },
        }

    @staticmethod
    def _get_reflection_prompt(phase: str) -> str:
        """Return a nudge for the LLM based on what phase it's in."""
        prompts = {
            "planning": (
                "Good. Now before calling analytics tools, think about: "
                "what data do you actually have? Is it fresh? Are there "
                "known gaps that will affect your analysis?"
            ),
            "pre_analysis": (
                "Before interpreting these results, consider: could the "
                "numbers be misleading? What would a skeptical analyst "
                "question about this data?"
            ),
            "post_analysis": (
                "You have results. Before presenting them, ask: does "
                "the confidence interval span both sides of 50%? Is "
                "this actually predictive or just noise? What would "
                "you want to verify?"
            ),
            "synthesis": (
                "You're combining multiple signals. Are they truly "
                "independent or are you double-counting the same "
                "underlying data? Where is the model most fragile?"
            ),
            "caveat_check": (
                "Final check: are you presenting uncertainty honestly? "
                "Would a sophisticated bettor find your caveats sufficient? "
                "What are you NOT telling the user?"
            ),
            "recommendation": (
                "If making a recommendation, is it justified by the "
                "confidence level? Would you bet your own money on this? "
                "What's the worst case if you're wrong?"
            ),
        }
        return prompts.get(phase, "Continue your analysis. Stay honest about what you don't know.")
