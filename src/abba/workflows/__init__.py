"""ABBA Workflows -- multi-step analytical pipelines.

Each workflow chains multiple tool calls into a coherent analysis.
Agents call run_workflow(name, **params) to execute a full pipeline
that would otherwise require 5-10 individual tool calls.

Workflows are the structured answer to questions like:
- "Who wins tonight's Rangers game?"
- "How are the Bruins doing this season?"
- "Any good NHL bets tonight?"
- "Can the Panthers afford a trade deadline deal?"
- "Tell me the story of the Avalanche's season"
"""

from .engine import WorkflowEngine, run_workflow, list_workflows

__all__ = ["WorkflowEngine", "run_workflow", "list_workflows"]
