"""Triage, the five research agents, and the supervisor — see docs/AGENTS.md."""

from __future__ import annotations

from amaris.agents.analyst import AnalystAgent
from amaris.agents.base_agent import AgentError, BaseAgent
from amaris.agents.critic import CriticAgent
from amaris.agents.planner import PlannerAgent
from amaris.agents.researcher import ResearcherAgent
from amaris.agents.supervisor import SupervisorAgent
from amaris.agents.triage import TriageAgent
from amaris.agents.writer import WriterAgent

# the graph builds nodes from this, so a new agent only needs adding here
AGENT_CLASSES = {
    "triage": TriageAgent,
    "supervisor": SupervisorAgent,
    "planner": PlannerAgent,
    "researcher": ResearcherAgent,
    "analyst": AnalystAgent,
    "writer": WriterAgent,
    "critic": CriticAgent,
}

__all__ = [
    "AGENT_CLASSES",
    "AgentError",
    "AnalystAgent",
    "BaseAgent",
    "CriticAgent",
    "PlannerAgent",
    "ResearcherAgent",
    "SupervisorAgent",
    "TriageAgent",
    "WriterAgent",
]
