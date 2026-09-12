"""The five research agents plus the supervisor — see docs/AGENTS.md."""

from __future__ import annotations

from amaris.agents.analyst import AnalystAgent
from amaris.agents.base_agent import AgentError, BaseAgent
from amaris.agents.critic import CriticAgent
from amaris.agents.planner import PlannerAgent
from amaris.agents.researcher import ResearcherAgent
from amaris.agents.supervisor import SupervisorAgent
from amaris.agents.writer import WriterAgent

# the graph builds nodes from this, so a new agent only needs adding here
AGENT_CLASSES = {
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
    "WriterAgent",
]
