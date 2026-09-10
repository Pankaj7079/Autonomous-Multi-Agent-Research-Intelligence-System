"""Tool access is a permission boundary — an agent must only get what its job needs."""

from __future__ import annotations

import pytest

from amaris.tools.mcp_server import AGENT_TOOLS, TOOL_REGISTRY, get_tools_for_agent


def test_researcher_gets_search_and_scrape_only() -> None:
    assert set(get_tools_for_agent("researcher")) == {"web_search", "scrape_webpage"}


def test_analyst_gets_code_and_knowledge_base_only() -> None:
    assert set(get_tools_for_agent("analyst")) == {"execute_python", "search_knowledge_base"}


def test_researcher_cannot_execute_code() -> None:
    """Code execution is the sharpest tool here — only the analyst reaches it."""
    assert "execute_python" not in get_tools_for_agent("researcher")


@pytest.mark.parametrize("agent", ["planner", "writer", "critic", "supervisor"])
def test_reasoning_only_agents_get_no_tools(agent: str) -> None:
    assert get_tools_for_agent(agent) == {}


def test_unknown_agent_gets_nothing_rather_than_everything() -> None:
    assert get_tools_for_agent("definitely_not_an_agent") == {}


def test_every_mapped_tool_actually_exists() -> None:
    """A typo in AGENT_TOOLS would only surface mid-run without this."""
    mapped = {name for names in AGENT_TOOLS.values() for name in names}
    assert mapped <= set(TOOL_REGISTRY)


def test_registry_exposes_all_four_tools() -> None:
    assert set(TOOL_REGISTRY) == {
        "web_search",
        "scrape_webpage",
        "execute_python",
        "search_knowledge_base",
    }
