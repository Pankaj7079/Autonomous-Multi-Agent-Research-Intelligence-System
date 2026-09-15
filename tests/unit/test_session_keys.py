"""A visitor's own API key must reach their own run and no one else's."""

from __future__ import annotations

import asyncio
import threading

from amaris.config.settings import get_settings, session_keys, use_session_keys

FIELD = "anthropic_api_key"
FAKE = "sk-ant-not-a-real-key"


def test_a_supplied_key_is_what_the_accessor_returns() -> None:
    settings = get_settings()
    use_session_keys({FIELD: FAKE})
    try:
        assert settings.key(FIELD) == FAKE
        assert "anthropic" in settings.configured_llm_providers
    finally:
        use_session_keys({})


def test_clearing_falls_back_to_the_configured_key() -> None:
    settings = get_settings()
    use_session_keys({FIELD: FAKE})
    use_session_keys({})
    assert settings.key(FIELD) != FAKE
    assert session_keys() == {}


def test_a_blank_entry_is_not_treated_as_a_key() -> None:
    """An empty text_input must not park an unusable provider at the head of the chain."""
    settings = get_settings()
    use_session_keys({FIELD: "   "})
    try:
        assert settings.key(FIELD) != "   "
        assert session_keys() == {}
    finally:
        use_session_keys({})


def test_one_visitors_key_never_reaches_another_visitors_run() -> None:
    """The whole reason this is a contextvar: one Streamlit process serves every visitor, and
    both os.environ and the settings singleton would hand this key to the next person."""
    settings = get_settings()
    seen: dict[str, str | None] = {}

    def other_visitor() -> None:
        # a separate thread is a separate context, exactly like a second streamlit session
        seen["value"] = settings.key(FIELD)

    use_session_keys({FIELD: FAKE})
    try:
        thread = threading.Thread(target=other_visitor)
        thread.start()
        thread.join()
        assert settings.key(FIELD) == FAKE
        assert seen["value"] != FAKE
    finally:
        use_session_keys({})


def test_the_key_survives_into_the_coroutines_a_run_spawns() -> None:
    """The pipeline runs under asyncio.run inside the session's own thread, so the binding has
    to reach every task it creates or the agents fall back to the shared key mid-run."""
    settings = get_settings()

    async def deep_inside() -> str | None:
        await asyncio.sleep(0)
        return await asyncio.create_task(asyncio.to_thread(settings.key, FIELD))

    async def nested() -> str | None:
        return await asyncio.create_task(deep_inside())

    use_session_keys({FIELD: FAKE})
    try:
        assert asyncio.run(nested()) == FAKE
    finally:
        use_session_keys({})


def test_a_callers_key_never_reaches_graph_state() -> None:
    """GraphState is checkpointed to sqlite and replayed by /trace. A key in there would be
    written to disk and printed by a replay, which is the opposite of session-scoped."""
    from amaris.api.schemas import ResearchRequest

    request = ResearchRequest(query="what is MCP?", api_keys={FIELD: FAKE})
    assert FAKE not in str(request.seed())
    # excluded from the dump too, so it cannot ride into a log line or the raw inspector
    assert "api_keys" not in request.model_dump()
    assert FAKE not in str(request.model_dump())


def test_the_api_binds_a_jobs_key_inside_the_job() -> None:
    """Bound in the request handler instead, the value would leak into every job the same
    event loop ran afterwards, because they would share that context."""
    from amaris.api.routes import research

    settings = get_settings()
    seen: dict[str, str | None] = {}

    async def drive() -> None:
        async def one_job() -> None:
            use_session_keys({FIELD: FAKE})
            seen["inside"] = settings.key(FIELD)

        async def later_job() -> None:
            seen["after"] = settings.key(FIELD)

        await asyncio.create_task(one_job())
        await asyncio.create_task(later_job())

    asyncio.run(drive())
    assert seen["inside"] == FAKE
    assert seen["after"] != FAKE
    # and the route really does pass them down rather than dropping them on the floor
    assert "api_keys" in research._run_job.__code__.co_varnames
