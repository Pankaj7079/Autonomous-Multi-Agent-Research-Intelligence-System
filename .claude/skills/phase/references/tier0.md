# Tier 0 — build phases 1-9

Extract one block with awk; do not read the whole file.

## Phase 1 — Foundation

Build `amaris/config/settings.py`, `amaris/graph/state.py`, `amaris/llm/router.py`.
(`amaris/observability/` already exists from scaffolding — wire settings into it,
don't rewrite it.)

- **settings.py** — pydantic-settings `BaseSettings`, `lru_cache` singleton
  `get_settings()`, one field per var in `.env.example`, sane defaults so the
  import never fails without a `.env`. `deployment_mode: Literal["local","cloud"]`.
- **state.py** — the full `GraphState` TypedDict. Must include `next_agent`,
  `research_quality`, `routing_hint` — those three carry the agentic behaviour.
  Also a `new_state(query, session_id)` factory so no caller hand-builds a dict.
- **router.py** — `get_llm(task_type)` returns the primary model for that task
  (reasoning → groq 70b, fast → groq 8b), `get_fallback_llm()`, and
  `invoke_with_fallback()` catching 429 / rate limit / timeout and switching to
  Gemini. Log every hop: `logger.bind(provider=..., to=..., reason=...).warning("llm.fallback")`.
  Tier 1 patch 1 extends this to a four-provider chain — leave room for it.

Verify: `uv run python -c "from amaris.llm.router import get_llm; print(get_llm('planning'))"`

## Phase 2 — Tools

Build `amaris/tools/`: search_tool, scraper_tool, code_executor, vector_tool, mcp_server.

- **search_tool** — DuckDuckGo via `ddgs` (the package rename, ADR-006) wrapped in
  `run_in_executor` because it is sync. Optional Tavily when the key exists.
  `smart_search()` merges both and dedupes by URL. Returns `[]` on total failure.
- **scraper_tool** — Crawl4AI, `scrape_url()` with a 15s timeout and an 8000 char
  cap. Lives behind the `scraping` extra: import lazily, fall back to the search
  snippet when crawl4ai is absent.
- **code_executor** — `subprocess` with `timeout=10` and a forbidden-import check.
  Never `eval`/`exec`. Returns stdout, stderr and a success flag.
- **vector_tool** — Qdrant search that works with both a local host/port and a
  cloud URL + API key.
- **mcp_server** — FastMCP exposing all four, plus `get_tools_for_agent(name)`.
  Behind the `mcp` extra.

Test each tool standalone before wiring anything.

## Phase 3 — Memory

Build `amaris/memory/redis_memory.py` and `mem0_memory.py`. Read docs/MEMORY.md first —
the store boundaries there are the spec.

- **redis_memory** — `redis.asyncio`, job status hash with 24h TTL, pubsub channel
  for progress. An `InMemoryJobStore` with the identical interface, chosen by
  `get_job_store()` on `deployment_mode` — and also used as the fallback when a
  local Redis connection fails at startup. A demo must not die because Docker is down.
- **mem0_memory** — local mode, HuggingFace MiniLM embedder, Qdrant backend, groq 8b
  for mem0's own calls. Surface is exactly `recall_related()` and
  `add_research_finding()`. Behind the `memory` extra: every method becomes a logged
  no-op when the import fails.

Add a `--selftest` entrypoint to each.
Verify: `docker compose up -d && uv run python -m amaris.memory.redis_memory --selftest`

## Phase 4 — Agents

Read docs/AGENTS.md fully first — **use those system prompts verbatim**.
Build `amaris/agents/`: base_agent, supervisor, planner, researcher, analyst, writer, critic.

- **base_agent** — abstract; LLM from the router, `_invoke()` with 3 retries and
  exponential backoff, structured-output helper, `agent_context()` around the run.
- **supervisor** — the LLM router. Outputs one word into `state["next_agent"]`.
  Validate the word against the allowed set; anything unexpected → FINISH.
- **researcher** — ReAct loop, max `MAX_REACT_ITERATIONS` per task,
  `asyncio.gather` across tasks, dedupe by URL, self-assess `research_quality`.
- **critic** — four dimension scores plus the `routing_hint` that lets a quality
  failure route back to research.

Test each agent alone with a mocked LLM. Behaviour, not prompt text.

## Phase 5 — Pipeline

Build `amaris/graph/nodes.py`, `edges.py`, `pipeline.py`.

- **nodes** — thin try/except wrappers, never raise, set `state["error"]`. Wrap each
  body in `agent_context(name)` and `timed("node.<name>")`.
- **edges** — only the supervisor conditional edge; every agent edges back to supervisor.
- **pipeline** — `StateGraph`, entry point supervisor, `SqliteSaver` checkpointer,
  `get_pipeline()` singleton, `thread_id = session_id`.

Add a `__main__` block. Log `run.start` / `run.complete` with the full agent path.
Verify: `uv run python -m amaris.graph.pipeline --query "What is LangGraph?"`

## Phase 6 — Layered Evaluation (ADR-017, replaces the original RAGAS-only plan)

RAGAS evaluates retrieve-then-generate; most of what determines AMARIS's output
quality happens in supervisor routing and the ReAct/revision loops, which RAGAS
never sees. Three layers instead of one metric:

- **Layer 1** `retrieval_eval.py` — RAGAS `context_precision` (+ `context_recall`
  when a golden query gives a `reference_answer`), scoped to sources vs the
  query only, never the report. Configure RAGAS to use Groq, never OpenAI.
- **Layer 2** `report_eval.py` — RAGAS `Faithfulness` + `ResponseRelevancy` on
  the final report vs its sources and vs the query. (Not DeepEval: its
  `ToolCorrectnessMetric` assumes one correct tool per intent, which fights
  the ReAct loop's deliberately adaptive tool choice — see ADR-017.)
- **Layer 3** `trajectory_eval.py` — custom, zero LLM calls, scores the
  *path*: `routing_accuracy`, `research_convergence`, `loop_efficiency`,
  `termination_quality`, `react_discipline`. Needs `decision_log` (supervisor)
  and `react_stats` (researcher) added to `GraphState`. Runs from the harness
  only, never inline on a real session.

Plus `base.py` (`EvalResult`/`EvalReport`/`Evaluator` protocol), `golden_set.py`
(loads `tests/golden/queries.yaml`), and `harness.py` (`run_evaluation`,
`compare_reports`, a CLI). Every evaluator degrades to a zero `EvalResult`
instead of raising. Layers 1+2 run inline in the evaluator node (cheap); Layer
3 runs only from `amaris.evaluation.harness`, off the request path.

## Phase 7 — API

Build `amaris/api/`: schemas.py, routes/health.py, routes/research.py, main.py.
Read docs/DESIGN.md for the route contracts and the progress weighting.

`POST /research` → job_id, pipeline as a background task. `GET /research/{job_id}` →
status, current_agent, progress_pct, result. `WS /research/{job_id}/stream` → live
progress off the Redis pubsub channel. Stream through `pipeline.astream()` and publish
each node transition. `configure_logging()` in the lifespan.

Verify with curl.

## Phase 8 — Streamlit frontend

Build `frontend/`: app.py, styles.py, components.py. Design tokens and component
contracts are in docs/DESIGN.md — use them, don't invent a second palette.

- **styles.py** — CSS injection, dark theme, variables for every colour. It must not
  look like default Streamlit.
- **components.py** — `agent_progress_tracker()`, `score_dashboard()`, `citation_list()`.
  Components render, they never fetch.
- **app.py** — dual mode: calls the API in local mode, imports and runs the pipeline
  in-process in cloud mode. Live progress, score cards, session_id copy box,
  expandable citations.

## Phase 9 — Tests, README, deploy

`tests/conftest.py` gains `sample_state` and a mocked LLM fixture (the logging
isolation fixture is already there).

Required tests:
- supervisor routes to researcher when `research_quality` < 0.60
- supervisor routes to researcher when `routing_hint == "need_more_research"`
- supervisor FINISHes at `revision_count >= 2`
- researcher ReAct loop terminates on `sufficient=true`
- researcher never exceeds `MAX_REACT_ITERATIONS`
- nodes return error state instead of raising

Then expand README (architecture diagram, agentic-vs-pipeline explanation, demo GIF
placeholder) and add `.streamlit/config.toml` for Community Cloud.

Verify: `uv run pytest tests/ --cov=amaris --cov-report=term-missing`
