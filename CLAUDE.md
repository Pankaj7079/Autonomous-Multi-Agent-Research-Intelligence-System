# AMARIS — Autonomous Multi-Agent Research Intelligence System

Read this fully before writing code. The patterns here are not suggestions.

---

## What this is

User asks a research question. Five agents autonomously research it, and the
system returns a structured report with real citations and quality scores in
roughly 60-90 seconds.

The important word is **agentic**, not "multi-agent pipeline". A Supervisor
agent with an LLM brain decides what runs next at every step. The Researcher
runs its own ReAct loop and decides when it has enough. The Critic can send
work back to the Researcher, not just the Writer. Nothing about the flow is
hard-coded — routing happens through LLM reasoning.

Everything runs on free tiers or open source. Monthly cost: $0.

---

## Toolchain — uv only, never pip

```bash
uv sync                      # core stack + dev group
uv sync --extra full         # + tavily, mcp, ragas, tracing
uv sync --extra scraping     # + crawl4ai (pulls playwright)
uv sync --extra memory       # + mem0 + sentence-transformers (pulls torch, ~2GB)
uv run python -m amaris.graph.pipeline --query "..."
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

`pyproject.toml` + `uv.lock` are the source of truth. There is no
requirements.txt and nothing installs with pip. Heavy dependencies are extras
on purpose — see ADR-009.

---

## Tooling this repo automates for you

`.claude/` is committed and active. Two things change how you should work:

- **Ruff runs itself.** A PostToolUse hook lints and formats every `.py` the
  moment it is written, and only reports what it could not autofix. Don't spend
  a turn running ruff after an edit — run the full gate (`/verify`) at the end.
- **Phase briefs live in a skill.** `/phase <n>` pulls one block out of
  `.claude/skills/phase/references/`. Never paste from
  `AMARIS_CLAUDE_CODE_MASTER.md` — it is 38 KB and the briefs there are stale
  on the uv migration.

Also available: `/verify` (gate + contract greps), `/trace <session_id>` (replay
a run from the JSONL log), and the `spec-auditor` / `dep-scout` subagents.
`pip` is blocked by a hook. `git commit` and `git push` prompt — Pankaj commits
to GitHub himself, so leave finished work in the tree and say it's ready.

---

## Two deployment modes (important architectural constraint)

**Local dev mode** — Docker available
  Redis + Qdrant via docker-compose, FastAPI backend, Streamlit frontend
  Full features: WebSocket streaming, persistent memory, checkpointing

**Cloud demo mode** — Streamlit Community Cloud, no Docker
  Streamlit runs the pipeline in-process, no FastAPI
  In-memory job state, Qdrant Cloud free tier (1GB), mem0 degrades gracefully

Every module must work in both modes. Detect via `settings.deployment_mode`.
Never assume Redis exists — always have an in-memory fallback path.

---

## Structure

```
amaris/
├── CLAUDE.md
├── pyproject.toml               deps + ruff + pytest config (uv is the pm)
├── uv.lock                      committed, reproducible installs
├── docker-compose.yml           redis + qdrant (+ langfuse optional)
├── .env.example
├── amaris_checkpoints.db        auto-created, gitignored
├── logs/                        rotating run logs, gitignored
│
├── docs/
│   ├── PRD.md                   what and why
│   ├── ARCHITECTURE.md          system design, agentic flow
│   ├── AGENTS.md                every agent's exact system prompt
│   ├── MEMORY.md                redis vs mem0 vs qdrant boundaries
│   ├── DESIGN.md                api + streamlit ui + css tokens
│   ├── OBSERVABILITY.md         logging contract, trace ids, log events
│   ├── DECISIONS.md             ADRs — read before changing the stack
│   └── RULES.md                 code conventions, comment style
│
├── amaris/
│   ├── config/settings.py       pydantic BaseSettings, lru_cache singleton
│   ├── observability/
│   │   ├── logging.py           loguru sink setup, JSONL file + pretty console
│   │   └── context.py           session_id / agent contextvars bound to logs
│   ├── llm/router.py            groq primary, gemini fallback, anthropic last resort
│   ├── agents/
│   │   ├── base_agent.py        abstract: llm setup, retry, structured output
│   │   ├── supervisor.py        THE AGENTIC CORE — llm decides routing
│   │   ├── planner.py           query → 3-5 concrete research tasks
│   │   ├── researcher.py        ReAct loop, self-terminates when satisfied
│   │   ├── analyst.py           conditional tool use, decides own approach
│   │   ├── writer.py            structured report + auto citations
│   │   └── critic.py            4-dim scoring + routing_hint for supervisor
│   ├── graph/
│   │   ├── state.py             GraphState TypedDict — the data contract
│   │   ├── nodes.py             thin wrappers, catch exceptions, never raise
│   │   ├── edges.py             supervisor routing only
│   │   └── pipeline.py          StateGraph + SqliteSaver
│   ├── tools/
│   │   ├── mcp_server.py        FastMCP exposing all tools
│   │   ├── search_tool.py       DuckDuckGo primary, Tavily optional
│   │   ├── scraper_tool.py      Crawl4AI, clean markdown
│   │   ├── code_executor.py     sandboxed subprocess, never eval()
│   │   └── vector_tool.py       Qdrant semantic search
│   ├── memory/
│   │   ├── redis_memory.py      job state + pubsub, with in-memory fallback
│   │   └── mem0_memory.py       episodic memory, local mode
│   ├── evaluation/ragas_eval.py non-blocking scoring
│   └── api/
│       ├── main.py              FastAPI + lifespan
│       ├── routes/research.py   POST /research, GET /{id}, WS /stream
│       ├── routes/health.py
│       └── schemas.py
│
├── frontend/
│   ├── app.py                   Streamlit, dual-mode (API or in-process)
│   ├── styles.py                CSS injection, design tokens
│   └── components.py            agent progress tracker, score cards
│
└── tests/
    ├── conftest.py              fixtures: sample_state, mock_llm
    ├── unit/                    fast, no external calls
    ├── agents/                  agent behavior with mocked LLMs
    └── integration/             needs docker-compose up
```

---

## The agentic core — read this before any agent code

```
Fixed pipeline (NOT what we build):
  planner → researcher → analyst → writer → critic → done
  every step always runs, edges are hard-coded

Agentic (what we build):
  START → supervisor
  supervisor thinks "no research yet"      → researcher
  supervisor thinks "quality 0.4, thin"    → researcher again
  supervisor thinks "quality 0.78, enough" → analyst
  supervisor thinks "analysis done"        → writer
  supervisor thinks "draft exists"         → critic
  critic sets routing_hint="need_more_research"
  supervisor reads hint                    → researcher (NOT writer)
  supervisor thinks "quality 0.81, done"   → FINISH → evaluator → END
```

Every one of those "thinks" is an LLM call in supervisor.py.
That single design choice is what makes this agentic rather than a chain.

Consequence: every agent returns to supervisor. No agent knows what comes next.

---

## Observability — built in, not bolted on

An unpredictable execution path is only debuggable if every run is traceable
from a single `session_id`. That is why logging is Phase 1, not Phase 9.

- `configure_logging()` runs once at process start — API lifespan, Streamlit
  boot, and every `__main__` entrypoint
- Two sinks: a coloured console sink for humans, and a JSONL file sink at
  `logs/amaris.jsonl` rotating at 10 MB / 7 days retention, for replaying a
  past run after the fact
- `bind_session(session_id)` / `bind_agent(name)` attach ids to every
  downstream log line through contextvars — no logger objects passed around
- Node wrappers log entry, exit, latency and the routing decision. A run's
  full agent path must be reconstructable from the log file alone.

Log line contract (full details in docs/OBSERVABILITY.md):

```python
logger.bind(next_agent=n, quality=q).info("supervisor.route")
```

Dotted event name as the message, structured values in `bind()`. Never bury a
value inside an f-string only — you lose the ability to filter on it later.

---

## Build phases — strictly in order

1. Foundation — settings.py, observability/{logging,context}.py, state.py, llm/router.py
2. Tools — search, scraper, code_executor, vector, mcp_server
3. Memory — redis_memory (with fallback), mem0_memory
4. Agents — base, supervisor, planner, researcher, analyst, writer, critic
5. Pipeline — nodes, edges, pipeline
6. Evaluation — ragas_eval
7. API — schemas, routes, main
8. Frontend — Streamlit + CSS
9. Tests + README + deploy

Do not start a phase until the previous one runs without errors.
At the end of each phase, stop and ask before continuing.

---

## GraphState fields that make it agentic

Beyond the obvious fields, these three carry the agentic behavior:

- `next_agent: str` — supervisor writes it, pipeline routes on it
- `research_quality: float` — researcher self-assesses 0-1, supervisor reads it
- `routing_hint: str` — critic writes "need_more_research" | "fix_writing" | "approve"

If an agent needs a new field, add it to GraphState first. Never pass loose
dicts between nodes.

---

## Hard rules

- Never eval() or exec() — subprocess with timeout=10, always
- Never synchronous redis in async code — use redis.asyncio
- Never raise from a node function — catch, set state["error"], return
- Never hard-code routing outside supervisor
- Never print() — use loguru
- Never assume Redis is running — check settings.deployment_mode
- Never commit .env or amaris_checkpoints.db
- Never pip — `uv add`, `uv sync`, `uv run`
- Type hints on every function, no exceptions

---

## Comment style

One line, plain English, says *why*. Never a multi-line comment block — if the
reason needs more than a line it belongs in a docstring or in docs/.

Good:

```python
# retry 3x: groq sometimes returns empty content on the first call
# gather, not a loop — a 4-task plan went from 40s serial to 11s
# escape braces or loguru re-reads a dict value as a format placeholder
```

Bad:

```python
# Retry up to 3 times          <- says what, not why
# Initialize the variable      <- says nothing
```

Docstrings: one line for most functions. Say what it returns and how it fails.

---

## Verify a phase is done

```bash
# Phase 1
uv run python -c "from amaris.llm.router import get_llm; print(get_llm('planning'))"

# Phase 3
docker compose up -d && uv run python -m amaris.memory.redis_memory --selftest

# Phase 5
uv run python -m amaris.graph.pipeline --query "What is LangGraph?"

# Phase 7
curl -X POST localhost:8000/research -H "Content-Type: application/json" \
     -d '{"query":"What is the MCP protocol?"}'

# Phase 9
uv run pytest tests/ --cov=amaris --cov-report=term-missing
```
