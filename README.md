# AMARIS — Autonomous Multi-Agent Research Intelligence System

Ask a research question. Five agents research it autonomously and return a
cited, self-scored report in roughly 60-90 seconds. Runs entirely on free
tiers — monthly cost $0.

The word that matters is **agentic**, not "multi-agent pipeline". A Supervisor
agent with an LLM brain decides what runs next at every step; the Critic can
send work back to the *Researcher*, not just the Writer. No routing is
hard-coded. See [CLAUDE.md](CLAUDE.md) for why that distinction is the whole
project.

> **Status: scaffolding complete (Phase 0).** Docs, tooling, package layout and
> the logging layer are in place. Application code lands phase by phase — see
> the table below.

## Quick start

```bash
uv sync                      # core stack + dev tools
cp .env.example .env         # PowerShell: Copy-Item .env.example .env
docker compose up -d         # redis + qdrant (local mode only)

uv run python -m amaris.observability   # logging smoke test — works today
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

uv only — never pip. `pyproject.toml` + `uv.lock` are the source of truth.
Heavy dependencies are opt-in extras (`--extra scraping`, `--extra memory`,
`--extra evaluation`, `--extra full`); every module degrades gracefully when an
extra is absent. Rationale in [ADR-009](docs/DECISIONS.md).

## Documentation

| Doc | What's in it |
|---|---|
| [CLAUDE.md](CLAUDE.md) | the contract — read first, before any code |
| [docs/PRD.md](docs/PRD.md) | problem, requirements, non-goals |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | layers, the one conditional edge, LLM strategy |
| [docs/AGENTS.md](docs/AGENTS.md) | every agent's exact system prompt |
| [docs/MEMORY.md](docs/MEMORY.md) | redis vs mem0 vs qdrant vs checkpoints |
| [docs/DESIGN.md](docs/DESIGN.md) | API surface, Streamlit UI, design tokens |
| [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md) | log event contract, replaying a run |
| [docs/DECISIONS.md](docs/DECISIONS.md) | ADRs — read before changing the stack |
| [docs/RULES.md](docs/RULES.md) | comments, logging, async, error and test conventions |
| [.claude/README.md](.claude/README.md) | Claude Code setup — hooks, skills, agents, permissions |

## Build phases

| # | Phase | Delivers | Status |
|---|---|---|---|
| 0 | Scaffolding | docs, uv project, package layout, logging | ✅ done |
| 1 | Foundation | settings.py, state.py, llm/router.py | ⬜ |
| 2 | Tools | search, scraper, code executor, vector, MCP | ⬜ |
| 3 | Memory | redis (with fallback), mem0 | ⬜ |
| 4 | Agents | supervisor + planner/researcher/analyst/writer/critic | ⬜ |
| 5 | Pipeline | nodes, edges, StateGraph, checkpointing | ⬜ |
| 6 | Evaluation | RAGAS scoring, non-blocking | ⬜ |
| 7 | API | FastAPI + WebSocket progress stream | ⬜ |
| 8 | Frontend | Streamlit, dual mode, custom CSS | ⬜ |
| 9 | Tests + deploy | full suite, README, Streamlit Cloud config | ⬜ |

Tier 1 hardening (4-provider fallback, structured-output resilience, startup
validation, readiness probes, PII guardrails, injection defense) follows Phase 9.

## Observability

Because an LLM chooses the execution path, two runs of the same query take
different routes — re-running is not a reproduction. Every run is therefore
keyed on a `session_id` that appears on every log line:

```
16:42:43.199 INFO  36333434:supervisor  supervisor.route  next_agent=researcher quality=0.0
16:42:43.199 DEBUG 36333434:researcher  researcher.react_step  task_id=t1 iteration=1
16:42:43.199 WARN  36333434:researcher  llm.fallback  provider=groq to=cerebras reason=429
```

The same records land in `logs/amaris.jsonl` as JSON, with errors mirrored to
`logs/errors.jsonl`. Filtering `supervisor.route` for one session prints the
exact agent path the system chose. Details in
[docs/OBSERVABILITY.md](docs/OBSERVABILITY.md).

## Tech stack

| Layer | Choice | Free tier |
|---|---|---|
| Orchestration | LangGraph + SqliteSaver | open source |
| LLM (primary) | Groq llama-3.3-70b / 3.1-8b | free, no card |
| LLM (fallbacks) | Cerebras, Google Gemini 2.0 Flash | free, no card |
| Search | DuckDuckGo (`ddgs`), Tavily optional | free / 1000 mo |
| Scraping | Crawl4AI | open source |
| Memory | mem0 + Qdrant | open source / 1GB cloud |
| Ops state | Redis, with in-memory fallback | open source |
| Evaluation | RAGAS on Groq | open source |
| API | FastAPI + WebSockets | — |
| UI | Streamlit + custom CSS | free public URL |
| Logs | loguru → console + JSONL | — |

## License

MIT — see [LICENSE](LICENSE).
