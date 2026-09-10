---
name: verify
description: Run the AMARIS quality gate — ruff check, ruff format check, pytest, and optionally the live phase verification commands. Use before declaring a phase done, before a commit, or when the user asks "is it clean", "run the checks", "verify".
---

# Verify

## The gate — always run this

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest
```

All three must pass. If `ruff format --check` fails, run `uv run ruff format .`
and say what it reformatted — do not hand-edit whitespace.

## Live checks — only what the current phase enables

Each needs something the earlier phases built, so skip the ones that do not
exist yet rather than reporting them as failures.

```bash
uv run python -m amaris.observability                      # phase 0, always works
uv run python -c "from amaris.llm.router import get_llm; print(get_llm('planning'))"   # phase 1
uv run python -m amaris.memory.redis_memory --selftest     # phase 3, needs docker
uv run python -m amaris.graph.pipeline --query "What is LangGraph?"                    # phase 5, needs keys
curl -s -X POST localhost:8000/research -H "Content-Type: application/json" -d '{"query":"What is MCP?"}'   # phase 7
```

Coverage, when the user asks for it:

```bash
uv run pytest --cov=amaris --cov-report=term-missing
```

## Contract checks the linter cannot see

Grep for the hard rules that have bitten this kind of system before:

```bash
grep -rn "print(" amaris/ frontend/ --include=*.py     # must be zero — loguru only
grep -rn "eval(\|exec(" amaris/ --include=*.py         # must be zero
grep -rn "time.sleep\|requests.get" amaris/ --include=*.py   # sync I/O in async paths
grep -rn "^import mem0\|^import crawl4ai\|^import ragas" amaris/ --include=*.py
                                                       # extras must be imported lazily (ADR-009)
```

A node function that can raise is the other one worth eyeballing — every
function in `amaris/graph/nodes.py` should have a `try` with a bare
`except Exception` that returns `{"error": ...}`.

## Reporting

Say what passed, what failed with the actual output, and what you could not
check because a dependency or key was missing. Never report a skipped check as
a pass. Do not commit — Pankaj commits himself.
