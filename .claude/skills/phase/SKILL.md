---
name: phase
description: Run one AMARIS build phase end to end — Tier 0 phases 1-9 (foundation, tools, memory, agents, pipeline, evaluation, api, frontend, tests) or Tier 1 hardening patches 1-6 (uv/fallback chain, structured output, config validation, health probes, guardrails, injection defense). Use when the user says "phase 3", "next phase", "start tier1 patch 2", or "continue the build".
---

# Run a build phase

## 1. Load only the phase you need

The briefs live in `references/tier0.md` (phases 1-9) and `references/tier1.md`
(patches 1-6). Do **not** read the whole file — pull out the one block:

```bash
# tier 0, phase 3
awk '/^## Phase 3 /,/^## Phase 4 /' .claude/skills/phase/references/tier0.md
# tier 1, patch 2
awk '/^## Patch 2 /,/^## Patch 3 /' .claude/skills/phase/references/tier1.md
```

If the user did not say which phase, run the one the SessionStart line reports
as next, and say which one you picked before starting.

## 2. Re-read the contracts that apply

`CLAUDE.md` is already in context. Read the specific doc the phase depends on,
nothing more:

| Phase | Read first |
|---|---|
| 1 Foundation | docs/ARCHITECTURE.md (LLM strategy), docs/OBSERVABILITY.md |
| 2 Tools | docs/RULES.md (async, security) |
| 3 Memory | docs/MEMORY.md — the store boundaries are the spec |
| 4 Agents | docs/AGENTS.md — **use those system prompts verbatim** |
| 5 Pipeline | docs/ARCHITECTURE.md (the one conditional edge) |
| 6 Evaluation | docs/DECISIONS.md ADR-017 (the layer split), docs/MEMORY.md (evaluator also writes to mem0) |
| 7 API | docs/DESIGN.md (routes, progress weighting) |
| 8 Frontend | docs/DESIGN.md (tokens, component contracts) |
| 9 Tests | docs/RULES.md (test behaviour, not implementation) |

## 3. Push back before building, if warranted

The brief is a plan written before any code existed — it can be wrong now.
Before implementing, check whether it names a library/model that's since been
renamed or retired, duplicates something already built, or would take a
simpler 90%-there path. Say so and propose the alternative before writing
code, per CLAUDE.md's Working agreement. Silent compliance with a stale brief
is not the goal; a correct Phase 6 is.

If the brief turns out to need more than ~3 new files or a change to working
code from an earlier phase, describe the plan in a few lines and get a nod
before writing it.

## 4. Build it

Non-negotiables, every phase:

- Type hints everywhere, `from __future__ import annotations` at the top
- Comments explain **why**, never what (docs/RULES.md has the examples)
- Anything importing an optional extra imports it lazily and degrades — a
  missing extra is a reduced run, never a crash (ADR-009)
- Log with the event-name contract: `logger.bind(k=v).info("noun.verb")`
- Node functions catch everything and set `state["error"]`; they never raise
- No `print`, no `eval`/`exec`, no sync I/O in async paths, no pip

Write the tests as part of the phase, not after it. Every new module gets a
matching test file under `tests/unit/` or `tests/agents/`.

## 5. Verify before claiming done

Run the phase's own verify command from the brief, then the standard gate:

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest
```

If a phase verify command needs API keys or Docker and they are not available,
say so plainly and report what you could not check — do not quietly skip it.

## 6. Close out

- Flip the phase row in the README build-phase table from ⬜ to ✅
- If the phase made a decision worth keeping, add an ADR to docs/DECISIONS.md
  (check the highest `## ADR-0NN` already there — Tier 1 patches reserve 011-016
  even before they're built, so Tier 0 phases 6-9 start at 017)
- Report what was built, what was verified, and what was **not** handled —
  scope deliberately left out, quota/API limits hit, anything that will need
  revisiting. Done-with-hidden-debt is worse than done-with-stated-caveats.
- **Do not commit.** Pankaj commits and pushes to GitHub himself — leave the
  work in the tree and say it is ready
- Stop and ask before starting the next phase
