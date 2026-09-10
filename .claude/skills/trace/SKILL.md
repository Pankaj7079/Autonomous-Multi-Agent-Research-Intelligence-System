---
name: trace
description: Replay an AMARIS run from logs/amaris.jsonl — the agent path the supervisor chose, per-node latency, provider fallbacks, and where it failed. Use when a run behaved oddly, took too long, produced a bad report, or when the user gives a session_id and asks what happened.
---

# Trace a run

Because an LLM picks the route, re-running a query does not reproduce a run.
The log is the reproduction. Every record carries `session_id` and `agent`.

Reading whole log files into context is denied on purpose — filter with the
commands below and read only what comes back.

## 1. Find the session

If the user did not give a session_id, list the recent runs:

```bash
uv run python .claude/skills/trace/scripts/trace.py --list
```

## 2. The agent path — start here

```bash
uv run python .claude/skills/trace/scripts/trace.py <session_id> --path
```

Prints the exact sequence of supervisor decisions, e.g.
`planner → researcher → researcher → analyst → writer → critic → writer → FINISH`.

What to look for:
- **researcher twice in a row** — first pass scored under the quality threshold
- **critic → researcher** — `routing_hint="need_more_research"` fired. This is
  the behaviour the whole architecture exists for; it is not a bug
- **FINISH straight after an error** — a node failed, look at step 4
- **writer → critic → writer → critic** — revision loop; check it stopped at
  `MAX_REVISIONS`

## 3. The full timeline

```bash
uv run python .claude/skills/trace/scripts/trace.py <session_id>
uv run python .claude/skills/trace/scripts/trace.py <session_id> --level WARNING
```

## 4. Failures and slow nodes

```bash
uv run python .claude/skills/trace/scripts/trace.py <session_id> --errors
uv run python .claude/skills/trace/scripts/trace.py <session_id> --slow
```

`--slow` sorts `node.*.complete` records by `ms`. A researcher node far slower
than the rest usually means the ReAct loop ran its full iteration budget
without setting `sufficient=true`.

## 5. Explain it

Report in this order: the path, where it diverged from the expected happy path,
the evidence (quoted log lines with their fields), then the likely cause. Tie
the cause to a specific file and line where you can. If the logs do not support
a conclusion, say that rather than inventing one.
