# `.claude/` — Claude Code setup for AMARIS

Committed on purpose: this is project configuration, not personal preference.
Anything personal goes in `.claude/settings.local.json`, which is gitignored.

```
.claude/
├── settings.json          permissions, env, hook registration
├── hooks/
│   ├── ruff_on_save.py    PostToolUse  — lint+format every .py the moment it's written
│   ├── guard_bash.py      PreToolUse   — blocks pip and other uv-only violations
│   └── session_context.py SessionStart — one line: next phase, branch, dirty count
├── skills/
│   ├── phase/             run one build phase (Tier 0 1-9, Tier 1 patches 1-6)
│   ├── verify/            the quality gate + contract greps
│   └── trace/             replay a run from logs/amaris.jsonl
└── agents/
    ├── spec-auditor.md    read-only audit of code against the docs
    └── dep-scout.md       verify package names / model ids before pinning
```

## Why each piece exists

**Hooks do the mechanical work.** The house rule is "linter passes before every
commit". Enforcing that from the model costs a tool call and a re-read on every
edit; `ruff_on_save.py` does it for free and only speaks up about things ruff
cannot autofix. `guard_bash.py` blocks pip because installing outside uv leaves
the environment and `uv.lock` silently disagreeing — nothing fails until someone
else clones the repo. `session_context.py` prints one line so a new session does
not spend four tool calls working out which phase is next.

**Skills keep the phase briefs out of the context window.** The original
`AMARIS_CLAUDE_CODE_MASTER.md` is 38 KB. `skills/phase/` holds the same briefs
split so that only the requested phase gets read — the skill's own instructions
tell Claude to `awk` out one block rather than load the file. Invoke with
`/phase 3`, `/verify`, `/trace <session_id>`.

**Agents are for jobs with a different failure mode than the main loop.**
`spec-auditor` needs to be read-only and adversarial about drift, which is
awkward to do in the same session that wrote the code. `dep-scout` needs the
web and must never guess a version. Both are opt-in — Claude will not spawn
them unless asked.

## Token behaviour

| Lever | Effect |
|---|---|
| Phase briefs in a skill, not pasted | the 38 KB master doc is never loaded whole |
| `awk` one phase block | ~300 tokens instead of ~4,000 |
| SessionStart line | ~40 tokens replaces several orientation tool calls |
| `deny: Read(uv.lock, .venv/**, logs/**)` | stops a 400 KB lockfile or log landing in context; grep and `/trace` still reach them |
| Hook autofix | no round trip for lint errors ruff can fix itself |

`CLAUDE.md` is read every session and is the one file worth keeping short —
detailed specs belong in `docs/`, which is read on demand.

## Permissions

`settings.json` allows the routine read-only and uv commands outright, and puts
`git commit`, `git push`, `git add` and `gh pr create` behind a prompt —
**Pankaj commits to GitHub manually**. It denies pip, `git reset --hard`,
`git clean -fd`, and reading `.env`.

## Changing this setup

Hook scripts are ordinary Python and take their payload as JSON on stdin. Test
one without a full session:

```bash
printf '{"tool_name":"Bash","tool_input":{"command":"uv add x"}}' | uv run --no-sync python .claude/hooks/guard_bash.py
echo "exit $?"    # 0 allows, 2 blocks and returns stderr to Claude
```

They are linted like the rest of the repo (`.claude/hooks/*.py` has its own
`per-file-ignores` entry in `pyproject.toml` because shelling out is their job).
