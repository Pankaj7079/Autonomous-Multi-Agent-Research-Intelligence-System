"""PreToolUse hook: block pip and other uv-only violations. Exit 2 = blocked."""

from __future__ import annotations

import json
import re
import sys

# kept short on purpose — a guard that blocks real work just gets switched off
BLOCKED: list[tuple[str, str]] = [
    (
        r"(?<!uv )\bpip3?\s+install\b",
        "This project is uv-only (CLAUDE.md, ADR-009). Use `uv add <pkg>` to add a "
        "dependency, or `uv sync` to install what uv.lock already pins.",
    ),
    (
        r"\bpython\s+-m\s+pip\b",
        "This project is uv-only (CLAUDE.md, ADR-009). Use `uv add <pkg>` / `uv sync`.",
    ),
    (
        r"\bpython\s+-m\s+(pytest|ruff)\b",
        "Run project tools through the locked environment: `uv run pytest`, `uv run ruff`.",
    ),
]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    if payload.get("tool_name") != "Bash":
        return 0

    command = (payload.get("tool_input") or {}).get("command", "")
    for pattern, guidance in BLOCKED:
        if re.search(pattern, command):
            sys.stderr.write(f"Blocked by .claude/hooks/guard_bash.py — {guidance}")
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
