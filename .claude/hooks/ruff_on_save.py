"""PostToolUse hook: ruff check --fix + format on every .py write. Exit 2 = tell Claude."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

RUFF = ["uv", "run", "--no-sync", "ruff"]


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # bad payload is the harness's problem, don't nag about it

    file_path = (payload.get("tool_input") or {}).get("file_path")
    if not file_path or not file_path.endswith(".py"):
        return 0

    target = Path(file_path)
    if not target.exists():
        return 0  # deleted or renamed, nothing to lint

    _run([*RUFF, "check", "--fix", "--quiet", str(target)])
    _run([*RUFF, "format", "--quiet", str(target)])

    remaining = _run([*RUFF, "check", "--quiet", str(target)])
    if remaining.returncode != 0:
        sys.stderr.write(
            "ruff found issues it cannot autofix in "
            f"{target.name} — fix them now, don't leave them for the commit:\n"
            f"{remaining.stdout}{remaining.stderr}"
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
