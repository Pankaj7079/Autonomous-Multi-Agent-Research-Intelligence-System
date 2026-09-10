"""SessionStart hook: one line saying which phase is next and if the tree is dirty."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def next_phase() -> str | None:
    """First unchecked row of the README build-phase table, as 'N Name — deliverables'."""
    readme = REPO / "README.md"
    if not readme.exists():
        return None
    for line in readme.read_text(encoding="utf-8").splitlines():
        if "⬜" not in line:  # empty checkbox = phase not done yet
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3:
            return f"{cells[0]} {cells[1]} ({cells[2]})"
    return None


def git_state() -> str | None:
    try:
        branch = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.splitlines()
    except (subprocess.SubprocessError, OSError):
        return None
    suffix = f", {len(dirty)} uncommitted" if dirty else ", clean"
    return f"branch {branch}{suffix}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parts = [p for p in (next_phase(), git_state()) if p]
    if not parts:
        return 0

    phase = parts[0]
    rest = parts[1:]
    line = f"AMARIS · next phase: {phase}"
    if rest:
        line += " · " + " · ".join(rest)
    sys.stdout.write(re.sub(r"\s+", " ", line) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
