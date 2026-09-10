"""Replay one AMARIS run from logs/amaris.jsonl, filtered small enough to read."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

LOG = Path(__file__).resolve().parents[4] / "logs" / "amaris.jsonl"

LEVELS = {"DEBUG": 10, "INFO": 20, "SUCCESS": 25, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


def read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        sys.exit(f"no log file at {path} — run something first")
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            records.append(json.loads(line)["record"])
        except (json.JSONDecodeError, KeyError):
            continue  # half-written line at the tail is normal
    return records


def fields(record: dict[str, Any]) -> str:
    extra = {k: v for k, v in record["extra"].items() if k not in ("session_id", "agent")}
    return " ".join(f"{k}={v}" for k, v in extra.items())


def stamp(record: dict[str, Any]) -> str:
    return record["time"]["repr"][11:23]


def list_sessions(records: list[dict[str, Any]]) -> None:
    seen: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "query": "", "last": ""})
    for r in records:
        sid = r["extra"].get("session_id", "-")
        if sid == "-":
            continue
        seen[sid]["n"] += 1
        seen[sid]["last"] = stamp(r)
        if r["message"] == "run.start":
            seen[sid]["query"] = str(r["extra"].get("query", ""))[:60]
    for sid, info in list(seen.items())[-15:]:
        print(f"{sid}  {info['last']}  {info['n']:>4} records  {info['query']}")


def agent_path(records: list[dict[str, Any]]) -> None:
    hops = [r for r in records if r["message"] == "supervisor.route"]
    if not hops:
        print("no supervisor.route records — the pipeline never routed")
        return
    print(" → ".join(str(r["extra"].get("next_agent", "?")) for r in hops))
    print()
    for r in hops:
        print(f"  {stamp(r)}  {fields(r)}")


def timeline(records: list[dict[str, Any]], min_level: str) -> None:
    floor = LEVELS.get(min_level.upper(), 0)
    for r in records:
        if r["level"]["no"] < floor:
            continue
        print(
            f"{stamp(r)} {r['level']['name']:<7} "
            f"{r['extra'].get('agent', '-'):<11} {r['message']:<28} {fields(r)}"
        )


def errors(records: list[dict[str, Any]]) -> None:
    found = [r for r in records if r["level"]["no"] >= LEVELS["ERROR"]]
    if not found:
        print("no errors in this session")
        return
    for r in found:
        print(f"{stamp(r)} {r['extra'].get('agent', '-')} {r['message']} {fields(r)}")
        if r.get("exception"):
            print(f"  {r['exception']}")


def slow(records: list[dict[str, Any]]) -> None:
    timed = [r for r in records if "ms" in r["extra"]]
    for r in sorted(timed, key=lambda x: float(x["extra"]["ms"]), reverse=True)[:15]:
        print(f"{float(r['extra']['ms']):>10.1f} ms  {r['message']:<30} {fields(r)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="replay an AMARIS run from the JSONL log")
    parser.add_argument("session_id", nargs="?", help="8-char run id; omit with --list")
    parser.add_argument("--list", action="store_true", help="recent sessions in the log")
    parser.add_argument("--path", action="store_true", help="supervisor routing decisions only")
    parser.add_argument("--errors", action="store_true", help="errors with tracebacks")
    parser.add_argument("--slow", action="store_true", help="slowest timed operations")
    parser.add_argument("--level", default="DEBUG", help="minimum level for the timeline")
    parser.add_argument("--log", default=str(LOG), help="path to amaris.jsonl")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    records = read(Path(args.log))

    if args.list or not args.session_id:
        list_sessions(records)
        return

    scoped = [r for r in records if r["extra"].get("session_id") == args.session_id]
    if not scoped:
        sys.exit(f"no records for session {args.session_id} — try --list")

    if args.path:
        agent_path(scoped)
    elif args.errors:
        errors(scoped)
    elif args.slow:
        slow(scoped)
    else:
        timeline(scoped, args.level)


if __name__ == "__main__":
    main()
