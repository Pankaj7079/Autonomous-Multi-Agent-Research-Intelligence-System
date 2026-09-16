"""Runs the golden set through the full live pipeline and scores every layer. Offline, not per-request.

This is the only place Layer 3 (trajectory) ever runs — it needs a whole finished decision_log,
which a real user session has no reason to wait around for.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from amaris.evaluation.base import EvalReport, EvalResult
from amaris.evaluation.golden_set import GoldenQuery, load_golden_set
from amaris.evaluation.report_eval import ReportEvaluator
from amaris.evaluation.retrieval_eval import RetrievalEvaluator
from amaris.evaluation.trajectory_eval import TrajectoryEvaluator
from amaris.graph.state import source_count
from amaris.observability.logging import logger

if TYPE_CHECKING:
    from amaris.graph.state import GraphState

REGRESSION_THRESHOLD = 0.05
EVALS_DIR = Path("evals")


class PipelineRunner(Protocol):
    """What run_research looks like — a Protocol so tests can inject a fake without a real run."""

    async def __call__(self, query: str) -> GraphState: ...


def _golden_checks(golden: GoldenQuery, state: GraphState) -> list[EvalResult]:
    """The golden query's own expectations — not a scoring layer, just pass/fail assertions."""
    layer = "golden_set"
    sources = source_count(state)
    results = [
        EvalResult(
            layer=layer,
            metric="min_sources",
            score=1.0 if sources >= golden.expected_min_sources else 0.0,
            passed=sources >= golden.expected_min_sources,
            detail=f"[{golden.id}] {sources} sources, expected >= {golden.expected_min_sources}",
        )
    ]

    involved = set(state["agent_path"])
    missing = set(golden.expected_agents_involved) - involved
    results.append(
        EvalResult(
            layer=layer,
            metric="agents_involved",
            score=1.0 if not missing else 0.0,
            passed=not missing,
            detail=f"[{golden.id}] missing {missing}" if missing else f"[{golden.id}] all present",
        )
    )

    forbidden = involved & set(golden.forbidden_agents)
    if golden.forbidden_agents:
        results.append(
            EvalResult(
                layer=layer,
                metric="agents_skipped",
                score=0.0 if forbidden else 1.0,
                passed=not forbidden,
                detail=f"[{golden.id}] ran {forbidden}, which this question should not need"
                if forbidden
                else f"[{golden.id}] skipped {set(golden.forbidden_agents)} as expected",
            )
        )

    # sizing drives every downstream budget, so a wrong depth is a wrong run even if it answers
    if golden.expected_depth:
        depth = state["query_depth"]
        ok = depth in golden.expected_depth
        results.append(
            EvalResult(
                layer=layer,
                metric="depth_sized",
                score=1.0 if ok else 0.0,
                passed=ok,
                detail=f"[{golden.id}] triaged as {depth}, expected one of {golden.expected_depth}",
            )
        )

    if golden.should_require_revision:
        revised = state["revision_count"] > 0
        results.append(
            EvalResult(
                layer=layer,
                metric="revision_expected",
                score=1.0 if revised else 0.0,
                passed=revised,
                detail=f"[{golden.id}] revision_count={state['revision_count']}",
            )
        )

    return results


async def _score_one(
    golden: GoldenQuery, state: GraphState, *, use_judge: bool = True
) -> list[EvalResult]:
    """All three layers for one query. Each evaluator already degrades to zero, never raises.

    use_judge=False skips retrieval and report — the two layers that spend a judge model call —
    and keeps trajectory and the golden checks, which read only state already recorded and cost
    nothing. That split is what makes a cheap smoke run possible on a free-tier quota.
    """
    prefix = f"[{golden.id}] "
    results: list[EvalResult] = []

    layers: list[tuple[Any, str]] = [(TrajectoryEvaluator(), "trajectory")]
    if use_judge:
        layers = [(RetrievalEvaluator(), "retrieval"), (ReportEvaluator(), "report"), *layers]

    for evaluator, label in layers:
        try:
            if label == "retrieval":
                scored = await evaluator.evaluate(state, reference=golden.reference_answer)
            else:
                scored = await evaluator.evaluate(state)
        except Exception as exc:
            # a harness run is worthless if one query's crash loses every other query's results
            logger.bind(query_id=golden.id, layer=label, error=str(exc)[:150]).warning(
                "harness.layer_crashed"
            )
            scored = [EvalResult(label, "error", 0.0, False, f"{label} raised: {exc}"[:200])]
        results.extend(
            EvalResult(r.layer, r.metric, r.score, r.passed, prefix + r.detail) for r in scored
        )

    results.extend(_golden_checks(golden, state))
    return results


async def run_evaluation(
    golden_set: list[GoldenQuery], pipeline: PipelineRunner, *, use_judge: bool = True
) -> EvalReport:
    """Runs every golden query through the pipeline and scores it on all three layers.

    use_judge=False is the smoke-test path: routing, depth-sizing and the citation-audit-free
    trajectory layer still get scored on every query, but no RAGAS judge call is made, so this
    is safe to run on a free-tier quota before spending it on the full judged report.
    """
    all_results: list[EvalResult] = []

    for golden in golden_set:
        logger.bind(query_id=golden.id, category=golden.category).info("harness.query_start")
        try:
            state = await pipeline(golden.query)
        except Exception as exc:
            logger.bind(query_id=golden.id, error=str(exc)[:200]).error("harness.pipeline_crashed")
            all_results.append(
                EvalResult("golden_set", "pipeline_error", 0.0, False, f"[{golden.id}] {exc}"[:200])
            )
            continue

        all_results.extend(await _score_one(golden, state, use_judge=use_judge))
        logger.bind(query_id=golden.id).info("harness.query_done")

    return EvalReport(
        results=all_results, query=f"{len(golden_set)} golden queries", session_id="golden_set_run"
    )


def compare_reports(old: EvalReport, new: EvalReport) -> list[str]:
    """Flags any metric whose average score dropped more than REGRESSION_THRESHOLD."""

    def _averages(report: EvalReport) -> dict[str, float]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for r in report.results:
            buckets[f"{r.layer}.{r.metric}"].append(r.score)
        return {key: sum(scores) / len(scores) for key, scores in buckets.items()}

    old_avg, new_avg = _averages(old), _averages(new)
    flags = []
    for key, old_score in old_avg.items():
        new_score = new_avg.get(key)
        if new_score is None:
            flags.append(f"{key}: metric missing from the new report (was {old_score:.3f})")
            continue
        drop = old_score - new_score
        if drop > REGRESSION_THRESHOLD:
            flags.append(f"{key}: {old_score:.3f} -> {new_score:.3f} (-{drop:.3f})")
    return flags


def _write_report(report: EvalReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"report_{stamp}.md"
    json_path = out_dir / f"report_{stamp}.json"
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return md_path, json_path


def _load_report(path: Path) -> EvalReport:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    results = [
        EvalResult(r["layer"], r["metric"], r["score"], r["passed"], r["detail"], r["timestamp"])
        for r in data["results"]
    ]
    return EvalReport(
        results=results, query=data.get("query", ""), session_id=data.get("session_id", "")
    )


async def _main() -> None:
    """uv run python -m amaris.evaluation.harness --golden tests/golden/queries.yaml"""
    from amaris.graph.pipeline import reset_pipeline, run_research
    from amaris.observability.logging import configure_from_settings

    parser = argparse.ArgumentParser(
        description="run the AMARIS golden set through all 3 eval layers"
    )
    parser.add_argument("--golden", default="tests/golden/queries.yaml")
    parser.add_argument("--compare-with", default=None, help="an earlier evals/report_*.json")
    parser.add_argument("--out-dir", default=str(EVALS_DIR))
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="skip retrieval/report (the two RAGAS-judged layers) — a free smoke run",
    )
    args = parser.parse_args()

    configure_from_settings()
    golden_set = load_golden_set(args.golden)
    logger.bind(count=len(golden_set), path=args.golden, judge=not args.no_judge).info(
        "harness.start"
    )

    report = await run_evaluation(golden_set, pipeline=run_research, use_judge=not args.no_judge)
    md_path, json_path = _write_report(report, Path(args.out_dir))

    logger.bind(
        overall=report.overall(), pass_rate=report.pass_rate(), md=str(md_path), json=str(json_path)
    ).info("harness.done")

    if args.compare_with:
        flags = compare_reports(_load_report(Path(args.compare_with)), report)
        if flags:
            logger.bind(count=len(flags)).warning("harness.regressions")
            for flag in flags:
                logger.bind(flag=flag).warning("harness.regression")
        else:
            logger.info("harness.no_regressions")

    await reset_pipeline()


if __name__ == "__main__":
    asyncio.run(_main())
