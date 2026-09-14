"""Shared shapes every evaluation layer returns. No layer imports another layer's module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from amaris.graph.state import GraphState


@dataclass(frozen=True)
class EvalResult:
    """One scored metric. `layer` is retrieval/report/trajectory — never a library name."""

    layer: str
    metric: str
    score: float
    passed: bool
    detail: str
    # false when the judge never produced a number — "not scored" and "scored zero" mean
    # opposite things, and consumers used to tell them apart by grepping detail for "NaN"
    scored: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "metric": self.metric,
            "score": round(self.score, 4),
            "passed": self.passed,
            "detail": self.detail,
            "scored": self.scored,
            "timestamp": self.timestamp,
        }


def zero_result(layer: str, metric: str, detail: str) -> EvalResult:
    """A metric that could not run. Every evaluator returns this instead of raising."""
    return EvalResult(
        layer=layer, metric=metric, score=0.0, passed=False, detail=detail, scored=False
    )


@dataclass
class EvalReport:
    """Aggregates results from one or many runs. Rendered to markdown for evals/report_*.md."""

    results: list[EvalResult]
    query: str = ""
    session_id: str = ""

    def overall(self) -> float:
        """Mean of the metrics that actually scored. An unavailable judge is not a zero."""
        scored = [r for r in self.results if r.scored]
        return round(sum(r.score for r in scored) / len(scored), 4) if scored else 0.0

    def by_layer(self) -> dict[str, list[EvalResult]]:
        grouped: dict[str, list[EvalResult]] = {}
        for result in self.results:
            grouped.setdefault(result.layer, []).append(result)
        return grouped

    def pass_rate(self) -> float:
        return (
            round(sum(1 for r in self.results if r.passed) / len(self.results), 4)
            if self.results
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "session_id": self.session_id,
            "overall": self.overall(),
            "pass_rate": self.pass_rate(),
            "results": [r.to_dict() for r in self.results],
        }

    def to_markdown(self) -> str:
        lines = [f"# Evaluation — {self.query or self.session_id or 'report'}", ""]
        lines.append(f"**Overall:** {self.overall():.2f} · **Pass rate:** {self.pass_rate():.0%}")
        lines.append("")
        for layer, results in self.by_layer().items():
            lines.append(f"## {layer}")
            lines.append("")
            lines.append("| Metric | Score | Passed | Detail |")
            lines.append("|---|---|---|---|")
            for r in results:
                mark = "✅" if r.passed else "❌"
                lines.append(f"| {r.metric} | {r.score:.2f} | {mark} | {r.detail} |")
            lines.append("")
        return "\n".join(lines)


class Evaluator(Protocol):
    """Every layer implements this. Non-blocking by contract: log and return zero, never raise."""

    async def evaluate(self, state: GraphState) -> list[EvalResult]: ...
