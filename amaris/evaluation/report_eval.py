"""Layer 2 — final report quality vs its sources and vs the query (ADR-017).

Originally speced as agent_eval.py on DeepEval (TaskCompletion, ToolCorrectness, AnswerRelevancy).
Replaced before building: ToolCorrectnessMetric assumes one correct tool per intent, but AMARIS's
ReAct loop deliberately chooses web_search vs scrape_webpage adaptively per iteration — grading
that against a fixed expected_tools mapping would fight the exact autonomy ADR-003 is about.
TaskCompletionMetric duplicates trajectory_eval's termination_quality with an LLM guess instead of
real state. That left only AnswerRelevancy worth keeping, which RAGAS already computes — and RAGAS's
Faithfulness metric (fabricated report -> 0.0, grounded report -> 1.0, verified live in Phase 6)
belongs here too: it is the one hallucination check in the whole evaluation design, and the
original layer split left it with no home. Net effect: no second heavy dependency, no default
telemetry call-home, same or better signal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from amaris.evaluation import _ragas_shared as shared
from amaris.evaluation.base import EvalResult, zero_result
from amaris.observability.logging import logger

if TYPE_CHECKING:
    from amaris.graph.state import GraphState

LAYER = "report"


class ReportEvaluator:
    """Faithfulness (report vs sources) and answer_relevancy (report vs query)."""

    async def evaluate(self, state: GraphState) -> list[EvalResult]:
        report = state["final_report"] or state["draft_report"]
        query = state["original_query"]
        contexts = shared.trim_contexts([item.get("content", "") for item in state["raw_research"]])

        if not report.strip() or not contexts:
            logger.bind(chars=len(report), contexts=len(contexts)).debug("report_eval.skipped")
            return [zero_result(LAYER, "faithfulness", "no report or no sources to check against")]

        if not shared.is_available():
            return [
                zero_result(LAYER, "faithfulness", shared.unavailable_reason() or "unavailable")
            ]

        with shared.quiet_deprecations():
            from ragas.metrics import Faithfulness

            metrics: list[Any] = [Faithfulness(llm=shared.judge())]
            embedder = shared.embedder()
            if embedder is not None:
                from ragas.metrics import ResponseRelevancy

                metrics.append(
                    ResponseRelevancy(
                        llm=shared.judge(),
                        embeddings=embedder,
                        strictness=shared.RELEVANCY_STRICTNESS,
                    )
                )

        raw = await shared.run_metrics(metrics, query, report, contexts, label=LAYER)
        if raw is None:
            return [zero_result(LAYER, "faithfulness", "ragas call failed or timed out")]

        results = []
        for key, value in raw.items():
            metric = "answer_relevancy" if "relevan" in key else "faithfulness"
            score = shared.clamp_or_none(value)
            if score is None:
                results.append(
                    zero_result(
                        LAYER, metric, "judge returned NaN — likely rate limited mid-scoring"
                    )
                )
                continue
            results.append(
                EvalResult(
                    layer=LAYER,
                    metric=metric,
                    score=score,
                    passed=score >= 0.5,
                    detail=f"report checked against {len(contexts)} sources",
                )
            )

        # a 0.0 that was actually a NaN needs to be visibly different from a real 0.0 in the log,
        # or "the judge got rate limited" and "the report genuinely hallucinated" look identical
        nan_metrics = [r.metric for r in results if "NaN" in r.detail]
        logger.bind(**{r.metric: r.score for r in results}, nan_metrics=nan_metrics or None).info(
            "report_eval.scored"
        )
        return results
