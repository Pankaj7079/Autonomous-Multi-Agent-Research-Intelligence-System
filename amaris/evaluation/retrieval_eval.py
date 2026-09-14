"""Layer 1 — RAGAS narrowed to what it actually measures well: retrieval, not generation (ADR-017).

Scope is deliberately narrow: the researcher's gathered sources vs the query. Never the final
report — that conflates retrieval quality with writing quality, which is exactly the mismatch
this layer exists to avoid. Report-level faithfulness/relevancy is report_eval.py's job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from amaris.evaluation import _ragas_shared as shared
from amaris.evaluation.base import EvalResult, zero_result
from amaris.observability.logging import logger

if TYPE_CHECKING:
    from amaris.graph.state import GraphState

LAYER = "retrieval"
# context_precision needs *some* text standing in for "what the research concluded" — using the
# writer's report here would blur retrieval quality with writing quality, so this layer only ever
# sees the analyst's intermediate synthesis, never the final cited report
_FALLBACK_SOURCE_LIMIT = 4
_FALLBACK_SNIPPET_CHARS = 200


def _response_proxy(state: GraphState) -> str:
    """What retrieval was 'for', without reaching into the writer's report.

    Falls back to content snippets, not bare titles — found live, titles alone give the precision
    judge almost nothing to assess relevance against and score close to 0 regardless of how good
    the sources actually are. This path should rarely fire in practice (analysis normally runs
    before the evaluator), but a run where the analyst produced nothing must still be scorable.
    """
    if state["analyzed_data"].strip():
        return state["analyzed_data"]
    snippets = [
        item.get("content", "")[:_FALLBACK_SNIPPET_CHARS]
        for item in state["raw_research"]
        if item.get("content")
    ]
    return " ".join(snippets[:_FALLBACK_SOURCE_LIMIT])


class RetrievalEvaluator:
    """context_precision always; context_recall only when a golden query supplies a reference."""

    async def evaluate(self, state: GraphState, reference: str | None = None) -> list[EvalResult]:
        contexts = shared.trim_contexts([item.get("content", "") for item in state["raw_research"]])
        query = state["original_query"]
        response = _response_proxy(state)

        if not contexts:
            logger.bind(query=query[:80]).debug("retrieval_eval.skipped")
            return [zero_result(LAYER, "context_precision", "no sources gathered")]

        if not shared.is_available():
            return [
                zero_result(
                    LAYER, "context_precision", shared.unavailable_reason() or "unavailable"
                )
            ]

        with shared.quiet_deprecations():
            from ragas.metrics import LLMContextPrecisionWithoutReference

            metrics: list[Any] = [LLMContextPrecisionWithoutReference(llm=shared.judge())]
            if reference:
                from ragas.metrics import LLMContextRecall

                metrics.append(LLMContextRecall(llm=shared.judge()))

        raw = await shared.run_metrics(
            metrics, query, response, contexts, reference=reference, label=LAYER
        )
        if raw is None:
            return [zero_result(LAYER, "context_precision", "ragas call failed or timed out")]

        results = []
        for key, value in raw.items():
            metric = "context_recall" if "recall" in key else "context_precision"
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
                    detail=f"{len(contexts)} sources scored against the query",
                )
            )

        # an unscored 0.0 needs to be visibly different from a real 0.0 in the log, or "the judge
        # got rate limited" and "the sources were genuinely useless" look identical
        unscored = [r.metric for r in results if not r.scored]
        logger.bind(**{r.metric: r.score for r in results}, unscored=unscored or None).info(
            "retrieval_eval.scored"
        )
        return results
