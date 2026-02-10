"""Tests for evaluate_rag_pipeline and LLM-as-judge scoring."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from requirements_graphrag_api.evaluation import (
    EvaluationExample,
    evaluate_rag_pipeline,
)
from requirements_graphrag_api.evaluation import (
    _llm_judge_metrics as llm_judge_metrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm_response(score: float, reasoning: str = "ok") -> MagicMock:
    """Create a mock LLM response with a JSON score payload."""
    resp = MagicMock()
    resp.content = f'{{"score": {score}, "reasoning": "{reasoning}"}}'
    return resp


def _make_judge_llm_mock(scores: dict[str, float]) -> MagicMock:
    """Return a mock ChatOpenAI whose ainvoke returns metric-specific scores.

    The mock inspects the prompt text to determine which metric is being scored
    and returns the corresponding score from the ``scores`` dict.
    """

    async def _side_effect(prompt: str) -> MagicMock:
        # Match prompt to metric by checking the prompt template text
        if "faithfulness" in prompt.lower()[:80]:
            return _mock_llm_response(scores.get("faithfulness", 0.5))
        if "relevancy" in prompt.lower()[:80]:
            return _mock_llm_response(scores.get("answer_relevancy", 0.5))
        if "precision" in prompt.lower()[:80]:
            return _mock_llm_response(scores.get("context_precision", 0.5))
        if "recall" in prompt.lower()[:80]:
            return _mock_llm_response(scores.get("context_recall", 0.5))
        return _mock_llm_response(0.5)

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=_side_effect)
    return llm


# ---------------------------------------------------------------------------
# _llm_judge_metrics
# ---------------------------------------------------------------------------


class TestLlmJudgeMetrics:
    """Tests for the _llm_judge_metrics helper."""

    @pytest.mark.asyncio
    async def test_returns_four_metrics(self) -> None:
        """All four RAGAS metrics should be present."""
        scores = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.85,
            "context_precision": 0.8,
            "context_recall": 0.75,
        }
        mock_llm = _make_judge_llm_mock(scores)

        with patch(
            "requirements_graphrag_api.evaluation._get_judge_llm",
            return_value=mock_llm,
        ):
            result = await llm_judge_metrics(
                question="What is X?",
                answer="X is a thing.",
                contexts=["X is defined as a thing."],
                ground_truth="X is a thing used for Y.",
            )

        assert set(result.keys()) == {
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        }
        assert result["faithfulness"] == 0.9
        assert result["answer_relevancy"] == 0.85
        assert result["context_precision"] == 0.8
        assert result["context_recall"] == 0.75

    @pytest.mark.asyncio
    async def test_empty_contexts_still_scores(self) -> None:
        """Empty context list should not crash; scores may be low."""
        mock_llm = _make_judge_llm_mock(
            {
                "faithfulness": 0.1,
                "answer_relevancy": 0.5,
                "context_precision": 0.0,
                "context_recall": 0.0,
            }
        )

        with patch(
            "requirements_graphrag_api.evaluation._get_judge_llm",
            return_value=mock_llm,
        ):
            result = await llm_judge_metrics(
                question="What is X?",
                answer="X is a thing.",
                contexts=[],
                ground_truth="X is a thing.",
            )

        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_llm_failure_returns_zero(self) -> None:
        """If the LLM call raises, the metric should fall back to 0.0."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=Exception("API error"))

        with patch(
            "requirements_graphrag_api.evaluation._get_judge_llm",
            return_value=llm,
        ):
            result = await llm_judge_metrics(
                question="What is X?",
                answer="X is a thing.",
                contexts=["context"],
                ground_truth="ground truth",
            )

        assert all(v == 0.0 for v in result.values())


# ---------------------------------------------------------------------------
# evaluate_rag_pipeline (integration-level, mocked)
# ---------------------------------------------------------------------------


class TestEvaluateRagPipeline:
    """Tests for the top-level evaluate_rag_pipeline function."""

    @pytest.mark.asyncio
    async def test_aggregates_scores_correctly(self) -> None:
        """Aggregate metrics should be the mean of per-example scores."""
        scores = {
            "faithfulness": 0.8,
            "answer_relevancy": 0.9,
            "context_precision": 0.7,
            "context_recall": 0.6,
        }
        mock_judge_llm = _make_judge_llm_mock(scores)

        examples = [
            EvaluationExample(
                id="test-1",
                question="What is X?",
                ground_truth="X is a concept.",
                category="definition",
            ),
        ]

        with (
            patch(
                "requirements_graphrag_api.evaluation._load_evaluation_dataset",
                return_value=examples,
            ),
            patch(
                "requirements_graphrag_api.evaluation._generate_answer",
                new_callable=AsyncMock,
                return_value=("X is a concept used in engineering.", ["Some context"]),
            ),
            patch(
                "requirements_graphrag_api.evaluation._get_judge_llm",
                return_value=mock_judge_llm,
            ),
        ):
            report = await evaluate_rag_pipeline(
                config=MagicMock(),
                retriever=MagicMock(),
                driver=MagicMock(),
                max_samples=1,
            )

        assert report.total_samples == 1
        assert report.passed_samples == 1
        assert report.aggregate_metrics["faithfulness"] == 0.8
        assert report.aggregate_metrics["answer_relevancy"] == 0.9
        assert report.aggregate_metrics["context_precision"] == 0.7
        assert report.aggregate_metrics["context_recall"] == 0.6
        expected_avg = (0.8 + 0.9 + 0.7 + 0.6) / 4
        assert abs(report.aggregate_metrics["avg_score"] - expected_avg) < 1e-6

    @pytest.mark.asyncio
    async def test_error_handling_records_error(self) -> None:
        """If _evaluate_single raises, the error should be captured."""
        examples = [
            EvaluationExample(
                id="err-1",
                question="Broken?",
                ground_truth="N/A",
            ),
        ]

        with (
            patch(
                "requirements_graphrag_api.evaluation._load_evaluation_dataset",
                return_value=examples,
            ),
            patch(
                "requirements_graphrag_api.evaluation._generate_answer",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
        ):
            report = await evaluate_rag_pipeline(
                config=MagicMock(),
                retriever=MagicMock(),
                driver=MagicMock(),
            )

        assert report.total_samples == 0
        assert len(report.errors) == 1
        assert "boom" in report.errors[0]

    @pytest.mark.asyncio
    async def test_low_scores_fail(self) -> None:
        """Examples with avg_score < 0.5 should not pass."""
        low_scores = {
            "faithfulness": 0.2,
            "answer_relevancy": 0.3,
            "context_precision": 0.1,
            "context_recall": 0.2,
        }
        mock_judge_llm = _make_judge_llm_mock(low_scores)

        examples = [
            EvaluationExample(
                id="low-1",
                question="What is X?",
                ground_truth="X is a concept.",
            ),
        ]

        with (
            patch(
                "requirements_graphrag_api.evaluation._load_evaluation_dataset",
                return_value=examples,
            ),
            patch(
                "requirements_graphrag_api.evaluation._generate_answer",
                new_callable=AsyncMock,
                return_value=("Wrong answer.", []),
            ),
            patch(
                "requirements_graphrag_api.evaluation._get_judge_llm",
                return_value=mock_judge_llm,
            ),
        ):
            report = await evaluate_rag_pipeline(
                config=MagicMock(),
                retriever=MagicMock(),
                driver=MagicMock(),
            )

        assert report.passed_samples == 0
        assert report.aggregate_metrics["avg_score"] < 0.5
