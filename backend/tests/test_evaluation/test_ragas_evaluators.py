"""Tests for evaluation/ragas_evaluators.py module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from requirements_graphrag_api.evaluation.ragas_evaluators import (
    _get_judge_llm,
    _parse_llm_score,
    answer_relevancy_evaluator,
    context_precision_evaluator,
    context_recall_evaluator,
    faithfulness_evaluator,
)


class TestParseLlmScore:
    """Tests for _parse_llm_score helper function."""

    def test_valid_json(self) -> None:
        """Test parsing valid JSON response."""
        score, reasoning = _parse_llm_score('{"score": 0.85, "reasoning": "Good answer"}')
        assert score == 0.85
        assert reasoning == "Good answer"

    def test_json_in_markdown(self) -> None:
        """Test parsing JSON in markdown code block."""
        response = '```json\n{"score": 0.9, "reasoning": "Excellent"}\n```'
        score, reasoning = _parse_llm_score(response)
        assert score == 0.9
        assert reasoning == "Excellent"

    def test_json_in_plain_markdown(self) -> None:
        """Test parsing JSON in plain markdown code block."""
        response = '```\n{"score": 0.75, "reasoning": "Okay"}\n```'
        score, reasoning = _parse_llm_score(response)
        assert score == 0.75
        assert reasoning == "Okay"

    def test_clamps_score_to_max(self) -> None:
        """Test that scores > 1.0 are clamped."""
        score, _ = _parse_llm_score('{"score": 1.5, "reasoning": "test"}')
        assert score == 1.0

    def test_clamps_score_to_min(self) -> None:
        """Test that scores < 0.0 are clamped."""
        score, _ = _parse_llm_score('{"score": -0.5, "reasoning": "test"}')
        assert score == 0.0

    def test_invalid_json(self) -> None:
        """Test that invalid JSON returns 0.0."""
        score, reasoning = _parse_llm_score("not valid json")
        assert score == 0.0
        assert "Parse error" in reasoning

    def test_missing_score_key(self) -> None:
        """Test that missing score key returns 0.0."""
        score, _ = _parse_llm_score('{"reasoning": "no score"}')
        assert score == 0.0


class TestGetJudgeLlm:
    """Tests for _get_judge_llm helper function."""

    def test_returns_chat_openai(self) -> None:
        """Test that function returns ChatOpenAI instance."""
        llm = _get_judge_llm()
        assert llm is not None
        assert llm.temperature == 0


class TestFaithfulnessEvaluator:
    """Tests for faithfulness evaluator."""

    @pytest.mark.asyncio
    async def test_missing_fields(self) -> None:
        """Test that missing fields return score 0.0."""
        run = MagicMock()
        run.inputs = {"question": "test"}
        run.outputs = {}

        result = await faithfulness_evaluator(run)

        assert result["key"] == "faithfulness"
        assert result["score"] == 0.0
        assert "Missing" in result["comment"]

    @pytest.mark.asyncio
    async def test_with_valid_inputs(self) -> None:
        """Test evaluator with valid inputs calls LLM."""
        run = MagicMock()
        run.inputs = {"question": "What is X?", "context": "X is a thing."}
        run.outputs = {"answer": "X is a thing that does stuff."}

        mock_response = MagicMock()
        mock_response.content = '{"score": 0.9, "reasoning": "Well grounded"}'

        with patch(
            "requirements_graphrag_api.evaluation.ragas_evaluators._get_judge_llm"
        ) as mock_llm:
            mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)
            result = await faithfulness_evaluator(run)

        assert result["key"] == "faithfulness"
        assert result["score"] == 0.9
        assert result["comment"] == "Well grounded"


class TestAnswerRelevancyEvaluator:
    """Tests for answer relevancy evaluator."""

    @pytest.mark.asyncio
    async def test_missing_fields(self) -> None:
        """Test that missing fields return score 0.0."""
        run = MagicMock()
        run.inputs = {}
        run.outputs = {"answer": "test"}

        result = await answer_relevancy_evaluator(run)

        assert result["key"] == "answer_relevancy"
        assert result["score"] == 0.0
        assert "Missing" in result["comment"]

    @pytest.mark.asyncio
    async def test_with_valid_inputs(self) -> None:
        """Test evaluator with valid inputs calls LLM."""
        run = MagicMock()
        run.inputs = {"question": "What is X?"}
        run.outputs = {"answer": "X is a thing."}

        mock_response = MagicMock()
        mock_response.content = '{"score": 0.85, "reasoning": "Relevant"}'

        with patch(
            "requirements_graphrag_api.evaluation.ragas_evaluators._get_judge_llm"
        ) as mock_llm:
            mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)
            result = await answer_relevancy_evaluator(run)

        assert result["key"] == "answer_relevancy"
        assert result["score"] == 0.85


class TestContextPrecisionEvaluator:
    """Tests for context precision evaluator."""

    @pytest.mark.asyncio
    async def test_missing_fields(self) -> None:
        """Test that missing fields return score 0.0."""
        run = MagicMock()
        run.inputs = {"question": "test"}
        run.outputs = {}

        result = await context_precision_evaluator(run)

        assert result["key"] == "context_precision"
        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_contexts_as_list(self) -> None:
        """Test that contexts list is joined properly."""
        run = MagicMock()
        run.inputs = {"question": "What is X?", "contexts": ["Context 1", "Context 2"]}
        run.outputs = {}

        mock_response = MagicMock()
        mock_response.content = '{"score": 0.8, "reasoning": "Good precision"}'

        with patch(
            "requirements_graphrag_api.evaluation.ragas_evaluators._get_judge_llm"
        ) as mock_llm:
            mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)
            result = await context_precision_evaluator(run)

        assert result["key"] == "context_precision"
        assert result["score"] == 0.8


class TestContextRecallEvaluator:
    """Tests for context recall evaluator."""

    @pytest.mark.asyncio
    async def test_missing_ground_truth(self) -> None:
        """Test that missing ground truth returns score 0.0."""
        run = MagicMock()
        run.inputs = {"question": "test", "contexts": ["ctx"]}
        run.outputs = {}

        result = await context_recall_evaluator(run, example=None)

        assert result["key"] == "context_recall"
        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_with_example_ground_truth(self) -> None:
        """Test that ground truth is extracted from example."""
        run = MagicMock()
        run.inputs = {"question": "What is X?"}
        run.outputs = {"contexts": ["X is defined as..."]}

        example = MagicMock()
        example.outputs = {"expected_answer": "X is a concept."}

        mock_response = MagicMock()
        mock_response.content = '{"score": 0.7, "reasoning": "Partial recall"}'

        with patch(
            "requirements_graphrag_api.evaluation.ragas_evaluators._get_judge_llm"
        ) as mock_llm:
            mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)
            result = await context_recall_evaluator(run, example=example)

        assert result["key"] == "context_recall"
        assert result["score"] == 0.7
