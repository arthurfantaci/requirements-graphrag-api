"""Tests for evaluation/ragas_evaluators.py module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.prompts import ChatPromptTemplate

from requirements_graphrag_api.evaluation.ragas_evaluators import (
    _cosine_similarity,
    _get_judge_llm,
    _parse_llm_json,
    _parse_llm_score,
    answer_correctness_evaluator,
    answer_relevancy_evaluator,
    answer_semantic_similarity_evaluator,
    context_entity_recall_evaluator,
    context_precision_evaluator,
    context_recall_evaluator,
    faithfulness_evaluator,
)

# A minimal template that accepts any variables via format_messages
_MOCK_TEMPLATE = ChatPromptTemplate.from_messages([("system", "Evaluate."), ("human", "Go.")])


class TestParseLlmScore:
    """Tests for _parse_llm_score helper function."""

    def test_valid_json(self) -> None:
        score, reasoning = _parse_llm_score('{"score": 0.85, "reasoning": "Good answer"}')
        assert score == 0.85
        assert reasoning == "Good answer"

    def test_json_in_markdown(self) -> None:
        response = '```json\n{"score": 0.9, "reasoning": "Excellent"}\n```'
        score, reasoning = _parse_llm_score(response)
        assert score == 0.9
        assert reasoning == "Excellent"

    def test_json_in_plain_markdown(self) -> None:
        response = '```\n{"score": 0.75, "reasoning": "Okay"}\n```'
        score, reasoning = _parse_llm_score(response)
        assert score == 0.75
        assert reasoning == "Okay"

    def test_clamps_score_to_max(self) -> None:
        score, _ = _parse_llm_score('{"score": 1.5, "reasoning": "test"}')
        assert score == 1.0

    def test_clamps_score_to_min(self) -> None:
        score, _ = _parse_llm_score('{"score": -0.5, "reasoning": "test"}')
        assert score == 0.0

    def test_invalid_json(self) -> None:
        score, reasoning = _parse_llm_score("not valid json")
        assert score == 0.0
        assert "Parse error" in reasoning

    def test_missing_score_key(self) -> None:
        score, _ = _parse_llm_score('{"reasoning": "no score"}')
        assert score == 0.0


class TestGetJudgeLlm:
    """Tests for _get_judge_llm helper function."""

    def test_returns_chat_openai(self) -> None:
        llm = _get_judge_llm()
        assert llm is not None
        assert llm.temperature == 0


class TestFaithfulnessEvaluator:
    """Tests for faithfulness evaluator."""

    @pytest.mark.asyncio
    async def test_missing_fields(self) -> None:
        run = MagicMock()
        run.inputs = {"question": "test"}
        run.outputs = {}

        result = await faithfulness_evaluator(run)

        assert result["key"] == "faithfulness"
        assert result["score"] == 0.0
        assert "Missing" in result["comment"]

    @pytest.mark.asyncio
    async def test_with_valid_inputs(self) -> None:
        run = MagicMock()
        run.inputs = {"question": "What is X?", "context": "X is a thing."}
        run.outputs = {"answer": "X is a thing that does stuff."}

        mock_response = MagicMock()
        mock_response.content = '{"score": 0.9, "reasoning": "Well grounded"}'

        with (
            patch(
                "requirements_graphrag_api.evaluation.ragas_evaluators.get_prompt",
                new_callable=AsyncMock,
                return_value=_MOCK_TEMPLATE,
            ),
            patch(
                "requirements_graphrag_api.evaluation.ragas_evaluators._get_judge_llm"
            ) as mock_llm,
        ):
            mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)
            result = await faithfulness_evaluator(run)

        assert result["key"] == "faithfulness"
        assert result["score"] == 0.9
        assert result["comment"] == "Well grounded"


class TestAnswerRelevancyEvaluator:
    """Tests for answer relevancy evaluator."""

    @pytest.mark.asyncio
    async def test_missing_fields(self) -> None:
        run = MagicMock()
        run.inputs = {}
        run.outputs = {"answer": "test"}

        result = await answer_relevancy_evaluator(run)

        assert result["key"] == "answer_relevancy"
        assert result["score"] == 0.0
        assert "Missing" in result["comment"]

    @pytest.mark.asyncio
    async def test_with_valid_inputs(self) -> None:
        run = MagicMock()
        run.inputs = {"question": "What is X?"}
        run.outputs = {"answer": "X is a thing."}

        mock_response = MagicMock()
        mock_response.content = '{"score": 0.85, "reasoning": "Relevant"}'

        with (
            patch(
                "requirements_graphrag_api.evaluation.ragas_evaluators.get_prompt",
                new_callable=AsyncMock,
                return_value=_MOCK_TEMPLATE,
            ),
            patch(
                "requirements_graphrag_api.evaluation.ragas_evaluators._get_judge_llm"
            ) as mock_llm,
        ):
            mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)
            result = await answer_relevancy_evaluator(run)

        assert result["key"] == "answer_relevancy"
        assert result["score"] == 0.85


class TestContextPrecisionEvaluator:
    """Tests for context precision evaluator."""

    @pytest.mark.asyncio
    async def test_missing_fields(self) -> None:
        run = MagicMock()
        run.inputs = {"question": "test"}
        run.outputs = {}

        result = await context_precision_evaluator(run)

        assert result["key"] == "context_precision"
        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_contexts_as_list(self) -> None:
        run = MagicMock()
        run.inputs = {"question": "What is X?", "contexts": ["Context 1", "Context 2"]}
        run.outputs = {}

        mock_response = MagicMock()
        mock_response.content = '{"score": 0.8, "reasoning": "Good precision"}'

        with (
            patch(
                "requirements_graphrag_api.evaluation.ragas_evaluators.get_prompt",
                new_callable=AsyncMock,
                return_value=_MOCK_TEMPLATE,
            ),
            patch(
                "requirements_graphrag_api.evaluation.ragas_evaluators._get_judge_llm"
            ) as mock_llm,
        ):
            mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)
            result = await context_precision_evaluator(run)

        assert result["key"] == "context_precision"
        assert result["score"] == 0.8


class TestContextRecallEvaluator:
    """Tests for context recall evaluator."""

    @pytest.mark.asyncio
    async def test_missing_ground_truth(self) -> None:
        run = MagicMock()
        run.inputs = {"question": "test", "contexts": ["ctx"]}
        run.outputs = {}

        result = await context_recall_evaluator(run, example=None)

        assert result["key"] == "context_recall"
        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_with_example_ground_truth(self) -> None:
        run = MagicMock()
        run.inputs = {"question": "What is X?"}
        run.outputs = {"contexts": ["X is defined as..."]}

        example = MagicMock()
        example.outputs = {"expected_answer": "X is a concept."}

        mock_response = MagicMock()
        mock_response.content = '{"score": 0.7, "reasoning": "Partial recall"}'

        with (
            patch(
                "requirements_graphrag_api.evaluation.ragas_evaluators.get_prompt",
                new_callable=AsyncMock,
                return_value=_MOCK_TEMPLATE,
            ),
            patch(
                "requirements_graphrag_api.evaluation.ragas_evaluators._get_judge_llm"
            ) as mock_llm,
        ):
            mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)
            result = await context_recall_evaluator(run, example=example)

        assert result["key"] == "context_recall"
        assert result["score"] == 0.7


class TestParseLlmJson:
    """Tests for _parse_llm_json helper function."""

    def test_valid_json(self) -> None:
        parsed = _parse_llm_json('{"entities": ["ISO 26262", "ASIL"]}')
        assert parsed == {"entities": ["ISO 26262", "ASIL"]}

    def test_json_in_markdown(self) -> None:
        response = '```json\n{"tp": 3, "fp": 1, "fn": 0}\n```'
        parsed = _parse_llm_json(response)
        assert parsed["tp"] == 3

    def test_invalid_json(self) -> None:
        parsed = _parse_llm_json("not valid json")
        assert parsed == {}


class TestCosineSimilarity:
    """Tests for _cosine_similarity helper function."""

    def test_identical_vectors(self) -> None:
        vec = [1.0, 0.0, 0.5]
        assert _cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestAnswerCorrectnessEvaluator:
    """Tests for answer correctness evaluator."""

    @pytest.mark.asyncio
    async def test_missing_fields(self) -> None:
        run = MagicMock()
        run.inputs = {"question": "test"}
        run.outputs = {"answer": "test"}

        result = await answer_correctness_evaluator(run, example=None)

        assert result["key"] == "answer_correctness"
        assert result["score"] == 0.0
        assert "Missing" in result["comment"]

    @pytest.mark.asyncio
    async def test_with_valid_inputs(self) -> None:
        run = MagicMock()
        run.inputs = {"question": "What is X?"}
        run.outputs = {"answer": "X is a concept used in engineering."}

        example = MagicMock()
        example.outputs = {"expected_answer": "X is a concept."}

        # LLM returns claim classifications: 2 TP, 1 FP, 0 FN → F1 = 4/5 = 0.8
        mock_response = MagicMock()
        mock_response.content = json.dumps(
            {
                "answer_claims": [
                    {"claim": "X is a concept", "classification": "TP"},
                    {"claim": "used in engineering", "classification": "FP"},
                ],
                "ground_truth_claims": [
                    {"claim": "X is a concept", "classification": "TP"},
                ],
                "tp": 2,
                "fp": 1,
                "fn": 0,
                "reasoning": "Mostly correct with one unsupported claim",
            }
        )

        with (
            patch(
                "requirements_graphrag_api.evaluation.ragas_evaluators.get_prompt",
                new_callable=AsyncMock,
                return_value=_MOCK_TEMPLATE,
            ),
            patch(
                "requirements_graphrag_api.evaluation.ragas_evaluators._get_judge_llm"
            ) as mock_llm,
        ):
            mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)
            result = await answer_correctness_evaluator(run, example=example)

        assert result["key"] == "answer_correctness"
        # F1 = 2*2 / (2*2 + 1 + 0) = 4/5 = 0.8
        assert result["score"] == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_f1_zero_when_no_tp(self) -> None:
        run = MagicMock()
        run.inputs = {"question": "What is X?"}
        run.outputs = {"answer": "Y is unrelated."}

        example = MagicMock()
        example.outputs = {"expected_answer": "X is a concept."}

        mock_response = MagicMock()
        mock_response.content = json.dumps({"tp": 0, "fp": 1, "fn": 1, "reasoning": "No overlap"})

        with (
            patch(
                "requirements_graphrag_api.evaluation.ragas_evaluators.get_prompt",
                new_callable=AsyncMock,
                return_value=_MOCK_TEMPLATE,
            ),
            patch(
                "requirements_graphrag_api.evaluation.ragas_evaluators._get_judge_llm"
            ) as mock_llm,
        ):
            mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)
            result = await answer_correctness_evaluator(run, example=example)

        assert result["score"] == 0.0


class TestAnswerSemanticSimilarityEvaluator:
    """Tests for answer semantic similarity evaluator."""

    @pytest.mark.asyncio
    async def test_missing_fields(self) -> None:
        run = MagicMock()
        run.outputs = {"answer": "test"}

        result = await answer_semantic_similarity_evaluator(run, example=None)

        assert result["key"] == "answer_semantic_similarity"
        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_identical_texts(self) -> None:
        run = MagicMock()
        run.outputs = {"answer": "X is a concept."}

        example = MagicMock()
        example.outputs = {"expected_answer": "X is a concept."}

        # Return identical embedding vectors
        vec = [0.5, 0.3, 0.8]

        with patch(
            "requirements_graphrag_api.evaluation.ragas_evaluators.OpenAIEmbeddings"
        ) as mock_embed_cls:
            mock_instance = MagicMock()
            mock_instance.aembed_documents = AsyncMock(return_value=[vec, vec])
            mock_embed_cls.return_value = mock_instance

            result = await answer_semantic_similarity_evaluator(run, example=example)

        assert result["key"] == "answer_semantic_similarity"
        assert result["score"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_orthogonal_embeddings(self) -> None:
        run = MagicMock()
        run.outputs = {"answer": "completely unrelated"}

        example = MagicMock()
        example.outputs = {"expected_answer": "X is a concept."}

        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]

        with patch(
            "requirements_graphrag_api.evaluation.ragas_evaluators.OpenAIEmbeddings"
        ) as mock_embed_cls:
            mock_instance = MagicMock()
            mock_instance.aembed_documents = AsyncMock(return_value=[vec_a, vec_b])
            mock_embed_cls.return_value = mock_instance

            result = await answer_semantic_similarity_evaluator(run, example=example)

        assert result["score"] == pytest.approx(0.0)


class TestContextEntityRecallEvaluator:
    """Tests for context entity recall evaluator."""

    @pytest.mark.asyncio
    async def test_missing_fields(self) -> None:
        run = MagicMock()
        run.inputs = {}
        run.outputs = {"contexts": ["some context"]}

        result = await context_entity_recall_evaluator(run, example=None)

        assert result["key"] == "context_entity_recall"
        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_with_valid_inputs(self) -> None:
        run = MagicMock()
        run.inputs = {}
        run.outputs = {"contexts": ["ISO 26262 defines ASIL levels."]}

        example = MagicMock()
        example.outputs = {"expected_answer": "ISO 26262 and ASIL are safety standards."}

        # Call 1 (context): returns ISO 26262, ASIL
        ctx_response = MagicMock()
        ctx_response.content = '{"entities": ["ISO 26262", "ASIL"]}'

        # Call 2 (ground truth): returns ISO 26262, ASIL, functional safety
        gt_response = MagicMock()
        gt_response.content = '{"entities": ["ISO 26262", "ASIL", "functional safety"]}'

        with (
            patch(
                "requirements_graphrag_api.evaluation.ragas_evaluators.get_prompt",
                new_callable=AsyncMock,
                return_value=_MOCK_TEMPLATE,
            ),
            patch(
                "requirements_graphrag_api.evaluation.ragas_evaluators._get_judge_llm"
            ) as mock_llm,
        ):
            mock_llm.return_value.ainvoke = AsyncMock(side_effect=[ctx_response, gt_response])
            result = await context_entity_recall_evaluator(run, example=example)

        assert result["key"] == "context_entity_recall"
        # 2 of 3 ground truth entities found → recall = 2/3 ≈ 0.667
        assert result["score"] == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_no_ground_truth_entities(self) -> None:
        run = MagicMock()
        run.inputs = {}
        run.outputs = {"contexts": ["Some generic text."]}

        example = MagicMock()
        example.outputs = {"expected_answer": "A short answer."}

        ctx_response = MagicMock()
        ctx_response.content = '{"entities": ["some entity"]}'

        gt_response = MagicMock()
        gt_response.content = '{"entities": []}'

        with (
            patch(
                "requirements_graphrag_api.evaluation.ragas_evaluators.get_prompt",
                new_callable=AsyncMock,
                return_value=_MOCK_TEMPLATE,
            ),
            patch(
                "requirements_graphrag_api.evaluation.ragas_evaluators._get_judge_llm"
            ) as mock_llm,
        ):
            mock_llm.return_value.ainvoke = AsyncMock(side_effect=[ctx_response, gt_response])
            result = await context_entity_recall_evaluator(run, example=example)

        # No ground truth entities → score 1.0 (vacuous truth)
        assert result["score"] == 1.0
