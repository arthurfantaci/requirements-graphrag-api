"""Tests for agentic RAG evaluators.

Tests cover:
- Tool selection evaluator
- Iteration efficiency evaluator
- Critic calibration evaluator
- Multi-hop reasoning evaluator
- LLM response parsing
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from requirements_graphrag_api.evaluation.agentic_evaluators import (
    _parse_llm_score,
    critic_calibration_evaluator,
    iteration_efficiency_evaluator,
    multi_hop_reasoning_evaluator,
    tool_selection_evaluator,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_run() -> MagicMock:
    """Create a mock LangSmith Run."""
    run = MagicMock()
    run.name = "test_run"
    run.inputs = {
        "question": "What is requirements traceability?",
        "query": "What is requirements traceability?",
    }
    run.outputs = {
        "answer": "Requirements traceability is the ability to trace requirements.",
        "final_answer": "Requirements traceability is the ability to trace requirements.",
        "tools_used": ["graph_search"],
        "tool_order": ["graph_search"],
        "iteration_count": 1,
        "critique": {
            "confidence": 0.8,
            "completeness": "complete",
            "missing_aspects": [],
        },
        "reasoning_chain": [
            "Identified query about requirements traceability",
            "Searched graph for relevant information",
            "Synthesized answer from context",
        ],
    }
    return run


@pytest.fixture
def mock_example() -> MagicMock:
    """Create a mock LangSmith Example."""
    example = MagicMock()
    example.inputs = {
        "question": "What is requirements traceability?",
        "complexity": "simple",
    }
    example.outputs = {
        "expected_answer": "Requirements traceability is the ability to trace requirements.",
        "expected_tools": ["graph_search"],
        "expected_iterations": 1,
        "expert_quality": 0.9,
        "expert_missing_aspects": [],
        "required_reasoning_steps": [
            "Define requirements traceability",
            "Explain its purpose",
        ],
    }
    return example


def create_mock_llm_response(score: float, reasoning: str) -> MagicMock:
    """Create a mock LLM response with score and reasoning."""
    response = MagicMock()
    response.content = json.dumps({"score": score, "reasoning": reasoning})
    return response


# =============================================================================
# PARSE LLM SCORE TESTS
# =============================================================================


class TestParseLLMScore:
    """Tests for the _parse_llm_score helper function."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        response = '{"score": 0.85, "reasoning": "Good tool selection"}'
        score, reasoning = _parse_llm_score(response)
        assert score == 0.85
        assert reasoning == "Good tool selection"

    def test_parse_json_with_code_block(self):
        """Test parsing JSON wrapped in code block."""
        response = '```json\n{"score": 0.75, "reasoning": "Adequate"}\n```'
        score, reasoning = _parse_llm_score(response)
        assert score == 0.75
        assert reasoning == "Adequate"

    def test_parse_json_with_generic_code_block(self):
        """Test parsing JSON wrapped in generic code block."""
        response = '```\n{"score": 0.9, "reasoning": "Excellent"}\n```'
        score, reasoning = _parse_llm_score(response)
        assert score == 0.9
        assert reasoning == "Excellent"

    def test_clamp_score_above_1(self):
        """Test that scores above 1.0 are clamped."""
        response = '{"score": 1.5, "reasoning": "Over max"}'
        score, _ = _parse_llm_score(response)
        assert score == 1.0

    def test_clamp_score_below_0(self):
        """Test that scores below 0.0 are clamped."""
        response = '{"score": -0.5, "reasoning": "Under min"}'
        score, _ = _parse_llm_score(response)
        assert score == 0.0

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns 0.0."""
        response = "This is not JSON"
        score, reasoning = _parse_llm_score(response)
        assert score == 0.0
        assert "Parse error" in reasoning

    def test_parse_missing_score(self):
        """Test parsing JSON without score field."""
        response = '{"reasoning": "No score provided"}'
        score, _ = _parse_llm_score(response)
        assert score == 0.0


# =============================================================================
# TOOL SELECTION EVALUATOR TESTS
# =============================================================================


class TestToolSelectionEvaluator:
    """Tests for the tool_selection_evaluator."""

    @pytest.mark.asyncio
    async def test_evaluator_with_matching_tools(self, mock_run, mock_example):
        """Test evaluation when actual tools match expected tools."""
        with patch(
            "requirements_graphrag_api.evaluation.agentic_evaluators._get_judge_llm"
        ) as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.ainvoke = AsyncMock(
                return_value=create_mock_llm_response(1.0, "Perfect tool selection")
            )
            mock_llm.return_value = mock_llm_instance

            result = await tool_selection_evaluator(mock_run, mock_example)

            assert result["key"] == "tool_selection"
            assert result["score"] == 1.0
            assert result["comment"] == "Perfect tool selection"

    @pytest.mark.asyncio
    async def test_evaluator_without_expected_tools(self, mock_run):
        """Test that missing expected tools returns 1.0 (skip evaluation)."""
        example = MagicMock()
        example.outputs = {}  # No expected_tools

        result = await tool_selection_evaluator(mock_run, example)

        assert result["key"] == "tool_selection"
        assert result["score"] == 1.0
        assert "No expected tools" in result["comment"]

    @pytest.mark.asyncio
    async def test_evaluator_without_example(self, mock_run):
        """Test evaluation without example."""
        result = await tool_selection_evaluator(mock_run, None)

        assert result["key"] == "tool_selection"
        assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_evaluator_extracts_tools_from_run_name(self):
        """Test that tools can be inferred from run name."""
        run = MagicMock()
        run.name = "graph_search_node"
        run.inputs = {"question": "Test question"}
        run.outputs = {"tools_used": [], "tool_order": []}

        example = MagicMock()
        example.outputs = {"expected_tools": ["graph_search"]}

        with patch(
            "requirements_graphrag_api.evaluation.agentic_evaluators._get_judge_llm"
        ) as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.ainvoke = AsyncMock(
                return_value=create_mock_llm_response(0.75, "Inferred from run name")
            )
            mock_llm.return_value = mock_llm_instance

            result = await tool_selection_evaluator(run, example)

            assert result["key"] == "tool_selection"
            # Verify the LLM was called
            mock_llm_instance.ainvoke.assert_called_once()


# =============================================================================
# ITERATION EFFICIENCY EVALUATOR TESTS
# =============================================================================


class TestIterationEfficiencyEvaluator:
    """Tests for the iteration_efficiency_evaluator."""

    @pytest.mark.asyncio
    async def test_evaluator_optimal_iterations(self, mock_run, mock_example):
        """Test evaluation when iterations match expected."""
        with patch(
            "requirements_graphrag_api.evaluation.agentic_evaluators._get_judge_llm"
        ) as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.ainvoke = AsyncMock(
                return_value=create_mock_llm_response(1.0, "Optimal iterations")
            )
            mock_llm.return_value = mock_llm_instance

            result = await iteration_efficiency_evaluator(mock_run, mock_example)

            assert result["key"] == "iteration_efficiency"
            assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_evaluator_uses_complexity_defaults(self, mock_run):
        """Test that complexity defaults are used when expected_iterations not set."""
        example = MagicMock()
        example.inputs = {"complexity": "complex"}
        example.outputs = {}  # No expected_iterations

        with patch(
            "requirements_graphrag_api.evaluation.agentic_evaluators._get_judge_llm"
        ) as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.ainvoke = AsyncMock(
                return_value=create_mock_llm_response(0.5, "More iterations needed")
            )
            mock_llm.return_value = mock_llm_instance

            result = await iteration_efficiency_evaluator(mock_run, example)

            assert result["key"] == "iteration_efficiency"
            # Verify the prompt includes complexity
            call_args = mock_llm_instance.ainvoke.call_args[0][0]
            assert "complex" in call_args

    @pytest.mark.asyncio
    async def test_evaluator_defaults_without_example(self, mock_run):
        """Test evaluation without example uses defaults."""
        with patch(
            "requirements_graphrag_api.evaluation.agentic_evaluators._get_judge_llm"
        ) as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.ainvoke = AsyncMock(
                return_value=create_mock_llm_response(0.75, "Using defaults")
            )
            mock_llm.return_value = mock_llm_instance

            result = await iteration_efficiency_evaluator(mock_run, None)

            assert result["key"] == "iteration_efficiency"
            assert result["score"] == 0.75


# =============================================================================
# CRITIC CALIBRATION EVALUATOR TESTS
# =============================================================================


class TestCriticCalibrationEvaluator:
    """Tests for the critic_calibration_evaluator."""

    @pytest.mark.asyncio
    async def test_evaluator_well_calibrated(self, mock_run, mock_example):
        """Test evaluation when critic is well-calibrated."""
        with patch(
            "requirements_graphrag_api.evaluation.agentic_evaluators._get_judge_llm"
        ) as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.ainvoke = AsyncMock(
                return_value=create_mock_llm_response(0.9, "Well calibrated")
            )
            mock_llm.return_value = mock_llm_instance

            result = await critic_calibration_evaluator(mock_run, mock_example)

            assert result["key"] == "critic_calibration"
            assert result["score"] == 0.9

    @pytest.mark.asyncio
    async def test_evaluator_handles_dict_critique(self, mock_run, mock_example):
        """Test that dict critique is properly extracted."""
        mock_run.outputs["critique"] = {
            "confidence": 0.7,
            "completeness": "partial",
            "missing_aspects": ["additional context needed"],
        }

        with patch(
            "requirements_graphrag_api.evaluation.agentic_evaluators._get_judge_llm"
        ) as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.ainvoke = AsyncMock(
                return_value=create_mock_llm_response(0.6, "Partial calibration")
            )
            mock_llm.return_value = mock_llm_instance

            _ = await critic_calibration_evaluator(mock_run, mock_example)

            # Verify the prompt includes the critique values
            call_args = mock_llm_instance.ainvoke.call_args[0][0]
            assert "0.7" in call_args
            assert "partial" in call_args

    @pytest.mark.asyncio
    async def test_evaluator_handles_non_dict_critique(self, mock_run, mock_example):
        """Test handling of non-dict critique."""
        mock_run.outputs["critique"] = "String critique"

        with patch(
            "requirements_graphrag_api.evaluation.agentic_evaluators._get_judge_llm"
        ) as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.ainvoke = AsyncMock(
                return_value=create_mock_llm_response(0.5, "Default values used")
            )
            mock_llm.return_value = mock_llm_instance

            result = await critic_calibration_evaluator(mock_run, mock_example)

            assert result["key"] == "critic_calibration"


# =============================================================================
# MULTI-HOP REASONING EVALUATOR TESTS
# =============================================================================


class TestMultiHopReasoningEvaluator:
    """Tests for the multi_hop_reasoning_evaluator."""

    @pytest.mark.asyncio
    async def test_evaluator_good_reasoning(self, mock_run, mock_example):
        """Test evaluation with good multi-hop reasoning."""
        with patch(
            "requirements_graphrag_api.evaluation.agentic_evaluators._get_judge_llm"
        ) as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.ainvoke = AsyncMock(
                return_value=create_mock_llm_response(0.85, "Good reasoning chain")
            )
            mock_llm.return_value = mock_llm_instance

            result = await multi_hop_reasoning_evaluator(mock_run, mock_example)

            assert result["key"] == "multi_hop_reasoning"
            assert result["score"] == 0.85

    @pytest.mark.asyncio
    async def test_evaluator_no_ground_truth(self, mock_run):
        """Test that missing ground truth returns 1.0 (skip evaluation)."""
        example = MagicMock()
        example.outputs = {}  # No expected_answer

        result = await multi_hop_reasoning_evaluator(mock_run, example)

        assert result["key"] == "multi_hop_reasoning"
        assert result["score"] == 1.0
        assert "No ground truth" in result["comment"]

    @pytest.mark.asyncio
    async def test_evaluator_includes_reasoning_chain(self, mock_run, mock_example):
        """Test that reasoning chain is included in evaluation."""
        with patch(
            "requirements_graphrag_api.evaluation.agentic_evaluators._get_judge_llm"
        ) as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.ainvoke = AsyncMock(
                return_value=create_mock_llm_response(0.75, "Partial chain")
            )
            mock_llm.return_value = mock_llm_instance

            _ = await multi_hop_reasoning_evaluator(mock_run, mock_example)

            # Verify the prompt includes reasoning chain
            call_args = mock_llm_instance.ainvoke.call_args[0][0]
            assert "Identified query" in call_args or "reasoning chain" in call_args.lower()

    @pytest.mark.asyncio
    async def test_evaluator_handles_empty_reasoning_chain(self, mock_run, mock_example):
        """Test handling of empty reasoning chain."""
        mock_run.outputs["reasoning_chain"] = []

        with patch(
            "requirements_graphrag_api.evaluation.agentic_evaluators._get_judge_llm"
        ) as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.ainvoke = AsyncMock(
                return_value=create_mock_llm_response(0.3, "No reasoning recorded")
            )
            mock_llm.return_value = mock_llm_instance

            _ = await multi_hop_reasoning_evaluator(mock_run, mock_example)

            # Verify the prompt includes "Not recorded" for empty chain
            call_args = mock_llm_instance.ainvoke.call_args[0][0]
            assert "Not recorded" in call_args


# =============================================================================
# SYNC WRAPPER TESTS
# =============================================================================


class TestSyncWrappers:
    """Tests for synchronous wrapper functions."""

    def test_tool_selection_sync_exists(self):
        """Test that sync wrapper is importable."""
        from requirements_graphrag_api.evaluation.agentic_evaluators import (
            tool_selection_evaluator_sync,
        )

        assert callable(tool_selection_evaluator_sync)

    def test_iteration_efficiency_sync_exists(self):
        """Test that sync wrapper is importable."""
        from requirements_graphrag_api.evaluation.agentic_evaluators import (
            iteration_efficiency_evaluator_sync,
        )

        assert callable(iteration_efficiency_evaluator_sync)

    def test_critic_calibration_sync_exists(self):
        """Test that sync wrapper is importable."""
        from requirements_graphrag_api.evaluation.agentic_evaluators import (
            critic_calibration_evaluator_sync,
        )

        assert callable(critic_calibration_evaluator_sync)

    def test_multi_hop_reasoning_sync_exists(self):
        """Test that sync wrapper is importable."""
        from requirements_graphrag_api.evaluation.agentic_evaluators import (
            multi_hop_reasoning_evaluator_sync,
        )

        assert callable(multi_hop_reasoning_evaluator_sync)


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


class TestEvaluatorIntegration:
    """Integration-style tests for evaluator imports and exports."""

    def test_evaluators_exported_from_package(self):
        """Test that evaluators are exported from the evaluation package."""
        from requirements_graphrag_api.evaluation import (
            critic_calibration_evaluator,
            critic_calibration_evaluator_sync,
            iteration_efficiency_evaluator,
            iteration_efficiency_evaluator_sync,
            multi_hop_reasoning_evaluator,
            multi_hop_reasoning_evaluator_sync,
            tool_selection_evaluator,
            tool_selection_evaluator_sync,
        )

        # Verify all are callable
        assert callable(tool_selection_evaluator)
        assert callable(tool_selection_evaluator_sync)
        assert callable(iteration_efficiency_evaluator)
        assert callable(iteration_efficiency_evaluator_sync)
        assert callable(critic_calibration_evaluator)
        assert callable(critic_calibration_evaluator_sync)
        assert callable(multi_hop_reasoning_evaluator)
        assert callable(multi_hop_reasoning_evaluator_sync)

    def test_evaluator_return_format(self, mock_run, mock_example):
        """Test that evaluators return correct format for LangSmith."""
        import asyncio

        with patch(
            "requirements_graphrag_api.evaluation.agentic_evaluators._get_judge_llm"
        ) as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.ainvoke = AsyncMock(
                return_value=create_mock_llm_response(0.8, "Test reasoning")
            )
            mock_llm.return_value = mock_llm_instance

            result = asyncio.get_event_loop().run_until_complete(
                tool_selection_evaluator(mock_run, mock_example)
            )

            # Verify LangSmith-compatible format
            assert "key" in result
            assert "score" in result
            assert "comment" in result
            assert isinstance(result["key"], str)
            assert isinstance(result["score"], float)
            assert 0.0 <= result["score"] <= 1.0
            assert isinstance(result["comment"], str)
