"""Tests for evaluation metrics module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jama_mcp_server_graphrag.config import AppConfig
from jama_mcp_server_graphrag.evaluation.metrics import (
    RAGMetrics,
    _parse_score,
    compute_answer_relevancy,
    compute_context_precision,
    compute_context_recall,
    compute_faithfulness,
)

_TEST_PASSWORD = "test"  # noqa: S105


@pytest.fixture
def mock_config() -> AppConfig:
    """Create a mock config for testing."""
    return AppConfig(
        neo4j_uri="neo4j://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password=_TEST_PASSWORD,
        openai_api_key="sk-test",
        chat_model="gpt-4o",
    )


class TestRAGMetrics:
    """Tests for RAGMetrics dataclass."""

    def test_metrics_creation(self) -> None:
        """Test creating metrics object."""
        metrics = RAGMetrics(
            faithfulness=0.9,
            answer_relevancy=0.85,
            context_precision=0.8,
            context_recall=0.75,
        )

        assert metrics.faithfulness == 0.9
        assert metrics.answer_relevancy == 0.85
        assert metrics.context_precision == 0.8
        assert metrics.context_recall == 0.75

    def test_average_calculation(self) -> None:
        """Test average score calculation."""
        metrics = RAGMetrics(
            faithfulness=1.0,
            answer_relevancy=0.8,
            context_precision=0.6,
            context_recall=0.4,
        )

        assert metrics.average == 0.7  # (1.0 + 0.8 + 0.6 + 0.4) / 4

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = RAGMetrics(
            faithfulness=0.9,
            answer_relevancy=0.85,
            context_precision=0.8,
            context_recall=0.75,
        )

        result = metrics.to_dict()

        assert result["faithfulness"] == 0.9
        assert result["answer_relevancy"] == 0.85
        assert result["context_precision"] == 0.8
        assert result["context_recall"] == 0.75
        assert "average" in result


class TestParseScore:
    """Tests for _parse_score function."""

    def test_parse_valid_score(self) -> None:
        """Test parsing a valid score."""
        assert _parse_score("0.85") == 0.85
        assert _parse_score("1.0") == 1.0
        assert _parse_score("0") == 0.0

    def test_parse_with_whitespace(self) -> None:
        """Test parsing score with whitespace."""
        assert _parse_score("  0.75  ") == 0.75

    def test_parse_clamps_above_one(self) -> None:
        """Test that scores above 1 are clamped."""
        assert _parse_score("1.5") == 1.0

    def test_parse_clamps_below_zero(self) -> None:
        """Test that scores below 0 are clamped."""
        assert _parse_score("-0.5") == 0.0

    def test_parse_invalid_returns_default(self) -> None:
        """Test that invalid input returns 0.5 default."""
        assert _parse_score("not a number") == 0.5
        assert _parse_score("") == 0.5


class TestComputeFaithfulness:
    """Tests for compute_faithfulness function."""

    @pytest.mark.asyncio
    async def test_returns_score(self, mock_config: AppConfig) -> None:
        """Test that compute_faithfulness returns a score."""
        with patch(
            "jama_mcp_server_graphrag.evaluation.metrics.ChatOpenAI"
        ) as mock_chat:
            # Create mock chain that returns a string when awaited
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(return_value="0.85")

            # Create mock LLM that returns the chain when piped
            mock_llm = MagicMock()
            mock_llm.__or__ = MagicMock(return_value=mock_chain)
            mock_chat.return_value = mock_llm

            score = await compute_faithfulness(
                mock_config,
                "What is traceability?",
                "Traceability is tracking requirements.",
                ["Requirements can be traced through their lifecycle."],
            )

            assert 0.0 <= score <= 1.0


class TestComputeAnswerRelevancy:
    """Tests for compute_answer_relevancy function."""

    @pytest.mark.asyncio
    async def test_returns_score(self, mock_config: AppConfig) -> None:
        """Test that compute_answer_relevancy returns a score."""
        with patch(
            "jama_mcp_server_graphrag.evaluation.metrics.ChatOpenAI"
        ) as mock_chat:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(return_value="0.9")

            mock_llm = MagicMock()
            mock_llm.__or__ = MagicMock(return_value=mock_chain)
            mock_chat.return_value = mock_llm

            score = await compute_answer_relevancy(
                mock_config,
                "What is traceability?",
                "Traceability is tracking requirements through their lifecycle.",
            )

            assert 0.0 <= score <= 1.0


class TestComputeContextPrecision:
    """Tests for compute_context_precision function."""

    @pytest.mark.asyncio
    async def test_returns_score(self, mock_config: AppConfig) -> None:
        """Test that compute_context_precision returns a score."""
        with patch(
            "jama_mcp_server_graphrag.evaluation.metrics.ChatOpenAI"
        ) as mock_chat:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(return_value="0.8")

            mock_llm = MagicMock()
            mock_llm.__or__ = MagicMock(return_value=mock_chain)
            mock_chat.return_value = mock_llm

            score = await compute_context_precision(
                mock_config,
                "What is traceability?",
                ["Context about traceability", "Relevant context"],
            )

            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_empty_contexts_returns_zero(self, mock_config: AppConfig) -> None:
        """Test that empty contexts return 0."""
        score = await compute_context_precision(
            mock_config,
            "What is traceability?",
            [],
        )

        assert score == 0.0


class TestComputeContextRecall:
    """Tests for compute_context_recall function."""

    @pytest.mark.asyncio
    async def test_returns_score(self, mock_config: AppConfig) -> None:
        """Test that compute_context_recall returns a score."""
        with patch(
            "jama_mcp_server_graphrag.evaluation.metrics.ChatOpenAI"
        ) as mock_chat:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(return_value="0.75")

            mock_llm = MagicMock()
            mock_llm.__or__ = MagicMock(return_value=mock_chain)
            mock_chat.return_value = mock_llm

            score = await compute_context_recall(
                mock_config,
                "What is traceability?",
                ["Context about traceability"],
                "Traceability is the ability to track requirements.",
            )

            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_empty_contexts_returns_zero(self, mock_config: AppConfig) -> None:
        """Test that empty contexts return 0."""
        score = await compute_context_recall(
            mock_config,
            "What is traceability?",
            [],
            "Traceability is important.",
        )

        assert score == 0.0
