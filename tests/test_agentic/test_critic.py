"""Tests for answer critic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jama_mcp_server_graphrag.agentic.critic import CritiqueResult, critique_answer
from tests.conftest import create_llm_mock

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock AppConfig."""
    config = MagicMock()
    config.chat_model = "gpt-4o"
    config.openai_api_key = "test-api-key"
    return config


# =============================================================================
# Critique Result Tests
# =============================================================================


class TestCritiqueResult:
    """Tests for CritiqueResult dataclass."""

    def test_critique_result_creation(self) -> None:
        """Test creating a CritiqueResult."""
        result = CritiqueResult(
            answerable=True,
            confidence=0.85,
            completeness="complete",
            missing_aspects=[],
            followup_query=None,
            reasoning="Context is sufficient",
        )

        assert result.answerable is True
        assert result.confidence == 0.85
        assert result.completeness == "complete"
        assert result.missing_aspects == []
        assert result.followup_query is None

    def test_critique_result_with_followup(self) -> None:
        """Test CritiqueResult with follow-up query."""
        result = CritiqueResult(
            answerable=False,
            confidence=0.3,
            completeness="insufficient",
            missing_aspects=["implementation details", "examples"],
            followup_query="What are specific examples of traceability?",
            reasoning="Context lacks specific examples",
        )

        assert result.answerable is False
        assert len(result.missing_aspects) == 2
        assert result.followup_query is not None


# =============================================================================
# Critique Answer Tests
# =============================================================================


class TestCritiqueAnswer:
    """Tests for critique_answer function."""

    @pytest.mark.asyncio
    async def test_critique_returns_result(self, mock_config: MagicMock) -> None:
        """Test that critique_answer returns a CritiqueResult."""
        response = (
            '{"answerable": true, "confidence": 0.9, "completeness": "complete", '
            '"missing_aspects": [], "followup_query": null, "reasoning": "Good context"}'
        )
        with patch("jama_mcp_server_graphrag.agentic.critic.ChatOpenAI") as mock_llm_class:
            mock_llm_class.return_value = create_llm_mock(response)

            result = await critique_answer(
                mock_config,
                "What is traceability?",
                "Traceability is the ability to trace requirements...",
            )

            assert isinstance(result, CritiqueResult)
            assert result.answerable is True
            assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_critique_identifies_insufficient_context(self, mock_config: MagicMock) -> None:
        """Test critique identifies when context is insufficient."""
        response = (
            '{"answerable": false, "confidence": 0.2, "completeness": "insufficient", '
            '"missing_aspects": ["FDA regulations"], '
            '"followup_query": "What are FDA requirements?", "reasoning": "No FDA info"}'
        )
        with patch("jama_mcp_server_graphrag.agentic.critic.ChatOpenAI") as mock_llm_class:
            mock_llm_class.return_value = create_llm_mock(response)

            result = await critique_answer(
                mock_config,
                "What FDA regulations apply?",
                "This is about automotive standards.",
            )

            assert result.answerable is False
            assert result.completeness == "insufficient"
            assert "FDA regulations" in result.missing_aspects

    @pytest.mark.asyncio
    async def test_critique_handles_invalid_json(self, mock_config: MagicMock) -> None:
        """Test that invalid JSON returns conservative defaults."""
        with patch("jama_mcp_server_graphrag.agentic.critic.ChatOpenAI") as mock_llm_class:
            mock_llm_class.return_value = create_llm_mock("Invalid JSON response")

            result = await critique_answer(mock_config, "Test", "Context")

            # Should return conservative defaults
            assert result.answerable is True  # Allow generation
            assert result.confidence == 0.5  # Low confidence
            assert result.completeness == "partial"

    @pytest.mark.asyncio
    async def test_critique_strips_markdown(self, mock_config: MagicMock) -> None:
        """Test that markdown code blocks are stripped."""
        response = (
            '```json\n{"answerable": true, "confidence": 0.8, '
            '"completeness": "complete", "missing_aspects": [], '
            '"followup_query": null, "reasoning": "OK"}\n```'
        )
        with patch("jama_mcp_server_graphrag.agentic.critic.ChatOpenAI") as mock_llm_class:
            mock_llm_class.return_value = create_llm_mock(response)

            result = await critique_answer(mock_config, "Test", "Context")

            assert result.answerable is True
            assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_critique_preserves_raw_response(self, mock_config: MagicMock) -> None:
        """Test that raw LLM response is preserved."""
        raw_response = (
            '{"answerable": true, "confidence": 0.9, "completeness": "complete", '
            '"missing_aspects": [], "followup_query": null, "reasoning": "Test"}'
        )

        with patch("jama_mcp_server_graphrag.agentic.critic.ChatOpenAI") as mock_llm_class:
            mock_llm_class.return_value = create_llm_mock(raw_response)

            result = await critique_answer(mock_config, "Test", "Context")

            assert result.raw_response == raw_response
