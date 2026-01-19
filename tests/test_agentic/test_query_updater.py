"""Tests for query updater."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jama_mcp_server_graphrag.agentic.query_updater import update_query
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


@pytest.fixture
def sample_previous_answers() -> list[dict[str, str]]:
    """Create sample previous Q&A pairs."""
    return [
        {
            "question": "What standards apply to automotive?",
            "answer": "ISO 26262 is the key standard for automotive functional safety.",
        },
        {
            "question": "What is ASIL?",
            "answer": "ASIL stands for Automotive Safety Integrity Level, with levels A through D.",
        },
    ]


# =============================================================================
# Update Query Tests
# =============================================================================


class TestUpdateQuery:
    """Tests for update_query function."""

    @pytest.mark.asyncio
    async def test_returns_original_when_no_history(self, mock_config: MagicMock) -> None:
        """Test that original query is returned when no previous answers."""
        result = await update_query(mock_config, "What are the requirements?", previous_answers=[])

        assert result == "What are the requirements?"

    @pytest.mark.asyncio
    async def test_updates_query_with_context(
        self,
        mock_config: MagicMock,
        sample_previous_answers: list[dict[str, str]],
    ) -> None:
        """Test that query is updated with context from previous answers."""
        with patch("jama_mcp_server_graphrag.agentic.query_updater.ChatOpenAI") as mock_llm_class:
            mock_llm_class.return_value = create_llm_mock(
                "What are the ISO 26262 ASIL-D requirements for automotive safety?"
            )

            result = await update_query(
                mock_config,
                "What are the highest level requirements?",
                previous_answers=sample_previous_answers,
            )

            assert isinstance(result, str)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_strips_whitespace(
        self,
        mock_config: MagicMock,
        sample_previous_answers: list[dict[str, str]],
    ) -> None:
        """Test that result is stripped of whitespace."""
        with patch("jama_mcp_server_graphrag.agentic.query_updater.ChatOpenAI") as mock_llm_class:
            mock_llm_class.return_value = create_llm_mock("  Updated question?  \n")

            result = await update_query(
                mock_config, "Original?", previous_answers=sample_previous_answers
            )

            assert result == "Updated question?"

    @pytest.mark.asyncio
    async def test_formats_previous_answers(
        self,
        mock_config: MagicMock,
        sample_previous_answers: list[dict[str, str]],
    ) -> None:
        """Test that previous answers are properly formatted and LLM is invoked."""
        with patch("jama_mcp_server_graphrag.agentic.query_updater.ChatOpenAI") as mock_llm_class:
            mock_llm_class.return_value = create_llm_mock("Updated question with ISO 26262 context")

            result = await update_query(
                mock_config, "Test", previous_answers=sample_previous_answers
            )

            # Verify the LLM was called and returned the expected result
            assert isinstance(result, str)
            assert len(result) > 0
            mock_llm_class.assert_called_once()  # ChatOpenAI was instantiated

    @pytest.mark.asyncio
    async def test_handles_single_previous_answer(self, mock_config: MagicMock) -> None:
        """Test handling of a single previous answer."""
        with patch("jama_mcp_server_graphrag.agentic.query_updater.ChatOpenAI") as mock_llm_class:
            mock_llm_class.return_value = create_llm_mock("What is ISO 26262 traceability?")

            result = await update_query(
                mock_config,
                "What is traceability for this standard?",
                previous_answers=[
                    {
                        "question": "What standard applies?",
                        "answer": "ISO 26262 applies.",
                    }
                ],
            )

            assert isinstance(result, str)
            assert "ISO 26262" in result
