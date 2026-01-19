"""Tests for step-back prompting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jama_mcp_server_graphrag.agentic.stepback import generate_stepback_query
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
# Generate Stepback Query Tests
# =============================================================================


class TestGenerateStepbackQuery:
    """Tests for generate_stepback_query function."""

    @pytest.mark.asyncio
    async def test_generates_broader_query(self, mock_config: MagicMock) -> None:
        """Test that stepback generates a broader query."""
        with patch("jama_mcp_server_graphrag.agentic.stepback.ChatOpenAI") as mock_llm_class:
            mock_llm_class.return_value = create_llm_mock(
                "What are the automotive industry safety standards?"
            )

            result = await generate_stepback_query(
                mock_config,
                "What ISO standard applies to automotive functional safety for ASIL-D?",
            )

            assert isinstance(result, str)
            assert len(result) > 0
            # Broader query should not contain specific details like "ASIL-D"
            assert "ASIL-D" not in result

    @pytest.mark.asyncio
    async def test_strips_whitespace(self, mock_config: MagicMock) -> None:
        """Test that result is stripped of whitespace."""
        with patch("jama_mcp_server_graphrag.agentic.stepback.ChatOpenAI") as mock_llm_class:
            mock_llm_class.return_value = create_llm_mock(
                "  What are requirements best practices?  \n"
            )

            result = await generate_stepback_query(mock_config, "Specific question")

            assert result == "What are requirements best practices?"
            assert not result.startswith(" ")
            assert not result.endswith(" ")

    @pytest.mark.asyncio
    async def test_handles_already_broad_query(self, mock_config: MagicMock) -> None:
        """Test handling of queries that are already broad."""
        with patch("jama_mcp_server_graphrag.agentic.stepback.ChatOpenAI") as mock_llm_class:
            # LLM might return the same query if it's already broad
            mock_llm_class.return_value = create_llm_mock("What is requirements management?")

            result = await generate_stepback_query(mock_config, "What is requirements management?")

            assert isinstance(result, str)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_uses_correct_model(self, mock_config: MagicMock) -> None:
        """Test that the configured chat model is used."""
        mock_config.chat_model = "gpt-4-turbo"

        with patch("jama_mcp_server_graphrag.agentic.stepback.ChatOpenAI") as mock_llm_class:
            mock_llm_class.return_value = create_llm_mock("Broader question")

            await generate_stepback_query(mock_config, "Test")

            mock_llm_class.assert_called_once()
            call_kwargs = mock_llm_class.call_args[1]
            assert call_kwargs["model"] == "gpt-4-turbo"
