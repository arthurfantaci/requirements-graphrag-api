"""Tests for retriever router."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jama_mcp_server_graphrag.agentic.router import (
    RETRIEVER_TOOLS,
    RoutingResult,
    route_query,
)
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
# Router Constants Tests
# =============================================================================


class TestRouterConstants:
    """Tests for router constants."""

    def test_retriever_tools_defined(self) -> None:
        """Test that all retriever tools are defined."""
        expected_tools = [
            "graphrag_vector_search",
            "graphrag_hybrid_search",
            "graphrag_graph_enriched_search",
            "graphrag_explore_entity",
            "graphrag_lookup_standard",
            "graphrag_lookup_term",
            "graphrag_text2cypher",
            "graphrag_chat",
        ]

        for tool in expected_tools:
            assert tool in RETRIEVER_TOOLS

    def test_retriever_tools_have_descriptions(self) -> None:
        """Test that all tools have non-empty descriptions."""
        for tool, description in RETRIEVER_TOOLS.items():
            assert description.strip(), f"Tool {tool} has empty description"


# =============================================================================
# Route Query Tests
# =============================================================================


class TestRouteQuery:
    """Tests for route_query function."""

    @pytest.mark.asyncio
    async def test_route_query_returns_routing_result(self, mock_config: MagicMock) -> None:
        """Test that route_query returns a RoutingResult."""
        response = (
            '{"selected_tools": ["graphrag_chat"], '
            '"reasoning": "Complex question", "tool_params": {}}'
        )
        with patch("jama_mcp_server_graphrag.agentic.router.ChatOpenAI") as mock_llm_class:
            mock_llm_class.return_value = create_llm_mock(response)

            result = await route_query(mock_config, "What is requirements traceability?")

            assert isinstance(result, RoutingResult)
            assert len(result.selected_tools) > 0
            assert result.reasoning != ""

    @pytest.mark.asyncio
    async def test_route_query_selects_appropriate_tool(self, mock_config: MagicMock) -> None:
        """Test that router selects appropriate tools based on query."""
        response = (
            '{"selected_tools": ["graphrag_lookup_term"], '
            '"reasoning": "Definition lookup", '
            '"tool_params": {"graphrag_lookup_term": {"query": "baseline"}}}'
        )
        with patch("jama_mcp_server_graphrag.agentic.router.ChatOpenAI") as mock_llm_class:
            mock_llm_class.return_value = create_llm_mock(response)

            result = await route_query(mock_config, "Define baseline")

            assert "graphrag_lookup_term" in result.selected_tools

    @pytest.mark.asyncio
    async def test_route_query_handles_invalid_json(self, mock_config: MagicMock) -> None:
        """Test that invalid JSON responses default to chat."""
        with patch("jama_mcp_server_graphrag.agentic.router.ChatOpenAI") as mock_llm_class:
            mock_llm_class.return_value = create_llm_mock("This is not valid JSON")

            result = await route_query(mock_config, "Test question")

            # Should default to chat
            assert "graphrag_chat" in result.selected_tools

    @pytest.mark.asyncio
    async def test_route_query_strips_markdown(self, mock_config: MagicMock) -> None:
        """Test that markdown code blocks are stripped from response."""
        response = (
            '```json\n{"selected_tools": ["graphrag_vector_search"], '
            '"reasoning": "Simple search", "tool_params": {}}\n```'
        )
        with patch("jama_mcp_server_graphrag.agentic.router.ChatOpenAI") as mock_llm_class:
            mock_llm_class.return_value = create_llm_mock(response)

            result = await route_query(mock_config, "Find info about requirements")

            assert "graphrag_vector_search" in result.selected_tools

    @pytest.mark.asyncio
    async def test_route_query_preserves_raw_response(self, mock_config: MagicMock) -> None:
        """Test that raw LLM response is preserved."""
        raw_response = (
            '{"selected_tools": ["graphrag_chat"], "reasoning": "Test", "tool_params": {}}'
        )

        with patch("jama_mcp_server_graphrag.agentic.router.ChatOpenAI") as mock_llm_class:
            mock_llm_class.return_value = create_llm_mock(raw_response)

            result = await route_query(mock_config, "Test")

            assert result.raw_response == raw_response
