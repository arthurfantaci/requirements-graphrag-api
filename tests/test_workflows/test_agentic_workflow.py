"""Tests for agentic RAG workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from jama_mcp_server_graphrag.workflows.agentic_workflow import (
    critique_node,
    refine_query_node,
    retrieve_node,
    route_node,
    should_refine,
)

if TYPE_CHECKING:
    from jama_mcp_server_graphrag.workflows.state import AgenticState

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock AppConfig."""
    config = MagicMock()
    config.chat_model = "gpt-4o"
    config.openai_api_key = "test-key"
    return config


@pytest.fixture
def mock_graph() -> MagicMock:
    """Create a mock Neo4jGraph."""
    return MagicMock()


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Create a mock Neo4jVector store."""
    return MagicMock()


@pytest.fixture
def initial_state() -> AgenticState:
    """Create an initial agentic state."""
    return {
        "question": "What is requirements traceability?",
        "refined_question": "",
        "selected_tools": [],
        "routing_reasoning": "",
        "documents": [],
        "context": "",
        "critique_result": None,
        "needs_more_context": False,
        "followup_query": None,
        "iteration": 0,
        "max_iterations": 3,
        "answer": "",
        "sources": [],
        "confidence": 0.0,
        "error": None,
    }


# =============================================================================
# Route Node Tests
# =============================================================================


class TestRouteNode:
    """Tests for route_node function."""

    @pytest.mark.asyncio
    async def test_route_node_success(
        self, initial_state: AgenticState, mock_config: MagicMock
    ) -> None:
        """Test successful routing."""
        with patch("jama_mcp_server_graphrag.workflows.agentic_workflow.route_query") as mock_route:
            mock_result = MagicMock()
            mock_result.selected_tools = ["graphrag_hybrid_search"]
            mock_result.reasoning = "Hybrid search for technical query"
            mock_route.return_value = mock_result

            result = await route_node(initial_state, config=mock_config)

            assert result["selected_tools"] == ["graphrag_hybrid_search"]
            assert "Hybrid search" in result["routing_reasoning"]

    @pytest.mark.asyncio
    async def test_route_node_error_defaults_to_hybrid(
        self, initial_state: AgenticState, mock_config: MagicMock
    ) -> None:
        """Test that routing errors default to hybrid search."""
        with patch("jama_mcp_server_graphrag.workflows.agentic_workflow.route_query") as mock_route:
            mock_route.side_effect = Exception("Routing failed")

            result = await route_node(initial_state, config=mock_config)

            assert result["selected_tools"] == ["graphrag_hybrid_search"]
            assert "defaulting to hybrid" in result["routing_reasoning"]


# =============================================================================
# Retrieve Node Tests
# =============================================================================


class TestRetrieveNode:
    """Tests for retrieve_node function."""

    @pytest.mark.asyncio
    async def test_retrieve_node_vector_search(
        self,
        initial_state: AgenticState,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test retrieval with vector search."""
        initial_state["selected_tools"] = ["graphrag_vector_search"]

        with patch(
            "jama_mcp_server_graphrag.workflows.agentic_workflow.vector_search"
        ) as mock_search:
            mock_search.return_value = [{"content": "Test", "score": 0.9, "metadata": {}}]

            result = await retrieve_node(
                initial_state, graph=mock_graph, vector_store=mock_vector_store
            )

            assert "documents" in result
            assert len(result["documents"]) == 1

    @pytest.mark.asyncio
    async def test_retrieve_node_hybrid_search(
        self,
        initial_state: AgenticState,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test retrieval with hybrid search."""
        initial_state["selected_tools"] = ["graphrag_hybrid_search"]

        with patch(
            "jama_mcp_server_graphrag.workflows.agentic_workflow.hybrid_search"
        ) as mock_search:
            mock_search.return_value = [{"content": "Test", "score": 0.9, "metadata": {}}]

            result = await retrieve_node(
                initial_state, graph=mock_graph, vector_store=mock_vector_store
            )

            assert "documents" in result
            mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_node_deduplicates(
        self,
        initial_state: AgenticState,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test that duplicate documents are deduplicated."""
        initial_state["selected_tools"] = [
            "graphrag_vector_search",
            "graphrag_hybrid_search",
        ]

        with (
            patch(
                "jama_mcp_server_graphrag.workflows.agentic_workflow.vector_search"
            ) as mock_vector,
            patch(
                "jama_mcp_server_graphrag.workflows.agentic_workflow.hybrid_search"
            ) as mock_hybrid,
        ):
            # Same content from both searches
            mock_vector.return_value = [
                {"content": "Same content here", "score": 0.8, "metadata": {}}
            ]
            mock_hybrid.return_value = [
                {"content": "Same content here", "score": 0.9, "metadata": {}}
            ]

            result = await retrieve_node(
                initial_state, graph=mock_graph, vector_store=mock_vector_store
            )

            # Should only have 1 document (deduplicated, higher score kept)
            assert len(result["documents"]) == 1
            assert result["documents"][0]["score"] == 0.9


# =============================================================================
# Critique Node Tests
# =============================================================================


class TestCritiqueNode:
    """Tests for critique_node function."""

    @pytest.mark.asyncio
    async def test_critique_node_sufficient_context(
        self, initial_state: AgenticState, mock_config: MagicMock
    ) -> None:
        """Test critique with sufficient context."""
        initial_state["context"] = "Detailed traceability information..."

        with patch(
            "jama_mcp_server_graphrag.workflows.agentic_workflow.critique_answer"
        ) as mock_critique:
            mock_result = MagicMock()
            mock_result.answerable = True
            mock_result.confidence = 0.9
            mock_result.completeness = "complete"
            mock_result.missing_aspects = []
            mock_result.followup_query = None
            mock_critique.return_value = mock_result

            result = await critique_node(initial_state, config=mock_config)

            assert result["needs_more_context"] is False
            assert result["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_critique_node_insufficient_context(
        self, initial_state: AgenticState, mock_config: MagicMock
    ) -> None:
        """Test critique with insufficient context."""
        initial_state["context"] = "Brief mention of traceability."

        with patch(
            "jama_mcp_server_graphrag.workflows.agentic_workflow.critique_answer"
        ) as mock_critique:
            mock_result = MagicMock()
            mock_result.answerable = False
            mock_result.confidence = 0.3
            mock_result.completeness = "insufficient"
            mock_result.missing_aspects = ["implementation details"]
            mock_result.followup_query = "What are traceability implementation details?"
            mock_critique.return_value = mock_result

            result = await critique_node(initial_state, config=mock_config)

            assert result["needs_more_context"] is True
            assert result["followup_query"] is not None

    @pytest.mark.asyncio
    async def test_critique_node_respects_max_iterations(
        self, initial_state: AgenticState, mock_config: MagicMock
    ) -> None:
        """Test that critique respects max iterations."""
        initial_state["context"] = "Some context"
        initial_state["iteration"] = 3  # At max iterations
        initial_state["max_iterations"] = 3

        with patch(
            "jama_mcp_server_graphrag.workflows.agentic_workflow.critique_answer"
        ) as mock_critique:
            mock_result = MagicMock()
            mock_result.answerable = False
            mock_result.confidence = 0.3
            mock_result.completeness = "insufficient"
            mock_result.missing_aspects = []
            mock_result.followup_query = "Follow up"
            mock_critique.return_value = mock_result

            result = await critique_node(initial_state, config=mock_config)

            # Should not need more context even though insufficient
            assert result["needs_more_context"] is False

    @pytest.mark.asyncio
    async def test_critique_node_empty_context(
        self, initial_state: AgenticState, mock_config: MagicMock
    ) -> None:
        """Test critique with empty context."""
        initial_state["context"] = ""

        result = await critique_node(initial_state, config=mock_config)

        assert result["needs_more_context"] is True
        assert result["critique_result"]["completeness"] == "insufficient"


# =============================================================================
# Refine Query Node Tests
# =============================================================================


class TestRefineQueryNode:
    """Tests for refine_query_node function."""

    @pytest.mark.asyncio
    async def test_refine_with_followup_query(self, initial_state: AgenticState) -> None:
        """Test refinement using followup query."""
        initial_state["followup_query"] = "What are specific examples?"

        result = await refine_query_node(initial_state)

        assert result["refined_question"] == "What are specific examples?"
        assert result["iteration"] == 1

    @pytest.mark.asyncio
    async def test_refine_with_missing_aspects(self, initial_state: AgenticState) -> None:
        """Test refinement using missing aspects."""
        initial_state["followup_query"] = None
        initial_state["critique_result"] = {"missing_aspects": ["implementation", "examples"]}

        result = await refine_query_node(initial_state)

        assert "Specifically about:" in result["refined_question"]
        assert "implementation" in result["refined_question"]

    @pytest.mark.asyncio
    async def test_refine_increments_iteration(self, initial_state: AgenticState) -> None:
        """Test that refinement increments iteration count."""
        initial_state["iteration"] = 1
        initial_state["critique_result"] = {}  # Prevent NoneType error

        result = await refine_query_node(initial_state)

        assert result["iteration"] == 2


# =============================================================================
# Should Refine Tests
# =============================================================================


class TestShouldRefine:
    """Tests for should_refine function."""

    def test_should_refine_when_needs_context(self, initial_state: AgenticState) -> None:
        """Test that refinement is triggered when context is needed."""
        initial_state["needs_more_context"] = True

        result = should_refine(initial_state)

        assert result == "refine"

    def test_should_generate_when_context_sufficient(self, initial_state: AgenticState) -> None:
        """Test that generation is triggered when context is sufficient."""
        initial_state["needs_more_context"] = False

        result = should_refine(initial_state)

        assert result == "generate"
