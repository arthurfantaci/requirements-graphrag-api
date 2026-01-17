"""Tests for RAG workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from jama_mcp_server_graphrag.workflows.rag_workflow import (
    format_context_node,
    generate_node,
    retrieve_node,
    run_rag_workflow,
)

if TYPE_CHECKING:
    from jama_mcp_server_graphrag.workflows.state import RAGState

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Create a mock Neo4jVector store."""
    return MagicMock()


@pytest.fixture
def mock_graph() -> MagicMock:
    """Create a mock Neo4jGraph."""
    return MagicMock()


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock AppConfig."""
    config = MagicMock()
    config.chat_model = "gpt-4o"
    config.openai_api_key = "test-key"
    return config


@pytest.fixture
def sample_state() -> RAGState:
    """Create a sample RAG state."""
    return {
        "question": "What is requirements traceability?",
        "documents": [],
        "context": "",
        "answer": "",
        "sources": [],
        "error": None,
    }


@pytest.fixture
def state_with_documents() -> RAGState:
    """Create a state with retrieved documents."""
    return {
        "question": "What is requirements traceability?",
        "documents": [
            {
                "content": "Requirements traceability is the ability to trace requirements.",
                "score": 0.95,
                "metadata": {
                    "title": "Article 1",
                    "url": "https://example.com/1",
                    "chunk_id": "chunk-1",
                },
            },
            {
                "content": "Traceability helps with change impact analysis.",
                "score": 0.85,
                "metadata": {
                    "title": "Article 2",
                    "url": "https://example.com/2",
                    "chunk_id": "chunk-2",
                },
            },
        ],
        "context": "",
        "answer": "",
        "sources": [],
        "error": None,
    }


# =============================================================================
# Retrieve Node Tests
# =============================================================================


class TestRetrieveNode:
    """Tests for retrieve_node function."""

    @pytest.mark.asyncio
    async def test_retrieve_node_success(
        self, sample_state: RAGState, mock_vector_store: MagicMock
    ) -> None:
        """Test successful document retrieval."""
        with patch("jama_mcp_server_graphrag.workflows.rag_workflow.vector_search") as mock_search:
            mock_search.return_value = [
                {
                    "content": "Test content",
                    "score": 0.9,
                    "metadata": {"title": "Test"},
                }
            ]

            result = await retrieve_node(sample_state, vector_store=mock_vector_store, limit=6)

            assert "documents" in result
            assert len(result["documents"]) == 1
            assert result["documents"][0]["score"] == 0.9

    @pytest.mark.asyncio
    async def test_retrieve_node_error(
        self, sample_state: RAGState, mock_vector_store: MagicMock
    ) -> None:
        """Test retrieval error handling."""
        with patch("jama_mcp_server_graphrag.workflows.rag_workflow.vector_search") as mock_search:
            mock_search.side_effect = Exception("Search failed")

            result = await retrieve_node(sample_state, vector_store=mock_vector_store, limit=6)

            assert "error" in result
            assert "Search failed" in result["error"]


# =============================================================================
# Format Context Node Tests
# =============================================================================


class TestFormatContextNode:
    """Tests for format_context_node function."""

    @pytest.mark.asyncio
    async def test_format_context_with_documents(self, state_with_documents: RAGState) -> None:
        """Test formatting context from documents."""
        result = await format_context_node(state_with_documents)

        assert "context" in result
        assert "sources" in result
        assert len(result["sources"]) == 2
        assert "[1]" in result["context"]
        assert "[2]" in result["context"]

    @pytest.mark.asyncio
    async def test_format_context_empty_documents(self, sample_state: RAGState) -> None:
        """Test formatting with no documents."""
        result = await format_context_node(sample_state)

        assert result["context"] == "No relevant documents found."
        assert result["sources"] == []

    @pytest.mark.asyncio
    async def test_format_context_with_error(self, sample_state: RAGState) -> None:
        """Test that errors short-circuit formatting."""
        sample_state["error"] = "Previous error"

        result = await format_context_node(sample_state)

        assert result == {}


# =============================================================================
# Generate Node Tests
# =============================================================================


class TestGenerateNode:
    """Tests for generate_node function."""

    @pytest.mark.asyncio
    async def test_generate_node_success(
        self,
        state_with_documents: RAGState,
        mock_config: MagicMock,
        mock_graph: MagicMock,
    ) -> None:
        """Test successful answer generation."""
        state_with_documents["context"] = "Test context about traceability."

        with patch("jama_mcp_server_graphrag.workflows.rag_workflow.generate_answer") as mock_gen:
            mock_gen.return_value = {"answer": "Traceability is important."}

            result = await generate_node(state_with_documents, config=mock_config, graph=mock_graph)

            assert "answer" in result
            assert result["answer"] == "Traceability is important."

    @pytest.mark.asyncio
    async def test_generate_node_no_context(
        self,
        sample_state: RAGState,
        mock_config: MagicMock,
        mock_graph: MagicMock,
    ) -> None:
        """Test generation with empty context."""
        result = await generate_node(sample_state, config=mock_config, graph=mock_graph)

        assert "couldn't find relevant information" in result["answer"]

    @pytest.mark.asyncio
    async def test_generate_node_error(
        self,
        state_with_documents: RAGState,
        mock_config: MagicMock,
        mock_graph: MagicMock,
    ) -> None:
        """Test generation error handling."""
        state_with_documents["context"] = "Test context."

        with patch("jama_mcp_server_graphrag.workflows.rag_workflow.generate_answer") as mock_gen:
            mock_gen.side_effect = Exception("Generation failed")

            result = await generate_node(state_with_documents, config=mock_config, graph=mock_graph)

            assert "error" in result
            assert "couldn't generate" in result["answer"]


# =============================================================================
# Full Workflow Tests
# =============================================================================


class TestRunRAGWorkflow:
    """Tests for run_rag_workflow function."""

    @pytest.mark.asyncio
    async def test_run_rag_workflow_end_to_end(
        self,
        mock_config: MagicMock,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test full RAG workflow execution."""
        with (
            patch("jama_mcp_server_graphrag.workflows.rag_workflow.vector_search") as mock_search,
            patch("jama_mcp_server_graphrag.workflows.rag_workflow.generate_answer") as mock_gen,
        ):
            mock_search.return_value = [
                {
                    "content": "Traceability content",
                    "score": 0.9,
                    "metadata": {"title": "Test", "url": "http://test.com"},
                }
            ]
            mock_gen.return_value = {"answer": "The answer is..."}

            result = await run_rag_workflow(
                config=mock_config,
                graph=mock_graph,
                vector_store=mock_vector_store,
                question="What is traceability?",
            )

            assert "answer" in result
            assert "sources" in result
            assert result["answer"] == "The answer is..."
