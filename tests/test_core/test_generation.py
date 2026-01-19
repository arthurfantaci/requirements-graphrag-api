"""Tests for core generation functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jama_mcp_server_graphrag.core.generation import chat, generate_answer
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
def mock_graph() -> MagicMock:
    """Create a mock Neo4jGraph."""
    graph = MagicMock()
    graph.query = MagicMock(return_value=[])
    return graph


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Create a mock Neo4jVector store."""
    store = MagicMock()
    store.similarity_search_with_score = MagicMock(
        return_value=[
            (
                MagicMock(
                    page_content="Requirements traceability is essential for quality.",
                    metadata={
                        "title": "Traceability Guide",
                        "url": "https://example.com/trace",
                        "chunk_id": "chunk-1",
                    },
                ),
                0.92,
            ),
        ]
    )
    return store


@pytest.fixture
def mock_search_results() -> list[dict]:
    """Create mock graph-enriched search results."""
    return [
        {
            "content": "Requirements traceability enables tracking requirements.",
            "score": 0.95,
            "metadata": {
                "title": "Traceability Article",
                "url": "https://example.com/article1",
                "chunk_id": "chunk-1",
            },
            "related_entities": ["requirements", "traceability"],
            "glossary_terms": ["traceability matrix"],
        },
        {
            "content": "Best practices for requirements management.",
            "score": 0.88,
            "metadata": {
                "title": "Best Practices",
                "url": "https://example.com/article2",
                "chunk_id": "chunk-2",
            },
            "related_entities": ["requirements management"],
            "glossary_terms": [],
        },
    ]


# =============================================================================
# Generate Answer Tests
# =============================================================================


class TestGenerateAnswer:
    """Tests for generate_answer function."""

    @pytest.mark.asyncio
    async def test_generate_answer_returns_complete_response(
        self,
        mock_config: MagicMock,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
        mock_search_results: list[dict],
    ) -> None:
        """Test that generate_answer returns a complete response."""
        with (
            patch("jama_mcp_server_graphrag.core.generation.graph_enriched_search") as mock_search,
            patch("jama_mcp_server_graphrag.core.generation.search_terms") as mock_terms,
            patch("jama_mcp_server_graphrag.core.generation.ChatOpenAI") as mock_llm_class,
        ):
            mock_search.return_value = mock_search_results
            mock_terms.return_value = []
            mock_llm_class.return_value = create_llm_mock(
                "Requirements traceability is the ability to track..."
            )

            result = await generate_answer(
                mock_config,
                mock_graph,
                mock_vector_store,
                "What is requirements traceability?",
            )

            assert "question" in result
            assert "answer" in result
            assert "sources" in result
            assert "entities" in result
            assert result["source_count"] > 0

    @pytest.mark.asyncio
    async def test_generate_answer_includes_sources(
        self,
        mock_config: MagicMock,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
        mock_search_results: list[dict],
    ) -> None:
        """Test that sources are properly formatted."""
        with (
            patch("jama_mcp_server_graphrag.core.generation.graph_enriched_search") as mock_search,
            patch("jama_mcp_server_graphrag.core.generation.search_terms") as mock_terms,
            patch("jama_mcp_server_graphrag.core.generation.ChatOpenAI") as mock_llm_class,
        ):
            mock_search.return_value = mock_search_results
            mock_terms.return_value = []
            mock_llm_class.return_value = create_llm_mock("Answer text")

            result = await generate_answer(
                mock_config,
                mock_graph,
                mock_vector_store,
                "Test question",
            )

            assert len(result["sources"]) == 2
            assert result["sources"][0]["title"] == "Traceability Article"
            assert "relevance_score" in result["sources"][0]

    @pytest.mark.asyncio
    async def test_generate_answer_respects_retrieval_limit(
        self,
        mock_config: MagicMock,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test that retrieval_limit parameter is passed to search."""
        with (
            patch("jama_mcp_server_graphrag.core.generation.graph_enriched_search") as mock_search,
            patch("jama_mcp_server_graphrag.core.generation.search_terms") as mock_terms,
            patch("jama_mcp_server_graphrag.core.generation.ChatOpenAI") as mock_llm_class,
        ):
            mock_search.return_value = []
            mock_terms.return_value = []
            mock_llm_class.return_value = create_llm_mock("No context found.")

            await generate_answer(
                mock_config,
                mock_graph,
                mock_vector_store,
                "Test",
                retrieval_limit=10,
            )

            mock_search.assert_called_once()
            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["limit"] == 10


# =============================================================================
# Chat Tests
# =============================================================================


class TestChat:
    """Tests for chat function."""

    @pytest.mark.asyncio
    async def test_chat_delegates_to_generate_answer(
        self,
        mock_config: MagicMock,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test that chat delegates to generate_answer."""
        with patch("jama_mcp_server_graphrag.core.generation.generate_answer") as mock_gen:
            mock_gen.return_value = {
                "question": "Test",
                "answer": "Answer",
                "sources": [],
                "entities": [],
                "source_count": 0,
            }

            result = await chat(
                mock_config,
                mock_graph,
                mock_vector_store,
                "Test message",
                max_sources=3,
            )

            mock_gen.assert_called_once()
            assert result["answer"] == "Answer"

    @pytest.mark.asyncio
    async def test_chat_passes_max_sources(
        self,
        mock_config: MagicMock,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test that max_sources is passed to generate_answer."""
        with patch("jama_mcp_server_graphrag.core.generation.generate_answer") as mock_gen:
            mock_gen.return_value = {
                "question": "Test",
                "answer": "Answer",
                "sources": [],
                "entities": [],
                "source_count": 0,
            }

            await chat(
                mock_config,
                mock_graph,
                mock_vector_store,
                "Test",
                max_sources=7,
            )

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["retrieval_limit"] == 7
