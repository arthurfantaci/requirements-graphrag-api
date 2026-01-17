"""Tests for core retrieval functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jama_mcp_server_graphrag.core.retrieval import (
    explore_entity,
    graph_enriched_search,
    hybrid_search,
    vector_search,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Create a mock Neo4jVector store."""
    store = MagicMock()
    store.similarity_search_with_score = MagicMock(
        return_value=[
            (
                MagicMock(
                    page_content="Test content about requirements",
                    metadata={
                        "title": "Test Article",
                        "url": "https://example.com/test",
                        "chunk_id": "chunk-1",
                    },
                ),
                0.95,
            ),
            (
                MagicMock(
                    page_content="More content about traceability",
                    metadata={
                        "title": "Another Article",
                        "url": "https://example.com/another",
                        "chunk_id": "chunk-2",
                    },
                ),
                0.85,
            ),
        ]
    )
    return store


@pytest.fixture
def mock_graph() -> MagicMock:
    """Create a mock Neo4jGraph."""
    graph = MagicMock()
    # Mock entity node as a dict-like object
    mock_entity = {
        "name": "requirements traceability",
        "definition": "The ability to trace requirements",
    }
    graph.query = MagicMock(
        return_value=[
            {
                "e": mock_entity,
                "labels": ["Entity", "Concept"],
            }
        ]
    )
    return graph


# =============================================================================
# Vector Search Tests
# =============================================================================


class TestVectorSearch:
    """Tests for vector_search function."""

    @pytest.mark.asyncio
    async def test_vector_search_returns_results(
        self, mock_vector_store: MagicMock
    ) -> None:
        """Test that vector search returns formatted results."""
        results = await vector_search(mock_vector_store, "requirements", limit=5)

        assert len(results) == 2
        assert results[0]["content"] == "Test content about requirements"
        assert results[0]["score"] == 0.95
        assert results[0]["metadata"]["title"] == "Test Article"

    @pytest.mark.asyncio
    async def test_vector_search_respects_limit(
        self, mock_vector_store: MagicMock
    ) -> None:
        """Test that limit parameter is passed to similarity search."""
        await vector_search(mock_vector_store, "test query", limit=10)

        mock_vector_store.similarity_search_with_score.assert_called_once_with(
            "test query", k=10
        )

    @pytest.mark.asyncio
    async def test_vector_search_empty_results(self) -> None:
        """Test handling of empty search results."""
        store = MagicMock()
        store.similarity_search_with_score = MagicMock(return_value=[])

        results = await vector_search(store, "nonexistent topic")

        assert results == []


# =============================================================================
# Hybrid Search Tests
# =============================================================================


class TestHybridSearch:
    """Tests for hybrid_search function."""

    @pytest.mark.asyncio
    async def test_hybrid_search_combines_results(self) -> None:
        """Test that hybrid search returns combined results."""
        # Create vector store with article_id in metadata (required for dedup)
        vector_store = MagicMock()
        vector_store.similarity_search_with_score = MagicMock(
            return_value=[
                (
                    MagicMock(
                        page_content="Test content",
                        metadata={
                            "title": "Test Article",
                            "article_id": "article-1",
                            "url": "https://example.com/test",
                        },
                    ),
                    0.95,
                ),
            ]
        )

        # Mock graph with keyword search failing (common case)
        graph = MagicMock()
        graph.query = MagicMock(side_effect=Exception("No fulltext index"))

        results = await hybrid_search(graph, vector_store, "requirements traceability")

        # Should still return vector results even if keyword search fails
        assert len(results) > 0
        vector_store.similarity_search_with_score.assert_called()

    @pytest.mark.asyncio
    async def test_hybrid_search_keyword_weight(self) -> None:
        """Test that keyword_weight parameter affects scoring."""
        # Create vector store with article_id in metadata
        vector_store = MagicMock()
        vector_store.similarity_search_with_score = MagicMock(
            return_value=[
                (
                    MagicMock(
                        page_content="Test content",
                        metadata={
                            "title": "Test Article",
                            "article_id": "article-1",
                            "url": "https://example.com/test",
                        },
                    ),
                    0.95,
                ),
            ]
        )

        graph = MagicMock()
        graph.query = MagicMock(side_effect=Exception("No fulltext index"))

        # With keyword_weight=0.3, vector score is multiplied by (1 - 0.3) = 0.7
        results_default = await hybrid_search(
            graph, vector_store, "test", keyword_weight=0.3
        )

        # Score should be adjusted: 0.95 * 0.7 = 0.665
        assert len(results_default) > 0
        assert abs(results_default[0]["score"] - 0.665) < 0.01


# =============================================================================
# Graph Enriched Search Tests
# =============================================================================


class TestGraphEnrichedSearch:
    """Tests for graph_enriched_search function."""

    @pytest.mark.asyncio
    async def test_graph_enriched_search_adds_entities(
        self, mock_graph: MagicMock, mock_vector_store: MagicMock
    ) -> None:
        """Test that graph enriched search adds related entities."""
        results = await graph_enriched_search(
            mock_graph, mock_vector_store, "requirements"
        )

        assert len(results) > 0
        # Graph should be queried for entities
        assert mock_graph.query.called

    @pytest.mark.asyncio
    async def test_graph_enriched_search_traversal_depth(
        self, mock_graph: MagicMock, mock_vector_store: MagicMock
    ) -> None:
        """Test that traversal_depth parameter is used."""
        await graph_enriched_search(
            mock_graph, mock_vector_store, "test", traversal_depth=2
        )

        # Should still work with different depths
        assert mock_vector_store.similarity_search_with_score.called


# =============================================================================
# Explore Entity Tests
# =============================================================================


class TestExploreEntity:
    """Tests for explore_entity function."""

    @pytest.mark.asyncio
    async def test_explore_entity_found(self) -> None:
        """Test exploring an existing entity."""
        graph = MagicMock()
        mock_entity = {"name": "requirements traceability", "definition": "Test def"}
        # First query returns entity, subsequent return empty lists
        graph.query = MagicMock(
            side_effect=[
                [{"e": mock_entity, "labels": ["Entity", "Concept"]}],  # Entity
                [],  # Related
                [],  # Mentions
            ]
        )

        result = await explore_entity(graph, "requirements traceability")

        assert result is not None
        assert result["name"] == "requirements traceability"
        assert "labels" in result

    @pytest.mark.asyncio
    async def test_explore_entity_not_found(self) -> None:
        """Test exploring a non-existent entity."""
        graph = MagicMock()
        graph.query = MagicMock(return_value=[])

        result = await explore_entity(graph, "nonexistent entity")

        assert result is None

    @pytest.mark.asyncio
    async def test_explore_entity_include_related(self) -> None:
        """Test that include_related fetches relationships."""
        graph = MagicMock()
        mock_entity = {"name": "test", "definition": "Test"}
        graph.query = MagicMock(
            side_effect=[
                [{"e": mock_entity, "labels": ["Entity"]}],  # Entity
                [{"name": "related", "relationship": "RELATED_TO", "labels": ["Entity"]}],
                [{"article": "Test Article", "heading": "Section", "url": "http://example.com"}],
            ]
        )

        result = await explore_entity(graph, "test", include_related=True)

        # Multiple queries should be made for related entities
        assert graph.query.call_count == 3
        assert "related" in result
        assert "mentioned_in" in result

    @pytest.mark.asyncio
    async def test_explore_entity_exclude_related(self) -> None:
        """Test that include_related=False skips relationship queries."""
        graph = MagicMock()
        mock_entity = {"name": "test", "definition": "Test"}
        graph.query = MagicMock(return_value=[{"e": mock_entity, "labels": ["Entity"]}])

        await explore_entity(graph, "test", include_related=False)

        # Only the main entity query should be made
        assert graph.query.call_count == 1
