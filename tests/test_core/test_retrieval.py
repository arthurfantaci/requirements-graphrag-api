"""Tests for core retrieval functions.

Updated Data Model (2026-01):
- Uses neo4j-graphrag VectorRetriever instead of LangChain Neo4jVector
- Uses direct Neo4j driver instead of LangChain Neo4jGraph
- Function signatures changed to take (retriever, driver, query, ...)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from jama_mcp_server_graphrag.core.retrieval import (
    explore_entity,
    graph_enriched_search,
    hybrid_search,
    vector_search,
)

# =============================================================================
# Mock Helpers
# =============================================================================


class MockRetrieverItem:
    """Mock for VectorRetriever result item."""

    def __init__(self, content: str, metadata: dict[str, Any]) -> None:
        self.content = content
        self.metadata = metadata


class MockRetrieverResult:
    """Mock for VectorRetriever search result."""

    def __init__(self, items: list[MockRetrieverItem]) -> None:
        self.items = items


def create_mock_retriever(items: list[dict[str, Any]]) -> MagicMock:
    """Create a mock VectorRetriever."""
    retriever = MagicMock()
    result_items = [
        MockRetrieverItem(
            content=item.get("text", ""),
            metadata=item.get("metadata", {}),
        )
        for item in items
    ]
    retriever.search.return_value = MockRetrieverResult(items=result_items)
    return retriever


class MockNeo4jNode(dict):
    """Mock for Neo4j Node that acts like dict and supports items()."""

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__(data)


def create_mock_record(data: dict[str, Any]) -> MagicMock:
    """Create a mock Neo4j record."""
    record = MagicMock()
    # Support __getitem__ access
    record.__getitem__ = lambda s, k: data.get(k)
    record.get = lambda k, d=None: data.get(k, d)
    record.data = lambda: data
    # For entity nodes, wrap in MockNeo4jNode
    if "e" in data:
        wrapped_data = dict(data)
        wrapped_data["e"] = MockNeo4jNode(data["e"])
        record.__getitem__ = lambda s, k: wrapped_data.get(k)
        record.get = lambda k, d=None: wrapped_data.get(k, d)
    return record


def create_mock_driver_with_results(results_sequence: list[list[dict[str, Any]]]) -> MagicMock:
    """Create a mock Neo4j driver that returns a sequence of results."""
    mock_driver = MagicMock()
    mock_session = MagicMock()

    call_index = [0]

    def run_side_effect(*args, **kwargs):
        idx = call_index[0]
        call_index[0] += 1

        mock_result = MagicMock()

        if idx < len(results_sequence):
            records = results_sequence[idx]
            mock_records = [create_mock_record(r) for r in records]
            mock_result.__iter__ = lambda self, recs=mock_records: iter(recs)
            mock_result.single.return_value = mock_records[0] if mock_records else None
        else:
            mock_result.__iter__ = lambda self: iter([])
            mock_result.single.return_value = None

        return mock_result

    mock_session.run = MagicMock(side_effect=run_side_effect)
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_driver.session.return_value = mock_session

    return mock_driver


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_retriever() -> MagicMock:
    """Create a mock VectorRetriever with test data."""
    return create_mock_retriever(
        [
            {
                "text": "Test content about requirements",
                "metadata": {
                    "id": "4:abc:123",  # neo4j-graphrag uses 'id' not 'element_id'
                    "title": "Test Article",
                    "score": 0.95,
                },
            },
            {
                "text": "More content about traceability",
                "metadata": {
                    "id": "4:abc:456",  # neo4j-graphrag uses 'id' not 'element_id'
                    "title": "Another Article",
                    "score": 0.85,
                },
            },
        ]
    )


@pytest.fixture
def mock_driver() -> MagicMock:
    """Create a mock Neo4j driver."""
    return create_mock_driver_with_results(
        [
            # Article context query results
            [
                {
                    "chunk_id": "4:abc:123",
                    "title": "Test Article",
                    "url": "https://example.com/test",
                    "article_id": "article-1",
                    "chapter": "Chapter 1",
                },
                {
                    "chunk_id": "4:abc:456",
                    "title": "Another Article",
                    "url": "https://example.com/another",
                    "article_id": "article-2",
                    "chapter": "Chapter 2",
                },
            ]
        ]
    )


@pytest.fixture
def empty_retriever() -> MagicMock:
    """Create a mock VectorRetriever with empty results."""
    return create_mock_retriever([])


@pytest.fixture
def empty_driver() -> MagicMock:
    """Create a mock Neo4j driver with empty results."""
    return create_mock_driver_with_results([[]])


# =============================================================================
# Vector Search Tests
# =============================================================================


class TestVectorSearch:
    """Tests for vector_search function."""

    @pytest.mark.asyncio
    async def test_vector_search_returns_results(
        self, mock_retriever: MagicMock, mock_driver: MagicMock
    ) -> None:
        """Test that vector search returns formatted results."""
        results = await vector_search(mock_retriever, mock_driver, "requirements", limit=5)

        assert len(results) == 2
        assert results[0]["content"] == "Test content about requirements"

    @pytest.mark.asyncio
    async def test_vector_search_respects_limit(
        self, mock_retriever: MagicMock, mock_driver: MagicMock
    ) -> None:
        """Test that limit parameter is passed to retriever."""
        await vector_search(mock_retriever, mock_driver, "test query", limit=10)

        mock_retriever.search.assert_called_once_with(query_text="test query", top_k=10)

    @pytest.mark.asyncio
    async def test_vector_search_empty_results(
        self, empty_retriever: MagicMock, empty_driver: MagicMock
    ) -> None:
        """Test handling of empty search results."""
        results = await vector_search(empty_retriever, empty_driver, "nonexistent topic")

        assert results == []


# =============================================================================
# Hybrid Search Tests
# =============================================================================


class TestHybridSearch:
    """Tests for hybrid_search function."""

    @pytest.mark.asyncio
    async def test_hybrid_search_combines_results(
        self, mock_retriever: MagicMock, mock_driver: MagicMock
    ) -> None:
        """Test that hybrid search returns combined results."""
        results = await hybrid_search(mock_retriever, mock_driver, "requirements traceability")

        assert len(results) > 0
        mock_retriever.search.assert_called()

    @pytest.mark.asyncio
    async def test_hybrid_search_keyword_weight(
        self, mock_retriever: MagicMock, mock_driver: MagicMock
    ) -> None:
        """Test that keyword_weight parameter affects scoring."""
        results = await hybrid_search(mock_retriever, mock_driver, "test", keyword_weight=0.3)

        assert len(results) > 0


# =============================================================================
# Graph Enriched Search Tests
# =============================================================================


class TestGraphEnrichedSearch:
    """Tests for graph_enriched_search function."""

    @pytest.mark.asyncio
    async def test_graph_enriched_search_adds_entities(
        self, mock_retriever: MagicMock, mock_driver: MagicMock
    ) -> None:
        """Test that graph enriched search adds related entities."""
        results = await graph_enriched_search(mock_retriever, mock_driver, "requirements")

        assert len(results) > 0
        mock_retriever.search.assert_called()

    @pytest.mark.asyncio
    async def test_graph_enriched_search_with_options(
        self, mock_retriever: MagicMock, mock_driver: MagicMock
    ) -> None:
        """Test that GraphEnrichmentOptions parameter is used."""
        from jama_mcp_server_graphrag.core.retrieval import GraphEnrichmentOptions

        # Create custom options with some features disabled
        options = GraphEnrichmentOptions(
            enable_window_expansion=False,
            enable_media_enrichment=False,
        )
        await graph_enriched_search(mock_retriever, mock_driver, "test", options=options)

        mock_retriever.search.assert_called()


# =============================================================================
# Explore Entity Tests
# =============================================================================


class TestExploreEntity:
    """Tests for explore_entity function."""

    @pytest.mark.asyncio
    async def test_explore_entity_found(self) -> None:
        """Test exploring an existing entity."""
        driver = create_mock_driver_with_results(
            [
                # Entity query returns entity node
                [
                    {
                        "e": {
                            "name": "requirements traceability",
                            "display_name": "Requirements Traceability",
                            "definition": "The ability to trace requirements",
                        },
                        "labels": ["Entity", "Concept"],
                    }
                ],
                # Related entities query
                [],
                # Mentioned in articles query
                [],
            ]
        )

        result = await explore_entity(driver, "requirements traceability")

        assert result is not None
        assert result["name"] == "requirements traceability"
        assert "labels" in result

    @pytest.mark.asyncio
    async def test_explore_entity_not_found(self, empty_driver: MagicMock) -> None:
        """Test exploring a non-existent entity."""
        result = await explore_entity(empty_driver, "nonexistent entity")

        assert result is None

    @pytest.mark.asyncio
    async def test_explore_entity_include_related(self) -> None:
        """Test that include_related fetches relationships."""
        driver = create_mock_driver_with_results(
            [
                # Entity query
                [
                    {
                        "e": {
                            "name": "test",
                            "display_name": "Test Entity",
                        },
                        "labels": ["Entity"],
                    }
                ],
                # Related entities query
                [
                    {
                        "name": "related",
                        "display_name": "Related Entity",
                        "relationship": "RELATED_TO",
                        "labels": ["Entity"],
                    }
                ],
                # Mentioned in articles query
                [
                    {
                        "article": "Test Article",
                        "heading": "Section",
                        "url": "http://example.com",
                    }
                ],
            ]
        )

        result = await explore_entity(driver, "test", include_related=True)

        mock_session = driver.session.return_value
        assert mock_session.run.call_count == 3
        assert "related" in result
        assert "mentioned_in" in result

    @pytest.mark.asyncio
    async def test_explore_entity_exclude_related(self) -> None:
        """Test that include_related=False skips relationship queries."""
        driver = create_mock_driver_with_results(
            [
                [
                    {
                        "e": {
                            "name": "test",
                            "display_name": "Test Entity",
                        },
                        "labels": ["Entity"],
                    }
                ]
            ]
        )

        await explore_entity(driver, "test", include_related=False)

        mock_session = driver.session.return_value
        assert mock_session.run.call_count == 1
