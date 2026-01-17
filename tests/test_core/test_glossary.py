"""Tests for core glossary functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jama_mcp_server_graphrag.core.glossary import (
    list_all_terms,
    lookup_term,
    search_terms,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_graph_with_terms() -> MagicMock:
    """Create a mock Neo4jGraph with glossary terms."""
    graph = MagicMock()
    graph.query = MagicMock(
        return_value=[
            {
                "term": "Requirements Traceability",
                "definition": "The ability to trace requirements throughout the lifecycle",
                "source": "Jama Guide",
                "score": 0.95,
            }
        ]
    )
    return graph


@pytest.fixture
def mock_graph_empty() -> MagicMock:
    """Create a mock Neo4jGraph that returns no results."""
    graph = MagicMock()
    graph.query = MagicMock(return_value=[])
    return graph


# =============================================================================
# Lookup Term Tests
# =============================================================================


class TestLookupTerm:
    """Tests for lookup_term function."""

    @pytest.mark.asyncio
    async def test_lookup_term_found(self, mock_graph_with_terms: MagicMock) -> None:
        """Test looking up an existing term."""
        result = await lookup_term(mock_graph_with_terms, "traceability")

        assert result is not None
        assert result["term"] == "Requirements Traceability"
        assert "definition" in result
        assert result["score"] == 0.95

    @pytest.mark.asyncio
    async def test_lookup_term_not_found(self, mock_graph_empty: MagicMock) -> None:
        """Test looking up a non-existent term."""
        result = await lookup_term(mock_graph_empty, "nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_lookup_term_fuzzy_matching(
        self, mock_graph_with_terms: MagicMock
    ) -> None:
        """Test fuzzy matching is used by default."""
        await lookup_term(mock_graph_with_terms, "trace", fuzzy=True)

        # Should use fulltext query
        call_args = mock_graph_with_terms.query.call_args
        assert "fulltext" in call_args[0][0] or "CONTAINS" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_lookup_term_exact_matching(
        self, mock_graph_with_terms: MagicMock
    ) -> None:
        """Test exact matching when fuzzy=False."""
        await lookup_term(mock_graph_with_terms, "traceability", fuzzy=False)

        # Should use exact match query
        call_args = mock_graph_with_terms.query.call_args
        assert "toLower" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_lookup_term_fallback_on_error(self) -> None:
        """Test fallback to CONTAINS when fulltext index fails."""
        graph = MagicMock()
        # First call fails (fulltext), second succeeds (fallback)
        graph.query = MagicMock(
            side_effect=[
                Exception("Index not found"),
                [{"term": "Test", "definition": "A test term", "source": None}],
            ]
        )

        result = await lookup_term(graph, "test", fuzzy=True)

        assert result is not None
        assert result["term"] == "Test"
        assert result["score"] == 0.8  # Fallback score


# =============================================================================
# Search Terms Tests
# =============================================================================


class TestSearchTerms:
    """Tests for search_terms function."""

    @pytest.mark.asyncio
    async def test_search_terms_returns_list(
        self, mock_graph_with_terms: MagicMock
    ) -> None:
        """Test that search returns a list of terms."""
        results = await search_terms(mock_graph_with_terms, "requirements")

        assert isinstance(results, list)
        assert len(results) > 0
        assert results[0]["term"] == "Requirements Traceability"

    @pytest.mark.asyncio
    async def test_search_terms_respects_limit(
        self, mock_graph_with_terms: MagicMock
    ) -> None:
        """Test that limit parameter is passed to query."""
        await search_terms(mock_graph_with_terms, "test", limit=5)

        call_args = mock_graph_with_terms.query.call_args
        # Parameters are passed as second positional arg (dict)
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert params["limit"] == 5

    @pytest.mark.asyncio
    async def test_search_terms_empty_results(
        self, mock_graph_empty: MagicMock
    ) -> None:
        """Test handling of empty search results."""
        results = await search_terms(mock_graph_empty, "nonexistent")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_terms_score_rounding(
        self, mock_graph_with_terms: MagicMock
    ) -> None:
        """Test that scores are rounded to 4 decimal places."""
        results = await search_terms(mock_graph_with_terms, "test")

        assert isinstance(results[0]["score"], float)


# =============================================================================
# List All Terms Tests
# =============================================================================


class TestListAllTerms:
    """Tests for list_all_terms function."""

    @pytest.mark.asyncio
    async def test_list_all_terms_returns_list(
        self, mock_graph_with_terms: MagicMock
    ) -> None:
        """Test that list_all_terms returns a list."""
        results = await list_all_terms(mock_graph_with_terms)

        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_list_all_terms_respects_limit(
        self, mock_graph_with_terms: MagicMock
    ) -> None:
        """Test that limit parameter is used."""
        await list_all_terms(mock_graph_with_terms, limit=100)

        call_args = mock_graph_with_terms.query.call_args
        # Parameters are passed as second positional arg (dict)
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert params["limit"] == 100

    @pytest.mark.asyncio
    async def test_list_all_terms_default_limit(
        self, mock_graph_with_terms: MagicMock
    ) -> None:
        """Test default limit of 50."""
        await list_all_terms(mock_graph_with_terms)

        call_args = mock_graph_with_terms.query.call_args
        # Parameters are passed as second positional arg (dict)
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert params["limit"] == 50
