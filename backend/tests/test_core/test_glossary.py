"""Tests for core definitions functions (renamed from glossary).

Updated Data Model (2026-01):
- Definition nodes replace GlossaryTerm
- Uses direct Neo4j driver instead of LangChain Neo4jGraph
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from requirements_graphrag_api.core.definitions import (
    list_all_terms,
    lookup_term,
    search_terms,
)

# =============================================================================
# Mock Helpers
# =============================================================================


def create_mock_record(data: dict[str, Any]) -> MagicMock:
    """Create a mock Neo4j record with proper __getitem__ support."""
    record = MagicMock()
    record.__getitem__ = lambda s, k: data.get(k)
    record.get = lambda k, d=None: data.get(k, d)
    record.data = lambda: data
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
def mock_driver_with_terms() -> MagicMock:
    """Create a mock Neo4j driver with definition terms."""
    return create_mock_driver_with_results(
        [
            [
                {
                    "term": "Requirements Traceability",
                    "definition": "The ability to trace requirements throughout the lifecycle",
                    "url": "https://example.com/glossary#traceability",
                    "term_id": "term-123",
                    "score": 0.95,
                }
            ]
        ]
    )


@pytest.fixture
def mock_driver_empty() -> MagicMock:
    """Create a mock Neo4j driver that returns no results."""
    return create_mock_driver_with_results([[]])


# =============================================================================
# Lookup Term Tests
# =============================================================================


class TestLookupTerm:
    """Tests for lookup_term function."""

    @pytest.mark.asyncio
    async def test_lookup_term_found(self, mock_driver_with_terms: MagicMock) -> None:
        """Test looking up an existing term."""
        result = await lookup_term(mock_driver_with_terms, "traceability")

        assert result is not None
        assert result["term"] == "Requirements Traceability"
        assert "definition" in result

    @pytest.mark.asyncio
    async def test_lookup_term_not_found(self, mock_driver_empty: MagicMock) -> None:
        """Test looking up a non-existent term."""
        result = await lookup_term(mock_driver_empty, "nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_lookup_term_fuzzy_uses_contains(self, mock_driver_with_terms: MagicMock) -> None:
        """Test fuzzy matching uses CONTAINS."""
        await lookup_term(mock_driver_with_terms, "trace", fuzzy=True)

        mock_session = mock_driver_with_terms.session.return_value.__enter__.return_value
        call_args = mock_session.run.call_args
        assert "CONTAINS" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_lookup_term_exact_matching(self, mock_driver_with_terms: MagicMock) -> None:
        """Test exact matching when fuzzy=False."""
        await lookup_term(mock_driver_with_terms, "traceability", fuzzy=False)

        mock_session = mock_driver_with_terms.session.return_value.__enter__.return_value
        call_args = mock_session.run.call_args
        # Exact match uses equals or toLower comparison
        assert "toLower" in call_args[0][0]


# =============================================================================
# Search Terms Tests
# =============================================================================


class TestSearchTerms:
    """Tests for search_terms function."""

    @pytest.mark.asyncio
    async def test_search_terms_returns_list(self, mock_driver_with_terms: MagicMock) -> None:
        """Test that search returns a list of terms."""
        results = await search_terms(mock_driver_with_terms, "requirements")

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_terms_respects_limit(self, mock_driver_with_terms: MagicMock) -> None:
        """Test that limit parameter is passed to query."""
        await search_terms(mock_driver_with_terms, "test", limit=5)

        mock_session = mock_driver_with_terms.session.return_value.__enter__.return_value
        call_args = mock_session.run.call_args
        # Parameters passed as keyword arguments
        assert call_args[1]["limit"] == 5

    @pytest.mark.asyncio
    async def test_search_terms_empty_results(self, mock_driver_empty: MagicMock) -> None:
        """Test handling of empty search results."""
        results = await search_terms(mock_driver_empty, "nonexistent")

        assert results == []


# =============================================================================
# List All Terms Tests
# =============================================================================


class TestListAllTerms:
    """Tests for list_all_terms function."""

    @pytest.mark.asyncio
    async def test_list_all_terms_returns_list(self, mock_driver_with_terms: MagicMock) -> None:
        """Test that list_all_terms returns a list."""
        results = await list_all_terms(mock_driver_with_terms)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_list_all_terms_respects_limit(self, mock_driver_with_terms: MagicMock) -> None:
        """Test that limit parameter is used."""
        await list_all_terms(mock_driver_with_terms, limit=100)

        mock_session = mock_driver_with_terms.session.return_value.__enter__.return_value
        call_args = mock_session.run.call_args
        # Parameters passed as keyword arguments
        assert call_args[1]["limit"] == 100

    @pytest.mark.asyncio
    async def test_list_all_terms_default_limit(self, mock_driver_with_terms: MagicMock) -> None:
        """Test default limit of 50."""
        await list_all_terms(mock_driver_with_terms)

        mock_session = mock_driver_with_terms.session.return_value.__enter__.return_value
        call_args = mock_session.run.call_args
        # Parameters passed as keyword arguments
        assert call_args[1]["limit"] == 50


# =============================================================================
# Acronym Normalization Tests
# =============================================================================


class TestNormalizeForAcronymMatch:
    """Tests for _normalize_for_acronym_match helper function."""

    def test_normalize_removes_periods(self) -> None:
        """Test that periods are removed."""
        from requirements_graphrag_api.core.definitions import _normalize_for_acronym_match

        assert _normalize_for_acronym_match("A.o.A.") == "aoa"

    def test_normalize_removes_spaces(self) -> None:
        """Test that spaces are removed."""
        from requirements_graphrag_api.core.definitions import _normalize_for_acronym_match

        assert _normalize_for_acronym_match("A o A") == "aoa"

    def test_normalize_lowercases(self) -> None:
        """Test that text is lowercased."""
        from requirements_graphrag_api.core.definitions import _normalize_for_acronym_match

        assert _normalize_for_acronym_match("AoA") == "aoa"

    def test_normalize_combined(self) -> None:
        """Test combined normalization."""
        from requirements_graphrag_api.core.definitions import _normalize_for_acronym_match

        # All variations should normalize to same value
        assert _normalize_for_acronym_match("AoA") == "aoa"
        assert _normalize_for_acronym_match("A.o.A.") == "aoa"
        assert _normalize_for_acronym_match("A o A") == "aoa"
        assert _normalize_for_acronym_match("a.o.a") == "aoa"


# =============================================================================
# Acronym Search Tests
# =============================================================================


@pytest.fixture
def mock_driver_with_acronym() -> MagicMock:
    """Create a mock Neo4j driver with a term that has an acronym."""
    return create_mock_driver_with_results(
        [
            [
                {
                    "term": "Analysis of Alternatives",
                    "definition": "A systematic evaluation of alternatives.",
                    "acronym": "AoA",
                    "url": "https://example.com/glossary#aoa",
                    "term_id": "term-aoa",
                    "score": 1.0,
                }
            ]
        ]
    )


class TestAcronymSearch:
    """Tests for acronym search functionality."""

    @pytest.mark.asyncio
    async def test_lookup_term_by_acronym(self, mock_driver_with_acronym: MagicMock) -> None:
        """Test looking up a term by its acronym."""
        result = await lookup_term(mock_driver_with_acronym, "AoA")

        assert result is not None
        assert result["term"] == "Analysis of Alternatives"
        assert result["acronym"] == "AoA"

    @pytest.mark.asyncio
    async def test_lookup_term_by_acronym_normalized(
        self, mock_driver_with_acronym: MagicMock
    ) -> None:
        """Test looking up by acronym with periods (e.g., A.o.A.)."""
        # The query should use normalized_term parameter
        await lookup_term(mock_driver_with_acronym, "A.o.A.")

        mock_session = mock_driver_with_acronym.session.return_value.__enter__.return_value
        call_args = mock_session.run.call_args
        # Check that normalized_term is passed to the query
        assert "normalized_term" in call_args[1]
        assert call_args[1]["normalized_term"] == "aoa"

    @pytest.mark.asyncio
    async def test_search_terms_includes_acronym_field(
        self, mock_driver_with_acronym: MagicMock
    ) -> None:
        """Test that search results include the acronym field."""
        results = await search_terms(mock_driver_with_acronym, "AoA")

        assert len(results) > 0
        assert "acronym" in results[0]
        assert results[0]["acronym"] == "AoA"

    @pytest.mark.asyncio
    async def test_search_cypher_includes_acronym_matching(
        self, mock_driver_with_acronym: MagicMock
    ) -> None:
        """Test that search Cypher query includes acronym matching logic."""
        await search_terms(mock_driver_with_acronym, "AoA")

        mock_session = mock_driver_with_acronym.session.return_value.__enter__.return_value
        call_args = mock_session.run.call_args
        cypher = call_args[0][0]
        # Verify acronym matching is in the query
        assert "d.acronym" in cypher
        assert "normalized_query" in call_args[1]

    @pytest.mark.asyncio
    async def test_lookup_returns_acronym_field(self, mock_driver_with_acronym: MagicMock) -> None:
        """Test that lookup returns the acronym field in result."""
        result = await lookup_term(mock_driver_with_acronym, "Analysis of Alternatives")

        assert result is not None
        assert "acronym" in result
        assert result["acronym"] == "AoA"
