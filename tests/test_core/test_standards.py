"""Tests for core standards functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jama_mcp_server_graphrag.core.standards import (
    get_standards_by_industry,
    list_all_standards,
    lookup_standard,
    search_standards,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_graph_with_standard() -> MagicMock:
    """Create a mock Neo4jGraph with a standard."""
    graph = MagicMock()
    # Return different results for different queries
    graph.query = MagicMock(
        return_value=[
            {
                "name": "ISO 26262",
                "definition": "Functional safety standard for automotive",
                "organization": "ISO",
                "types": "Standard",
                "labels": ["Standard", "Entity"],
            }
        ]
    )
    return graph


@pytest.fixture
def mock_graph_with_standard_and_related() -> MagicMock:
    """Create a mock graph that returns standard and related data."""
    graph = MagicMock()
    graph.query = MagicMock(
        side_effect=[
            # First query: standard lookup
            [
                {
                    "name": "ISO 26262",
                    "definition": "Functional safety standard for automotive",
                    "organization": "ISO",
                    "types": "Standard",
                    "labels": ["Standard", "Entity"],
                }
            ],
            # Second query: related entities
            [
                {
                    "name": "ASIL",
                    "relationship": "DEFINES",
                    "labels": ["Concept"],
                }
            ],
            # Third query: articles mentioning
            [
                {
                    "title": "Automotive Safety",
                    "url": "https://example.com/automotive",
                }
            ],
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
# Lookup Standard Tests
# =============================================================================


class TestLookupStandard:
    """Tests for lookup_standard function."""

    @pytest.mark.asyncio
    async def test_lookup_standard_found(
        self, mock_graph_with_standard_and_related: MagicMock
    ) -> None:
        """Test looking up an existing standard."""
        result = await lookup_standard(
            mock_graph_with_standard_and_related, "ISO 26262"
        )

        assert result is not None
        assert result["name"] == "ISO 26262"
        assert result["organization"] == "ISO"
        assert "definition" in result

    @pytest.mark.asyncio
    async def test_lookup_standard_not_found(
        self, mock_graph_empty: MagicMock
    ) -> None:
        """Test looking up a non-existent standard."""
        result = await lookup_standard(mock_graph_empty, "FAKE-12345")

        assert result is None

    @pytest.mark.asyncio
    async def test_lookup_standard_include_related(
        self, mock_graph_with_standard_and_related: MagicMock
    ) -> None:
        """Test that include_related fetches relationships."""
        result = await lookup_standard(
            mock_graph_with_standard_and_related, "ISO 26262", include_related=True
        )

        # Multiple queries should be made
        assert mock_graph_with_standard_and_related.query.call_count == 3
        assert "related" in result
        assert "mentioned_in" in result

    @pytest.mark.asyncio
    async def test_lookup_standard_exclude_related(
        self, mock_graph_with_standard: MagicMock
    ) -> None:
        """Test that include_related=False skips extra queries."""
        await lookup_standard(
            mock_graph_with_standard, "ISO 26262", include_related=False
        )

        # Only the main query
        assert mock_graph_with_standard.query.call_count == 1


# =============================================================================
# Search Standards Tests
# =============================================================================


class TestSearchStandards:
    """Tests for search_standards function."""

    @pytest.mark.asyncio
    async def test_search_standards_returns_list(
        self, mock_graph_with_standard: MagicMock
    ) -> None:
        """Test that search returns a list of standards."""
        results = await search_standards(mock_graph_with_standard, "automotive")

        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_standards_with_industry_filter(
        self, mock_graph_with_standard: MagicMock
    ) -> None:
        """Test filtering by industry."""
        await search_standards(
            mock_graph_with_standard, "safety", industry="automotive"
        )

        # Query should include industry filter
        call_args = mock_graph_with_standard.query.call_args
        assert "automotive" in str(call_args)

    @pytest.mark.asyncio
    async def test_search_standards_respects_limit(
        self, mock_graph_with_standard: MagicMock
    ) -> None:
        """Test that limit parameter is used."""
        await search_standards(mock_graph_with_standard, "test", limit=5)

        call_args = mock_graph_with_standard.query.call_args
        # Parameters are passed as second positional arg (dict)
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert params["limit"] == 5


# =============================================================================
# Get Standards by Industry Tests
# =============================================================================


class TestGetStandardsByIndustry:
    """Tests for get_standards_by_industry function."""

    @pytest.mark.asyncio
    async def test_get_standards_automotive(
        self, mock_graph_with_standard: MagicMock
    ) -> None:
        """Test getting automotive standards."""
        results = await get_standards_by_industry(
            mock_graph_with_standard, "automotive"
        )

        assert isinstance(results, list)
        # Query should include automotive patterns
        call_args = mock_graph_with_standard.query.call_args
        assert "automotive" in call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_get_standards_medical(
        self, mock_graph_with_standard: MagicMock
    ) -> None:
        """Test getting medical standards."""
        await get_standards_by_industry(mock_graph_with_standard, "medical")

        call_args = mock_graph_with_standard.query.call_args
        assert "medical" in call_args[0][0].lower() or "fda" in call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_get_standards_unknown_industry(
        self, mock_graph_with_standard: MagicMock
    ) -> None:
        """Test getting standards for unknown industry uses industry name."""
        await get_standards_by_industry(mock_graph_with_standard, "custom_industry")

        call_args = mock_graph_with_standard.query.call_args
        assert "custom_industry" in call_args[0][0].lower()


# =============================================================================
# List All Standards Tests
# =============================================================================


class TestListAllStandards:
    """Tests for list_all_standards function."""

    @pytest.mark.asyncio
    async def test_list_all_standards_returns_list(
        self, mock_graph_with_standard: MagicMock
    ) -> None:
        """Test that list returns a list of standards."""
        results = await list_all_standards(mock_graph_with_standard)

        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_list_all_standards_default_limit(
        self, mock_graph_with_standard: MagicMock
    ) -> None:
        """Test default limit of 50."""
        await list_all_standards(mock_graph_with_standard)

        call_args = mock_graph_with_standard.query.call_args
        # Parameters are passed as second positional arg (dict)
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert params["limit"] == 50

    @pytest.mark.asyncio
    async def test_list_all_standards_custom_limit(
        self, mock_graph_with_standard: MagicMock
    ) -> None:
        """Test custom limit parameter."""
        await list_all_standards(mock_graph_with_standard, limit=100)

        call_args = mock_graph_with_standard.query.call_args
        # Parameters are passed as second positional arg (dict)
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert params["limit"] == 100
