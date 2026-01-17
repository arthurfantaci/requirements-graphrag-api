"""Tests for core standards functions.

Updated Data Model (2026-01):
- Uses direct Neo4j driver instead of LangChain Neo4jGraph
- Standards connect to industries via APPLIES_TO relationship
"""

from __future__ import annotations

from typing import Any
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


def create_mock_result(records: list[dict[str, Any]]) -> MagicMock:
    """Create a mock Neo4j result with records."""
    mock_result = MagicMock()
    # For single() call (returns first record or None)
    if records:
        first_record = MagicMock()
        for key, value in records[0].items():
            first_record.__getitem__ = lambda s, k=key, v=value: records[0].get(k)
            first_record.get = lambda k, d=None: records[0].get(k, d)
        first_record.__getitem__ = lambda s, k: records[0].get(k)
        mock_result.single.return_value = first_record
    else:
        mock_result.single.return_value = None

    # For iteration (list comprehension)
    mock_records = []
    for rec in records:
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda s, k, r=rec: r.get(k)
        mock_record.get = lambda k, d=None, r=rec: r.get(k, d)
        mock_records.append(mock_record)
    mock_result.__iter__ = lambda self: iter(mock_records)

    return mock_result


def create_mock_driver_with_results(results_sequence: list[list[dict[str, Any]]]) -> MagicMock:
    """Create a mock Neo4j driver that returns a sequence of results."""
    mock_driver = MagicMock()
    mock_session = MagicMock()

    # Create mock results for each call
    mock_results = [create_mock_result(records) for records in results_sequence]

    # Handle multiple calls via side_effect
    call_index = [0]

    def run_side_effect(*args, **kwargs):
        idx = call_index[0]
        call_index[0] += 1
        if idx < len(mock_results):
            return mock_results[idx]
        return create_mock_result([])

    mock_session.run = MagicMock(side_effect=run_side_effect)
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_driver.session.return_value = mock_session

    return mock_driver


@pytest.fixture
def mock_driver_with_standard() -> MagicMock:
    """Create a mock Neo4j driver with a standard."""
    return create_mock_driver_with_results(
        [
            [
                {
                    "name": "ISO 26262",
                    "display_name": "ISO 26262 Functional Safety",
                    "organization": "ISO",
                    "domain": "Automotive",
                    "labels": ["Standard", "Entity"],
                }
            ]
        ]
    )


@pytest.fixture
def mock_driver_with_standard_and_related() -> MagicMock:
    """Create a mock driver that returns standard and related data."""
    return create_mock_driver_with_results(
        [
            # First query: standard lookup
            [
                {
                    "name": "ISO 26262",
                    "display_name": "ISO 26262 Functional Safety",
                    "organization": "ISO",
                    "domain": "Automotive",
                    "labels": ["Standard", "Entity"],
                }
            ],
            # Second query: related entities
            [
                {
                    "name": "ASIL",
                    "display_name": "ASIL Levels",
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


@pytest.fixture
def mock_driver_empty() -> MagicMock:
    """Create a mock Neo4j driver that returns no results."""
    return create_mock_driver_with_results([[]])


# =============================================================================
# Lookup Standard Tests
# =============================================================================


class TestLookupStandard:
    """Tests for lookup_standard function."""

    @pytest.mark.asyncio
    async def test_lookup_standard_found(
        self, mock_driver_with_standard_and_related: MagicMock
    ) -> None:
        """Test looking up an existing standard."""
        result = await lookup_standard(mock_driver_with_standard_and_related, "ISO 26262")

        assert result is not None
        assert result["name"] == "ISO 26262"
        assert result["organization"] == "ISO"

    @pytest.mark.asyncio
    async def test_lookup_standard_not_found(self, mock_driver_empty: MagicMock) -> None:
        """Test looking up a non-existent standard."""
        result = await lookup_standard(mock_driver_empty, "FAKE-12345")

        assert result is None

    @pytest.mark.asyncio
    async def test_lookup_standard_include_related(
        self, mock_driver_with_standard_and_related: MagicMock
    ) -> None:
        """Test that include_related fetches relationships."""
        result = await lookup_standard(
            mock_driver_with_standard_and_related, "ISO 26262", include_related=True
        )

        # Multiple session calls should be made
        mock_session = mock_driver_with_standard_and_related.session.return_value
        assert mock_session.run.call_count == 3
        assert "related" in result
        assert "mentioned_in" in result

    @pytest.mark.asyncio
    async def test_lookup_standard_exclude_related(
        self, mock_driver_with_standard: MagicMock
    ) -> None:
        """Test that include_related=False skips extra queries."""
        await lookup_standard(mock_driver_with_standard, "ISO 26262", include_related=False)

        # Only the main query
        mock_session = mock_driver_with_standard.session.return_value
        assert mock_session.run.call_count == 1


# =============================================================================
# Search Standards Tests
# =============================================================================


class TestSearchStandards:
    """Tests for search_standards function."""

    @pytest.mark.asyncio
    async def test_search_standards_returns_list(
        self, mock_driver_with_standard: MagicMock
    ) -> None:
        """Test that search returns a list of standards."""
        results = await search_standards(mock_driver_with_standard, "automotive")

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_standards_with_industry_filter(
        self, mock_driver_with_standard: MagicMock
    ) -> None:
        """Test filtering by industry."""
        await search_standards(mock_driver_with_standard, "safety", industry="automotive")

        # Query should include industry filter
        mock_session = mock_driver_with_standard.session.return_value.__enter__.return_value
        call_args = mock_session.run.call_args
        assert call_args[1]["industry"] == "automotive"

    @pytest.mark.asyncio
    async def test_search_standards_respects_limit(
        self, mock_driver_with_standard: MagicMock
    ) -> None:
        """Test that limit parameter is used."""
        await search_standards(mock_driver_with_standard, "test", limit=5)

        mock_session = mock_driver_with_standard.session.return_value.__enter__.return_value
        call_args = mock_session.run.call_args
        assert call_args[1]["limit"] == 5


# =============================================================================
# Get Standards by Industry Tests
# =============================================================================


class TestGetStandardsByIndustry:
    """Tests for get_standards_by_industry function."""

    @pytest.mark.asyncio
    async def test_get_standards_automotive(self, mock_driver_with_standard: MagicMock) -> None:
        """Test getting automotive standards."""
        results = await get_standards_by_industry(mock_driver_with_standard, "automotive")

        assert isinstance(results, list)
        # Query should include automotive industry param
        mock_session = mock_driver_with_standard.session.return_value.__enter__.return_value
        call_args = mock_session.run.call_args
        assert call_args[1]["industry"] == "automotive"

    @pytest.mark.asyncio
    async def test_get_standards_medical(self, mock_driver_with_standard: MagicMock) -> None:
        """Test getting medical standards."""
        await get_standards_by_industry(mock_driver_with_standard, "medical")

        mock_session = mock_driver_with_standard.session.return_value.__enter__.return_value
        call_args = mock_session.run.call_args
        assert call_args[1]["industry"] == "medical"

    @pytest.mark.asyncio
    async def test_get_standards_unknown_industry(
        self, mock_driver_with_standard: MagicMock
    ) -> None:
        """Test getting standards for unknown industry uses industry name."""
        await get_standards_by_industry(mock_driver_with_standard, "custom_industry")

        mock_session = mock_driver_with_standard.session.return_value.__enter__.return_value
        call_args = mock_session.run.call_args
        assert call_args[1]["industry"] == "custom_industry"


# =============================================================================
# List All Standards Tests
# =============================================================================


class TestListAllStandards:
    """Tests for list_all_standards function."""

    @pytest.mark.asyncio
    async def test_list_all_standards_returns_list(
        self, mock_driver_with_standard: MagicMock
    ) -> None:
        """Test that list returns a list of standards."""
        results = await list_all_standards(mock_driver_with_standard)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_list_all_standards_default_limit(
        self, mock_driver_with_standard: MagicMock
    ) -> None:
        """Test default limit of 50."""
        await list_all_standards(mock_driver_with_standard)

        mock_session = mock_driver_with_standard.session.return_value.__enter__.return_value
        call_args = mock_session.run.call_args
        assert call_args[1]["limit"] == 50

    @pytest.mark.asyncio
    async def test_list_all_standards_custom_limit(
        self, mock_driver_with_standard: MagicMock
    ) -> None:
        """Test custom limit parameter."""
        await list_all_standards(mock_driver_with_standard, limit=100)

        mock_session = mock_driver_with_standard.session.return_value.__enter__.return_value
        call_args = mock_session.run.call_args
        assert call_args[1]["limit"] == 100
