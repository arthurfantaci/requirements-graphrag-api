"""Tests for core text2cypher functions.

Updated Data Model (2026-01):
- Uses direct Neo4j driver instead of LangChain Neo4jGraph
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from jama_mcp_server_graphrag.core.text2cypher import generate_cypher, text2cypher_query
from tests.conftest import create_llm_mock

# =============================================================================
# Mock Helpers
# =============================================================================


def create_mock_record(data: dict[str, Any]) -> MagicMock:
    """Create a mock Neo4j record."""
    record = MagicMock()
    record.__getitem__ = lambda s, k: data.get(k)
    record.get = lambda k, d=None: data.get(k, d)
    record.data = lambda: data
    # For dict(record) conversion
    record.keys = lambda: data.keys()
    record.values = lambda: data.values()
    record.items = lambda: data.items()
    return record


def create_mock_driver_with_results(results_sequence: list[list[dict[str, Any]]]) -> MagicMock:
    """Create a mock Neo4j driver."""
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
        else:
            mock_result.__iter__ = lambda self: iter([])

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
def mock_config() -> MagicMock:
    """Create a mock AppConfig."""
    config = MagicMock()
    config.chat_model = "gpt-4o"
    config.openai_api_key = "test-api-key"
    return config


@pytest.fixture
def mock_driver() -> MagicMock:
    """Create a mock Neo4j driver."""
    return create_mock_driver_with_results(
        [
            # Schema query result
            [{"label": "Entity", "count": 100}, {"label": "Concept", "count": 50}],
            # Query execution result
            [{"count": 100}],
        ]
    )


# =============================================================================
# Generate Cypher Tests
# =============================================================================


class TestGenerateCypher:
    """Tests for generate_cypher function."""

    @pytest.mark.asyncio
    async def test_generate_cypher_returns_string(
        self, mock_config: MagicMock, mock_driver: MagicMock
    ) -> None:
        """Test that generate_cypher returns a Cypher query string."""
        with patch("jama_mcp_server_graphrag.core.text2cypher.ChatOpenAI") as mock_llm_class:
            mock_llm_class.return_value = create_llm_mock(
                "MATCH (n:Entity) RETURN count(n) AS count"
            )

            result = await generate_cypher(mock_config, mock_driver, "How many entities are there?")

            assert isinstance(result, str)
            assert "MATCH" in result or "RETURN" in result

    @pytest.mark.asyncio
    async def test_generate_cypher_strips_markdown(
        self, mock_config: MagicMock, mock_driver: MagicMock
    ) -> None:
        """Test that markdown code blocks are stripped from response."""
        with patch("jama_mcp_server_graphrag.core.text2cypher.ChatOpenAI") as mock_llm_class:
            mock_llm_class.return_value = create_llm_mock("```cypher\nMATCH (n) RETURN n\n```")

            result = await generate_cypher(mock_config, mock_driver, "Get all nodes")

            assert "```" not in result
            assert "MATCH (n) RETURN n" in result


# =============================================================================
# Text2Cypher Query Tests
# =============================================================================


class TestText2CypherQuery:
    """Tests for text2cypher_query function."""

    @pytest.mark.asyncio
    async def test_text2cypher_query_with_execution(self, mock_config: MagicMock) -> None:
        """Test query generation and execution."""
        # Create a driver that returns 1 result for the execution query
        driver = create_mock_driver_with_results(
            [
                [{"count": 100}],
            ]
        )

        with patch("jama_mcp_server_graphrag.core.text2cypher.generate_cypher") as mock_gen:
            mock_gen.return_value = "MATCH (n:Entity) RETURN count(n) AS count"

            result = await text2cypher_query(
                mock_config, driver, "How many entities?", execute=True
            )

            assert "question" in result
            assert "cypher" in result
            assert "results" in result
            assert result["row_count"] == 1

    @pytest.mark.asyncio
    async def test_text2cypher_query_without_execution(
        self, mock_config: MagicMock, mock_driver: MagicMock
    ) -> None:
        """Test query generation without execution."""
        with patch("jama_mcp_server_graphrag.core.text2cypher.generate_cypher") as mock_gen:
            mock_gen.return_value = "MATCH (n) RETURN n LIMIT 10"

            result = await text2cypher_query(mock_config, mock_driver, "Get nodes", execute=False)

            assert "cypher" in result
            assert "results" not in result

    @pytest.mark.asyncio
    async def test_text2cypher_query_blocks_write_operations(
        self, mock_config: MagicMock, mock_driver: MagicMock
    ) -> None:
        """Test that write operations are blocked."""
        forbidden_queries = [
            "DELETE n",
            "CREATE (n:Test)",
            "MERGE (n:Test)",
            "SET n.prop = 'value'",
            "REMOVE n.prop",
            "DROP INDEX test",
        ]

        for forbidden in forbidden_queries:
            with patch("jama_mcp_server_graphrag.core.text2cypher.generate_cypher") as mock_gen:
                mock_gen.return_value = f"MATCH (n) {forbidden}"

                result = await text2cypher_query(mock_config, mock_driver, "test", execute=True)

                assert "error" in result
                assert result["row_count"] == 0

    @pytest.mark.asyncio
    async def test_text2cypher_query_handles_execution_error(self, mock_config: MagicMock) -> None:
        """Test handling of query execution errors."""
        # Create a driver that raises an exception when run is called
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_session.run = MagicMock(side_effect=Exception("Query failed"))
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_session

        with patch("jama_mcp_server_graphrag.core.text2cypher.generate_cypher") as mock_gen:
            mock_gen.return_value = "MATCH (n) RETURN n"

            result = await text2cypher_query(mock_config, mock_driver, "test", execute=True)

            assert "error" in result
            assert "Query failed" in result["error"]
            assert result["row_count"] == 0
