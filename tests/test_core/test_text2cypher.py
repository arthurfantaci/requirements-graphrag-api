"""Tests for core text2cypher functions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jama_mcp_server_graphrag.core.text2cypher import generate_cypher, text2cypher_query

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
    graph.schema = "Node labels: Entity, Concept, Standard, GlossaryTerm"
    graph.query = MagicMock(
        return_value=[
            {"count": 100},
        ]
    )
    return graph


# =============================================================================
# Generate Cypher Tests
# =============================================================================


class TestGenerateCypher:
    """Tests for generate_cypher function."""

    @pytest.mark.asyncio
    async def test_generate_cypher_returns_string(
        self, mock_config: MagicMock, mock_graph: MagicMock
    ) -> None:
        """Test that generate_cypher returns a Cypher query string."""
        with patch(
            "jama_mcp_server_graphrag.core.text2cypher.ChatOpenAI"
        ) as mock_llm_class:
            # Setup mock LLM chain
            mock_llm = MagicMock()
            mock_llm.__or__ = MagicMock(return_value=mock_llm)
            mock_llm.ainvoke = AsyncMock(
                return_value="MATCH (n:Entity) RETURN count(n) AS count"
            )
            mock_llm_class.return_value = mock_llm

            result = await generate_cypher(
                mock_config, mock_graph, "How many entities are there?"
            )

            assert isinstance(result, str)
            assert "MATCH" in result or "RETURN" in result

    @pytest.mark.asyncio
    async def test_generate_cypher_strips_markdown(
        self, mock_config: MagicMock, mock_graph: MagicMock
    ) -> None:
        """Test that markdown code blocks are stripped from response."""
        with patch(
            "jama_mcp_server_graphrag.core.text2cypher.ChatOpenAI"
        ) as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.__or__ = MagicMock(return_value=mock_llm)
            mock_llm.ainvoke = AsyncMock(
                return_value="```cypher\nMATCH (n) RETURN n\n```"
            )
            mock_llm_class.return_value = mock_llm

            result = await generate_cypher(mock_config, mock_graph, "Get all nodes")

            assert "```" not in result
            assert "MATCH (n) RETURN n" in result


# =============================================================================
# Text2Cypher Query Tests
# =============================================================================


class TestText2CypherQuery:
    """Tests for text2cypher_query function."""

    @pytest.mark.asyncio
    async def test_text2cypher_query_with_execution(
        self, mock_config: MagicMock, mock_graph: MagicMock
    ) -> None:
        """Test query generation and execution."""
        with patch(
            "jama_mcp_server_graphrag.core.text2cypher.generate_cypher"
        ) as mock_gen:
            mock_gen.return_value = "MATCH (n:Entity) RETURN count(n) AS count"

            result = await text2cypher_query(
                mock_config, mock_graph, "How many entities?", execute=True
            )

            assert "question" in result
            assert "cypher" in result
            assert "results" in result
            assert result["row_count"] == 1

    @pytest.mark.asyncio
    async def test_text2cypher_query_without_execution(
        self, mock_config: MagicMock, mock_graph: MagicMock
    ) -> None:
        """Test query generation without execution."""
        with patch(
            "jama_mcp_server_graphrag.core.text2cypher.generate_cypher"
        ) as mock_gen:
            mock_gen.return_value = "MATCH (n) RETURN n LIMIT 10"

            result = await text2cypher_query(
                mock_config, mock_graph, "Get nodes", execute=False
            )

            assert "cypher" in result
            assert "results" not in result
            mock_graph.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_text2cypher_query_blocks_write_operations(
        self, mock_config: MagicMock, mock_graph: MagicMock
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
            with patch(
                "jama_mcp_server_graphrag.core.text2cypher.generate_cypher"
            ) as mock_gen:
                mock_gen.return_value = f"MATCH (n) {forbidden}"

                result = await text2cypher_query(
                    mock_config, mock_graph, "test", execute=True
                )

                assert "error" in result
                assert result["row_count"] == 0

    @pytest.mark.asyncio
    async def test_text2cypher_query_handles_execution_error(
        self, mock_config: MagicMock, mock_graph: MagicMock
    ) -> None:
        """Test handling of query execution errors."""
        mock_graph.query = MagicMock(side_effect=Exception("Query failed"))

        with patch(
            "jama_mcp_server_graphrag.core.text2cypher.generate_cypher"
        ) as mock_gen:
            mock_gen.return_value = "MATCH (n) RETURN n"

            result = await text2cypher_query(
                mock_config, mock_graph, "test", execute=True
            )

            assert "error" in result
            assert "Query failed" in result["error"]
            assert result["row_count"] == 0
