"""Tests for core text2cypher functions.

Updated Data Model (2026-01):
- Uses direct Neo4j driver instead of LangChain Neo4jGraph
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j.exceptions import ServiceUnavailable

from requirements_graphrag_api.core.text2cypher import (
    Text2CypherResult,
    _execute_cypher,
    _validate_cypher,
    generate_cypher,
    text2cypher_query,
)
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
        with patch("requirements_graphrag_api.core.text2cypher.ChatOpenAI") as mock_llm_class:
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
        with patch("requirements_graphrag_api.core.text2cypher.ChatOpenAI") as mock_llm_class:
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

        with patch("requirements_graphrag_api.core.text2cypher.generate_cypher") as mock_gen:
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
        with patch("requirements_graphrag_api.core.text2cypher.generate_cypher") as mock_gen:
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
            with patch("requirements_graphrag_api.core.text2cypher.generate_cypher") as mock_gen:
                mock_gen.return_value = f"MATCH (n) {forbidden}"

                result = await text2cypher_query(mock_config, mock_driver, "test", execute=True)

                assert "error" in result
                assert result["row_count"] == 0

    @pytest.mark.asyncio
    async def test_text2cypher_query_handles_execution_error(self, mock_config: MagicMock) -> None:
        """Test handling of query execution errors."""
        # Create a driver that raises a Neo4j exception when run is called
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_session.run = MagicMock(side_effect=ServiceUnavailable("Query failed"))
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_session

        with patch("requirements_graphrag_api.core.text2cypher.generate_cypher") as mock_gen:
            mock_gen.return_value = "MATCH (n) RETURN n"

            result = await text2cypher_query(mock_config, mock_driver, "test", execute=True)

            assert "error" in result
            assert "Query failed" in result["error"]
            assert result["row_count"] == 0

    @pytest.mark.asyncio
    async def test_empty_results_include_message(self, mock_config: MagicMock) -> None:
        """Test that empty query results include a user-friendly message."""
        driver = create_mock_driver_with_results([[]])

        with patch("requirements_graphrag_api.core.text2cypher.generate_cypher") as mock_gen:
            mock_gen.return_value = "MATCH (n:Entity) RETURN n LIMIT 10"

            result = await text2cypher_query(mock_config, driver, "Find entities", execute=True)

            assert result["row_count"] == 0
            assert "message" in result
            assert "no matching results" in result["message"]

    @pytest.mark.asyncio
    async def test_non_empty_results_no_message(self, mock_config: MagicMock) -> None:
        """Test that non-empty results do not include a message field."""
        driver = create_mock_driver_with_results([[{"name": "Entity1"}]])

        with patch("requirements_graphrag_api.core.text2cypher.generate_cypher") as mock_gen:
            mock_gen.return_value = "MATCH (n:Entity) RETURN n LIMIT 10"

            result = await text2cypher_query(mock_config, driver, "Find entities", execute=True)

            assert result["row_count"] == 1
            assert "message" not in result


# =============================================================================
# Validate Cypher Tests
# =============================================================================


class TestValidateCypher:
    """Tests for _validate_cypher helper."""

    def test_valid_cypher_returns_none(self) -> None:
        """Valid MATCH query should pass validation."""
        assert _validate_cypher("MATCH (n) RETURN n") is None

    def test_non_cypher_returns_error(self) -> None:
        """Natural language response should fail validation."""
        result = _validate_cypher("I cannot generate a query for that")
        assert result is not None
        assert "cannot be answered" in result

    def test_forbidden_keyword_returns_error(self) -> None:
        """DELETE keyword should be rejected."""
        result = _validate_cypher("MATCH (n) DELETE n")
        assert result is not None
        assert "forbidden keyword" in result.lower()

    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (n) RETURN n OFFSET 10",
            "MATCH (n) WHERE n.name = 'RESET' RETURN n",
            "MATCH (n) WHERE n.type = 'DATASET' RETURN n",
            "MATCH (n) WHERE n.name CONTAINS 'CREATIVE' RETURN n",
            "MATCH (n) WHERE n.action = 'REMOVED_BY' RETURN n",
            "MATCH (n) WHERE n.desc = 'DROPDOWN_MENU' RETURN n",
        ],
    )
    def test_word_boundary_prevents_false_positives(self, cypher: str) -> None:
        """Words containing forbidden substrings should not be rejected."""
        assert _validate_cypher(cypher) is None

    @pytest.mark.parametrize(
        ("cypher", "keyword"),
        [
            ("MATCH (n) DELETE n", "DELETE"),
            ("MATCH (n) MERGE (m:Test)", "MERGE"),
            ("MATCH (n) SET n.x = 1", "SET"),
            ("MATCH (n) REMOVE n.x", "REMOVE"),
            # CREATE/DROP don't start with valid starters, so they fail at the
            # CYPHER_STARTERS check first. Test them in valid contexts:
            ("MATCH (n) WITH n CREATE (m:Test)", "CREATE"),
            ("CALL { DROP INDEX test }", "DROP"),
        ],
    )
    def test_word_boundary_catches_real_keywords(self, cypher: str, keyword: str) -> None:
        """Actual forbidden keywords should still be rejected."""
        result = _validate_cypher(cypher)
        assert result is not None
        assert keyword in result


# =============================================================================
# Execute Cypher Tests
# =============================================================================


class TestExecuteCypher:
    """Tests for _execute_cypher helper."""

    def test_successful_execution(self) -> None:
        """Successful execution returns results and no error."""
        driver = create_mock_driver_with_results([[{"name": "Test"}]])
        results, error = _execute_cypher(driver, "MATCH (n) RETURN n", timeout=15.0)
        assert len(results) == 1
        assert error is None

    def test_execution_error(self) -> None:
        """Execution error returns empty list and error string."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_session.run = MagicMock(side_effect=ServiceUnavailable("Connection lost"))
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_session

        results, error = _execute_cypher(mock_driver, "MATCH (n) RETURN n", timeout=15.0)
        assert results == []
        assert error is not None
        assert "Connection lost" in error


# =============================================================================
# Text2Cypher Timeout Tests
# =============================================================================


class TestText2CypherTimeout:
    """Tests for timeout and retry behavior in text2cypher_query."""

    @pytest.mark.asyncio
    async def test_llm_timeout_triggers_retry(self, mock_config: MagicMock) -> None:
        """Timeout on first attempt should retry, succeed on second."""
        driver = create_mock_driver_with_results([[{"count": 10}]])

        call_count = 0

        async def generate_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("LLM timed out")
            return "MATCH (n) RETURN count(n) AS count"

        with patch(
            "requirements_graphrag_api.core.text2cypher.generate_cypher",
            side_effect=generate_side_effect,
        ):
            result = await text2cypher_query(mock_config, driver, "Count nodes", max_retries=1)

            assert call_count == 2
            assert "error" not in result
            assert result["row_count"] == 1

    @pytest.mark.asyncio
    async def test_neo4j_timeout_uses_query_object(self, mock_config: MagicMock) -> None:
        """Neo4j execution should use Query object with timeout."""
        from neo4j import Query

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([])
        mock_session.run = MagicMock(return_value=mock_result)
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_session

        with patch(
            "requirements_graphrag_api.core.text2cypher.generate_cypher",
            new_callable=AsyncMock,
            return_value="MATCH (n) RETURN n",
        ):
            await text2cypher_query(mock_config, mock_driver, "test", neo4j_timeout=15.0)

            # Verify Query object was passed (not raw string)
            call_args = mock_session.run.call_args
            query_arg = call_args[0][0]
            assert isinstance(query_arg, Query)
            assert query_arg.timeout == 15.0

    @pytest.mark.asyncio
    async def test_validation_error_no_retry(self, mock_config: MagicMock) -> None:
        """Validation errors (forbidden keyword) should not trigger retry."""
        mock_gen = AsyncMock(return_value="MATCH (n) DELETE n")

        with patch(
            "requirements_graphrag_api.core.text2cypher.generate_cypher",
            mock_gen,
        ):
            result = await text2cypher_query(mock_config, MagicMock(), "test", max_retries=2)

            assert mock_gen.call_count == 1
            assert "error" in result

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self, mock_config: MagicMock) -> None:
        """All retries exhausted should return timeout error."""
        mock_gen = AsyncMock(side_effect=TimeoutError("LLM timed out"))

        with patch(
            "requirements_graphrag_api.core.text2cypher.generate_cypher",
            mock_gen,
        ):
            result = await text2cypher_query(mock_config, MagicMock(), "test", max_retries=2)

            assert mock_gen.call_count == 3  # 1 initial + 2 retries
            assert "timed out" in result["error"]
            assert result["row_count"] == 0

    @pytest.mark.asyncio
    async def test_timeout_error_includes_run_id(self, mock_config: MagicMock) -> None:
        """Timeout error response should include run_id when available."""
        mock_gen = AsyncMock(side_effect=TimeoutError("LLM timed out"))
        mock_run_tree = MagicMock()
        mock_run_tree.id = "test-run-id-123"

        with (
            patch(
                "requirements_graphrag_api.core.text2cypher.generate_cypher",
                mock_gen,
            ),
            patch(
                "requirements_graphrag_api.core.text2cypher.get_current_run_tree",
                return_value=mock_run_tree,
            ),
        ):
            result = await text2cypher_query(mock_config, MagicMock(), "test", max_retries=0)

            assert result["run_id"] == "test-run-id-123"

    @pytest.mark.asyncio
    async def test_actual_timeout_behavior(self, mock_config: MagicMock) -> None:
        """Verify asyncio.timeout actually cancels a slow LLM call."""

        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(100)
            return "MATCH (n) RETURN n"

        with patch(
            "requirements_graphrag_api.core.text2cypher.generate_cypher",
            side_effect=slow_generate,
        ):
            result = await text2cypher_query(
                mock_config, MagicMock(), "test", llm_timeout=0.1, max_retries=0
            )

            assert "timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_non_timeout_llm_error_retries(self, mock_config: MagicMock) -> None:
        """Non-timeout LLM errors (API errors) should also trigger retry."""
        call_count = 0

        async def error_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("API rate limited")
            return "MATCH (n) RETURN count(n) AS count"

        driver = create_mock_driver_with_results([[{"count": 10}]])

        with patch(
            "requirements_graphrag_api.core.text2cypher.generate_cypher",
            side_effect=error_then_success,
        ):
            result = await text2cypher_query(mock_config, driver, "Count nodes", max_retries=1)

            assert call_count == 2
            assert "error" not in result

    @pytest.mark.asyncio
    async def test_non_timeout_llm_error_exhausted(self, mock_config: MagicMock) -> None:
        """All retries exhausted for non-timeout LLM errors should return error."""
        mock_gen = AsyncMock(side_effect=RuntimeError("API is down"))

        with patch(
            "requirements_graphrag_api.core.text2cypher.generate_cypher",
            mock_gen,
        ):
            result = await text2cypher_query(mock_config, MagicMock(), "test", max_retries=1)

            assert mock_gen.call_count == 2
            assert "failed" in result["error"]
            assert result["row_count"] == 0


# =============================================================================
# Timeout Param Validation Tests
# =============================================================================


class TestTimeoutParamValidation:
    """Tests for timeout parameter validation guards."""

    @pytest.mark.asyncio
    async def test_negative_llm_timeout_raises(self, mock_config: MagicMock) -> None:
        """Negative llm_timeout should raise ValueError."""
        with pytest.raises(ValueError, match="llm_timeout must be positive"):
            await text2cypher_query(mock_config, MagicMock(), "test", llm_timeout=-1.0)

    @pytest.mark.asyncio
    async def test_zero_llm_timeout_raises(self, mock_config: MagicMock) -> None:
        """Zero llm_timeout should raise ValueError."""
        with pytest.raises(ValueError, match="llm_timeout must be positive"):
            await text2cypher_query(mock_config, MagicMock(), "test", llm_timeout=0)

    @pytest.mark.asyncio
    async def test_negative_neo4j_timeout_raises(self, mock_config: MagicMock) -> None:
        """Negative neo4j_timeout should raise ValueError."""
        with pytest.raises(ValueError, match="neo4j_timeout must be positive"):
            await text2cypher_query(mock_config, MagicMock(), "test", neo4j_timeout=-5.0)

    @pytest.mark.asyncio
    async def test_zero_neo4j_timeout_raises(self, mock_config: MagicMock) -> None:
        """Zero neo4j_timeout should raise ValueError."""
        with pytest.raises(ValueError, match="neo4j_timeout must be positive"):
            await text2cypher_query(mock_config, MagicMock(), "test", neo4j_timeout=0)


# =============================================================================
# Text2CypherResult TypedDict Tests
# =============================================================================


class TestText2CypherResult:
    """Tests for Text2CypherResult TypedDict."""

    def test_typeddict_has_expected_keys(self) -> None:
        """TypedDict should have all expected annotation keys."""
        annotations = Text2CypherResult.__annotations__
        expected = {"question", "cypher", "results", "row_count", "error", "message", "run_id"}
        assert set(annotations.keys()) == expected

    def test_result_is_dict_compatible(self) -> None:
        """TypedDict instances should be regular dicts at runtime."""
        result = Text2CypherResult(question="test", cypher="MATCH (n) RETURN n")
        assert isinstance(result, dict)
        assert result["question"] == "test"
