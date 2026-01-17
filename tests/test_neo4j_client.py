"""Tests for Neo4j client module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from neo4j.exceptions import AuthError, ServiceUnavailable

from jama_mcp_server_graphrag.exceptions import Neo4jConnectionError
from jama_mcp_server_graphrag.neo4j_client import (
    create_driver,
    execute_read_query,
    execute_read_with_bookmark,
    execute_write_query,
)


class TestCreateDriver:
    """Tests for create_driver function."""

    def test_create_driver_success(self, mock_config, mock_neo4j_driver):
        """Test successful driver creation with connectivity verification."""
        driver = create_driver(mock_config)

        assert driver is mock_neo4j_driver
        mock_neo4j_driver.verify_connectivity.assert_called_once()

    def test_create_driver_with_correct_params(self, mock_config):
        """Test that driver is created with correct parameters."""
        with patch("jama_mcp_server_graphrag.neo4j_client.GraphDatabase") as mock_gdb:
            mock_driver = MagicMock()
            mock_gdb.driver.return_value = mock_driver

            create_driver(mock_config)

            mock_gdb.driver.assert_called_once_with(
                mock_config.neo4j_uri,
                auth=(mock_config.neo4j_username, mock_config.neo4j_password),
                max_connection_pool_size=mock_config.neo4j_max_connection_pool_size,
                connection_acquisition_timeout=mock_config.neo4j_connection_acquisition_timeout,
            )

    def test_create_driver_auth_error_raises_connection_error(self, mock_config):
        """Test that AuthError is wrapped in Neo4jConnectionError."""
        with patch("jama_mcp_server_graphrag.neo4j_client.GraphDatabase") as mock_gdb:
            mock_driver = MagicMock()
            mock_driver.verify_connectivity.side_effect = AuthError("Invalid credentials")
            mock_gdb.driver.return_value = mock_driver

            with pytest.raises(Neo4jConnectionError, match="Authentication failed"):
                create_driver(mock_config)

    def test_create_driver_service_unavailable_raises_connection_error(self, mock_config):
        """Test that ServiceUnavailable is wrapped in Neo4jConnectionError."""
        with patch("jama_mcp_server_graphrag.neo4j_client.GraphDatabase") as mock_gdb:
            mock_driver = MagicMock()
            mock_driver.verify_connectivity.side_effect = ServiceUnavailable("Cannot connect")
            mock_gdb.driver.return_value = mock_driver

            with pytest.raises(Neo4jConnectionError, match="Service unavailable"):
                create_driver(mock_config)

    def test_create_driver_generic_error_raises_connection_error(self, mock_config):
        """Test that generic errors are wrapped in Neo4jConnectionError."""
        with patch("jama_mcp_server_graphrag.neo4j_client.GraphDatabase") as mock_gdb:
            mock_gdb.driver.side_effect = RuntimeError("Unexpected error")

            with pytest.raises(Neo4jConnectionError, match="Driver creation failed"):
                create_driver(mock_config)

    def test_create_driver_preserves_original_exception(self, mock_config):
        """Test that original exception is chained."""
        with patch("jama_mcp_server_graphrag.neo4j_client.GraphDatabase") as mock_gdb:
            original_error = AuthError("Invalid credentials")
            mock_driver = MagicMock()
            mock_driver.verify_connectivity.side_effect = original_error
            mock_gdb.driver.return_value = mock_driver

            with pytest.raises(Neo4jConnectionError) as exc_info:
                create_driver(mock_config)

            assert exc_info.value.__cause__ is original_error


class TestExecuteReadQuery:
    """Tests for execute_read_query function."""

    def test_execute_read_query_returns_results(self):
        """Test that execute_read_query returns query results."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        expected_results = [{"name": "test1"}, {"name": "test2"}]
        mock_session.execute_read.return_value = expected_results

        results = execute_read_query(
            mock_driver,
            "MATCH (n) RETURN n.name as name",
            database="neo4j",
        )

        assert results == expected_results

    def test_execute_read_query_uses_execute_read(self):
        """Test that execute_read_query uses session.execute_read for cluster routing."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        execute_read_query(mock_driver, "MATCH (n) RETURN n")

        mock_session.execute_read.assert_called_once()

    def test_execute_read_query_with_parameters(self):
        """Test that execute_read_query passes parameters correctly."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        query = "MATCH (n) WHERE n.name = $name RETURN n"
        parameters = {"name": "test"}

        execute_read_query(mock_driver, query, parameters=parameters)

        # Verify execute_read was called with the query and parameters
        call_args = mock_session.execute_read.call_args
        assert call_args is not None
        # The first positional arg is the transaction function
        # The remaining args are query and parameters
        assert call_args[0][1] == query
        assert call_args[0][2] == parameters

    def test_execute_read_query_uses_specified_database(self):
        """Test that execute_read_query uses the specified database."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        execute_read_query(mock_driver, "MATCH (n) RETURN n", database="custom_db")

        mock_driver.session.assert_called_once_with(database="custom_db")


class TestExecuteWriteQuery:
    """Tests for execute_write_query function."""

    def test_execute_write_query_returns_results(self):
        """Test that execute_write_query returns query results."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        expected_results = [{"created": 1}]
        mock_session.execute_write.return_value = expected_results

        results = execute_write_query(
            mock_driver,
            "CREATE (n:Node {name: $name}) RETURN count(n) as created",
            parameters={"name": "test"},
        )

        assert results == expected_results

    def test_execute_write_query_uses_execute_write(self):
        """Test that execute_write_query uses session.execute_write for leader routing."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        execute_write_query(mock_driver, "CREATE (n:Node)")

        mock_session.execute_write.assert_called_once()

    def test_execute_write_query_with_parameters(self):
        """Test that execute_write_query passes parameters correctly."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        query = "CREATE (n:Node {name: $name})"
        parameters = {"name": "test"}

        execute_write_query(mock_driver, query, parameters=parameters)

        call_args = mock_session.execute_write.call_args
        assert call_args is not None
        assert call_args[0][1] == query
        assert call_args[0][2] == parameters


class TestExecuteReadWithBookmark:
    """Tests for execute_read_with_bookmark function."""

    def test_execute_read_with_bookmark_returns_results_and_bookmark(self):
        """Test that execute_read_with_bookmark returns results and new bookmark."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        expected_results = [{"name": "test"}]
        expected_bookmark = "bookmark:12345"
        mock_session.execute_read.return_value = expected_results
        mock_session.last_bookmarks.return_value = expected_bookmark

        results, bookmark = execute_read_with_bookmark(
            mock_driver, "MATCH (n) RETURN n.name as name"
        )

        assert results == expected_results
        assert bookmark == expected_bookmark

    def test_execute_read_with_bookmark_passes_bookmarks_to_session(self):
        """Test that bookmarks are passed to the session."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        previous_bookmarks = ["bookmark:prev"]

        execute_read_with_bookmark(
            mock_driver,
            "MATCH (n) RETURN n",
            bookmarks=previous_bookmarks,
        )

        mock_driver.session.assert_called_once_with(database="neo4j", bookmarks=previous_bookmarks)


class TestQueryParameterSecurity:
    """Tests verifying query parameter usage for security."""

    def test_read_query_uses_parameters_not_string_concat(self):
        """Test that queries use parameters instead of string concatenation."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        # Safe query with parameter placeholder
        safe_query = "MATCH (n) WHERE n.name = $name RETURN n"
        user_input = "test'; DROP DATABASE neo4j; --"

        execute_read_query(mock_driver, safe_query, parameters={"name": user_input})

        # Verify the query template was passed, not a concatenated string
        call_args = mock_session.execute_read.call_args
        passed_query = call_args[0][1]
        assert "$name" in passed_query
        assert user_input not in passed_query
