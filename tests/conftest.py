"""Pytest fixtures for Jama MCP Server GraphRAG tests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from jama_mcp_server_graphrag.config import AppConfig

if TYPE_CHECKING:
    from collections.abc import Generator

# Test credentials - not real secrets
_TEST_PASSWORD = "test-password"  # noqa: S105
_TEST_API_KEY = "sk-test-key"


@pytest.fixture
def mock_config() -> AppConfig:
    """Create a mock configuration for testing."""
    return AppConfig(
        neo4j_uri="neo4j+s://test.databases.neo4j.io",
        neo4j_username="neo4j",
        neo4j_password=_TEST_PASSWORD,
        neo4j_database="neo4j",
        openai_api_key=_TEST_API_KEY,
        chat_model="gpt-4o",
        embedding_model="text-embedding-ada-002",
        vector_index_name="chunk_embeddings",
        similarity_k=6,
        log_level="INFO",
        neo4j_max_connection_pool_size=5,
        neo4j_connection_acquisition_timeout=30.0,
    )


@pytest.fixture
def mock_local_config() -> AppConfig:
    """Create a mock configuration with local Neo4j URI for testing."""
    return AppConfig(
        neo4j_uri="neo4j://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password=_TEST_PASSWORD,
    )


@pytest.fixture
def mock_neo4j_driver() -> Generator[MagicMock, None, None]:
    """Create a mock Neo4j driver for testing."""
    with patch("jama_mcp_server_graphrag.neo4j_client.GraphDatabase") as mock_gdb:
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver
        yield mock_driver


@pytest.fixture
def env_vars() -> Generator[dict[str, str], None, None]:
    """Set up environment variables for testing."""
    test_vars = {
        "NEO4J_URI": "neo4j+s://test.databases.neo4j.io",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": _TEST_PASSWORD,
        "NEO4J_DATABASE": "neo4j",
        "OPENAI_API_KEY": _TEST_API_KEY,
        "OPENAI_MODEL": "gpt-4o",
        "EMBEDDING_MODEL": "text-embedding-ada-002",
        "VECTOR_INDEX_NAME": "chunk_embeddings",
        "SIMILARITY_K": "6",
        "LOG_LEVEL": "INFO",
        "NEO4J_MAX_POOL_SIZE": "5",
        "NEO4J_CONNECTION_TIMEOUT": "30.0",
    }
    with patch.dict("os.environ", test_vars, clear=False):
        yield test_vars


@pytest.fixture
def minimal_env_vars() -> Generator[dict[str, str], None, None]:
    """Set up minimal required environment variables for testing."""
    test_vars = {
        "NEO4J_URI": "neo4j+s://test.databases.neo4j.io",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": _TEST_PASSWORD,
    }
    with patch.dict("os.environ", test_vars, clear=True):
        yield test_vars
