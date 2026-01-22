"""Pytest fixtures for Jama GraphRAG API tests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from requirements_graphrag_api.config import AppConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


def create_llm_mock(response: str, *, streaming: bool = False) -> MagicMock:
    """Create a mock LLM that works with LangChain's RunnableSequence.

    This helper creates a mock that properly handles being invoked in a chain
    via either `ainvoke`, `invoke`, `astream`, or direct call.

    Args:
        response: The string response the mock should return.
        streaming: If True, configures astream to yield tokens word by word.

    Returns:
        A configured MagicMock that can be used in place of ChatOpenAI.
    """
    mock_llm = MagicMock()

    # Configure ainvoke for async chain invocation
    mock_llm.ainvoke = AsyncMock(return_value=response)

    # Configure invoke for sync chain invocation
    mock_llm.invoke = MagicMock(return_value=response)

    # Configure for direct call (used by some LangChain internals)
    mock_llm.return_value = response

    # Configure astream for async streaming
    async def mock_astream(*_args, **_kwargs):
        """Simulate streaming by yielding words as tokens."""
        words = response.split()
        for i, word in enumerate(words):
            # Add space before word except for first token
            yield word if i == 0 else f" {word}"

    mock_llm.astream = mock_astream

    # Allow the | operator to work with RunnableSequence
    # Don't override __or__ - let LangChain's default behavior handle it
    # The mock will be treated as a Runnable in the sequence

    return mock_llm


@pytest.fixture
def llm_mock_factory() -> Callable[[str], MagicMock]:
    """Factory fixture to create LLM mocks with specific responses.

    Usage:
        def test_something(llm_mock_factory):
            mock_llm = llm_mock_factory('{"key": "value"}')
            with patch("module.ChatOpenAI", return_value=mock_llm):
                # test code
    """
    return create_llm_mock


# Test credentials - not real secrets
_TEST_PASSWORD = "test-password"
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
