"""Tests for observability module."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import pytest

from jama_mcp_server_graphrag.config import AppConfig
from jama_mcp_server_graphrag.observability import (
    configure_tracing,
    disable_tracing,
    get_tracing_status,
    traceable,
)

if TYPE_CHECKING:
    from collections.abc import Generator

_TEST_API_KEY = "lsv2_test_api_key_12345"
_TEST_PASSWORD = "test"  # noqa: S105


@pytest.fixture
def clean_env() -> Generator[None, None, None]:
    """Clean LangSmith environment variables before and after tests."""
    env_vars = [
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGCHAIN_PROJECT",
    ]

    # Store original values
    original = {k: os.environ.get(k) for k in env_vars}

    # Remove all
    for var in env_vars:
        os.environ.pop(var, None)

    yield

    # Restore original values
    for var in env_vars:
        if original[var] is not None:
            os.environ[var] = original[var]
        else:
            os.environ.pop(var, None)


def _make_config_tracing_enabled() -> AppConfig:
    """Create a config with tracing enabled."""
    return AppConfig(
        neo4j_uri="neo4j://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password=_TEST_PASSWORD,
        langsmith_api_key=_TEST_API_KEY,
        langsmith_project="test-project",
        langsmith_tracing_enabled=True,
    )


def _make_config_tracing_disabled() -> AppConfig:
    """Create a config with tracing disabled."""
    return AppConfig(
        neo4j_uri="neo4j://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password=_TEST_PASSWORD,
        langsmith_api_key=_TEST_API_KEY,
        langsmith_project="test-project",
        langsmith_tracing_enabled=False,
    )


def _make_config_no_api_key() -> AppConfig:
    """Create a config with tracing enabled but no API key."""
    return AppConfig(
        neo4j_uri="neo4j://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password=_TEST_PASSWORD,
        langsmith_api_key="",
        langsmith_project="test-project",
        langsmith_tracing_enabled=True,
    )


class TestConfigureTracing:
    """Tests for configure_tracing function."""

    def test_configure_tracing_enabled(self, clean_env: None) -> None:
        """Test that tracing is configured when enabled with API key."""
        _ = clean_env  # Ensure environment is clean
        config = _make_config_tracing_enabled()
        result = configure_tracing(config)

        assert result is True
        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGSMITH_API_KEY"] == _TEST_API_KEY
        assert os.environ["LANGCHAIN_API_KEY"] == _TEST_API_KEY
        assert os.environ["LANGSMITH_PROJECT"] == "test-project"
        assert os.environ["LANGCHAIN_PROJECT"] == "test-project"

    def test_configure_tracing_disabled(self, clean_env: None) -> None:
        """Test that tracing is not configured when disabled."""
        _ = clean_env  # Ensure environment is clean
        config = _make_config_tracing_disabled()
        result = configure_tracing(config)

        assert result is False
        # Environment should not be modified
        assert os.environ.get("LANGSMITH_TRACING") is None

    def test_configure_tracing_no_api_key(self, clean_env: None) -> None:
        """Test that tracing is not configured without API key."""
        _ = clean_env  # Ensure environment is clean
        config = _make_config_no_api_key()
        result = configure_tracing(config)

        assert result is False
        # Environment should not be modified
        assert os.environ.get("LANGSMITH_TRACING") is None


class TestDisableTracing:
    """Tests for disable_tracing function."""

    def test_disable_tracing(self, clean_env: None) -> None:
        """Test that disable_tracing sets environment variables to false."""
        _ = clean_env  # Ensure environment is clean
        # First enable tracing
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

        disable_tracing()

        assert os.environ["LANGSMITH_TRACING"] == "false"
        assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


class TestGetTracingStatus:
    """Tests for get_tracing_status function."""

    def test_get_status_enabled(self, clean_env: None) -> None:
        """Test status when tracing is enabled."""
        _ = clean_env  # Ensure environment is clean
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = "my-project"
        os.environ["LANGSMITH_API_KEY"] = _TEST_API_KEY

        status = get_tracing_status()

        assert status["tracing_enabled"] is True
        assert status["project"] == "my-project"
        assert status["api_key_set"] is True

    def test_get_status_disabled(self, clean_env: None) -> None:
        """Test status when tracing is disabled."""
        _ = clean_env  # Ensure environment is clean
        status = get_tracing_status()

        assert status["tracing_enabled"] is False
        assert status["project"] == "default"
        assert status["api_key_set"] is False

    def test_get_status_partial_config(self, clean_env: None) -> None:
        """Test status with partial configuration."""
        _ = clean_env  # Ensure environment is clean
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGSMITH_PROJECT"] = "partial-project"

        status = get_tracing_status()

        assert status["tracing_enabled"] is False
        assert status["project"] == "partial-project"
        assert status["api_key_set"] is False


class TestTraceableReexport:
    """Tests for traceable decorator re-export."""

    def test_traceable_is_importable(self) -> None:
        """Test that traceable can be imported from observability module."""
        assert traceable is not None

    def test_traceable_is_callable(self) -> None:
        """Test that traceable is a callable decorator."""
        assert callable(traceable)

    def test_traceable_can_decorate_function(self) -> None:
        """Test that traceable can be used as a decorator."""

        @traceable(name="test_func", run_type="chain")
        def sample_function(x: int) -> int:
            return x * 2

        # Function should still work when decorated
        result = sample_function(5)
        assert result == 10

    def test_traceable_can_decorate_async_function(self) -> None:
        """Test that traceable can be used with async functions."""

        @traceable(name="async_test_func", run_type="chain")
        async def async_sample_function(x: int) -> int:
            return x * 2

        # Function should still work when decorated
        result = asyncio.run(async_sample_function(5))
        assert result == 10
