"""Tests for observability module."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import pytest

from requirements_graphrag_api.config import AppConfig
from requirements_graphrag_api.observability import (
    REDACTED,
    configure_tracing,
    create_thread_metadata,
    disable_tracing,
    get_tracing_status,
    sanitize_inputs,
    traceable,
    traceable_safe,
)

if TYPE_CHECKING:
    from collections.abc import Generator

_TEST_API_KEY = "lsv2_test_api_key_12345"
_TEST_PASSWORD = "test"


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


class TestSanitizeInputs:
    """Tests for sanitize_inputs function."""

    def test_sanitize_password_field(self) -> None:
        """Test that password fields are redacted."""
        inputs = {"username": "user", "password": "secret123"}
        result = sanitize_inputs(inputs)

        assert result["username"] == "user"
        assert result["password"] == REDACTED

    def test_sanitize_api_key_field(self) -> None:
        """Test that api_key fields are redacted."""
        inputs = {"model": "gpt-4", "openai_api_key": "sk-12345"}
        result = sanitize_inputs(inputs)

        assert result["model"] == "gpt-4"
        assert result["openai_api_key"] == REDACTED

    def test_sanitize_nested_dict(self) -> None:
        """Test that nested dictionaries are sanitized."""
        inputs = {
            "config": {
                "neo4j_password": "db-secret",
                "neo4j_username": "neo4j",
            },
            "query": "test query",
        }
        result = sanitize_inputs(inputs)

        assert result["query"] == "test query"
        assert result["config"]["neo4j_username"] == "neo4j"
        assert result["config"]["neo4j_password"] == REDACTED

    def test_sanitize_dataclass(self) -> None:
        """Test that dataclasses are converted and sanitized."""
        config = AppConfig(
            neo4j_uri="neo4j://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="secret-password",  # noqa: S106
            openai_api_key="sk-openai-key",
            langsmith_api_key="lsv2_key",
        )
        inputs = {"config": config, "message": "hello"}
        result = sanitize_inputs(inputs)

        assert result["message"] == "hello"
        # Config should be converted to dict and sanitized
        assert isinstance(result["config"], dict)
        assert result["config"]["neo4j_username"] == "neo4j"
        assert result["config"]["neo4j_password"] == REDACTED
        assert result["config"]["openai_api_key"] == REDACTED
        assert result["config"]["langsmith_api_key"] == REDACTED

    def test_sanitize_list_of_dicts(self) -> None:
        """Test that lists containing dicts are sanitized."""
        inputs = {
            "credentials": [
                {"service": "db", "password": "pass1"},
                {"service": "api", "api_key": "key2"},
            ]
        }
        result = sanitize_inputs(inputs)

        assert result["credentials"][0]["service"] == "db"
        assert result["credentials"][0]["password"] == REDACTED
        assert result["credentials"][1]["service"] == "api"
        assert result["credentials"][1]["api_key"] == REDACTED

    def test_sanitize_preserves_non_sensitive_data(self) -> None:
        """Test that non-sensitive data is preserved."""
        inputs = {
            "message": "What is a requirement?",
            "max_sources": 5,
            "include_entities": True,
            "metadata": {"topic": "requirements"},
        }
        result = sanitize_inputs(inputs)

        assert result == inputs  # Should be identical

    def test_sanitize_various_sensitive_patterns(self) -> None:
        """Test various sensitive field name patterns."""
        inputs = {
            "password": "pass1",
            "neo4j_password": "pass2",
            "api_key": "key1",
            "openai_api_key": "key2",
            "langsmith_api_key": "key3",
            "secret": "sec1",
            "client_secret": "sec2",
            "auth_token": "tok1",
            "bearer_token": "tok2",
            "credentials": "cred1",
        }
        result = sanitize_inputs(inputs)

        # All should be redacted
        for key in inputs:
            assert result[key] == REDACTED, f"Expected {key} to be redacted"

    def test_sanitize_case_insensitive(self) -> None:
        """Test that field matching is case-insensitive."""
        inputs = {
            "PASSWORD": "pass1",
            "Api_Key": "key1",
            "SecReT": "sec1",
        }
        result = sanitize_inputs(inputs)

        assert result["PASSWORD"] == REDACTED
        assert result["Api_Key"] == REDACTED
        assert result["SecReT"] == REDACTED

    def test_sanitize_empty_inputs(self) -> None:
        """Test sanitization of empty inputs."""
        assert sanitize_inputs({}) == {}

    def test_sanitize_non_dict_returns_unchanged(self) -> None:
        """Test that non-dict inputs are returned unchanged."""
        assert sanitize_inputs("string") == "string"  # type: ignore[arg-type]
        assert sanitize_inputs(123) == 123  # type: ignore[arg-type]
        assert sanitize_inputs(None) is None  # type: ignore[arg-type]


class TestTraceableSafe:
    """Tests for traceable_safe decorator."""

    def test_traceable_safe_sync_function(self) -> None:
        """Test that traceable_safe works with sync functions."""

        @traceable_safe(name="test_sync", run_type="chain")
        def sync_func(x: int) -> int:
            return x * 2

        result = sync_func(5)
        assert result == 10

    def test_traceable_safe_async_function(self) -> None:
        """Test that traceable_safe works with async functions."""

        @traceable_safe(name="test_async", run_type="chain")
        async def async_func(x: int) -> int:
            return x * 2

        result = asyncio.run(async_func(5))
        assert result == 10

    def test_traceable_safe_preserves_function_name(self) -> None:
        """Test that traceable_safe preserves function metadata."""

        @traceable_safe(name="my_func", run_type="chain")
        async def my_function(x: int) -> int:
            """My docstring."""
            return x

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_traceable_safe_with_config_param(self) -> None:
        """Test that traceable_safe works with AppConfig parameter."""
        config = AppConfig(
            neo4j_uri="neo4j://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="secret",  # noqa: S106
            openai_api_key="sk-key",
        )

        @traceable_safe(name="test_with_config", run_type="chain")
        async def func_with_config(config: AppConfig, query: str) -> str:
            return f"Query: {query}"

        result = asyncio.run(func_with_config(config, "test"))
        assert result == "Query: test"


class TestCreateThreadMetadata:
    """Tests for create_thread_metadata function."""

    def test_create_thread_metadata_with_id(self) -> None:
        """Test that metadata is created with thread_id for LangSmith Threads."""
        result = create_thread_metadata("conversation-123")

        assert result is not None
        assert "metadata" in result
        # LangSmith requires 'thread_id' key for Thread grouping
        assert result["metadata"]["thread_id"] == "conversation-123"

    def test_create_thread_metadata_without_id(self) -> None:
        """Test that None is returned without conversation_id."""
        assert create_thread_metadata(None) is None
        assert create_thread_metadata("") is None

    def test_create_thread_metadata_preserves_uuid_format(self) -> None:
        """Test that UUID-style conversation IDs are preserved."""
        uuid_id = "550e8400-e29b-41d4-a716-446655440000"
        result = create_thread_metadata(uuid_id)

        assert result is not None
        # LangSmith requires 'thread_id' key for Thread grouping
        assert result["metadata"]["thread_id"] == uuid_id
