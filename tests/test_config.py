"""Tests for configuration module."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from jama_mcp_server_graphrag.config import (
    SECURE_NEO4J_SCHEMES,
    VALID_NEO4J_SCHEMES,
    AppConfig,
    ConfigurationError,
    get_config,
)


class TestAppConfig:
    """Tests for AppConfig dataclass."""

    def test_valid_config_with_secure_uri(self, mock_config):
        """Test that valid configuration with secure URI is accepted."""
        assert mock_config.neo4j_uri == "neo4j+s://test.databases.neo4j.io"
        assert mock_config.neo4j_username == "neo4j"
        assert mock_config.neo4j_password == "test-password"  # noqa: S105
        assert mock_config.similarity_k == 6

    def test_valid_config_with_local_uri(self, mock_local_config):
        """Test that valid configuration with local URI is accepted."""
        assert mock_local_config.neo4j_uri == "neo4j://localhost:7687"
        assert mock_local_config.neo4j_database == "neo4j"  # default

    def test_invalid_uri_scheme_raises_error(self):
        """Test that invalid URI scheme raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Invalid Neo4j URI scheme"):
            AppConfig(
                neo4j_uri="http://localhost:7687",
                neo4j_username="neo4j",
                neo4j_password="password",  # noqa: S106
            )

    @pytest.mark.parametrize("scheme", VALID_NEO4J_SCHEMES)
    def test_all_valid_schemes_accepted(self, scheme):
        """Test that all valid URI schemes are accepted."""
        config = AppConfig(
            neo4j_uri=f"{scheme}localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="password",  # noqa: S106
        )
        assert config.neo4j_uri.startswith(scheme)

    def test_similarity_k_too_low_raises_error(self):
        """Test that similarity_k below 1 raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="similarity_k must be between 1 and 100"):
            AppConfig(
                neo4j_uri="neo4j://localhost:7687",
                neo4j_username="neo4j",
                neo4j_password="password",  # noqa: S106
                similarity_k=0,
            )

    def test_similarity_k_too_high_raises_error(self):
        """Test that similarity_k above 100 raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="similarity_k must be between 1 and 100"):
            AppConfig(
                neo4j_uri="neo4j://localhost:7687",
                neo4j_username="neo4j",
                neo4j_password="password",  # noqa: S106
                similarity_k=101,
            )

    def test_connection_pool_size_too_low_raises_error(self):
        """Test that pool size below 1 raises ConfigurationError."""
        with pytest.raises(
            ConfigurationError, match="neo4j_max_connection_pool_size must be between 1 and 100"
        ):
            AppConfig(
                neo4j_uri="neo4j://localhost:7687",
                neo4j_username="neo4j",
                neo4j_password="password",  # noqa: S106
                neo4j_max_connection_pool_size=0,
            )

    def test_connection_pool_size_too_high_raises_error(self):
        """Test that pool size above 100 raises ConfigurationError."""
        with pytest.raises(
            ConfigurationError, match="neo4j_max_connection_pool_size must be between 1 and 100"
        ):
            AppConfig(
                neo4j_uri="neo4j://localhost:7687",
                neo4j_username="neo4j",
                neo4j_password="password",  # noqa: S106
                neo4j_max_connection_pool_size=101,
            )

    def test_config_is_immutable(self, mock_config):
        """Test that configuration is immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            mock_config.neo4j_uri = "neo4j://new-uri:7687"

    def test_insecure_production_uri_logs_warning(self, caplog):
        """Test that insecure production URI logs a warning."""
        with caplog.at_level(logging.WARNING):
            AppConfig(
                neo4j_uri="neo4j://production.aura.neo4j.io:7687",
                neo4j_username="neo4j",
                neo4j_password="password",  # noqa: S106
            )
        assert "insecure Neo4j connection scheme" in caplog.text

    def test_secure_production_uri_no_warning(self, caplog):
        """Test that secure production URI does not log a warning."""
        with caplog.at_level(logging.WARNING):
            AppConfig(
                neo4j_uri="neo4j+s://production.aura.neo4j.io",
                neo4j_username="neo4j",
                neo4j_password="password",  # noqa: S106
            )
        assert "insecure" not in caplog.text

    def test_default_values(self):
        """Test that default values are correctly applied."""
        config = AppConfig(
            neo4j_uri="neo4j://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="password",  # noqa: S106
        )
        assert config.neo4j_database == "neo4j"
        assert config.openai_api_key == ""
        assert config.chat_model == "gpt-4o"
        assert config.embedding_model == "text-embedding-3-small"
        assert config.vector_index_name == "chunk_embeddings"
        assert config.similarity_k == 6
        assert config.log_level == "INFO"
        assert config.neo4j_max_connection_pool_size == 5
        assert config.neo4j_connection_acquisition_timeout == 30.0


class TestGetConfig:
    """Tests for get_config function."""

    def test_get_config_loads_all_env_vars(self, env_vars):
        """Test that get_config loads all environment variables."""
        config = get_config()

        assert config.neo4j_uri == env_vars["NEO4J_URI"]
        assert config.neo4j_username == env_vars["NEO4J_USERNAME"]
        assert config.neo4j_password == env_vars["NEO4J_PASSWORD"]
        assert config.neo4j_database == env_vars["NEO4J_DATABASE"]
        assert config.openai_api_key == env_vars["OPENAI_API_KEY"]
        assert config.chat_model == env_vars["OPENAI_MODEL"]
        assert config.embedding_model == env_vars["EMBEDDING_MODEL"]
        assert config.vector_index_name == env_vars["VECTOR_INDEX_NAME"]
        assert config.similarity_k == int(env_vars["SIMILARITY_K"])
        assert config.log_level == env_vars["LOG_LEVEL"]

    def test_get_config_with_minimal_env_vars(self, minimal_env_vars):
        """Test that get_config works with only required env vars."""
        config = get_config()

        assert config.neo4j_uri == minimal_env_vars["NEO4J_URI"]
        assert config.neo4j_username == minimal_env_vars["NEO4J_USERNAME"]
        assert config.neo4j_password == minimal_env_vars["NEO4J_PASSWORD"]
        # Defaults should be applied
        assert config.neo4j_database == "neo4j"
        assert config.chat_model == "gpt-4o"

    def test_get_config_missing_uri_raises_error(self):
        """Test that missing NEO4J_URI raises ConfigurationError."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ConfigurationError, match="Missing required environment variables"),
        ):
            get_config()

    def test_get_config_missing_username_raises_error(self):
        """Test that missing NEO4J_USERNAME raises ConfigurationError."""
        with (
            patch.dict("os.environ", {"NEO4J_URI": "neo4j://localhost:7687"}, clear=True),
            pytest.raises(ConfigurationError, match="NEO4J_USERNAME"),
        ):
            get_config()

    def test_get_config_missing_password_raises_error(self):
        """Test that missing NEO4J_PASSWORD raises ConfigurationError."""
        with patch.dict(
            "os.environ",
            {"NEO4J_URI": "neo4j://localhost:7687", "NEO4J_USERNAME": "neo4j"},
            clear=True,
        ), pytest.raises(ConfigurationError, match="NEO4J_PASSWORD"):
            get_config()

    def test_get_config_lists_all_missing_vars(self):
        """Test that error message lists all missing required variables."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                get_config()
            error_msg = str(exc_info.value)
            assert "NEO4J_URI" in error_msg
            assert "NEO4J_USERNAME" in error_msg
            assert "NEO4J_PASSWORD" in error_msg


class TestSchemeConstants:
    """Tests for URI scheme constants."""

    def test_secure_schemes_are_subset_of_valid(self):
        """Test that all secure schemes are also valid schemes."""
        for scheme in SECURE_NEO4J_SCHEMES:
            assert scheme in VALID_NEO4J_SCHEMES

    def test_valid_schemes_include_both_neo4j_and_bolt(self):
        """Test that valid schemes include both neo4j and bolt protocols."""
        neo4j_schemes = [s for s in VALID_NEO4J_SCHEMES if s.startswith("neo4j")]
        bolt_schemes = [s for s in VALID_NEO4J_SCHEMES if s.startswith("bolt")]
        assert len(neo4j_schemes) > 0
        assert len(bolt_schemes) > 0
