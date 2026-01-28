"""Configuration management for GraphRAG API.

Provides immutable configuration using dataclasses with validation,
environment variable loading, and sensible defaults.

Neo4j Connection Best Practices:
- Use neo4j+s:// for production (Aura, clusters) - enables TLS and routing
- Use neo4j:// only for local development without TLS
- Never use bolt:// with clusters (no routing support)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Final

logger = logging.getLogger(__name__)

# Valid Neo4j URI schemes
VALID_NEO4J_SCHEMES: Final[tuple[str, ...]] = (
    "neo4j://",
    "neo4j+s://",
    "neo4j+ssc://",
    "bolt://",
    "bolt+s://",
    "bolt+ssc://",
)

# Recommended schemes for production (TLS-enabled)
SECURE_NEO4J_SCHEMES: Final[tuple[str, ...]] = (
    "neo4j+s://",
    "neo4j+ssc://",
    "bolt+s://",
    "bolt+ssc://",
)

# Validation bounds
MIN_SIMILARITY_K: Final[int] = 1
MAX_SIMILARITY_K: Final[int] = 100
MIN_CONNECTION_POOL_SIZE: Final[int] = 1
MAX_CONNECTION_POOL_SIZE: Final[int] = 100


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Immutable application configuration.

    Attributes:
        neo4j_uri: Neo4j connection URI (prefer neo4j+s:// for production).
        neo4j_username: Neo4j username.
        neo4j_password: Neo4j password.
        neo4j_database: Neo4j database name.
        openai_api_key: OpenAI API key.
        chat_model: Chat model name for generation.
        embedding_model: Embedding model name.
        vector_index_name: Name of the vector index in Neo4j.
        similarity_k: Default number of results for similarity search.
        log_level: Logging level.
        neo4j_max_connection_pool_size: Max connections (reduce for serverless).
        neo4j_connection_acquisition_timeout: Timeout for acquiring connections.
        langsmith_api_key: LangSmith API key for observability.
        langsmith_project: LangSmith project name.
        langsmith_tracing_enabled: Whether to enable LangSmith tracing.
    """

    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str = "neo4j"
    openai_api_key: str = ""
    chat_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    vector_index_name: str = "chunk_embeddings"
    similarity_k: int = 6
    log_level: str = "INFO"
    # Neo4j driver settings optimized for serverless (Vercel/Lambda)
    neo4j_max_connection_pool_size: int = 5
    neo4j_connection_acquisition_timeout: float = 30.0
    neo4j_liveness_check_timeout: float = 30.0  # Timeout for connection liveness checks
    neo4j_max_connection_lifetime: int = 300  # Max seconds a connection can live (5 min)
    # LangSmith observability settings
    langsmith_api_key: str = ""
    langsmith_project: str = "graphrag-api-dev"
    langsmith_tracing_enabled: bool = False
    langsmith_workspace_id: str = ""  # Required for org-scoped API keys
    # Prompt catalog settings
    langsmith_org: str = ""
    prompt_environment: str = "development"
    prompt_cache_ttl: int = 300
    prompt_hub_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate Neo4j URI scheme
        if not any(self.neo4j_uri.startswith(scheme) for scheme in VALID_NEO4J_SCHEMES):
            msg = f"Invalid Neo4j URI scheme. Must start with one of: {VALID_NEO4J_SCHEMES}"
            raise ConfigurationError(msg)

        # Warn if using insecure scheme in production-like URI
        is_secure = any(self.neo4j_uri.startswith(s) for s in SECURE_NEO4J_SCHEMES)
        is_local = any(
            local in self.neo4j_uri for local in ["localhost", "127.0.0.1", "host.docker.internal"]
        )
        is_production_uri = (
            "aura" in self.neo4j_uri.lower() or "neo4j.io" in self.neo4j_uri.lower() or not is_local
        )

        if is_production_uri and not is_secure:
            logger.warning(
                "Using insecure Neo4j connection scheme for production URI. "
                "Consider using neo4j+s:// for TLS encryption and certificate validation."
            )

        # Validate similarity_k
        if not MIN_SIMILARITY_K <= self.similarity_k <= MAX_SIMILARITY_K:
            msg = f"similarity_k must be between {MIN_SIMILARITY_K} and {MAX_SIMILARITY_K}"
            raise ConfigurationError(msg)

        # Validate connection pool size
        pool_size = self.neo4j_max_connection_pool_size
        if not MIN_CONNECTION_POOL_SIZE <= pool_size <= MAX_CONNECTION_POOL_SIZE:
            msg = (
                f"neo4j_max_connection_pool_size must be between "
                f"{MIN_CONNECTION_POOL_SIZE} and {MAX_CONNECTION_POOL_SIZE}"
            )
            raise ConfigurationError(msg)


def get_config() -> AppConfig:
    """Load configuration from environment variables.

    Returns:
        AppConfig instance with values from environment.

    Raises:
        ConfigurationError: If required environment variables are missing.
    """
    required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        msg = f"Missing required environment variables: {', '.join(missing)}"
        raise ConfigurationError(msg)

    # Check for LangSmith tracing (supports both environment variable names)
    langsmith_tracing = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "false"))
    tracing_enabled = langsmith_tracing.lower() in ("true", "1", "yes")

    # Check for prompt hub enabled
    prompt_hub_enabled = os.getenv("PROMPT_HUB_ENABLED", "true").lower() in ("true", "1", "yes")

    return AppConfig(
        neo4j_uri=os.environ["NEO4J_URI"],
        neo4j_username=os.environ["NEO4J_USERNAME"],
        neo4j_password=os.environ["NEO4J_PASSWORD"],
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        chat_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        vector_index_name=os.getenv("VECTOR_INDEX_NAME", "chunk_embeddings"),
        similarity_k=int(os.getenv("SIMILARITY_K", "6")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        neo4j_max_connection_pool_size=int(os.getenv("NEO4J_MAX_POOL_SIZE", "5")),
        neo4j_connection_acquisition_timeout=float(os.getenv("NEO4J_CONNECTION_TIMEOUT", "30.0")),
        neo4j_liveness_check_timeout=float(os.getenv("NEO4J_LIVENESS_CHECK_TIMEOUT", "30.0")),
        neo4j_max_connection_lifetime=int(os.getenv("NEO4J_MAX_CONNECTION_LIFETIME", "300")),
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY", os.getenv("LANGCHAIN_API_KEY", "")),
        langsmith_project=os.getenv(
            "LANGSMITH_PROJECT", os.getenv("LANGCHAIN_PROJECT", "requirements-graphrag")
        ),
        langsmith_tracing_enabled=tracing_enabled,
        langsmith_workspace_id=os.getenv("LANGSMITH_WORKSPACE_ID", ""),
        langsmith_org=os.getenv("LANGSMITH_ORG", ""),
        prompt_environment=os.getenv("PROMPT_ENVIRONMENT", "development"),
        prompt_cache_ttl=int(os.getenv("PROMPT_CACHE_TTL", "300")),
        prompt_hub_enabled=prompt_hub_enabled,
    )
