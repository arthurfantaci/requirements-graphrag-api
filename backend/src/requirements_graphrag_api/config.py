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
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# Load .env file for local development
# Checks backend/.env first, then falls back to root .env
# Does not override existing environment variables (deployment configs take precedence)
# Path from config.py: backend/src/requirements_graphrag_api/config.py
#   -> 3 parents up = backend/
#   -> 4 parents up = project root
_backend_env = Path(__file__).parent.parent.parent / ".env"
_root_env = Path(__file__).parent.parent.parent.parent / ".env"

if _backend_env.exists():
    load_dotenv(_backend_env, override=False)
elif _root_env.exists():
    load_dotenv(_root_env, override=False)

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
    # Conversational model (lightweight, for meta-conversation queries)
    conversational_model: str = "gpt-4o-mini"

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


@dataclass(frozen=True, slots=True)
class AuthConfig:
    """Authentication and authorization configuration.

    Attributes:
        require_api_key: Whether to require API keys for all requests.
            When False (default), anonymous access is allowed.
        api_key_header: Header name for API key authentication.
        rate_limit_free: Rate limit for free tier (requests/time).
        rate_limit_standard: Rate limit for standard tier.
        rate_limit_premium: Rate limit for premium tier.
        rate_limit_enterprise: Rate limit for enterprise tier.
        audit_enabled: Whether to enable audit logging.
        audit_log_requests: Whether to log incoming requests.
        audit_log_responses: Whether to log outgoing responses.
    """

    require_api_key: bool = False
    api_key_header: str = "X-API-Key"

    # Rate limits by tier
    rate_limit_free: str = "10/minute"
    rate_limit_standard: str = "50/minute"
    rate_limit_premium: str = "200/minute"
    rate_limit_enterprise: str = "1000/minute"

    # Audit logging
    audit_enabled: bool = True
    audit_log_requests: bool = True
    audit_log_responses: bool = True

    @property
    def rate_limits(self) -> dict[str, str]:
        """Get rate limits by tier as a dictionary."""
        return {
            "free": self.rate_limit_free,
            "standard": self.rate_limit_standard,
            "premium": self.rate_limit_premium,
            "enterprise": self.rate_limit_enterprise,
        }


@dataclass(frozen=True, slots=True)
class GuardrailConfig:
    """Configuration for guardrail features.

    Attributes:
        prompt_injection_enabled: Enable prompt injection detection.
        pii_detection_enabled: Enable PII detection and redaction.
        rate_limiting_enabled: Enable request rate limiting.
        injection_block_threshold: Risk level at which to block (low/medium/high/critical).
        pii_entities: PII entity types to detect.
        pii_score_threshold: Minimum confidence score for PII detection.
        pii_anonymize_type: How to anonymize PII (replace/redact/hash).
        rate_limit_chat: Rate limit for /chat endpoint.
        rate_limit_search: Rate limit for /search endpoints.
        rate_limit_default: Default rate limit for other endpoints.
    """

    # Feature flags — Phase 1 (Critical Security)
    prompt_injection_enabled: bool = True
    pii_detection_enabled: bool = True
    rate_limiting_enabled: bool = True

    # Feature flags — Phase 2 (Content Safety)
    toxicity_enabled: bool = True
    toxicity_use_full_check: bool = False  # OpenAI Moderation API (~500ms extra)
    topic_guard_enabled: bool = True
    topic_guard_use_llm: bool = True  # LLM classification for borderline queries
    topic_guard_allow_borderline: bool = True
    output_filter_enabled: bool = True
    output_filter_confidence_threshold: float = 0.6
    hallucination_enabled: bool = True

    # Prompt injection settings
    injection_block_threshold: str = "high"  # low, medium, high, critical

    # PII settings
    pii_entities: tuple[str, ...] = (
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "US_SSN",
    )
    pii_score_threshold: float = 0.7
    pii_anonymize_type: str = "replace"  # replace, redact, hash

    # Rate limiting
    rate_limit_chat: str = "20/minute"
    rate_limit_search: str = "60/minute"
    rate_limit_default: str = "100/minute"


def get_guardrail_config() -> GuardrailConfig:
    """Load guardrail configuration from environment variables.

    Returns:
        GuardrailConfig instance with values from environment.
    """

    def str_to_bool(value: str, default: bool = True) -> bool:
        return value.lower() in ("true", "1", "yes") if value else default

    pii_entities_str = os.getenv(
        "GUARDRAIL_PII_ENTITIES", "PERSON,EMAIL_ADDRESS,PHONE_NUMBER,CREDIT_CARD,US_SSN"
    )
    pii_entities = tuple(e.strip() for e in pii_entities_str.split(",") if e.strip())

    return GuardrailConfig(
        prompt_injection_enabled=str_to_bool(
            os.getenv("GUARDRAIL_PROMPT_INJECTION_ENABLED", "true")
        ),
        pii_detection_enabled=str_to_bool(os.getenv("GUARDRAIL_PII_DETECTION_ENABLED", "true")),
        rate_limiting_enabled=str_to_bool(os.getenv("GUARDRAIL_RATE_LIMITING_ENABLED", "true")),
        toxicity_enabled=str_to_bool(os.getenv("GUARDRAIL_TOXICITY_ENABLED", "true")),
        toxicity_use_full_check=str_to_bool(
            os.getenv("GUARDRAIL_TOXICITY_FULL_CHECK", "false"), default=False
        ),
        topic_guard_enabled=str_to_bool(os.getenv("GUARDRAIL_TOPIC_GUARD_ENABLED", "true")),
        topic_guard_use_llm=str_to_bool(os.getenv("GUARDRAIL_TOPIC_GUARD_USE_LLM", "true")),
        topic_guard_allow_borderline=str_to_bool(
            os.getenv("GUARDRAIL_TOPIC_GUARD_ALLOW_BORDERLINE", "true")
        ),
        output_filter_enabled=str_to_bool(os.getenv("GUARDRAIL_OUTPUT_FILTER_ENABLED", "true")),
        output_filter_confidence_threshold=float(
            os.getenv("GUARDRAIL_OUTPUT_FILTER_CONFIDENCE_THRESHOLD", "0.6")
        ),
        hallucination_enabled=str_to_bool(os.getenv("GUARDRAIL_HALLUCINATION_ENABLED", "true")),
        injection_block_threshold=os.getenv("GUARDRAIL_PROMPT_INJECTION_THRESHOLD", "high"),
        pii_entities=pii_entities,
        pii_score_threshold=float(os.getenv("GUARDRAIL_PII_SCORE_THRESHOLD", "0.7")),
        pii_anonymize_type=os.getenv("GUARDRAIL_PII_ANONYMIZE_TYPE", "replace"),
        rate_limit_chat=os.getenv("GUARDRAIL_RATE_LIMIT_CHAT", "20/minute"),
        rate_limit_search=os.getenv("GUARDRAIL_RATE_LIMIT_SEARCH", "60/minute"),
        rate_limit_default=os.getenv("GUARDRAIL_RATE_LIMIT_DEFAULT", "100/minute"),
    )


def get_auth_config() -> AuthConfig:
    """Load authentication configuration from environment variables.

    Environment variables:
    - REQUIRE_API_KEY: Whether to require API keys (default: false)
    - AUTH_RATE_LIMIT_FREE: Rate limit for free tier (default: 10/minute)
    - AUTH_RATE_LIMIT_STANDARD: Rate limit for standard tier (default: 50/minute)
    - AUTH_RATE_LIMIT_PREMIUM: Rate limit for premium tier (default: 200/minute)
    - AUTH_RATE_LIMIT_ENTERPRISE: Rate limit for enterprise tier (default: 1000/minute)
    - AUTH_AUDIT_ENABLED: Whether to enable audit logging (default: true)
    - AUTH_AUDIT_LOG_REQUESTS: Whether to log requests (default: true)
    - AUTH_AUDIT_LOG_RESPONSES: Whether to log responses (default: true)

    Returns:
        AuthConfig instance with values from environment.
    """

    def str_to_bool(value: str, default: bool) -> bool:
        return value.lower() in ("true", "1", "yes") if value else default

    return AuthConfig(
        require_api_key=str_to_bool(os.getenv("REQUIRE_API_KEY", "false"), False),
        rate_limit_free=os.getenv("AUTH_RATE_LIMIT_FREE", "10/minute"),
        rate_limit_standard=os.getenv("AUTH_RATE_LIMIT_STANDARD", "50/minute"),
        rate_limit_premium=os.getenv("AUTH_RATE_LIMIT_PREMIUM", "200/minute"),
        rate_limit_enterprise=os.getenv("AUTH_RATE_LIMIT_ENTERPRISE", "1000/minute"),
        audit_enabled=str_to_bool(os.getenv("AUTH_AUDIT_ENABLED", "true"), True),
        audit_log_requests=str_to_bool(os.getenv("AUTH_AUDIT_LOG_REQUESTS", "true"), True),
        audit_log_responses=str_to_bool(os.getenv("AUTH_AUDIT_LOG_RESPONSES", "true"), True),
    )


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
        conversational_model=os.getenv("CONVERSATIONAL_MODEL", "gpt-4o-mini"),
    )
