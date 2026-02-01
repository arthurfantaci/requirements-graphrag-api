"""Authentication and authorization module for GraphRAG API.

This module provides enterprise-grade access control:
- API key authentication with secure hashing
- Scope-based authorization for fine-grained access control
- Request context management for route handlers
- Comprehensive audit logging for compliance

Phase 3 of the guardrails implementation.

Usage:
    # In FastAPI app setup
    from requirements_graphrag_api.auth import (
        AuthMiddleware,
        InMemoryAPIKeyStore,
        configure_audit_logging,
    )

    app.state.api_key_store = InMemoryAPIKeyStore()
    app.add_middleware(AuthMiddleware, require_auth=False)
    configure_audit_logging()

    # In route handlers
    from requirements_graphrag_api.auth import (
        get_current_client,
        require_scopes,
        Scope,
    )

    @router.post("/chat")
    async def chat(
        request: Request,
        client: APIKeyInfo = Depends(require_scopes(Scope.CHAT))
    ):
        ...

    # Key management
    from requirements_graphrag_api.auth import generate_api_key, hash_api_key

    raw_key = generate_api_key()  # rgapi_<random>
    key_hash = hash_api_key(raw_key)  # Store this in database
"""

from __future__ import annotations

from requirements_graphrag_api.auth.api_key import (
    API_KEY_HEADER_NAME,
    API_KEY_PREFIX,
    TIER_RATE_LIMITS,
    APIKeyInfo,
    APIKeyStore,
    APIKeyTier,
    InMemoryAPIKeyStore,
    create_anonymous_key_info,
    generate_api_key,
    hash_api_key,
    validate_api_key_format,
    verify_api_key,
)
from requirements_graphrag_api.auth.audit import (
    AuditEvent,
    AuditEventType,
    AuditHandler,
    AuditLogger,
    LoggingAuditHandler,
    audit_logger,
    configure_audit_logging,
    log_api_error,
    log_api_request,
    log_api_response,
    log_auth_failure,
    log_auth_success,
    log_guardrail_event,
    log_rate_limit_event,
)
from requirements_graphrag_api.auth.middleware import (
    AuthMiddleware,
    get_current_client,
    get_current_request_id,
    get_request_duration_ms,
)
from requirements_graphrag_api.auth.scopes import (
    DEFAULT_TIER_SCOPES,
    ENDPOINT_SCOPES,
    Scope,
    ScopeChecker,
    check_scopes,
    require_scope,
    require_scopes,
)

__all__ = [
    "API_KEY_HEADER_NAME",
    "API_KEY_PREFIX",
    "DEFAULT_TIER_SCOPES",
    "ENDPOINT_SCOPES",
    "TIER_RATE_LIMITS",
    "APIKeyInfo",
    "APIKeyStore",
    "APIKeyTier",
    "AuditEvent",
    "AuditEventType",
    "AuditHandler",
    "AuditLogger",
    "AuthMiddleware",
    "InMemoryAPIKeyStore",
    "LoggingAuditHandler",
    "Scope",
    "ScopeChecker",
    "audit_logger",
    "check_scopes",
    "configure_audit_logging",
    "create_anonymous_key_info",
    "generate_api_key",
    "get_current_client",
    "get_current_request_id",
    "get_request_duration_ms",
    "hash_api_key",
    "log_api_error",
    "log_api_request",
    "log_api_response",
    "log_auth_failure",
    "log_auth_success",
    "log_guardrail_event",
    "log_rate_limit_event",
    "require_scope",
    "require_scopes",
    "validate_api_key_format",
    "verify_api_key",
]
