# Phase 3: Access Control Implementation

## Overview
Implement authentication, authorization, and audit logging for enterprise-grade access control.

**Timeline**: Week 5-6
**Priority**: P1 (High)
**Prerequisites**: Phase 1 & 2 complete

---

## 3.1 API Key Authentication

### Files to Create

#### `backend/src/requirements_graphrag_api/auth/__init__.py`

```python
"""Authentication and authorization module."""

from requirements_graphrag_api.auth.api_key import (
    APIKeyAuth,
    APIKeyInfo,
    verify_api_key,
    generate_api_key,
    hash_api_key,
)
from requirements_graphrag_api.auth.middleware import (
    AuthMiddleware,
    get_current_client,
)

__all__ = [
    "APIKeyAuth",
    "APIKeyInfo",
    "verify_api_key",
    "generate_api_key",
    "hash_api_key",
    "AuthMiddleware",
    "get_current_client",
]
```

#### `backend/src/requirements_graphrag_api/auth/api_key.py`

**Purpose**: API key generation, validation, and management.

```python
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

if TYPE_CHECKING:
    from collections.abc import Mapping

# API key header configuration
API_KEY_HEADER_NAME = "X-API-Key"
API_KEY_PREFIX = "rgapi_"  # Requirements GraphRAG API

api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


class APIKeyTier(StrEnum):
    """API key access tiers with different rate limits."""
    FREE = "free"           # 10 requests/min
    STANDARD = "standard"   # 50 requests/min
    PREMIUM = "premium"     # 200 requests/min
    ENTERPRISE = "enterprise"  # Custom limits


@dataclass(frozen=True, slots=True)
class APIKeyInfo:
    """Information associated with an API key."""
    key_id: str              # Unique identifier (not the key itself)
    name: str                # Human-readable name
    tier: APIKeyTier         # Access tier
    organization: str | None # Organization name
    created_at: datetime
    expires_at: datetime | None
    rate_limit: str          # e.g., "50/minute"
    is_active: bool
    scopes: tuple[str, ...]  # e.g., ("chat", "search", "feedback")
    metadata: dict           # Additional metadata


def generate_api_key() -> str:
    """Generate a new API key.

    Format: rgapi_<32 random bytes as base64url>
    Total length: ~50 characters
    """
    random_bytes = secrets.token_urlsafe(32)
    return f"{API_KEY_PREFIX}{random_bytes}"


def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage.

    Uses SHA-256 for fast lookups while maintaining security.
    The prefix is stripped before hashing.
    """
    # Strip prefix if present
    key_value = api_key.removeprefix(API_KEY_PREFIX)
    return hashlib.sha256(key_value.encode()).hexdigest()


def validate_api_key_format(api_key: str) -> bool:
    """Validate API key format."""
    if not api_key:
        return False
    if not api_key.startswith(API_KEY_PREFIX):
        return False
    if len(api_key) < 40:  # Minimum reasonable length
        return False
    return True


class APIKeyStore:
    """Abstract base for API key storage.

    In production, implement with database backend.
    """

    async def get_by_hash(self, key_hash: str) -> APIKeyInfo | None:
        """Retrieve API key info by hash."""
        raise NotImplementedError

    async def create(self, api_key: str, info: APIKeyInfo) -> None:
        """Store a new API key."""
        raise NotImplementedError

    async def revoke(self, key_id: str) -> bool:
        """Revoke an API key."""
        raise NotImplementedError


class InMemoryAPIKeyStore(APIKeyStore):
    """In-memory API key store for development/testing."""

    def __init__(self):
        self._keys: dict[str, APIKeyInfo] = {}

    async def get_by_hash(self, key_hash: str) -> APIKeyInfo | None:
        return self._keys.get(key_hash)

    async def create(self, api_key: str, info: APIKeyInfo) -> None:
        key_hash = hash_api_key(api_key)
        self._keys[key_hash] = info

    async def revoke(self, key_id: str) -> bool:
        for key_hash, info in list(self._keys.items()):
            if info.key_id == key_id:
                del self._keys[key_hash]
                return True
        return False


async def verify_api_key(
    api_key: str = Security(api_key_header),
    key_store: APIKeyStore = None,  # Injected via dependency
) -> APIKeyInfo:
    """Verify API key and return associated info.

    Raises HTTPException if key is invalid.
    """
    # Check if auth is required
    if key_store is None:
        # Auth disabled, return anonymous info
        return APIKeyInfo(
            key_id="anonymous",
            name="Anonymous",
            tier=APIKeyTier.FREE,
            organization=None,
            created_at=datetime.utcnow(),
            expires_at=None,
            rate_limit="20/minute",
            is_active=True,
            scopes=("chat", "search"),
            metadata={},
        )

    # Validate key presence
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "missing_api_key",
                "message": f"API key required. Include '{API_KEY_HEADER_NAME}' header.",
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Validate key format
    if not validate_api_key_format(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "invalid_api_key_format",
                "message": "Invalid API key format.",
            },
        )

    # Look up key
    key_hash = hash_api_key(api_key)
    key_info = await key_store.get_by_hash(key_hash)

    if key_info is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "invalid_api_key",
                "message": "API key not found or invalid.",
            },
        )

    # Check if active
    if not key_info.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "api_key_revoked",
                "message": "API key has been revoked.",
            },
        )

    # Check expiration
    if key_info.expires_at and datetime.utcnow() > key_info.expires_at:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "api_key_expired",
                "message": "API key has expired.",
            },
        )

    return key_info
```

---

#### `backend/src/requirements_graphrag_api/auth/middleware.py`

**Purpose**: FastAPI middleware for authentication and request context.

```python
from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

from fastapi import Depends, Request
from starlette.middleware.base import BaseHTTPMiddleware

from requirements_graphrag_api.auth.api_key import (
    APIKeyInfo,
    verify_api_key,
)

if TYPE_CHECKING:
    from starlette.responses import Response

# Context variable for current client info
_current_client: ContextVar[APIKeyInfo | None] = ContextVar(
    "current_client", default=None
)


def get_current_client() -> APIKeyInfo | None:
    """Get the current authenticated client from context."""
    return _current_client.get()


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to handle authentication and set request context."""

    def __init__(self, app, require_auth: bool = False):
        super().__init__(app)
        self.require_auth = require_auth

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip auth for health checks and docs
        if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        # Extract and validate API key if present
        api_key = request.headers.get("X-API-Key")

        if api_key or self.require_auth:
            try:
                key_store = request.app.state.api_key_store
                client_info = await verify_api_key(api_key, key_store)
                _current_client.set(client_info)

                # Add client info to request state for logging
                request.state.client = client_info
            except Exception:
                if self.require_auth:
                    raise
                # If auth not required, continue as anonymous
                request.state.client = None
        else:
            request.state.client = None

        try:
            response = await call_next(request)
            return response
        finally:
            _current_client.set(None)
```

---

## 3.2 Scope-Based Authorization

#### `backend/src/requirements_graphrag_api/auth/scopes.py`

**Purpose**: Define and check endpoint access scopes.

```python
from __future__ import annotations

from enum import StrEnum
from functools import wraps
from typing import TYPE_CHECKING

from fastapi import HTTPException, status

if TYPE_CHECKING:
    from requirements_graphrag_api.auth.api_key import APIKeyInfo


class Scope(StrEnum):
    """API access scopes."""
    CHAT = "chat"           # Access to /chat endpoint
    SEARCH = "search"       # Access to /search/* endpoints
    FEEDBACK = "feedback"   # Access to /feedback endpoint
    ADMIN = "admin"         # Access to admin endpoints


# Endpoint to required scopes mapping
ENDPOINT_SCOPES = {
    "/chat": [Scope.CHAT],
    "/search/hybrid": [Scope.SEARCH],
    "/search/vector": [Scope.SEARCH],
    "/search/graph": [Scope.SEARCH],
    "/feedback": [Scope.FEEDBACK],
    "/definitions": [Scope.SEARCH],
    "/glossary": [Scope.SEARCH],
    "/standards": [Scope.SEARCH],
}


def require_scope(*required_scopes: Scope):
    """Decorator to require specific scopes for an endpoint."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, client: APIKeyInfo = None, **kwargs):
            if client is None:
                # Anonymous access - check if endpoint allows it
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            # Check if client has required scopes
            client_scopes = set(client.scopes)
            missing = set(required_scopes) - client_scopes

            if missing:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "error": "insufficient_scope",
                        "message": f"Missing required scopes: {', '.join(missing)}",
                        "required": list(required_scopes),
                        "granted": list(client.scopes),
                    },
                )

            return await func(*args, client=client, **kwargs)
        return wrapper
    return decorator
```

---

## 3.3 Audit Logging

#### `backend/src/requirements_graphrag_api/auth/audit.py`

**Purpose**: Comprehensive audit logging for compliance and security analysis.

```python
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import Request
    from requirements_graphrag_api.auth.api_key import APIKeyInfo

logger = logging.getLogger("audit")


class AuditEventType(StrEnum):
    """Types of audit events."""
    # Authentication events
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    AUTH_KEY_CREATED = "auth.key_created"
    AUTH_KEY_REVOKED = "auth.key_revoked"

    # API access events
    API_REQUEST = "api.request"
    API_RESPONSE = "api.response"
    API_ERROR = "api.error"

    # Guardrail events
    GUARDRAIL_TRIGGERED = "guardrail.triggered"
    GUARDRAIL_BLOCKED = "guardrail.blocked"

    # Rate limiting
    RATE_LIMIT_WARNING = "rate_limit.warning"
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"


@dataclass
class AuditEvent:
    """Structured audit event."""
    event_type: AuditEventType
    timestamp: datetime
    request_id: str
    client_id: str | None
    client_ip: str | None
    endpoint: str
    method: str
    status_code: int | None
    duration_ms: float | None
    details: dict[str, Any]

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["event_type"] = self.event_type.value
        return data


class AuditLogger:
    """Audit logger with configurable backends."""

    def __init__(self):
        self._handlers: list[AuditHandler] = []

    def add_handler(self, handler: AuditHandler) -> None:
        self._handlers.append(handler)

    async def log(self, event: AuditEvent) -> None:
        """Log an audit event to all handlers."""
        for handler in self._handlers:
            try:
                await handler.handle(event)
            except Exception as e:
                logger.error(f"Audit handler error: {e}")


class AuditHandler:
    """Base class for audit handlers."""

    async def handle(self, event: AuditEvent) -> None:
        raise NotImplementedError


class LoggingAuditHandler(AuditHandler):
    """Handler that logs to Python logging."""

    async def handle(self, event: AuditEvent) -> None:
        logger.info(
            "Audit: %s",
            event.event_type.value,
            extra={"audit_event": event.to_dict()},
        )


class LangSmithAuditHandler(AuditHandler):
    """Handler that sends events to LangSmith as feedback."""

    async def handle(self, event: AuditEvent) -> None:
        # Could integrate with LangSmith for analysis
        pass


# Global audit logger instance
audit_logger = AuditLogger()
audit_logger.add_handler(LoggingAuditHandler())


async def log_api_request(
    request: Request,
    client: APIKeyInfo | None,
    request_id: str,
) -> None:
    """Log an API request."""
    await audit_logger.log(AuditEvent(
        event_type=AuditEventType.API_REQUEST,
        timestamp=datetime.utcnow(),
        request_id=request_id,
        client_id=client.key_id if client else None,
        client_ip=request.client.host if request.client else None,
        endpoint=request.url.path,
        method=request.method,
        status_code=None,
        duration_ms=None,
        details={
            "query_params": dict(request.query_params),
            "content_type": request.headers.get("content-type"),
        },
    ))


async def log_api_response(
    request: Request,
    client: APIKeyInfo | None,
    request_id: str,
    status_code: int,
    duration_ms: float,
) -> None:
    """Log an API response."""
    await audit_logger.log(AuditEvent(
        event_type=AuditEventType.API_RESPONSE,
        timestamp=datetime.utcnow(),
        request_id=request_id,
        client_id=client.key_id if client else None,
        client_ip=request.client.host if request.client else None,
        endpoint=request.url.path,
        method=request.method,
        status_code=status_code,
        duration_ms=duration_ms,
        details={},
    ))
```

---

### Integration Points

#### Update `backend/src/requirements_graphrag_api/api.py`

```python
from requirements_graphrag_api.auth import (
    AuthMiddleware,
    InMemoryAPIKeyStore,
)

def create_app() -> FastAPI:
    app = FastAPI(...)

    # Initialize API key store
    # In production, use database-backed store
    app.state.api_key_store = InMemoryAPIKeyStore()

    # Add authentication middleware
    auth_required = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
    app.add_middleware(AuthMiddleware, require_auth=auth_required)

    return app
```

#### Update Routes to Use Authentication

```python
from fastapi import Depends
from requirements_graphrag_api.auth import verify_api_key, APIKeyInfo

@router.post("/chat")
async def chat_endpoint(
    request: Request,
    body: ChatRequest,
    client: APIKeyInfo = Depends(verify_api_key),
):
    # client contains API key info for logging/rate limiting
    ...
```

---

### Configuration

```python
@dataclass(frozen=True, slots=True)
class AuthConfig:
    """Authentication configuration."""
    require_api_key: bool = False  # Enable in production
    api_key_header: str = "X-API-Key"

    # Rate limits by tier
    rate_limits: dict[str, str] = field(default_factory=lambda: {
        "free": "10/minute",
        "standard": "50/minute",
        "premium": "200/minute",
        "enterprise": "1000/minute",
    })

    # Audit logging
    audit_enabled: bool = True
    audit_log_requests: bool = True
    audit_log_responses: bool = True
```

---

### Tests

```python
"""Tests for API key authentication."""

import pytest
from requirements_graphrag_api.auth import (
    generate_api_key,
    hash_api_key,
    validate_api_key_format,
    verify_api_key,
)

class TestAPIKeyGeneration:
    def test_generates_valid_format(self):
        key = generate_api_key()
        assert key.startswith("rgapi_")
        assert len(key) > 40

    def test_generates_unique_keys(self):
        keys = [generate_api_key() for _ in range(100)]
        assert len(set(keys)) == 100

class TestAPIKeyHashing:
    def test_consistent_hashing(self):
        key = generate_api_key()
        hash1 = hash_api_key(key)
        hash2 = hash_api_key(key)
        assert hash1 == hash2

    def test_different_keys_different_hashes(self):
        key1 = generate_api_key()
        key2 = generate_api_key()
        assert hash_api_key(key1) != hash_api_key(key2)
```

---

### Acceptance Criteria

- [ ] API keys generated with secure random bytes
- [ ] Keys hashed before storage (SHA-256)
- [ ] Authentication middleware validates keys
- [ ] Invalid keys return 401/403 with clear errors
- [ ] Rate limits enforced per API key tier
- [ ] All requests logged with client context
- [ ] Scope-based authorization working
- [ ] Feature flag to enable/disable auth
- [ ] Admin endpoints for key management
- [ ] 100% test coverage on auth module
