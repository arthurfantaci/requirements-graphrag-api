"""FastAPI middleware for authentication and request context.

This module provides authentication middleware that:
- Validates API keys on incoming requests
- Sets up request context with client information
- Supports optional authentication (disabled by default)
- Skips authentication for health checks and documentation endpoints

Usage:
    from requirements_graphrag_api.auth.middleware import AuthMiddleware

    # In FastAPI app setup
    app.add_middleware(AuthMiddleware, require_auth=False)

    # In route handlers
    from requirements_graphrag_api.auth.middleware import get_current_client
    client = get_current_client()  # Returns APIKeyInfo or None
"""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar
from datetime import UTC
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware

from requirements_graphrag_api.auth.api_key import (
    API_KEY_HEADER_NAME,
    APIKeyInfo,
    APIKeyStore,
    create_anonymous_key_info,
    hash_api_key,
    validate_api_key_format,
)
from requirements_graphrag_api.auth.audit import (
    log_api_request,
    log_api_response,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import Request
    from starlette.responses import Response
    from starlette.types import ASGIApp

    # Type alias for the call_next function
    RequestResponseEndpoint = Callable[[Request], Response]

logger = logging.getLogger(__name__)

# Context variables for request-scoped state
# These are async-safe and provide isolation between concurrent requests
_current_client: ContextVar[APIKeyInfo | None] = ContextVar("current_client", default=None)
_current_request_id: ContextVar[str | None] = ContextVar("current_request_id", default=None)
_request_start_time: ContextVar[float | None] = ContextVar("request_start_time", default=None)


def get_current_client() -> APIKeyInfo | None:
    """Get the current authenticated client from request context.

    Call this from anywhere within a request handler to access
    the authenticated client's information.

    Returns:
        APIKeyInfo for the authenticated client, or None if anonymous.

    Example:
        @router.get("/protected")
        async def protected_endpoint():
            client = get_current_client()
            if client:
                return {"message": f"Hello, {client.name}"}
            return {"message": "Anonymous access"}
    """
    return _current_client.get()


def get_current_request_id() -> str | None:
    """Get the current request ID from context.

    Returns:
        The unique request ID string, or None if not in a request context.
    """
    return _current_request_id.get()


def get_request_duration_ms() -> float | None:
    """Get the current request duration in milliseconds.

    Returns:
        Duration since request started, or None if not in a request context.
    """
    start = _request_start_time.get()
    if start is None:
        return None
    return (time.perf_counter() - start) * 1000


# Paths that skip authentication
SKIP_AUTH_PATHS: frozenset[str] = frozenset(
    {
        "/",
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
    }
)


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to handle authentication and set request context.

    This middleware:
    1. Generates a unique request ID for tracing
    2. Records request start time for duration metrics
    3. Extracts and validates API keys from headers
    4. Sets up context variables for access in route handlers
    5. Cleans up context after request completion

    Args:
        app: The ASGI application to wrap.
        require_auth: If True, requests without valid API keys are rejected.
            If False (default), anonymous access is allowed.

    Attributes:
        require_auth: Whether authentication is required for API access.

    Example:
        app = FastAPI()
        app.add_middleware(AuthMiddleware, require_auth=False)
    """

    def __init__(self, app: ASGIApp, require_auth: bool = False) -> None:
        """Initialize the authentication middleware.

        Args:
            app: The ASGI application to wrap.
            require_auth: Whether to require authentication.
        """
        super().__init__(app)
        self.require_auth = require_auth

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process the request through the authentication pipeline.

        Args:
            request: The incoming HTTP request.
            call_next: The next handler in the middleware chain.

        Returns:
            The HTTP response from the route handler.
        """
        from fastapi import HTTPException
        from starlette.responses import JSONResponse

        # Generate request ID and start timing
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        # Set context variables
        _current_request_id.set(request_id)
        _request_start_time.set(start_time)

        # Add request ID to response headers for tracing
        # This will be done after call_next

        # Skip auth for specific paths and CORS preflight requests
        # OPTIONS requests are sent by browsers before actual requests to check CORS
        if request.url.path in SKIP_AUTH_PATHS or request.method == "OPTIONS":
            request.state.client = None
            request.state.request_id = request_id
            try:
                response = await call_next(request)
                response.headers["X-Request-ID"] = request_id
                return response
            finally:
                self._cleanup_context()

        # Extract API key from header
        api_key = request.headers.get(API_KEY_HEADER_NAME)

        # Handle authentication - catch HTTPException and return proper response
        try:
            client_info = await self._authenticate(request, api_key)
        except HTTPException as exc:
            self._cleanup_context()
            response = JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers,
            )
            response.headers["X-Request-ID"] = request_id
            return response

        # Store in request state for route handlers
        request.state.client = client_info
        request.state.request_id = request_id

        # Set context variable for access anywhere in the request
        _current_client.set(client_info)

        # Audit log: request received
        await log_api_request(request, client_info, request_id)

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id

            # Audit log: response sent
            duration_ms = (time.perf_counter() - start_time) * 1000
            await log_api_response(
                request, client_info, request_id, response.status_code, duration_ms
            )

            return response
        finally:
            self._cleanup_context()

    async def _authenticate(self, request: Request, api_key: str | None) -> APIKeyInfo | None:
        """Authenticate the request using the provided API key.

        Args:
            request: The incoming HTTP request.
            api_key: The API key from the header, if present.

        Returns:
            APIKeyInfo for authenticated requests, None for anonymous.

        Raises:
            HTTPException: If authentication fails and is required.
        """
        from fastapi import HTTPException, status

        # Get key store from app state (initialized during app startup)
        key_store: APIKeyStore | None = getattr(request.app.state, "api_key_store", None)

        # No key provided
        if not api_key:
            if self.require_auth:
                logger.warning(
                    "Missing API key on protected endpoint",
                    extra={
                        "path": request.url.path,
                        "method": request.method,
                        "client_ip": request.client.host if request.client else None,
                    },
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={
                        "error": "missing_api_key",
                        "message": f"API key required. Include '{API_KEY_HEADER_NAME}' header.",
                    },
                    headers={"WWW-Authenticate": "ApiKey"},
                )
            # Anonymous access allowed
            return create_anonymous_key_info()

        # Validate format first (fast path for obviously invalid keys)
        if not validate_api_key_format(api_key):
            logger.warning(
                "Invalid API key format",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "client_ip": request.client.host if request.client else None,
                },
            )
            if self.require_auth:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={
                        "error": "invalid_api_key_format",
                        "message": "Invalid API key format. Keys must start with 'rgapi_'.",
                    },
                )
            return create_anonymous_key_info()

        # No store configured, allow with anonymous access
        if key_store is None:
            return create_anonymous_key_info()

        # Look up key in store
        key_hash = hash_api_key(api_key)
        key_info = await key_store.get_by_hash(key_hash)

        if key_info is None:
            logger.warning(
                "API key not found",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "client_ip": request.client.host if request.client else None,
                },
            )
            if self.require_auth:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "error": "invalid_api_key",
                        "message": "API key not found or invalid.",
                    },
                )
            return create_anonymous_key_info()

        # Check if active
        if not key_info.is_active:
            logger.warning(
                "Revoked API key used",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "key_id": key_info.key_id,
                    "client_ip": request.client.host if request.client else None,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "api_key_revoked",
                    "message": "API key has been revoked.",
                },
            )

        # Check expiration
        from datetime import datetime

        if key_info.expires_at and datetime.now(UTC) > key_info.expires_at:
            logger.warning(
                "Expired API key used",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "key_id": key_info.key_id,
                    "client_ip": request.client.host if request.client else None,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "api_key_expired",
                    "message": "API key has expired.",
                },
            )

        # Successfully authenticated
        logger.debug(
            "API key authenticated",
            extra={
                "key_id": key_info.key_id,
                "tier": key_info.tier,
                "organization": key_info.organization,
            },
        )

        return key_info

    def _cleanup_context(self) -> None:
        """Reset context variables after request completion."""
        _current_client.set(None)
        _current_request_id.set(None)
        _request_start_time.set(None)
