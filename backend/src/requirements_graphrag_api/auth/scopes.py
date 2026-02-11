"""Scope-based authorization for API endpoints.

This module implements fine-grained access control using scopes:
- Each scope grants access to specific functionality
- API keys are assigned scopes based on their tier and use case
- Endpoints can require one or more scopes for access

Scopes follow a coarse-grained model appropriate for this API:
- chat: Access to chat/RAG endpoints
- search: Access to search endpoints (hybrid, vector, graph)
- feedback: Access to submit feedback
- admin: Access to administrative endpoints

Usage:
    from requirements_graphrag_api.auth.scopes import Scope, require_scopes

    @router.post("/chat")
    @require_scopes(Scope.CHAT)
    async def chat_endpoint(request: Request):
        ...

    # Or as FastAPI dependency
    @router.post("/admin/keys")
    async def create_key(
        _: None = Depends(require_scopes(Scope.ADMIN))
    ):
        ...

Note:
    This module intentionally does NOT use `from __future__ import annotations`
    because FastAPI's dependency injection needs to inspect the actual `Request`
    type at runtime to properly inject the request object. With deferred annotations,
    the type becomes a string and FastAPI can't recognize it.
"""

from enum import StrEnum
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, status

if TYPE_CHECKING:
    from requirements_graphrag_api.auth.api_key import APIKeyInfo


class Scope(StrEnum):
    """API access scopes.

    Scopes define what actions an API key is authorized to perform.
    Keys can have multiple scopes, and endpoints can require multiple scopes.

    Attributes:
        CHAT: Access to /chat endpoint for RAG-powered Q&A.
        SEARCH: Access to /search/* endpoints for semantic search.
        FEEDBACK: Access to /feedback endpoint for submitting feedback.
        ADMIN: Access to administrative endpoints (key management, metrics).
    """

    CHAT = "chat"
    SEARCH = "search"
    FEEDBACK = "feedback"
    ADMIN = "admin"


# Default scopes for each API key tier
DEFAULT_TIER_SCOPES: dict[str, tuple[str, ...]] = {
    "free": (Scope.CHAT, Scope.SEARCH),
    "standard": (Scope.CHAT, Scope.SEARCH, Scope.FEEDBACK),
    "premium": (Scope.CHAT, Scope.SEARCH, Scope.FEEDBACK),
    "enterprise": (Scope.CHAT, Scope.SEARCH, Scope.FEEDBACK, Scope.ADMIN),
}

# Endpoint to required scopes mapping
# Used for documentation and automatic scope checking
ENDPOINT_SCOPES: dict[str, list[Scope]] = {
    "/chat": [Scope.CHAT],
    "/search/hybrid": [Scope.SEARCH],
    "/search/vector": [Scope.SEARCH],
    "/search/graph": [Scope.SEARCH],
    "/feedback": [Scope.FEEDBACK],
    "/definitions": [Scope.SEARCH],
    "/definitions/search": [Scope.SEARCH],
    "/glossary": [Scope.SEARCH],
    "/standards": [Scope.SEARCH],
    "/standards/list": [Scope.SEARCH],
}


def check_scopes(
    client_scopes: tuple[str, ...] | list[str],
    required_scopes: tuple[Scope, ...] | list[Scope],
) -> tuple[bool, set[str]]:
    """Check if client has required scopes.

    Args:
        client_scopes: Scopes assigned to the client's API key.
        required_scopes: Scopes required for the operation.

    Returns:
        Tuple of (has_all_scopes, missing_scopes).

    Example:
        >>> has_scopes, missing = check_scopes(("chat", "search"), (Scope.CHAT,))
        >>> has_scopes
        True
        >>> missing
        set()
    """
    client_scope_set = set(client_scopes)
    required_scope_set = {str(s) for s in required_scopes}
    missing = required_scope_set - client_scope_set
    return len(missing) == 0, missing


def get_client_from_request(request: Request) -> "APIKeyInfo | None":
    """Extract client info from request state.

    The AuthMiddleware sets request.state.client during authentication.

    Args:
        request: The FastAPI request object.

    Returns:
        APIKeyInfo if authenticated, None otherwise.
    """
    return getattr(request.state, "client", None)


class ScopeChecker:
    """FastAPI dependency for checking scopes.

    Use this as a dependency to enforce scope requirements on endpoints.
    It extracts the client from request state (set by AuthMiddleware)
    and verifies they have the required scopes.

    Example:
        @router.post("/admin/keys")
        async def create_key(
            request: Request,
            _: None = Depends(ScopeChecker(Scope.ADMIN))
        ):
            ...
    """

    def __init__(self, *required_scopes: Scope, allow_anonymous: bool = False) -> None:
        """Initialize the scope checker.

        Args:
            *required_scopes: Scopes required for access.
            allow_anonymous: If True, allow access without authentication
                (but still check scopes if authenticated).
        """
        self.required_scopes = required_scopes
        self.allow_anonymous = allow_anonymous

    async def __call__(self, request: Request) -> "APIKeyInfo | None":
        """Check scopes when used as a dependency.

        Args:
            request: The FastAPI request object.

        Returns:
            The APIKeyInfo if authorized.

        Raises:
            HTTPException: 401 if not authenticated and required.
            HTTPException: 403 if missing required scopes.
        """
        client = get_client_from_request(request)

        # Handle unauthenticated requests
        if client is None:
            if self.allow_anonymous:
                return None
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "authentication_required",
                    "message": "Authentication required for this endpoint.",
                },
                headers={"WWW-Authenticate": "ApiKey"},
            )

        # Check if client has required scopes
        has_scopes, missing = check_scopes(client.scopes, self.required_scopes)

        if not has_scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "insufficient_scope",
                    "message": f"Missing required scopes: {', '.join(sorted(missing))}",
                    "required": [str(s) for s in self.required_scopes],
                    "granted": list(client.scopes),
                },
            )

        return client


def require_scopes(*required_scopes: Scope, allow_anonymous: bool = False) -> ScopeChecker:
    """Create a FastAPI dependency that requires specific scopes.

    This is a convenience function that creates a ScopeChecker dependency.

    Args:
        *required_scopes: Scopes required for access.
        allow_anonymous: If True, allow unauthenticated access.

    Returns:
        A ScopeChecker dependency.

    Example:
        @router.post("/chat")
        async def chat(
            request: Request,
            client: APIKeyInfo = Depends(require_scopes(Scope.CHAT))
        ):
            # client is guaranteed to have CHAT scope
            ...
    """
    return ScopeChecker(*required_scopes, allow_anonymous=allow_anonymous)
