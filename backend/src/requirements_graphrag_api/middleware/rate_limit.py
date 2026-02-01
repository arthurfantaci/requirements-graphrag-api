"""Rate limiting middleware using SlowAPI.

This module provides request rate limiting to prevent abuse and control costs.

Rate Limits by Endpoint:
    - /chat: 20/minute (LLM calls are expensive)
    - /search/*: 60/minute (Vector search is cheaper)
    - /feedback: 30/minute (Prevent spam)
    - /health: 120/minute (Monitoring tools)
    - Default: 100/minute (Catch-all)
"""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded  # noqa: TC002 - used at runtime in handler
from slowapi.util import get_remote_address
from starlette.requests import Request  # noqa: TC002 - used at runtime for request attrs
from starlette.responses import JSONResponse

from requirements_graphrag_api.config import get_guardrail_config
from requirements_graphrag_api.guardrails.events import (
    create_rate_limit_event,
    log_guardrail_event,
)

logger = logging.getLogger(__name__)


def get_rate_limit_key(request: Request) -> str:
    """Get the rate limit key for a request.

    Uses API key if present, otherwise falls back to IP address.
    This allows different rate limits per authenticated user.

    Args:
        request: The incoming request.

    Returns:
        String key for rate limiting (IP or API key hash).
    """
    # Check for API key in Authorization header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        # Use a hash of the API key as the limit key
        # This allows per-key rate limiting without exposing the key
        api_key = auth_header[7:]
        import hashlib

        return f"api:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"

    # Check for API key in query parameter (alternative auth method)
    api_key = request.query_params.get("api_key")
    if api_key:
        import hashlib

        return f"api:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"

    # Fall back to IP address
    return get_remote_address(request)


@lru_cache(maxsize=1)
def get_rate_limiter() -> Limiter:
    """Get or create the rate limiter instance (singleton).

    Returns:
        Configured Limiter instance.

    Note:
        The limiter uses in-memory storage by default. For distributed
        deployments, configure a Redis backend.
    """
    config = get_guardrail_config()

    limiter = Limiter(
        key_func=get_rate_limit_key,
        default_limits=[config.rate_limit_default],
        enabled=config.rate_limiting_enabled,
        # Storage options:
        # - "memory://" (default): In-memory, lost on restart
        # - "redis://localhost:6379": Redis for distributed
        # - "memcached://localhost:11211": Memcached
        storage_uri="memory://",
        strategy="fixed-window",  # "fixed-window" or "moving-window"
    )

    logger.info(
        "Rate limiter initialized: enabled=%s, default=%s",
        config.rate_limiting_enabled,
        config.rate_limit_default,
    )

    return limiter


async def rate_limit_exceeded_handler(
    request: Request,
    exc: RateLimitExceeded,
) -> JSONResponse:
    """Custom handler for rate limit exceeded errors.

    Provides a structured JSON response with retry information
    and logs the event for monitoring.

    Args:
        request: The rate-limited request.
        exc: The rate limit exception.

    Returns:
        JSONResponse with 429 status and retry information.
    """
    # Parse the retry-after from the exception detail
    # exc.detail format: "Rate limit exceeded: X per Y"
    limit_str = str(exc.detail).replace("Rate limit exceeded: ", "")

    # Calculate retry-after seconds (approximate)
    retry_after = _parse_retry_after(limit_str)

    # Generate request ID for tracking
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])

    # Get client identifier
    client_ip = get_remote_address(request)

    # Log the rate limit event
    event = create_rate_limit_event(
        request_id=request_id,
        limit=limit_str,
        endpoint=request.url.path,
        user_ip=client_ip,
    )
    log_guardrail_event(event)

    response_body = {
        "error": "rate_limit_exceeded",
        "message": f"Rate limit exceeded. Try again in {retry_after} seconds.",
        "retry_after": retry_after,
        "limit": limit_str,
    }

    return JSONResponse(
        status_code=429,
        content=response_body,
        headers={
            "Retry-After": str(retry_after),
            "X-RateLimit-Limit": limit_str,
            "X-Request-ID": request_id,
        },
    )


def _parse_retry_after(limit_str: str) -> int:
    """Parse retry-after seconds from limit string.

    Args:
        limit_str: Rate limit string like "20 per minute".

    Returns:
        Approximate seconds to wait before retry.
    """
    limit_lower = limit_str.lower()

    if "second" in limit_lower:
        return 1
    if "minute" in limit_lower:
        return 60
    if "hour" in limit_lower:
        return 3600
    if "day" in limit_lower:
        return 86400

    # Default to 60 seconds
    return 60


# Rate limit decorators for different endpoints
# These are used in route definitions

# Chat endpoint - expensive LLM calls
CHAT_RATE_LIMIT = "20/minute"

# Search endpoints - cheaper vector operations
SEARCH_RATE_LIMIT = "60/minute"

# Feedback endpoint - prevent spam
FEEDBACK_RATE_LIMIT = "30/minute"

# Health endpoint - allow more for monitoring
HEALTH_RATE_LIMIT = "120/minute"


def get_endpoint_limit(endpoint: str) -> str:
    """Get the rate limit for a specific endpoint.

    Args:
        endpoint: The endpoint path.

    Returns:
        Rate limit string for the endpoint.
    """
    config = get_guardrail_config()

    if "/chat" in endpoint:
        return config.rate_limit_chat
    if "/search" in endpoint:
        return config.rate_limit_search
    if "/feedback" in endpoint:
        return FEEDBACK_RATE_LIMIT
    if "/health" in endpoint:
        return HEALTH_RATE_LIMIT

    return config.rate_limit_default
