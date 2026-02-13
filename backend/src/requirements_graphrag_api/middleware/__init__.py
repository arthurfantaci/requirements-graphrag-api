"""Middleware module for request/response processing.

This module provides middleware components for the GraphRAG API:
- Rate limiting to prevent abuse (Phase 1)
- Request size limits (Phase 4)
- Timeout handling (Phase 4)

Usage:
    from requirements_graphrag_api.middleware import (
        SizeLimitMiddleware,
        with_timeout,
        TIMEOUTS,
    )

    # Add middleware to app
    app.add_middleware(SizeLimitMiddleware)

    # Use timeout decorator
    @with_timeout(TIMEOUTS["chat"])
    async def chat_handler():
        ...
"""

from __future__ import annotations

from requirements_graphrag_api.middleware.rate_limit import (
    get_rate_limit_key,
    get_rate_limiter,
    rate_limit_exceeded_handler,
)
from requirements_graphrag_api.middleware.size_limit import (
    MAX_REQUEST_SIZE,
    MAX_RESPONSE_SIZE,
    SizeLimitMiddleware,
    check_request_size,
)
from requirements_graphrag_api.middleware.timeout import (
    TIMEOUTS,
    get_timeout_for_endpoint,
    run_with_timeout,
    with_timeout,
)
from requirements_graphrag_api.middleware.tracing import (
    TraceCorrelationMiddleware,
)

__all__ = [
    "MAX_REQUEST_SIZE",
    "MAX_RESPONSE_SIZE",
    "TIMEOUTS",
    "SizeLimitMiddleware",
    "TraceCorrelationMiddleware",
    "check_request_size",
    "get_rate_limit_key",
    "get_rate_limiter",
    "get_timeout_for_endpoint",
    "rate_limit_exceeded_handler",
    "run_with_timeout",
    "with_timeout",
]
