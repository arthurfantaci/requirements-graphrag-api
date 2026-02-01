"""Middleware module for request/response processing.

This module provides middleware components for the GraphRAG API:
- Rate limiting to prevent abuse
- Request size limits (Phase 4)
- Timeout handling (Phase 4)
"""

from __future__ import annotations

from requirements_graphrag_api.middleware.rate_limit import (
    get_rate_limit_key,
    get_rate_limiter,
    rate_limit_exceeded_handler,
)

__all__ = [
    "get_rate_limit_key",
    "get_rate_limiter",
    "rate_limit_exceeded_handler",
]
