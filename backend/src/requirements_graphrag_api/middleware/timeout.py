"""Request timeout middleware and decorators.

This module provides timeout protection for async operations to prevent
hanging requests and resource exhaustion.

Default Timeouts by Operation:
    - chat: 60s (LLM generation can be slow)
    - search: 30s (Vector/graph search should be fast)
    - cypher: 30s (Graph queries)
    - feedback: 10s (Simple DB operation)
    - health: 5s (Should be instant)

Usage:
    from requirements_graphrag_api.middleware.timeout import with_timeout, TIMEOUTS

    @router.post("/chat")
    @with_timeout(TIMEOUTS["chat"])
    async def chat(request: Request):
        ...

    # Or with custom timeout
    @with_timeout(45.0)
    async def custom_operation():
        ...
"""

from __future__ import annotations

import asyncio
import logging
from functools import wraps
from typing import TYPE_CHECKING, ParamSpec, TypeVar

from fastapi import HTTPException, status

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


# Default timeouts by operation type (in seconds)
TIMEOUTS: dict[str, float] = {
    "chat": 60.0,  # LLM generation can be slow
    "search": 30.0,  # Vector search should be fast
    "cypher": 30.0,  # Graph queries
    "feedback": 10.0,  # Simple DB operation
    "health": 5.0,  # Should be instant
    "default": 30.0,  # Catch-all
}


def with_timeout(
    seconds: float,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator to add timeout to async functions.

    Wraps an async function with asyncio.wait_for to enforce a timeout.
    If the function doesn't complete within the timeout, raises HTTP 504.

    Args:
        seconds: Maximum execution time in seconds.

    Returns:
        Decorator function.

    Example:
        @with_timeout(30.0)
        async def slow_operation():
            await some_external_call()
            return result

    Raises:
        HTTPException: 504 if operation times out.
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds,
                )
            except TimeoutError:
                func_name = func.__name__
                logger.warning(
                    "Operation timed out: function=%s, timeout=%.1fs",
                    func_name,
                    seconds,
                )
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail={
                        "error": "request_timeout",
                        "message": f"Request timed out after {seconds:.0f} seconds",
                        "timeout_seconds": seconds,
                    },
                ) from None

        return wrapper

    return decorator


async def run_with_timeout[T](
    coro: Awaitable[T],
    timeout: float,
    operation: str = "operation",
) -> T:
    """Run a coroutine with a timeout.

    Utility function for one-off timeout wrapping when decorator isn't suitable.

    Args:
        coro: The coroutine to execute.
        timeout: Maximum execution time in seconds.
        operation: Description of the operation (for error messages).

    Returns:
        Result of the coroutine.

    Raises:
        HTTPException: 504 if operation times out.

    Example:
        result = await run_with_timeout(
            external_api.fetch_data(),
            timeout=30.0,
            operation="fetch_data"
        )
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except TimeoutError:
        logger.warning(
            "Operation timed out: operation=%s, timeout=%.1fs",
            operation,
            timeout,
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "error": "request_timeout",
                "message": f"{operation} timed out after {timeout:.0f} seconds",
                "timeout_seconds": timeout,
            },
        ) from None


def get_timeout_for_endpoint(endpoint: str) -> float:
    """Get the appropriate timeout for an endpoint.

    Args:
        endpoint: The endpoint path (e.g., "/chat", "/search/hybrid").

    Returns:
        Timeout in seconds for the endpoint.

    Example:
        timeout = get_timeout_for_endpoint("/chat")
        # Returns 60.0
    """
    if "/chat" in endpoint:
        return TIMEOUTS["chat"]
    if "/search" in endpoint:
        return TIMEOUTS["search"]
    if "/cypher" in endpoint or "/graph" in endpoint:
        return TIMEOUTS["cypher"]
    if "/feedback" in endpoint:
        return TIMEOUTS["feedback"]
    if "/health" in endpoint:
        return TIMEOUTS["health"]

    return TIMEOUTS["default"]
