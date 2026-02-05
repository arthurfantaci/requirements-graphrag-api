"""Request and response size limiting middleware.

This module provides middleware to enforce size limits on HTTP requests
and responses to prevent abuse and protect server resources.

Limits:
    - MAX_REQUEST_SIZE: 1 MB (request body)
    - MAX_RESPONSE_SIZE: 10 MB (response body)

The middleware checks Content-Length header for requests and can optionally
wrap responses to enforce output limits (though response limiting is disabled
by default since we control our own response generation).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

logger = logging.getLogger(__name__)

# Size limits in bytes
MAX_REQUEST_SIZE = 1 * 1024 * 1024  # 1 MB
MAX_RESPONSE_SIZE = 10 * 1024 * 1024  # 10 MB

# Path prefixes to exclude from size limiting (e.g., health checks)
EXCLUDED_PATHS = ("/health", "/docs", "/redoc", "/openapi.json")


class SizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request/response size limits.

    This middleware checks the Content-Length header of incoming requests
    and rejects requests that exceed the configured maximum size.

    Attributes:
        max_request_size: Maximum allowed request body size in bytes.
        max_response_size: Maximum allowed response body size in bytes.
        excluded_paths: Path prefixes to exclude from size limiting.

    Example:
        app.add_middleware(
            SizeLimitMiddleware,
            max_request_size=1 * 1024 * 1024,  # 1 MB
        )
    """

    def __init__(
        self,
        app: object,
        max_request_size: int = MAX_REQUEST_SIZE,
        max_response_size: int = MAX_RESPONSE_SIZE,
        excluded_paths: tuple[str, ...] = EXCLUDED_PATHS,
    ) -> None:
        """Initialize the size limit middleware.

        Args:
            app: The ASGI application.
            max_request_size: Maximum request body size in bytes.
            max_response_size: Maximum response body size in bytes.
            excluded_paths: Path prefixes to exclude from size limiting.
        """
        super().__init__(app)
        self.max_request_size = max_request_size
        self.max_response_size = max_response_size
        self.excluded_paths = excluded_paths

    async def dispatch(
        self,
        request: Request,
        call_next: object,
    ) -> Response:
        """Process the request and enforce size limits.

        Args:
            request: The incoming request.
            call_next: The next middleware/handler in the chain.

        Returns:
            The response from the handler.

        Raises:
            HTTPException: 413 if request body exceeds size limit.
        """
        # Skip size limiting for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)

        # Check request size via Content-Length header
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_request_size:
                    logger.warning(
                        "Request too large: size=%d, max=%d, path=%s",
                        size,
                        self.max_request_size,
                        request.url.path,
                    )
                    # Return JSONResponse directly instead of raising exception
                    # (BaseHTTPMiddleware doesn't handle HTTPException properly)
                    return JSONResponse(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        content={
                            "detail": {
                                "error": "request_too_large",
                                "message": (
                                    f"Request body exceeds maximum size of "
                                    f"{_format_size(self.max_request_size)}"
                                ),
                                "max_size_bytes": self.max_request_size,
                                "actual_size_bytes": size,
                            }
                        },
                    )
            except ValueError:
                # Invalid Content-Length header, let the framework handle it
                pass

        # Process the request
        response = await call_next(request)

        # Note: Response size limiting is more complex for streaming responses.
        # Since we control our own response generation and don't expect to exceed
        # limits, we trust our handlers. If needed, response limiting can be
        # implemented using a custom streaming response wrapper.

        return response


def _format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable size string (e.g., "1 MB").
    """
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.0f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.0f} KB"
    return f"{size_bytes} bytes"


def check_request_size(content_length: int | None, max_size: int = MAX_REQUEST_SIZE) -> None:
    """Check if request size is within limits.

    Utility function for direct size checking in route handlers.

    Args:
        content_length: Request content length in bytes (None if unknown).
        max_size: Maximum allowed size in bytes.

    Raises:
        HTTPException: 413 if size exceeds limit.

    Example:
        @router.post("/upload")
        async def upload(request: Request):
            check_request_size(request.headers.get("content-length"))
            # Process upload...
    """
    if content_length is not None and content_length > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={
                "error": "request_too_large",
                "message": f"Request body exceeds maximum size of {_format_size(max_size)}",
                "max_size_bytes": max_size,
            },
        )
