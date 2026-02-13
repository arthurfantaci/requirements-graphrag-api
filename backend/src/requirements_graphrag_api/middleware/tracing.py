"""Trace correlation middleware (raw ASGI, NOT BaseHTTPMiddleware).

Reads the OTel trace ID from the current span context (set by
FastAPIInstrumentor), stores it in ``scope["state"]["trace_id"]``,
and injects ``X-Trace-ID`` response header.

Must be registered AFTER ``FastAPIInstrumentor.instrument_app(app)``
so the OTel span is active when this middleware runs.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class TraceCorrelationMiddleware:
    """Raw ASGI middleware for trace ID propagation."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:  # noqa: D102
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        trace_id = _get_otel_trace_id() or str(uuid.uuid4())

        # Store in ASGI scope state (accessible via request.state.trace_id)
        scope.setdefault("state", {})
        scope["state"]["trace_id"] = trace_id

        async def send_with_trace_header(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-trace-id", trace_id.encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_trace_header)


def _get_otel_trace_id() -> str | None:
    """Extract trace ID from the active OTel span, if available."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.trace_id:
            return format(ctx.trace_id, "032x")
    except Exception:
        logger.debug("OTel trace ID extraction failed", exc_info=True)
    return None


__all__ = ["TraceCorrelationMiddleware"]
