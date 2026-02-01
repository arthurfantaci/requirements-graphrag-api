"""Comprehensive audit logging for compliance and security analysis.

This module provides structured audit logging for:
- Authentication events (success, failure, key creation, revocation)
- API access events (requests, responses, errors)
- Guardrail events (triggered, blocked)
- Rate limiting events (warnings, exceeded)

Audit events are immutable, timestamped, and include request context
for compliance and security analysis.

Usage:
    from requirements_graphrag_api.auth.audit import (
        audit_logger,
        log_api_request,
        log_api_response,
        AuditEventType,
    )

    # Log an API request
    await log_api_request(request, client, request_id)

    # Log a response
    await log_api_response(request, client, request_id, status_code, duration_ms)

    # Custom event
    await audit_logger.log(AuditEvent(
        event_type=AuditEventType.AUTH_FAILURE,
        ...
    ))
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import Request

    from requirements_graphrag_api.auth.api_key import APIKeyInfo

# Dedicated audit logger with its own handlers
# Configure in your logging setup to route to appropriate destination
audit_log = logging.getLogger("audit")


class AuditEventType(StrEnum):
    """Types of audit events.

    Events are categorized by domain for easier filtering and analysis:
    - auth.*: Authentication and authorization events
    - api.*: API request/response lifecycle events
    - guardrail.*: Security guardrail events
    - rate_limit.*: Rate limiting events
    """

    # Authentication events
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    AUTH_KEY_CREATED = "auth.key_created"
    AUTH_KEY_REVOKED = "auth.key_revoked"
    AUTH_KEY_EXPIRED = "auth.key_expired"

    # API access events
    API_REQUEST = "api.request"
    API_RESPONSE = "api.response"
    API_ERROR = "api.error"

    # Guardrail events
    GUARDRAIL_TRIGGERED = "guardrail.triggered"
    GUARDRAIL_BLOCKED = "guardrail.blocked"
    GUARDRAIL_PASSED = "guardrail.passed"

    # Rate limiting events
    RATE_LIMIT_WARNING = "rate_limit.warning"
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"


@dataclass(frozen=True)
class AuditEvent:
    """Structured audit event.

    Immutable (frozen) to ensure audit records cannot be modified
    after creation, maintaining integrity for compliance.

    Attributes:
        event_type: The type of audit event.
        timestamp: When the event occurred (UTC).
        request_id: Unique identifier for request correlation.
        client_id: API key ID if authenticated, None otherwise.
        client_ip: Client IP address if available.
        endpoint: The API endpoint path.
        method: HTTP method (GET, POST, etc.).
        status_code: HTTP status code (for response events).
        duration_ms: Request duration in milliseconds.
        details: Additional event-specific details.
    """

    event_type: AuditEventType
    timestamp: datetime
    request_id: str
    client_id: str | None
    client_ip: str | None
    endpoint: str
    method: str
    status_code: int | None = None
    duration_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization.

        Returns:
            Dictionary representation with ISO timestamp.
        """
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["event_type"] = self.event_type.value
        return data

    def to_json(self) -> str:
        """Convert to JSON string for logging.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), default=str)


class AuditHandler:
    """Base class for audit event handlers.

    Subclass this to implement custom audit event destinations
    (e.g., database, external service, file).

    Handlers are async to support non-blocking I/O operations.
    """

    async def handle(self, event: AuditEvent) -> None:
        """Handle an audit event.

        Args:
            event: The audit event to process.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def close(self) -> None:
        """Clean up handler resources.

        Called during application shutdown.
        """


class LoggingAuditHandler(AuditHandler):
    """Handler that logs audit events to Python logging.

    Uses structured logging with JSON format for easy parsing
    by log aggregation systems.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the logging handler.

        Args:
            logger: Logger to use. Defaults to the 'audit' logger.
        """
        self.logger = logger or audit_log

    async def handle(self, event: AuditEvent) -> None:
        """Log the audit event.

        Args:
            event: The audit event to log.
        """
        # Determine log level based on event type
        if event.event_type in (
            AuditEventType.AUTH_FAILURE,
            AuditEventType.GUARDRAIL_BLOCKED,
            AuditEventType.RATE_LIMIT_EXCEEDED,
            AuditEventType.API_ERROR,
        ):
            level = logging.WARNING
        elif event.event_type in (
            AuditEventType.RATE_LIMIT_WARNING,
            AuditEventType.GUARDRAIL_TRIGGERED,
        ):
            level = logging.INFO
        else:
            level = logging.INFO

        self.logger.log(
            level,
            "Audit: %s %s %s",
            event.event_type.value,
            event.method,
            event.endpoint,
            extra={"audit_event": event.to_dict()},
        )


class LangSmithAuditHandler(AuditHandler):
    """Handler that sends audit events to LangSmith as feedback.

    Integrates audit events with LangSmith for correlation with
    LLM traces and performance analysis.
    """

    def __init__(self, enabled: bool = False) -> None:
        """Initialize the LangSmith handler.

        Args:
            enabled: Whether to actually send events to LangSmith.
        """
        self.enabled = enabled

    async def handle(self, event: AuditEvent) -> None:
        """Send audit event to LangSmith.

        Currently a placeholder - implement integration with LangSmith
        feedback API when needed.

        Args:
            event: The audit event to send.
        """
        if not self.enabled:
            return

        # TODO: Implement LangSmith integration
        # Could use langsmith.Client().create_feedback() to correlate
        # audit events with LLM traces


class AuditLogger:
    """Audit logger with configurable handlers.

    Central point for logging audit events to multiple destinations.
    Handlers are processed in order, and failures in one handler
    don't prevent others from receiving the event.

    Example:
        logger = AuditLogger()
        logger.add_handler(LoggingAuditHandler())
        logger.add_handler(DatabaseAuditHandler(db))

        await logger.log(event)
    """

    def __init__(self) -> None:
        """Initialize the audit logger with no handlers."""
        self._handlers: list[AuditHandler] = []

    def add_handler(self, handler: AuditHandler) -> None:
        """Add a handler to receive audit events.

        Args:
            handler: The handler to add.
        """
        self._handlers.append(handler)

    def remove_handler(self, handler: AuditHandler) -> None:
        """Remove a handler.

        Args:
            handler: The handler to remove.
        """
        self._handlers.remove(handler)

    async def log(self, event: AuditEvent) -> None:
        """Log an audit event to all handlers.

        Errors in individual handlers are caught and logged,
        but don't prevent other handlers from receiving the event.

        Args:
            event: The audit event to log.
        """
        for handler in self._handlers:
            try:
                await handler.handle(event)
            except Exception as e:
                # Use standard logger to avoid infinite recursion
                logging.getLogger(__name__).exception(
                    "Audit handler error: %s",
                    e,
                    extra={"handler": handler.__class__.__name__},
                )

    async def close(self) -> None:
        """Close all handlers.

        Call during application shutdown to clean up resources.
        """
        for handler in self._handlers:
            try:
                await handler.close()
            except Exception as e:
                logging.getLogger(__name__).exception("Error closing audit handler: %s", e)


# Global audit logger instance
# Add handlers during application startup
audit_logger = AuditLogger()


def configure_audit_logging(
    enable_logging: bool = True,
    enable_langsmith: bool = False,
) -> None:
    """Configure the global audit logger with default handlers.

    Call this during application startup to set up audit logging.

    Args:
        enable_logging: Whether to enable Python logging handler.
        enable_langsmith: Whether to enable LangSmith handler.
    """
    if enable_logging:
        audit_logger.add_handler(LoggingAuditHandler())
    if enable_langsmith:
        audit_logger.add_handler(LangSmithAuditHandler(enabled=True))


# Convenience functions for common audit events


def _get_client_ip(request: Request) -> str | None:
    """Extract client IP from request.

    Handles X-Forwarded-For header for proxied requests.

    Args:
        request: The FastAPI request.

    Returns:
        Client IP address or None.
    """
    # Check for forwarded IP first (behind proxy/load balancer)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take the first IP in the chain (original client)
        return forwarded.split(",")[0].strip()
    # Direct connection
    if request.client:
        return request.client.host
    return None


async def log_api_request(
    request: Request,
    client: APIKeyInfo | None,
    request_id: str,
) -> None:
    """Log an API request event.

    Call this at the start of request processing for comprehensive
    request logging.

    Args:
        request: The FastAPI request.
        client: The authenticated client info, or None.
        request_id: Unique request identifier.
    """
    await audit_logger.log(
        AuditEvent(
            event_type=AuditEventType.API_REQUEST,
            timestamp=datetime.now(UTC),
            request_id=request_id,
            client_id=client.key_id if client else None,
            client_ip=_get_client_ip(request),
            endpoint=request.url.path,
            method=request.method,
            status_code=None,
            duration_ms=None,
            details={
                "query_params": dict(request.query_params),
                "content_type": request.headers.get("content-type"),
                "user_agent": request.headers.get("user-agent"),
                "tier": client.tier if client else None,
            },
        )
    )


async def log_api_response(
    request: Request,
    client: APIKeyInfo | None,
    request_id: str,
    status_code: int,
    duration_ms: float,
) -> None:
    """Log an API response event.

    Call this after request processing completes.

    Args:
        request: The FastAPI request.
        client: The authenticated client info, or None.
        request_id: Unique request identifier.
        status_code: HTTP response status code.
        duration_ms: Request duration in milliseconds.
    """
    await audit_logger.log(
        AuditEvent(
            event_type=AuditEventType.API_RESPONSE,
            timestamp=datetime.now(UTC),
            request_id=request_id,
            client_id=client.key_id if client else None,
            client_ip=_get_client_ip(request),
            endpoint=request.url.path,
            method=request.method,
            status_code=status_code,
            duration_ms=duration_ms,
            details={
                "tier": client.tier if client else None,
            },
        )
    )


async def log_api_error(
    request: Request,
    client: APIKeyInfo | None,
    request_id: str,
    status_code: int,
    error_type: str,
    error_message: str,
    duration_ms: float | None = None,
) -> None:
    """Log an API error event.

    Call this when an error occurs during request processing.

    Args:
        request: The FastAPI request.
        client: The authenticated client info, or None.
        request_id: Unique request identifier.
        status_code: HTTP error status code.
        error_type: Error type/code for categorization.
        error_message: Human-readable error message.
        duration_ms: Request duration if available.
    """
    await audit_logger.log(
        AuditEvent(
            event_type=AuditEventType.API_ERROR,
            timestamp=datetime.now(UTC),
            request_id=request_id,
            client_id=client.key_id if client else None,
            client_ip=_get_client_ip(request),
            endpoint=request.url.path,
            method=request.method,
            status_code=status_code,
            duration_ms=duration_ms,
            details={
                "error_type": error_type,
                "error_message": error_message,
                "tier": client.tier if client else None,
            },
        )
    )


async def log_auth_success(
    request: Request,
    client: APIKeyInfo,
    request_id: str,
) -> None:
    """Log a successful authentication event.

    Args:
        request: The FastAPI request.
        client: The authenticated client info.
        request_id: Unique request identifier.
    """
    await audit_logger.log(
        AuditEvent(
            event_type=AuditEventType.AUTH_SUCCESS,
            timestamp=datetime.now(UTC),
            request_id=request_id,
            client_id=client.key_id,
            client_ip=_get_client_ip(request),
            endpoint=request.url.path,
            method=request.method,
            details={
                "tier": client.tier,
                "organization": client.organization,
                "scopes": list(client.scopes),
            },
        )
    )


async def log_auth_failure(
    request: Request,
    request_id: str,
    reason: str,
) -> None:
    """Log a failed authentication attempt.

    Args:
        request: The FastAPI request.
        request_id: Unique request identifier.
        reason: Why authentication failed.
    """
    await audit_logger.log(
        AuditEvent(
            event_type=AuditEventType.AUTH_FAILURE,
            timestamp=datetime.now(UTC),
            request_id=request_id,
            client_id=None,
            client_ip=_get_client_ip(request),
            endpoint=request.url.path,
            method=request.method,
            details={
                "reason": reason,
            },
        )
    )


async def log_guardrail_event(
    request: Request,
    client: APIKeyInfo | None,
    request_id: str,
    guardrail_name: str,
    triggered: bool,
    blocked: bool,
    details: dict[str, Any] | None = None,
) -> None:
    """Log a guardrail event.

    Args:
        request: The FastAPI request.
        client: The authenticated client info, or None.
        request_id: Unique request identifier.
        guardrail_name: Name of the guardrail (e.g., "prompt_injection").
        triggered: Whether the guardrail was triggered.
        blocked: Whether the request was blocked.
        details: Additional guardrail-specific details.
    """
    if blocked:
        event_type = AuditEventType.GUARDRAIL_BLOCKED
    elif triggered:
        event_type = AuditEventType.GUARDRAIL_TRIGGERED
    else:
        event_type = AuditEventType.GUARDRAIL_PASSED

    await audit_logger.log(
        AuditEvent(
            event_type=event_type,
            timestamp=datetime.now(UTC),
            request_id=request_id,
            client_id=client.key_id if client else None,
            client_ip=_get_client_ip(request),
            endpoint=request.url.path,
            method=request.method,
            details={
                "guardrail": guardrail_name,
                "triggered": triggered,
                "blocked": blocked,
                **(details or {}),
            },
        )
    )


async def log_rate_limit_event(
    request: Request,
    client: APIKeyInfo | None,
    request_id: str,
    exceeded: bool,
    limit: str,
    current_count: int | None = None,
) -> None:
    """Log a rate limit event.

    Args:
        request: The FastAPI request.
        client: The authenticated client info, or None.
        request_id: Unique request identifier.
        exceeded: Whether the rate limit was exceeded.
        limit: The rate limit string (e.g., "50/minute").
        current_count: Current request count if available.
    """
    event_type = (
        AuditEventType.RATE_LIMIT_EXCEEDED if exceeded else AuditEventType.RATE_LIMIT_WARNING
    )

    await audit_logger.log(
        AuditEvent(
            event_type=event_type,
            timestamp=datetime.now(UTC),
            request_id=request_id,
            client_id=client.key_id if client else None,
            client_ip=_get_client_ip(request),
            endpoint=request.url.path,
            method=request.method,
            details={
                "limit": limit,
                "current_count": current_count,
                "tier": client.tier if client else None,
            },
        )
    )
