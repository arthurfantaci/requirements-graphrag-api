"""Structured event logging for guardrail operations.

This module provides a consistent event logging interface for security
monitoring and compliance tracking.

Event Types:
    - PROMPT_INJECTION_DETECTED: Detected but not blocked
    - PROMPT_INJECTION_BLOCKED: Detected and blocked
    - PII_DETECTED: PII found in input
    - PII_REDACTED: PII was redacted from input
    - RATE_LIMIT_EXCEEDED: Request exceeded rate limit
    - TOXICITY_DETECTED: Toxic content detected (Phase 2)
    - TOPIC_OUT_OF_SCOPE: Query outside allowed topics (Phase 2)
    - OUTPUT_FILTERED: LLM output was filtered (Phase 2)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

# Configure guardrail-specific logger
guardrail_logger = logging.getLogger("guardrails")


class GuardrailEventType(StrEnum):
    """Types of guardrail events for structured logging."""

    # Phase 1 - Critical Security
    PROMPT_INJECTION_DETECTED = "prompt_injection_detected"
    PROMPT_INJECTION_BLOCKED = "prompt_injection_blocked"
    PII_DETECTED = "pii_detected"
    PII_REDACTED = "pii_redacted"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

    # Phase 2 - Content Safety (for future use)
    TOXICITY_DETECTED = "toxicity_detected"
    TOXICITY_BLOCKED = "toxicity_blocked"
    TOPIC_OUT_OF_SCOPE = "topic_out_of_scope"
    OUTPUT_FILTERED = "output_filtered"

    # Phase 3 - Access Control (for future use)
    AUTH_FAILED = "auth_failed"
    SCOPE_VIOLATION = "scope_violation"

    # Phase 4 - Advanced (for future use)
    HALLUCINATION_DETECTED = "hallucination_detected"
    CONVERSATION_INVALID = "conversation_invalid"


class ActionTaken(StrEnum):
    """Actions taken in response to guardrail triggers."""

    ALLOWED = "allowed"  # Logged but allowed through
    WARNED = "warned"  # Logged with warning, allowed through
    BLOCKED = "blocked"  # Request was blocked
    REDACTED = "redacted"  # Content was modified (PII removed)
    FILTERED = "filtered"  # Output was filtered


@dataclass(slots=True)
class GuardrailEvent:
    """A guardrail event for structured logging.

    Attributes:
        event_type: The type of guardrail event.
        request_id: Unique identifier for the request.
        action_taken: What action was taken in response.
        timestamp: When the event occurred (UTC).
        user_ip: IP address of the requester (optional).
        api_key_id: Hashed API key identifier (optional).
        input_hash: SHA256 hash of input for correlation (not the actual input).
        risk_level: Risk level if applicable.
        details: Additional event-specific details.
    """

    event_type: GuardrailEventType
    request_id: str
    action_taken: ActionTaken
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    user_ip: str | None = None
    api_key_id: str | None = None
    input_hash: str | None = None
    risk_level: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the event.
        """
        return {
            "event_type": self.event_type.value,
            "request_id": self.request_id,
            "action_taken": self.action_taken.value,
            "timestamp": self.timestamp.isoformat(),
            "user_ip": self.user_ip,
            "api_key_id": self.api_key_id,
            "input_hash": self.input_hash,
            "risk_level": self.risk_level,
            "details": self.details,
        }


def compute_input_hash(text: str) -> str:
    """Compute SHA256 hash of input for correlation without storing content.

    Args:
        text: The input text to hash.

    Returns:
        Hex-encoded SHA256 hash (first 16 characters for brevity).
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def log_guardrail_event(
    event: GuardrailEvent,
    *,
    include_details: bool = True,
) -> None:
    """Log a guardrail event with structured format.

    Logs events to the 'guardrails' logger with appropriate severity levels
    based on the action taken.

    Args:
        event: The GuardrailEvent to log.
        include_details: Whether to include the details dict in the log.

    Example:
        >>> event = GuardrailEvent(
        ...     event_type=GuardrailEventType.PROMPT_INJECTION_BLOCKED,
        ...     request_id="req-123",
        ...     action_taken=ActionTaken.BLOCKED,
        ...     risk_level="high",
        ...     details={"patterns": ["instruction_override"]},
        ... )
        >>> log_guardrail_event(event)
    """
    # Determine log level based on action
    if event.action_taken == ActionTaken.BLOCKED or event.action_taken == ActionTaken.WARNED:
        log_level = logging.WARNING
    elif event.action_taken in (ActionTaken.REDACTED, ActionTaken.FILTERED):
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    # Build log message
    message_parts = [
        f"[{event.event_type.value}]",
        f"request_id={event.request_id}",
        f"action={event.action_taken.value}",
    ]

    if event.risk_level:
        message_parts.append(f"risk={event.risk_level}")

    if event.user_ip:
        message_parts.append(f"ip={event.user_ip}")

    if event.input_hash:
        message_parts.append(f"hash={event.input_hash}")

    if include_details and event.details:
        # Format details concisely
        details_str = ", ".join(f"{k}={v}" for k, v in event.details.items())
        message_parts.append(f"details=[{details_str}]")

    message = " ".join(message_parts)

    guardrail_logger.log(log_level, message, extra={"guardrail_event": event.to_dict()})


def create_injection_event(
    request_id: str,
    risk_level: str,
    patterns: tuple[str, ...],
    blocked: bool,
    user_ip: str | None = None,
    input_text: str | None = None,
) -> GuardrailEvent:
    """Create a prompt injection event.

    Args:
        request_id: Unique request identifier.
        risk_level: Detected risk level.
        patterns: Names of patterns that matched.
        blocked: Whether the request was blocked.
        user_ip: Client IP address.
        input_text: Original input (for hashing, not stored).

    Returns:
        Configured GuardrailEvent.
    """
    event_type = (
        GuardrailEventType.PROMPT_INJECTION_BLOCKED
        if blocked
        else GuardrailEventType.PROMPT_INJECTION_DETECTED
    )
    action = ActionTaken.BLOCKED if blocked else ActionTaken.WARNED

    return GuardrailEvent(
        event_type=event_type,
        request_id=request_id,
        action_taken=action,
        user_ip=user_ip,
        input_hash=compute_input_hash(input_text) if input_text else None,
        risk_level=risk_level,
        details={
            "patterns": list(patterns),
            "pattern_count": len(patterns),
        },
    )


def create_pii_event(
    request_id: str,
    entity_types: tuple[str, ...],
    entity_count: int,
    redacted: bool,
    user_ip: str | None = None,
    input_text: str | None = None,
) -> GuardrailEvent:
    """Create a PII detection event.

    Args:
        request_id: Unique request identifier.
        entity_types: Types of PII detected.
        entity_count: Number of PII entities found.
        redacted: Whether PII was redacted.
        user_ip: Client IP address.
        input_text: Original input (for hashing, not stored).

    Returns:
        Configured GuardrailEvent.
    """
    event_type = GuardrailEventType.PII_REDACTED if redacted else GuardrailEventType.PII_DETECTED
    action = ActionTaken.REDACTED if redacted else ActionTaken.WARNED

    return GuardrailEvent(
        event_type=event_type,
        request_id=request_id,
        action_taken=action,
        user_ip=user_ip,
        input_hash=compute_input_hash(input_text) if input_text else None,
        details={
            "entity_types": list(set(entity_types)),
            "entity_count": entity_count,
        },
    )


def create_rate_limit_event(
    request_id: str,
    limit: str,
    endpoint: str,
    user_ip: str | None = None,
    api_key_id: str | None = None,
) -> GuardrailEvent:
    """Create a rate limit exceeded event.

    Args:
        request_id: Unique request identifier.
        limit: The rate limit that was exceeded.
        endpoint: The endpoint that was rate limited.
        user_ip: Client IP address.
        api_key_id: Hashed API key identifier.

    Returns:
        Configured GuardrailEvent.
    """
    return GuardrailEvent(
        event_type=GuardrailEventType.RATE_LIMIT_EXCEEDED,
        request_id=request_id,
        action_taken=ActionTaken.BLOCKED,
        user_ip=user_ip,
        api_key_id=api_key_id,
        details={
            "limit": limit,
            "endpoint": endpoint,
        },
    )


# =============================================================================
# Phase 2 - Content Safety Events
# =============================================================================


def create_toxicity_event(
    request_id: str,
    categories: tuple[str, ...],
    confidence: float,
    blocked: bool,
    check_type: str,
    user_ip: str | None = None,
    input_text: str | None = None,
) -> GuardrailEvent:
    """Create a toxicity detection event.

    Args:
        request_id: Unique request identifier.
        categories: Detected toxicity categories.
        confidence: Confidence score of detection.
        blocked: Whether the request was blocked.
        check_type: Type of check performed ("fast" or "full").
        user_ip: Client IP address.
        input_text: Original input (for hashing, not stored).

    Returns:
        Configured GuardrailEvent.
    """
    event_type = (
        GuardrailEventType.TOXICITY_BLOCKED if blocked else GuardrailEventType.TOXICITY_DETECTED
    )
    action = ActionTaken.BLOCKED if blocked else ActionTaken.WARNED

    return GuardrailEvent(
        event_type=event_type,
        request_id=request_id,
        action_taken=action,
        user_ip=user_ip,
        input_hash=compute_input_hash(input_text) if input_text else None,
        risk_level="high" if blocked else "medium",
        details={
            "categories": list(categories),
            "confidence": confidence,
            "check_type": check_type,
        },
    )


def create_topic_event(
    request_id: str,
    classification: str,
    confidence: float,
    reasoning: str | None,
    check_type: str,
    user_ip: str | None = None,
    input_text: str | None = None,
) -> GuardrailEvent:
    """Create a topic classification event.

    Args:
        request_id: Unique request identifier.
        classification: Topic classification result.
        confidence: Confidence score of classification.
        reasoning: Explanation for the classification.
        check_type: Type of check performed ("keyword" or "llm").
        user_ip: Client IP address.
        input_text: Original input (for hashing, not stored).

    Returns:
        Configured GuardrailEvent.
    """
    # Only create event for out-of-scope classifications
    action = ActionTaken.WARNED if classification == "out_of_scope" else ActionTaken.ALLOWED

    return GuardrailEvent(
        event_type=GuardrailEventType.TOPIC_OUT_OF_SCOPE,
        request_id=request_id,
        action_taken=action,
        user_ip=user_ip,
        input_hash=compute_input_hash(input_text) if input_text else None,
        details={
            "classification": classification,
            "confidence": confidence,
            "reasoning": reasoning,
            "check_type": check_type,
        },
    )


def create_output_filter_event(
    request_id: str,
    is_safe: bool,
    confidence_score: float,
    warnings: tuple[str, ...],
    modifications: tuple[str, ...],
    blocked_reason: str | None = None,
    user_ip: str | None = None,
) -> GuardrailEvent:
    """Create an output filter event.

    Args:
        request_id: Unique request identifier.
        is_safe: Whether the output passed safety checks.
        confidence_score: Confidence score of the output.
        warnings: List of warnings about the output.
        modifications: List of modifications made.
        blocked_reason: Reason if output was blocked.
        user_ip: Client IP address.

    Returns:
        Configured GuardrailEvent.
    """
    if not is_safe:
        action = ActionTaken.BLOCKED
    elif modifications:
        action = ActionTaken.FILTERED
    else:
        action = ActionTaken.ALLOWED

    return GuardrailEvent(
        event_type=GuardrailEventType.OUTPUT_FILTERED,
        request_id=request_id,
        action_taken=action,
        user_ip=user_ip,
        risk_level="high" if not is_safe else ("medium" if warnings else "low"),
        details={
            "is_safe": is_safe,
            "confidence_score": confidence_score,
            "warnings": list(warnings),
            "modifications": list(modifications),
            "blocked_reason": blocked_reason,
        },
    )
