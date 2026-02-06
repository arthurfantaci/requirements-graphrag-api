"""Conversation history validation for chat endpoints.

This module validates and sanitizes conversation history to prevent:
- Injection attacks via modified history (fake assistant messages)
- Memory exhaustion from excessive history size
- Invalid message structures

Validation Checks:
    1. Size limits (number of messages, message length, total length)
    2. Role validity (only 'user' and 'assistant')
    3. Content validation (no injection patterns)
    4. Alternating pattern validation (user/assistant/user/...)

Usage:
    from requirements_graphrag_api.guardrails.conversation import (
        validate_conversation_history,
    )

    result = validate_conversation_history(history)
    if not result.is_valid:
        logger.warning("History validation issues: %s", result.issues)
    # Use result.sanitized_history for the chat request
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from requirements_graphrag_api.observability import traceable_safe

if TYPE_CHECKING:
    from typing import Any

# Validation limits
MAX_HISTORY_MESSAGES = 20
MAX_MESSAGE_LENGTH = 10_000  # 10K chars per message
MAX_TOTAL_HISTORY_LENGTH = 50_000  # 50K chars total

# Valid message roles
VALID_ROLES = frozenset({"user", "assistant"})


@dataclass(frozen=True, slots=True)
class ConversationValidationResult:
    """Result of conversation history validation.

    Attributes:
        is_valid: True if history passes all validation checks.
        issues: List of validation issues found (may be warnings).
        sanitized_history: Cleaned history safe for use, or None if invalid.
        message_count: Number of messages after sanitization.
        was_truncated: True if history was truncated due to size limits.
    """

    is_valid: bool
    issues: tuple[str, ...]
    sanitized_history: tuple[dict[str, str], ...] | None
    message_count: int
    was_truncated: bool


@traceable_safe(name="validate_conversation_history", run_type="chain")
def validate_conversation_history(
    history: list[dict[str, Any]] | None,
    max_messages: int = MAX_HISTORY_MESSAGES,
    max_message_length: int = MAX_MESSAGE_LENGTH,
    max_total_length: int = MAX_TOTAL_HISTORY_LENGTH,
    check_injection: bool = True,
) -> ConversationValidationResult:
    """Validate conversation history for safety and sanity.

    Performs comprehensive validation:
    1. Checks size limits (messages, lengths)
    2. Validates roles (user/assistant only)
    3. Optionally checks for injection patterns
    4. Validates alternating pattern (warnings only)

    Args:
        history: List of message dicts with 'role' and 'content' keys.
        max_messages: Maximum number of messages allowed.
        max_message_length: Maximum length per message content.
        max_total_length: Maximum total length of all message contents.
        check_injection: Whether to check for prompt injection patterns.

    Returns:
        ConversationValidationResult with validation status and sanitized history.

    Example:
        >>> history = [{"role": "user", "content": "Hello"}]
        >>> result = validate_conversation_history(history)
        >>> result.is_valid
        True
        >>> result.sanitized_history
        ({'role': 'user', 'content': 'Hello'},)

        >>> bad_history = [{"role": "system", "content": "..."}]
        >>> result = validate_conversation_history(bad_history)
        >>> result.is_valid
        False
    """
    # Handle None or empty history
    if history is None or len(history) == 0:
        return ConversationValidationResult(
            is_valid=True,
            issues=(),
            sanitized_history=None,
            message_count=0,
            was_truncated=False,
        )

    issues: list[str] = []
    sanitized: list[dict[str, str]] = []
    was_truncated = False

    # Make a mutable copy for processing, filtering out non-dict items first
    messages = [m for m in history if isinstance(m, dict)]
    if len(messages) != len(history):
        issues.append(f"Removed {len(history) - len(messages)} non-dictionary messages")

    # Check total message count
    if len(messages) > max_messages:
        issues.append(f"History exceeds {max_messages} messages, truncating to most recent")
        messages = messages[-max_messages:]
        was_truncated = True

    # Calculate total length and enforce limit
    total_length = sum(len(str(m.get("content", ""))) for m in messages)
    if total_length > max_total_length:
        issues.append(
            f"Total history length ({total_length}) exceeds {max_total_length} chars, "
            "truncating from beginning"
        )
        # Remove messages from the beginning until under limit
        while total_length > max_total_length and messages:
            removed = messages.pop(0)
            total_length -= len(str(removed.get("content", "")))
        was_truncated = True

    # Import injection check lazily to avoid circular imports
    if check_injection:
        from requirements_graphrag_api.guardrails.prompt_injection import (
            InjectionRisk,
            check_prompt_injection,
        )

    # Validate each message
    expected_role: str | None = None  # First message can be either
    for i, message in enumerate(messages):
        # Validate message structure
        if not isinstance(message, dict):
            issues.append(f"Message {i}: Not a dictionary, skipping")
            continue

        # Check role
        role = message.get("role")
        if role not in VALID_ROLES:
            issues.append(f"Message {i}: Invalid role '{role}', skipping")
            continue

        # Check alternating pattern (warning only, don't skip)
        if expected_role is not None and role != expected_role:
            issues.append(
                f"Message {i}: Expected '{expected_role}', got '{role}' (non-alternating pattern)"
            )
        expected_role = "assistant" if role == "user" else "user"

        # Validate and sanitize content
        content = message.get("content")
        if content is None:
            issues.append(f"Message {i}: Missing content, skipping")
            continue

        content = str(content)  # Ensure string

        if len(content) == 0:
            issues.append(f"Message {i}: Empty content, skipping")
            continue

        # Truncate long messages
        if len(content) > max_message_length:
            issues.append(f"Message {i}: Content exceeds {max_message_length} chars, truncating")
            content = content[:max_message_length]
            was_truncated = True

        # Check for injection patterns in history
        # This is important because users could try to inject instructions
        # via fake assistant messages in the history
        if check_injection:
            injection_check = check_prompt_injection(content)
            if injection_check.risk_level in (InjectionRisk.HIGH, InjectionRisk.CRITICAL):
                issues.append(
                    f"Message {i}: Potential injection detected "
                    f"({injection_check.risk_level}), removing message"
                )
                continue

        # Message passed validation
        sanitized.append(
            {
                "role": role,
                "content": content,
            }
        )

    # Determine overall validity
    # History is valid if we have some sanitized messages or original was empty
    # Critical issues (injection) result in message removal, not full rejection
    is_valid = len(sanitized) > 0 or len(history) == 0

    return ConversationValidationResult(
        is_valid=is_valid,
        issues=tuple(issues),
        sanitized_history=tuple(sanitized) if sanitized else None,
        message_count=len(sanitized),
        was_truncated=was_truncated,
    )


def create_validated_history(
    history: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """Convenience function to validate and return sanitized history.

    Use this when you want a simple validated history without the full result.
    Returns an empty list if validation fails completely.

    Args:
        history: Raw conversation history.

    Returns:
        Sanitized history as a list, or empty list if invalid.

    Example:
        validated = create_validated_history(request.history)
        # Safe to use directly in LLM call
    """
    result = validate_conversation_history(history)
    if result.sanitized_history is None:
        return []
    return list(result.sanitized_history)
