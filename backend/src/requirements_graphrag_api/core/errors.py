"""LLM provider error classification — maps raw exceptions to safe user messages.

The CALLER is responsible for logging the real exception via logger.exception()
before calling classify_llm_error(). This module only produces the safe outbound text.
"""

from __future__ import annotations

_CAPACITY_MESSAGE = "The assistant is temporarily at capacity. Please try again shortly."
_CONFIG_MESSAGE = "The assistant is temporarily unavailable."
_UNKNOWN_MESSAGE = "Something went wrong generating a response."


def classify_llm_error(exc: BaseException) -> tuple[str, str]:
    """Map an LLM provider exception to a safe user message and a category string.

    Args:
        exc: The caught exception from an LLM provider call.

    Returns:
        ``(user_safe_message, category)`` where category is one of
        ``"capacity"``, ``"config"``, or ``"unknown"``.
    """
    try:
        import openai

        if isinstance(exc, openai.RateLimitError):
            return _CAPACITY_MESSAGE, "capacity"

        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return _CONFIG_MESSAGE, "config"

    except ImportError:
        pass

    return _UNKNOWN_MESSAGE, "unknown"
