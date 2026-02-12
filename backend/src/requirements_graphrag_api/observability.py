"""LangSmith observability and tracing utilities.

Provides centralized configuration for LangSmith tracing, including:
- Environment setup for automatic LangChain/LangGraph tracing
- Traceable decorator re-exports for consistent usage
- Input sanitization to prevent sensitive data from being logged
- OpenAI client wrapping for tracing non-LangChain calls

Usage:
    from requirements_graphrag_api.observability import configure_tracing, traceable_safe

    # Configure at startup
    configure_tracing(config)

    # Decorate functions for tracing (with automatic input sanitization)
    @traceable_safe(name="my_function", run_type="chain")
    async def my_function(config: AppConfig, query: str) -> str:
        ...
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, is_dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, Final

from langsmith import traceable

if TYPE_CHECKING:
    from collections.abc import Callable

    from requirements_graphrag_api.config import AppConfig

logger = logging.getLogger(__name__)

# Patterns for sensitive field names (case-insensitive)
SENSITIVE_FIELD_PATTERNS: Final[tuple[str, ...]] = (
    r".*password.*",
    r".*secret.*",
    r".*api_key.*",
    r".*apikey.*",
    r".*token.*",
    r".*credential.*",
    r".*auth.*",
)

# Compiled regex for performance
_SENSITIVE_PATTERN = re.compile(
    "|".join(SENSITIVE_FIELD_PATTERNS),
    re.IGNORECASE,
)

# Placeholder for redacted values
REDACTED: Final[str] = "[REDACTED]"


def _is_sensitive_key(key: str) -> bool:
    """Check if a key name indicates sensitive data.

    Args:
        key: The field/key name to check.

    Returns:
        True if the key matches a sensitive pattern.
    """
    return bool(_SENSITIVE_PATTERN.match(key))


def sanitize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Recursively sanitize inputs by redacting sensitive values.

    This function traverses dictionaries and dataclasses to find and
    redact any fields that match sensitive patterns (passwords, API keys, etc.).

    Only primitive values (strings, numbers) are redacted. Nested structures
    (dicts, lists, dataclasses) are recursively traversed even if their key
    matches a sensitive pattern, to ensure we don't lose structural information.

    Args:
        inputs: Dictionary of function inputs to sanitize.

    Returns:
        A new dictionary with sensitive values replaced by [REDACTED].
    """
    if not isinstance(inputs, dict):
        return inputs

    sanitized: dict[str, Any] = {}

    for key, value in inputs.items():
        if is_dataclass(value) and not isinstance(value, type):
            # Convert dataclass to dict and sanitize recursively
            sanitized[key] = sanitize_inputs(asdict(value))
        elif isinstance(value, dict):
            # Recurse into nested dicts
            sanitized[key] = sanitize_inputs(value)
        elif isinstance(value, list):
            # Handle lists (may contain dicts or dataclasses)
            sanitized[key] = [
                sanitize_inputs(item) if isinstance(item, dict) else item for item in value
            ]
        elif hasattr(value, "__class__") and value.__class__.__name__ in (
            "Neo4jDriver",
            "VectorRetriever",
            "Driver",
            "ChatOpenAI",
            "AsyncOpenAI",
        ):
            # Replace driver/retriever objects with type info only
            sanitized[key] = f"<{value.__class__.__name__}>"
        elif _is_sensitive_key(key):
            # Redact sensitive primitive values (strings, numbers, etc.)
            sanitized[key] = REDACTED
        else:
            sanitized[key] = value

    return sanitized


def traceable_safe(
    name: str | None = None,
    run_type: str = "chain",
    **kwargs: Any,
) -> Callable:
    """Decorator for LangSmith tracing with automatic input sanitization.

    This is a wrapper around langsmith.traceable that automatically
    sanitizes inputs to prevent sensitive data (API keys, passwords, etc.)
    from being logged to LangSmith.

    Args:
        name: Name for the trace (defaults to function name).
        run_type: Type of run (chain, llm, retriever, tool, etc.).
        **kwargs: Additional arguments passed to langsmith.traceable.

    Returns:
        Decorated function with sanitized tracing.

    Example:
        @traceable_safe(name="stream_chat", run_type="chain")
        async def stream_chat(config: AppConfig, message: str) -> str:
            ...
    """

    def decorator(func: Callable) -> Callable:
        # LangSmith's @traceable detects a `config` parameter and treats it
        # as a LangChain RunnableConfig (calls .get() on it, or strips it).
        # This breaks when `config` is a dataclass (ToxicityConfig, etc.).
        #
        # Fix: stash the config value in a ContextVar before entering
        # LangSmith, then retrieve and re-inject it in the inner function.
        import asyncio
        import contextvars
        import inspect

        sig = inspect.signature(func)
        config_param = sig.parameters.get("config")
        has_config_param = config_param is not None

        if has_config_param:
            config_idx = list(sig.parameters.keys()).index("config")
            config_is_positional = config_param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )

            _sentinel = object()
            _cfg_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
                f"_traceable_cfg_{func.__module__}.{func.__qualname__}",
                default=_sentinel,
            )
            _cfg_was_positional: contextvars.ContextVar[bool] = contextvars.ContextVar(
                f"_traceable_cfg_pos_{func.__module__}.{func.__qualname__}",
                default=False,
            )

            if asyncio.iscoroutinefunction(func):

                async def _inner(*args: Any, **kw: Any) -> Any:
                    cfg_val = _cfg_var.get(_sentinel)
                    if cfg_val is not _sentinel:
                        was_positional = _cfg_was_positional.get(False)
                        if was_positional and len(args) >= config_idx:
                            args = (*args[:config_idx], cfg_val, *args[config_idx:])
                        else:
                            kw["config"] = cfg_val
                    return await func(*args, **kw)
            else:

                def _inner(*args: Any, **kw: Any) -> Any:
                    cfg_val = _cfg_var.get(_sentinel)
                    if cfg_val is not _sentinel:
                        was_positional = _cfg_was_positional.get(False)
                        if was_positional and len(args) >= config_idx:
                            args = (*args[:config_idx], cfg_val, *args[config_idx:])
                        else:
                            kw["config"] = cfg_val
                    return func(*args, **kw)

            _inner.__name__ = func.__name__
            _inner.__qualname__ = func.__qualname__
            target = _inner
        else:
            target = func

        traced_func = traceable(
            name=name,
            run_type=run_type,
            process_inputs=sanitize_inputs,
            **kwargs,
        )(target)

        if has_config_param:

            @wraps(func)
            async def async_wrapper(*args: Any, **func_kwargs: Any) -> Any:
                cfg_token = None
                pos_token = None
                try:
                    if "config" in func_kwargs:
                        cfg_token = _cfg_var.set(func_kwargs.pop("config"))
                        pos_token = _cfg_was_positional.set(False)
                    elif config_is_positional and len(args) > config_idx:
                        cfg_token = _cfg_var.set(args[config_idx])
                        pos_token = _cfg_was_positional.set(True)
                        args = args[:config_idx] + args[config_idx + 1 :]
                    return await traced_func(*args, **func_kwargs)
                finally:
                    if cfg_token is not None:
                        _cfg_var.reset(cfg_token)
                    if pos_token is not None:
                        _cfg_was_positional.reset(pos_token)

            @wraps(func)
            def sync_wrapper(*args: Any, **func_kwargs: Any) -> Any:
                cfg_token = None
                pos_token = None
                try:
                    if "config" in func_kwargs:
                        cfg_token = _cfg_var.set(func_kwargs.pop("config"))
                        pos_token = _cfg_was_positional.set(False)
                    elif config_is_positional and len(args) > config_idx:
                        cfg_token = _cfg_var.set(args[config_idx])
                        pos_token = _cfg_was_positional.set(True)
                        args = args[:config_idx] + args[config_idx + 1 :]
                    return traced_func(*args, **func_kwargs)
                finally:
                    if cfg_token is not None:
                        _cfg_var.reset(cfg_token)
                    if pos_token is not None:
                        _cfg_was_positional.reset(pos_token)
        else:

            @wraps(func)
            async def async_wrapper(*args: Any, **func_kwargs: Any) -> Any:
                return await traced_func(*args, **func_kwargs)

            @wraps(func)
            def sync_wrapper(*args: Any, **func_kwargs: Any) -> Any:
                return traced_func(*args, **func_kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def create_thread_metadata(conversation_id: str | None) -> dict[str, Any] | None:
    """Create LangSmith metadata for conversation threading.

    This helper creates the metadata dict needed to group traces into
    LangSmith Threads. Use this with the `langsmith_extra` parameter
    when calling @traceable decorated functions.

    Note: LangSmith requires the key to be `thread_id` (not `conversation_id`)
    for runs to be grouped into Threads in the LangSmith UI.

    Args:
        conversation_id: Unique ID for the conversation thread.

    Returns:
        Metadata dict with thread_id, or None if no ID provided.

    Example:
        @traceable_safe(name="stream_chat", run_type="chain")
        async def stream_chat(..., langsmith_extra=None):
            ...

        # At call time:
        await stream_chat(
            ...,
            langsmith_extra=create_thread_metadata("conversation-123")
        )
    """
    if not conversation_id:
        return None
    return {"metadata": {"thread_id": conversation_id}}


def configure_sentry() -> bool:
    """Configure Sentry error tracking and performance monitoring.

    Initializes Sentry SDK with FastAPI, LangChain, and LangGraph integrations.
    No-op if SENTRY_DSN environment variable is not set.

    Call this at module level before the FastAPI app is created.

    Returns:
        True if Sentry was initialized, False otherwise.
    """
    dsn = os.getenv("SENTRY_DSN", "")
    if not dsn:
        logger.info("Sentry disabled (SENTRY_DSN not set)")
        return False

    import sentry_sdk

    sentry_sdk.init(
        dsn=dsn,
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0")),
        profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.0")),
        environment=os.getenv("SENTRY_ENVIRONMENT", "development"),
        send_default_pii=False,
    )
    env = os.getenv("SENTRY_ENVIRONMENT", "development")
    logger.info("Sentry initialized (environment=%s)", env)
    return True


def configure_tracing(config: AppConfig) -> bool:
    """Configure LangSmith tracing from application config.

    Sets up environment variables required for automatic tracing of
    LangChain and LangGraph operations. Call this at application startup.

    Args:
        config: Application configuration with LangSmith settings.

    Returns:
        True if tracing was enabled, False otherwise.
    """
    if not config.langsmith_tracing_enabled:
        logger.info("LangSmith tracing disabled")
        return False

    if not config.langsmith_api_key:
        logger.warning(
            "LangSmith tracing enabled but LANGSMITH_API_KEY not set. "
            "Tracing will not work without an API key."
        )
        return False

    # Set environment variables for LangChain/LangGraph auto-tracing
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Legacy compatibility
    os.environ["LANGSMITH_API_KEY"] = config.langsmith_api_key
    os.environ["LANGCHAIN_API_KEY"] = config.langsmith_api_key  # Legacy
    os.environ["LANGSMITH_PROJECT"] = config.langsmith_project
    os.environ["LANGCHAIN_PROJECT"] = config.langsmith_project  # Legacy

    # Set workspace ID for org-scoped API keys (X-Tenant-ID header)
    if config.langsmith_workspace_id:
        os.environ["LANGSMITH_WORKSPACE_ID"] = config.langsmith_workspace_id
        logger.info(
            "LangSmith tracing enabled for project: %s (workspace: %s)",
            config.langsmith_project,
            config.langsmith_workspace_id[:8] + "...",  # Truncate for logging
        )
    else:
        logger.info(
            "LangSmith tracing enabled for project: %s",
            config.langsmith_project,
        )
    return True


def disable_tracing() -> None:
    """Disable LangSmith tracing.

    Useful for testing or when tracing should be temporarily disabled.
    """
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    logger.info("LangSmith tracing disabled")


def get_tracing_status() -> dict[str, str | bool]:
    """Get current tracing configuration status.

    Returns:
        Dictionary with tracing status information.
    """
    return {
        "tracing_enabled": os.getenv("LANGSMITH_TRACING", "false").lower() == "true",
        "project": os.getenv("LANGSMITH_PROJECT", "default"),
        "api_key_set": bool(os.getenv("LANGSMITH_API_KEY")),
        "workspace_id_set": bool(os.getenv("LANGSMITH_WORKSPACE_ID")),
    }


__all__ = [
    "REDACTED",
    "SENSITIVE_FIELD_PATTERNS",
    "configure_sentry",
    "configure_tracing",
    "create_thread_metadata",
    "disable_tracing",
    "get_tracing_status",
    "sanitize_inputs",
    "traceable",  # Re-export for backward compatibility
    "traceable_safe",
]
