"""LangSmith observability and tracing utilities.

Provides centralized configuration for LangSmith tracing, including:
- Environment setup for automatic LangChain/LangGraph tracing
- Traceable decorator re-exports for consistent usage
- OpenAI client wrapping for tracing non-LangChain calls

Usage:
    from jama_mcp_server_graphrag.observability import configure_tracing, traceable

    # Configure at startup
    configure_tracing(config)

    # Decorate functions for tracing
    @traceable(name="my_function", run_type="chain")
    async def my_function(query: str) -> str:
        ...
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

# Re-export traceable for convenience
from langsmith import traceable  # noqa: F401

if TYPE_CHECKING:
    from jama_mcp_server_graphrag.config import AppConfig

logger = logging.getLogger(__name__)


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
