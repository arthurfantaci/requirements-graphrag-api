"""PostgresSaver configuration for conversation persistence.

This module configures the LangGraph checkpoint system:
1. AsyncPostgresSaver setup with connection pooling
2. Thread/conversation ID management
3. Checkpoint configuration helpers

Database Schema:
    LangGraph manages its own checkpoint tables via checkpointer.setup().
    We configure connection pooling and provide thread_id from the API layer.

Usage:
    from requirements_graphrag_api.core.agentic.checkpoints import (
        create_async_checkpointer,
        get_thread_config,
    )

    async with create_async_checkpointer(database_url) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        config = get_thread_config(thread_id)
        result = await graph.ainvoke(state, config=config)

Connection Pooling:
    Uses psycopg connection pooling internally via AsyncPostgresSaver.
    Pool is created once at startup and shared across requests.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CHECKPOINT_DATABASE_URL = os.getenv("CHECKPOINT_DATABASE_URL")


async def create_async_checkpointer(
    database_url: str | None = None,
    *,
    setup: bool = True,
) -> AsyncPostgresSaver:
    """Create an async PostgreSQL checkpointer.

    This creates an AsyncPostgresSaver configured with the provided database URL.
    The checkpointer should be used as an async context manager or manually closed.

    Args:
        database_url: PostgreSQL connection URL. If None, uses CHECKPOINT_DATABASE_URL
                     environment variable.
        setup: Whether to run setup() to create checkpoint tables. Default True.

    Returns:
        Configured AsyncPostgresSaver instance.

    Raises:
        ValueError: If no database URL is provided or configured.

    Example:
        async with await create_async_checkpointer() as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)
    """
    url = database_url or DEFAULT_CHECKPOINT_DATABASE_URL
    if not url:
        msg = (
            "No database URL provided. Set CHECKPOINT_DATABASE_URL environment "
            "variable or pass database_url parameter."
        )
        raise ValueError(msg)

    logger.info("Creating async checkpointer with PostgreSQL")
    checkpointer = AsyncPostgresSaver.from_conn_string(url)

    if setup:
        await checkpointer.setup()
        logger.info("Checkpoint tables created/verified")

    return checkpointer


@asynccontextmanager
async def async_checkpointer_context(
    database_url: str | None = None,
) -> AsyncGenerator[AsyncPostgresSaver, None]:
    """Context manager for async PostgreSQL checkpointer.

    Handles setup and cleanup of the checkpointer connection.

    Args:
        database_url: PostgreSQL connection URL. If None, uses CHECKPOINT_DATABASE_URL.

    Yields:
        Configured AsyncPostgresSaver instance.

    Example:
        async with async_checkpointer_context() as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)
            result = await graph.ainvoke(state, config=get_thread_config("thread-1"))
    """
    url = database_url or DEFAULT_CHECKPOINT_DATABASE_URL
    if not url:
        msg = "No database URL provided for checkpointer"
        raise ValueError(msg)

    async with AsyncPostgresSaver.from_conn_string(url) as checkpointer:
        await checkpointer.setup()
        logger.info("Async checkpointer initialized")
        yield checkpointer
        logger.info("Async checkpointer closed")


def get_thread_config(
    thread_id: str,
    *,
    checkpoint_id: str | None = None,
    user_id: str | None = None,
    run_id: str | None = None,
    **extra_configurable: Any,
) -> RunnableConfig:
    """Create a RunnableConfig with thread configuration for checkpointing.

    This helper creates the config dict needed by LangGraph for checkpoint
    persistence. The thread_id is required for persistence to work.

    Args:
        thread_id: Unique identifier for the conversation thread.
        checkpoint_id: Optional specific checkpoint to resume from.
        user_id: Optional user identifier for memory namespacing.
        run_id: Optional LangSmith run ID for tracing correlation.
        **extra_configurable: Additional configurable parameters.

    Returns:
        RunnableConfig suitable for graph.ainvoke() or graph.astream().

    Example:
        config = get_thread_config("conversation-123")
        result = await graph.ainvoke(state, config=config)

        # Resume from specific checkpoint
        config = get_thread_config("conv-123", checkpoint_id="cp-456")

        # With user namespacing
        config = get_thread_config("conv-123", user_id="user-789")
    """
    configurable: dict[str, Any] = {
        "thread_id": thread_id,
        **extra_configurable,
    }

    if checkpoint_id:
        configurable["checkpoint_id"] = checkpoint_id

    if user_id:
        configurable["user_id"] = user_id

    config: RunnableConfig = {"configurable": configurable}

    if run_id:
        config["run_id"] = run_id

    return config


async def get_conversation_history_from_checkpoint(
    checkpointer: AsyncPostgresSaver,
    thread_id: str,
) -> list[dict[str, str]]:
    """Read conversation history from a checkpoint for a given thread.

    We use aget_tuple() (low-level API) because this function is called
    before the graph is instantiated. graph.aget_state() would require
    creating a compiled graph just to read state.

    Args:
        checkpointer: Async PostgreSQL checkpointer.
        thread_id: Thread ID to look up.

    Returns:
        List of {"role": "user"|"assistant", "content": "..."} dicts.
        Returns empty list on any error (graceful fallback).
    """
    try:
        config = get_thread_config(thread_id)
        tuple_result = await checkpointer.aget_tuple(config)

        # aget_tuple() returns None when no checkpoint exists
        if tuple_result is None:
            return []

        checkpoint = tuple_result.checkpoint
        messages = checkpoint.get("channel_values", {}).get("messages", [])

        history: list[dict[str, str]] = []
        for msg in messages:
            # Handle both LangChain message objects and raw dicts
            if hasattr(msg, "content") and hasattr(msg, "type"):
                role = "user" if msg.type == "human" else "assistant"
                history.append({"role": role, "content": msg.content})
            elif isinstance(msg, dict):
                role = msg.get("role", msg.get("type", "user"))
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                history.append({"role": role, "content": msg.get("content", "")})

        return history

    except Exception:
        logger.warning("Failed to read checkpoint for thread %s", thread_id, exc_info=True)
        return []


def get_thread_id_from_config(config: RunnableConfig) -> str | None:
    """Extract thread_id from a RunnableConfig.

    Args:
        config: RunnableConfig that may contain thread configuration.

    Returns:
        The thread_id if present, None otherwise.
    """
    configurable = config.get("configurable", {})
    return configurable.get("thread_id")


__all__ = [
    "async_checkpointer_context",
    "create_async_checkpointer",
    "get_conversation_history_from_checkpoint",
    "get_thread_config",
    "get_thread_id_from_config",
]
