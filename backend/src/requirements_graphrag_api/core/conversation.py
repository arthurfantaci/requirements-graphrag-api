"""Conversational intent handler for meta-conversation queries.

Handles queries about the conversation itself (recall, summarize, repeat)
using conversation history from the frontend or checkpoint fallback.

This module intentionally bypasses the LangGraph StateGraph because:
- Conversational queries are lightweight recall tasks
- They don't produce state worth checkpointing
- Running through the full graph would add ~100-200ms overhead with no benefit
- The frontend conversation_history is the primary data source
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from langchain_openai import ChatOpenAI
from langsmith import get_current_run_tree

from requirements_graphrag_api.core.definitions import StreamEventType
from requirements_graphrag_api.observability import traceable_safe
from requirements_graphrag_api.prompts import PromptName, get_prompt_sync

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langchain_core.messages import BaseMessage

    from requirements_graphrag_api.config import AppConfig

logger = logging.getLogger(__name__)

# Graceful response for empty history
_EMPTY_HISTORY_RESPONSE = (
    "I don't have any previous conversation to reference. "
    "This appears to be the start of our conversation. "
    "Feel free to ask me about requirements management topics!"
)


def _format_conversation_history(history: list[dict[str, str]]) -> str:
    r"""Format conversation history into a numbered string for the prompt.

    Args:
        history: List of {"role": "user"|"assistant", "content": "..."} dicts.

    Returns:
        Formatted string like "1. User: ...\n2. Assistant: ..."
    """
    if not history:
        return "(No conversation history available)"

    lines = []
    for i, msg in enumerate(history, 1):
        role = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{i}. {role}: {msg.get('content', '')}")
    return "\n".join(lines)


@traceable_safe(name="conversational_llm_stream", run_type="chain")
async def _run_conversational_llm(
    config: AppConfig,
    messages: list[BaseMessage],
    *,
    langsmith_extra: dict[str, Any] | None = None,
) -> tuple[Any, str | None]:
    """Run the conversational LLM and return the stream + run_id.

    Extracted as a regular async function (not generator) so that
    @traceable_safe can wrap it correctly — asyncio.iscoroutinefunction()
    returns False for async generators, breaking the ContextVar stash/restore.

    Args:
        config: Application configuration.
        messages: Formatted prompt messages for the LLM.
        langsmith_extra: Optional LangSmith metadata for thread grouping.

    Returns:
        Tuple of (async_stream, run_id_or_none).
    """
    llm = ChatOpenAI(
        model=config.conversational_model,
        temperature=0.1,
        api_key=config.openai_api_key,
    )

    # Propagate LangSmith callbacks for thread grouping visibility
    callbacks = (langsmith_extra or {}).get("callbacks", [])
    stream = llm.astream(messages, config={"callbacks": callbacks} if callbacks else {})

    # Capture run_id while inside the traceable span
    run_id = None
    try:
        run_tree = get_current_run_tree()
        if run_tree:
            run_id = str(run_tree.id)
    except Exception:
        logger.debug("Could not get run_id for conversational LLM span")

    return stream, run_id


async def stream_conversational_events(
    config: AppConfig,
    question: str,
    conversation_history: list[dict[str, str]],
    *,
    langsmith_extra: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    """Stream SSE events for a conversational query using true token streaming.

    Uses llm.astream() for real-time token delivery (~60-180ms TTFT)
    instead of ainvoke() + simulated chunking (~210-530ms).

    Args:
        config: Application configuration.
        question: The user's meta-conversation question.
        conversation_history: Previous messages as list of role/content dicts.
        langsmith_extra: Optional LangSmith metadata for thread grouping.

    Yields:
        Formatted SSE event strings (token events + done event).
    """
    # Handle empty history without LLM call
    if not conversation_history:
        yield f"event: {StreamEventType.TOKEN.value}\n"
        yield f"data: {json.dumps({'token': _EMPTY_HISTORY_RESPONSE})}\n\n"

        run_id = None
        try:
            run_tree = get_current_run_tree()
            if run_tree:
                run_id = str(run_tree.id)
        except Exception:
            logger.debug("Could not get run_id for empty history streaming")

        yield f"event: {StreamEventType.DONE.value}\n"
        done_data: dict[str, Any] = {
            "full_answer": _EMPTY_HISTORY_RESPONSE,
            "source_count": 0,
        }
        if run_id:
            done_data["run_id"] = run_id
        yield f"data: {json.dumps(done_data)}\n\n"
        return

    formatted_history = _format_conversation_history(conversation_history)
    prompt_template = get_prompt_sync(PromptName.CONVERSATIONAL)

    # Build prompt messages for astream
    messages = prompt_template.format_messages(history=formatted_history, question=question)

    # Get stream + run_id from traceable inner function
    stream, run_id = await _run_conversational_llm(
        config, messages, langsmith_extra=langsmith_extra
    )

    # True token streaming via llm.astream()
    accumulated: list[str] = []
    try:
        async for chunk in stream:
            token = chunk.content if hasattr(chunk, "content") else ""
            if token:
                accumulated.append(token)
                yield f"event: {StreamEventType.TOKEN.value}\n"
                yield f"data: {json.dumps({'token': token})}\n\n"
    except Exception:
        logger.exception("Error during conversational streaming")
        yield f"event: {StreamEventType.ERROR.value}\n"
        yield f"data: {json.dumps({'error': 'Conversational response failed'})}\n\n"
        return

    full_response = "".join(accumulated)

    yield f"event: {StreamEventType.DONE.value}\n"
    done_data = {
        "full_answer": full_response,
        "source_count": 0,
    }
    if run_id:
        done_data["run_id"] = run_id
    yield f"data: {json.dumps(done_data)}\n\n"


__all__ = [
    "stream_conversational_events",
]
