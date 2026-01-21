"""Chat endpoint with SSE streaming for RAG-powered Q&A.

Updated Data Model (2026-01):
- Uses neo4j Driver directly instead of LangChain Neo4jGraph
- Uses VectorRetriever instead of Neo4jVector

Streaming Support (2026-01):
- Endpoint now returns Server-Sent Events (SSE) for real-time token streaming
- Enables LangSmith TTFT (Time to First Token) metrics
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from jama_mcp_server_graphrag.core import stream_chat

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from neo4j import Driver
    from neo4j_graphrag.retrievers import VectorRetriever

    from jama_mcp_server_graphrag.config import AppConfig

router = APIRouter()


class ChatOptions(BaseModel):
    """Options for chat request."""

    retrieval_strategy: str = Field(
        default="hybrid",
        description="Retrieval strategy: 'vector', 'hybrid', or 'graph'",
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include source citations",
    )
    max_sources: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of sources to cite",
    )


class ChatMessage(BaseModel):
    """A message in conversation history."""

    role: str = Field(
        ...,
        pattern="^(user|assistant)$",
        description="Role of the message sender (user or assistant)",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Content of the message",
    )


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User message to respond to",
    )
    conversation_history: list[ChatMessage] | None = Field(
        default=None,
        description="Previous messages for multi-turn conversation context",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Optional conversation ID for tracking",
    )
    options: ChatOptions = Field(
        default_factory=ChatOptions,
        description="Chat options",
    )


async def _generate_sse_events(
    config: AppConfig,
    retriever: VectorRetriever,
    driver: Driver,
    request: ChatRequest,
) -> AsyncIterator[str]:
    """Generate SSE events from streaming chat response.

    Args:
        config: Application configuration.
        retriever: VectorRetriever for semantic search.
        driver: Neo4j driver for graph queries.
        request: Chat request with message and options.

    Yields:
        Formatted SSE event strings.
    """
    # Convert ChatMessage models to dicts for stream_chat
    history: list[dict[str, str]] | None = None
    if request.conversation_history:
        history = [
            {"role": msg.role, "content": msg.content} for msg in request.conversation_history
        ]

    async for event in stream_chat(
        config,
        retriever,
        driver,
        request.message,
        conversation_history=history,
        max_sources=request.options.max_sources,
    ):
        # Format as SSE: event type and JSON data
        yield f"event: {event.event_type.value}\n"
        yield f"data: {json.dumps(event.data)}\n\n"


@router.post("/chat")
async def chat_endpoint(
    request: Request,
    body: ChatRequest,
) -> StreamingResponse:
    """Stream chat response via Server-Sent Events.

    This endpoint provides RAG-powered Q&A with real-time token streaming.
    It retrieves relevant content from the knowledge graph and streams
    the generated answer token by token.

    **SSE Event Sequence:**
    1. `sources` - Retrieved context, entities, and images
    2. `token` - Individual tokens as they're generated (multiple events)
    3. `done` - Complete answer with source count
    4. `error` - If an error occurs (replaces done)

    **Example Response Stream:**
    ```
    event: sources
    data: {"sources": [...], "entities": [...], "images": [...]}

    event: token
    data: {"token": "Requirements"}

    event: token
    data: {"token": " traceability"}

    event: done
    data: {"full_answer": "Requirements traceability is...", "source_count": 3}
    ```

    Args:
        request: FastAPI request object.
        body: Chat request body.

    Returns:
        StreamingResponse with SSE media type.
    """
    config: AppConfig = request.app.state.config
    retriever: VectorRetriever = request.app.state.retriever
    driver: Driver = request.app.state.driver

    return StreamingResponse(
        _generate_sse_events(config, retriever, driver, body),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
