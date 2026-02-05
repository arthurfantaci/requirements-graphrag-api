"""Chat endpoint with SSE streaming and automatic query routing.

Updated Data Model (2026-01):
- Uses neo4j Driver directly instead of LangChain Neo4jGraph
- Uses VectorRetriever instead of Neo4jVector

Streaming Support (2026-01):
- Endpoint now returns Server-Sent Events (SSE) for real-time token streaming
- Enables LangSmith TTFT (Time to First Token) metrics

Automatic Routing (2026-01):
- Queries are automatically classified as EXPLANATORY or STRUCTURED
- EXPLANATORY queries use Agentic RAG with LangGraph orchestrator
- STRUCTURED queries use Text2Cypher for direct graph queries

Agentic RAG (2026-02):
- Replaced routed RAG with full LangGraph-based agentic system
- Orchestrator composes RAG, Research, and Synthesis subgraphs
- Supports conversation persistence via thread_id
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from slowapi.util import get_remote_address

from requirements_graphrag_api.core import (
    QueryIntent,
    StreamEventType,
    classify_intent,
    get_routing_guide,
    text2cypher_query,
)
from requirements_graphrag_api.core.agentic import (
    OrchestratorState,
    async_checkpointer_context,
    create_orchestrator_graph,
    get_thread_config,
    stream_agentic_events,
)
from requirements_graphrag_api.guardrails import (
    InjectionRisk,
    check_prompt_injection,
    detect_and_redact_pii,
    log_guardrail_event,
)
from requirements_graphrag_api.guardrails.events import (
    create_injection_event,
    create_pii_event,
)
from requirements_graphrag_api.middleware.rate_limit import CHAT_RATE_LIMIT, get_rate_limiter
from requirements_graphrag_api.observability import create_thread_metadata

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from neo4j import Driver
    from neo4j_graphrag.retrievers import VectorRetriever

    from requirements_graphrag_api.config import AppConfig, GuardrailConfig

logger = logging.getLogger(__name__)

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
    auto_route: bool = Field(
        default=True,
        description="Automatically route queries based on intent classification",
    )
    force_intent: str | None = Field(
        default=None,
        pattern="^(explanatory|structured)$",
        description="Force a specific intent (overrides auto_route)",
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
    guardrail_config: GuardrailConfig,
    user_ip: str | None = None,
    request_id: str | None = None,
) -> AsyncIterator[str]:
    """Generate SSE events from streaming chat response with automatic routing.

    Args:
        config: Application configuration.
        retriever: VectorRetriever for semantic search.
        driver: Neo4j driver for graph queries.
        request: Chat request with message and options.
        guardrail_config: Guardrail configuration.
        user_ip: Client IP address for logging.
        request_id: Unique request identifier.

    Yields:
        Formatted SSE event strings.
    """
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]

    try:
        # === INPUT GUARDRAILS ===
        safe_message = request.message

        # 1. Check for prompt injection
        if guardrail_config.prompt_injection_enabled:
            injection_result = check_prompt_injection(
                request.message,
                block_threshold=InjectionRisk(guardrail_config.injection_block_threshold),
            )

            if injection_result.should_warn or injection_result.should_block:
                event = create_injection_event(
                    request_id=request_id,
                    risk_level=injection_result.risk_level.value,
                    patterns=injection_result.detected_patterns,
                    blocked=injection_result.should_block,
                    user_ip=user_ip,
                    input_text=request.message,
                )
                log_guardrail_event(event)

            if injection_result.should_block:
                yield f"event: {StreamEventType.ERROR.value}\n"
                yield f"data: {json.dumps({'error': 'Request blocked by safety filter'})}\n\n"
                return

        # 2. Detect and redact PII
        if guardrail_config.pii_detection_enabled:
            pii_result = detect_and_redact_pii(
                safe_message,
                entities=guardrail_config.pii_entities,
                score_threshold=guardrail_config.pii_score_threshold,
                anonymize_type=guardrail_config.pii_anonymize_type,
            )

            if pii_result.contains_pii:
                entity_types = tuple(e.entity_type for e in pii_result.detected_entities)
                event = create_pii_event(
                    request_id=request_id,
                    entity_types=entity_types,
                    entity_count=pii_result.entity_count,
                    redacted=True,
                    user_ip=user_ip,
                    input_text=request.message,
                )
                log_guardrail_event(event)

                # Use the anonymized text for processing
                safe_message = pii_result.anonymized_text
                logger.info(
                    "PII detected and redacted: %d entities, types=%s",
                    pii_result.entity_count,
                    entity_types,
                )

        # === QUERY PROCESSING ===

        # Determine query intent (use safe_message for classification)
        intent: QueryIntent
        if request.options.force_intent:
            intent = QueryIntent(request.options.force_intent)
            logger.info("Using forced intent: %s", intent)
        elif request.options.auto_route:
            intent = await classify_intent(config, safe_message)
            logger.info("Auto-classified intent: %s", intent)
        else:
            # Default to explanatory if auto_route is disabled
            intent = QueryIntent.EXPLANATORY
            logger.info("Auto-route disabled, using default: %s", intent)

        # Emit routing event so frontend knows which handler is being used
        yield f"event: {StreamEventType.ROUTING.value}\n"
        yield f"data: {json.dumps({'intent': intent.value})}\n\n"

        if intent == QueryIntent.STRUCTURED:
            # Use Text2Cypher for structured queries
            async for event_str in _generate_structured_events(
                config, driver, request, safe_message
            ):
                yield event_str
        else:
            # Use RAG for explanatory queries
            async for event_str in _generate_explanatory_events(
                config, retriever, driver, request, safe_message
            ):
                yield event_str

    except Exception as e:
        logger.exception("Error in SSE generation")
        yield f"event: {StreamEventType.ERROR.value}\n"
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


async def _generate_explanatory_events(
    config: AppConfig,
    retriever: VectorRetriever,
    driver: Driver,
    request: ChatRequest,
    safe_message: str,
) -> AsyncIterator[str]:
    """Generate SSE events for explanatory (RAG) queries using agentic orchestrator.

    Uses the LangGraph-based agentic system that composes:
    - RAG subgraph: Query expansion + parallel retrieval + deduplication
    - Research subgraph: Entity identification + conditional exploration
    - Synthesis subgraph: Draft + critique + revision loop

    Args:
        config: Application configuration.
        retriever: VectorRetriever for semantic search.
        driver: Neo4j driver for graph queries.
        request: Chat request with message and options.
        safe_message: Sanitized message with PII redacted.

    Yields:
        Formatted SSE event strings.
    """
    # Create checkpointer for conversation persistence (graceful fallback)
    async with AsyncExitStack() as stack:
        checkpointer = None
        if os.getenv("CHECKPOINT_DATABASE_URL"):
            checkpointer = await stack.enter_async_context(async_checkpointer_context())
        else:
            logger.warning("CHECKPOINT_DATABASE_URL not set — chat memory disabled")

        # Create the orchestrator graph with persistence
        graph = create_orchestrator_graph(config, driver, retriever, checkpointer=checkpointer)

        # Build conversation history as LangChain messages
        messages: list[HumanMessage] = []
        if request.conversation_history:
            for msg in request.conversation_history:
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))

        # Add the current query as the final message
        messages.append(HumanMessage(content=safe_message))

        # Build initial state
        initial_state: OrchestratorState = {
            "messages": messages,
            "query": safe_message,
        }

        # Get thread configuration for persistence (uses conversation_id as thread_id)
        thread_id = request.conversation_id or str(uuid.uuid4())
        runnable_config = get_thread_config(thread_id)

        # Stream events from the agentic orchestrator
        async for sse_event in stream_agentic_events(
            graph,
            initial_state,
            runnable_config,
            app_config=config,
        ):
            yield sse_event


async def _generate_structured_events(
    config: AppConfig,
    driver: Driver,
    request: ChatRequest,
    safe_message: str,
) -> AsyncIterator[str]:
    """Generate SSE events for structured (Text2Cypher) queries.

    Args:
        config: Application configuration.
        driver: Neo4j driver for graph queries.
        request: Chat request with message and options.
        safe_message: Sanitized message with PII redacted.

    Yields:
        Formatted SSE event strings.
    """
    try:
        # Create LangSmith thread metadata for conversation grouping
        thread_metadata = create_thread_metadata(request.conversation_id)

        # Generate and execute Cypher query
        result = await text2cypher_query(
            config,
            driver,
            safe_message,
            execute=True,
            langsmith_extra=thread_metadata,
        )

        # Emit Cypher query event
        yield f"event: {StreamEventType.CYPHER.value}\n"
        yield f"data: {json.dumps({'query': result.get('cypher', '')})}\n\n"

        # Emit results event
        yield f"event: {StreamEventType.RESULTS.value}\n"
        results_data = {
            "results": result.get("results", []),
            "row_count": result.get("row_count", 0),
        }
        yield f"data: {json.dumps(results_data)}\n\n"

        # Get run_id from result (captured inside text2cypher_query for correct context)
        run_id = result.get("run_id")

        # Emit done event
        if "error" in result:
            yield f"event: {StreamEventType.ERROR.value}\n"
            yield f"data: {json.dumps({'error': result['error']})}\n\n"
        else:
            yield f"event: {StreamEventType.DONE.value}\n"
            done_data = {
                "query": result.get("cypher", ""),
                "row_count": result.get("row_count", 0),
                "run_id": run_id,
            }
            yield f"data: {json.dumps(done_data)}\n\n"

    except Exception as e:
        logger.exception("Error in structured query")
        yield f"event: {StreamEventType.ERROR.value}\n"
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# Get rate limiter instance for decorator
limiter = get_rate_limiter()


@router.post("/chat")
@limiter.limit(CHAT_RATE_LIMIT)
async def chat_endpoint(
    request: Request,
    body: ChatRequest,
) -> StreamingResponse:
    """Stream chat response via Server-Sent Events with automatic routing.

    This endpoint automatically routes queries based on intent classification:
    - **EXPLANATORY** queries use RAG with hybrid search and graph enrichment
    - **STRUCTURED** queries use Text2Cypher for direct graph queries

    **Security Guardrails:**
    - Rate limiting: 20 requests/minute per IP or API key
    - Prompt injection detection: Blocks malicious instruction manipulation
    - PII detection: Automatically redacts personal information

    **SSE Event Sequence (Explanatory - RAG):**
    1. `routing` - Intent classification result
    2. `sources` - Retrieved context, entities, and resources
    3. `token` - Individual tokens as they're generated (multiple events)
    4. `done` - Complete answer with source count

    **SSE Event Sequence (Structured - Cypher):**
    1. `routing` - Intent classification result
    2. `cypher` - Generated Cypher query
    3. `results` - Query execution results
    4. `done` - Completion with row count

    **Example Explanatory Response:**
    ```
    event: routing
    data: {"intent": "explanatory"}

    event: sources
    data: {"sources": [...], "entities": [...], "resources": {...}}

    event: token
    data: {"token": "Requirements"}

    event: done
    data: {"full_answer": "Requirements traceability is...", "source_count": 3}
    ```

    **Example Structured Response:**
    ```
    event: routing
    data: {"intent": "structured"}

    event: cypher
    data: {"query": "MATCH (w:Webinar) RETURN w.title, w.url"}

    event: results
    data: {"results": [...], "row_count": 5}

    event: done
    data: {"query": "...", "row_count": 5}
    ```

    **Routing Tips:**
    - Use "list all", "show me all", "how many" for structured queries
    - Use "what is", "how do I", "explain" for explanatory queries
    - Set `options.force_intent` to override automatic classification

    Args:
        request: FastAPI request object.
        body: Chat request body.

    Returns:
        StreamingResponse with SSE media type.
    """
    config: AppConfig = request.app.state.config
    retriever: VectorRetriever = request.app.state.retriever
    driver: Driver = request.app.state.driver
    guardrail_config: GuardrailConfig = request.app.state.guardrail_config

    # Get client info for logging
    user_ip = get_remote_address(request)
    request_id = str(uuid.uuid4())[:8]

    return StreamingResponse(
        _generate_sse_events(
            config,
            retriever,
            driver,
            body,
            guardrail_config,
            user_ip=user_ip,
            request_id=request_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "X-Request-ID": request_id,
        },
    )


@router.get("/chat/routing-guide")
async def routing_guide_endpoint() -> dict:
    """Get user-facing documentation for query routing.

    Returns guidance on how to phrase queries for optimal routing.
    This can be displayed in the frontend to help users understand
    how to get the best answers.

    Returns:
        Dictionary with routing guidance, examples, and tips.
    """
    return get_routing_guide()


# =============================================================================
# CONVERSATION MANAGEMENT ENDPOINTS (Phase 5.3)
# =============================================================================


class ConversationMessage(BaseModel):
    """A message in conversation state."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ConversationStateResponse(BaseModel):
    """Response model for conversation state."""

    thread_id: str = Field(..., description="Conversation thread ID")
    messages: list[ConversationMessage] = Field(
        default_factory=list, description="Conversation messages"
    )
    current_phase: str | None = Field(None, description="Current execution phase if in progress")
    last_query: str | None = Field(None, description="Last query processed")
    last_answer: str | None = Field(None, description="Last answer generated")
    message_count: int = Field(0, description="Total number of messages")


class ContinueChatRequest(BaseModel):
    """Request body for continuing a conversation."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Follow-up message to continue the conversation",
    )
    options: ChatOptions = Field(
        default_factory=ChatOptions,
        description="Chat options",
    )


@router.get("/chat/{thread_id}")
async def get_conversation_state(
    request: Request,
    thread_id: str,
) -> ConversationStateResponse:
    """Get the state of an existing conversation.

    Retrieves the conversation history and current state for a given thread ID.
    This is useful for resuming conversations or displaying conversation history.

    **Note:** Requires checkpoint persistence to be configured via
    `CHECKPOINT_DATABASE_URL` environment variable.

    Args:
        request: FastAPI request object.
        thread_id: The conversation thread ID.

    Returns:
        ConversationStateResponse with conversation details.

    Raises:
        HTTPException: 404 if conversation not found, 503 if checkpointing unavailable.
    """
    from fastapi import HTTPException

    config: AppConfig = request.app.state.config
    driver: Driver = request.app.state.driver
    retriever: VectorRetriever = request.app.state.retriever

    # Check if checkpoint database URL is configured (from env only to avoid mock issues)
    checkpoint_url = os.getenv("CHECKPOINT_DATABASE_URL")

    if not checkpoint_url:
        raise HTTPException(
            status_code=503,
            detail="Checkpoint persistence is not configured. Set CHECKPOINT_DATABASE_URL.",
        )

    try:
        # Use context manager for proper cleanup
        async with async_checkpointer_context(checkpoint_url) as checkpointer:
            # Create the orchestrator graph with checkpointer
            graph = create_orchestrator_graph(config, driver, retriever, checkpointer=checkpointer)

            # Get the thread configuration
            thread_config = get_thread_config(thread_id)

            # Get the current state
            state = await graph.aget_state(thread_config)

            if state is None or state.values is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Conversation with thread_id '{thread_id}' not found",
                )

            # Extract messages from state
            messages: list[ConversationMessage] = []
            state_values = state.values

            if "messages" in state_values:
                for msg in state_values["messages"]:
                    if hasattr(msg, "content"):
                        role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
                        messages.append(ConversationMessage(role=role, content=msg.content))

            return ConversationStateResponse(
                thread_id=thread_id,
                messages=messages,
                current_phase=state_values.get("current_phase"),
                last_query=state_values.get("query"),
                last_answer=state_values.get("final_answer"),
                message_count=len(messages),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting conversation state")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving conversation state: {e}",
        ) from e


@router.post("/chat/{thread_id}/continue")
@limiter.limit(CHAT_RATE_LIMIT)
async def continue_conversation(
    request: Request,
    thread_id: str,
    body: ContinueChatRequest,
) -> StreamingResponse:
    """Continue an existing conversation with a new message.

    Resumes a conversation using the stored checkpoint state and adds
    a new message. This enables multi-turn conversations with memory.

    **Note:** Requires checkpoint persistence to be configured via
    `CHECKPOINT_DATABASE_URL` environment variable.

    **SSE Event Sequence:**
    Same as POST /chat endpoint.

    Args:
        request: FastAPI request object.
        thread_id: The conversation thread ID to continue.
        body: Request body with new message.

    Returns:
        StreamingResponse with SSE events.
    """
    config: AppConfig = request.app.state.config
    retriever: VectorRetriever = request.app.state.retriever
    driver: Driver = request.app.state.driver
    guardrail_config: GuardrailConfig = request.app.state.guardrail_config

    # Get client info for logging
    user_ip = get_remote_address(request)
    request_id = str(uuid.uuid4())[:8]

    # Create a ChatRequest with the thread_id as conversation_id
    chat_request = ChatRequest(
        message=body.message,
        conversation_id=thread_id,
        options=body.options,
    )

    return StreamingResponse(
        _generate_sse_events(
            config,
            retriever,
            driver,
            chat_request,
            guardrail_config,
            user_ip=user_ip,
            request_id=request_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
            "X-Thread-ID": thread_id,
        },
    )
