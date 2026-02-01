"""Chat endpoint with SSE streaming and automatic query routing.

Updated Data Model (2026-01):
- Uses neo4j Driver directly instead of LangChain Neo4jGraph
- Uses VectorRetriever instead of Neo4jVector

Streaming Support (2026-01):
- Endpoint now returns Server-Sent Events (SSE) for real-time token streaming
- Enables LangSmith TTFT (Time to First Token) metrics

Automatic Routing (2026-01):
- Queries are automatically classified as EXPLANATORY or STRUCTURED
- EXPLANATORY queries use RAG with hybrid search and graph enrichment
- STRUCTURED queries use Text2Cypher for direct graph queries
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi.util import get_remote_address

from requirements_graphrag_api.core import (
    QueryIntent,
    StreamEventType,
    classify_intent,
    get_routing_guide,
    stream_chat,
    text2cypher_query,
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
    """Generate SSE events for explanatory (RAG) queries.

    Args:
        config: Application configuration.
        retriever: VectorRetriever for semantic search.
        driver: Neo4j driver for graph queries.
        request: Chat request with message and options.
        safe_message: Sanitized message with PII redacted.

    Yields:
        Formatted SSE event strings.
    """
    # Convert ChatMessage models to dicts for stream_chat
    history: list[dict[str, str]] | None = None
    if request.conversation_history:
        history = [
            {"role": msg.role, "content": msg.content} for msg in request.conversation_history
        ]

    # Create LangSmith thread metadata for conversation grouping
    thread_metadata = create_thread_metadata(request.conversation_id)

    async for event in stream_chat(
        config,
        retriever,
        driver,
        safe_message,
        conversation_history=history,
        conversation_id=request.conversation_id,
        max_sources=request.options.max_sources,
        langsmith_extra=thread_metadata,
    ):
        # Format as SSE: event type and JSON data
        yield f"event: {event.event_type.value}\n"
        yield f"data: {json.dumps(event.data)}\n\n"


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
