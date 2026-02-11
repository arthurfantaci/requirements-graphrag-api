"""Chat endpoint with SSE streaming and automatic query routing.

Request models, input guardrail orchestration, intent routing, and
the /chat endpoint live here. Response handlers (explanatory, structured,
conversational) are in routes/handlers.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
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
)
from requirements_graphrag_api.guardrails import (
    InjectionRisk,
    TopicClassification,
    check_prompt_injection,
    check_topic_relevance,
    check_toxicity,
    detect_and_redact_pii,
    log_guardrail_event,
    validate_conversation_history,
)
from requirements_graphrag_api.guardrails.events import (
    create_injection_event,
    create_pii_event,
    create_topic_event,
    create_toxicity_event,
)
from requirements_graphrag_api.middleware.rate_limit import CHAT_RATE_LIMIT, get_rate_limiter
from requirements_graphrag_api.routes.handlers import (
    generate_conversational_events,
    generate_explanatory_events,
    generate_structured_events,
)

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
    force_intent: QueryIntent | None = Field(
        default=None,
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

            if pii_result.check_failed:
                logger.warning("PII detection failed — processing request with unchecked input")

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

        # 3. Toxicity check (fast keyword check on input)
        if guardrail_config.toxicity_enabled:
            toxicity_result = await check_toxicity(
                safe_message,
                use_full_check=guardrail_config.toxicity_use_full_check,
            )
            if toxicity_result.should_block:
                event = create_toxicity_event(
                    request_id=request_id,
                    categories=tuple(c.value for c in toxicity_result.categories),
                    confidence=toxicity_result.confidence,
                    blocked=True,
                    check_type=toxicity_result.check_type,
                    user_ip=user_ip,
                    input_text=request.message,
                )
                log_guardrail_event(event)
                yield f"event: {StreamEventType.ERROR.value}\n"
                err = {"error": "Request blocked by content safety filter"}
                yield f"data: {json.dumps(err)}\n\n"
                return
            if toxicity_result.should_warn:
                event = create_toxicity_event(
                    request_id=request_id,
                    categories=tuple(c.value for c in toxicity_result.categories),
                    confidence=toxicity_result.confidence,
                    blocked=False,
                    check_type=toxicity_result.check_type,
                    user_ip=user_ip,
                    input_text=request.message,
                )
                log_guardrail_event(event)

        # 4. Topic guard + intent classification (parallel when both need LLM)
        # Run topic guard and intent classification concurrently to reduce TTFT.
        # If topic guard blocks (OUT_OF_SCOPE), the intent result is discarded.

        # 4a. Validate conversation history (doesn't need LLM, run inline)
        if request.conversation_history:
            history_dicts = [
                {"role": msg.role, "content": msg.content} for msg in request.conversation_history
            ]
            validation = validate_conversation_history(history_dicts)
            if validation.issues:
                logger.info(
                    "Conversation history validation: %d issues: %s",
                    len(validation.issues),
                    validation.issues,
                )
            if validation.sanitized_history is not None:
                # Replace with sanitized history
                request = request.model_copy(
                    update={
                        "conversation_history": [
                            ChatMessage(role=m["role"], content=m["content"])
                            for m in validation.sanitized_history
                        ]
                    }
                )
            elif not validation.is_valid:
                # All messages were removed (e.g., all contained injection)
                request = request.model_copy(update={"conversation_history": None})

        # 4b. Build topic guard coroutine (if enabled)
        topic_guard_coro = None
        if guardrail_config.topic_guard_enabled:
            from requirements_graphrag_api.guardrails.topic_guard import TopicGuardConfig

            topic_config = TopicGuardConfig(
                enabled=True,
                allow_borderline=guardrail_config.topic_guard_allow_borderline,
            )
            topic_guard_coro = check_topic_relevance(
                safe_message,
                config=topic_config,
            )

        # 4c. Determine intent resolution strategy
        topic_result = None
        intent: QueryIntent

        if request.options.force_intent:
            intent = request.options.force_intent
            logger.info("Using forced intent: %s", intent)
            # Still need to run topic guard if enabled
            if topic_guard_coro:
                topic_result = await topic_guard_coro
        elif request.options.auto_route:
            # Both topic guard and intent may need LLM — run concurrently
            intent_coro = classify_intent(config, safe_message)
            if topic_guard_coro:
                topic_result, intent = await asyncio.gather(topic_guard_coro, intent_coro)
                logger.info("Parallel topic guard + intent classification complete")
            else:
                intent = await intent_coro
            logger.info("Auto-classified intent: %s", intent)
        else:
            intent = QueryIntent.EXPLANATORY
            logger.info("Auto-route disabled, using default: %s", intent)
            if topic_guard_coro:
                topic_result = await topic_guard_coro

        # 4e. Conditional topic guard bypass for conversational intent
        # Cross-check: if query also contains out-of-scope topics, don't bypass.
        # Uses word-boundary regex (not substring) to avoid false positives
        # like "stock" matching "stockpile" or "news" matching "newness".
        if intent == QueryIntent.CONVERSATIONAL and topic_result:
            from requirements_graphrag_api.guardrails.topic_guard import OUT_OF_SCOPE_TOPICS

            has_out_of_scope = any(
                re.search(rf"\b{re.escape(topic)}\b", safe_message, re.IGNORECASE)
                for topic in OUT_OF_SCOPE_TOPICS
            )
            if not has_out_of_scope:
                topic_result = None  # Safe to bypass — pure meta-conversation

        # 4f. Check topic guard result (may short-circuit)
        if topic_result and topic_result.classification == TopicClassification.OUT_OF_SCOPE:
            event = create_topic_event(
                request_id=request_id,
                classification=topic_result.classification.value,
                confidence=topic_result.confidence,
                reasoning=topic_result.reasoning,
                check_type=topic_result.check_type,
                user_ip=user_ip,
                input_text=request.message,
            )
            log_guardrail_event(event)
            # Return a polite redirect instead of processing the query
            redirect = (
                topic_result.suggested_response or "This question is outside my area of expertise."
            )
            yield f"event: {StreamEventType.ROUTING.value}\n"
            yield f"data: {json.dumps({'intent': 'explanatory'})}\n\n"
            yield f"event: {StreamEventType.TOKEN.value}\n"
            yield f"data: {json.dumps({'token': redirect})}\n\n"
            yield f"event: {StreamEventType.DONE.value}\n"
            yield f"data: {json.dumps({'full_answer': redirect, 'source_count': 0})}\n\n"
            return

        # Record successful guardrail pass

        # Emit routing event so frontend knows which handler is being used
        yield f"event: {StreamEventType.ROUTING.value}\n"
        yield f"data: {json.dumps({'intent': intent.value})}\n\n"

        if intent == QueryIntent.CONVERSATIONAL:
            # Use lightweight conversation handler for meta-conversation queries
            async for event_str in generate_conversational_events(
                config, request, safe_message, guardrail_config
            ):
                yield event_str
        elif intent == QueryIntent.STRUCTURED:
            # Use Text2Cypher for structured queries
            async for event_str in generate_structured_events(
                config, driver, request, safe_message
            ):
                yield event_str
        else:
            # Use RAG for explanatory queries
            async for event_str in generate_explanatory_events(
                config,
                retriever,
                driver,
                request,
                safe_message,
                guardrail_config=guardrail_config,
            ):
                yield event_str

    except Exception as e:
        logger.exception("Error in SSE generation")
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

    **SSE Event Sequence (Conversational - Recall):**
    1. `routing` - Intent classification result (`{"intent": "conversational"}`)
    2. `token` - Individual tokens as they're generated (true streaming)
    3. `done` - Complete answer with `run_id` for feedback

    **Routing Tips:**
    - Use "list all", "show me all", "how many" for structured queries
    - Use "what is", "how do I", "explain" for explanatory queries
    - Use "what was my first question", "summarize our conversation" for recall
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
