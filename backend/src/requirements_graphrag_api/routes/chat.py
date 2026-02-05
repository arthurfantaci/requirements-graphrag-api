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
from langchain_core.messages import AIMessage, HumanMessage
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
    TopicClassification,
    check_prompt_injection,
    check_topic_relevance,
    check_toxicity,
    detect_and_redact_pii,
    log_guardrail_event,
    metrics,
    validate_conversation_history,
)
from requirements_graphrag_api.guardrails.events import (
    create_injection_event,
    create_pii_event,
    create_topic_event,
    create_toxicity_event,
)
from requirements_graphrag_api.middleware.rate_limit import CHAT_RATE_LIMIT, get_rate_limiter
from requirements_graphrag_api.observability import create_thread_metadata
from requirements_graphrag_api.prompts import PromptName, get_prompt_sync

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

            if injection_result.should_warn or injection_result.should_block:
                metrics.record_prompt_injection(blocked=injection_result.should_block)

            if injection_result.should_block:
                metrics.record_request(blocked=True)
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
                metrics.record_pii(redacted=True)
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
                metrics.record_toxicity(blocked=True)
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
                metrics.record_request(blocked=True)
                yield f"event: {StreamEventType.ERROR.value}\n"
                err = {"error": "Request blocked by content safety filter"}
                yield f"data: {json.dumps(err)}\n\n"
                return
            if toxicity_result.should_warn:
                metrics.record_toxicity(blocked=False)
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

        # 4. Topic guard (keyword + LLM classification for borderline)
        if guardrail_config.topic_guard_enabled:
            from langchain_openai import ChatOpenAI

            from requirements_graphrag_api.guardrails.topic_guard import TopicGuardConfig

            topic_config = TopicGuardConfig(
                enabled=True,
                use_llm_classification=guardrail_config.topic_guard_use_llm,
                allow_borderline=guardrail_config.topic_guard_allow_borderline,
            )
            topic_llm = None
            if guardrail_config.topic_guard_use_llm and config.openai_api_key:
                topic_llm = ChatOpenAI(
                    model=config.chat_model,
                    temperature=0,
                    api_key=config.openai_api_key,
                    max_tokens=20,
                )
            topic_result = await check_topic_relevance(
                safe_message,
                llm=topic_llm,
                config=topic_config,
            )
            if topic_result.classification == TopicClassification.OUT_OF_SCOPE:
                metrics.record_topic_out_of_scope()
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
                    topic_result.suggested_response
                    or "This question is outside my area of expertise."
                )
                yield f"event: {StreamEventType.ROUTING.value}\n"
                yield f"data: {json.dumps({'intent': 'explanatory'})}\n\n"
                yield f"event: {StreamEventType.TOKEN.value}\n"
                yield f"data: {json.dumps({'token': redirect})}\n\n"
                yield f"event: {StreamEventType.DONE.value}\n"
                yield f"data: {json.dumps({'full_answer': redirect, 'source_count': 0})}\n\n"
                return

        # 5. Validate conversation history
        if request.conversation_history:
            history_dicts = [
                {"role": msg.role, "content": msg.content} for msg in request.conversation_history
            ]
            validation = validate_conversation_history(history_dicts)
            if validation.issues:
                metrics.record_conversation_validation_issue()
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

        # Record successful guardrail pass
        metrics.record_request(blocked=False)

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


async def _run_output_guardrails(
    full_response: str,
    original_query: str,
    retrieved_sources: list[dict],
    config: AppConfig,
    guardrail_config: GuardrailConfig,
) -> AsyncIterator[str]:
    """Run output guardrails on accumulated response and yield warning events.

    Checks the complete LLM response for:
    1. Output safety (toxicity, confidence scoring, disclaimers)
    2. Hallucination detection (factual grounding against sources)

    Args:
        full_response: The complete accumulated LLM response.
        original_query: The original user query.
        retrieved_sources: Sources retrieved during RAG.
        config: Application configuration.
        guardrail_config: Guardrail configuration.

    Yields:
        SSE guardrail_warning events if issues are detected.
    """
    warnings: list[str] = []

    try:
        # 1. Output filter (toxicity + confidence scoring)
        if guardrail_config.output_filter_enabled:
            from requirements_graphrag_api.guardrails import (
                OutputFilterConfig,
                filter_output,
            )
            from requirements_graphrag_api.guardrails.events import (
                create_output_filter_event,
            )

            output_config = OutputFilterConfig(
                enabled=True,
                confidence_threshold=guardrail_config.output_filter_confidence_threshold,
            )
            filter_result = await filter_output(
                full_response,
                original_query,
                retrieved_sources=retrieved_sources or None,
                config=output_config,
            )
            if not filter_result.is_safe:
                warnings.append(filter_result.filtered_content)
                event = create_output_filter_event(
                    request_id="output",
                    is_safe=False,
                    confidence_score=filter_result.confidence_score,
                    warnings=filter_result.warnings,
                    modifications=filter_result.modifications,
                    blocked_reason=filter_result.blocked_reason,
                )
                log_guardrail_event(event)
            elif filter_result.should_add_disclaimer:
                warnings.append(
                    filter_result.filtered_content.replace(
                        filter_result.original_content, ""
                    ).strip()
                )
                event = create_output_filter_event(
                    request_id="output",
                    is_safe=True,
                    confidence_score=filter_result.confidence_score,
                    warnings=filter_result.warnings,
                    modifications=filter_result.modifications,
                )
                log_guardrail_event(event)

        # 2. Hallucination check (grounding against sources)
        if guardrail_config.hallucination_enabled and retrieved_sources:
            from langchain_openai import ChatOpenAI

            from requirements_graphrag_api.guardrails import (
                check_hallucination,
            )

            llm = ChatOpenAI(
                model=config.chat_model,
                temperature=0.0,
                api_key=config.openai_api_key,
            )
            hal_result = await check_hallucination(
                response=full_response,
                sources=retrieved_sources,
                llm=llm,
            )
            if hal_result.should_add_warning:
                metrics.record_hallucination_warning()
                from requirements_graphrag_api.guardrails import (
                    HALLUCINATION_WARNING_SHORT,
                )

                warnings.append(HALLUCINATION_WARNING_SHORT.strip())
                logger.info(
                    "Hallucination check: level=%s, unsupported=%d",
                    hal_result.grounding_level.value,
                    len(hal_result.unsupported_claims),
                )

    except Exception:
        logger.exception("Output guardrail check failed")

    # Emit warning events
    for warning in warnings:
        yield f"event: {StreamEventType.GUARDRAIL_WARNING.value}\n"
        yield f"data: {json.dumps({'warning': warning})}\n\n"


async def _generate_explanatory_events(
    config: AppConfig,
    retriever: VectorRetriever,
    driver: Driver,
    request: ChatRequest,
    safe_message: str,
    guardrail_config: GuardrailConfig | None = None,
) -> AsyncIterator[str]:
    """Generate SSE events for explanatory (RAG) queries using agentic orchestrator.

    Uses the LangGraph-based agentic system that composes:
    - RAG subgraph: Query expansion + parallel retrieval + deduplication
    - Research subgraph: Entity identification + conditional exploration
    - Synthesis subgraph: Draft + critique + revision loop

    After streaming completes, output guardrails (output_filter, hallucination)
    run on the accumulated response and emit warning events if needed.

    Args:
        config: Application configuration.
        retriever: VectorRetriever for semantic search.
        driver: Neo4j driver for graph queries.
        request: Chat request with message and options.
        safe_message: Sanitized message with PII redacted.
        guardrail_config: Guardrail configuration for output checks.

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

        # Refine query for multi-turn context (resolve pronouns, add context)
        refined_query = safe_message
        if request.conversation_history:
            try:
                from langchain_core.output_parsers import StrOutputParser
                from langchain_openai import ChatOpenAI

                previous_answers = "\n".join(
                    f"Q: {msg.content}" if msg.role == "user" else f"A: {msg.content}"
                    for msg in request.conversation_history
                )
                updater_template = get_prompt_sync(PromptName.QUERY_UPDATER)
                llm = ChatOpenAI(
                    model=config.chat_model,
                    temperature=0.1,
                    api_key=config.openai_api_key,
                )
                chain = updater_template | llm | StrOutputParser()
                refined_query = await chain.ainvoke(
                    {"previous_answers": previous_answers, "question": safe_message}
                )
                if refined_query and refined_query.strip():
                    refined_query = refined_query.strip()
                    logger.info(
                        "Query refined for multi-turn: '%s' -> '%s'", safe_message, refined_query
                    )
                else:
                    refined_query = safe_message
            except Exception:
                logger.exception("Query refinement failed, using original query")
                refined_query = safe_message

        # Build conversation history as LangChain messages
        messages: list[HumanMessage | AIMessage] = []
        if request.conversation_history:
            for msg in request.conversation_history:
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    messages.append(AIMessage(content=msg.content))

        # Add the current (potentially refined) query as the final message
        messages.append(HumanMessage(content=refined_query))

        # Build initial state
        initial_state: OrchestratorState = {
            "messages": messages,
            "query": refined_query,
        }

        # Get thread configuration for persistence (uses conversation_id as thread_id)
        thread_id = request.conversation_id or str(uuid.uuid4())
        runnable_config = get_thread_config(thread_id)

        # Stream events from the agentic orchestrator, accumulating for output guardrails
        accumulated_tokens: list[str] = []
        retrieved_sources: list[dict] = []
        last_event_type: str | None = None

        async for sse_event in stream_agentic_events(
            graph,
            initial_state,
            runnable_config,
            app_config=config,
        ):
            # Track event data for output guardrails
            if sse_event.startswith("event: "):
                last_event_type = sse_event.strip().removeprefix("event: ")
            elif sse_event.startswith("data: ") and last_event_type:
                try:
                    data = json.loads(sse_event[6:].strip())
                    if last_event_type == StreamEventType.TOKEN.value and "token" in data:
                        accumulated_tokens.append(data["token"])
                    elif last_event_type == StreamEventType.SOURCES.value and "sources" in data:
                        retrieved_sources = data["sources"]
                except (json.JSONDecodeError, TypeError):
                    pass
            yield sse_event

        # === OUTPUT GUARDRAILS (post-stream) ===
        full_response = "".join(accumulated_tokens)
        if full_response and guardrail_config:
            async for warning_event in _run_output_guardrails(
                full_response,
                refined_query,
                retrieved_sources,
                config,
                guardrail_config,
            ):
                yield warning_event


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
