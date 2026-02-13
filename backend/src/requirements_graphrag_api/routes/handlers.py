"""Chat response handlers for different query intents.

Each handler is an async generator that yields SSE event strings.
Extracted from routes/chat.py to keep the endpoint module focused
on request parsing, guardrail orchestration, and routing dispatch.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage

from requirements_graphrag_api.core import (
    StreamEventType,
    text2cypher_query,
)
from requirements_graphrag_api.core.agentic import (
    OrchestratorState,
    async_checkpointer_context,
    create_orchestrator_graph,
    get_conversation_history_from_checkpoint,
    get_thread_config,
    stream_agentic_events,
)
from requirements_graphrag_api.core.conversation import stream_conversational_events
from requirements_graphrag_api.evaluation.cost_analysis import get_global_cost_tracker
from requirements_graphrag_api.guardrails import log_guardrail_event
from requirements_graphrag_api.observability import create_thread_metadata
from requirements_graphrag_api.prompts import PromptName, get_prompt

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from neo4j import Driver
    from neo4j_graphrag.retrievers import VectorRetriever

    from requirements_graphrag_api.config import AppConfig, GuardrailConfig
    from requirements_graphrag_api.routes.chat import ChatMessage, ChatRequest

logger = logging.getLogger(__name__)


async def run_output_guardrails(
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
            from requirements_graphrag_api.guardrails import (
                check_hallucination,
            )

            hal_result = await check_hallucination(
                response=full_response,
                sources=retrieved_sources,
            )
            if hal_result.should_add_warning:
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


async def generate_conversational_events(
    config: AppConfig,
    request: ChatRequest,
    safe_message: str,
    guardrail_config: GuardrailConfig,
    *,
    trace_id: str | None = None,
) -> AsyncIterator[str]:
    """Generate SSE events for conversational (meta-conversation) queries.

    Uses conversation_history from the request as primary data source.
    Falls back to checkpoint read if history is empty but conversation_id
    exists and CHECKPOINT_DATABASE_URL is configured.

    Runs output_filter (toxicity only, no disclaimers) on the response.

    Args:
        config: Application configuration.
        request: Chat request with message, conversation_history, and options.
        safe_message: Sanitized message with PII redacted.
        guardrail_config: Guardrail configuration for output checks.
        trace_id: OTel trace ID for cross-system correlation.

    Yields:
        Formatted SSE event strings.
    """
    # Build conversation history from request (primary source)
    history: list[dict[str, str]] = []
    if request.conversation_history:
        history = [
            {"role": msg.role, "content": msg.content} for msg in request.conversation_history
        ]

    # Checkpoint fallback: if history is empty but we have a conversation_id
    if not history and request.conversation_id and os.getenv("CHECKPOINT_DATABASE_URL"):
        try:
            async with async_checkpointer_context() as checkpointer:
                history = await get_conversation_history_from_checkpoint(
                    checkpointer, request.conversation_id
                )
                if history:
                    logger.info(
                        "Loaded %d messages from checkpoint for thread %s",
                        len(history),
                        request.conversation_id,
                    )
        except Exception:
            logger.warning("Checkpoint fallback failed", exc_info=True)

    # Create LangSmith metadata for thread grouping + intent tracking
    langsmith_extra = create_thread_metadata(request.conversation_id) or {}
    langsmith_extra.setdefault("metadata", {})["intent"] = "conversational"
    if trace_id:
        langsmith_extra["metadata"]["otel_trace_id"] = trace_id

    # Emit empty sources event so frontend SSE parser proceeds to token rendering
    empty_sources = {
        "sources": [],
        "entities": [],
        "resources": {"images": [], "webinars": [], "videos": []},
    }
    yield f"event: {StreamEventType.SOURCES.value}\n"
    yield f"data: {json.dumps(empty_sources)}\n\n"

    # Stream tokens from the conversational handler and accumulate for output filter
    accumulated_tokens: list[str] = []
    async for sse_event in stream_conversational_events(
        config, safe_message, history, langsmith_extra=langsmith_extra, trace_id=trace_id
    ):
        # Track accumulated tokens for output guardrails
        if sse_event.startswith("data: ") and '"token"' in sse_event:
            try:
                data = json.loads(sse_event[6:].strip())
                if "token" in data:
                    accumulated_tokens.append(data["token"])
            except (json.JSONDecodeError, TypeError):
                pass
        yield sse_event

    # Run toxicity-only output filter (no disclaimers — sourceless responses
    # get penalized -0.3 confidence, producing misleading disclaimers)
    full_response = "".join(accumulated_tokens)
    if full_response and guardrail_config.output_filter_enabled:
        from requirements_graphrag_api.guardrails import (
            OutputFilterConfig,
            filter_output,
        )

        output_config = OutputFilterConfig(
            enabled=True,
            add_disclaimers=False,
            confidence_threshold=guardrail_config.output_filter_confidence_threshold,
        )
        filter_result = await filter_output(
            full_response,
            safe_message,
            retrieved_sources=None,
            config=output_config,
        )
        if not filter_result.is_safe:
            from requirements_graphrag_api.guardrails.events import (
                create_output_filter_event,
            )

            event = create_output_filter_event(
                request_id="output",
                is_safe=False,
                confidence_score=filter_result.confidence_score,
                warnings=filter_result.warnings,
                modifications=filter_result.modifications,
                blocked_reason=filter_result.blocked_reason,
            )
            log_guardrail_event(event)
            yield f"event: {StreamEventType.GUARDRAIL_WARNING.value}\n"
            yield f"data: {json.dumps({'warning': filter_result.filtered_content})}\n\n"


async def generate_explanatory_events(
    config: AppConfig,
    retriever: VectorRetriever,
    driver: Driver,
    request: ChatRequest,
    safe_message: str,
    guardrail_config: GuardrailConfig | None = None,
    *,
    trace_id: str | None = None,
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
        trace_id: OTel trace ID for cross-system correlation.

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
                from langchain_openai import ChatOpenAI

                previous_answers = "\n".join(
                    f"Q: {msg.content}" if msg.role == "user" else f"A: {msg.content}"
                    for msg in request.conversation_history
                )
                updater_template = await get_prompt(PromptName.QUERY_UPDATER)
                llm = ChatOpenAI(
                    model=config.conversational_model,
                    temperature=0.1,
                    api_key=config.openai_api_key,
                )
                chain = updater_template | llm
                response = await chain.ainvoke(
                    {"previous_answers": previous_answers, "question": safe_message}
                )
                get_global_cost_tracker().record_from_response(
                    config.conversational_model, response, operation="query_updater"
                )
                refined_query = response.content
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
        runnable_config.setdefault("metadata", {})["intent"] = "explanatory"
        if trace_id:
            runnable_config["metadata"]["otel_trace_id"] = trace_id

        # Stream events from the agentic orchestrator, accumulating for output guardrails
        accumulated_tokens: list[str] = []
        retrieved_sources: list[dict] = []
        last_event_type: str | None = None

        async for sse_event in stream_agentic_events(
            graph,
            initial_state,
            runnable_config,
            app_config=config,
            trace_id=trace_id,
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
            async for warning_event in run_output_guardrails(
                full_response,
                refined_query,
                retrieved_sources,
                config,
                guardrail_config,
            ):
                yield warning_event


async def resolve_coreferences(
    config: AppConfig,
    safe_message: str,
    conversation_history: list[ChatMessage] | None,
) -> str:
    """Resolve pronoun/reference expressions using conversation history.

    Uses the COREFERENCE_RESOLVER prompt with a fast model to substitute
    references like "those two industries" with their concrete antecedents
    from conversation history. Returns the original message unchanged if
    no history is provided, resolution fails, or times out.

    Args:
        config: Application configuration.
        safe_message: Sanitized user message.
        conversation_history: Previous messages for context.

    Returns:
        Resolved message with references substituted, or original on failure.
    """
    if not conversation_history:
        return safe_message

    try:
        async with asyncio.timeout(5.0):
            from langchain_openai import ChatOpenAI

            history_text = "\n".join(
                f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}"
                for msg in conversation_history
            )

            template = await get_prompt(PromptName.COREFERENCE_RESOLVER)
            llm = ChatOpenAI(
                model=config.conversational_model,
                temperature=0,
                api_key=config.openai_api_key,
            )
            chain = template | llm
            response = await chain.ainvoke({"history": history_text, "question": safe_message})
            get_global_cost_tracker().record_from_response(
                config.conversational_model, response, operation="coreference"
            )
            resolved = response.content

            if resolved and resolved.strip():
                resolved = resolved.strip()
                if resolved != safe_message:
                    logger.info("Coreference resolved: '%s' -> '%s'", safe_message, resolved)
                return resolved
            return safe_message
    except Exception:
        logger.exception("Coreference resolution failed, using original")
        return safe_message


async def generate_structured_events(
    config: AppConfig,
    driver: Driver,
    request: ChatRequest,
    safe_message: str,
    *,
    trace_id: str | None = None,
) -> AsyncIterator[str]:
    """Generate SSE events for structured (Text2Cypher) queries.

    Args:
        config: Application configuration.
        driver: Neo4j driver for graph queries.
        request: Chat request with message and options.
        safe_message: Sanitized message with PII redacted.
        trace_id: OTel trace ID for cross-system correlation.

    Yields:
        Formatted SSE event strings.
    """
    try:
        # Resolve coreferences in multi-turn conversations before Cypher generation
        refined_query = await resolve_coreferences(
            config, safe_message, request.conversation_history
        )

        # Create LangSmith thread metadata for conversation grouping
        thread_metadata = create_thread_metadata(request.conversation_id) or {}
        thread_metadata.setdefault("metadata", {})["intent"] = "structured"
        if trace_id:
            thread_metadata["metadata"]["otel_trace_id"] = trace_id

        # Generate and execute Cypher query
        result = await text2cypher_query(
            config,
            driver,
            refined_query,
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
            if trace_id:
                done_data["trace_id"] = trace_id
            if result.get("message"):
                done_data["message"] = result["message"]
            yield f"data: {json.dumps(done_data)}\n\n"

    except Exception as e:
        logger.exception("Error in structured query")
        yield f"event: {StreamEventType.ERROR.value}\n"
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
