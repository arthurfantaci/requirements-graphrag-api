"""SSE streaming utilities for the agentic RAG system.

This module provides streaming support for real-time response delivery:
- AgenticEventType: Enum of event types for agentic streaming
- AgenticEvent: Dataclass for structured event data
- stream_agentic_events: Async generator yielding SSE-formatted events

Event Types (backward-compatible with existing frontend):
- routing: Query routing/classification result
- sources: Retrieved context sources
- token: Individual token from LLM response
- done: Stream completion with final answer
- error: Error event with details

Agentic-specific events:
- phase: Current execution phase (rag, research, synthesis)
- progress: Subgraph progress update
- entities: Explored entity information

SSE Format:
    data: {json_data}

The frontend detects event type by payload shape (e.g., data.token !== undefined),
so we maintain the same JSON structure for backward compatibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from langchain_core.runnables import RunnableConfig

    from requirements_graphrag_api.config import AppConfig
    from requirements_graphrag_api.core.agentic.orchestrator import StateGraph
    from requirements_graphrag_api.core.agentic.state import OrchestratorState

logger = logging.getLogger(__name__)


class AgenticEventType(StrEnum):
    """Event types for agentic streaming.

    Maintains backward compatibility with existing frontend event detection.
    The frontend uses payload shape detection, not event type field.
    """

    # Backward-compatible events (frontend expects these payload shapes)
    ROUTING = "routing"  # {"intent": "..."}
    SOURCES = "sources"  # {"sources": [...], "entities": [...]}
    TOKEN = "token"  # noqa: S105 - {"token": "..."}
    DONE = "done"  # {"full_answer": "...", "run_id": "..."}
    ERROR = "error"  # {"error": "..."}

    # Agentic-specific events
    PHASE = "phase"  # {"phase": "rag|research|synthesis", "message": "..."}
    PROGRESS = "progress"  # {"step": "...", "detail": "..."}
    ENTITIES = "entities"  # {"entities": [...]}


@dataclass(slots=True)
class AgenticEvent:
    """Structured event for SSE streaming.

    Attributes:
        event_type: The type of event.
        data: Event payload data.
        metadata: Optional metadata (not sent to frontend).
    """

    event_type: AgenticEventType
    data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        r"""Format as SSE data line.

        Returns:
            SSE-formatted string: 'data: {json}\n\n'
        """
        return f"data: {json.dumps(self.data)}\n\n"


def create_routing_event(intent: str) -> AgenticEvent:
    """Create a routing event indicating query intent."""
    return AgenticEvent(
        event_type=AgenticEventType.ROUTING,
        data={"intent": intent},
    )


def create_sources_event(
    sources: list[dict[str, Any]],
    entities: list[dict[str, Any]] | None = None,
    resources: dict[str, Any] | None = None,
) -> AgenticEvent:
    """Create a sources event with retrieved context."""
    data: dict[str, Any] = {"sources": sources}
    if entities:
        data["entities"] = entities
    if resources:
        data["resources"] = resources
    return AgenticEvent(
        event_type=AgenticEventType.SOURCES,
        data=data,
    )


def create_token_event(token: str) -> AgenticEvent:
    """Create a token event for streaming LLM output."""
    return AgenticEvent(
        event_type=AgenticEventType.TOKEN,
        data={"token": token},
    )


def create_done_event(
    full_answer: str,
    source_count: int = 0,
    run_id: str | None = None,
) -> AgenticEvent:
    """Create a done event with final answer."""
    data: dict[str, Any] = {
        "full_answer": full_answer,
        "source_count": source_count,
    }
    if run_id:
        data["run_id"] = run_id
    return AgenticEvent(
        event_type=AgenticEventType.DONE,
        data=data,
    )


def create_error_event(error: str) -> AgenticEvent:
    """Create an error event."""
    return AgenticEvent(
        event_type=AgenticEventType.ERROR,
        data={"error": error},
    )


def create_phase_event(phase: str, message: str | None = None) -> AgenticEvent:
    """Create a phase event indicating current execution stage."""
    data: dict[str, Any] = {"phase": phase}
    if message:
        data["message"] = message
    return AgenticEvent(
        event_type=AgenticEventType.PHASE,
        data=data,
    )


def create_progress_event(step: str, detail: str | None = None) -> AgenticEvent:
    """Create a progress event for subgraph updates."""
    data: dict[str, Any] = {"step": step}
    if detail:
        data["detail"] = detail
    return AgenticEvent(
        event_type=AgenticEventType.PROGRESS,
        data=data,
    )


def create_entities_event(entities: list[dict[str, Any]]) -> AgenticEvent:
    """Create an entities event with explored entity information."""
    return AgenticEvent(
        event_type=AgenticEventType.ENTITIES,
        data={"entities": entities},
    )


async def stream_agentic_events(
    graph: StateGraph,
    initial_state: OrchestratorState,
    config: RunnableConfig,
    *,
    app_config: AppConfig | None = None,
) -> AsyncGenerator[str, None]:
    r"""Stream SSE events from agentic graph execution.

    This async generator yields SSE-formatted strings as the orchestrator
    processes queries through RAG, Research, and Synthesis subgraphs.

    Args:
        graph: Compiled orchestrator graph.
        initial_state: Initial state with query and messages.
        config: RunnableConfig with thread_id for persistence.
        app_config: Optional app config for additional settings.

    Yields:
        SSE-formatted strings (data: {json}\n\n).
    """
    run_id: str | None = None
    query = initial_state.get("query", "")

    logger.info("Starting agentic stream for query: %s", query[:50] if query else "N/A")

    try:
        # Note: Routing event is emitted by the caller (routes/chat.py)
        # based on the intent classification result

        # Emit initial phase event
        yield create_phase_event("rag", "Retrieving relevant context...").to_sse()

        # Stream graph execution
        last_phase = None
        sources_emitted = False
        source_count = 0
        final_answer = ""

        async for event in graph.astream_events(
            initial_state,
            config=config,
            version="v2",
        ):
            event_kind = event.get("event")
            event_name = event.get("name", "")
            event_data = event.get("data", {})

            # Capture run_id from the first event (P0-1 fix)
            if run_id is None and event.get("run_id"):
                run_id = event["run_id"]
                logger.debug("Captured run_id from astream_events: %s", run_id)

            # Track phase changes
            if event_kind == "on_chain_start":
                if "run_rag" in event_name and last_phase != "rag":
                    last_phase = "rag"
                    yield create_phase_event("rag", "Searching knowledge base...").to_sse()
                elif "run_research" in event_name and last_phase != "research":
                    last_phase = "research"
                    yield create_phase_event("research", "Exploring related entities...").to_sse()
                elif "run_synthesis" in event_name and last_phase != "synthesis":
                    last_phase = "synthesis"
                    yield create_phase_event("synthesis", "Generating answer...").to_sse()

            # Emit sources when RAG completes
            if event_kind == "on_chain_end" and "run_rag" in event_name:
                output = event_data.get("output", {})
                ranked_results = output.get("ranked_results", [])

                if ranked_results and not sources_emitted:
                    sources = []
                    entities: dict[str, dict[str, Any]] = {}
                    all_webinars: list[dict[str, Any]] = []
                    all_images: list[dict[str, Any]] = []
                    all_videos: list[dict[str, Any]] = []
                    seen_urls: set[str] = set()

                    for doc in ranked_results[:10]:
                        # Extract source info
                        if hasattr(doc, "source"):
                            content = doc.content
                            if len(content) > 200:
                                content = content[:200] + "..."
                            metadata = doc.metadata if hasattr(doc, "metadata") else {}
                            sources.append(
                                {
                                    "title": doc.source,
                                    "content": content,
                                    "url": metadata.get("url", ""),
                                    "chunk_id": metadata.get("chunk_id"),
                                    "relevance_score": doc.score,
                                }
                            )

                            # Extract entities
                            for entity in metadata.get("entities", []):
                                if isinstance(entity, dict) and entity.get("name"):
                                    name = entity["name"]
                                    if name not in entities:
                                        entities[name] = {
                                            "definition": entity.get("definition"),
                                            "label": entity.get("type", "Entity"),
                                        }

                            # Extract glossary definitions as entities
                            for defn in metadata.get("glossary_definitions", []):
                                if isinstance(defn, dict) and defn.get("term"):
                                    term = defn["term"]
                                    if term not in entities:
                                        entities[term] = {
                                            "definition": defn.get("definition"),
                                            "label": "Definition",
                                        }

                            # Extract media (webinars, images, videos)
                            media = metadata.get("media", {})
                            for webinar in media.get("webinars", []):
                                url = webinar.get("url")
                                if url and url not in seen_urls:
                                    seen_urls.add(url)
                                    all_webinars.append(
                                        {
                                            "title": webinar.get("title", "Webinar"),
                                            "url": url,
                                            "thumbnail_url": webinar.get("thumbnail_url"),
                                        }
                                    )

                            for img in media.get("images", []):
                                url = img.get("url")
                                if url and url not in seen_urls:
                                    seen_urls.add(url)
                                    all_images.append(
                                        {
                                            "title": img.get("alt_text", "Image"),
                                            "url": url,
                                            "alt_text": img.get("alt_text"),
                                        }
                                    )

                            for video in media.get("videos", []):
                                url = video.get("url")
                                if url and url not in seen_urls:
                                    seen_urls.add(url)
                                    all_videos.append(
                                        {
                                            "title": video.get("title", "Video"),
                                            "url": url,
                                        }
                                    )

                        elif isinstance(doc, dict):
                            content = doc.get("content", "")[:200]
                            sources.append(
                                {
                                    "title": doc.get("source", "Unknown"),
                                    "content": content,
                                    "url": doc.get("url", ""),
                                    "relevance_score": doc.get("score", 0),
                                }
                            )

                    if sources:
                        # Build entities list for frontend
                        entities_list = [
                            {
                                "name": name,
                                "definition": info.get("definition"),
                                "label": info.get("label"),
                            }
                            for name, info in entities.items()
                        ]

                        # Build resources dict for frontend
                        resources = {}
                        if all_webinars:
                            resources["webinars"] = all_webinars[:5]
                        if all_images:
                            resources["images"] = all_images[:5]
                        if all_videos:
                            resources["videos"] = all_videos[:5]

                        yield create_sources_event(
                            sources,
                            entities=entities_list if entities_list else None,
                            resources=resources if resources else None,
                        ).to_sse()
                        source_count = len(sources)
                    sources_emitted = True

            # Emit entities when research completes
            if event_kind == "on_chain_end" and "run_research" in event_name:
                output = event_data.get("output", {})
                entity_contexts = output.get("entity_contexts", [])

                if entity_contexts:
                    entities = []
                    for entity in entity_contexts:
                        if hasattr(entity, "name"):
                            entities.append(
                                {
                                    "name": entity.name,
                                    "type": entity.entity_type,
                                    "description": entity.description,
                                }
                            )
                        elif isinstance(entity, dict):
                            entities.append(
                                {
                                    "name": entity.get("name", ""),
                                    "type": entity.get("entity_type", ""),
                                    "description": entity.get("description", ""),
                                }
                            )
                    yield create_entities_event(entities).to_sse()

            # Capture fallback answer when quality gate fails
            if event_kind == "on_chain_end" and "format_fallback" in event_name:
                output = event_data.get("output", {})
                final_answer = output.get("final_answer", "")
                if not sources_emitted:
                    yield create_sources_event([]).to_sse()
                    sources_emitted = True

            # Note: We don't stream raw LLM tokens during synthesis because the
            # SYNTHESIS prompt outputs JSON that needs parsing. Instead, we capture
            # the parsed final_answer and stream it after synthesis completes.

            # Capture final answer when synthesis completes
            if event_kind == "on_chain_end" and "run_synthesis" in event_name:
                output = event_data.get("output", {})
                final_answer = output.get("final_answer", "")

        # Stream the parsed answer in chunks for smooth visual effect
        if final_answer:
            # Larger chunks + shorter delay: ~200ms vs ~3.75s for 2000-char answer
            chunk_size = 50
            for i in range(0, len(final_answer), chunk_size):
                chunk = final_answer[i : i + chunk_size]
                yield create_token_event(chunk).to_sse()
                await asyncio.sleep(0.005)  # 5ms between chunks

        # Emit done event (source_count tracked when sources were emitted)
        yield create_done_event(
            full_answer=final_answer,
            source_count=source_count,
            run_id=run_id,
        ).to_sse()

        logger.info("Agentic stream completed successfully")

    except Exception as e:
        logger.exception("Agentic stream error")
        yield create_error_event(str(e)).to_sse()


__all__ = [
    "AgenticEvent",
    "AgenticEventType",
    "create_done_event",
    "create_entities_event",
    "create_error_event",
    "create_phase_event",
    "create_progress_event",
    "create_routing_event",
    "create_sources_event",
    "create_token_event",
    "stream_agentic_events",
]
