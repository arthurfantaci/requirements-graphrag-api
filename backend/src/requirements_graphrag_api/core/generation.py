"""Answer generation with RAG for conversational Q&A.

Combines retrieval results with LLM generation to produce
grounded answers with citations.

This module uses the centralized prompt catalog for prompt management,
enabling version control, A/B testing, and monitoring via LangSmith Hub.

Updated Data Model (2026-01):
- Uses VectorRetriever and Driver instead of Neo4jGraph/Neo4jVector

Streaming Support (2026-01):
- stream_chat() provides SSE streaming for REST API
- Enables LangSmith TTFT and token latency metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from requirements_graphrag_api.core.definitions import search_terms
from requirements_graphrag_api.core.retrieval import graph_enriched_search
from requirements_graphrag_api.observability import traceable_safe
from requirements_graphrag_api.prompts import PromptName, get_prompt_sync

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from neo4j import Driver
    from neo4j_graphrag.retrievers import VectorRetriever

    from requirements_graphrag_api.config import AppConfig

logger = logging.getLogger(__name__)

DEFINITION_RELEVANCE_THRESHOLD: Final[float] = 0.5


class StreamEventType(StrEnum):
    """Types of events emitted during streaming chat."""

    # Explanatory (RAG) events
    SOURCES = "sources"
    TOKEN = "token"  # noqa: S105 - not a password
    DONE = "done"
    ERROR = "error"

    # Structured (Cypher) events
    ROUTING = "routing"  # Intent classification result
    CYPHER = "cypher"  # Generated Cypher query
    RESULTS = "results"  # Query results


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """A single event in the streaming response.

    Attributes:
        event_type: The type of event (sources, token, done, error).
        data: The event payload as a dictionary.
    """

    event_type: StreamEventType
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Resource:
    """A single resource (webinar, video, or image).

    Attributes:
        title: Display title for the resource.
        url: URL to access the resource.
        alt_text: Alternative text (primarily for images).
        source_title: Title of the source this resource came from.
    """

    title: str
    url: str
    alt_text: str = ""
    source_title: str = ""


@dataclass(frozen=True, slots=True)
class ContextBuildResult:
    """Result of building context from retrieval results.

    Attributes:
        sources: List of source dictionaries with metadata.
        entities: List of entity names.
        context: Formatted context string for LLM.
        entities_str: Comma-separated entity string for LLM.
        resources: Dictionary mapping resource types to lists of Resource objects.
    """

    sources: list[dict[str, Any]]
    entities: list[str]
    context: str
    entities_str: str
    resources: dict[str, list[Resource]] = field(default_factory=dict)


@traceable_safe(name="generate_answer", run_type="chain")
async def generate_answer(
    config: AppConfig,
    retriever: VectorRetriever,
    driver: Driver,
    question: str,
    *,
    retrieval_limit: int = 5,
    include_entities: bool = True,
) -> dict[str, Any]:
    """Generate an answer using RAG (Retrieval-Augmented Generation).

    The prompt is fetched from the centralized catalog, enabling:
    - Version control via LangSmith Hub
    - A/B testing between prompt variants
    - Performance monitoring and evaluation

    Args:
        config: Application configuration.
        retriever: VectorRetriever for semantic search.
        driver: Neo4j driver for graph queries.
        question: User's question.
        retrieval_limit: Number of sources to retrieve.
        include_entities: Whether to include related entities in context.

    Returns:
        Dictionary with answer, sources, entities, and resources.
    """
    logger.info("Generating RAG answer for: '%s'", question)

    # Search for relevant definitions/glossary terms
    definitions = await search_terms(driver, question, limit=3)

    # Retrieve relevant context from chunks
    search_results = await graph_enriched_search(
        retriever,
        driver,
        question,
        limit=retrieval_limit,
    )

    # Build context and extract resources
    build_result = _build_context_from_results(
        definitions,
        search_results,
        include_entities=include_entities,
    )

    # Get prompt from catalog (uses cache if available)
    prompt_template = get_prompt_sync(PromptName.RAG_GENERATION)

    # Generate answer with LLM
    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0.1,
        api_key=config.openai_api_key,
    )

    # Use the prompt template from the catalog
    chain = prompt_template | llm | StrOutputParser()

    answer = await chain.ainvoke(
        {
            "context": build_result.context,
            "entities": build_result.entities_str,
            "question": question,
        }
    )

    # Convert Resource dataclasses to dicts for JSON serialization
    resources_dict = {
        "images": [
            {
                "title": r.title,
                "url": r.url,
                "alt_text": r.alt_text,
                "source_title": r.source_title,
            }
            for r in build_result.resources.get("images", [])
        ],
        "webinars": [
            {
                "title": r.title,
                "url": r.url,
                "source_title": r.source_title,
            }
            for r in build_result.resources.get("webinars", [])
        ],
        "videos": [
            {
                "title": r.title,
                "url": r.url,
                "source_title": r.source_title,
            }
            for r in build_result.resources.get("videos", [])
        ],
    }

    response = {
        "question": question,
        "answer": answer,
        "sources": build_result.sources,
        "entities": build_result.entities,
        "resources": resources_dict,
        "source_count": len(build_result.sources),
    }

    logger.info(
        "Generated answer with %d sources, %d entities, and %d resources",
        len(build_result.sources),
        len(build_result.entities),
        sum(len(v) for v in resources_dict.values()),
    )
    return response


def _build_context_from_results(
    definitions: list[dict[str, Any]],
    search_results: list[dict[str, Any]],
    *,
    include_entities: bool = True,
    max_resources_per_type: int = 3,
) -> ContextBuildResult:
    """Build context from retrieval results.

    Shared logic between generate_answer and stream_chat to avoid duplication.
    Extracts and formats all resources (images, webinars, videos) for both
    LLM context and frontend display.

    Args:
        definitions: Definition search results.
        search_results: Graph-enriched search results.
        include_entities: Whether to extract entities from results.
        max_resources_per_type: Maximum resources per type per source (default 3).

    Returns:
        ContextBuildResult with sources, entities, context string, and resources.
    """
    context_parts: list[str] = []
    sources: list[dict[str, Any]] = []
    all_entities: set[str] = set()

    # Track resources globally for deduplication by URL
    seen_urls: set[str] = set()
    all_images: list[Resource] = []
    all_webinars: list[Resource] = []
    all_videos: list[Resource] = []

    # Add definitions to context first (if any found)
    if definitions:
        for defn in definitions:
            if defn.get("score", 0) >= DEFINITION_RELEVANCE_THRESHOLD:
                defn_url = defn.get("url", "")
                # Include acronym in display if available
                term_display = defn["term"]
                if defn.get("acronym"):
                    term_display = f"{defn['term']} ({defn['acronym']})"
                context_parts.append(
                    f"[Definition: {term_display}]\n{defn['definition']}\nURL: {defn_url}\n"
                )
                sources.append(
                    {
                        "title": f"Definition: {term_display}",
                        "url": defn_url,
                        "chunk_id": None,
                        "relevance_score": defn.get("score", 0.5),
                    }
                )
                all_entities.add(defn["term"])

    for i, result in enumerate(search_results, 1):
        title = result["metadata"].get("title", "Unknown")
        content = result["content"]
        url = result["metadata"].get("url", "")

        # Build source content section
        source_context = f"[Source {i}: {title}]\n{content}\n"

        sources.append(
            {
                "title": title,
                "content": content,
                "url": url,
                "chunk_id": result["metadata"].get("chunk_id"),
                "relevance_score": result["score"],
            }
        )

        # Collect entities from enriched structure
        if include_entities:
            for entity in result.get("entities", []):
                if isinstance(entity, dict) and entity.get("name"):
                    all_entities.add(entity["name"])
                elif isinstance(entity, str) and entity:
                    all_entities.add(entity)
            for defn in result.get("glossary_definitions", []):
                if isinstance(defn, dict) and defn.get("term"):
                    all_entities.add(defn["term"])
                elif isinstance(defn, str) and defn:
                    all_entities.add(defn)

        # Extract resources from media (with per-source limits and global deduplication)
        source_resources: list[str] = []
        if result.get("media"):
            media = result["media"]

            # Extract images (limit per source)
            source_image_count = 0
            for img in media.get("images", []):
                img_url = img.get("url")
                if not img_url or img_url in seen_urls:
                    continue
                if source_image_count >= max_resources_per_type:
                    break
                seen_urls.add(img_url)
                source_image_count += 1
                alt_text = img.get("alt_text", "")
                all_images.append(
                    Resource(
                        title=alt_text or "Image",
                        url=img_url,
                        alt_text=alt_text,
                        source_title=title,
                    )
                )
                source_resources.append(f'- 🖼️ Image: "{alt_text}" - {img_url}')

            # Extract webinars (limit per source)
            source_webinar_count = 0
            for webinar in media.get("webinars", []):
                webinar_url = webinar.get("url")
                if not webinar_url or webinar_url in seen_urls:
                    continue
                if source_webinar_count >= max_resources_per_type:
                    break
                seen_urls.add(webinar_url)
                source_webinar_count += 1
                webinar_title = webinar.get("title", "Webinar")
                all_webinars.append(
                    Resource(
                        title=webinar_title,
                        url=webinar_url,
                        source_title=title,
                    )
                )
                source_resources.append(f'- 📹 Webinar: "{webinar_title}" - {webinar_url}')

            # Extract videos (limit per source)
            source_video_count = 0
            for video in media.get("videos", []):
                video_url = video.get("url")
                if not video_url or video_url in seen_urls:
                    continue
                if source_video_count >= max_resources_per_type:
                    break
                seen_urls.add(video_url)
                source_video_count += 1
                video_title = video.get("title", "Video")
                all_videos.append(
                    Resource(
                        title=video_title,
                        url=video_url,
                        source_title=title,
                    )
                )
                source_resources.append(f'- 🎬 Video: "{video_title}" - {video_url}')

        # Add resources section to source context if any resources found
        if source_resources:
            source_context += "\nResources from this source:\n"
            source_context += "\n".join(source_resources)
            source_context += "\n"

        context_parts.append(source_context)

    context = "\n".join(context_parts) if context_parts else "No relevant context found."
    entities_str = ", ".join(sorted(all_entities)[:20]) if all_entities else "None identified"
    entities_list = list(all_entities)[:20]

    return ContextBuildResult(
        sources=sources,
        entities=entities_list,
        context=context,
        entities_str=entities_str,
        resources={
            "images": all_images,
            "webinars": all_webinars,
            "videos": all_videos,
        },
    )


@traceable_safe(name="stream_chat", run_type="chain")
async def stream_chat(
    config: AppConfig,
    retriever: VectorRetriever,
    driver: Driver,
    message: str,
    *,
    conversation_history: list[dict[str, str]] | None = None,
    conversation_id: str | None = None,
    max_sources: int = 5,
    langsmith_extra: dict[str, Any] | None = None,  # For thread metadata
) -> AsyncIterator[StreamEvent]:
    """Stream chat response with progressive events.

    Emits events as retrieval and generation progress:
    1. SOURCES event with retrieved context, entities, and resources
    2. TOKEN events for each generated token
    3. DONE event with complete answer

    This enables LangSmith to capture streaming metrics like TTFT.
    When conversation_id is provided, traces are grouped into LangSmith Threads.

    Args:
        config: Application configuration.
        retriever: VectorRetriever for semantic search.
        driver: Neo4j driver for graph queries.
        message: User's message.
        conversation_history: Optional list of previous messages for multi-turn.
        conversation_id: Optional conversation ID for LangSmith thread grouping.
        max_sources: Maximum sources to retrieve.
        langsmith_extra: Optional LangSmith metadata (used internally for threading).

    Yields:
        StreamEvent objects for sources, tokens, and completion.
    """
    # Note: langsmith_extra is consumed by the @traceable decorator, not used here
    _ = langsmith_extra  # Suppress unused variable warning
    logger.info("Streaming chat: message='%s', conversation_id=%s", message[:50], conversation_id)

    try:
        # 1. Retrieval (non-streamed)
        definitions = await search_terms(driver, message, limit=3)
        search_results = await graph_enriched_search(
            retriever,
            driver,
            message,
            limit=max_sources,
        )

        # 2. Build context and collect metadata
        build_result = _build_context_from_results(
            definitions,
            search_results,
            include_entities=True,
        )

        # 3. Emit sources/entities/resources immediately
        # Convert Resource dataclasses to dicts for JSON serialization
        resources_dict = {
            "images": [
                {
                    "title": r.title,
                    "url": r.url,
                    "alt_text": r.alt_text,
                    "source_title": r.source_title,
                }
                for r in build_result.resources.get("images", [])
            ],
            "webinars": [
                {
                    "title": r.title,
                    "url": r.url,
                    "source_title": r.source_title,
                }
                for r in build_result.resources.get("webinars", [])
            ],
            "videos": [
                {
                    "title": r.title,
                    "url": r.url,
                    "source_title": r.source_title,
                }
                for r in build_result.resources.get("videos", [])
            ],
        }
        yield StreamEvent(
            event_type=StreamEventType.SOURCES,
            data={
                "sources": build_result.sources,
                "entities": build_result.entities,
                "resources": resources_dict,
            },
        )

        # 4. Convert conversation history to LangChain message format
        history_messages: list[HumanMessage | AIMessage] = []
        if conversation_history:
            for msg in conversation_history:
                if msg.get("role") == "user":
                    history_messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    history_messages.append(AIMessage(content=msg.get("content", "")))

        # 5. Get prompt and set up streaming chain
        prompt_template = get_prompt_sync(PromptName.RAG_GENERATION)
        llm = ChatOpenAI(
            model=config.chat_model,
            temperature=0.1,
            api_key=config.openai_api_key,
        )
        chain = prompt_template | llm | StrOutputParser()

        # 6. Stream LLM tokens
        full_answer = ""
        chain_input: dict[str, Any] = {
            "context": build_result.context,
            "entities": build_result.entities_str,
            "question": message,
        }

        # Add history if the prompt supports it
        if history_messages:
            chain_input["history"] = history_messages

        async for token in chain.astream(chain_input):
            full_answer += token
            yield StreamEvent(
                event_type=StreamEventType.TOKEN,
                data={"token": token},
            )

        # 7. Emit completion event
        yield StreamEvent(
            event_type=StreamEventType.DONE,
            data={"full_answer": full_answer, "source_count": len(build_result.sources)},
        )

    except Exception as e:
        logger.exception("Error in stream_chat")
        yield StreamEvent(
            event_type=StreamEventType.ERROR,
            data={"error": str(e)},
        )


__all__ = [
    "ContextBuildResult",
    "Resource",
    "StreamEvent",
    "StreamEventType",
    "_build_context_from_results",
    "generate_answer",
    "stream_chat",
]
