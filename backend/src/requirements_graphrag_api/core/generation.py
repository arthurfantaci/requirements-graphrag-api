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
from dataclasses import dataclass
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

    SOURCES = "sources"
    TOKEN = "token"  # noqa: S105 - not a password
    DONE = "done"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """A single event in the streaming response.

    Attributes:
        event_type: The type of event (sources, token, done, error).
        data: The event payload as a dictionary.
    """

    event_type: StreamEventType
    data: dict[str, Any]


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
        Dictionary with answer, sources, and entities.
    """
    logger.info("Generating RAG answer for: '%s'", question)

    # Search for relevant definitions/glossary terms
    # This helps answer questions like "What does X mean?" or "What is the definition of Y?"
    definitions = await search_terms(driver, question, limit=3)

    # Retrieve relevant context from chunks
    search_results = await graph_enriched_search(
        retriever,
        driver,
        question,
        limit=retrieval_limit,
    )

    # Build context string
    context_parts = []
    sources = []
    all_entities: set[str] = set()
    all_images: list[dict[str, str]] = []
    seen_image_urls: set[str] = set()

    # Add definitions to context first (if any found)
    if definitions:
        for defn in definitions:
            if defn.get("score", 0) >= DEFINITION_RELEVANCE_THRESHOLD:
                context_parts.append(f"[Definition: {defn['term']}]\n{defn['definition']}\n")
                # Add definition as a source
                sources.append(
                    {
                        "title": f"Definition: {defn['term']}",
                        "url": defn.get("url", ""),
                        "chunk_id": None,
                        "relevance_score": defn.get("score", 0.5),
                    }
                )
                all_entities.add(defn["term"])

    for i, result in enumerate(search_results, 1):
        title = result["metadata"].get("title", "Unknown")
        content = result["content"]
        url = result["metadata"].get("url", "")

        context_parts.append(f"[Source {i}: {title}]\n{content}\n")
        sources.append(
            {
                "title": title,
                "content": content,  # Include content for evaluation metrics
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

        # Collect images from media enrichment (deduplicated by URL)
        if result.get("media"):
            for img in result["media"].get("images", []):
                img_url = img.get("url")
                if img_url and img_url not in seen_image_urls:
                    seen_image_urls.add(img_url)
                    all_images.append(
                        {
                            "url": img_url,
                            "alt_text": img.get("alt_text", ""),
                            "context": img.get("context", ""),
                            "source_title": result["metadata"].get("title", ""),
                        }
                    )

    context = "\n".join(context_parts) if context_parts else "No relevant context found."
    entities_str = ", ".join(sorted(all_entities)[:20]) if all_entities else "None identified"

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
            "context": context,
            "entities": entities_str,
            "question": question,
        }
    )

    response = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "entities": list(all_entities)[:20],
        "images": all_images[:5],  # Limit to 5 images
        "source_count": len(sources),
    }

    logger.info(
        "Generated answer with %d sources, %d entities, and %d images",
        len(sources),
        len(all_entities),
        len(all_images[:5]),
    )
    return response


def _build_context_from_results(
    definitions: list[dict[str, Any]],
    search_results: list[dict[str, Any]],
    *,
    include_entities: bool = True,
) -> tuple[
    list[dict[str, Any]],
    list[str],
    list[dict[str, str]],
    str,
    str,
]:
    """Build context from retrieval results.

    Shared logic between generate_answer and stream_chat to avoid duplication.

    Args:
        definitions: Definition search results.
        search_results: Graph-enriched search results.
        include_entities: Whether to extract entities from results.

    Returns:
        Tuple of (sources, entities_list, images, context_str, entities_str).
    """
    context_parts: list[str] = []
    sources: list[dict[str, Any]] = []
    all_entities: set[str] = set()
    all_images: list[dict[str, str]] = []
    seen_image_urls: set[str] = set()

    # Add definitions to context first (if any found)
    if definitions:
        for defn in definitions:
            if defn.get("score", 0) >= DEFINITION_RELEVANCE_THRESHOLD:
                context_parts.append(f"[Definition: {defn['term']}]\n{defn['definition']}\n")
                sources.append(
                    {
                        "title": f"Definition: {defn['term']}",
                        "url": defn.get("url", ""),
                        "chunk_id": None,
                        "relevance_score": defn.get("score", 0.5),
                    }
                )
                all_entities.add(defn["term"])

    for i, result in enumerate(search_results, 1):
        title = result["metadata"].get("title", "Unknown")
        content = result["content"]
        url = result["metadata"].get("url", "")

        context_parts.append(f"[Source {i}: {title}]\n{content}\n")
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

        # Collect images from media enrichment (deduplicated by URL)
        if result.get("media"):
            for img in result["media"].get("images", []):
                img_url = img.get("url")
                if img_url and img_url not in seen_image_urls:
                    seen_image_urls.add(img_url)
                    all_images.append(
                        {
                            "url": img_url,
                            "alt_text": img.get("alt_text", ""),
                            "context": img.get("context", ""),
                            "source_title": result["metadata"].get("title", ""),
                        }
                    )

    context = "\n".join(context_parts) if context_parts else "No relevant context found."
    entities_str = ", ".join(sorted(all_entities)[:20]) if all_entities else "None identified"
    entities_list = list(all_entities)[:20]

    return sources, entities_list, all_images[:5], context, entities_str


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
    1. SOURCES event with retrieved context, entities, and images
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
        sources, entities, images, context, entities_str = _build_context_from_results(
            definitions,
            search_results,
            include_entities=True,
        )

        # 3. Emit sources/entities/images immediately
        yield StreamEvent(
            event_type=StreamEventType.SOURCES,
            data={"sources": sources, "entities": entities, "images": images},
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
            "context": context,
            "entities": entities_str,
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
            data={"full_answer": full_answer, "source_count": len(sources)},
        )

    except Exception as e:
        logger.exception("Error in stream_chat")
        yield StreamEvent(
            event_type=StreamEventType.ERROR,
            data={"error": str(e)},
        )


__all__ = ["StreamEvent", "StreamEventType", "generate_answer", "stream_chat"]
