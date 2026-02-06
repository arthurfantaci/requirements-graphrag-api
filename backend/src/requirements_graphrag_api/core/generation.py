"""Shared types and context-building for RAG answer generation.

This module provides shared data types (StreamEventType, StreamEvent, Resource,
ContextBuildResult) and the _build_context_from_results() function used by both
the agentic orchestrator and the evaluation pipeline.

For the agentic RAG path, see:
- agentic/orchestrator.py - Main graph composition
- agentic/streaming.py - Agentic SSE streaming
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

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

    # Guardrail events
    GUARDRAIL_WARNING = "guardrail_warning"  # Post-stream safety warning


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
        thumbnail_url: Thumbnail image URL (primarily for webinars).
    """

    title: str
    url: str
    alt_text: str = ""
    source_title: str = ""
    thumbnail_url: str = ""


@dataclass(frozen=True, slots=True)
class ContextBuildResult:
    """Result of building context from retrieval results.

    Attributes:
        sources: List of source dictionaries with metadata.
        entities: List of entity dicts with name and optional definition.
        context: Formatted context string for LLM.
        entities_str: Comma-separated entity string for LLM.
        resources: Dictionary mapping resource types to lists of Resource objects.
    """

    sources: list[dict[str, Any]]
    entities: list[dict[str, Any]]
    context: str
    entities_str: str
    resources: dict[str, list[Resource]] = field(default_factory=dict)


def _build_context_from_results(
    definitions: list[dict[str, Any]],
    search_results: list[dict[str, Any]],
    *,
    include_entities: bool = True,
    max_resources_per_type: int = 3,
) -> ContextBuildResult:
    """Build context from retrieval results.

    Shared logic used by the evaluation pipeline and agentic orchestrator.
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
    # Map entity names to {definition, label} for tracking node labels
    # label is the Neo4j node label (e.g., "Concept", "Entity", "Definition")
    all_entities: dict[str, dict[str, Any]] = {}

    # Track resources globally for deduplication by URL
    seen_urls: set[str] = set()
    all_images: list[Resource] = []
    all_webinars: list[Resource] = []
    all_videos: list[Resource] = []

    # Pre-collect webinar thumbnail URLs to exclude from images (Issue #54)
    webinar_thumbnail_urls: set[str] = set()
    for r in search_results:
        for w in r.get("media", {}).get("webinars", []):
            thumb = w.get("thumbnail_url")
            if thumb:
                webinar_thumbnail_urls.add(thumb)

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
                # Store entity with its definition and label (Definition node)
                all_entities[defn["term"]] = {
                    "definition": defn.get("definition"),
                    "label": "Definition",
                }

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
                    name = entity["name"]
                    label = entity.get("type", "Entity")  # Neo4j node label
                    definition = entity.get("definition")
                    # Don't overwrite if we already have a definition
                    if name not in all_entities:
                        all_entities[name] = {"definition": definition, "label": label}
                    elif definition and not all_entities[name].get("definition"):
                        # Update with definition if we didn't have one
                        all_entities[name]["definition"] = definition
                elif isinstance(entity, str) and entity:
                    if entity not in all_entities:
                        all_entities[entity] = {"definition": None, "label": "Entity"}
            # Glossary definitions include the actual definition text
            for defn in result.get("glossary_definitions", []):
                if isinstance(defn, dict) and defn.get("term"):
                    term = defn["term"]
                    definition = defn.get("definition")
                    # Prefer definition over None, mark as Definition label
                    if term not in all_entities:
                        all_entities[term] = {"definition": definition, "label": "Definition"}
                    elif definition and not all_entities[term].get("definition"):
                        all_entities[term]["definition"] = definition
                        all_entities[term]["label"] = "Definition"
                elif isinstance(defn, str) and defn:
                    if defn not in all_entities:
                        all_entities[defn] = {"definition": None, "label": "Entity"}

        # Extract resources from media (with per-source limits and global deduplication)
        source_resources: list[str] = []
        if result.get("media"):
            media = result["media"]

            # Extract images (limit per source)
            source_image_count = 0
            for img in media.get("images", []):
                img_url = img.get("url")
                if not img_url or img_url in seen_urls or img_url in webinar_thumbnail_urls:
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
                webinar_thumbnail = webinar.get("thumbnail_url", "")
                all_webinars.append(
                    Resource(
                        title=webinar_title,
                        url=webinar_url,
                        source_title=title,
                        thumbnail_url=webinar_thumbnail,
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
    # Sort by name and limit to 20 entities
    sorted_names = sorted(all_entities.keys())[:20]
    entities_str = ", ".join(sorted_names) if sorted_names else "None identified"
    # Build list of entity objects with name, definition, and node label
    entities_list = [
        {
            "name": name,
            "definition": all_entities[name].get("definition"),
            "label": all_entities[name].get("label", "Entity"),
        }
        for name in sorted_names
    ]

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


__all__ = [
    "ContextBuildResult",
    "Resource",
    "StreamEvent",
    "StreamEventType",
    "_build_context_from_results",
]
