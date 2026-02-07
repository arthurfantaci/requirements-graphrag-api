"""Definition lookup functions for term definitions.

Provides access to the Definition nodes in the knowledge graph
for looking up requirements management terminology.

Updated Data Model (2026-01):
- Definition nodes replace the former GlossaryTerm nodes
- Properties: term, definition, url, term_id
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from neo4j import Driver

logger = logging.getLogger(__name__)


async def lookup_term(
    driver: Driver,
    term: str,
    *,
    fuzzy: bool = True,
) -> dict[str, Any] | None:
    """Look up a definition term.

    Args:
        driver: Neo4j driver instance.
        term: Term to look up.
        fuzzy: If True, use fuzzy matching; if False, exact match only.

    Returns:
        Dictionary with term details or None if not found.
    """
    logger.info("Looking up definition term: '%s' (fuzzy=%s)", term, fuzzy)

    # Normalize for acronym matching
    normalized_term = _normalize_for_acronym_match(term)

    with driver.session() as session:
        if fuzzy:
            # Use CONTAINS for fuzzy matching, including acronym search
            result = session.run(
                """
                MATCH (d:Definition)
                WHERE toLower(d.term) CONTAINS toLower($term)
                   OR toLower(d.definition) CONTAINS toLower($term)
                   OR (d.acronym IS NOT NULL
                       AND replace(replace(toLower(d.acronym), '.', ''), ' ', '')
                         = $normalized_term)
                WITH d, CASE
                    WHEN toLower(d.term) = toLower($term) THEN 1.0
                    WHEN d.acronym IS NOT NULL
                         AND replace(replace(toLower(d.acronym), '.', ''), ' ', '')
                           = $normalized_term THEN 1.0
                    WHEN toLower(d.term) STARTS WITH toLower($term) THEN 0.9
                    WHEN toLower(d.term) CONTAINS toLower($term) THEN 0.7
                    ELSE 0.5
                END AS score
                RETURN d.term AS term,
                       d.definition AS definition,
                       d.acronym AS acronym,
                       d.url AS url,
                       d.term_id AS term_id,
                       score
                ORDER BY score DESC
                LIMIT 1
                """,
                term=term,
                normalized_term=normalized_term,
            )
        else:
            # Exact match (case-insensitive) on term or acronym
            result = session.run(
                """
                MATCH (d:Definition)
                WHERE toLower(d.term) = toLower($term)
                   OR (d.acronym IS NOT NULL
                       AND replace(replace(toLower(d.acronym), '.', ''), ' ', '')
                         = $normalized_term)
                RETURN d.term AS term,
                       d.definition AS definition,
                       d.acronym AS acronym,
                       d.url AS url,
                       d.term_id AS term_id,
                       1.0 AS score
                LIMIT 1
                """,
                term=term,
                normalized_term=normalized_term,
            )

        record = result.single()
        if record:
            logger.info("Found definition term: '%s'", record["term"])
            return {
                "term": record["term"],
                "definition": record["definition"],
                "acronym": record.get("acronym"),
                "url": record.get("url"),
                "term_id": record.get("term_id"),
                "score": record["score"],
            }

        logger.info("Definition term not found: '%s'", term)
        return None


def _normalize_for_acronym_match(text: str) -> str:
    """Normalize text for acronym matching.

    Removes periods and spaces to handle variations like:
    - "AoA" vs "A.o.A." vs "A o A"
    - "ALM" vs "A.L.M."

    Args:
        text: Text to normalize.

    Returns:
        Lowercase text with periods and spaces removed.
    """
    import re

    return re.sub(r"[\s.]", "", text.lower())


async def search_terms(
    driver: Driver,
    query: str,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search for definition terms matching a query.

    Searches across term names, definitions, and acronyms with intelligent
    scoring. Handles acronym variations (AoA, A.o.A., A o A).

    Args:
        driver: Neo4j driver instance.
        query: Search query.
        limit: Maximum number of results.

    Returns:
        List of matching terms with definitions and acronyms.
    """
    logger.info("Searching definition terms: '%s', limit=%d", query, limit)

    # Normalize query for acronym matching
    normalized_query = _normalize_for_acronym_match(query)

    with driver.session() as session:
        # Use bidirectional matching: term in query OR query in term/definition
        # Also search acronym field for matches like "AoA" -> "Analysis of Alternatives"
        result = session.run(
            """
            MATCH (d:Definition)
            WHERE toLower(d.term) CONTAINS toLower($search_term)
               OR toLower(d.definition) CONTAINS toLower($search_term)
               OR toLower($search_term) CONTAINS toLower(d.term)
               // Acronym matching (handle NULL safely)
               OR (d.acronym IS NOT NULL
                   AND replace(replace(toLower(d.acronym), '.', ''), ' ', '')
                     = $normalized_query)
               OR (d.acronym IS NOT NULL
                   AND $normalized_query CONTAINS
                       replace(replace(toLower(d.acronym), '.', ''), ' ', ''))
            WITH d, CASE
                // Exact term match
                WHEN toLower(d.term) = toLower($search_term) THEN 1.0
                // Exact acronym match (e.g., "AoA" -> "Analysis of Alternatives")
                WHEN d.acronym IS NOT NULL
                     AND replace(replace(toLower(d.acronym), '.', ''), ' ', '')
                       = $normalized_query THEN 1.0
                // Acronym found within query (e.g., "What is AoA?")
                WHEN d.acronym IS NOT NULL
                     AND size(d.acronym) >= 2
                     AND $normalized_query CONTAINS
                         replace(replace(toLower(d.acronym), '.', ''), ' ', '')
                    THEN 0.9
                // Term found within query
                WHEN toLower($search_term) CONTAINS toLower(d.term)
                     AND size(d.term) > 3 THEN 0.85
                WHEN toLower(d.term) STARTS WITH toLower($search_term) THEN 0.8
                WHEN toLower(d.term) CONTAINS toLower($search_term) THEN 0.7
                ELSE 0.5
            END AS score
            RETURN d.term AS term,
                   d.definition AS definition,
                   d.acronym AS acronym,
                   d.url AS url,
                   score
            ORDER BY score DESC, size(d.term) DESC, d.term
            LIMIT $limit
            """,
            search_term=query,
            normalized_query=normalized_query,
            limit=limit,
        )

        terms = [
            {
                "term": r["term"],
                "definition": r["definition"],
                "acronym": r.get("acronym"),
                "url": r.get("url"),
                "score": round(float(r["score"]), 4),
            }
            for r in result
        ]

    logger.info("Found %d definition terms", len(terms))
    return terms


async def list_all_terms(
    driver: Driver,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List all definition terms alphabetically.

    Args:
        driver: Neo4j driver instance.
        limit: Maximum number of terms to return.

    Returns:
        List of all terms with definitions.
    """
    logger.info("Listing all definition terms, limit=%d", limit)

    with driver.session() as session:
        result = session.run(
            """
            MATCH (d:Definition)
            RETURN d.term AS term,
                   d.definition AS definition,
                   d.url AS url
            ORDER BY d.term
            LIMIT $limit
            """,
            limit=limit,
        )

        terms = [
            {
                "term": r["term"],
                "definition": r["definition"],
                "url": r.get("url"),
            }
            for r in result
        ]

    logger.info("Listed %d definition terms", len(terms))
    return terms


# =============================================================================
# Shared types and context-building for RAG answer generation
# (moved from core/generation.py)
# =============================================================================

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
    """A single event in the streaming response."""

    event_type: StreamEventType
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Resource:
    """A single resource (webinar, video, or image)."""

    title: str
    url: str
    alt_text: str = ""
    source_title: str = ""
    thumbnail_url: str = ""


@dataclass(frozen=True, slots=True)
class ContextBuildResult:
    """Result of building context from retrieval results."""

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
    """
    context_parts: list[str] = []
    sources: list[dict[str, Any]] = []
    all_entities: dict[str, dict[str, Any]] = {}

    seen_urls: set[str] = set()
    all_images: list[Resource] = []
    all_webinars: list[Resource] = []
    all_videos: list[Resource] = []

    webinar_thumbnail_urls: set[str] = set()
    for r in search_results:
        for w in r.get("media", {}).get("webinars", []):
            thumb = w.get("thumbnail_url")
            if thumb:
                webinar_thumbnail_urls.add(thumb)

    if definitions:
        for defn in definitions:
            if defn.get("score", 0) >= DEFINITION_RELEVANCE_THRESHOLD:
                defn_url = defn.get("url", "")
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
                all_entities[defn["term"]] = {
                    "definition": defn.get("definition"),
                    "label": "Definition",
                }

    for i, result in enumerate(search_results, 1):
        title = result["metadata"].get("title", "Unknown")
        content = result["content"]
        url = result["metadata"].get("url", "")

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

        if include_entities:
            for entity in result.get("entities", []):
                if isinstance(entity, dict) and entity.get("name"):
                    name = entity["name"]
                    label = entity.get("type", "Entity")
                    definition = entity.get("definition")
                    if name not in all_entities:
                        all_entities[name] = {"definition": definition, "label": label}
                    elif definition and not all_entities[name].get("definition"):
                        all_entities[name]["definition"] = definition
                elif isinstance(entity, str) and entity:
                    if entity not in all_entities:
                        all_entities[entity] = {"definition": None, "label": "Entity"}
            for defn in result.get("glossary_definitions", []):
                if isinstance(defn, dict) and defn.get("term"):
                    term = defn["term"]
                    definition = defn.get("definition")
                    if term not in all_entities:
                        all_entities[term] = {"definition": definition, "label": "Definition"}
                    elif definition and not all_entities[term].get("definition"):
                        all_entities[term]["definition"] = definition
                        all_entities[term]["label"] = "Definition"
                elif isinstance(defn, str) and defn:
                    if defn not in all_entities:
                        all_entities[defn] = {"definition": None, "label": "Entity"}

        source_resources: list[str] = []
        if result.get("media"):
            media = result["media"]

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
                source_resources.append(f'- \U0001f5bc\ufe0f Image: "{alt_text}" - {img_url}')

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
                source_resources.append(f'- \U0001f4f9 Webinar: "{webinar_title}" - {webinar_url}')

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
                source_resources.append(f'- \U0001f3ac Video: "{video_title}" - {video_url}')

        if source_resources:
            source_context += "\nResources from this source:\n"
            source_context += "\n".join(source_resources)
            source_context += "\n"

        context_parts.append(source_context)

    context = "\n".join(context_parts) if context_parts else "No relevant context found."
    sorted_names = sorted(all_entities.keys())[:20]
    entities_str = ", ".join(sorted_names) if sorted_names else "None identified"
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
