"""Definition lookup functions for term definitions.

Provides access to the Definition nodes in the knowledge graph
for looking up requirements management terminology.

Updated Data Model (2026-01):
- Definition nodes replace the former GlossaryTerm nodes
- Properties: term, definition, url, term_id
"""

from __future__ import annotations

import asyncio
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

    def _query() -> dict[str, Any] | None:
        with driver.session() as session:
            if fuzzy:
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
                return {
                    "term": record["term"],
                    "definition": record["definition"],
                    "acronym": record.get("acronym"),
                    "url": record.get("url"),
                    "term_id": record.get("term_id"),
                    "score": record["score"],
                }
            return None

    found = await asyncio.to_thread(_query)
    if found:
        logger.info("Found definition term: '%s'", found["term"])
    else:
        logger.info("Definition term not found: '%s'", term)
    return found


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

    normalized_query = _normalize_for_acronym_match(query)

    def _search() -> list[dict[str, Any]]:
        with driver.session() as session:
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

            return [
                {
                    "term": r["term"],
                    "definition": r["definition"],
                    "acronym": r.get("acronym"),
                    "url": r.get("url"),
                    "score": round(float(r["score"]), 4),
                }
                for r in result
            ]

    terms = await asyncio.to_thread(_search)
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

    def _query() -> list[dict[str, Any]]:
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
            return [
                {
                    "term": r["term"],
                    "definition": r["definition"],
                    "url": r.get("url"),
                }
                for r in result
            ]

    terms = await asyncio.to_thread(_query)
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

    Delegates to :func:`~requirements_graphrag_api.core.context.format_context`
    for entity/glossary/relationship/standards formatting, then appends
    inline media resource sections for evaluation pipeline compatibility.

    The evaluation pipeline expects media URLs embedded in the context
    string, unlike the production SSE handler which sends media via
    separate events.
    """
    from requirements_graphrag_api.core.context import NormalizedDocument, format_context

    # Normalize raw dicts → NormalizedDocument
    normalized = [NormalizedDocument.from_raw_result(r) for r in search_results]

    # Delegate to shared format_context (handles entities, glossary,
    # relationships, standards, media extraction, deduplication)
    formatted = format_context(
        normalized,
        definitions=definitions if include_entities else None,
        max_resources_per_type=max_resources_per_type,
    )

    # ---- Evaluation-specific: embed media in context string ----
    # format_context() extracts media to resources (for SSE), but the
    # evaluation pipeline expects them inline with emoji formatting.
    context = _embed_media_in_context(
        formatted.context,
        search_results,
        formatted.resources,
        max_resources_per_type,
    )

    # Build entities_list in the ContextBuildResult format
    sorted_names = sorted(formatted.entities_by_name.keys())[:20]
    entities_list = [
        {
            "name": name,
            "definition": formatted.entities_by_name[name].get("definition"),
            "label": formatted.entities_by_name[name].get("label", "Entity"),
        }
        for name in sorted_names
    ]

    return ContextBuildResult(
        sources=formatted.sources,
        entities=entities_list,
        context=context,
        entities_str=formatted.entities_str,
        resources=formatted.resources,
    )


def _embed_media_in_context(
    context: str,
    search_results: list[dict[str, Any]],
    resources: dict[str, list[Resource]],
    max_per_type: int,
) -> str:
    """Embed media resource lines into per-source context sections.

    This preserves the evaluation pipeline's format where each source
    block includes ``Resources from this source:`` with emoji-prefixed
    media lines.
    """
    # Build per-source media lines from the collected resources
    # Map source_title -> list of media lines
    source_media: dict[str, list[str]] = {}
    for img in resources.get("images", []):
        key = img.source_title
        source_media.setdefault(key, []).append(
            f'- \U0001f5bc\ufe0f Image: "{img.alt_text}" - {img.url}'
        )
    for webinar in resources.get("webinars", []):
        key = webinar.source_title
        source_media.setdefault(key, []).append(
            f'- \U0001f4f9 Webinar: "{webinar.title}" - {webinar.url}'
        )
    for video in resources.get("videos", []):
        key = video.source_title
        source_media.setdefault(key, []).append(
            f'- \U0001f3ac Video: "{video.title}" - {video.url}'
        )

    if not source_media:
        return context

    # Insert media lines after each [Source N: title] block
    lines = context.split("\n")
    result_lines: list[str] = []
    current_source_title = ""

    for line in lines:
        result_lines.append(line)
        # Detect source headers: "[Source N: Title]"
        if line.startswith("[Source ") and line.endswith("]"):
            # Extract title from "[Source N: Title]"
            colon_pos = line.find(": ")
            if colon_pos > 0:
                current_source_title = line[colon_pos + 2 : -1]

        # Detect chunk separator — insert media for previous source
        if line == "---" and current_source_title in source_media:
            media_lines = source_media.pop(current_source_title)
            # Insert before the separator
            separator = result_lines.pop()  # remove "---"
            # Remove trailing empty line if present
            if result_lines and result_lines[-1] == "":
                result_lines.pop()
            result_lines.append("")
            result_lines.append("Resources from this source:")
            result_lines.extend(media_lines)
            result_lines.append("")
            result_lines.append(separator)
            current_source_title = ""

    # Handle media for the last source (no trailing "---")
    if current_source_title in source_media:
        media_lines = source_media.pop(current_source_title)
        result_lines.append("")
        result_lines.append("Resources from this source:")
        result_lines.extend(media_lines)

    return "\n".join(result_lines)
