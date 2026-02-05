"""Definition lookup functions for term definitions.

Provides access to the Definition nodes in the knowledge graph
for looking up requirements management terminology.

Updated Data Model (2026-01):
- Definition nodes replace the former GlossaryTerm nodes
- Properties: term, definition, url, term_id
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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
