"""Standards and compliance lookup functions.

Provides access to Standard nodes for regulatory compliance,
industry standards, and certification requirements.

Updated Data Model (2026-01):
- Standards connect to industries via APPLIES_TO relationship
- Standards link to chunks via MENTIONED_IN (Standard -> Chunk direction)
- Use FROM_ARTICLE to navigate from chunks to articles
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neo4j import Driver

logger = logging.getLogger(__name__)


async def lookup_standard(
    driver: Driver,
    name: str,
    *,
    include_related: bool = True,
) -> dict[str, Any] | None:
    """Look up a specific standard by name.

    Args:
        driver: Neo4j driver instance.
        name: Standard name (e.g., "ISO 26262", "FDA", "DO-178C").
        include_related: Whether to include related entities.

    Returns:
        Dictionary with standard details or None if not found.
    """
    logger.info("Looking up standard: '%s'", name)

    with driver.session() as session:
        result = session.run(
            """
            MATCH (s:Standard)
            WHERE toLower(s.name) CONTAINS toLower($name)
               OR toLower(s.display_name) CONTAINS toLower($name)
            RETURN s.name AS name,
                   s.display_name AS display_name,
                   s.organization AS organization,
                   s.domain AS domain,
                   labels(s) AS labels
            LIMIT 1
            """,
            name=name,
        )
        record = result.single()

    if not record:
        logger.info("Standard not found: '%s'", name)
        return None

    response = {
        "name": record["name"],
        "display_name": record.get("display_name"),
        "organization": record.get("organization"),
        "domain": record.get("domain"),
        "labels": record["labels"],
    }

    if include_related:
        with driver.session() as session:
            # Get related entities
            related_result = session.run(
                """
                MATCH (s:Standard {name: $name})-[r]-(related)
                WHERE NOT related:Chunk AND NOT related:Article
                RETURN type(r) AS relationship,
                       related.name AS name,
                       related.display_name AS display_name,
                       labels(related) AS labels
                LIMIT 10
                """,
                name=record["name"],
            )
            response["related"] = [
                {
                    "name": r["name"],
                    "display_name": r["display_name"],
                    "relationship": r["relationship"],
                    "labels": r["labels"],
                }
                for r in related_result
            ]

            # Get articles mentioning this standard via MENTIONED_IN -> FROM_ARTICLE
            articles_result = session.run(
                """
                MATCH (s:Standard {name: $name})-[:MENTIONED_IN]->(c:Chunk)
                MATCH (c)-[:FROM_ARTICLE]->(a:Article)
                RETURN DISTINCT a.article_title AS title, a.url AS url
                LIMIT 5
                """,
                name=record["name"],
            )
            response["mentioned_in"] = [
                {"title": a["title"], "url": a["url"]} for a in articles_result
            ]

    logger.info("Found standard: '%s'", record["name"])
    return response


async def search_standards(
    driver: Driver,
    query: str,
    *,
    industry: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search for standards matching criteria.

    Args:
        driver: Neo4j driver instance.
        query: Search query (name, organization, or industry).
        industry: Optional filter by industry (e.g., "automotive", "medical").
        limit: Maximum number of results.

    Returns:
        List of matching standards.
    """
    logger.info("Searching standards: query='%s', industry='%s'", query, industry)

    with driver.session() as session:
        if industry:
            result = session.run(
                """
                MATCH (s:Standard)
                WHERE (toLower(s.name) CONTAINS toLower($query)
                       OR toLower(s.display_name) CONTAINS toLower($query)
                       OR toLower(s.organization) CONTAINS toLower($query))
                OPTIONAL MATCH (s)-[:APPLIES_TO]->(i:Industry)
                WHERE toLower(i.name) CONTAINS toLower($industry)
                   OR toLower(i.display_name) CONTAINS toLower($industry)
                WITH s, i
                WHERE i IS NOT NULL
                   OR toLower(s.display_name) CONTAINS toLower($industry)
                   OR toLower(s.domain) CONTAINS toLower($industry)
                RETURN s.name AS name,
                       s.display_name AS display_name,
                       s.organization AS organization,
                       s.domain AS domain
                LIMIT $limit
                """,
                query=query,
                industry=industry,
                limit=limit,
            )
        else:
            result = session.run(
                """
                MATCH (s:Standard)
                WHERE toLower(s.name) CONTAINS toLower($query)
                   OR toLower(s.display_name) CONTAINS toLower($query)
                   OR toLower(s.organization) CONTAINS toLower($query)
                RETURN s.name AS name,
                       s.display_name AS display_name,
                       s.organization AS organization,
                       s.domain AS domain
                LIMIT $limit
                """,
                query=query,
                limit=limit,
            )

        standards = [
            {
                "name": r["name"],
                "display_name": r.get("display_name"),
                "organization": r.get("organization"),
                "domain": r.get("domain"),
            }
            for r in result
        ]

    logger.info("Found %d standards", len(standards))
    return standards


async def get_standards_by_industry(
    driver: Driver,
    industry: str,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Get standards applicable to a specific industry.

    Args:
        driver: Neo4j driver instance.
        industry: Industry name (e.g., "automotive", "medical", "aerospace").
        limit: Maximum number of results.

    Returns:
        List of standards for the industry.
    """
    logger.info("Getting standards for industry: '%s'", industry)

    with driver.session() as session:
        # Try direct APPLIES_TO relationship first
        result = session.run(
            """
            MATCH (s:Standard)-[:APPLIES_TO]->(i:Industry)
            WHERE toLower(i.name) CONTAINS toLower($industry)
               OR toLower(i.display_name) CONTAINS toLower($industry)
            RETURN DISTINCT s.name AS name,
                   s.display_name AS display_name,
                   s.organization AS organization,
                   s.domain AS domain,
                   i.display_name AS industry_name
            LIMIT $limit
            """,
            industry=industry,
            limit=limit,
        )
        standards = list(result)

        # If no direct relationships, fallback to text search
        if not standards:
            # Map common industry names to search patterns
            industry_mappings = {
                "automotive": ["automotive", "vehicle", "ISO 26262", "ASPICE"],
                "medical": ["medical", "healthcare", "FDA", "IEC 62304", "ISO 13485"],
                "aerospace": ["aerospace", "aviation", "DO-178", "DO-254", "airborne"],
                "defense": ["defense", "military", "MIL-STD", "DoD"],
                "rail": ["rail", "railway", "EN 50128", "CENELEC"],
            }

            patterns = industry_mappings.get(industry.lower(), [industry])
            pattern_conditions = " OR ".join(
                [f"toLower(s.name) CONTAINS '{p.lower()}'" for p in patterns]
                + [f"toLower(s.display_name) CONTAINS '{p.lower()}'" for p in patterns]
            )

            fallback_result = session.run(
                f"""
                MATCH (s:Standard)
                WHERE {pattern_conditions}
                RETURN DISTINCT s.name AS name,
                       s.display_name AS display_name,
                       s.organization AS organization,
                       s.domain AS domain
                LIMIT $limit
                """,
                limit=limit,
            )
            standards = list(fallback_result)

        result_list = [
            {
                "name": r["name"],
                "display_name": r.get("display_name"),
                "organization": r.get("organization"),
                "domain": r.get("domain"),
            }
            for r in standards
        ]

    logger.info("Found %d standards for industry '%s'", len(result_list), industry)
    return result_list


async def list_all_standards(
    driver: Driver,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List all standards in the knowledge graph.

    Args:
        driver: Neo4j driver instance.
        limit: Maximum number of standards to return.

    Returns:
        List of all standards.
    """
    logger.info("Listing all standards, limit=%d", limit)

    with driver.session() as session:
        result = session.run(
            """
            MATCH (s:Standard)
            RETURN s.name AS name,
                   s.display_name AS display_name,
                   s.organization AS organization,
                   s.domain AS domain
            ORDER BY s.name
            LIMIT $limit
            """,
            limit=limit,
        )

        standards = [
            {
                "name": r["name"],
                "display_name": r.get("display_name"),
                "organization": r.get("organization"),
                "domain": r.get("domain"),
            }
            for r in result
        ]

    logger.info("Listed %d standards", len(standards))
    return standards
