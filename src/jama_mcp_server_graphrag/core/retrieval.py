"""Core retrieval functions for GraphRAG.

This module contains the shared retrieval logic used by both MCP tools
and REST API routes. Keeping logic here ensures DRY principle is followed.

Updated Data Model (2026-01):
- Chunks contain text directly, linked via FROM_ARTICLE to Articles
- Entities link to chunks via MENTIONED_IN (Entity -> Chunk direction)
- Definitions replace GlossaryTerm nodes
- NEXT_CHUNK provides sequential ordering of chunks

Retrieval Patterns:
1. Vector Search - Pure semantic similarity using embeddings
2. Hybrid Search - Combines vector + keyword (fulltext) search
3. Graph-Enriched - Adds related entities via graph traversal
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from jama_mcp_server_graphrag.observability import traceable

if TYPE_CHECKING:
    from neo4j import Driver
    from neo4j_graphrag.retrievers import VectorRetriever

    from jama_mcp_server_graphrag.config import AppConfig

logger = logging.getLogger(__name__)


def create_vector_retriever(
    driver: Driver,
    config: AppConfig,
) -> VectorRetriever:
    """Create a VectorRetriever for semantic search.

    Uses neo4j-graphrag library for direct Neo4j vector index integration.

    Args:
        driver: Neo4j driver instance.
        config: Application configuration.

    Returns:
        Configured VectorRetriever instance.
    """
    from neo4j_graphrag.embeddings import OpenAIEmbeddings  # noqa: PLC0415
    from neo4j_graphrag.retrievers import VectorRetriever  # noqa: PLC0415

    embedder = OpenAIEmbeddings(
        model=config.embedding_model,
        api_key=config.openai_api_key or os.getenv("OPENAI_API_KEY"),
    )

    return VectorRetriever(
        driver=driver,
        index_name=config.vector_index_name,
        embedder=embedder,
        return_properties=["text", "index"],
    )


@traceable(name="vector_search", run_type="retriever")
async def vector_search(
    retriever: VectorRetriever,
    driver: Driver,
    query: str,
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Perform semantic similarity search using vector embeddings.

    Uses the neo4j-graphrag VectorRetriever for efficient vector search
    on the chunk_embeddings index.

    Args:
        retriever: Configured VectorRetriever instance.
        driver: Neo4j driver for additional context queries.
        query: Natural language search query.
        limit: Maximum number of results to return.

    Returns:
        List of results with content, score, and metadata.

    Example:
        >>> results = await vector_search(retriever, driver, "requirements traceability", limit=5)
        >>> for r in results:
        ...     print(f"{r['score']:.3f}: {r['metadata']['title']}")
    """
    logger.info("Vector search: query='%s', limit=%d", query, limit)

    # Perform similarity search with scores
    results = retriever.search(query_text=query, top_k=limit)

    # Get article context for each chunk
    chunk_ids = []
    for item in results.items:
        element_id = item.metadata.get("element_id") if item.metadata else None
        if element_id:
            chunk_ids.append(element_id)

    # Fetch article metadata for chunks
    article_context = {}
    if chunk_ids:
        with driver.session() as session:
            context_result = session.run(
                """
                MATCH (c:Chunk)-[:FROM_ARTICLE]->(a:Article)
                WHERE elementId(c) IN $chunk_ids
                RETURN elementId(c) AS chunk_id,
                       a.article_title AS title,
                       a.url AS url,
                       a.article_id AS article_id,
                       a.chapter_title AS chapter
                """,
                chunk_ids=chunk_ids,
            )
            for record in context_result:
                article_context[record["chunk_id"]] = {
                    "title": record["title"],
                    "url": record["url"],
                    "article_id": record["article_id"],
                    "chapter": record["chapter"],
                }

    # Format results
    formatted = []
    for item in results.items:
        element_id = item.metadata.get("element_id") if item.metadata else None
        score = item.metadata.get("score", 0.0) if item.metadata else 0.0

        # Extract text content
        if isinstance(item.content, dict):
            content = item.content.get("text", "")
        else:
            content = str(item.content) if item.content else ""

        # Build metadata
        metadata = article_context.get(element_id, {})
        metadata["chunk_id"] = element_id
        if item.metadata:
            metadata["chunk_index"] = item.metadata.get("index")

        formatted.append(
            {
                "content": content,
                "score": round(float(score), 4),
                "metadata": metadata,
            }
        )

    logger.info("Vector search returned %d results", len(formatted))
    return formatted


@traceable(name="hybrid_search", run_type="retriever")
async def hybrid_search(
    retriever: VectorRetriever,
    driver: Driver,
    query: str,
    *,
    limit: int = 6,
    keyword_weight: float = 0.3,
) -> list[dict[str, Any]]:
    """Perform hybrid search combining vector similarity and keyword matching.

    Combines semantic understanding (vectors) with exact term matching (keywords)
    for improved recall on queries with specific technical terms.

    Args:
        retriever: Configured VectorRetriever instance.
        driver: Neo4j driver for keyword search.
        query: Natural language search query.
        limit: Maximum number of results to return.
        keyword_weight: Weight for keyword results (0-1). Vector weight = 1 - keyword_weight.

    Returns:
        List of results with content, score, source, and metadata.
    """
    logger.info(
        "Hybrid search: query='%s', limit=%d, keyword_weight=%.2f",
        query,
        limit,
        keyword_weight,
    )

    # Get vector search results
    vector_results = await vector_search(retriever, driver, query, limit=limit * 2)

    # Get keyword search results using fulltext index on chunks
    keyword_results = []
    try:
        with driver.session() as session:
            keyword_result = session.run(
                """
                CALL db.index.fulltext.queryNodes('chunk_fulltext', $query)
                YIELD node, score
                MATCH (node)-[:FROM_ARTICLE]->(a:Article)
                RETURN node.text AS content,
                       score,
                       {
                           article_id: a.article_id,
                           title: a.article_title,
                           chapter: a.chapter_title,
                           url: a.url,
                           chunk_id: elementId(node)
                       } AS metadata
                LIMIT $limit
                """,
                query=query,
                limit=limit * 2,
            )
            keyword_results = [
                {
                    "content": r["content"],
                    "score": float(r["score"]),
                    "metadata": dict(r["metadata"]) if r["metadata"] else {},
                    "source": "keyword",
                }
                for r in keyword_result
            ]
    except Exception as e:
        # Fulltext index may not exist - fall back to vector only
        logger.warning("Keyword search failed (fulltext index may not exist): %s", e)

    # Merge and deduplicate results
    seen_ids = set()
    merged = []

    # Add vector results with adjusted score
    for r in vector_results:
        article_id = r["metadata"].get("article_id")
        if article_id and article_id not in seen_ids:
            seen_ids.add(article_id)
            merged.append(
                {
                    **r,
                    "score": r["score"] * (1 - keyword_weight),
                    "source": "vector",
                }
            )

    # Add keyword results with adjusted score
    for r in keyword_results:
        article_id = r["metadata"].get("article_id")
        if article_id and article_id not in seen_ids:
            seen_ids.add(article_id)
            merged.append(
                {
                    **r,
                    "score": r["score"] * keyword_weight,
                    "source": "keyword",
                }
            )
        elif article_id in seen_ids:
            # Boost score for results found in both
            for m in merged:
                if m["metadata"].get("article_id") == article_id:
                    m["score"] += r["score"] * keyword_weight
                    m["source"] = "hybrid"
                    break

    # Sort by combined score and limit
    merged.sort(key=lambda x: x["score"], reverse=True)
    results = merged[:limit]

    logger.info("Hybrid search returned %d results", len(results))
    return results


@traceable(name="graph_enriched_search", run_type="retriever")
async def graph_enriched_search(
    retriever: VectorRetriever,
    driver: Driver,
    query: str,
    *,
    limit: int = 6,
    traversal_depth: int = 1,  # noqa: ARG001
) -> list[dict[str, Any]]:
    """Perform vector search enriched with graph context.

    Combines semantic search with graph traversal to add related
    entities, concepts, and context to search results.

    Note: Uses the reversed MENTIONED_IN relationship direction where
    entities point to chunks: (Entity)-[:MENTIONED_IN]->(Chunk)

    Args:
        retriever: Configured VectorRetriever instance.
        driver: Neo4j driver for traversal.
        query: Natural language search query.
        limit: Maximum number of base results.
        traversal_depth: How many hops to traverse for related entities.

    Returns:
        List of enriched results with content, score, metadata, and related entities.
    """
    logger.info(
        "Graph-enriched search: query='%s', limit=%d",
        query,
        limit,
    )

    # Get base vector results
    base_results = await vector_search(retriever, driver, query, limit=limit)

    # Collect all chunk IDs for batch query
    chunk_ids = [
        r["metadata"].get("chunk_id") for r in base_results if r["metadata"].get("chunk_id")
    ]

    if not chunk_ids:
        return base_results

    # Batch query for entities mentioned in chunks
    # Note: MENTIONED_IN direction is Entity -> Chunk
    entities_by_chunk: dict[str, list[str]] = {}
    definitions_by_chunk: dict[str, list[str]] = {}

    with driver.session() as session:
        # Get entities that mention these chunks
        entity_result = session.run(
            """
            MATCH (entity)-[:MENTIONED_IN]->(c:Chunk)
            WHERE elementId(c) IN $chunk_ids
            WITH elementId(c) AS chunk_id,
                 collect(DISTINCT entity.display_name)[..10] AS entities
            RETURN chunk_id, entities
            """,
            chunk_ids=chunk_ids,
        )
        for record in entity_result:
            entities_by_chunk[record["chunk_id"]] = record["entities"]

        # Get any definitions that might be related (via entity relationships)
        # Since Definition nodes exist, we can try to find related definitions
        def_result = session.run(
            """
            MATCH (entity)-[:MENTIONED_IN]->(c:Chunk)
            WHERE elementId(c) IN $chunk_ids
            OPTIONAL MATCH (d:Definition)
            WHERE toLower(entity.name) CONTAINS toLower(d.term)
               OR toLower(entity.display_name) CONTAINS toLower(d.term)
            WITH elementId(c) AS chunk_id,
                 collect(DISTINCT d.term)[..5] AS terms
            WHERE terms IS NOT NULL AND size(terms) > 0
            RETURN chunk_id, terms
            """,
            chunk_ids=chunk_ids,
        )
        for record in def_result:
            if record["terms"]:
                definitions_by_chunk[record["chunk_id"]] = record["terms"]

    # Enrich results with graph context
    enriched = []
    for result in base_results:
        chunk_id = result["metadata"].get("chunk_id")
        if chunk_id:
            result["related_entities"] = entities_by_chunk.get(chunk_id, [])
            result["glossary_terms"] = definitions_by_chunk.get(chunk_id, [])
        enriched.append(result)

    logger.info("Graph-enriched search returned %d results", len(enriched))
    return enriched


def get_entities_from_chunks(
    driver: Driver,
    chunk_ids: list[str],
) -> list[dict[str, Any]]:
    """Find entities mentioned in the retrieved chunks.

    Uses the MENTIONED_IN relationship with direction Entity -> Chunk.

    Args:
        driver: Neo4j driver instance.
        chunk_ids: List of chunk element IDs.

    Returns:
        List of entity dictionaries with label, name, display_name, definition, mentions.
    """
    if not chunk_ids:
        return []

    with driver.session() as session:
        result = session.run(
            """
            MATCH (entity)-[:MENTIONED_IN]->(c:Chunk)
            WHERE elementId(c) IN $chunk_ids
            WITH entity, labels(entity)[0] AS label, count(*) AS mentions
            RETURN label,
                   entity.name AS name,
                   entity.display_name AS display_name,
                   entity.definition AS definition,
                   mentions
            ORDER BY mentions DESC, label, name
            LIMIT 20
            """,
            chunk_ids=chunk_ids,
        )
        return [dict(record) for record in result]


def search_entities_by_name(
    driver: Driver,
    search_term: str,
) -> list[dict[str, Any]]:
    """Direct search for entities containing the search term.

    Args:
        driver: Neo4j driver instance.
        search_term: Term to search for in entity names.

    Returns:
        List of matching entities with their connection counts.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n)
            WHERE (n:Concept OR n:Challenge OR n:Bestpractice OR n:Standard
                   OR n:Methodology OR n:Artifact OR n:Tool OR n:Role
                   OR n:Processstage OR n:Industry)
                  AND (toLower(n.name) CONTAINS toLower($term)
                       OR toLower(n.display_name) CONTAINS toLower($term))
            WITH n, labels(n)[0] AS label
            OPTIONAL MATCH (n)-[r]-(related)
            WITH n, label, count(DISTINCT related) AS connections
            RETURN label,
                   n.name AS name,
                   n.display_name AS display_name,
                   n.definition AS definition,
                   connections
            ORDER BY connections DESC
            LIMIT 10
            """,
            term=search_term,
        )
        return [dict(record) for record in result]


def get_related_entities(
    driver: Driver,
    entity_name: str,
) -> list[dict[str, Any]]:
    """Get entities related to a specific entity.

    Args:
        driver: Neo4j driver instance.
        entity_name: Name of the entity to find relationships for.

    Returns:
        List of related entities with relationship information.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n {name: $name})-[r]-(related)
            WHERE NOT related:Chunk AND NOT related:Article
            WITH type(r) AS rel_type,
                 labels(related)[0] AS related_label,
                 related.name AS related_name,
                 related.display_name AS related_display,
                 startNode(r) = n AS outgoing
            RETURN rel_type,
                   CASE WHEN outgoing THEN '->' ELSE '<-' END AS direction,
                   related_label,
                   related_name,
                   related_display
            ORDER BY rel_type, related_label
            """,
            name=entity_name.lower(),
        )
        return [dict(record) for record in result]


@traceable(name="explore_entity", run_type="retriever")
async def explore_entity(
    driver: Driver,
    entity_name: str,
    *,
    include_related: bool = True,
    related_limit: int = 10,
) -> dict[str, Any] | None:
    """Explore a specific entity and its relationships.

    Args:
        driver: Neo4j driver instance.
        entity_name: Name of the entity to explore.
        include_related: Whether to include related entities.
        related_limit: Maximum number of related entities per type.

    Returns:
        Dictionary with entity details and relationships, or None if not found.
    """
    logger.info("Exploring entity: '%s'", entity_name)

    # Find the entity (case-insensitive search)
    with driver.session() as session:
        entity_result = session.run(
            """
            MATCH (e)
            WHERE (e:Concept OR e:Challenge OR e:Bestpractice OR e:Standard
                   OR e:Methodology OR e:Artifact OR e:Tool OR e:Role
                   OR e:Processstage OR e:Industry)
              AND (toLower(e.name) CONTAINS toLower($name)
                   OR toLower(e.display_name) CONTAINS toLower($name))
            RETURN e, labels(e) AS labels
            LIMIT 1
            """,
            name=entity_name,
        )
        result = list(entity_result)

    if not result:
        logger.info("Entity not found: '%s'", entity_name)
        return None

    entity = dict(result[0]["e"])
    labels = result[0]["labels"]

    response: dict[str, Any] = {
        "name": entity.get("name"),
        "display_name": entity.get("display_name"),
        "labels": labels,
        "properties": {k: v for k, v in entity.items() if k not in ("name", "display_name")},
    }

    if include_related:
        with driver.session() as session:
            # Get related entities
            related_result = session.run(
                """
                MATCH (e {name: $name})-[r]-(related)
                WHERE NOT related:Chunk AND NOT related:Article
                RETURN type(r) AS relationship,
                       related.name AS name,
                       related.display_name AS display_name,
                       labels(related) AS labels
                LIMIT $limit
                """,
                name=entity.get("name"),
                limit=related_limit,
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

            # Get chunks that mention this entity (reversed direction)
            chunks_result = session.run(
                """
                MATCH (e {name: $name})-[:MENTIONED_IN]->(c:Chunk)
                MATCH (c)-[:FROM_ARTICLE]->(a:Article)
                RETURN DISTINCT a.article_title AS article,
                       a.url AS url
                LIMIT 5
                """,
                name=entity.get("name"),
            )
            response["mentioned_in"] = [
                {"article": m["article"], "url": m["url"]} for m in chunks_result
            ]

    related_count = len(response.get("related", []))
    logger.info("Entity exploration complete: found %d related entities", related_count)
    return response
