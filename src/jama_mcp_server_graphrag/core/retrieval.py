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
3. Graph-Enriched - Multi-level graph traversal with:
   - Window expansion (NEXT_CHUNK)
   - Entity extraction with properties
   - Semantic relationship traversal (RELATED_TO, ADDRESSES, REQUIRES)
   - Industry-aware context (APPLIES_TO)
   - Media enrichment (Images, Webinars)
"""

from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jama_mcp_server_graphrag.observability import traceable

if TYPE_CHECKING:
    from neo4j import Driver
    from neo4j_graphrag.retrievers import VectorRetriever

    from jama_mcp_server_graphrag.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GraphEnrichmentOptions:
    """Configuration options for graph enrichment levels.

    Controls which enrichment features are enabled and their limits.
    All options are enabled by default for maximum context.
    """

    # Level 1: Window expansion
    enable_window_expansion: bool = True
    window_size: int = 1  # Number of adjacent chunks to include

    # Level 2: Entity extraction
    enable_entity_extraction: bool = True
    max_entities_per_chunk: int = 10
    include_entity_properties: bool = True  # Include definition, benefit, impact

    # Level 3: Semantic relationships
    enable_semantic_traversal: bool = True
    max_related_per_entity: int = 5
    relationship_types: tuple[str, ...] = ("RELATED_TO", "ADDRESSES", "REQUIRES", "COMPONENT_OF")

    # Level 4: Domain context
    enable_industry_context: bool = True
    enable_media_enrichment: bool = True
    enable_cross_references: bool = True
    max_media_items: int = 3


# Default options instance
DEFAULT_ENRICHMENT_OPTIONS = GraphEnrichmentOptions()


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
    # Note: neo4j-graphrag VectorRetriever returns 'id' not 'element_id'
    chunk_ids = []
    for item in results.items:
        node_id = item.metadata.get("id") if item.metadata else None
        if node_id:
            chunk_ids.append(node_id)

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
        # neo4j-graphrag VectorRetriever returns 'id' not 'element_id'
        node_id = item.metadata.get("id") if item.metadata else None
        score = item.metadata.get("score", 0.0) if item.metadata else 0.0

        # Extract text content
        # VectorRetriever may return content as dict or as string representation of dict
        if isinstance(item.content, dict):
            content = item.content.get("text", "")
        elif isinstance(item.content, str):
            # Content might be a string like "{'index': 3, 'text': '...'}"
            # Try to extract text from it
            try:
                parsed = ast.literal_eval(item.content)
                content = parsed.get("text", "") if isinstance(parsed, dict) else item.content
            except (ValueError, SyntaxError):
                content = item.content
        else:
            content = str(item.content) if item.content else ""

        # Build metadata
        metadata = article_context.get(node_id, {})
        metadata["chunk_id"] = node_id
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


# =============================================================================
# Graph Enrichment Helper Functions
# =============================================================================


def _enrich_with_window_context(
    driver: Driver,
    chunk_ids: list[str],
    window_size: int = 1,  # noqa: ARG001 - reserved for multi-hop expansion
) -> dict[str, dict[str, str | None]]:
    """Expand context window using NEXT_CHUNK relationships.

    Retrieves adjacent chunks to provide surrounding context for each
    retrieved chunk. This helps when chunks are split mid-paragraph.

    Args:
        driver: Neo4j driver instance.
        chunk_ids: List of chunk element IDs.
        window_size: Number of chunks before/after to include.

    Returns:
        Dictionary mapping chunk_id to {prev_context, next_context}.
    """
    if not chunk_ids:
        return {}

    window_context: dict[str, dict[str, str | None]] = {}

    with driver.session() as session:
        result = session.run(
            """
            UNWIND $chunk_ids AS cid
            MATCH (c:Chunk) WHERE elementId(c) = cid
            OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(c)
            OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(next:Chunk)
            RETURN cid AS chunk_id,
                   prev.text AS prev_context,
                   next.text AS next_context
            """,
            chunk_ids=chunk_ids,
        )
        for record in result:
            window_context[record["chunk_id"]] = {
                "prev_context": record["prev_context"],
                "next_context": record["next_context"],
            }

    logger.debug("Window expansion: enriched %d chunks", len(window_context))
    return window_context


def _enrich_with_entities(
    driver: Driver,
    chunk_ids: list[str],
    max_entities: int = 10,
    include_properties: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Extract entities mentioned in chunks with their properties.

    Uses MENTIONED_IN relationship (Entity -> Chunk direction) and
    returns entity properties including definition, benefit, and impact.

    Args:
        driver: Neo4j driver instance.
        chunk_ids: List of chunk element IDs.
        max_entities: Maximum entities per chunk.
        include_properties: Whether to include definition/benefit/impact.

    Returns:
        Dictionary mapping chunk_id to list of entity dictionaries.
    """
    if not chunk_ids:
        return {}

    entities_by_chunk: dict[str, list[dict[str, Any]]] = {}

    # Build return clause based on whether we want properties
    if include_properties:
        return_clause = """
            collect(DISTINCT {
                name: entity.display_name,
                type: labels(entity)[0],
                definition: entity.definition,
                benefit: entity.benefit,
                impact: entity.impact
            })[..$max_entities] AS entities
        """
    else:
        return_clause = """
            collect(DISTINCT entity.display_name)[..$max_entities] AS entities
        """

    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (entity)-[:MENTIONED_IN]->(c:Chunk)
            WHERE elementId(c) IN $chunk_ids
            WITH elementId(c) AS chunk_id, entity
            ORDER BY size((entity)-[:MENTIONED_IN]->()) DESC
            WITH chunk_id, collect(entity)[..$max_entities] AS top_entities
            UNWIND top_entities AS entity
            WITH chunk_id,
                 {return_clause}
            RETURN chunk_id, entities
            """,
            chunk_ids=chunk_ids,
            max_entities=max_entities,
        )
        for record in result:
            entities_by_chunk[record["chunk_id"]] = record["entities"] or []

    logger.debug("Entity extraction: found entities for %d chunks", len(entities_by_chunk))
    return entities_by_chunk


def _enrich_with_semantic_relationships(
    driver: Driver,
    chunk_ids: list[str],
    relationship_types: tuple[str, ...],
    max_related: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Traverse semantic relationships from entities in chunks.

    Follows relationships like RELATED_TO, ADDRESSES, REQUIRES to find
    related concepts, challenges addressed, and dependencies.

    Args:
        driver: Neo4j driver instance.
        chunk_ids: List of chunk element IDs.
        relationship_types: Tuple of relationship types to traverse.
        max_related: Maximum related entities per chunk.

    Returns:
        Dictionary mapping chunk_id to list of related entity info.
    """
    if not chunk_ids or not relationship_types:
        return {}

    relationships_by_chunk: dict[str, list[dict[str, Any]]] = {}

    # Build dynamic relationship pattern
    rel_pattern = "|".join(relationship_types)

    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (entity)-[:MENTIONED_IN]->(c:Chunk)
            WHERE elementId(c) IN $chunk_ids
            OPTIONAL MATCH (entity)-[r:{rel_pattern}]->(related)
            WHERE related IS NOT NULL
            WITH elementId(c) AS chunk_id,
                 collect(DISTINCT {{
                     from_entity: entity.display_name,
                     relationship: type(r),
                     to_entity: related.display_name,
                     to_type: labels(related)[0],
                     to_definition: related.definition
                 }})[..$max_related] AS relationships
            WHERE size(relationships) > 0
            RETURN chunk_id, relationships
            """,
            chunk_ids=chunk_ids,
            max_related=max_related,
        )
        for record in result:
            if record["relationships"]:
                relationships_by_chunk[record["chunk_id"]] = record["relationships"]

    logger.debug(
        "Semantic traversal: found relationships for %d chunks",
        len(relationships_by_chunk),
    )
    return relationships_by_chunk


def _enrich_with_industry_context(
    driver: Driver,
    chunk_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Find industry-specific standards that apply to entities in chunks.

    Uses APPLIES_TO relationship to surface relevant industry standards
    when industry or standard entities are mentioned.

    Args:
        driver: Neo4j driver instance.
        chunk_ids: List of chunk element IDs.

    Returns:
        Dictionary mapping chunk_id to list of industry/standard info.
    """
    if not chunk_ids:
        return {}

    industry_context: dict[str, list[dict[str, Any]]] = {}

    with driver.session() as session:
        result = session.run(
            """
            MATCH (entity)-[:MENTIONED_IN]->(c:Chunk)
            WHERE elementId(c) IN $chunk_ids
              AND (entity:Industry OR entity:Standard)
            OPTIONAL MATCH (standard:Standard)-[:APPLIES_TO]->(industry:Industry)
            WHERE standard = entity OR industry = entity
            WITH elementId(c) AS chunk_id,
                 collect(DISTINCT {
                     industry: industry.display_name,
                     standard: standard.display_name,
                     organization: standard.organization,
                     standard_definition: standard.definition
                 })[..5] AS context
            WHERE size(context) > 0
            RETURN chunk_id, context
            """,
            chunk_ids=chunk_ids,
        )
        for record in result:
            if record["context"]:
                industry_context[record["chunk_id"]] = record["context"]

    logger.debug("Industry context: found for %d chunks", len(industry_context))
    return industry_context


def _enrich_with_media(
    driver: Driver,
    chunk_ids: list[str],
    max_items: int = 3,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Find related media (images, webinars, videos) from source articles.

    Args:
        driver: Neo4j driver instance.
        chunk_ids: List of chunk element IDs.
        max_items: Maximum media items per type.

    Returns:
        Dictionary mapping chunk_id to {images, webinars, videos}.
    """
    if not chunk_ids:
        return {}

    media_by_chunk: dict[str, dict[str, list[dict[str, Any]]]] = {}

    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Chunk)-[:FROM_ARTICLE]->(a:Article)
            WHERE elementId(c) IN $chunk_ids
            OPTIONAL MATCH (a)-[:HAS_IMAGE]->(img:Image)
            OPTIONAL MATCH (a)-[:HAS_WEBINAR]->(web:Webinar)
            OPTIONAL MATCH (a)-[:HAS_VIDEO]->(vid:Video)
            WITH elementId(c) AS chunk_id,
                 collect(DISTINCT {
                     url: img.url,
                     alt_text: img.alt_text,
                     context: img.context
                 })[..$max_items] AS images,
                 collect(DISTINCT {title: web.title, url: web.url})[..$max_items] AS webinars,
                 collect(DISTINCT {title: vid.title, url: vid.url})[..$max_items] AS videos
            RETURN chunk_id, images, webinars, videos
            """,
            chunk_ids=chunk_ids,
            max_items=max_items,
        )
        for record in result:
            # Filter out null entries from collections
            images = [i for i in (record["images"] or []) if i.get("url")]
            webinars = [w for w in (record["webinars"] or []) if w.get("url")]
            videos = [v for v in (record["videos"] or []) if v.get("url")]

            if images or webinars or videos:
                media_by_chunk[record["chunk_id"]] = {
                    "images": images,
                    "webinars": webinars,
                    "videos": videos,
                }

    logger.debug("Media enrichment: found for %d chunks", len(media_by_chunk))
    return media_by_chunk


def _enrich_with_cross_references(
    driver: Driver,
    chunk_ids: list[str],
) -> dict[str, list[dict[str, str]]]:
    """Find cross-referenced articles via REFERENCES relationship.

    Args:
        driver: Neo4j driver instance.
        chunk_ids: List of chunk element IDs.

    Returns:
        Dictionary mapping chunk_id to list of referenced articles.
    """
    if not chunk_ids:
        return {}

    references_by_chunk: dict[str, list[dict[str, str]]] = {}

    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Chunk)-[:FROM_ARTICLE]->(a:Article)-[:REFERENCES]->(ref:Article)
            WHERE elementId(c) IN $chunk_ids
            WITH elementId(c) AS chunk_id,
                 collect(DISTINCT {
                     title: ref.article_title,
                     url: ref.url,
                     chapter: ref.chapter_title
                 })[..5] AS references
            WHERE size(references) > 0
            RETURN chunk_id, references
            """,
            chunk_ids=chunk_ids,
        )
        for record in result:
            if record["references"]:
                references_by_chunk[record["chunk_id"]] = record["references"]

    logger.debug("Cross-references: found for %d chunks", len(references_by_chunk))
    return references_by_chunk


def _enrich_with_definitions(
    driver: Driver,
    chunk_ids: list[str],
) -> dict[str, list[dict[str, str]]]:
    """Find glossary definitions related to entities in chunks.

    Uses direct Definition node lookup based on entity names.

    Args:
        driver: Neo4j driver instance.
        chunk_ids: List of chunk element IDs.

    Returns:
        Dictionary mapping chunk_id to list of definitions.
    """
    if not chunk_ids:
        return {}

    definitions_by_chunk: dict[str, list[dict[str, str]]] = {}

    with driver.session() as session:
        result = session.run(
            """
            MATCH (entity)-[:MENTIONED_IN]->(c:Chunk)
            WHERE elementId(c) IN $chunk_ids
            WITH elementId(c) AS chunk_id, collect(DISTINCT entity.name) AS entity_names
            UNWIND entity_names AS ename
            MATCH (d:Definition)
            WHERE toLower(d.term) = toLower(ename)
               OR toLower(ename) CONTAINS toLower(d.term)
            WITH chunk_id,
                 collect(DISTINCT {
                     term: d.term,
                     definition: d.definition,
                     url: d.url
                 })[..5] AS definitions
            WHERE size(definitions) > 0
            RETURN chunk_id, definitions
            """,
            chunk_ids=chunk_ids,
        )
        for record in result:
            if record["definitions"]:
                definitions_by_chunk[record["chunk_id"]] = record["definitions"]

    logger.debug("Definitions: found for %d chunks", len(definitions_by_chunk))
    return definitions_by_chunk


def _assemble_enriched_result(
    result: dict[str, Any],
    enrichment_data: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Assemble enrichment data into a single result dictionary.

    Args:
        result: Base search result with content, score, metadata.
        enrichment_data: Dictionary containing all enrichment lookups.

    Returns:
        Enriched result with all graph context attached.
    """
    chunk_id = result["metadata"].get("chunk_id")
    if not chunk_id:
        return result

    # Level 1: Window context
    if chunk_id in enrichment_data["window"]:
        result["context_window"] = enrichment_data["window"][chunk_id]

    # Level 2: Entities with properties
    result["entities"] = enrichment_data["entities"].get(chunk_id, [])

    # Level 3: Semantic relationships
    if chunk_id in enrichment_data["relationships"]:
        result["semantic_relationships"] = enrichment_data["relationships"][chunk_id]

    # Level 4: Domain context
    if chunk_id in enrichment_data["industry"]:
        result["industry_standards"] = enrichment_data["industry"][chunk_id]

    if chunk_id in enrichment_data["media"]:
        result["media"] = enrichment_data["media"][chunk_id]

    if chunk_id in enrichment_data["references"]:
        result["related_articles"] = enrichment_data["references"][chunk_id]

    if chunk_id in enrichment_data["definitions"]:
        result["glossary_definitions"] = enrichment_data["definitions"][chunk_id]

    return result


# =============================================================================
# Main Graph-Enriched Search Function
# =============================================================================


@traceable(name="graph_enriched_search", run_type="retriever")
async def graph_enriched_search(
    retriever: VectorRetriever,
    driver: Driver,
    query: str,
    *,
    limit: int = 6,
    options: GraphEnrichmentOptions | None = None,
) -> list[dict[str, Any]]:
    """Perform hybrid search enriched with multi-level graph context.

    Combines hybrid search (vector + keyword) with comprehensive graph
    traversal to provide rich context for RAG:

    Level 1: Window Expansion (NEXT_CHUNK)
        - Adds previous/next chunk text for context continuity

    Level 2: Entity Extraction
        - Entities mentioned in chunks with properties (definition, benefit, impact)

    Level 3: Semantic Relationship Traversal
        - RELATED_TO: Related concepts
        - ADDRESSES: Challenges that concepts address
        - REQUIRES: Dependencies
        - COMPONENT_OF: Parent concepts

    Level 4: Domain Context
        - Industry standards (APPLIES_TO)
        - Media content (images, webinars, videos)
        - Cross-article references (REFERENCES)
        - Glossary definitions

    Args:
        retriever: Configured VectorRetriever instance.
        driver: Neo4j driver for traversal.
        query: Natural language search query.
        limit: Maximum number of base results.
        options: Configuration for enrichment levels. Uses defaults if None.

    Returns:
        List of enriched results with content, score, metadata, and graph context.
    """
    opts = options or DEFAULT_ENRICHMENT_OPTIONS

    logger.info(
        "Graph-enriched search: query='%s', limit=%d, window=%s, semantic=%s",
        query,
        limit,
        opts.enable_window_expansion,
        opts.enable_semantic_traversal,
    )

    # Get base results using hybrid search (vector + keyword) for better recall
    base_results = await hybrid_search(retriever, driver, query, limit=limit)

    # Collect all chunk IDs for batch queries
    chunk_ids = [
        r["metadata"].get("chunk_id") for r in base_results if r["metadata"].get("chunk_id")
    ]

    if not chunk_ids:
        return base_results

    # ==========================================================================
    # Level 1: Window Expansion
    # ==========================================================================
    window_context: dict[str, dict[str, str | None]] = {}
    if opts.enable_window_expansion:
        window_context = _enrich_with_window_context(
            driver, chunk_ids, window_size=opts.window_size
        )

    # ==========================================================================
    # Level 2: Entity Extraction with Properties
    # ==========================================================================
    entities_by_chunk: dict[str, list[dict[str, Any]]] = {}
    if opts.enable_entity_extraction:
        entities_by_chunk = _enrich_with_entities(
            driver,
            chunk_ids,
            max_entities=opts.max_entities_per_chunk,
            include_properties=opts.include_entity_properties,
        )

    # ==========================================================================
    # Level 3: Semantic Relationship Traversal
    # ==========================================================================
    relationships_by_chunk: dict[str, list[dict[str, Any]]] = {}
    if opts.enable_semantic_traversal:
        relationships_by_chunk = _enrich_with_semantic_relationships(
            driver,
            chunk_ids,
            relationship_types=opts.relationship_types,
            max_related=opts.max_related_per_entity,
        )

    # ==========================================================================
    # Level 4: Domain Context
    # ==========================================================================
    industry_context: dict[str, list[dict[str, Any]]] = {}
    if opts.enable_industry_context:
        industry_context = _enrich_with_industry_context(driver, chunk_ids)

    media_by_chunk: dict[str, dict[str, list[dict[str, Any]]]] = {}
    if opts.enable_media_enrichment:
        media_by_chunk = _enrich_with_media(
            driver, chunk_ids, max_items=opts.max_media_items
        )

    references_by_chunk: dict[str, list[dict[str, str]]] = {}
    if opts.enable_cross_references:
        references_by_chunk = _enrich_with_cross_references(driver, chunk_ids)

    # Always include definitions (they're core to understanding)
    definitions_by_chunk = _enrich_with_definitions(driver, chunk_ids)

    # ==========================================================================
    # Assemble Enriched Results
    # ==========================================================================
    # Bundle all enrichment data for assembly
    enrichment_data = {
        "window": window_context,
        "entities": entities_by_chunk,
        "relationships": relationships_by_chunk,
        "industry": industry_context,
        "media": media_by_chunk,
        "references": references_by_chunk,
        "definitions": definitions_by_chunk,
    }

    enriched = [
        _assemble_enriched_result(result, enrichment_data)
        for result in base_results
    ]

    # Log enrichment summary
    enrichment_stats = {
        "window": len(window_context),
        "entities": len(entities_by_chunk),
        "relationships": len(relationships_by_chunk),
        "industry": len(industry_context),
        "media": len(media_by_chunk),
        "references": len(references_by_chunk),
        "definitions": len(definitions_by_chunk),
    }
    logger.info(
        "Graph-enriched search returned %d results with enrichment: %s",
        len(enriched),
        enrichment_stats,
    )
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
