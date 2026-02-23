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
3. Community Search - Vector search on community summary embeddings
4. Graph-Enriched - Multi-level graph traversal with:
   - Window expansion (NEXT_CHUNK)
   - Entity extraction with properties
   - Semantic relationship traversal (RELATED_TO, ADDRESSES, REQUIRES)
   - Industry-aware context (APPLIES_TO)
   - Media enrichment (Images, Webinars)
"""

from __future__ import annotations

import ast
import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from requirements_graphrag_api.observability import traceable_safe

if TYPE_CHECKING:
    import neo4j
    from neo4j import Driver
    from neo4j_graphrag.retrievers import HybridRetriever, VectorRetriever

    from requirements_graphrag_api.config import AppConfig

logger = structlog.get_logger()

# Minimum meaningful content length for chunk filtering (Issue #202)
_MIN_CONTENT_LENGTH = 20


def _is_meaningful_content(content: str) -> bool:
    """Check if chunk content has enough non-whitespace text to be useful.

    Filters degenerate chunks like bare markdown headings ("#### Traceable")
    that score highly in vector search but contain no information.

    Args:
        content: Chunk text content.

    Returns:
        True if content has enough meaningful text.
    """
    stripped = content.strip()
    # Remove markdown heading markers
    if stripped.startswith("#"):
        stripped = stripped.lstrip("#").strip()
    return len(stripped) >= _MIN_CONTENT_LENGTH


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
    from neo4j_graphrag.retrievers import VectorRetriever

    from requirements_graphrag_api.core.embeddings import VoyageAIEmbeddings

    embedder = VoyageAIEmbeddings(
        model=config.embedding_model,
        input_type="query",
        dimensions=config.embedding_dimensions,
        api_key=config.voyage_api_key,
    )

    return VectorRetriever(
        driver=driver,
        index_name=config.vector_index_name,
        embedder=embedder,
        return_properties=["text", "index"],
    )


def _hybrid_result_formatter(record: neo4j.Record) -> Any:
    """Format HybridRetriever results with element ID and clean text content.

    The default neo4j-graphrag formatter loses the element ID needed for
    article metadata joins. This custom formatter extracts it from the
    record's top-level ``elementId`` key.

    Args:
        record: Neo4j record from HybridRetriever's internal Cypher query.

    Returns:
        RetrieverResultItem with text content and metadata including element ID.
    """
    from neo4j_graphrag.types import RetrieverResultItem

    node = record.get("node")
    text = node.get("text", "") if isinstance(node, dict) else ""
    index = node.get("index") if isinstance(node, dict) else None

    return RetrieverResultItem(
        content=text,
        metadata={
            "score": record.get("score", 0.0),
            "id": record.get("elementId"),
            "index": index,
        },
    )


def create_hybrid_retriever(
    driver: Driver,
    config: AppConfig,
) -> HybridRetriever:
    """Create a HybridRetriever for combined vector + keyword search.

    Uses the neo4j-graphrag library's HybridRetriever with a custom result
    formatter that preserves element IDs for article metadata joins.

    Args:
        driver: Neo4j driver instance.
        config: Application configuration.

    Returns:
        Configured HybridRetriever instance.
    """
    from neo4j_graphrag.retrievers import HybridRetriever

    from requirements_graphrag_api.core.embeddings import VoyageAIEmbeddings

    embedder = VoyageAIEmbeddings(
        model=config.embedding_model,
        input_type="query",
        dimensions=config.embedding_dimensions,
        api_key=config.voyage_api_key,
    )

    return HybridRetriever(
        driver=driver,
        vector_index_name=config.vector_index_name,
        fulltext_index_name=config.fulltext_index_name,
        embedder=embedder,
        return_properties=["text", "index"],
        result_formatter=_hybrid_result_formatter,
    )


@traceable_safe(name="vector_search", run_type="retriever")
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

    # Perform similarity search with scores (sync API, offload to thread)
    results = await asyncio.to_thread(retriever.search, query_text=query, top_k=limit)

    # Get article context for each chunk
    # Note: neo4j-graphrag VectorRetriever returns 'id' not 'element_id'
    chunk_ids = []
    for item in results.items:
        node_id = item.metadata.get("id") if item.metadata else None
        if node_id:
            chunk_ids.append(node_id)

    # Fetch article metadata for chunks
    article_context: dict[str, dict[str, Any]] = {}
    if chunk_ids:

        def _fetch_article_context() -> dict[str, dict[str, Any]]:
            ctx: dict[str, dict[str, Any]] = {}
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
                    ctx[record["chunk_id"]] = {
                        "title": record["title"],
                        "url": record["url"],
                        "article_id": record["article_id"],
                        "chapter": record["chapter"],
                    }
            return ctx

        article_context = await asyncio.to_thread(_fetch_article_context)

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

    # Filter degenerate chunks (bare headings without content — Issue #202)
    formatted = [r for r in formatted if _is_meaningful_content(r["content"])]

    logger.info("Vector search returned %d results", len(formatted))
    return formatted


async def _legacy_hybrid_search(
    retriever: VectorRetriever,
    driver: Driver,
    query: str,
    *,
    limit: int = 6,
    keyword_weight: float = 0.3,
) -> list[dict[str, Any]]:
    """Legacy hybrid search using manual Cypher queries.

    Preserved behind ``use_legacy_hybrid_search`` config flag for rollback.
    Will be removed after one release cycle.

    Note: The keyword leg uses 'chunk_fulltext' (indexed on non-existent
    ``heading`` property), so it effectively returns vector-only results.
    """
    logger.info(
        "Legacy hybrid search: query='%s', limit=%d, keyword_weight=%.2f",
        query,
        limit,
        keyword_weight,
    )

    # Get vector search results
    vector_results = await vector_search(retriever, driver, query, limit=limit * 2)

    # Get keyword search results using fulltext index on chunks
    def _keyword_search() -> list[dict[str, Any]]:
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
                return [
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
            return []

    keyword_results = await asyncio.to_thread(_keyword_search)

    # Filter degenerate chunks from keyword results (Issue #202)
    keyword_results = [r for r in keyword_results if _is_meaningful_content(r.get("content", ""))]

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

    logger.info("Legacy hybrid search returned %d results", len(results))
    return results


@traceable_safe(name="hybrid_search", run_type="retriever")
async def hybrid_search(
    retriever: VectorRetriever,
    driver: Driver,
    query: str,
    *,
    limit: int = 6,
    keyword_weight: float = 0.3,
    hybrid_retriever: HybridRetriever | None = None,
    use_legacy: bool = False,
) -> list[dict[str, Any]]:
    """Perform hybrid search combining vector similarity and keyword matching.

    Uses the neo4j-graphrag HybridRetriever with LINEAR ranker when available,
    falling back to the legacy manual implementation if ``use_legacy`` is True
    or no ``hybrid_retriever`` is provided.

    The HybridRetriever handles score fusion internally using:
        ``score = alpha * vector_normalized + (1-alpha) * fulltext_normalized``
    where ``alpha = 1 - keyword_weight``.

    Post-processing preserves:
    - Article-level deduplication (HybridRetriever only dedupes at chunk level)
    - ``source`` provenance field tagging
    - Article metadata via separate FROM_ARTICLE join

    Args:
        retriever: Configured VectorRetriever instance (used for legacy path).
        driver: Neo4j driver for article metadata queries.
        query: Natural language search query.
        limit: Maximum number of results to return.
        keyword_weight: Weight for keyword results (0-1). Maps to alpha = 1 - keyword_weight.
        hybrid_retriever: Optional HybridRetriever instance for new path.
        use_legacy: Force legacy implementation (config flag for rollback).

    Returns:
        List of results with content, score, source, and metadata.
    """
    # Dispatch to legacy if requested or no hybrid retriever available
    if use_legacy or hybrid_retriever is None:
        return await _legacy_hybrid_search(
            retriever, driver, query, limit=limit, keyword_weight=keyword_weight
        )

    alpha = 1 - keyword_weight
    logger.info(
        "Hybrid search: query='%s', limit=%d, keyword_weight=%.2f, alpha=%.2f",
        query,
        limit,
        keyword_weight,
        alpha,
    )

    # Use HybridRetriever with LINEAR ranker (sync API, offload to thread)
    # Request extra results for article-level dedup filtering
    raw_results = await asyncio.to_thread(
        hybrid_retriever.search,
        query_text=query,
        top_k=limit * 2,
        ranker="linear",
        alpha=alpha,
    )

    # Filter degenerate chunks (bare headings without content — Issue #202)
    items = [item for item in raw_results.items if _is_meaningful_content(item.content or "")]

    # Collect chunk IDs for article metadata join
    chunk_ids = [item.metadata["id"] for item in items if item.metadata.get("id")]

    # Fetch article metadata for chunks
    article_context: dict[str, dict[str, Any]] = {}
    if chunk_ids:

        def _fetch_article_context() -> dict[str, dict[str, Any]]:
            ctx: dict[str, dict[str, Any]] = {}
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
                    ctx[record["chunk_id"]] = {
                        "title": record["title"],
                        "url": record["url"],
                        "article_id": record["article_id"],
                        "chapter": record["chapter"],
                    }
            return ctx

        article_context = await asyncio.to_thread(_fetch_article_context)

    # Article-level deduplication + provenance tagging
    seen_articles: set[str] = set()
    formatted: list[dict[str, Any]] = []

    for item in items:
        chunk_id = item.metadata.get("id")
        score = item.metadata.get("score", 0.0)
        metadata = article_context.get(chunk_id, {}).copy()
        metadata["chunk_id"] = chunk_id
        metadata["chunk_index"] = item.metadata.get("index")

        # Article-level dedup
        article_id = metadata.get("article_id")
        if article_id and article_id in seen_articles:
            continue
        if article_id:
            seen_articles.add(article_id)

        formatted.append(
            {
                "content": item.content or "",
                "score": round(float(score), 4),
                "metadata": metadata,
                "source": "hybrid",
            }
        )

        if len(formatted) >= limit:
            break

    logger.info("Hybrid search returned %d results", len(formatted))
    return formatted


# =============================================================================
# Community Index Check
# =============================================================================


def check_community_index(driver: Driver, index_name: str) -> bool:
    """Check whether the community summary vector index exists.

    Shared helper used by both ``api.py`` lifespan and ``graph.py`` dev
    server to avoid duplicating the ``SHOW VECTOR INDEXES`` query.

    Args:
        driver: Neo4j driver instance.
        index_name: Expected community vector index name.

    Returns:
        True if the index exists, False otherwise.
    """
    try:
        with driver.session() as session:
            index_result = session.run("SHOW VECTOR INDEXES YIELD name")
            index_names = [r["name"] for r in index_result]
            available = index_name in index_names
        if available:
            logger.info("Community index available: %s", index_name)
        else:
            logger.warning(
                "Community index '%s' not found — community search disabled",
                index_name,
            )
        return available
    except Exception as e:
        # Distinguish connection failures from general errors
        try:
            from neo4j.exceptions import ServiceUnavailable

            if isinstance(e, ServiceUnavailable):
                logger.error("Neo4j unavailable during community index check: %s", e)
            else:
                logger.exception("Failed to check community index")
        except ImportError:
            logger.exception("Failed to check community index")
        return False


# =============================================================================
# Community Search
# =============================================================================


@traceable_safe(name="community_search", run_type="retriever")
async def community_search(
    driver: Driver,
    config: AppConfig,
    query: str,
    *,
    limit: int = 3,
    max_members: int = 8,
) -> list[dict[str, Any]]:
    """Search community summaries for thematic/global context.

    Performs vector similarity search on the ``community_summary_embeddings``
    index and fetches top member entities via ``IN_COMMUNITY`` relationships.

    Community results complement chunk-level retrieval by providing
    high-level thematic context (e.g., "What are the main themes in
    requirements management?").

    Args:
        driver: Neo4j driver instance.
        config: Application configuration (provides embedder settings).
        query: Natural language search query.
        limit: Maximum number of community results.
        max_members: Maximum member entities per community.

    Returns:
        List of community dicts with summary, score, community_id, members.
        Empty list if the community index doesn't exist or search fails.
    """
    from neo4j_graphrag.retrievers import VectorRetriever

    from requirements_graphrag_api.core.embeddings import VoyageAIEmbeddings

    logger.info("Community search: query='%s', limit=%d", query, limit)

    # Embedder + retriever construction is NOT wrapped in try/except — if
    # VOYAGE_API_KEY is missing or config is invalid, we fail loud (design
    # decision: no silent fallback for missing credentials).
    embedder = VoyageAIEmbeddings(
        model=config.embedding_model,
        input_type="query",
        dimensions=config.embedding_dimensions,
        api_key=config.voyage_api_key,
    )

    community_retriever = VectorRetriever(
        driver=driver,
        index_name=config.community_index_name,
        embedder=embedder,
        return_properties=["summary", "communityId", "member_count"],
    )

    try:
        results = await asyncio.to_thread(community_retriever.search, query_text=query, top_k=limit)
    except Exception:
        logger.exception("Community search failed")
        return []

    if not results.items:
        logger.info("Community search returned 0 results")
        return []

    # Parse results and collect community IDs for member lookup
    communities: list[dict[str, Any]] = []
    community_ids: list[str] = []

    for item in results.items:
        # VectorRetriever returns content as string repr of dict — use
        # ast.literal_eval (safe: only parses Python literals, same as
        # vector_search above)
        if isinstance(item.content, dict):
            summary = item.content.get("summary", "")
            community_id = item.content.get("communityId", "")
        elif isinstance(item.content, str):
            try:
                parsed = ast.literal_eval(item.content)
                summary = parsed.get("summary", "") if isinstance(parsed, dict) else item.content
                community_id = parsed.get("communityId", "") if isinstance(parsed, dict) else ""
            except (ValueError, SyntaxError):
                logger.debug("Could not parse community content as dict, using raw string")
                summary = item.content
                community_id = ""
        else:
            summary = str(item.content) if item.content else ""
            community_id = ""

        score = item.metadata.get("score", 0.0) if item.metadata else 0.0

        communities.append(
            {
                "summary": summary,
                "score": round(float(score), 4),
                "community_id": community_id,
                "members": [],
            }
        )
        if community_id:
            community_ids.append(community_id)

    # Fetch member entities for all communities in a single query
    if community_ids:

        def _fetch_members() -> dict[str, list[dict[str, str]]]:
            members_by_community: dict[str, list[dict[str, str]]] = {}
            with driver.session() as session:
                result = session.run(
                    """
                    UNWIND $community_ids AS cid
                    MATCH (e)-[:IN_COMMUNITY]->(c:Community {communityId: cid})
                    WITH cid,
                         [l IN labels(e) WHERE NOT l STARTS WITH '__'][0] AS type,
                         e.display_name AS name
                    WHERE type IS NOT NULL
                    WITH cid, type, name
                    ORDER BY type, name
                    WITH cid, collect({name: name, type: type})[..$max_members] AS members
                    RETURN cid, members
                    """,
                    community_ids=community_ids,
                    max_members=max_members,
                )
                for record in result:
                    members_by_community[record["cid"]] = record["members"]
            return members_by_community

        try:
            members_lookup = await asyncio.to_thread(_fetch_members)
            for community in communities:
                cid = community["community_id"]
                if cid in members_lookup:
                    community["members"] = members_lookup[cid]
        except Exception:
            logger.exception("Failed to fetch community members")

    logger.info("Community search returned %d results", len(communities))
    return communities


# =============================================================================
# Graph Enrichment Helper Functions
# =============================================================================


def _enrich_with_window_context(
    driver: Driver,
    chunk_ids: list[str],
    window_size: int = 1,
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
            ORDER BY COUNT {{ (entity)-[:MENTIONED_IN]->() }} DESC
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
            MATCH (entity)-[r:{rel_pattern}]->(related)
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
            OPTIONAL MATCH (a)-[:HAS_WEBINAR]->(web:Webinar)
            WITH c, a,
                 [x IN collect(DISTINCT web.thumbnail_url)
                  WHERE x IS NOT NULL] AS _thumb_urls,
                 collect(DISTINCT {
                     title: web.title,
                     url: web.url,
                     thumbnail_url: web.thumbnail_url
                 })[..$max_items] AS webinars
            OPTIONAL MATCH (a)-[:HAS_IMAGE]->(img:Image)
            WHERE NOT img.url IN _thumb_urls
            WITH c, a,
                 collect(DISTINCT {
                     url: img.url,
                     alt_text: img.alt_text,
                     context: img.context
                 })[..$max_items] AS images,
                 webinars
            OPTIONAL MATCH (a)-[:HAS_VIDEO]->(vid:Video)
            WITH elementId(c) AS chunk_id,
                 images,
                 webinars,
                 collect(DISTINCT {
                     title: vid.title,
                     url: vid.url
                 })[..$max_items] AS videos
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
    seen_relationships: set[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    """Assemble enrichment data into a single result dictionary.

    Args:
        result: Base search result with content, score, metadata.
        enrichment_data: Dictionary containing all enrichment lookups.
        seen_relationships: Optional set tracking relationship keys across results
            for deduplication (Issue #202). Mutated in-place.

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

    # Level 3: Semantic relationships (deduplicate across results — Issue #202)
    if chunk_id in enrichment_data["relationships"]:
        rels = enrichment_data["relationships"][chunk_id]
        if seen_relationships is not None:
            deduped = []
            for rel in rels:
                key = (
                    rel.get("from_entity", ""),
                    rel.get("relationship", ""),
                    rel.get("to_entity", ""),
                )
                if key not in seen_relationships:
                    seen_relationships.add(key)
                    deduped.append(rel)
            rels = deduped
        if rels:
            result["semantic_relationships"] = rels

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


@traceable_safe(name="graph_enriched_search", run_type="retriever")
async def graph_enriched_search(
    retriever: VectorRetriever,
    driver: Driver,
    query: str,
    *,
    limit: int = 6,
    options: GraphEnrichmentOptions | None = None,
    hybrid_retriever: HybridRetriever | None = None,
    use_legacy: bool = False,
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
        retriever: Configured VectorRetriever instance (used for legacy path).
        driver: Neo4j driver for traversal.
        query: Natural language search query.
        limit: Maximum number of base results.
        options: Configuration for enrichment levels. Uses defaults if None.
        hybrid_retriever: Optional HybridRetriever for new hybrid search path.
        use_legacy: Force legacy hybrid search implementation.

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
    base_results = await hybrid_search(
        retriever,
        driver,
        query,
        limit=limit,
        hybrid_retriever=hybrid_retriever,
        use_legacy=use_legacy,
    )

    # Collect all chunk IDs for batch queries
    chunk_ids = [
        r["metadata"].get("chunk_id") for r in base_results if r["metadata"].get("chunk_id")
    ]

    if not chunk_ids:
        return base_results

    # ==========================================================================
    # Run all enrichment queries in parallel via asyncio.gather()
    # Each sync Neo4j call is offloaded to the thread pool to avoid
    # blocking the event loop. Expected latency improvement: 3-5x.
    # ==========================================================================

    async def _noop() -> dict[str, Any]:
        return {}

    (
        window_context,
        entities_by_chunk,
        relationships_by_chunk,
        industry_context,
        media_by_chunk,
        references_by_chunk,
        definitions_by_chunk,
    ) = await asyncio.gather(
        asyncio.to_thread(_enrich_with_window_context, driver, chunk_ids, opts.window_size)
        if opts.enable_window_expansion
        else _noop(),
        asyncio.to_thread(
            _enrich_with_entities,
            driver,
            chunk_ids,
            opts.max_entities_per_chunk,
            opts.include_entity_properties,
        )
        if opts.enable_entity_extraction
        else _noop(),
        asyncio.to_thread(
            _enrich_with_semantic_relationships,
            driver,
            chunk_ids,
            opts.relationship_types,
            opts.max_related_per_entity,
        )
        if opts.enable_semantic_traversal
        else _noop(),
        asyncio.to_thread(_enrich_with_industry_context, driver, chunk_ids)
        if opts.enable_industry_context
        else _noop(),
        asyncio.to_thread(_enrich_with_media, driver, chunk_ids, opts.max_media_items)
        if opts.enable_media_enrichment
        else _noop(),
        asyncio.to_thread(_enrich_with_cross_references, driver, chunk_ids)
        if opts.enable_cross_references
        else _noop(),
        # Always include definitions (they're core to understanding)
        asyncio.to_thread(_enrich_with_definitions, driver, chunk_ids),
    )

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

    # Track seen relationships across results for deduplication (Issue #202)
    seen_rels: set[tuple[str, str, str]] = set()
    enriched = [
        _assemble_enriched_result(result, enrichment_data, seen_rels) for result in base_results
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


async def get_entities_from_chunks(
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

    def _query() -> list[dict[str, Any]]:
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

    return await asyncio.to_thread(_query)


async def search_entities_by_name(
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

    def _query() -> list[dict[str, Any]]:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE (n:Concept OR n:Challenge OR n:Bestpractice OR n:Standard
                       OR n:Methodology OR n:Artifact OR n:Tool OR n:Role
                       OR n:Processstage OR n:Industry OR n:Organization
                       OR n:Outcome)
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

    return await asyncio.to_thread(_query)


async def get_related_entities(
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

    def _query() -> list[dict[str, Any]]:
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

    return await asyncio.to_thread(_query)


@traceable_safe(name="explore_entity", run_type="retriever")
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
    def _find_entity() -> list[Any]:
        with driver.session() as session:
            entity_result = session.run(
                """
                MATCH (e)
                WHERE (e:Concept OR e:Challenge OR e:Bestpractice OR e:Standard
                       OR e:Methodology OR e:Artifact OR e:Tool OR e:Role
                       OR e:Processstage OR e:Industry OR e:Organization
                       OR e:Outcome)
                  AND (toLower(e.name) CONTAINS toLower($name)
                       OR toLower(e.display_name) CONTAINS toLower($name))
                RETURN e, labels(e) AS labels
                LIMIT 1
                """,
                name=entity_name,
            )
            return list(entity_result)

    result = await asyncio.to_thread(_find_entity)

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

        def _get_related() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            with driver.session() as session:
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
                related = [
                    {
                        "name": r["name"],
                        "display_name": r["display_name"],
                        "relationship": r["relationship"],
                        "labels": r["labels"],
                    }
                    for r in related_result
                ]

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
                mentioned = [{"article": m["article"], "url": m["url"]} for m in chunks_result]
                return related, mentioned

        related_list, mentioned_list = await asyncio.to_thread(_get_related)
        response["related"] = related_list
        response["mentioned_in"] = mentioned_list

    related_count = len(response.get("related", []))
    logger.info("Entity exploration complete: found %d related entities", related_count)
    return response
