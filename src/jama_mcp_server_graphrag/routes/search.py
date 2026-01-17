"""Search endpoints for vector, hybrid, and graph-enriched search.

Updated Data Model (2026-01):
- Uses neo4j Driver directly instead of LangChain Neo4jGraph
- Uses VectorRetriever instead of Neo4jVector
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from jama_mcp_server_graphrag.core import (
    graph_enriched_search,
    hybrid_search,
    vector_search,
)

if TYPE_CHECKING:
    from neo4j import Driver
    from neo4j_graphrag.retrievers import VectorRetriever

router = APIRouter()


class SearchRequest(BaseModel):
    """Request body for search endpoints."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of results",
    )


class HybridSearchRequest(SearchRequest):
    """Request body for hybrid search."""

    keyword_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for keyword vs vector (0.0-1.0)",
    )


class GraphSearchRequest(SearchRequest):
    """Request body for graph-enriched search."""

    traversal_depth: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Graph traversal depth (1-3)",
    )


class SearchResult(BaseModel):
    """A single search result."""

    content: str
    score: float
    metadata: dict[str, Any]
    related_entities: list[str] | None = None
    glossary_terms: list[str] | None = None


class SearchResponse(BaseModel):
    """Response from search endpoints."""

    results: list[SearchResult]
    total: int


@router.post("/search/vector", response_model=SearchResponse)
async def vector_search_endpoint(
    request: Request,
    body: SearchRequest,
) -> dict[str, Any]:
    """Search using semantic vector similarity.

    Performs vector similarity search on chunk embeddings to find
    semantically similar content.

    Args:
        request: FastAPI request object.
        body: Search request body.

    Returns:
        Search results with content and metadata.
    """
    retriever: VectorRetriever = request.app.state.retriever
    driver: Driver = request.app.state.driver

    results = await vector_search(retriever, driver, body.query, limit=body.limit)

    return {
        "results": [SearchResult(**r) for r in results],
        "total": len(results),
    }


@router.post("/search/hybrid", response_model=SearchResponse)
async def hybrid_search_endpoint(
    request: Request,
    body: HybridSearchRequest,
) -> dict[str, Any]:
    """Search using combined vector and keyword matching.

    Combines vector similarity with keyword matching for better
    results when queries contain specific terms.

    Args:
        request: FastAPI request object.
        body: Hybrid search request body.

    Returns:
        Search results with combined scores.
    """
    retriever: VectorRetriever = request.app.state.retriever
    driver: Driver = request.app.state.driver

    results = await hybrid_search(
        retriever,
        driver,
        body.query,
        limit=body.limit,
        keyword_weight=body.keyword_weight,
    )

    return {
        "results": [SearchResult(**r) for r in results],
        "total": len(results),
    }


@router.post("/search/graph", response_model=SearchResponse)
async def graph_search_endpoint(
    request: Request,
    body: GraphSearchRequest,
) -> dict[str, Any]:
    """Search with knowledge graph enrichment.

    Combines vector search with graph traversal to include
    related entities and glossary terms.

    Args:
        request: FastAPI request object.
        body: Graph search request body.

    Returns:
        Search results enriched with graph context.
    """
    retriever: VectorRetriever = request.app.state.retriever
    driver: Driver = request.app.state.driver

    results = await graph_enriched_search(
        retriever,
        driver,
        body.query,
        limit=body.limit,
        traversal_depth=body.traversal_depth,
    )

    return {
        "results": [SearchResult(**r) for r in results],
        "total": len(results),
    }
