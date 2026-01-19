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


class EntityInfo(BaseModel):
    """Entity information from graph enrichment."""

    name: str | None = None
    type: str | None = None
    definition: str | None = None
    benefit: str | None = None
    impact: str | None = None


class GlossaryDefinition(BaseModel):
    """Glossary definition from graph enrichment."""

    term: str | None = None
    definition: str | None = None
    url: str | None = None


class SemanticRelationship(BaseModel):
    """Semantic relationship between entities."""

    from_entity: str | None = None
    relationship: str | None = None
    to_entity: str | None = None
    to_type: str | None = None
    to_definition: str | None = None


class ContextWindow(BaseModel):
    """Adjacent chunk context."""

    prev_context: str | None = None
    next_context: str | None = None


class ImageContent(BaseModel):
    """Image content from source articles."""

    url: str
    alt_text: str = ""
    context: str = ""


class WebinarContent(BaseModel):
    """Webinar content from source articles."""

    title: str | None = None
    url: str | None = None


class VideoContent(BaseModel):
    """Video content from source articles."""

    title: str | None = None
    url: str | None = None


class MediaContent(BaseModel):
    """Media content from source articles."""

    images: list[ImageContent] = []
    webinars: list[WebinarContent] = []
    videos: list[VideoContent] = []


class SearchResult(BaseModel):
    """A single search result with graph enrichment."""

    content: str
    score: float
    metadata: dict[str, Any]
    # Level 1: Window context
    context_window: ContextWindow | None = None
    # Level 2: Entities with properties
    entities: list[EntityInfo] = []
    # Level 3: Semantic relationships
    semantic_relationships: list[SemanticRelationship] = []
    # Level 4: Domain context
    industry_standards: list[dict[str, Any]] = []
    media: MediaContent | None = None
    related_articles: list[dict[str, Any]] = []
    glossary_definitions: list[GlossaryDefinition] = []


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
    body: SearchRequest,
) -> dict[str, Any]:
    """Search with knowledge graph enrichment.

    Combines hybrid search (vector + keyword) with multi-level graph
    traversal to provide rich context:
    - Level 1: Window expansion (adjacent chunks)
    - Level 2: Entity extraction with properties
    - Level 3: Semantic relationship traversal
    - Level 4: Industry standards, media, cross-references

    Args:
        request: FastAPI request object.
        body: Search request body.

    Returns:
        Search results enriched with comprehensive graph context.
    """
    retriever: VectorRetriever = request.app.state.retriever
    driver: Driver = request.app.state.driver

    results = await graph_enriched_search(
        retriever,
        driver,
        body.query,
        limit=body.limit,
    )

    return {
        "results": [SearchResult(**r) for r in results],
        "total": len(results),
    }
