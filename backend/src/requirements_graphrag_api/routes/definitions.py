"""Definition endpoints for term lookup and search.

Updated Data Model (2026-01):
- Uses Definition nodes instead of GlossaryTerm nodes
- Uses neo4j Driver directly instead of LangChain Neo4jGraph
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict

from requirements_graphrag_api.core import list_all_terms, lookup_term, search_terms

if TYPE_CHECKING:
    from neo4j import Driver

router = APIRouter()


class TermResponse(BaseModel):
    """Response from single term lookup."""

    model_config = ConfigDict(extra="allow")

    term: str
    definition: str
    acronym: str | None = None
    url: str | None = None
    term_id: str | None = None
    score: float


class TermSummary(BaseModel):
    """Summary of a definition term."""

    model_config = ConfigDict(extra="allow")

    term: str
    definition: str
    acronym: str | None = None
    url: str | None = None
    score: float | None = None


class TermListResponse(BaseModel):
    """Response from term listing/search."""

    model_config = ConfigDict(extra="allow")

    terms: list[TermSummary]
    total: int


@router.get("/definitions/{term}", response_model=TermResponse)
async def get_term(
    request: Request,
    term: str,
    fuzzy: bool = Query(default=True, description="Use fuzzy matching"),
) -> dict[str, Any]:
    """Look up a specific definition term.

    Args:
        request: FastAPI request object.
        term: Term to look up.
        fuzzy: Whether to use fuzzy matching.

    Returns:
        Term definition or 404 if not found.
    """
    driver: Driver = request.app.state.driver

    result = await lookup_term(driver, term, fuzzy=fuzzy)

    if result is None:
        raise HTTPException(status_code=404, detail=f"Term '{term}' not found")

    return result


@router.get("/definitions", response_model=TermListResponse)
async def list_definitions(
    request: Request,
    q: str | None = Query(default=None, description="Search query"),
    limit: int = Query(default=50, ge=1, le=200, description="Max results"),
) -> dict[str, Any]:
    """List or search definition terms.

    Args:
        request: FastAPI request object.
        q: Optional search query. If provided, searches terms.
        limit: Maximum number of results.

    Returns:
        List of terms with definitions.
    """
    driver: Driver = request.app.state.driver

    if q:
        terms = await search_terms(driver, q, limit=limit)
    else:
        terms = await list_all_terms(driver, limit=limit)

    return {
        "terms": terms,
        "total": len(terms),
    }
