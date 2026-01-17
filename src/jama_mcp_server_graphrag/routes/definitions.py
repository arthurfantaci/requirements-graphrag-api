"""Definition endpoints for term lookup and search.

Updated Data Model (2026-01):
- Uses Definition nodes instead of GlossaryTerm nodes
- Uses neo4j Driver directly instead of LangChain Neo4jGraph
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Query, Request

from jama_mcp_server_graphrag.core import list_all_terms, lookup_term, search_terms

if TYPE_CHECKING:
    from neo4j import Driver

router = APIRouter()


@router.get("/definitions/{term}")
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


@router.get("/definitions")
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


# Backward compatibility: alias for glossary routes
@router.get("/glossary/{term}")
async def get_glossary_term(
    request: Request,
    term: str,
    fuzzy: bool = Query(default=True, description="Use fuzzy matching"),
) -> dict[str, Any]:
    """Look up a specific glossary term (alias for definitions endpoint).

    Args:
        request: FastAPI request object.
        term: Term to look up.
        fuzzy: Whether to use fuzzy matching.

    Returns:
        Term definition or 404 if not found.
    """
    return await get_term(request, term, fuzzy)


@router.get("/glossary")
async def list_glossary(
    request: Request,
    q: str | None = Query(default=None, description="Search query"),
    limit: int = Query(default=50, ge=1, le=200, description="Max results"),
) -> dict[str, Any]:
    """List or search glossary terms (alias for definitions endpoint).

    Args:
        request: FastAPI request object.
        q: Optional search query. If provided, searches terms.
        limit: Maximum number of results.

    Returns:
        List of terms with definitions.
    """
    return await list_definitions(request, q, limit)
