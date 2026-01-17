"""Standards endpoints for industry standards lookup.

Updated Data Model (2026-01):
- Uses neo4j Driver directly instead of LangChain Neo4jGraph
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Query, Request

from jama_mcp_server_graphrag.core import (
    get_standards_by_industry,
    list_all_standards,
    lookup_standard,
    search_standards,
)

if TYPE_CHECKING:
    from neo4j import Driver

router = APIRouter()


@router.get("/standards/{name}")
async def get_standard(
    request: Request,
    name: str,
    include_related: bool = Query(
        default=True, description="Include related entities"
    ),
) -> dict[str, Any]:
    """Look up a specific standard.

    Args:
        request: FastAPI request object.
        name: Standard name (e.g., "ISO 26262").
        include_related: Whether to include related entities.

    Returns:
        Standard details or 404 if not found.
    """
    driver: Driver = request.app.state.driver

    result = await lookup_standard(driver, name, include_related=include_related)

    if result is None:
        raise HTTPException(status_code=404, detail=f"Standard '{name}' not found")

    return result


@router.get("/standards")
async def list_standards(
    request: Request,
    q: str | None = Query(default=None, description="Search query"),
    industry: str | None = Query(default=None, description="Filter by industry"),
    limit: int = Query(default=50, ge=1, le=200, description="Max results"),
) -> dict[str, Any]:
    """List or search standards.

    Args:
        request: FastAPI request object.
        q: Optional search query.
        industry: Optional industry filter.
        limit: Maximum number of results.

    Returns:
        List of standards.
    """
    driver: Driver = request.app.state.driver

    if industry and not q:
        # Get standards by industry
        standards = await get_standards_by_industry(driver, industry, limit=limit)
    elif q:
        # Search standards
        standards = await search_standards(driver, q, industry=industry, limit=limit)
    else:
        # List all standards
        standards = await list_all_standards(driver, limit=limit)

    return {
        "standards": standards,
        "total": len(standards),
    }


@router.get("/standards/industry/{industry}")
async def get_by_industry(
    request: Request,
    industry: str,
    limit: int = Query(default=10, ge=1, le=50, description="Max results"),
) -> dict[str, Any]:
    """Get standards for a specific industry.

    Supported industries: automotive, medical, aerospace, defense, rail.

    Args:
        request: FastAPI request object.
        industry: Industry name.
        limit: Maximum number of results.

    Returns:
        List of standards applicable to the industry.
    """
    driver: Driver = request.app.state.driver

    standards = await get_standards_by_industry(driver, industry, limit=limit)

    return {
        "industry": industry,
        "standards": standards,
        "total": len(standards),
    }
