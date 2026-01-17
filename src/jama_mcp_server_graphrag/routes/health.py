"""Health check endpoint.

Updated Data Model (2026-01):
- Uses direct Neo4j driver instead of LangChain Neo4jGraph
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request

if TYPE_CHECKING:
    from neo4j import Driver

router = APIRouter()


@router.get("/health")
async def health_check(request: Request) -> dict[str, Any]:
    """Check the health of the API and its dependencies.

    Returns:
        Health status including Neo4j connectivity.
    """
    driver: Driver | None = getattr(request.app.state, "driver", None)

    # Check Neo4j connectivity
    neo4j_status = "not configured"
    if driver:
        try:
            with driver.session() as session:
                session.run("RETURN 1 AS connected")
            neo4j_status = "connected"
        except Exception:
            neo4j_status = "disconnected"

    return {
        "status": "healthy" if neo4j_status == "connected" else "degraded",
        "neo4j": neo4j_status,
        "version": "1.0.0",
    }
