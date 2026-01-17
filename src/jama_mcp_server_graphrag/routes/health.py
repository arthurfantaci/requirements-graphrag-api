"""Health check endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request

if TYPE_CHECKING:
    from langchain_neo4j import Neo4jGraph

router = APIRouter()


@router.get("/health")
async def health_check(request: Request) -> dict[str, Any]:
    """Check the health of the API and its dependencies.

    Returns:
        Health status including Neo4j connectivity.
    """
    graph: Neo4jGraph = request.app.state.graph

    # Check Neo4j connectivity
    try:
        graph.query("RETURN 1 AS connected")
        neo4j_status = "connected"
    except Exception:
        neo4j_status = "disconnected"

    return {
        "status": "healthy" if neo4j_status == "connected" else "degraded",
        "neo4j": neo4j_status,
        "version": "1.0.0",
    }
