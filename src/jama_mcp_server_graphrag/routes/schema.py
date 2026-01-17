"""Schema endpoint for exploring the knowledge graph structure.

Updated Data Model (2026-01):
- Uses neo4j Driver directly instead of LangChain Neo4jGraph
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request

from jama_mcp_server_graphrag.core import explore_entity

if TYPE_CHECKING:
    from neo4j import Driver

router = APIRouter()


@router.get("/schema")
async def get_schema(request: Request) -> dict[str, Any]:
    """Get the knowledge graph schema.

    Returns node labels, relationship types, and their counts
    to help understand the graph structure.

    Args:
        request: FastAPI request object.

    Returns:
        Schema information including node and relationship counts.
    """
    driver: Driver = request.app.state.driver

    with driver.session() as session:
        # Get node label counts
        node_result = session.run(
            """
            MATCH (n)
            WITH labels(n) AS labels
            UNWIND labels AS label
            RETURN label, count(*) AS count
            ORDER BY count DESC
            """
        )
        node_counts = [dict(record) for record in node_result]

        # Get relationship type counts
        rel_result = session.run(
            """
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(*) AS count
            ORDER BY count DESC
            LIMIT 20
            """
        )
        rel_counts = [dict(record) for record in rel_result]

    return {
        "node_labels": node_counts,
        "relationship_types": rel_counts,
    }


@router.get("/schema/entity/{name}")
async def get_entity(
    request: Request,
    name: str,
) -> dict[str, Any]:
    """Explore a specific entity in the knowledge graph.

    Returns detailed information about an entity including
    its relationships and mentions.

    Args:
        request: FastAPI request object.
        name: Entity name to explore.

    Returns:
        Entity details including relationships.
    """
    driver: Driver = request.app.state.driver

    result = await explore_entity(driver, name, include_related=True)

    if result is None:
        return {"name": name, "found": False, "message": "Entity not found"}

    return {**result, "found": True}
