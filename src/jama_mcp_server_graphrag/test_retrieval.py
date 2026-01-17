"""Integration test for retrieval functions against live Neo4j database.

Run with: uv run python -m jama_mcp_server_graphrag.test_retrieval
"""

from __future__ import annotations

import asyncio
import logging
import sys

from dotenv import load_dotenv

from jama_mcp_server_graphrag.config import get_config
from jama_mcp_server_graphrag.core.retrieval import (
    explore_entity,
    graph_enriched_search,
    vector_search,
)
from jama_mcp_server_graphrag.server import create_graph, create_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SEPARATOR = "=" * 60


async def main() -> int:
    """Run retrieval integration tests."""
    load_dotenv()

    logger.info(SEPARATOR)
    logger.info("Retrieval Integration Tests")
    logger.info(SEPARATOR)

    # Setup
    logger.info("\n[Setup] Initializing connections...")
    config = get_config()
    graph = create_graph(config)
    vector_store = create_vector_store(config, graph)

    # Test 1: Vector Search
    logger.info("\n[Test 1] Vector Search")
    logger.info("-" * 40)
    results = await vector_search(vector_store, "requirements traceability", limit=3)
    logger.info("Query: 'requirements traceability'")
    logger.info("Results: %d", len(results))
    for i, r in enumerate(results):
        logger.info("  [%d] %.4f - %s", i + 1, r["score"], r["metadata"].get("title", "N/A"))
    assert len(results) > 0, "Vector search should return results"
    logger.info("PASSED")

    # Test 2: Graph-Enriched Search
    logger.info("\n[Test 2] Graph-Enriched Search")
    logger.info("-" * 40)
    results = await graph_enriched_search(
        graph, vector_store, "ISO 26262 automotive safety", limit=3
    )
    logger.info("Query: 'ISO 26262 automotive safety'")
    logger.info("Results: %d", len(results))
    for i, r in enumerate(results):
        entities = r.get("related_entities", [])[:3]
        logger.info(
            "  [%d] %.4f - %s (entities: %s)",
            i + 1,
            r["score"],
            r["metadata"].get("title", "N/A"),
            entities if entities else "none",
        )
    assert len(results) > 0, "Graph-enriched search should return results"
    logger.info("PASSED")

    # Test 3: Entity Exploration
    logger.info("\n[Test 3] Entity Exploration")
    logger.info("-" * 40)
    entity = await explore_entity(graph, "traceability", include_related=True)
    if entity:
        logger.info("Entity: %s", entity.get("name"))
        logger.info("Labels: %s", entity.get("labels"))
        logger.info("Related: %d entities", len(entity.get("related", [])))
        logger.info("Mentioned in: %d articles", len(entity.get("mentioned_in", [])))
        logger.info("PASSED")
    else:
        logger.warning("Entity 'traceability' not found - checking alternative...")
        entity = await explore_entity(graph, "requirements", include_related=True)
        if entity:
            logger.info("Entity: %s", entity.get("name"))
            logger.info("PASSED (with alternative entity)")
        else:
            logger.warning("SKIPPED - no entities found")

    logger.info("\n%s", SEPARATOR)
    logger.info("All retrieval tests completed!")
    logger.info(SEPARATOR)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
