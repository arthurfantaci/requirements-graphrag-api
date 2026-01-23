"""Neo4j client utilities for GraphRAG.

Provides helper functions for Neo4j driver management following best practices:
- Create driver once, reuse across requests
- Use neo4j+s:// for production (Aura, clusters) - enables TLS and routing
- Verify connectivity at startup
- Small connection pool for serverless (5-10 connections)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neo4j import Driver

    from requirements_graphrag_api.config import AppConfig

logger = logging.getLogger(__name__)


def create_driver(config: AppConfig) -> Driver:
    """Create a Neo4j driver with configuration from AppConfig.

    Uses serverless-optimized settings for connection pool management.

    Args:
        config: Application configuration containing Neo4j credentials.

    Returns:
        Configured Neo4j driver instance.

    Note:
        The caller is responsible for closing the driver when done.
        Best practice is to create once during app lifespan and reuse.
    """
    from neo4j import GraphDatabase

    logger.info(
        "Creating Neo4j driver: %s (pool_size=%d)",
        config.neo4j_uri.split("@")[-1] if "@" in config.neo4j_uri else config.neo4j_uri,
        config.neo4j_max_connection_pool_size,
    )

    return GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_username, config.neo4j_password),
        max_connection_pool_size=config.neo4j_max_connection_pool_size,
        connection_acquisition_timeout=config.neo4j_connection_acquisition_timeout,
    )
