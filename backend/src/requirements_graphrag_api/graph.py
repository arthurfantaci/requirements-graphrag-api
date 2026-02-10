"""LangGraph dev server entry point.

Constructs and exports the compiled orchestrator graph for use with
the LangGraph development server (`langgraph dev`). Dependencies are
initialized from environment variables at import time.

Usage:
    cd backend
    langgraph dev --config langgraph.json
"""

from __future__ import annotations

import logging

from neo4j import GraphDatabase

from requirements_graphrag_api.config import get_config
from requirements_graphrag_api.core.agentic.orchestrator import create_orchestrator_graph
from requirements_graphrag_api.core.retrieval import create_vector_retriever

logger = logging.getLogger(__name__)

# Load config from environment (.env is auto-loaded by config module)
config = get_config()

# Initialize Neo4j driver
driver = GraphDatabase.driver(
    config.neo4j_uri,
    auth=(config.neo4j_username, config.neo4j_password),
    max_connection_pool_size=config.neo4j_max_connection_pool_size,
    connection_acquisition_timeout=config.neo4j_connection_acquisition_timeout,
    liveness_check_timeout=config.neo4j_liveness_check_timeout,
    max_connection_lifetime=config.neo4j_max_connection_lifetime,
)

# Initialize VectorRetriever
retriever = create_vector_retriever(driver, config)

# Compile the orchestrator graph (no checkpointer for dev server)
graph = create_orchestrator_graph(config, driver, retriever)

logger.info("LangGraph dev server: orchestrator graph compiled")
