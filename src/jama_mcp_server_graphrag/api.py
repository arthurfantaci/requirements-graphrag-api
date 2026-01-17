"""FastAPI REST API entry point for React frontend integration.

This module provides REST endpoints that wrap the same core GraphRAG
logic used by the MCP server, enabling a React chatbot frontend.

Updated Data Model (2026-01):
- Uses neo4j Driver directly with neo4j-graphrag VectorRetriever
- Chunks contain text directly, linked via FROM_ARTICLE
- MENTIONED_IN relationship direction: Entity -> Chunk
- Definition nodes replace GlossaryTerm

Usage:
    # Development
    uv run uvicorn jama_mcp_server_graphrag.api:app --reload

    # Production
    uv run uvicorn jama_mcp_server_graphrag.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from neo4j import GraphDatabase

from jama_mcp_server_graphrag.config import get_config
from jama_mcp_server_graphrag.core import create_vector_retriever
from jama_mcp_server_graphrag.observability import configure_tracing
from jama_mcp_server_graphrag.routes import chat, definitions, health, schema, search, standards

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle - initialize and cleanup resources.

    Applies the same Neo4j best practices as the MCP server:
    - Create resources once at startup, reuse across requests
    - Verify connectivity immediately (fail fast)
    - Clean up on shutdown

    Args:
        app: FastAPI application instance.

    Yields:
        None after initializing resources.
    """
    logger.info("Starting Jama GraphRAG API")

    config = get_config()

    # Configure LangSmith tracing if enabled
    if configure_tracing(config):
        logger.info("LangSmith tracing configured for project: %s", config.langsmith_project)

    # Log configuration (without sensitive data)
    logger.info(
        "Configuration loaded: database=%s, model=%s, embedding=%s",
        config.neo4j_database,
        config.chat_model,
        config.embedding_model,
    )

    # Initialize Neo4j driver
    logger.info("Connecting to Neo4j at %s", config.neo4j_uri.split("@")[-1])
    driver = GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_username, config.neo4j_password),
        max_connection_pool_size=config.neo4j_max_connection_pool_size,
        connection_acquisition_timeout=config.neo4j_connection_acquisition_timeout,
    )

    # Verify connectivity
    driver.verify_connectivity()
    logger.info("Neo4j connectivity verified")

    # Initialize vector retriever
    logger.info("Initializing vector retriever with index '%s'", config.vector_index_name)
    retriever = create_vector_retriever(driver, config)
    logger.info("Vector retriever initialized")

    # Store in app state for route access
    app.state.config = config
    app.state.driver = driver
    app.state.retriever = retriever

    logger.info("Jama GraphRAG API ready to accept requests")
    yield

    # Cleanup on shutdown
    logger.info("Shutting down Jama GraphRAG API")
    driver.close()


# Get CORS origins from environment or use defaults
cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173",
).split(",")

app = FastAPI(
    title="Jama GraphRAG API",
    description="""
GraphRAG backend for Jama Requirements Management knowledge base.

This API provides intelligent access to requirements engineering knowledge including:
- RAG-powered Q&A with source citations
- Vector, hybrid, and graph-enriched search
- Definition term lookup
- Industry standards reference
- Knowledge graph exploration

Based on the Jama Software "Essential Guide to Requirements Management and Traceability".
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health.router, tags=["health"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(search.router, prefix="/api/v1", tags=["search"])
app.include_router(schema.router, prefix="/api/v1", tags=["schema"])
app.include_router(definitions.router, prefix="/api/v1", tags=["definitions"])
app.include_router(standards.router, prefix="/api/v1", tags=["standards"])


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": "Jama GraphRAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }
