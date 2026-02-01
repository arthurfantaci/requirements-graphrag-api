"""FastAPI REST API for GraphRAG.

This module provides the FastAPI application for the GraphRAG REST API,
deployable to Vercel serverless functions.

Features:
- Health check endpoint
- Vector, hybrid, and graph-enriched search
- RAG-powered chat with SSE streaming
- Definition/glossary term lookups
- Industry standards queries
- Knowledge graph schema exploration

Updated Data Model (2026-01):
- Uses neo4j-graphrag VectorRetriever
- Uses neo4j Driver directly instead of LangChain Neo4jGraph
- Optimized connection pool for serverless (5-10 connections)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded

from requirements_graphrag_api.auth import (
    AuthMiddleware,
    InMemoryAPIKeyStore,
    configure_audit_logging,
)
from requirements_graphrag_api.config import get_auth_config, get_config, get_guardrail_config
from requirements_graphrag_api.core.retrieval import create_vector_retriever
from requirements_graphrag_api.middleware import (
    SizeLimitMiddleware,
    get_rate_limiter,
    rate_limit_exceeded_handler,
)
from requirements_graphrag_api.observability import configure_tracing
from requirements_graphrag_api.routes import (
    admin_router,
    chat_router,
    definitions_router,
    feedback_router,
    health_router,
    schema_router,
    search_router,
    standards_router,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifespan for resource initialization and cleanup.

    This context manager handles:
    - Neo4j driver creation and verification
    - VectorRetriever initialization
    - LangSmith tracing configuration
    - Resource cleanup on shutdown

    Neo4j Best Practices:
    - Create driver once, reuse across requests
    - Verify connectivity at startup
    - Small connection pool for serverless (5-10 connections)
    """
    # Import neo4j here to allow the module to load without it during discovery
    from neo4j import GraphDatabase

    # Load configuration
    config = get_config()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Configure LangSmith tracing
    configure_tracing(config)

    # Create Neo4j driver with serverless-optimized settings
    logger.info(
        "Connecting to Neo4j: %s (pool_size=%d, max_lifetime=%ds)",
        config.neo4j_uri.split("@")[-1] if "@" in config.neo4j_uri else config.neo4j_uri,
        config.neo4j_max_connection_pool_size,
        config.neo4j_max_connection_lifetime,
    )

    driver = GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_username, config.neo4j_password),
        max_connection_pool_size=config.neo4j_max_connection_pool_size,
        connection_acquisition_timeout=config.neo4j_connection_acquisition_timeout,
        liveness_check_timeout=config.neo4j_liveness_check_timeout,
        max_connection_lifetime=config.neo4j_max_connection_lifetime,
    )

    # Verify connectivity (fail fast)
    try:
        driver.verify_connectivity()
        logger.info("Neo4j connection verified successfully")
    except Exception as e:
        logger.error("Failed to connect to Neo4j: %s", e)
        driver.close()
        raise

    # Create VectorRetriever
    retriever = create_vector_retriever(driver, config)
    logger.info("VectorRetriever initialized with index: %s", config.vector_index_name)

    # Initialize guardrails
    guardrail_config = get_guardrail_config()
    limiter = get_rate_limiter()

    # Initialize authentication
    auth_config = get_auth_config()
    api_key_store = InMemoryAPIKeyStore()

    # Configure audit logging if enabled
    if auth_config.audit_enabled:
        configure_audit_logging(enable_logging=True, enable_langsmith=False)
        logger.info("Audit logging enabled")

    # Store in app state for route handlers
    app.state.config = config
    app.state.driver = driver
    app.state.retriever = retriever
    app.state.guardrail_config = guardrail_config
    app.state.limiter = limiter
    app.state.auth_config = auth_config
    app.state.api_key_store = api_key_store

    logger.info(
        "Guardrails initialized: injection=%s, pii=%s, rate_limit=%s",
        guardrail_config.prompt_injection_enabled,
        guardrail_config.pii_detection_enabled,
        guardrail_config.rate_limiting_enabled,
    )
    logger.info(
        "Authentication initialized: require_api_key=%s",
        auth_config.require_api_key,
    )
    logger.info("API startup complete")

    yield

    # Cleanup
    logger.info("Shutting down API...")
    driver.close()
    logger.info("Neo4j driver closed")


# Create FastAPI app
app = FastAPI(
    title="Requirements GraphRAG API",
    description=(
        "GraphRAG REST API for Requirements Management Knowledge Graph. "
        "Provides RAG-powered Q&A, semantic search, and knowledge graph exploration."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Register rate limit exception handler
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# Configure CORS for frontend access
# Use CORS_ORIGINS env var (comma-separated) or defaults for local development
_cors_origins_env = os.getenv("CORS_ORIGINS", "")
_cors_origins = (
    [origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()]
    if _cors_origins_env
    else ["http://localhost:3000", "http://localhost:5173"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add authentication middleware
# Note: Middleware runs in reverse order of registration, so auth runs after CORS
_auth_required = os.getenv("REQUIRE_API_KEY", "false").lower() in ("true", "1", "yes")
app.add_middleware(AuthMiddleware, require_auth=_auth_required)

# Add size limit middleware (Phase 4)
# Prevents oversized requests from consuming server resources
app.add_middleware(SizeLimitMiddleware)

# Mount routers
app.include_router(health_router, tags=["Health"])
app.include_router(chat_router, tags=["Chat"])
app.include_router(feedback_router, tags=["Feedback"])
app.include_router(search_router, tags=["Search"])
app.include_router(definitions_router, tags=["Definitions"])
app.include_router(standards_router, tags=["Standards"])
app.include_router(schema_router, tags=["Schema"])
app.include_router(admin_router, tags=["Admin"])  # Phase 4: Compliance dashboard


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": "Requirements GraphRAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


def run() -> None:
    """Run the API server (for local development)."""
    import uvicorn

    uvicorn.run(
        "requirements_graphrag_api.api:app",
        host="0.0.0.0",  # noqa: S104 - development server
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    run()
