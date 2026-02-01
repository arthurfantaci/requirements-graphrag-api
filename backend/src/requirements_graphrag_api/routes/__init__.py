"""FastAPI route handlers for GraphRAG REST API.

This package contains all route handlers organized by domain:
- chat: RAG-powered Q&A with SSE streaming
- search: Vector, hybrid, and graph-enriched search
- definitions: Definition/glossary term lookups
- standards: Industry standards queries
- schema: Knowledge graph schema introspection
- health: Health check endpoints
- feedback: User feedback collection for LangSmith
- admin: Administrative endpoints for compliance dashboard (Phase 4)
"""

from __future__ import annotations

from requirements_graphrag_api.routes.admin import router as admin_router
from requirements_graphrag_api.routes.chat import router as chat_router
from requirements_graphrag_api.routes.definitions import router as definitions_router
from requirements_graphrag_api.routes.feedback import router as feedback_router
from requirements_graphrag_api.routes.health import router as health_router
from requirements_graphrag_api.routes.schema import router as schema_router
from requirements_graphrag_api.routes.search import router as search_router
from requirements_graphrag_api.routes.standards import router as standards_router

__all__ = [
    "admin_router",
    "chat_router",
    "definitions_router",
    "feedback_router",
    "health_router",
    "schema_router",
    "search_router",
    "standards_router",
]
