"""FastAPI route handlers for REST API.

Updated Data Model (2026-01):
- Uses neo4j Driver directly instead of LangChain Neo4jGraph
- Uses VectorRetriever instead of Neo4jVector
- Definitions route replaces Glossary route
"""

from __future__ import annotations

from jama_mcp_server_graphrag.routes import chat, definitions, health, schema, search, standards

__all__ = [
    "chat",
    "definitions",
    "health",
    "schema",
    "search",
    "standards",
]
