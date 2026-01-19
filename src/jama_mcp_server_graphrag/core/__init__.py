"""Core GraphRAG logic shared between MCP tools and REST API routes.

Updated Data Model (2026-01):
- Uses neo4j-graphrag VectorRetriever for vector search
- Uses neo4j Driver directly instead of LangChain Neo4jGraph
- Definitions module replaces Glossary module
"""

from __future__ import annotations

from jama_mcp_server_graphrag.core.definitions import (
    list_all_terms,
    lookup_term,
    search_terms,
)
from jama_mcp_server_graphrag.core.generation import chat, generate_answer
from jama_mcp_server_graphrag.core.retrieval import (
    DEFAULT_ENRICHMENT_OPTIONS,
    GraphEnrichmentOptions,
    create_vector_retriever,
    explore_entity,
    get_entities_from_chunks,
    get_related_entities,
    graph_enriched_search,
    hybrid_search,
    search_entities_by_name,
    vector_search,
)
from jama_mcp_server_graphrag.core.standards import (
    get_standards_by_industry,
    list_all_standards,
    lookup_standard,
    search_standards,
)
from jama_mcp_server_graphrag.core.text2cypher import generate_cypher, text2cypher_query

__all__ = [
    "DEFAULT_ENRICHMENT_OPTIONS",
    "GraphEnrichmentOptions",
    "chat",
    "create_vector_retriever",
    "explore_entity",
    "generate_answer",
    "generate_cypher",
    "get_entities_from_chunks",
    "get_related_entities",
    "get_standards_by_industry",
    "graph_enriched_search",
    "hybrid_search",
    "list_all_standards",
    "list_all_terms",
    "lookup_standard",
    "lookup_term",
    "search_entities_by_name",
    "search_standards",
    "search_terms",
    "text2cypher_query",
    "vector_search",
]
