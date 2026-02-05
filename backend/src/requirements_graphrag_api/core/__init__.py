"""Core GraphRAG logic shared between MCP tools and REST API routes.

This module provides two execution paths:

1. **Non-Agentic Path** (generation.py, routing.py):
   - Intent classification routes to RAG or Text2Cypher
   - Direct streaming via stream_chat()
   - Used when agentic mode is disabled

2. **Agentic Path** (agentic/ subpackage):
   - LangGraph orchestrator with RAG, Research, Synthesis subgraphs
   - PostgreSQL checkpointing for conversation persistence
   - SSE streaming via stream_agentic_events()

Updated Data Model (2026-01):
- Uses neo4j-graphrag VectorRetriever for vector search
- Uses neo4j Driver directly instead of LangChain Neo4jGraph
- Definitions module replaces Glossary module

Agentic RAG (2026-02):
- Subgraph-based orchestration (rag.py, research.py, synthesis.py)
- Self-critique and revision loop in synthesis
- Entity exploration for complex queries
"""

from __future__ import annotations

from requirements_graphrag_api.core.definitions import (
    list_all_terms,
    lookup_term,
    search_terms,
)
from requirements_graphrag_api.core.generation import (
    StreamEvent,
    StreamEventType,
)
from requirements_graphrag_api.core.retrieval import (
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
from requirements_graphrag_api.core.routing import (
    QueryIntent,
    classify_intent,
    get_routing_guide,
)
from requirements_graphrag_api.core.standards import (
    get_standards_by_industry,
    list_all_standards,
    lookup_standard,
    search_standards,
)
from requirements_graphrag_api.core.text2cypher import generate_cypher, text2cypher_query

__all__ = [
    "DEFAULT_ENRICHMENT_OPTIONS",
    "GraphEnrichmentOptions",
    "QueryIntent",
    "StreamEvent",
    "StreamEventType",
    "classify_intent",
    "create_vector_retriever",
    "explore_entity",
    "generate_cypher",
    "get_entities_from_chunks",
    "get_related_entities",
    "get_routing_guide",
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
