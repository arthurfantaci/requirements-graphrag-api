"""Core GraphRAG logic shared between MCP tools and REST API routes.

Execution:
- **Agentic Path** (agentic/ subpackage):
  LangGraph orchestrator with RAG, Research, Synthesis subgraphs.
  PostgreSQL checkpointing for conversation persistence.
  SSE streaming via stream_agentic_events().

- **Structured Path** (text2cypher.py):
  Intent classifier routes list/count queries to Cypher generation.

Supporting modules:
- routing.py: Intent classification (EXPLANATORY vs STRUCTURED vs CONVERSATIONAL)
- definitions.py: Glossary term lookup + shared types and context-building for RAG
- retrieval.py: Vector + graph-enriched search
- standards.py: Industry standards lookup
"""

from __future__ import annotations

from requirements_graphrag_api.core.context import (
    FormattedContext,
    NormalizedDocument,
    format_context,
)
from requirements_graphrag_api.core.definitions import (
    StreamEvent,
    StreamEventType,
    list_all_terms,
    lookup_term,
    search_terms,
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
    "FormattedContext",
    "GraphEnrichmentOptions",
    "NormalizedDocument",
    "QueryIntent",
    "StreamEvent",
    "StreamEventType",
    "classify_intent",
    "create_vector_retriever",
    "explore_entity",
    "format_context",
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
