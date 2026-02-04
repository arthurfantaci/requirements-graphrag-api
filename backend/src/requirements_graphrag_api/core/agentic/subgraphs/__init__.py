"""Subgraph modules for the agentic RAG system.

This package contains modular subgraphs that can be composed by the orchestrator:

- rag: RAG retrieval subgraph for parallel search and ranking
- research: Entity exploration subgraph for deep dives
- synthesis: Answer synthesis subgraph with self-critique

Subgraph Design Principles:
1. Each subgraph is self-contained and testable
2. Subgraphs communicate via well-defined state interfaces
3. Use conditional edges for dynamic control flow
4. Include iteration limits to prevent infinite loops
"""

from __future__ import annotations

from requirements_graphrag_api.core.agentic.subgraphs.rag import create_rag_subgraph
from requirements_graphrag_api.core.agentic.subgraphs.research import (
    create_research_subgraph,
)
from requirements_graphrag_api.core.agentic.subgraphs.synthesis import (
    create_synthesis_subgraph,
)

__all__ = [
    "create_rag_subgraph",
    "create_research_subgraph",
    "create_synthesis_subgraph",
]
