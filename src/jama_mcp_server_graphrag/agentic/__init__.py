"""Agentic RAG patterns for intelligent retrieval.

This module implements agentic patterns from Essential GraphRAG Chapter 5:
- Router: Selects optimal retrieval tool based on query characteristics
- Stepback: Transforms specific queries to broader ones for better retrieval
- Critic: Validates whether retrieved context answers the question
- Query Updater: Updates queries with context from previous answers
"""

from __future__ import annotations

from jama_mcp_server_graphrag.agentic.critic import CritiqueResult, critique_answer
from jama_mcp_server_graphrag.agentic.query_updater import update_query
from jama_mcp_server_graphrag.agentic.router import RoutingResult, route_query
from jama_mcp_server_graphrag.agentic.stepback import generate_stepback_query

__all__ = [
    "CritiqueResult",
    "RoutingResult",
    "critique_answer",
    "generate_stepback_query",
    "route_query",
    "update_query",
]
