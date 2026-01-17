"""LangGraph workflows for RAG pipelines.

Provides workflow implementations for different RAG strategies:
- Basic RAG: Simple retrieve-then-generate
- Agentic RAG: Routing, critique, and iterative refinement

Usage:
    from jama_mcp_server_graphrag.workflows import run_rag_workflow, run_agentic_workflow

    # Basic RAG
    result = await run_rag_workflow(config, graph, vector_store, question)

    # Agentic RAG with routing and critique
    result = await run_agentic_workflow(config, graph, vector_store, question)
"""

from __future__ import annotations

from jama_mcp_server_graphrag.workflows.agentic_workflow import (
    create_agentic_workflow,
    run_agentic_workflow,
)
from jama_mcp_server_graphrag.workflows.rag_workflow import (
    create_rag_workflow,
    run_rag_workflow,
)
from jama_mcp_server_graphrag.workflows.state import (
    AgenticState,
    ChatMessage,
    ConversationalState,
    DocumentResult,
    RAGState,
)

__all__ = [
    "AgenticState",
    "ChatMessage",
    "ConversationalState",
    "DocumentResult",
    "RAGState",
    "create_agentic_workflow",
    "create_rag_workflow",
    "run_agentic_workflow",
    "run_rag_workflow",
]
