"""Agentic RAG implementation using LangGraph.

This module provides a full agentic RAG system that replaces the routed RAG
approach with an LLM-driven agent loop. The agent can dynamically select tools,
iterate on queries, and synthesize comprehensive answers.

Architecture:
- State: TypedDict definitions for graph state management
- Tools: LangChain tool wrappers around existing retrieval functions
- Nodes: Graph node functions for processing state
- Subgraphs: Modular subgraphs for RAG, Research, and Synthesis
- Orchestrator: Main composed graph coordinating subgraphs
- Streaming: SSE streaming utilities for real-time responses
- Checkpoints: PostgresSaver configuration for conversation persistence

Usage:
    from requirements_graphrag_api.core.agentic import (
        create_orchestrator_graph,
        create_agent_tools,
        get_thread_config,
        OrchestratorState,
    )

    # Create the orchestrator graph
    graph = create_orchestrator_graph(config, driver, retriever)

    # Invoke with thread configuration for persistence
    config = get_thread_config("thread-123")
    result = await graph.ainvoke({"query": "..."}, config=config)
"""

from __future__ import annotations

from requirements_graphrag_api.core.agentic.checkpoints import (
    async_checkpointer_context,
    create_async_checkpointer,
    get_thread_config,
    get_thread_id_from_config,
)
from requirements_graphrag_api.core.agentic.orchestrator import (
    create_orchestrator_graph,
)
from requirements_graphrag_api.core.agentic.state import (
    AgentState,
    CriticEvaluation,
    EntityInfo,
    OrchestratorState,
    RAGState,
    ResearchState,
    RetrievedDocument,
    SynthesisState,
)
from requirements_graphrag_api.core.agentic.streaming import (
    AgenticEvent,
    AgenticEventType,
    stream_agentic_events,
)
from requirements_graphrag_api.core.agentic.tools import create_agent_tools

__all__ = [
    "AgentState",
    "AgenticEvent",
    "AgenticEventType",
    "CriticEvaluation",
    "EntityInfo",
    "OrchestratorState",
    "RAGState",
    "ResearchState",
    "RetrievedDocument",
    "SynthesisState",
    "async_checkpointer_context",
    "create_agent_tools",
    "create_async_checkpointer",
    "create_orchestrator_graph",
    "get_thread_config",
    "get_thread_id_from_config",
    "stream_agentic_events",
]
