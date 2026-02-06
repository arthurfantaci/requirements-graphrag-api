"""Agentic RAG implementation using LangGraph.

This is the ONLY active path for EXPLANATORY queries. The orchestrator
composes subgraphs for retrieval, research, and synthesis.

Architecture:
- State: TypedDict definitions for graph state management
- Subgraphs: Modular subgraphs for RAG, Research, and Synthesis
- Orchestrator: Main composed graph coordinating subgraphs
- Streaming: SSE streaming utilities for real-time responses
- Checkpoints: PostgresSaver configuration for conversation persistence

Usage:
    from requirements_graphrag_api.core.agentic import (
        create_orchestrator_graph,
        get_thread_config,
        OrchestratorState,
    )

    graph = create_orchestrator_graph(config, driver, retriever)
    config = get_thread_config("thread-123")
    result = await graph.ainvoke({"query": "..."}, config=config)
"""

from __future__ import annotations

from requirements_graphrag_api.core.agentic.checkpoints import (
    async_checkpointer_context,
    create_async_checkpointer,
    get_conversation_history_from_checkpoint,
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
    "create_async_checkpointer",
    "create_orchestrator_graph",
    "get_conversation_history_from_checkpoint",
    "get_thread_config",
    "get_thread_id_from_config",
    "stream_agentic_events",
]
