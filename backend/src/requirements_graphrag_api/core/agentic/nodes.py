"""Graph node functions for the agentic RAG system.

This module contains the node functions that process state in the LangGraph.
Each node is a function that takes state and returns updated state.

Node Categories:
1. Agent Nodes: LLM calls for reasoning and tool selection
2. Tool Nodes: Execute tool calls and format results
3. Processing Nodes: Transform and validate state
4. Control Nodes: Routing and iteration control

Node Design Principles:
1. Nodes are pure functions of state -> state updates
2. Side effects (API calls) should be explicit and traceable
3. Use LangSmith tracing for observability
4. Include clear logging for debugging
"""

from __future__ import annotations

__all__: list[str] = []
