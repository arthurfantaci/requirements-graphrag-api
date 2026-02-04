"""TypedDict state definitions for the agentic RAG system.

This module defines all state types used across the LangGraph implementation:
- AgentState: Core agent state with messages and tool results
- RAGState: State for the RAG retrieval subgraph
- ResearchState: State for entity exploration subgraph
- SynthesisState: State for answer synthesis subgraph
- OrchestratorState: Main orchestrator state composing subgraph states

State Design Principles:
1. Use TypedDict for type safety and IDE support
2. Include Annotated fields with reducers for message lists
3. Keep states minimal - only include what's needed for that subgraph
4. Use optional fields for data that may not be present
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage  # noqa: TC002 - needed at runtime for LangGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# =============================================================================
# RETRIEVED DOCUMENT TYPE
# =============================================================================


@dataclass(frozen=True, slots=True)
class RetrievedDocument:
    """A retrieved document with content and metadata.

    Attributes:
        content: The text content of the document.
        source: Source identifier (article title, URL, etc.).
        score: Relevance score from retrieval (0.0-1.0).
        metadata: Additional metadata (chunk index, entities, etc.).
    """

    content: str
    source: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# CRITIC EVALUATION TYPE
# =============================================================================


@dataclass(frozen=True, slots=True)
class CriticEvaluation:
    """Result of critic self-evaluation.

    Attributes:
        answerable: Whether the question can be answered from context.
        confidence: Confidence score (0.0-1.0).
        completeness: Assessment of context completeness.
        missing_aspects: List of missing information.
        followup_query: Suggested query to fill gaps.
        reasoning: Brief explanation of assessment.
    """

    answerable: bool
    confidence: float
    completeness: Literal["complete", "partial", "insufficient"]
    missing_aspects: list[str] = field(default_factory=list)
    followup_query: str | None = None
    reasoning: str = ""


# =============================================================================
# ENTITY INFO TYPE
# =============================================================================


@dataclass(frozen=True, slots=True)
class EntityInfo:
    """Information about an explored entity.

    Attributes:
        name: Entity name.
        entity_type: Type of entity (Tool, Concept, Standard, etc.).
        description: Entity description or display name.
        related_entities: List of related entity names.
        mentioned_in: List of article titles where mentioned.
    """

    name: str
    entity_type: str
    description: str = ""
    related_entities: list[str] = field(default_factory=list)
    mentioned_in: list[str] = field(default_factory=list)


# =============================================================================
# RAG SUBGRAPH STATE
# =============================================================================


class RAGState(TypedDict, total=False):
    """State for the RAG retrieval subgraph.

    This subgraph handles query expansion and parallel retrieval.

    Required:
        query: The original user query.

    Optional:
        expanded_queries: List of expanded/reformulated queries.
        raw_results: Raw retrieval results before ranking.
        ranked_results: Final ranked and deduplicated results.
        retrieval_metadata: Metadata about the retrieval process.
    """

    # Required
    query: str

    # Populated during subgraph execution
    expanded_queries: list[str]
    raw_results: Annotated[list[dict[str, Any]], operator.add]
    ranked_results: list[RetrievedDocument]
    retrieval_metadata: dict[str, Any]


# =============================================================================
# RESEARCH SUBGRAPH STATE
# =============================================================================


class ResearchState(TypedDict, total=False):
    """State for the entity exploration subgraph.

    This subgraph handles deep entity exploration for complex queries.

    Required:
        query: The original user query.
        context: Initial context from RAG retrieval.

    Optional:
        identified_entities: Entities identified for exploration.
        explored_entities: Entities that have been explored.
        entity_contexts: Detailed information for each explored entity.
        exploration_complete: Whether exploration is complete.
    """

    # Required
    query: str
    context: str

    # Populated during subgraph execution
    identified_entities: list[str]
    explored_entities: Annotated[list[str], operator.add]
    entity_contexts: Annotated[list[EntityInfo], operator.add]
    exploration_complete: bool


# =============================================================================
# SYNTHESIS SUBGRAPH STATE
# =============================================================================


class SynthesisState(TypedDict, total=False):
    """State for the answer synthesis subgraph.

    This subgraph handles answer generation with self-critique.

    Required:
        query: The original user query.
        context: Retrieved context for answer generation.

    Optional:
        draft_answer: Initial generated answer.
        critique: Critic evaluation of the draft.
        revision_count: Number of revisions performed.
        final_answer: Final answer after critique/revision.
        citations: List of source citations used.
    """

    # Required
    query: str
    context: str

    # Populated during subgraph execution
    draft_answer: str
    critique: CriticEvaluation
    revision_count: int
    final_answer: str
    citations: list[str]


# =============================================================================
# AGENT STATE (for tool-calling agent loop)
# =============================================================================


class AgentState(TypedDict, total=False):
    """Core state for the tool-calling agent loop.

    This state is used by the main agent for tool selection and invocation.
    Uses add_messages reducer to properly handle message updates.

    Required:
        messages: Conversation messages (uses add_messages reducer).

    Optional:
        tool_calls_count: Number of tool calls made.
        iteration_count: Number of agent loop iterations.
        should_continue: Whether the agent should continue iterating.
    """

    # Required - messages with add_messages reducer for proper handling
    messages: Annotated[list[AnyMessage], add_messages]

    # Control flow
    tool_calls_count: int
    iteration_count: int
    should_continue: bool


# =============================================================================
# ORCHESTRATOR STATE (main composed graph)
# =============================================================================


class OrchestratorState(TypedDict, total=False):
    """Main orchestrator state composing all subgraph states.

    This is the top-level state for the composed agentic graph.
    It includes fields from all subgraphs plus orchestration control.

    The orchestrator routes between subgraphs based on query complexity
    and manages the overall agent execution flow.

    Required:
        messages: Conversation messages (uses add_messages reducer).
        query: The current user query being processed.

    Shared with subgraphs:
        context: Retrieved context (shared with RAG, Research, Synthesis).
        expanded_queries: From RAG subgraph.
        ranked_results: From RAG subgraph.
        entity_contexts: From Research subgraph.
        final_answer: From Synthesis subgraph.
        citations: From Synthesis subgraph.

    Orchestration control:
        current_phase: Current execution phase.
        iteration_count: Total iterations across all phases.
        max_iterations: Maximum allowed iterations.
        error: Error message if something went wrong.
        run_id: LangSmith run ID for tracing correlation.
        thread_id: Conversation thread ID for persistence.
    """

    # Required - messages with add_messages reducer
    messages: Annotated[list[AnyMessage], add_messages]
    query: str

    # Shared with RAG subgraph
    expanded_queries: list[str]
    ranked_results: list[RetrievedDocument]
    context: str

    # Shared with Research subgraph
    entity_contexts: Annotated[list[EntityInfo], operator.add]

    # Shared with Synthesis subgraph
    draft_answer: str
    critique: CriticEvaluation
    final_answer: str
    citations: list[str]

    # Orchestration control
    current_phase: Literal["rag", "research", "synthesis", "complete", "error"]
    iteration_count: int
    max_iterations: int
    error: str | None

    # Tracing and persistence
    run_id: str | None
    thread_id: str | None


__all__ = [
    "AgentState",
    "CriticEvaluation",
    "EntityInfo",
    "OrchestratorState",
    "RAGState",
    "ResearchState",
    "RetrievedDocument",
    "SynthesisState",
]
