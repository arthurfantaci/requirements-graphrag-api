"""TypedDict state definitions for the agentic RAG system.

This module defines all state types used across the LangGraph implementation:
- RAGState: State for the RAG retrieval subgraph
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
from pydantic import BaseModel, Field
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
# DOCUMENT GRADING MODEL (for retrieval quality gate)
# =============================================================================


class GradeDocuments(BaseModel):
    """Binary relevance grade for a retrieved document.

    Used with `.with_structured_output()` to constrain the LLM
    to a yes/no relevance decision per document.
    """

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the document could provide useful context "
        "or background knowledge for the question, 'no' otherwise.",
    )


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
        relevant_count: Number of documents graded as relevant.
        total_count: Total number of documents graded.
        quality_pass: Whether retrieval passed the quality gate.
    """

    # Required
    query: str

    # Populated during subgraph execution
    expanded_queries: list[str]
    raw_results: Annotated[list[dict[str, Any]], operator.add]
    ranked_results: list[RetrievedDocument]
    retrieval_metadata: dict[str, Any]

    # Quality gate (populated by dedupe_and_rank node)
    relevant_count: int
    total_count: int
    quality_pass: bool


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

    # Multi-turn conversation context
    previous_context: str

    # Populated during subgraph execution
    draft_answer: str
    critique: CriticEvaluation
    revision_count: int
    final_answer: str
    citations: list[str]


# =============================================================================
# ORCHESTRATOR STATE (main composed graph)
# =============================================================================


class OrchestratorState(TypedDict, total=False):
    """Main orchestrator state composing all subgraph states.

    This is the top-level state for the composed agentic graph.
    It includes fields from all subgraphs plus orchestration control.

    The orchestrator routes between subgraphs based on retrieval quality
    and manages the overall agent execution flow.

    Required:
        messages: Conversation messages (uses add_messages reducer).
        query: The current user query being processed.

    Shared with subgraphs:
        context: Retrieved context (shared with RAG, Synthesis).
        expanded_queries: From RAG subgraph.
        ranked_results: From RAG subgraph.
        final_answer: From Synthesis subgraph.
        citations: From Synthesis subgraph.

    Orchestration control:
        current_phase: Current execution phase.
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
    quality_pass: bool

    # Shared with Synthesis subgraph
    draft_answer: str
    critique: CriticEvaluation
    final_answer: str
    citations: list[str]

    # Orchestration control
    current_phase: Literal["rag", "synthesis", "complete", "error"]
    error: str | None

    # Tracing and persistence
    run_id: str | None
    thread_id: str | None


__all__ = [
    "CriticEvaluation",
    "GradeDocuments",
    "OrchestratorState",
    "RAGState",
    "RetrievedDocument",
    "SynthesisState",
]
