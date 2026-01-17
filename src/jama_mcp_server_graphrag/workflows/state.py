"""Workflow state definitions for LangGraph RAG pipelines.

Defines TypedDict state schemas used by RAG and agentic workflows.
State flows through the graph and is updated by each node.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from typing_extensions import TypedDict


def add_to_list(existing: list[Any], new: Any) -> list[Any]:
    """Reducer function to append items to a list.

    Used with Annotated to enable incremental state updates.

    Args:
        existing: Current list in state.
        new: New item(s) to add.

    Returns:
        Updated list with new items appended.
    """
    if new is None:
        return existing
    if isinstance(new, list):
        return [*existing, *new]
    return [*existing, new]


class DocumentResult(TypedDict):
    """Retrieved document with metadata."""

    content: str
    score: float
    metadata: dict[str, Any]


class RAGState(TypedDict):
    """State for basic RAG workflow.

    Attributes:
        question: Original user question.
        documents: Retrieved documents with scores.
        context: Formatted context string from documents.
        answer: Generated answer.
        sources: Source citations for the answer.
        error: Error message if any step failed.
    """

    question: str
    documents: Annotated[list[DocumentResult], add_to_list]
    context: str
    answer: str
    sources: list[dict[str, Any]]
    error: str | None


class AgenticState(TypedDict):
    """State for agentic RAG workflow with routing and critique.

    Extends RAGState with routing decisions, critique results,
    and iteration tracking for multi-step retrieval.

    Attributes:
        question: Original user question.
        refined_question: Question after step-back or refinement.
        selected_tools: Tools selected by router.
        routing_reasoning: Explanation for tool selection.
        documents: Retrieved documents with scores.
        context: Formatted context string from documents.
        critique_result: Answer quality assessment.
        needs_more_context: Whether additional retrieval is needed.
        followup_query: Suggested follow-up query from critic.
        iteration: Current iteration count (for limiting retries).
        max_iterations: Maximum allowed iterations.
        answer: Generated answer.
        sources: Source citations for the answer.
        confidence: Confidence score for the answer.
        error: Error message if any step failed.
    """

    question: str
    refined_question: str
    selected_tools: list[str]
    routing_reasoning: str
    documents: Annotated[list[DocumentResult], add_to_list]
    context: str
    critique_result: dict[str, Any] | None
    needs_more_context: bool
    followup_query: str | None
    iteration: int
    max_iterations: int
    answer: str
    sources: list[dict[str, Any]]
    confidence: float
    error: str | None


class ChatMessage(TypedDict):
    """Chat message structure."""

    role: Literal["user", "assistant", "system"]
    content: str


class ConversationalState(TypedDict):
    """State for conversational RAG with history.

    Supports multi-turn conversations with memory.

    Attributes:
        messages: Conversation history.
        current_question: Current user question.
        documents: Retrieved documents for current turn.
        context: Formatted context string.
        answer: Generated answer for current turn.
        sources: Source citations.
        session_id: Unique session identifier.
    """

    messages: Annotated[list[ChatMessage], add_to_list]
    current_question: str
    documents: list[DocumentResult]
    context: str
    answer: str
    sources: list[dict[str, Any]]
    session_id: str
