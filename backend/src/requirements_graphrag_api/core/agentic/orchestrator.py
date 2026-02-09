"""Main orchestrator graph composing all subgraphs.

This module defines the top-level StateGraph that:
1. Analyzes query complexity and routes appropriately
2. Composes RAG, Research, and Synthesis subgraphs
3. Manages iteration limits and termination conditions
4. Integrates with PostgresSaver for checkpoint persistence

Flow:
    START -> initialize -> run_rag -> should_proceed? -> [research?] -> synthesis -> END
                                          ↓ (quality_pass=False)
                                     format_fallback -> END

The orchestrator uses the INTENT_CLASSIFIER prompt for initial query analysis,
then routes through the appropriate subgraphs based on complexity.

Usage:
    from requirements_graphrag_api.core.agentic.orchestrator import create_orchestrator_graph

    graph = create_orchestrator_graph(config, driver, retriever)
    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "..."}})
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from requirements_graphrag_api.core.agentic.state import (
    EntityInfo,
    OrchestratorState,
    RAGState,
    ResearchState,
    RetrievedDocument,
    SynthesisState,
)
from requirements_graphrag_api.core.agentic.subgraphs import (
    create_rag_subgraph,
    create_research_subgraph,
    create_synthesis_subgraph,
)
from requirements_graphrag_api.core.context import NormalizedDocument, format_context

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from neo4j import Driver
    from neo4j_graphrag.retrievers import VectorRetriever

    from requirements_graphrag_api.config import AppConfig

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_ITERATIONS = 3
RESEARCH_ENTITY_THRESHOLD = 2  # Min entities to trigger research
COMPARISON_KEYWORDS = frozenset(
    [
        "compare",
        "comparison",
        "versus",
        "vs",
        "difference between",
        "differences",
        "similarities",
        "contrast",
        "relationship between",
    ]
)


def create_orchestrator_graph(
    config: AppConfig,
    driver: Driver,
    retriever: VectorRetriever,
    *,
    checkpointer: AsyncPostgresSaver | None = None,
) -> StateGraph:
    """Create the main orchestrator graph composing all subgraphs.

    Args:
        config: Application configuration.
        driver: Neo4j driver instance.
        retriever: Vector retriever instance.
        checkpointer: Optional PostgresSaver for conversation persistence.

    Returns:
        Compiled orchestrator graph.
    """
    # Create subgraphs
    rag_subgraph = create_rag_subgraph(config, driver, retriever)
    research_subgraph = create_research_subgraph(config, driver)
    synthesis_subgraph = create_synthesis_subgraph(config)

    # -------------------------------------------------------------------------
    # Node: initialize
    # -------------------------------------------------------------------------
    async def initialize(state: OrchestratorState) -> dict[str, Any]:
        """Initialize the orchestrator state from user message.

        Extracts the query from messages and sets up initial state.
        """
        messages = state.get("messages", [])
        query = state.get("query", "")

        # Extract query from last human message if not explicitly set
        if not query and messages:
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    query = msg.content
                    break

        if not query:
            logger.warning("No query found in orchestrator state")
            return {
                "current_phase": "error",
                "error": "No query provided",
            }

        logger.info("Initializing orchestrator for query: %s", query[:50])

        return {
            "query": query,
            "current_phase": "rag",
            "iteration_count": 0,
            "max_iterations": DEFAULT_MAX_ITERATIONS,
        }

    # -------------------------------------------------------------------------
    # Node: run_rag
    # -------------------------------------------------------------------------
    async def run_rag(state: OrchestratorState) -> dict[str, Any]:
        """Execute the RAG subgraph for retrieval.

        Invokes the RAG subgraph and extracts results.
        """
        query = state["query"]
        logger.info("Running RAG subgraph for: %s", query[:50])

        # Prepare RAG state
        rag_input: RAGState = {"query": query}

        try:
            # Invoke RAG subgraph
            rag_result = await rag_subgraph.ainvoke(rag_input)

            # Extract results
            ranked_results = rag_result.get("ranked_results", [])
            expanded_queries = rag_result.get("expanded_queries", [])
            quality_pass = rag_result.get("quality_pass", True)

            # Build hybrid context string (inline entities + KG section)
            normalized = []
            for doc in ranked_results[:10]:
                if isinstance(doc, RetrievedDocument):
                    normalized.append(NormalizedDocument.from_retrieved_document(doc))
                elif isinstance(doc, dict):
                    normalized.append(NormalizedDocument.from_raw_result(doc))
            formatted = format_context(normalized)
            context = formatted.context

            logger.info(
                "RAG complete: %d results, %d queries, quality_pass=%s",
                len(ranked_results),
                len(expanded_queries),
                quality_pass,
            )

            return {
                "ranked_results": ranked_results,
                "expanded_queries": expanded_queries,
                "context": context,
                "quality_pass": quality_pass,
                "current_phase": "research",
                "iteration_count": state.get("iteration_count", 0) + 1,
            }

        except Exception:
            logger.exception("RAG subgraph failed")
            return {
                "current_phase": "error",
                "error": "RAG retrieval failed",
            }

    # -------------------------------------------------------------------------
    # Node: run_research
    # -------------------------------------------------------------------------
    async def run_research(state: OrchestratorState) -> dict[str, Any]:
        """Execute the Research subgraph for entity exploration.

        Invokes the Research subgraph if context warrants deeper exploration.
        """
        query = state["query"]
        context = state.get("context", "")

        logger.info("Running Research subgraph")

        # Prepare Research state
        research_input: ResearchState = {
            "query": query,
            "context": context,
        }

        try:
            # Invoke Research subgraph
            research_result = await research_subgraph.ainvoke(research_input)

            # Extract entity contexts
            entity_contexts = research_result.get("entity_contexts", [])

            # Augment context with entity information
            if entity_contexts:
                entity_info_parts = []
                for entity in entity_contexts:
                    if isinstance(entity, EntityInfo):
                        info = f"**{entity.name}** ({entity.entity_type})"
                        if entity.description:
                            info += f": {entity.description}"
                        if entity.related_entities:
                            info += f"\n  Related: {', '.join(entity.related_entities[:5])}"
                        entity_info_parts.append(info)

                if entity_info_parts:
                    entity_section = "\n\n## Explored Entities\n" + "\n\n".join(entity_info_parts)
                    context = state.get("context", "") + entity_section

            logger.info("Research complete: %d entities explored", len(entity_contexts))

            return {
                "entity_contexts": entity_contexts,
                "context": context,
                "current_phase": "synthesis",
            }

        except Exception:
            logger.exception("Research subgraph failed")
            # Continue to synthesis even if research fails
            return {
                "current_phase": "synthesis",
            }

    # -------------------------------------------------------------------------
    # Node: run_synthesis
    # -------------------------------------------------------------------------
    async def run_synthesis(state: OrchestratorState) -> dict[str, Any]:
        """Execute the Synthesis subgraph for answer generation.

        Invokes the Synthesis subgraph to generate the final answer.
        """
        query = state["query"]
        context = state.get("context", "")

        logger.info("Running Synthesis subgraph")

        # Build previous_context from conversation history (all messages before current)
        previous_context = ""
        messages = state.get("messages", [])
        if len(messages) > 1:
            parts = []
            for msg in messages[:-1]:
                role = "Q" if isinstance(msg, HumanMessage) else "A"
                parts.append(f"{role}: {msg.content}")
            previous_context = "\n".join(parts)

        # Prepare Synthesis state
        synthesis_input: SynthesisState = {
            "query": query,
            "context": context,
            "previous_context": previous_context,
        }

        try:
            # Invoke Synthesis subgraph
            synthesis_result = await synthesis_subgraph.ainvoke(synthesis_input)

            # Extract results
            final_answer = synthesis_result.get("final_answer", "")
            citations = synthesis_result.get("citations", [])
            critique = synthesis_result.get("critique")

            logger.info(
                "Synthesis complete: %d chars, %d citations",
                len(final_answer),
                len(citations),
            )

            # Create AI message with the answer
            ai_message = AIMessage(content=final_answer)

            return {
                "final_answer": final_answer,
                "citations": citations,
                "critique": critique,
                "messages": [ai_message],
                "current_phase": "complete",
            }

        except Exception:
            logger.exception("Synthesis subgraph failed")
            return {
                "final_answer": "I encountered an error generating the answer.",
                "messages": [AIMessage(content="I encountered an error generating the answer.")],
                "current_phase": "error",
                "error": "Synthesis failed",
            }

    # -------------------------------------------------------------------------
    # Node: format_fallback
    # -------------------------------------------------------------------------
    async def format_fallback(state: OrchestratorState) -> dict[str, Any]:
        """Format a fallback response when quality gate fails.

        Skips research and synthesis entirely, returning a structured
        low-confidence message to the user.
        """
        query = state.get("query", "the topic")
        logger.info("Quality gate failed — formatting fallback for: %s", query[:50])

        fallback_answer = (
            "I couldn't find specific information about this topic "
            "in our knowledge base. Try rephrasing your question "
            "with different terminology, or ask about a more "
            "specific aspect of the topic."
        )

        return {
            "final_answer": fallback_answer,
            "messages": [AIMessage(content=fallback_answer)],
            "current_phase": "complete",
        }

    # -------------------------------------------------------------------------
    # Conditional edge: route_after_rag
    # -------------------------------------------------------------------------
    def route_after_rag(
        state: OrchestratorState,
    ) -> Literal["format_fallback", "run_research", "run_synthesis"]:
        """Route after RAG: quality gate check, then research decision.

        First checks quality_pass (from grade_documents). If False,
        routes to fallback. Otherwise applies research heuristics.
        """
        # --- Quality gate ---
        if not state.get("quality_pass", True):
            logger.info("Quality gate failed — routing to fallback")
            return "format_fallback"

        # --- Research heuristics (unchanged) ---
        ranked_results = state.get("ranked_results", [])
        context = state.get("context", "")
        query = state.get("query", "")
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)

        # Skip research if we're at iteration limit
        if iteration_count >= max_iterations:
            logger.info("Skipping research: max iterations reached")
            return "run_synthesis"

        # Skip research if we have very few results
        if len(ranked_results) < RESEARCH_ENTITY_THRESHOLD:
            logger.info("Skipping research: insufficient results for entity exploration")
            return "run_synthesis"

        # Skip research if context is too short
        if len(context) < 200:
            logger.info("Skipping research: context too short")
            return "run_synthesis"

        # Skip research for simple queries: short + no comparison keywords
        query_lower = query.lower()
        word_count = len(query.split())
        has_comparison = any(kw in query_lower for kw in COMPARISON_KEYWORDS)

        if word_count < 12 and not has_comparison:
            # Also skip if top retrieval score is high (confident context)
            top_score = 0.0
            if ranked_results:
                first = ranked_results[0]
                top_score = first.score if hasattr(first, "score") else first.get("score", 0)
            if top_score > 0.85:
                logger.info(
                    "Skipping research: simple query (%d words, top_score=%.2f)",
                    word_count,
                    top_score,
                )
                return "run_synthesis"

        logger.info("Proceeding to research phase")
        return "run_research"

    # -------------------------------------------------------------------------
    # Conditional edge: check_error
    # -------------------------------------------------------------------------
    def check_error(state: OrchestratorState) -> Literal["run_rag", END]:
        """Check if initialization failed."""
        if state.get("current_phase") == "error":
            return END
        return "run_rag"

    # -------------------------------------------------------------------------
    # Build the orchestrator graph
    # -------------------------------------------------------------------------
    builder = StateGraph(OrchestratorState)

    # Add nodes
    builder.add_node("initialize", initialize)
    builder.add_node("run_rag", run_rag)
    builder.add_node("format_fallback", format_fallback)
    builder.add_node("run_research", run_research)
    builder.add_node("run_synthesis", run_synthesis)

    # Add edges
    builder.add_edge(START, "initialize")
    builder.add_conditional_edges(
        "initialize",
        check_error,
        ["run_rag", END],
    )
    builder.add_conditional_edges(
        "run_rag",
        route_after_rag,
        ["format_fallback", "run_research", "run_synthesis"],
    )
    builder.add_edge("run_research", "run_synthesis")
    builder.add_edge("run_synthesis", END)
    builder.add_edge("format_fallback", END)

    # Compile with optional checkpointer
    return builder.compile(checkpointer=checkpointer)


__all__ = ["create_orchestrator_graph"]
