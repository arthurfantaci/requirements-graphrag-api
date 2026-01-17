"""Agentic RAG workflow using LangGraph.

Implements an advanced RAG pipeline with:
1. Router: Analyzes query and selects optimal retrieval tools
2. Retrieve: Executes selected retrieval strategy
3. Critique: Evaluates if context is sufficient
4. Generate: Creates answer with citations (or loops back for more context)

This workflow supports iterative refinement when initial
retrieval doesn't provide sufficient context.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from langgraph.graph import END, START, StateGraph

from jama_mcp_server_graphrag.agentic.critic import critique_answer
from jama_mcp_server_graphrag.agentic.router import route_query
from jama_mcp_server_graphrag.agentic.stepback import generate_stepback_query
from jama_mcp_server_graphrag.core.generation import generate_answer
from jama_mcp_server_graphrag.core.retrieval import (
    graph_enriched_search,
    hybrid_search,
    vector_search,
)
from jama_mcp_server_graphrag.workflows.state import AgenticState

if TYPE_CHECKING:
    from langchain_neo4j import Neo4jGraph, Neo4jVector

    from jama_mcp_server_graphrag.config import AppConfig

logger = logging.getLogger(__name__)

# Maximum iterations to prevent infinite loops
DEFAULT_MAX_ITERATIONS = 3

# Minimum confidence threshold for context sufficiency
MIN_CONFIDENCE_THRESHOLD = 0.6


async def route_node(
    state: AgenticState,
    *,
    config: AppConfig,
) -> dict[str, Any]:
    """Route query to optimal retrieval tools.

    Args:
        state: Current workflow state.
        config: Application configuration.

    Returns:
        State update with routing decision.
    """
    question = state.get("refined_question") or state["question"]
    logger.info("Routing query: '%s'", question[:50])

    try:
        result = await route_query(config, question)
    except Exception as e:
        logger.exception("Routing failed")
        # Default to hybrid search on failure
        return {
            "selected_tools": ["graphrag_hybrid_search"],
            "routing_reasoning": f"Routing failed ({e}), defaulting to hybrid search",
        }
    else:
        return {
            "selected_tools": result.selected_tools,
            "routing_reasoning": result.reasoning,
        }


async def stepback_node(
    state: AgenticState,
    *,
    config: AppConfig,
) -> dict[str, Any]:
    """Generate step-back question for broader context.

    Args:
        state: Current workflow state.
        config: Application configuration.

    Returns:
        State update with refined question.
    """
    logger.info("Generating step-back question")

    try:
        stepback = await generate_stepback_query(config, state["question"])
    except Exception as e:
        logger.warning("Step-back generation failed: %s", e)
        return {"refined_question": state["question"]}
    else:
        return {"refined_question": stepback}


async def retrieve_node(
    state: AgenticState,
    *,
    graph: Neo4jGraph,
    vector_store: Neo4jVector,
) -> dict[str, Any]:
    """Execute retrieval based on selected tools.

    Args:
        state: Current workflow state with routing decision.
        graph: Neo4j graph connection.
        vector_store: Vector store for similarity search.

    Returns:
        State update with retrieved documents.
    """
    question = state.get("refined_question") or state["question"]
    selected_tools = state.get("selected_tools", ["graphrag_hybrid_search"])

    logger.info("Retrieving with tools: %s", selected_tools)

    all_documents: list[dict[str, Any]] = []

    try:
        for tool in selected_tools:
            if tool == "graphrag_vector_search":
                results = await vector_search(vector_store, question, limit=6)
            elif tool == "graphrag_hybrid_search":
                results = await hybrid_search(graph, vector_store, question, limit=6)
            elif tool == "graphrag_graph_enriched_search":
                results = await graph_enriched_search(
                    graph, vector_store, question, limit=6, traversal_depth=2
                )
            else:
                # Skip non-retrieval tools
                continue

            all_documents.extend(
                {
                    "content": r["content"],
                    "score": r["score"],
                    "metadata": r["metadata"],
                }
                for r in results
            )

        # Deduplicate by content (keep highest score)
        seen_content: dict[str, dict[str, Any]] = {}
        for doc in all_documents:
            content_key = doc["content"][:100]  # Use first 100 chars as key
            if content_key not in seen_content or doc["score"] > seen_content[content_key]["score"]:
                seen_content[content_key] = doc

        unique_docs = list(seen_content.values())
        # Sort by score descending
        unique_docs.sort(key=lambda x: x["score"], reverse=True)

    except Exception as e:
        logger.exception("Retrieval failed")
        return {"error": f"Retrieval failed: {e}"}
    else:
        logger.info("Retrieved %d unique documents", len(unique_docs))
        return {"documents": unique_docs}


async def format_context_node(state: AgenticState) -> dict[str, Any]:
    """Format retrieved documents into context string.

    Args:
        state: Current workflow state with documents.

    Returns:
        State update with formatted context.
    """
    if state.get("error"):
        return {}

    documents = state.get("documents", [])
    if not documents:
        return {"context": "No relevant documents found.", "sources": []}

    # Format documents into context
    context_parts = []
    sources = []

    for i, doc in enumerate(documents, 1):
        content = doc["content"]
        metadata = doc["metadata"]

        context_parts.append(f"[{i}] {content}")

        sources.append(
            {
                "index": i,
                "title": metadata.get("title", "Unknown"),
                "url": metadata.get("url", ""),
                "chunk_id": metadata.get("chunk_id", ""),
                "relevance_score": doc["score"],
            }
        )

    context = "\n\n".join(context_parts)
    logger.info("Formatted context from %d documents", len(documents))

    return {"context": context, "sources": sources}


async def critique_node(
    state: AgenticState,
    *,
    config: AppConfig,
) -> dict[str, Any]:
    """Evaluate if context is sufficient to answer the question.

    Args:
        state: Current workflow state with context.
        config: Application configuration.

    Returns:
        State update with critique result.
    """
    if state.get("error"):
        return {"needs_more_context": False}

    context = state.get("context", "")
    if not context or context == "No relevant documents found.":
        return {
            "needs_more_context": True,
            "critique_result": {
                "answerable": False,
                "confidence": 0.0,
                "completeness": "insufficient",
            },
        }

    logger.info("Critiquing context quality")

    try:
        result = await critique_answer(config, state["question"], context)
    except Exception as e:
        logger.warning("Critique failed: %s", e)
        return {"needs_more_context": False, "confidence": 0.5}

    # Determine if we need more context
    needs_more = (
        not result.answerable
        or result.confidence < MIN_CONFIDENCE_THRESHOLD
        or result.completeness == "insufficient"
    )

    # Only loop if we haven't exceeded iterations
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    if iteration >= max_iterations:
        needs_more = False
        logger.info("Max iterations reached, proceeding with available context")

    return {
        "critique_result": {
            "answerable": result.answerable,
            "confidence": result.confidence,
            "completeness": result.completeness,
            "missing_aspects": result.missing_aspects,
        },
        "needs_more_context": needs_more,
        "followup_query": result.followup_query if needs_more else None,
        "confidence": result.confidence,
    }


async def refine_query_node(state: AgenticState) -> dict[str, Any]:
    """Refine query based on critique feedback.

    Args:
        state: Current workflow state with critique result.

    Returns:
        State update with refined question and incremented iteration.
    """
    followup = state.get("followup_query")
    iteration = state.get("iteration", 0)

    if followup:
        logger.info("Refining query to: '%s'", followup[:50])
        return {
            "refined_question": followup,
            "iteration": iteration + 1,
        }

    # Combine original with missing aspects
    critique = state.get("critique_result", {})
    missing = critique.get("missing_aspects", [])

    if missing:
        refined = f"{state['question']} Specifically about: {', '.join(missing)}"
        logger.info("Refining query with missing aspects")
        return {
            "refined_question": refined,
            "iteration": iteration + 1,
        }

    return {"iteration": iteration + 1}


async def generate_node(
    state: AgenticState,
    *,
    config: AppConfig,
    graph: Neo4jGraph,
) -> dict[str, Any]:
    """Generate answer from context.

    Args:
        state: Current workflow state with context.
        config: Application configuration.
        graph: Neo4j graph for entity enrichment.

    Returns:
        State update with generated answer.
    """
    if state.get("error"):
        return {"answer": "Sorry, an error occurred during processing."}

    context = state.get("context", "")
    if not context or context == "No relevant documents found.":
        return {"answer": "I couldn't find relevant information to answer your question."}

    try:
        result = await generate_answer(
            config=config,
            graph=graph,
            question=state["question"],
            context=context,
        )
    except Exception as e:
        logger.exception("Generation failed")
        return {
            "error": f"Generation failed: {e}",
            "answer": "Sorry, I couldn't generate an answer.",
        }
    else:
        logger.info("Generated answer with %d characters", len(result["answer"]))
        return {"answer": result["answer"]}


def should_refine(state: AgenticState) -> Literal["refine", "generate"]:
    """Determine whether to refine query or proceed to generation.

    Args:
        state: Current workflow state.

    Returns:
        Next node to execute.
    """
    if state.get("needs_more_context", False):
        return "refine"
    return "generate"


def create_agentic_workflow(
    config: AppConfig,
    graph: Neo4jGraph,
    vector_store: Neo4jVector,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> StateGraph:
    """Create a compiled agentic RAG workflow graph.

    The workflow follows this pattern:
    1. Route -> Select optimal retrieval tools
    2. Retrieve -> Execute retrieval
    3. Format -> Build context
    4. Critique -> Evaluate sufficiency
    5a. If insufficient -> Refine query and loop back to retrieve
    5b. If sufficient -> Generate answer

    Args:
        config: Application configuration.
        graph: Neo4j graph connection.
        vector_store: Vector store for retrieval.
        max_iterations: Maximum retrieval iterations (used in initial state).

    Returns:
        Compiled LangGraph workflow.
    """
    # Store max_iterations for use in run_agentic_workflow
    _ = max_iterations  # Used when initializing state

    workflow = StateGraph(AgenticState)

    # Define nodes with bound parameters
    async def route(state: AgenticState) -> dict[str, Any]:
        return await route_node(state, config=config)

    async def retrieve(state: AgenticState) -> dict[str, Any]:
        return await retrieve_node(state, graph=graph, vector_store=vector_store)

    async def format_context(state: AgenticState) -> dict[str, Any]:
        return await format_context_node(state)

    async def critique(state: AgenticState) -> dict[str, Any]:
        return await critique_node(state, config=config)

    async def refine(state: AgenticState) -> dict[str, Any]:
        return await refine_query_node(state)

    async def generate(state: AgenticState) -> dict[str, Any]:
        return await generate_node(state, config=config, graph=graph)

    # Add nodes
    workflow.add_node("route", route)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("format_context", format_context)
    workflow.add_node("critique", critique)
    workflow.add_node("refine", refine)
    workflow.add_node("generate", generate)

    # Define edges
    workflow.add_edge(START, "route")
    workflow.add_edge("route", "retrieve")
    workflow.add_edge("retrieve", "format_context")
    workflow.add_edge("format_context", "critique")

    # Conditional edge based on critique result
    workflow.add_conditional_edges(
        "critique",
        should_refine,
        {
            "refine": "refine",
            "generate": "generate",
        },
    )

    # Refine loops back to retrieve
    workflow.add_edge("refine", "retrieve")

    # Generate ends the workflow
    workflow.add_edge("generate", END)

    return workflow.compile()


async def run_agentic_workflow(
    config: AppConfig,
    graph: Neo4jGraph,
    vector_store: Neo4jVector,
    question: str,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> dict[str, Any]:
    """Run the agentic RAG workflow for a question.

    Convenience function that creates and executes the workflow.

    Args:
        config: Application configuration.
        graph: Neo4j graph connection.
        vector_store: Vector store for retrieval.
        question: User's question.
        max_iterations: Maximum retrieval iterations.

    Returns:
        Final workflow state with answer, sources, and confidence.
    """
    logger.info("Running agentic workflow for: '%s'", question[:50])

    workflow = create_agentic_workflow(
        config=config,
        graph=graph,
        vector_store=vector_store,
        max_iterations=max_iterations,
    )

    # Initialize state
    initial_state: AgenticState = {
        "question": question,
        "refined_question": "",
        "selected_tools": [],
        "routing_reasoning": "",
        "documents": [],
        "context": "",
        "critique_result": None,
        "needs_more_context": False,
        "followup_query": None,
        "iteration": 0,
        "max_iterations": max_iterations,
        "answer": "",
        "sources": [],
        "confidence": 0.0,
        "error": None,
    }

    # Execute workflow
    result = await workflow.ainvoke(initial_state)

    logger.info(
        "Agentic workflow completed (iterations: %d, confidence: %.2f)",
        result.get("iteration", 0),
        result.get("confidence", 0.0),
    )

    return result
