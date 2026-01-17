"""Basic RAG workflow using LangGraph.

Implements a straightforward retrieve-then-generate pipeline:
1. Retrieve: Get relevant documents via vector search
2. Format: Build context from retrieved documents
3. Generate: Create answer with citations

This workflow is suitable for simple Q&A without advanced
routing or quality validation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from jama_mcp_server_graphrag.core.generation import generate_answer
from jama_mcp_server_graphrag.core.retrieval import vector_search
from jama_mcp_server_graphrag.workflows.state import RAGState

if TYPE_CHECKING:
    from langchain_neo4j import Neo4jGraph, Neo4jVector

    from jama_mcp_server_graphrag.config import AppConfig

logger = logging.getLogger(__name__)


async def retrieve_node(
    state: RAGState,
    *,
    vector_store: Neo4jVector,
    limit: int = 6,
) -> dict[str, Any]:
    """Retrieve relevant documents for the question.

    Args:
        state: Current workflow state.
        vector_store: Neo4j vector store for similarity search.
        limit: Maximum number of documents to retrieve.

    Returns:
        State update with retrieved documents.
    """
    logger.info("Retrieving documents for: '%s'", state["question"][:50])

    try:
        results = await vector_search(vector_store, state["question"], limit=limit)

        documents = [
            {
                "content": r["content"],
                "score": r["score"],
                "metadata": r["metadata"],
            }
            for r in results
        ]

    except Exception as e:
        logger.exception("Retrieval failed")
        return {"error": f"Retrieval failed: {e}"}
    else:
        logger.info("Retrieved %d documents", len(documents))
        return {"documents": documents}


async def format_context_node(state: RAGState) -> dict[str, Any]:
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


async def generate_node(
    state: RAGState,
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
        return {}

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


def create_rag_workflow(
    config: AppConfig,
    graph: Neo4jGraph,
    vector_store: Neo4jVector,
    limit: int = 6,
) -> StateGraph:
    """Create a compiled RAG workflow graph.

    Args:
        config: Application configuration.
        graph: Neo4j graph connection.
        vector_store: Vector store for retrieval.
        limit: Maximum documents to retrieve.

    Returns:
        Compiled LangGraph workflow.
    """
    # Create workflow with state schema
    workflow = StateGraph(RAGState)

    # Define nodes with bound parameters
    async def retrieve(state: RAGState) -> dict[str, Any]:
        return await retrieve_node(state, vector_store=vector_store, limit=limit)

    async def format_context(state: RAGState) -> dict[str, Any]:
        return await format_context_node(state)

    async def generate(state: RAGState) -> dict[str, Any]:
        return await generate_node(state, config=config, graph=graph)

    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("format_context", format_context)
    workflow.add_node("generate", generate)

    # Define edges (linear flow)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "format_context")
    workflow.add_edge("format_context", "generate")
    workflow.add_edge("generate", END)

    # Compile and return
    return workflow.compile()


async def run_rag_workflow(
    config: AppConfig,
    graph: Neo4jGraph,
    vector_store: Neo4jVector,
    question: str,
    limit: int = 6,
) -> dict[str, Any]:
    """Run the RAG workflow for a question.

    Convenience function that creates and executes the workflow.

    Args:
        config: Application configuration.
        graph: Neo4j graph connection.
        vector_store: Vector store for retrieval.
        question: User's question.
        limit: Maximum documents to retrieve.

    Returns:
        Final workflow state with answer and sources.
    """
    logger.info("Running RAG workflow for: '%s'", question[:50])

    workflow = create_rag_workflow(
        config=config,
        graph=graph,
        vector_store=vector_store,
        limit=limit,
    )

    # Initialize state
    initial_state: RAGState = {
        "question": question,
        "documents": [],
        "context": "",
        "answer": "",
        "sources": [],
        "error": None,
    }

    # Execute workflow
    result = await workflow.ainvoke(initial_state)

    logger.info("RAG workflow completed")
    return result
