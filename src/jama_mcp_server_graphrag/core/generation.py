"""Answer generation with RAG for conversational Q&A.

Combines retrieval results with LLM generation to produce
grounded answers with citations.

Updated Data Model (2026-01):
- Uses VectorRetriever and Driver instead of Neo4jGraph/Neo4jVector
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from jama_mcp_server_graphrag.core.definitions import search_terms
from jama_mcp_server_graphrag.core.retrieval import graph_enriched_search
from jama_mcp_server_graphrag.observability import traceable

if TYPE_CHECKING:
    from neo4j import Driver
    from neo4j_graphrag.retrievers import VectorRetriever

    from jama_mcp_server_graphrag.config import AppConfig

logger = logging.getLogger(__name__)

DEFINITION_RELEVANCE_THRESHOLD: Final[float] = 0.5

RAG_SYSTEM_PROMPT: Final[str] = """You are a requirements management expert assistant.
Your knowledge comes from Jama Software's "Essential Guide to Requirements Management
and Traceability".

INSTRUCTIONS:
1. Answer questions based ONLY on the provided context
2. If the context doesn't contain enough information, say so
3. Cite your sources by mentioning article titles
4. Be concise but comprehensive
5. Use bullet points for lists
6. Highlight key terms and concepts

CONTEXT:
{context}

RELATED ENTITIES:
{entities}
"""


@traceable(name="generate_answer", run_type="chain")
async def generate_answer(  # noqa: PLR0913
    config: AppConfig,
    retriever: VectorRetriever,
    driver: Driver,
    question: str,
    *,
    retrieval_limit: int = 5,
    include_entities: bool = True,
) -> dict[str, Any]:
    """Generate an answer using RAG (Retrieval-Augmented Generation).

    Args:
        config: Application configuration.
        retriever: VectorRetriever for semantic search.
        driver: Neo4j driver for graph queries.
        question: User's question.
        retrieval_limit: Number of sources to retrieve.
        include_entities: Whether to include related entities in context.

    Returns:
        Dictionary with answer, sources, and entities.
    """
    logger.info("Generating RAG answer for: '%s'", question)

    # Search for relevant definitions/glossary terms
    # This helps answer questions like "What does X mean?" or "What is the definition of Y?"
    definitions = await search_terms(driver, question, limit=3)

    # Retrieve relevant context from chunks
    search_results = await graph_enriched_search(
        retriever,
        driver,
        question,
        limit=retrieval_limit,
    )

    # Build context string
    context_parts = []
    sources = []
    all_entities: set[str] = set()

    # Add definitions to context first (if any found)
    if definitions:
        for defn in definitions:
            if defn.get("score", 0) >= DEFINITION_RELEVANCE_THRESHOLD:
                context_parts.append(f"[Definition: {defn['term']}]\n{defn['definition']}\n")
                # Add definition as a source
                sources.append(
                    {
                        "title": f"Definition: {defn['term']}",
                        "url": defn.get("url", ""),
                        "chunk_id": None,
                        "relevance_score": defn.get("score", 0.5),
                    }
                )
                all_entities.add(defn["term"])

    for i, result in enumerate(search_results, 1):
        title = result["metadata"].get("title", "Unknown")
        content = result["content"]
        url = result["metadata"].get("url", "")

        context_parts.append(f"[Source {i}: {title}]\n{content}\n")
        sources.append(
            {
                "title": title,
                "content": content,  # Include content for evaluation metrics
                "url": url,
                "chunk_id": result["metadata"].get("chunk_id"),
                "relevance_score": result["score"],
            }
        )

        # Collect entities from enriched structure
        if include_entities:
            for entity in result.get("entities", []):
                if isinstance(entity, dict) and entity.get("name"):
                    all_entities.add(entity["name"])
                elif isinstance(entity, str) and entity:
                    all_entities.add(entity)
            for defn in result.get("glossary_definitions", []):
                if isinstance(defn, dict) and defn.get("term"):
                    all_entities.add(defn["term"])
                elif isinstance(defn, str) and defn:
                    all_entities.add(defn)

    context = "\n".join(context_parts) if context_parts else "No relevant context found."
    entities_str = ", ".join(sorted(all_entities)[:20]) if all_entities else "None identified"

    # Generate answer with LLM
    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0.1,
        api_key=config.openai_api_key,
    )

    system_message = SystemMessage(
        content=RAG_SYSTEM_PROMPT.format(
            context=context,
            entities=entities_str,
        )
    )

    human_message = HumanMessage(content=question)

    chain = llm | StrOutputParser()
    answer = await chain.ainvoke([system_message, human_message])

    response = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "entities": list(all_entities)[:20],
        "source_count": len(sources),
    }

    logger.info("Generated answer with %d sources and %d entities", len(sources), len(all_entities))
    return response


@traceable(name="chat", run_type="chain")
async def chat(  # noqa: PLR0913
    config: AppConfig,
    retriever: VectorRetriever,
    driver: Driver,
    message: str,
    *,
    conversation_history: list[dict[str, str]] | None = None,  # noqa: ARG001
    retrieval_strategy: str = "hybrid",
    max_sources: int = 5,
) -> dict[str, Any]:
    """Handle a chat message with optional conversation history.

    Args:
        config: Application configuration.
        retriever: VectorRetriever for semantic search.
        driver: Neo4j driver for graph queries.
        message: User's message.
        conversation_history: Optional list of previous messages.
        retrieval_strategy: Strategy for retrieval ("vector", "hybrid", "graph").
        max_sources: Maximum sources to include.

    Returns:
        Dictionary with answer, sources, and entities.
    """
    logger.info("Chat: message='%s', strategy=%s", message[:50], retrieval_strategy)

    # For now, treat each message independently
    # Future: Use conversation history for context
    return await generate_answer(
        config,
        retriever,
        driver,
        message,
        retrieval_limit=max_sources,
        include_entities=True,
    )
