"""RAG retrieval subgraph for the agentic system.

This subgraph handles the retrieval phase:
1. Query expansion using QUERY_EXPANSION prompt
2. Parallel retrieval across multiple queries
3. Result deduplication and ranking
4. Document grading (quality gate)

Flow:
    START -> expand_queries -> parallel_retrieve -> dedupe_and_rank
         -> grade_documents -> END

State:
    RAGState with query, expanded_queries, raw_results, ranked_results,
    quality gate fields (relevant_count, total_count, quality_pass)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from requirements_graphrag_api.core.agentic.state import (
    GradeDocuments,
    RAGState,
    RetrievedDocument,
)
from requirements_graphrag_api.prompts import PromptName, get_prompt_sync

if TYPE_CHECKING:
    from neo4j import Driver
    from neo4j_graphrag.retrievers import VectorRetriever

    from requirements_graphrag_api.config import AppConfig

logger = logging.getLogger(__name__)

# Constants
DEFAULT_RETRIEVAL_LIMIT = 6
MAX_RESULTS_PER_QUERY = 6  # Increased from 4 to get more coverage per query
MAX_TOTAL_RESULTS = 10
QUALITY_GATE_THRESHOLD = 0.5  # At least 50% of docs must be relevant


def create_rag_subgraph(
    config: AppConfig,
    driver: Driver,
    retriever: VectorRetriever,
) -> StateGraph:
    """Create the RAG retrieval subgraph.

    Args:
        config: Application configuration.
        driver: Neo4j driver instance.
        retriever: Vector retriever instance.

    Returns:
        Compiled RAG subgraph.
    """
    # Import here to avoid circular imports
    from requirements_graphrag_api.core.retrieval import graph_enriched_search

    # -------------------------------------------------------------------------
    # Node: expand_queries
    # -------------------------------------------------------------------------
    async def expand_queries(state: RAGState) -> dict[str, Any]:
        """Expand the user query into multiple search queries.

        Uses the QUERY_EXPANSION prompt to generate step-back, synonym,
        and aspect-specific queries for better retrieval coverage.
        """
        query = state["query"]
        logger.info("Expanding query: %s", query[:50])

        try:
            prompt_template = get_prompt_sync(PromptName.QUERY_EXPANSION)
            llm = ChatOpenAI(
                model=config.conversational_model,
                temperature=0.3,
                api_key=config.openai_api_key,
            )

            chain = prompt_template | llm | StrOutputParser()
            result = await chain.ainvoke({"question": query})

            # Parse JSON response
            try:
                parsed = json.loads(result)
                queries = [q["query"] for q in parsed.get("queries", [])]
            except (json.JSONDecodeError, KeyError):
                # Fallback: use original query only
                logger.warning("Failed to parse query expansion, using original")
                queries = []

            # Always include original query
            if query not in queries:
                queries.insert(0, query)

            # Limit to 4 queries max
            queries = queries[:4]

            logger.info("Expanded to %d queries", len(queries))
            return {
                "expanded_queries": queries,
                "retrieval_metadata": {"expansion_count": len(queries)},
            }

        except Exception as e:
            logger.exception("Query expansion failed")
            return {
                "expanded_queries": [query],
                "retrieval_metadata": {"expansion_error": str(e)},
            }

    # -------------------------------------------------------------------------
    # Node: parallel_retrieve
    # -------------------------------------------------------------------------
    async def parallel_retrieve(state: RAGState) -> dict[str, Any]:
        """Perform parallel retrieval across all expanded queries.

        Executes graph_enriched_search for each query concurrently.
        """
        queries = state.get("expanded_queries", [state["query"]])
        logger.info("Retrieving for %d queries", len(queries))

        async def retrieve_single(q: str) -> list[dict[str, Any]]:
            """Retrieve results for a single query."""
            try:
                results = await graph_enriched_search(
                    retriever=retriever,
                    driver=driver,
                    query=q,
                    limit=MAX_RESULTS_PER_QUERY,
                )
                # Tag results with source query
                for r in results:
                    r["_source_query"] = q
                return results
            except Exception as e:
                logger.warning("Retrieval failed for query '%s': %s", q[:30], e)
                return []

        # Execute all retrievals in parallel
        all_results = await asyncio.gather(
            *[retrieve_single(q) for q in queries],
            return_exceptions=True,
        )

        # Flatten results, filtering out exceptions
        raw_results = []
        for result in all_results:
            if isinstance(result, list):
                raw_results.extend(result)

        logger.info("Retrieved %d total raw results", len(raw_results))
        return {"raw_results": raw_results}

    # -------------------------------------------------------------------------
    # Node: dedupe_and_rank
    # -------------------------------------------------------------------------
    async def dedupe_and_rank(state: RAGState) -> dict[str, Any]:
        """Deduplicate and rank retrieval results.

        Removes duplicates based on chunk ID, then ranks by score.
        Converts to RetrievedDocument format.
        """
        raw_results = state.get("raw_results", [])
        logger.info("Deduplicating %d results", len(raw_results))

        # Deduplicate by chunk_id (or text hash if no ID)
        seen: set[str] = set()
        unique_results: list[dict[str, Any]] = []

        for result in raw_results:
            # Use chunk_id or hash of text as dedup key
            # chunk_id may be at top level or nested in metadata
            metadata = result.get("metadata", {})
            chunk_id = (
                result.get("chunk_id")
                or metadata.get("chunk_id")
                or result.get("id")
                or metadata.get("id")
            )
            if not chunk_id:
                # Fallback to content hash
                text = result.get("content") or result.get("text", "")
                chunk_id = str(hash(text)) if text else str(id(result))

            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_results.append(result)

        # Sort by score descending
        unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Limit total results
        unique_results = unique_results[:MAX_TOTAL_RESULTS]

        # Convert to RetrievedDocument format
        ranked_results = []
        for result in unique_results:
            # Get metadata dict (graph_enriched_search returns nested metadata)
            metadata = result.get("metadata", {})

            # Extract content - may be at 'text' or 'content' depending on source
            content = result.get("text") or result.get("content", "")

            # Extract source title from metadata (title) or top-level (article_title)
            source_title = (
                metadata.get("title")
                or metadata.get("article_title")
                or result.get("article_title")
                or "Unknown"
            )

            doc = RetrievedDocument(
                content=content,
                source=source_title,
                score=result.get("score", 0.0),
                metadata={
                    "chunk_id": metadata.get("chunk_id") or result.get("chunk_id"),
                    "url": metadata.get("url") or result.get("url"),
                    "chapter": metadata.get("chapter"),
                    "entities": result.get("entities", []),
                    "source_query": result.get("_source_query"),
                    # Preserve enriched data for frontend display
                    "media": result.get("media", {}),
                    "glossary_definitions": result.get("glossary_definitions", []),
                    "industry_standards": result.get("industry_standards", []),
                    "semantic_relationships": result.get("semantic_relationships", {}),
                },
            )
            ranked_results.append(doc)

        logger.info("Ranked %d unique results", len(ranked_results))

        # Update metadata
        metadata = state.get("retrieval_metadata", {})
        metadata.update(
            {
                "raw_count": len(raw_results),
                "unique_count": len(unique_results),
                "final_count": len(ranked_results),
            }
        )

        return {
            "ranked_results": ranked_results,
            "retrieval_metadata": metadata,
        }

    # -------------------------------------------------------------------------
    # Node: grade_documents
    # -------------------------------------------------------------------------
    async def grade_documents(state: RAGState) -> dict[str, Any]:
        """Grade each retrieved document for relevance.

        Uses an LLM with structured output to make a binary yes/no
        relevance decision per document. Filters out irrelevant docs
        and sets quality_pass based on the ratio of relevant documents.
        """
        ranked_results = state.get("ranked_results", [])
        query = state["query"]

        if not ranked_results:
            logger.info("No documents to grade")
            return {
                "ranked_results": [],
                "relevant_count": 0,
                "total_count": 0,
                "quality_pass": False,
            }

        llm = ChatOpenAI(
            model=config.conversational_model,
            temperature=0,
            api_key=config.openai_api_key,
        )
        grader = llm.with_structured_output(GradeDocuments)

        relevant_docs: list[RetrievedDocument] = []
        total = len(ranked_results)

        for doc in ranked_results:
            try:
                grade: GradeDocuments = await grader.ainvoke(
                    [
                        {
                            "role": "system",
                            "content": "You are a relevance grader. "
                            "Determine if the document is relevant "
                            "to the user's question. Respond with "
                            "binary_score 'yes' or 'no'.",
                        },
                        {
                            "role": "user",
                            "content": f"Document:\n{doc.content[:500]}\n\nQuestion: {query}",
                        },
                    ]
                )
                if grade.binary_score == "yes":
                    relevant_docs.append(doc)
            except Exception:
                logger.warning(
                    "Grading failed for doc '%s', keeping it",
                    doc.source[:30],
                )
                relevant_docs.append(doc)

        relevant_count = len(relevant_docs)
        quality_pass = (relevant_count / total) >= QUALITY_GATE_THRESHOLD

        logger.info(
            "Graded %d docs: %d relevant (%.0f%%), quality_pass=%s",
            total,
            relevant_count,
            (relevant_count / total) * 100,
            quality_pass,
        )

        return {
            "ranked_results": relevant_docs,
            "relevant_count": relevant_count,
            "total_count": total,
            "quality_pass": quality_pass,
        }

    # -------------------------------------------------------------------------
    # Build the subgraph
    # -------------------------------------------------------------------------
    builder = StateGraph(RAGState)

    # Add nodes
    builder.add_node("expand_queries", expand_queries)
    builder.add_node("parallel_retrieve", parallel_retrieve)
    builder.add_node("dedupe_and_rank", dedupe_and_rank)
    builder.add_node("grade_documents", grade_documents)

    # Add edges (linear flow)
    builder.add_edge(START, "expand_queries")
    builder.add_edge("expand_queries", "parallel_retrieve")
    builder.add_edge("parallel_retrieve", "dedupe_and_rank")
    builder.add_edge("dedupe_and_rank", "grade_documents")
    builder.add_edge("grade_documents", END)

    return builder.compile()


__all__ = ["create_rag_subgraph"]
