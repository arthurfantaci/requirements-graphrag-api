"""LangChain tool definitions for the agentic RAG system.

This module wraps existing retrieval functions as LangChain tools,
enabling the agent to dynamically select and invoke them.

Tools:
- graph_search: Wraps graph_enriched_search for hybrid vector+graph retrieval
- text2cypher: Wraps text2cypher_query for structured data queries
- explore_entity: Wraps explore_entity for entity relationship exploration
- lookup_standard: Wraps standard lookup for standards information
- search_definitions: Wraps search_terms for terminology search
- lookup_term: Wraps lookup_term for specific term definitions

Tool Design Principles:
1. Tools are thin wrappers - business logic stays in core modules
2. Include clear descriptions for LLM tool selection
3. Use Pydantic models for structured input validation
4. Handle errors gracefully and return informative messages
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from neo4j import Driver
    from neo4j_graphrag.retrievers import VectorRetriever

    from requirements_graphrag_api.config import AppConfig

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL INPUT SCHEMAS (Pydantic models for validation)
# =============================================================================


class GraphSearchInput(BaseModel):
    """Input schema for graph_search tool."""

    query: str = Field(
        description="The search query to find relevant content in the knowledge base."
    )
    limit: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Maximum number of results to return (1-20).",
    )


class Text2CypherInput(BaseModel):
    """Input schema for text2cypher tool."""

    question: str = Field(
        description=(
            "Natural language question to convert to Cypher. "
            "Use for structured queries like counts, lists, or specific lookups."
        )
    )
    execute: bool = Field(
        default=True,
        description="Whether to execute the generated query (default: True).",
    )


class ExploreEntityInput(BaseModel):
    """Input schema for explore_entity tool."""

    entity_name: str = Field(
        description="Name of the entity to explore (e.g., 'ISO 26262', 'Jama Connect')."
    )
    include_related: bool = Field(
        default=True,
        description="Whether to include related entities.",
    )
    related_limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of related entities to return.",
    )


class LookupStandardInput(BaseModel):
    """Input schema for lookup_standard tool."""

    name: str = Field(description="Name of the standard to look up (e.g., 'ISO 26262', 'DO-178C').")
    include_related: bool = Field(
        default=True,
        description="Whether to include related standards and industries.",
    )


class SearchStandardsInput(BaseModel):
    """Input schema for search_standards tool."""

    query: str = Field(description="Search query for finding standards.")
    industry: str | None = Field(
        default=None,
        description="Optional industry filter (e.g., 'automotive', 'aerospace', 'medical').",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of results.",
    )


class SearchDefinitionsInput(BaseModel):
    """Input schema for search_definitions tool."""

    query: str = Field(description="Search query for finding terminology definitions.")
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of definitions to return.",
    )


class LookupTermInput(BaseModel):
    """Input schema for lookup_term tool."""

    term: str = Field(description="The term to look up (e.g., 'traceability', 'ASIL', 'V&V').")
    fuzzy: bool = Field(
        default=True,
        description="Whether to use fuzzy matching for acronyms and variations.",
    )


# =============================================================================
# TOOL FACTORY FUNCTION
# =============================================================================


def create_agent_tools(
    config: AppConfig,
    driver: Driver,
    retriever: VectorRetriever,
) -> list[StructuredTool]:
    """Create all agent tools with bound dependencies.

    This factory function creates LangChain tools with the application
    dependencies (config, driver, retriever) already bound. This allows
    the agent to call tools without needing to pass these dependencies.

    Args:
        config: Application configuration.
        driver: Neo4j driver instance.
        retriever: Vector retriever instance.

    Returns:
        List of StructuredTool instances ready for agent use.
    """
    # Import here to avoid circular imports
    from requirements_graphrag_api.core.definitions import lookup_term, search_terms
    from requirements_graphrag_api.core.retrieval import (
        explore_entity as _explore_entity,
    )
    from requirements_graphrag_api.core.retrieval import (
        graph_enriched_search,
    )
    from requirements_graphrag_api.core.standards import (
        lookup_standard as _lookup_standard,
    )
    from requirements_graphrag_api.core.standards import (
        search_standards as _search_standards,
    )
    from requirements_graphrag_api.core.text2cypher import text2cypher_query

    # -------------------------------------------------------------------------
    # graph_search: Hybrid vector + graph retrieval
    # -------------------------------------------------------------------------
    async def graph_search_impl(query: str, limit: int = 6) -> str:
        """Search the knowledge base using hybrid vector + graph retrieval."""
        try:
            results = await graph_enriched_search(
                retriever=retriever,
                driver=driver,
                query=query,
                limit=limit,
            )
            if not results:
                return "No results found for the query."

            # Format results for the agent
            formatted = []
            for i, doc in enumerate(results, 1):
                source = doc.get("article_title", "Unknown")
                text = doc.get("text", "")[:500]  # Truncate for context
                score = doc.get("score", 0)
                formatted.append(f"[{i}] {source} (score: {score:.2f}):\n{text}")

            return "\n\n".join(formatted)
        except Exception as e:
            logger.exception("graph_search failed")
            return f"Error searching knowledge base: {e}"

    graph_search_tool = StructuredTool.from_function(
        func=graph_search_impl,
        coroutine=graph_search_impl,
        name="graph_search",
        description=(
            "Search the Requirements Management knowledge base using hybrid "
            "vector similarity + graph context. Use this for finding explanations, "
            "concepts, best practices, and detailed information. Returns ranked "
            "content chunks with source citations."
        ),
        args_schema=GraphSearchInput,
    )

    # -------------------------------------------------------------------------
    # text2cypher: Natural language to Cypher queries
    # -------------------------------------------------------------------------
    async def text2cypher_impl(question: str, execute: bool = True) -> str:
        """Convert natural language to Cypher and optionally execute."""
        try:
            result = await text2cypher_query(
                config=config,
                driver=driver,
                question=question,
                execute=execute,
            )
            cypher = result.get("cypher", "")
            if not execute:
                return f"Generated Cypher:\n```cypher\n{cypher}\n```"

            if "error" in result:
                return f"Query error: {result['error']}\nGenerated Cypher: {cypher}"

            results = result.get("results", [])
            row_count = result.get("row_count", 0)

            if not results:
                return f"Query returned no results.\nCypher: {cypher}"

            # Format results as a table-like string
            formatted = [f"Query returned {row_count} results:"]
            for row in results[:25]:  # Limit output
                formatted.append(str(row))

            formatted.append(f"\nCypher: {cypher}")
            return "\n".join(formatted)
        except Exception as e:
            logger.exception("text2cypher failed")
            return f"Error executing Cypher query: {e}"

    text2cypher_tool = StructuredTool.from_function(
        func=text2cypher_impl,
        coroutine=text2cypher_impl,
        name="text2cypher",
        description=(
            "Convert natural language questions to Cypher queries for the Neo4j "
            "knowledge graph. Use this for structured queries like: counts, lists, "
            "enumerations, specific lookups, or queries about relationships between "
            "entities. Examples: 'How many articles mention ISO 26262?', "
            "'List all webinars', 'Which standards apply to automotive?'"
        ),
        args_schema=Text2CypherInput,
    )

    # -------------------------------------------------------------------------
    # explore_entity: Deep entity exploration
    # -------------------------------------------------------------------------
    async def explore_entity_impl(
        entity_name: str,
        include_related: bool = True,
        related_limit: int = 10,
    ) -> str:
        """Explore an entity and its relationships."""
        try:
            result = await _explore_entity(
                driver=driver,
                entity_name=entity_name,
                include_related=include_related,
                related_limit=related_limit,
            )
            if not result:
                return f"Entity '{entity_name}' not found in the knowledge graph."

            # Format entity information
            lines = [
                f"Entity: {result.get('display_name', entity_name)}",
                f"Type: {result.get('type', 'Unknown')}",
            ]

            if result.get("description"):
                lines.append(f"Description: {result['description']}")

            related = result.get("related_entities", [])
            if related:
                lines.append(f"\nRelated Entities ({len(related)}):")
                for rel in related[:10]:
                    lines.append(f"  - {rel.get('name', 'Unknown')} ({rel.get('type', '')})")

            articles = result.get("mentioned_in", [])
            if articles:
                lines.append(f"\nMentioned in {len(articles)} articles:")
                for art in articles[:5]:
                    lines.append(f"  - {art}")

            return "\n".join(lines)
        except Exception as e:
            logger.exception("explore_entity failed")
            return f"Error exploring entity: {e}"

    explore_entity_tool = StructuredTool.from_function(
        func=explore_entity_impl,
        coroutine=explore_entity_impl,
        name="explore_entity",
        description=(
            "Explore a specific entity (tool, concept, standard, methodology) "
            "in the knowledge graph. Returns the entity's details, related "
            "entities, and articles where it's mentioned. Use this to deep-dive "
            "into a specific topic or understand relationships."
        ),
        args_schema=ExploreEntityInput,
    )

    # -------------------------------------------------------------------------
    # lookup_standard: Standard information lookup
    # -------------------------------------------------------------------------
    async def lookup_standard_impl(
        name: str,
        include_related: bool = True,
    ) -> str:
        """Look up a specific standard."""
        try:
            result = await _lookup_standard(
                driver=driver,
                name=name,
                include_related=include_related,
            )
            if not result:
                return f"Standard '{name}' not found."

            lines = [
                f"Standard: {result.get('display_name', name)}",
                f"Organization: {result.get('organization', 'Unknown')}",
            ]

            industries = result.get("industries", [])
            if industries:
                lines.append(f"Industries: {', '.join(industries)}")

            related = result.get("related_standards", [])
            if related:
                lines.append(f"\nRelated Standards: {', '.join(related)}")

            articles = result.get("mentioned_in", [])
            if articles:
                lines.append(f"\nMentioned in {len(articles)} articles")

            return "\n".join(lines)
        except Exception as e:
            logger.exception("lookup_standard failed")
            return f"Error looking up standard: {e}"

    lookup_standard_tool = StructuredTool.from_function(
        func=lookup_standard_impl,
        coroutine=lookup_standard_impl,
        name="lookup_standard",
        description=(
            "Look up information about a specific industry standard "
            "(e.g., ISO 26262, DO-178C, IEC 62304). Returns the standard's "
            "details, applicable industries, and related standards."
        ),
        args_schema=LookupStandardInput,
    )

    # -------------------------------------------------------------------------
    # search_standards: Search for standards
    # -------------------------------------------------------------------------
    async def search_standards_impl(
        query: str,
        industry: str | None = None,
        limit: int = 10,
    ) -> str:
        """Search for standards matching criteria."""
        try:
            results = await _search_standards(
                driver=driver,
                query=query,
                industry=industry,
                limit=limit,
            )
            if not results:
                filter_msg = f" for industry '{industry}'" if industry else ""
                return f"No standards found matching '{query}'{filter_msg}."

            lines = [f"Found {len(results)} standards:"]
            for std in results:
                name = std.get("display_name", std.get("name", "Unknown"))
                org = std.get("organization", "")
                org_str = f" ({org})" if org else ""
                lines.append(f"  - {name}{org_str}")

            return "\n".join(lines)
        except Exception as e:
            logger.exception("search_standards failed")
            return f"Error searching standards: {e}"

    search_standards_tool = StructuredTool.from_function(
        func=search_standards_impl,
        coroutine=search_standards_impl,
        name="search_standards",
        description=(
            "Search for industry standards in the knowledge base. Can filter "
            "by industry (automotive, aerospace, medical, etc.). Use this to "
            "find relevant standards for a domain or topic."
        ),
        args_schema=SearchStandardsInput,
    )

    # -------------------------------------------------------------------------
    # search_definitions: Search terminology definitions
    # -------------------------------------------------------------------------
    async def search_definitions_impl(query: str, limit: int = 10) -> str:
        """Search for terminology definitions."""
        try:
            results = await search_terms(
                driver=driver,
                query=query,
                limit=limit,
            )
            if not results:
                return f"No definitions found for '{query}'."

            lines = [f"Found {len(results)} definitions:"]
            for defn in results:
                term = defn.get("term", "Unknown")
                acronym = defn.get("acronym", "")
                definition = defn.get("definition", "")[:200]
                acronym_str = f" ({acronym})" if acronym else ""
                lines.append(f"\n**{term}{acronym_str}**")
                lines.append(f"  {definition}...")

            return "\n".join(lines)
        except Exception as e:
            logger.exception("search_definitions failed")
            return f"Error searching definitions: {e}"

    search_definitions_tool = StructuredTool.from_function(
        func=search_definitions_impl,
        coroutine=search_definitions_impl,
        name="search_definitions",
        description=(
            "Search for terminology definitions in the Requirements Management "
            "glossary. Use this to find definitions of technical terms, acronyms, "
            "or concepts mentioned in the knowledge base."
        ),
        args_schema=SearchDefinitionsInput,
    )

    # -------------------------------------------------------------------------
    # lookup_term: Look up specific term definition
    # -------------------------------------------------------------------------
    async def lookup_term_impl(term: str, fuzzy: bool = True) -> str:
        """Look up a specific term definition."""
        try:
            result = await lookup_term(
                driver=driver,
                term=term,
                fuzzy=fuzzy,
            )
            if not result:
                return f"Definition for '{term}' not found."

            lines = [
                f"**{result.get('term', term)}**",
            ]

            acronym = result.get("acronym")
            if acronym:
                lines.append(f"Acronym: {acronym}")

            definition = result.get("definition", "No definition available.")
            lines.append(f"\nDefinition: {definition}")

            url = result.get("url")
            if url:
                lines.append(f"\nSource: {url}")

            return "\n".join(lines)
        except Exception as e:
            logger.exception("lookup_term failed")
            return f"Error looking up term: {e}"

    lookup_term_tool = StructuredTool.from_function(
        func=lookup_term_impl,
        coroutine=lookup_term_impl,
        name="lookup_term",
        description=(
            "Look up the definition of a specific term or acronym. Use this "
            "when you need the precise definition of a Requirements Management "
            "term (e.g., 'traceability', 'ASIL', 'V&V', 'MBSE')."
        ),
        args_schema=LookupTermInput,
    )

    # -------------------------------------------------------------------------
    # Return all tools
    # -------------------------------------------------------------------------
    return [
        graph_search_tool,
        text2cypher_tool,
        explore_entity_tool,
        lookup_standard_tool,
        search_standards_tool,
        search_definitions_tool,
        lookup_term_tool,
    ]


__all__ = [
    "ExploreEntityInput",
    "GraphSearchInput",
    "LookupStandardInput",
    "LookupTermInput",
    "SearchDefinitionsInput",
    "SearchStandardsInput",
    "Text2CypherInput",
    "create_agent_tools",
]
