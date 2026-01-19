"""FastMCP server entry point for Requirements Management GraphRAG.

This module initializes the MCP server with GraphRAG tools,
managing the lifecycle of Neo4j connections and vector stores.

Updated Data Model (2026-01):
- Uses neo4j Driver directly with neo4j-graphrag VectorRetriever
- Chunks contain text directly, linked via FROM_ARTICLE
- MENTIONED_IN relationship direction: Entity -> Chunk
- Definition nodes replace GlossaryTerm

Neo4j Driver Best Practices Applied:
- Single driver instance created once and reused
- Connectivity verified at startup (fail fast)
- Connection pool sized for serverless (Vercel)
- Proper cleanup on shutdown
"""

from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from neo4j import GraphDatabase

from jama_mcp_server_graphrag.config import get_config
from jama_mcp_server_graphrag.core import (
    chat,
    create_vector_retriever,
    explore_entity,
    get_standards_by_industry,
    graph_enriched_search,
    hybrid_search,
    list_all_standards,
    list_all_terms,
    lookup_standard,
    lookup_term,
    search_standards,
    search_terms,
    text2cypher_query,
    vector_search,
)
from jama_mcp_server_graphrag.exceptions import Neo4jConnectionError
from jama_mcp_server_graphrag.observability import configure_tracing

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from neo4j import Driver
    from neo4j_graphrag.retrievers import VectorRetriever

    from jama_mcp_server_graphrag.config import AppConfig

# Load environment variables
load_dotenv()

logger: logging.Logger = logging.getLogger(__name__)


# Server resources holder (initialized in lifespan)
class _ServerResources:
    """Holder for server-wide shared resources."""

    config: AppConfig | None = None
    driver: Driver | None = None
    retriever: VectorRetriever | None = None


_resources = _ServerResources()


def create_driver(config: AppConfig) -> Driver:
    """Create Neo4j driver with best practices.

    Applies Neo4j driver best practices:
    - Uses neo4j+s:// scheme for secure connections
    - Verifies connectivity immediately (fail fast)
    - Configures pool size for serverless

    Args:
        config: Application configuration.

    Returns:
        Connected Neo4j Driver instance.

    Raises:
        Neo4jConnectionError: If connection verification fails.
    """
    logger.info("Creating Neo4j connection to %s", config.neo4j_uri.split("@")[-1])

    try:
        driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_username, config.neo4j_password),
            max_connection_pool_size=config.neo4j_max_connection_pool_size,
            connection_acquisition_timeout=config.neo4j_connection_acquisition_timeout,
        )

        # Best Practice: Verify connectivity immediately
        driver.verify_connectivity()
    except Exception as e:
        logger.exception("Failed to connect to Neo4j")
        raise Neo4jConnectionError(f"Failed to connect to Neo4j at {config.neo4j_uri}: {e}") from e
    else:
        logger.info("Neo4j connectivity verified successfully")
        return driver


@asynccontextmanager
async def server_lifespan(_mcp_server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Manage server lifecycle and resources.

    Implements Neo4j driver best practices:
    - Creates single driver instance (reused across all requests)
    - Verifies connectivity at startup (fail fast on bad config)
    - Properly cleans up resources on shutdown

    Args:
        mcp_server: The FastMCP server instance.

    Yields:
        Empty dictionary (resources stored in module-level variables).
    """
    logger.info("Starting Requirements GraphRAG MCP Server")

    _resources.config = get_config()

    # Configure LangSmith tracing if enabled
    if configure_tracing(_resources.config):
        logger.info(
            "LangSmith tracing configured for project: %s",
            _resources.config.langsmith_project,
        )

    # Log configuration (without sensitive data)
    logger.info(
        "Configuration loaded: database=%s, model=%s, embedding=%s",
        _resources.config.neo4j_database,
        _resources.config.chat_model,
        _resources.config.embedding_model,
    )

    # Initialize Neo4j driver
    _resources.driver = create_driver(_resources.config)

    # Initialize vector retriever for similarity search
    logger.info(
        "Initializing vector retriever with index '%s'",
        _resources.config.vector_index_name,
    )
    _resources.retriever = create_vector_retriever(_resources.driver, _resources.config)
    logger.info("Vector retriever initialized successfully")

    logger.info("All resources initialized - server ready to accept requests")

    # Yield empty dict (resources accessed via _resources holder)
    yield {}

    # Cleanup on shutdown
    logger.info("Shutting down GraphRAG MCP Server - cleaning up resources")
    if _resources.driver:
        _resources.driver.close()
    _resources.config = None
    _resources.driver = None
    _resources.retriever = None


# Initialize the MCP server
mcp = FastMCP(
    name="jama_graphrag_mcp",
    instructions="""
    Requirements Management GraphRAG Server - Intelligent access to requirements
    engineering knowledge including best practices, industry standards, tools,
    and methodologies from the Jama Software "Essential Guide to Requirements
    Management and Traceability".

    Available capabilities:

    **Search & Retrieval:**
    - `graphrag_vector_search` - Semantic similarity search on chunk embeddings
    - `graphrag_hybrid_search` - Combined vector + keyword search
    - `graphrag_graph_enriched_search` - Vector search + knowledge graph context
    - `graphrag_explore_entity` - Deep dive into specific entities

    **Definitions:**
    - `graphrag_lookup_term` - Look up a specific term definition
    - `graphrag_search_terms` - Search definition terms
    - `graphrag_list_terms` - List all definition terms

    **Standards:**
    - `graphrag_lookup_standard` - Look up a specific standard (ISO, FDA, etc.)
    - `graphrag_search_standards` - Search standards by name or description
    - `graphrag_standards_by_industry` - Get standards for an industry
    - `graphrag_list_standards` - List all standards

    **Advanced:**
    - `graphrag_text2cypher` - Convert natural language to Cypher queries
    - `graphrag_chat` - RAG-powered Q&A with citations
    - `graphrag_schema` - Explore the knowledge graph structure

    **Resources (read via resource:// URIs):**
    - `schema://graph` - Knowledge graph schema and structure
    - `glossary://terms` - Complete glossary of RM terminology
    - `standards://list` - Industry standards reference
    - `guide://overview` - Guide overview and usage tips

    **Domain Coverage:**
    - Requirements writing, gathering, and management processes
    - Traceability and impact analysis
    - Validation and verification
    - Industry standards (ISO, FDA, DO-178C, INCOSE, MIL-STD)
    - Industry applications (Automotive, Medical, Aerospace, Defense)
    - Tools (Jama Connect, DOORS, traditional approaches)
    """,
    lifespan=server_lifespan,
)


@mcp.tool()
async def graphrag_vector_search(
    query: str,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Search for relevant content using semantic similarity.

    Performs vector similarity search on chunk embeddings to find
    the most relevant passages from the requirements management guide.

    Args:
        query: Natural language search query.
        limit: Maximum number of results to return (default: 6).

    Returns:
        List of relevant chunks with content and metadata.

    Example queries:
        - "What is requirements traceability?"
        - "How to validate requirements?"
        - "ISO 26262 ASIL levels"
    """
    if _resources.retriever is None or _resources.driver is None:
        msg = "Server not initialized. Resources unavailable."
        raise RuntimeError(msg)

    logger.info("MCP tool: graphrag_vector_search query='%s', limit=%d", query, limit)
    return await vector_search(_resources.retriever, _resources.driver, query, limit=limit)


@mcp.tool()
async def graphrag_hybrid_search(
    query: str,
    limit: int = 6,
    keyword_weight: float = 0.3,
) -> list[dict[str, Any]]:
    """Search using combined vector similarity and keyword matching.

    Hybrid search provides better results when queries contain specific
    terms that should match exactly (e.g., standard names, acronyms).

    Args:
        query: Natural language search query.
        limit: Maximum number of results (default: 6).
        keyword_weight: Weight for keyword vs vector (0.0-1.0, default: 0.3).

    Returns:
        List of results with combined scores.

    Example queries:
        - "FDA 21 CFR Part 11 validation"
        - "ASPICE process areas"
        - "Jama Connect traceability"
    """
    if _resources.retriever is None or _resources.driver is None:
        msg = "Server not initialized. Resources unavailable."
        raise RuntimeError(msg)

    logger.info("MCP tool: graphrag_hybrid_search query='%s'", query)
    return await hybrid_search(
        _resources.retriever, _resources.driver, query, limit=limit, keyword_weight=keyword_weight
    )


@mcp.tool()
async def graphrag_graph_enriched_search(
    query: str,
    limit: int = 6,
    traversal_depth: int = 1,
) -> list[dict[str, Any]]:
    """Search with knowledge graph enrichment.

    Combines vector search with graph traversal to include related
    entities, glossary terms, and context from the knowledge graph.

    Args:
        query: Natural language search query.
        limit: Maximum number of results (default: 6).
        traversal_depth: How many hops to traverse (1-2, default: 1).

    Returns:
        Results enriched with related entities and terms.

    Best for: Understanding concepts in context with relationships.
    """
    if _resources.retriever is None or _resources.driver is None:
        msg = "Server not initialized. Resources unavailable."
        raise RuntimeError(msg)

    logger.info("MCP tool: graphrag_graph_enriched_search query='%s'", query)
    return await graph_enriched_search(
        _resources.retriever, _resources.driver, query, limit=limit, traversal_depth=traversal_depth
    )


@mcp.tool()
async def graphrag_explore_entity(
    entity_name: str,
    include_related: bool = True,
) -> dict[str, Any] | None:
    """Explore a specific entity in the knowledge graph.

    Returns detailed information about an entity including its
    definition, relationships, and mentions in articles.

    Args:
        entity_name: Name of the entity to explore.
        include_related: Whether to include related entities (default: True).

    Returns:
        Entity details or None if not found.

    Example entities: "requirements traceability", "V-model", "ASIL"
    """
    if _resources.driver is None:
        msg = "Server not initialized. Driver unavailable."
        raise RuntimeError(msg)

    logger.info("MCP tool: graphrag_explore_entity name='%s'", entity_name)
    return await explore_entity(_resources.driver, entity_name, include_related=include_related)


@mcp.tool()
async def graphrag_schema() -> dict[str, Any]:
    """Explore the knowledge graph schema.

    Returns information about node labels, relationship types,
    and their counts to help understand the graph structure.

    Returns:
        Dictionary with node_labels, relationship_types, and counts.
    """
    if _resources.driver is None:
        msg = "Server not initialized. Driver unavailable."
        raise RuntimeError(msg)

    logger.info("Fetching graph schema")

    with _resources.driver.session() as session:
        # Get node label counts
        node_result = session.run(
            """
            MATCH (n)
            WITH labels(n) AS labels
            UNWIND labels AS label
            RETURN label, count(*) AS count
            ORDER BY count DESC
            """
        )
        node_counts = [dict(record) for record in node_result]

        # Get relationship type counts
        rel_result = session.run(
            """
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(*) AS count
            ORDER BY count DESC
            LIMIT 20
            """
        )
        rel_counts = [dict(record) for record in rel_result]

    return {
        "node_labels": node_counts,
        "relationship_types": rel_counts,
    }


# =============================================================================
# Definition Tools (formerly Glossary)
# =============================================================================


@mcp.tool()
async def graphrag_lookup_term(
    term: str,
    fuzzy: bool = True,
) -> dict[str, Any] | None:
    """Look up a definition term.

    Searches the definitions for a specific term and returns its definition.
    Supports fuzzy matching for partial or misspelled terms.

    Args:
        term: Term to look up (e.g., "traceability", "SDLC").
        fuzzy: Use fuzzy matching (default: True).

    Returns:
        Term definition or None if not found.

    Example terms: "requirement", "verification", "validation"
    """
    if _resources.driver is None:
        msg = "Server not initialized. Driver unavailable."
        raise RuntimeError(msg)

    logger.info("MCP tool: graphrag_lookup_term term='%s'", term)
    return await lookup_term(_resources.driver, term, fuzzy=fuzzy)


@mcp.tool()
async def graphrag_search_terms(
    query: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search for definition terms matching a query.

    Searches term names and definitions to find relevant terminology.

    Args:
        query: Search query.
        limit: Maximum number of results (default: 10).

    Returns:
        List of matching terms with definitions.
    """
    if _resources.driver is None:
        msg = "Server not initialized. Driver unavailable."
        raise RuntimeError(msg)

    logger.info("MCP tool: graphrag_search_terms query='%s'", query)
    return await search_terms(_resources.driver, query, limit=limit)


@mcp.tool()
async def graphrag_list_terms(
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List all definition terms alphabetically.

    Args:
        limit: Maximum number of terms to return (default: 50).

    Returns:
        List of all terms with definitions.
    """
    if _resources.driver is None:
        msg = "Server not initialized. Driver unavailable."
        raise RuntimeError(msg)

    logger.info("MCP tool: graphrag_list_terms limit=%d", limit)
    return await list_all_terms(_resources.driver, limit=limit)


# =============================================================================
# Standards Tools
# =============================================================================


@mcp.tool()
async def graphrag_lookup_standard(
    name: str,
    include_related: bool = True,
) -> dict[str, Any] | None:
    """Look up a specific industry standard.

    Returns detailed information about standards including
    related entities and articles that mention them.

    Args:
        name: Standard name (e.g., "ISO 26262", "FDA", "DO-178C").
        include_related: Include related entities (default: True).

    Returns:
        Standard details or None if not found.

    Example standards: "ISO 26262", "FDA", "IEC 62304", "ASPICE"
    """
    if _resources.driver is None:
        msg = "Server not initialized. Driver unavailable."
        raise RuntimeError(msg)

    logger.info("MCP tool: graphrag_lookup_standard name='%s'", name)
    return await lookup_standard(_resources.driver, name, include_related=include_related)


@mcp.tool()
async def graphrag_search_standards(
    query: str,
    industry: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search for standards matching criteria.

    Args:
        query: Search query (name, organization, or description).
        industry: Optional filter by industry (e.g., "automotive").
        limit: Maximum number of results (default: 10).

    Returns:
        List of matching standards.
    """
    if _resources.driver is None:
        msg = "Server not initialized. Driver unavailable."
        raise RuntimeError(msg)

    logger.info("MCP tool: graphrag_search_standards query='%s', industry='%s'", query, industry)
    return await search_standards(_resources.driver, query, industry=industry, limit=limit)


@mcp.tool()
async def graphrag_standards_by_industry(
    industry: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Get standards applicable to a specific industry.

    Supported industries: automotive, medical, aerospace, defense, rail.

    Args:
        industry: Industry name.
        limit: Maximum number of results (default: 10).

    Returns:
        List of standards for the industry.

    Example: graphrag_standards_by_industry("automotive")
    """
    if _resources.driver is None:
        msg = "Server not initialized. Driver unavailable."
        raise RuntimeError(msg)

    logger.info("MCP tool: graphrag_standards_by_industry industry='%s'", industry)
    return await get_standards_by_industry(_resources.driver, industry, limit=limit)


@mcp.tool()
async def graphrag_list_standards(
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List all standards in the knowledge graph.

    Args:
        limit: Maximum number of standards to return (default: 50).

    Returns:
        List of all standards.
    """
    if _resources.driver is None:
        msg = "Server not initialized. Driver unavailable."
        raise RuntimeError(msg)

    logger.info("MCP tool: graphrag_list_standards limit=%d", limit)
    return await list_all_standards(_resources.driver, limit=limit)


# =============================================================================
# Text2Cypher Tool
# =============================================================================


@mcp.tool()
async def graphrag_text2cypher(
    question: str,
    execute: bool = True,
) -> dict[str, Any]:
    """Convert natural language to Cypher and optionally execute.

    Uses LLM to translate questions into Cypher queries for the
    requirements management knowledge graph. Only read queries allowed.

    Args:
        question: Natural language question about the graph.
        execute: Whether to execute the query (default: True).

    Returns:
        Dictionary with generated query and optional results.

    Example questions:
        - "How many chapters are there?"
        - "Which entities are mentioned most?"
        - "What standards apply to automotive?"
    """
    if _resources.config is None or _resources.driver is None:
        msg = "Server not initialized. Resources unavailable."
        raise RuntimeError(msg)

    logger.info("MCP tool: graphrag_text2cypher question='%s'", question)
    return await text2cypher_query(_resources.config, _resources.driver, question, execute=execute)


# =============================================================================
# Chat Tool
# =============================================================================


@mcp.tool()
async def graphrag_chat(
    message: str,
    max_sources: int = 5,
) -> dict[str, Any]:
    """Chat with RAG-powered Q&A including citations and images.

    Answers questions about requirements management using the
    knowledge graph for retrieval and LLM for generation.
    Responses include source citations and relevant images.

    Args:
        message: User's question or message.
        max_sources: Maximum sources to cite (default: 5).

    Returns:
        Dictionary with:
        - answer: Generated response text
        - sources: List of cited sources with title, url, relevance_score
        - entities: Related entities mentioned in the sources
        - images: List of relevant images with url, alt_text, context, source_title

    Example messages:
        - "What is requirements traceability and why is it important?"
        - "Explain the V-model approach to verification"
        - "How do I write good requirements?"
    """
    if _resources.config is None or _resources.driver is None or _resources.retriever is None:
        msg = "Server not initialized. Resources unavailable."
        raise RuntimeError(msg)

    logger.info("MCP tool: graphrag_chat message='%s'", message[:50])
    return await chat(
        _resources.config,
        _resources.retriever,
        _resources.driver,
        message,
        max_sources=max_sources,
    )


# =============================================================================
# MCP Resources
# =============================================================================


@mcp.resource("schema://graph")
async def resource_graph_schema() -> str:
    """Knowledge graph schema with node labels and relationships.

    Provides the complete schema of the requirements management
    knowledge graph including all node types and relationships.
    """
    if _resources.driver is None:
        msg = "Server not initialized. Driver unavailable."
        raise RuntimeError(msg)

    with _resources.driver.session() as session:
        # Get node label counts
        node_result = session.run(
            """
            MATCH (n)
            WITH labels(n) AS labels
            UNWIND labels AS label
            RETURN label, count(*) AS count
            ORDER BY count DESC
            """
        )
        node_counts = [dict(record) for record in node_result]

        # Get relationship type counts
        rel_result = session.run(
            """
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(*) AS count
            ORDER BY count DESC
            LIMIT 20
            """
        )
        rel_counts = [dict(record) for record in rel_result]

    # Format as readable text
    lines = [
        "# Requirements Management Knowledge Graph Schema",
        "",
        "## Node Labels",
    ]

    lines.extend(f"- **{node['label']}**: {node['count']} nodes" for node in node_counts)

    lines.extend(
        [
            "",
            "## Relationship Types",
        ]
    )

    lines.extend(f"- **{rel['type']}**: {rel['count']} relationships" for rel in rel_counts)

    lines.extend(
        [
            "",
            "## Key Patterns",
            "- `(Chapter)-[:CONTAINS]->(Article)` - Book structure",
            "- `(Chunk)-[:FROM_ARTICLE]->(Article)` - Chunk to article link",
            "- `(Entity)-[:MENTIONED_IN]->(Chunk)` - Entity mentions in chunks",
            "- `(Article)-[:HAS_IMAGE]->(Image)` - Article images",
            "- `(Concept)-[:ADDRESSES]->(Challenge)` - Concepts solving challenges",
            "- `(Standard)-[:APPLIES_TO]->(Industry)` - Industry standards",
        ]
    )

    logger.info("MCP resource: schema://graph accessed")
    return "\n".join(lines)


@mcp.resource("glossary://terms")
async def resource_glossary_terms() -> str:
    """List of all definition terms and their definitions.

    Provides a complete glossary of requirements management terminology
    from the knowledge base.
    """
    if _resources.driver is None:
        msg = "Server not initialized. Driver unavailable."
        raise RuntimeError(msg)

    with _resources.driver.session() as session:
        result = session.run(
            """
            MATCH (d:Definition)
            RETURN d.term AS term, d.definition AS definition
            ORDER BY d.term
            LIMIT 100
            """
        )
        terms = [dict(record) for record in result]

    lines = [
        "# Requirements Management Glossary",
        "",
        "Key terms and definitions from the Jama Software guide.",
        "",
    ]

    for term in terms:
        term_name = term.get("term", "Unknown")
        definition = term.get("definition", "No definition available")
        lines.append(f"## {term_name}")
        lines.append(f"{definition}")
        lines.append("")

    logger.info("MCP resource: glossary://terms accessed, %d terms", len(terms))
    return "\n".join(lines)


@mcp.resource("standards://list")
async def resource_standards_list() -> str:
    """List of industry standards covered in the knowledge base.

    Provides information about requirements management standards
    including ISO, FDA, DO-178C, ASPICE, and more.
    """
    if _resources.driver is None:
        msg = "Server not initialized. Driver unavailable."
        raise RuntimeError(msg)

    with _resources.driver.session() as session:
        result = session.run(
            """
            MATCH (s:Standard)
            RETURN s.name AS name, s.display_name AS display_name, s.organization AS org
            ORDER BY s.name
            LIMIT 50
            """
        )
        standards = [dict(record) for record in result]

    lines = [
        "# Industry Standards for Requirements Management",
        "",
        "Standards covered in the knowledge base:",
        "",
    ]

    for std in standards:
        name = std.get("display_name") or std.get("name", "Unknown")
        org = std.get("org", "")
        org_str = f" ({org})" if org else ""
        lines.append(f"- **{name}**{org_str}")

    logger.info("MCP resource: standards://list accessed, %d standards", len(standards))
    return "\n".join(lines)


@mcp.resource("guide://overview")
async def resource_guide_overview() -> str:
    """Overview of the requirements management knowledge base.

    Provides context about the Jama Software Essential Guide
    and how to use the GraphRAG system effectively.
    """
    return """# Jama Software Essential Guide to Requirements Management

## About This Knowledge Base

This GraphRAG system provides intelligent access to the complete content of
Jama Software's "Essential Guide to Requirements Management and Traceability".

## Coverage Areas

### Requirements Engineering
- Requirements elicitation and gathering techniques
- Writing effective requirements (SMART criteria)
- Requirements prioritization methods
- Change management processes

### Traceability
- Forward and backward traceability
- Impact analysis
- Coverage analysis
- Compliance traceability

### Verification & Validation
- V-model approach
- Testing strategies
- Review techniques
- Acceptance criteria

### Industry Standards
- **Automotive**: ISO 26262, ASPICE
- **Medical**: FDA 21 CFR Part 11, IEC 62304
- **Aerospace**: DO-178C, DO-254
- **Defense**: MIL-STD-498, MIL-STD-882
- **General**: ISO/IEC 12207, IEEE 830

### Tools & Methodologies
- Jama Connect features
- Traditional approaches (spreadsheets, documents)
- Agile requirements management
- Model-based systems engineering (MBSE)

## How to Use

1. **Search**: Use `graphrag_vector_search` for semantic queries
2. **Explore**: Use `graphrag_explore_entity` to dive deep into concepts
3. **Standards**: Use `graphrag_lookup_standard` for compliance guidance
4. **Glossary**: Use `graphrag_lookup_term` for terminology
5. **Chat**: Use `graphrag_chat` for conversational Q&A with citations

## Sample Questions

- "What is requirements traceability and why is it important?"
- "How do ASIL levels work in ISO 26262?"
- "What are the best practices for writing requirements?"
- "Explain the V-model approach to verification"
- "What standards apply to medical device software?"
"""


def main() -> None:
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="Requirements GraphRAG MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport (default: 8000)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the server
    if args.transport == "http":
        mcp.run(transport="streamable-http", port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
