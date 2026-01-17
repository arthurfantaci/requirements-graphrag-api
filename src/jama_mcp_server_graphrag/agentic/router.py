"""Retriever router for agentic RAG.

Analyzes user queries and routes them to the most appropriate
retrieval tool based on query characteristics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from jama_mcp_server_graphrag.config import AppConfig

logger = logging.getLogger(__name__)

# Tool descriptions for routing decisions
RETRIEVER_TOOLS: Final[dict[str, str]] = {
    "graphrag_vector_search": """
        Basic semantic search using vector embeddings.
        Use for: General content lookup, finding relevant passages.
        Example: "What does the guide say about traceability?"
    """,
    "graphrag_hybrid_search": """
        Combined vector + keyword search for improved accuracy.
        Use for: Queries with specific technical terms or acronyms.
        Example: "ISO 26262 ASIL requirements"
    """,
    "graphrag_graph_enriched_search": """
        Graph-enriched retrieval with entity relationships.
        Use for: Understanding how concepts relate to each other.
        Example: "How does V-Model relate to requirements validation?"
    """,
    "graphrag_explore_entity": """
        Deep dive into specific entities and their relationships.
        Use for: Learning about a specific concept, standard, or tool.
        Example: "Tell me about ISO 26262" or "What is requirements traceability?"
    """,
    "graphrag_lookup_standard": """
        Standards and compliance lookup.
        Use for: Regulatory requirements, industry standards, certifications.
        Example: "What standards apply to medical devices?"
    """,
    "graphrag_lookup_term": """
        Term definitions from the glossary.
        Use for: Understanding specific terminology.
        Example: "Define baseline" or "What is an atomic requirement?"
    """,
    "graphrag_text2cypher": """
        Natural language to Cypher query generation.
        Use for: Complex queries, aggregations, specific patterns.
        Example: "Which chapter has the most entities?" or "List all tools"
    """,
    "graphrag_chat": """
        Full RAG conversational Q&A with citations.
        Use for: Complex questions requiring synthesis from multiple sources.
        Example: "How should I approach requirements management for a medical device?"
    """,
}

ROUTER_SYSTEM_PROMPT: Final[str] = """
You are a retrieval router for a Requirements Management knowledge graph.

Your task is to analyze the user's question and select the best retrieval tool(s).

Available tools:
{tools}

Selection Guidelines:
1. For simple lookups or general questions -> graphrag_vector_search or graphrag_hybrid_search
2. For questions about how concepts relate -> graphrag_graph_enriched_search
3. For deep dives into specific entities -> graphrag_explore_entity
4. For regulatory/compliance questions -> graphrag_lookup_standard
5. For terminology definitions -> graphrag_lookup_term
6. For aggregations or complex patterns -> graphrag_text2cypher
7. For multi-faceted questions requiring synthesis -> graphrag_chat

You may select multiple tools if the question has multiple parts.

Return a JSON object with:
{{
    "selected_tools": ["tool_name1", "tool_name2"],
    "reasoning": "Brief explanation of why these tools were selected",
    "tool_params": {{
        "tool_name1": {{"query": "refined query for this tool"}},
        "tool_name2": {{"query": "refined query for this tool"}}
    }}
}}

User Question: {question}
"""


@dataclass
class RoutingResult:
    """Result of routing a query to retrieval tools."""

    selected_tools: list[str]
    reasoning: str
    tool_params: dict[str, dict[str, Any]]
    raw_response: str


async def route_query(
    config: AppConfig,
    question: str,
) -> RoutingResult:
    """Route a query to the most appropriate retrieval tool(s).

    Uses an LLM to analyze the query and select the optimal retrieval
    strategy based on query characteristics.

    Args:
        config: Application configuration.
        question: User's question to route.

    Returns:
        RoutingResult with selected tools and parameters.
    """
    logger.info("Routing query: '%s'", question[:50])

    # Format tools for prompt
    tools_text = "\n".join(
        f"- {name}: {desc.strip()}" for name, desc in RETRIEVER_TOOLS.items()
    )

    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0,
        api_key=config.openai_api_key,
    )

    system_message = SystemMessage(
        content=ROUTER_SYSTEM_PROMPT.format(tools=tools_text, question=question)
    )

    human_message = HumanMessage(
        content="Analyze the question and return the routing decision as JSON."
    )

    chain = llm | StrOutputParser()
    response = await chain.ainvoke([system_message, human_message])

    # Parse JSON response
    try:
        # Clean up response if needed
        response_clean = response.strip()
        if response_clean.startswith("```"):
            lines = response_clean.split("\n")
            response_clean = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        data = json.loads(response_clean)

        result = RoutingResult(
            selected_tools=data.get("selected_tools", ["graphrag_chat"]),
            reasoning=data.get("reasoning", ""),
            tool_params=data.get("tool_params", {}),
            raw_response=response,
        )
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse routing response: %s", e)
        # Default to chat for complex questions
        result = RoutingResult(
            selected_tools=["graphrag_chat"],
            reasoning="Failed to parse routing decision, defaulting to chat",
            tool_params={"graphrag_chat": {"query": question}},
            raw_response=response,
        )

    logger.info("Routed to: %s", result.selected_tools)
    return result
