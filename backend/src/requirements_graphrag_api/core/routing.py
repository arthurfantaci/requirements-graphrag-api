"""Query intent classification and routing.

Classifies user queries to route them to the appropriate handler:
- EXPLANATORY: Uses the agentic RAG orchestrator (LangGraph subgraphs)
- STRUCTURED: Uses Text2Cypher for direct graph queries

Uses a two-stage approach: fast keyword matching for obvious cases,
then LLM-based classification (INTENT_CLASSIFIER prompt) for ambiguous queries.
"""

from __future__ import annotations

import json
import logging
import re
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from requirements_graphrag_api.observability import traceable_safe
from requirements_graphrag_api.prompts import PromptName, get_prompt_sync

if TYPE_CHECKING:
    from requirements_graphrag_api.config import AppConfig

logger = logging.getLogger(__name__)


class QueryIntent(StrEnum):
    """Types of query intent for routing decisions."""

    EXPLANATORY = "explanatory"
    STRUCTURED = "structured"


# Keywords that strongly suggest structured intent (case-insensitive)
STRUCTURED_KEYWORDS: frozenset[str] = frozenset(
    [
        "list all",
        "show all",
        "show me all",
        "list every",
        "show every",
        "how many",
        "count",
        "total number",
        "table of",
        "enumerate",
        "give me a list",
        "provide a list",
        "provide a table",
    ]
)

# Patterns that suggest structured intent
STRUCTURED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwhich\s+\w+s\b", re.IGNORECASE),  # "which articles", "which tools"
    re.compile(r"\bwhat\s+\w+s\s+are\s+(there|available)\b", re.IGNORECASE),
)


def _quick_classify(question: str) -> QueryIntent | None:
    """Perform fast keyword-based classification.

    This is a performance optimization to avoid LLM calls for obvious cases.

    Args:
        question: User's question.

    Returns:
        QueryIntent if confident, None if LLM classification needed.
    """
    question_lower = question.lower()

    # Check for structured keywords
    for keyword in STRUCTURED_KEYWORDS:
        if keyword in question_lower:
            logger.debug("Quick classify: STRUCTURED (keyword: %s)", keyword)
            return QueryIntent.STRUCTURED

    # Check for structured patterns
    for pattern in STRUCTURED_PATTERNS:
        if pattern.search(question):
            logger.debug("Quick classify: STRUCTURED (pattern match)")
            return QueryIntent.STRUCTURED

    return None


@traceable_safe(name="classify_intent", run_type="llm")
async def classify_intent(
    config: AppConfig,
    question: str,
    *,
    use_quick_classify: bool = True,
) -> QueryIntent:
    """Classify the intent of a user query.

    Uses a two-stage approach:
    1. Quick keyword-based classification for obvious cases
    2. LLM-based classification for ambiguous queries

    Args:
        config: Application configuration.
        question: User's question to classify.
        use_quick_classify: Whether to try quick classification first (default True).

    Returns:
        QueryIntent indicating how to route the query.
    """
    logger.info("Classifying intent for: '%s'", question[:50])

    # Stage 1: Quick keyword-based classification
    if use_quick_classify:
        quick_result = _quick_classify(question)
        if quick_result is not None:
            logger.info("Intent classified (quick): %s", quick_result)
            return quick_result

    # Stage 2: LLM-based classification
    prompt_template = get_prompt_sync(PromptName.INTENT_CLASSIFIER)

    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0,
        api_key=config.openai_api_key,
    )

    chain = prompt_template | llm | StrOutputParser()

    response = await chain.ainvoke({"question": question})

    # Parse JSON response
    try:
        # Handle potential markdown code blocks
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(line for line in lines if not line.startswith("```"))
            response = response.strip()

        result = json.loads(response)
        intent_str = result.get("intent", "explanatory").lower()
        intent = QueryIntent.STRUCTURED if intent_str == "structured" else QueryIntent.EXPLANATORY

        logger.info("Intent classified (LLM): %s", intent)
        return intent

    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        logger.warning("Failed to parse intent response, defaulting to explanatory: %s", e)
        return QueryIntent.EXPLANATORY


def get_routing_guide() -> dict[str, Any]:
    """Get user-facing documentation for query routing.

    Returns a structured guide explaining how to phrase queries
    for optimal routing.

    Returns:
        Dictionary with routing guidance for display to users.
    """
    return {
        "title": "How to Ask Questions",
        "description": (
            "Your questions are automatically routed to the best handler. "
            "Here's how to get the most relevant answers."
        ),
        "query_types": [
            {
                "type": "Understanding & Explanation",
                "intent": "explanatory",
                "description": "Get synthesized answers with context and citations",
                "examples": [
                    "What is requirements traceability?",
                    "How do I implement change management?",
                    "What are best practices for verification?",
                    "Why is traceability important for compliance?",
                ],
                "keywords": ["what is", "how do I", "explain", "best practices", "why"],
            },
            {
                "type": "Lists & Structured Data",
                "intent": "structured",
                "description": "Get complete lists, counts, and structured query results",
                "examples": [
                    "List all webinars",
                    "Show me all videos",
                    "How many articles mention ISO 26262?",
                    "Which standards apply to automotive?",
                ],
                "keywords": ["list all", "show me all", "how many", "which", "table of"],
            },
        ],
        "tips": [
            "For complete lists of resources, use 'list all' or 'show me all'",
            "For explanations and guidance, phrase as 'what is' or 'how do I'",
            "Ambiguous queries default to explanation mode for richer context",
        ],
    }


__all__ = [
    "QueryIntent",
    "classify_intent",
    "get_routing_guide",
]
