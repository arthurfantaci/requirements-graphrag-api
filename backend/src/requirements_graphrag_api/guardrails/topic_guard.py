"""Topic boundary enforcement for query relevance.

This module ensures queries and responses stay within the intended domain
(requirements management). It uses two methods:
1. Fast keyword matching for obvious out-of-scope queries
2. LLM classification for ambiguous cases

In-Scope Topics:
    - Requirements management and traceability
    - Systems engineering and product development
    - Compliance standards (ISO, IEC, FDA, etc.)
    - Jama Software products and features
    - Related technical documentation practices

Out-of-Scope Topics:
    - Politics, religion, personal relationships
    - Medical diagnosis/treatment, legal advice
    - Financial investment advice
    - Entertainment, sports, current events
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from requirements_graphrag_api.observability import traceable_safe

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class TopicClassification(StrEnum):
    """Classification result for topic relevance."""

    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"
    BORDERLINE = "borderline"


@dataclass(frozen=True, slots=True)
class TopicGuardConfig:
    """Configuration for topic boundary enforcement.

    Attributes:
        enabled: Whether topic guard is enabled.
        use_llm_classification: Whether to use LLM for ambiguous cases.
        allow_borderline: Whether to allow borderline queries.
        out_of_scope_response: Default response for out-of-scope queries.
    """

    enabled: bool = True
    use_llm_classification: bool = True
    allow_borderline: bool = True
    out_of_scope_response: str = """I'm a specialized assistant for Requirements Management topics.

I can help you with:
- Requirements traceability and management
- Systems engineering best practices
- Compliance with standards (ISO, IEC, FDA)
- Jama Software features and workflows

Would you like to ask about any of these topics instead?"""


@dataclass(frozen=True, slots=True)
class TopicCheckResult:
    """Result of topic relevance check.

    Attributes:
        classification: The topic classification result.
        confidence: Confidence score (0-1) for the classification.
        suggested_response: For out-of-scope, a polite redirect message.
        reasoning: Explanation for the classification.
        check_type: Type of check performed ("keyword" or "llm").
    """

    classification: TopicClassification
    confidence: float
    suggested_response: str | None
    reasoning: str | None
    check_type: str


# Topics that are clearly in scope for requirements management
IN_SCOPE_TOPICS: tuple[str, ...] = (
    "requirements management",
    "requirements traceability",
    "traceability matrix",
    "systems engineering",
    "product development",
    "software requirements",
    "hardware requirements",
    "compliance",
    "regulatory",
    "iso ",
    "iec ",
    "fda ",
    "verification and validation",
    "v&v",
    "change management",
    "configuration management",
    "risk management",
    "jama software",
    "jama connect",
    "requirements documentation",
    "specification writing",
    "test case",
    "test management",
    "srs ",
    "software requirement specification",
    "system requirement specification",
    "functional requirement",
    "non-functional requirement",
    "nfr",
    "use case",
    "user story",
    "acceptance criteria",
    "requirement elicitation",
    "stakeholder",
    "trace link",
    "impact analysis",
    "baseline",
    "requirement attribute",
    "requirement type",
    "derived requirement",
    "parent requirement",
    "child requirement",
)

# Topics that are clearly out of scope
OUT_OF_SCOPE_TOPICS: tuple[str, ...] = (
    "politics",
    "political",
    "election",
    "vote",
    "republican",
    "democrat",
    "religion",
    "religious",
    "church",
    "prayer",
    "personal relationship",
    "dating",
    "boyfriend",
    "girlfriend",
    "medical diagnosis",
    "medical treatment",
    "medication",
    "prescription",
    "legal advice",
    "lawsuit",
    "attorney",
    "lawyer",
    "financial advice",
    "stock",
    "investment",
    "cryptocurrency",
    "crypto",
    "bitcoin",
    "competitor product",
    "current events",
    "news",
    "entertainment",
    "movie",
    "music",
    "celebrity",
    "pop culture",
    "cooking",
    "recipe",
    "sports",
    "football",
    "basketball",
    "soccer",
    "weather",
    "horoscope",
    "joke",
)

# Meta-conversation patterns — word-boundary regex to identify queries about the
# conversation itself. These get IN_SCOPE classification with high confidence so
# they bypass the topic guard (the conversational handler answers from history).
META_CONVERSATION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
    for kw in (
        "what was my",
        "summarize our conversation",
        "my first question",
        "you said earlier",
        "what did I ask",
        "our conversation",
        "my previous question",
        "repeat what you said",
        "what did you say",
        "what have we discussed",
        "recap our discussion",
        "my last question",
        "what we talked about",
        "earlier you mentioned",
        "you told me",
        "go back to what",
        "remind me what",
        "what did you tell me",
    )
)


# LLM prompt for topic classification
TOPIC_CLASSIFIER_PROMPT = """You are a topic classifier for a Requirements Management \
knowledge base chatbot.

The chatbot should ONLY answer questions about:
- Requirements management and traceability
- Systems engineering and product development
- Compliance standards (ISO, IEC, FDA, etc.)
- Jama Software products and features
- Related technical documentation practices

Classify the following user query as one of:
- IN_SCOPE: Related to the topics above
- OUT_OF_SCOPE: Unrelated to requirements management
- BORDERLINE: Could be related depending on context

User Query: {query}

Classification (respond with only IN_SCOPE, OUT_OF_SCOPE, or BORDERLINE):"""


# Default configuration
DEFAULT_CONFIG = TopicGuardConfig()


def _get_redirect_response(detected_topic: str | None = None) -> str:
    """Get a polite redirect response for out-of-scope queries.

    Args:
        detected_topic: The detected out-of-scope topic (optional).

    Returns:
        A polite message redirecting to in-scope topics.
    """
    base_response = """I'm a specialized assistant for Requirements Management topics.

I can help you with:
- Requirements traceability and management
- Systems engineering best practices
- Compliance with standards (ISO, IEC, FDA)
- Jama Software features and workflows

Would you like to ask about any of these topics instead?"""

    return base_response


def _check_keywords(text: str) -> tuple[TopicClassification, str | None, float]:
    """Fast keyword-based topic check.

    Args:
        text: The text to check.

    Returns:
        Tuple of (classification, detected_topic, confidence).
    """
    # Check meta-conversation patterns first — these are always in-scope
    for pattern in META_CONVERSATION_PATTERNS:
        if pattern.search(text):
            return (TopicClassification.IN_SCOPE, "meta-conversation", 0.95)

    text_lower = text.lower()

    # Check for out-of-scope topics
    for topic in OUT_OF_SCOPE_TOPICS:
        if topic.lower() in text_lower:
            return (TopicClassification.OUT_OF_SCOPE, topic, 0.9)

    # Check for in-scope topics
    for topic in IN_SCOPE_TOPICS:
        if topic.lower() in text_lower:
            return (TopicClassification.IN_SCOPE, topic, 0.9)

    # No keyword matches - uncertain
    return (TopicClassification.BORDERLINE, None, 0.5)


def _parse_classification(response_text: str) -> TopicClassification:
    """Parse LLM classification response.

    Args:
        response_text: The raw LLM response.

    Returns:
        Parsed TopicClassification.
    """
    text = response_text.strip().upper()

    if "OUT_OF_SCOPE" in text or "OUT OF SCOPE" in text:
        return TopicClassification.OUT_OF_SCOPE
    elif "IN_SCOPE" in text or "IN SCOPE" in text:
        return TopicClassification.IN_SCOPE
    else:
        return TopicClassification.BORDERLINE


async def check_topic_relevance_fast(
    query: str,
    config: TopicGuardConfig = DEFAULT_CONFIG,
) -> TopicCheckResult:
    """Fast keyword-based topic relevance check.

    Args:
        query: The user query to check.
        config: Topic guard configuration.

    Returns:
        TopicCheckResult with keyword-based classification.

    Example:
        >>> result = await check_topic_relevance_fast("What is requirements traceability?")
        >>> result.classification
        <TopicClassification.IN_SCOPE: 'in_scope'>
    """
    if not query or not query.strip():
        return TopicCheckResult(
            classification=TopicClassification.IN_SCOPE,
            confidence=0.5,
            suggested_response=None,
            reasoning="Empty query",
            check_type="keyword",
        )

    classification, detected_topic, confidence = _check_keywords(query)

    suggested_response = None
    if classification == TopicClassification.OUT_OF_SCOPE:
        suggested_response = _get_redirect_response(detected_topic)

    reasoning = None
    if detected_topic:
        if classification == TopicClassification.OUT_OF_SCOPE:
            reasoning = f"Query contains out-of-scope topic: {detected_topic}"
        else:
            reasoning = f"Query contains in-scope topic: {detected_topic}"
    else:
        reasoning = "No keyword matches found"

    return TopicCheckResult(
        classification=classification,
        confidence=confidence,
        suggested_response=suggested_response,
        reasoning=reasoning,
        check_type="keyword",
    )


@traceable_safe(name="check_topic_relevance", run_type="llm")
async def check_topic_relevance(
    query: str,
    llm: ChatOpenAI | None = None,
    config: TopicGuardConfig = DEFAULT_CONFIG,
) -> TopicCheckResult:
    """Check if query is within the chatbot's intended scope.

    Performs keyword-based check first, then uses LLM classification
    for ambiguous cases if configured.

    Args:
        query: The user query to check.
        llm: LangChain ChatOpenAI instance for LLM classification.
        config: Topic guard configuration.

    Returns:
        TopicCheckResult with classification and suggested response.

    Example:
        >>> result = await check_topic_relevance("What is requirements traceability?")
        >>> result.classification
        <TopicClassification.IN_SCOPE: 'in_scope'>

        >>> result = await check_topic_relevance("What's your favorite movie?")
        >>> result.classification
        <TopicClassification.OUT_OF_SCOPE: 'out_of_scope'>
    """
    if not config.enabled:
        return TopicCheckResult(
            classification=TopicClassification.IN_SCOPE,
            confidence=1.0,
            suggested_response=None,
            reasoning="Topic guard disabled",
            check_type="disabled",
        )

    # Fast keyword check first
    fast_result = await check_topic_relevance_fast(query, config)

    # If keyword check gives high confidence result, use it
    if fast_result.confidence >= 0.9:
        return fast_result

    # If keyword check is uncertain and LLM is available, use LLM classification
    if config.use_llm_classification and llm is not None:
        try:
            response = await llm.ainvoke(TOPIC_CLASSIFIER_PROMPT.format(query=query))
            classification = _parse_classification(str(response.content))

            suggested_response = None
            is_out_of_scope = classification == TopicClassification.OUT_OF_SCOPE
            is_blocked_borderline = (
                classification == TopicClassification.BORDERLINE and not config.allow_borderline
            )
            if is_out_of_scope or is_blocked_borderline:
                suggested_response = _get_redirect_response()

            return TopicCheckResult(
                classification=classification,
                confidence=0.85,
                suggested_response=suggested_response,
                reasoning=f"LLM classified as: {classification.value}",
                check_type="llm",
            )

        except Exception:
            logger.exception("Error during LLM topic classification")
            # Fall back to keyword result on error
            return fast_result

    # Return keyword result if no LLM check
    return fast_result
