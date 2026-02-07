"""Topic boundary enforcement for query relevance.

Ensures queries stay within the intended domain (requirements management)
using fast keyword matching. LLM classification removed -- keyword matching
is sufficient for a single-user authenticated API.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from requirements_graphrag_api.core.routing import CONVERSATIONAL_PATTERNS
from requirements_graphrag_api.observability import traceable_safe

# Re-export so existing imports from topic_guard still work
META_CONVERSATION_PATTERNS = CONVERSATIONAL_PATTERNS


class TopicClassification(StrEnum):
    """Classification result for topic relevance."""

    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"
    BORDERLINE = "borderline"


@dataclass(frozen=True, slots=True)
class TopicGuardConfig:
    """Configuration for topic boundary enforcement."""

    enabled: bool = True
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
    """Result of topic relevance check."""

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

# Default configuration
DEFAULT_CONFIG = TopicGuardConfig()

_REDIRECT_RESPONSE = """I'm a specialized assistant for Requirements Management topics.

I can help you with:
- Requirements traceability and management
- Systems engineering best practices
- Compliance with standards (ISO, IEC, FDA)
- Jama Software features and workflows

Would you like to ask about any of these topics instead?"""


def _check_keywords(text: str) -> tuple[TopicClassification, str | None, float]:
    """Fast keyword-based topic check."""
    # Check meta-conversation patterns first -- these are always in-scope
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

    # No keyword matches -- uncertain
    return (TopicClassification.BORDERLINE, None, 0.5)


async def check_topic_relevance_fast(
    query: str,
    config: TopicGuardConfig = DEFAULT_CONFIG,
) -> TopicCheckResult:
    """Fast keyword-based topic relevance check."""
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
        suggested_response = _REDIRECT_RESPONSE

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


@traceable_safe(name="check_topic_relevance", run_type="chain")
async def check_topic_relevance(
    query: str,
    config: TopicGuardConfig = DEFAULT_CONFIG,
    **_kwargs: object,
) -> TopicCheckResult:
    """Check if query is within the chatbot's intended scope.

    Keyword-only classification. The ``llm`` parameter previously accepted
    is silently ignored via ``**_kwargs`` for backwards compatibility.
    """
    if not config.enabled:
        return TopicCheckResult(
            classification=TopicClassification.IN_SCOPE,
            confidence=1.0,
            suggested_response=None,
            reasoning="Topic guard disabled",
            check_type="disabled",
        )

    return await check_topic_relevance_fast(query, config)
