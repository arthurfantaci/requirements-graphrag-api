"""Domain-specific evaluation metrics for requirements management content.

Implements custom metrics tailored to the requirements management domain:
- Citation Accuracy: Standards references are correct
- Traceability Coverage: Traceability concepts are included when relevant
- Technical Precision: Domain terms are used correctly
- Completeness Score: All aspects of the query are addressed
- Regulatory Alignment: Standard references are accurate and applicable
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from jama_mcp_server_graphrag.observability import traceable

if TYPE_CHECKING:
    from jama_mcp_server_graphrag.config import AppConfig

logger = logging.getLogger(__name__)


# =============================================================================
# DOMAIN TERMINOLOGY KNOWLEDGE BASE
# =============================================================================

# Standards commonly referenced in requirements management
KNOWN_STANDARDS: frozenset[str] = frozenset(
    [
        "ISO 26262",
        "IEC 62304",
        "DO-178C",
        "DO-178B",
        "ISO 14971",
        "ISO 13485",
        "ISO 9001",
        "ASPICE",
        "CMMI",
        "FDA 21 CFR Part 11",
        "ISO 21434",
        "IEC 61508",
        "AUTOSAR",
        "ISO/IEC 15504",
        "MIL-STD-498",
        "IEEE 830",
        "IEEE 29148",
    ]
)

# Domain-specific technical terms for requirements management
DOMAIN_TERMS: frozenset[str] = frozenset(
    [
        "traceability",
        "requirements",
        "verification",
        "validation",
        "ASIL",
        "FMEA",
        "FTA",
        "V-model",
        "hazard analysis",
        "risk assessment",
        "safety case",
        "impact analysis",
        "change management",
        "baseline",
        "configuration management",
        "test case",
        "functional safety",
        "software safety class",
        "DAL",
        "safety integrity level",
        "bidirectional traceability",
        "requirements elicitation",
        "stakeholder",
        "use case",
        "acceptance criteria",
    ]
)


# =============================================================================
# DATACLASS
# =============================================================================


@dataclass
class DomainMetrics:
    """Container for domain-specific evaluation metrics.

    Attributes:
        citation_accuracy: Score (0-1) measuring correctness of standards citations.
        traceability_coverage: Score (0-1) measuring inclusion of traceability concepts.
        technical_precision: Score (0-1) measuring correct use of domain terms.
        completeness_score: Score (0-1) measuring coverage of all query aspects.
        regulatory_alignment: Score (0-1) measuring accuracy of regulatory references.
    """

    citation_accuracy: float
    traceability_coverage: float
    technical_precision: float
    completeness_score: float
    regulatory_alignment: float

    @property
    def average(self) -> float:
        """Calculate average score across all domain metrics."""
        return (
            self.citation_accuracy
            + self.traceability_coverage
            + self.technical_precision
            + self.completeness_score
            + self.regulatory_alignment
        ) / 5

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary format."""
        return {
            "citation_accuracy": self.citation_accuracy,
            "traceability_coverage": self.traceability_coverage,
            "technical_precision": self.technical_precision,
            "completeness_score": self.completeness_score,
            "regulatory_alignment": self.regulatory_alignment,
            "average": self.average,
        }


# =============================================================================
# EVALUATION PROMPTS
# =============================================================================

EVALUATION_SYSTEM_MESSAGE = "You are an evaluation assistant for requirements management content."

CITATION_ACCURACY_PROMPT = """You are evaluating the citation accuracy of an AI-generated answer
about requirements management.

Citation accuracy measures whether standards and regulations mentioned in the answer are:
1. Actually cited correctly (correct standard names, version numbers if given)
2. Relevant to the context discussed
3. Not fabricated or hallucinated

Expected Standards (if any): {expected_standards}

Answer: {answer}

Rate the citation accuracy on a scale of 0 to 1:
- 1.0: All standards mentioned are correct and properly cited
- 0.5: Some citations are correct, some have minor errors
- 0.0: Citations are incorrect, fabricated, or completely missing when expected

If no standards are mentioned or expected, rate based on whether that's appropriate.

Respond with ONLY a decimal number between 0 and 1."""

TRACEABILITY_COVERAGE_PROMPT = """You are evaluating whether an AI-generated answer \
about requirements management properly covers traceability concepts when relevant.

Traceability coverage measures whether the answer includes appropriate traceability concepts:
- Forward/backward traceability when discussing requirements flow
- Impact analysis when discussing changes
- Trace links when discussing relationships between artifacts
- Coverage matrices when discussing verification

Question: {question}

Answer: {answer}

Expected Entities (traceability concepts that should be covered): {expected_entities}

Rate the traceability coverage on a scale of 0 to 1:
- 1.0: All relevant traceability concepts are properly covered
- 0.5: Some traceability concepts covered, some missing
- 0.0: Missing critical traceability concepts that should have been included

If traceability is not relevant to the question, rate 1.0.

Respond with ONLY a decimal number between 0 and 1."""

TECHNICAL_PRECISION_PROMPT = """You are evaluating the technical precision of domain terminology
in an AI-generated answer about requirements management.

Technical precision measures whether domain-specific terms are:
1. Used correctly in the proper context
2. Defined or explained accurately
3. Not confused with similar terms

Key domain terms to check: ASIL, FMEA, FTA, V-model, DAL, verification vs validation,
functional vs non-functional requirements, traceability matrix, safety case, etc.

Question: {question}

Answer: {answer}

Rate the technical precision on a scale of 0 to 1:
- 1.0: All domain terms are used correctly and precisely
- 0.5: Some terms used correctly, some imprecisely
- 0.0: Domain terms are misused or confused

If no technical terms are used, rate based on whether that's appropriate for the question.

Respond with ONLY a decimal number between 0 and 1."""

COMPLETENESS_SCORE_PROMPT = """You are evaluating whether an AI-generated answer completely
addresses all aspects of a requirements management question.

Completeness measures whether the answer covers all parts of the question:
- For multi-part questions: each part should be addressed
- For comparison questions: both items should be compared
- For procedural questions: all key steps should be included

Question: {question}

Answer: {answer}

Rate the completeness on a scale of 0 to 1:
- 1.0: All aspects of the question are fully addressed
- 0.5: Some aspects addressed, some missing or incomplete
- 0.0: The answer misses major aspects of the question

Respond with ONLY a decimal number between 0 and 1."""

REGULATORY_ALIGNMENT_PROMPT = """You are evaluating the regulatory alignment of an AI-generated
answer about requirements management.

Regulatory alignment measures whether regulatory/standard references are:
1. Accurate (correct standard for the industry/context)
2. Applicable (relevant to the specific situation discussed)
3. Current (not referencing obsolete standards when newer exist)

Industries and their key standards:
- Automotive: ISO 26262, ASPICE, ISO 21434
- Medical: IEC 62304, ISO 14971, FDA 21 CFR Part 11, ISO 13485
- Aerospace: DO-178C, MIL-STD-498

Expected Standards (if specified): {expected_standards}

Question: {question}

Answer: {answer}

Rate the regulatory alignment on a scale of 0 to 1:
- 1.0: All regulatory references are accurate and applicable
- 0.5: Some references are accurate, some questionable
- 0.0: Regulatory references are wrong or misapplied

If no regulatory references are needed, rate 1.0.

Respond with ONLY a decimal number between 0 and 1."""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def _parse_score(response: str) -> float:
    """Parse a score from LLM response.

    Args:
        response: LLM response containing a score.

    Returns:
        Parsed score clamped between 0 and 1.
    """
    try:
        cleaned = response.strip()
        # Extract first number from response (handles "0.85", "-0.5", or "Score: 0.85")
        match = re.search(r"(-?\d+\.?\d*)", cleaned)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
    except ValueError:
        logger.warning("Failed to parse score from response: %s", response[:50])
    return 0.5


def extract_standards_from_text(text: str) -> list[str]:
    """Extract standards references from text.

    Args:
        text: Text to search for standard references.

    Returns:
        List of recognized standards found in the text.
    """
    text_upper = text.upper()
    return [standard for standard in KNOWN_STANDARDS if standard.upper() in text_upper]


def extract_domain_terms_from_text(text: str) -> list[str]:
    """Extract domain-specific terms from text.

    Args:
        text: Text to search for domain terms.

    Returns:
        List of recognized domain terms found in the text.
    """
    text_lower = text.lower()
    return [term for term in DOMAIN_TERMS if term.lower() in text_lower]


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================


@traceable(name="compute_citation_accuracy", run_type="chain")
async def compute_citation_accuracy(
    config: AppConfig,
    answer: str,
    expected_standards: list[str] | None = None,
) -> float:
    """Compute citation accuracy for standards references.

    Measures whether standards and regulations mentioned in the answer
    are correctly cited and relevant.

    Args:
        config: Application configuration.
        answer: The generated answer to evaluate.
        expected_standards: List of standards expected to be cited.

    Returns:
        Citation accuracy score between 0 and 1.
    """
    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0,
        api_key=config.openai_api_key,
    )

    expected_str = ", ".join(expected_standards) if expected_standards else "None specified"

    messages = [
        SystemMessage(content=EVALUATION_SYSTEM_MESSAGE),
        HumanMessage(
            content=CITATION_ACCURACY_PROMPT.format(
                expected_standards=expected_str,
                answer=answer,
            )
        ),
    ]

    chain = llm | StrOutputParser()
    response = await chain.ainvoke(messages)
    return _parse_score(response)


@traceable(name="compute_traceability_coverage", run_type="chain")
async def compute_traceability_coverage(
    config: AppConfig,
    question: str,
    answer: str,
    expected_entities: list[str] | None = None,
) -> float:
    """Compute traceability coverage score.

    Measures whether the answer properly covers traceability concepts
    when they are relevant to the question.

    Args:
        config: Application configuration.
        question: The user's question.
        answer: The generated answer to evaluate.
        expected_entities: List of traceability concepts expected.

    Returns:
        Traceability coverage score between 0 and 1.
    """
    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0,
        api_key=config.openai_api_key,
    )

    expected_str = ", ".join(expected_entities) if expected_entities else "None specified"

    messages = [
        SystemMessage(content=EVALUATION_SYSTEM_MESSAGE),
        HumanMessage(
            content=TRACEABILITY_COVERAGE_PROMPT.format(
                question=question,
                answer=answer,
                expected_entities=expected_str,
            )
        ),
    ]

    chain = llm | StrOutputParser()
    response = await chain.ainvoke(messages)
    return _parse_score(response)


@traceable(name="compute_technical_precision", run_type="chain")
async def compute_technical_precision(
    config: AppConfig,
    question: str,
    answer: str,
) -> float:
    """Compute technical precision score.

    Measures whether domain-specific terms are used correctly
    and precisely in the answer.

    Args:
        config: Application configuration.
        question: The user's question.
        answer: The generated answer to evaluate.

    Returns:
        Technical precision score between 0 and 1.
    """
    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0,
        api_key=config.openai_api_key,
    )

    messages = [
        SystemMessage(content=EVALUATION_SYSTEM_MESSAGE),
        HumanMessage(
            content=TECHNICAL_PRECISION_PROMPT.format(
                question=question,
                answer=answer,
            )
        ),
    ]

    chain = llm | StrOutputParser()
    response = await chain.ainvoke(messages)
    return _parse_score(response)


@traceable(name="compute_completeness_score", run_type="chain")
async def compute_completeness_score(
    config: AppConfig,
    question: str,
    answer: str,
) -> float:
    """Compute completeness score.

    Measures whether the answer addresses all aspects of the question.

    Args:
        config: Application configuration.
        question: The user's question.
        answer: The generated answer to evaluate.

    Returns:
        Completeness score between 0 and 1.
    """
    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0,
        api_key=config.openai_api_key,
    )

    messages = [
        SystemMessage(content=EVALUATION_SYSTEM_MESSAGE),
        HumanMessage(
            content=COMPLETENESS_SCORE_PROMPT.format(
                question=question,
                answer=answer,
            )
        ),
    ]

    chain = llm | StrOutputParser()
    response = await chain.ainvoke(messages)
    return _parse_score(response)


@traceable(name="compute_regulatory_alignment", run_type="chain")
async def compute_regulatory_alignment(
    config: AppConfig,
    question: str,
    answer: str,
    expected_standards: list[str] | None = None,
) -> float:
    """Compute regulatory alignment score.

    Measures whether regulatory and standards references are accurate
    and applicable to the context discussed.

    Args:
        config: Application configuration.
        question: The user's question.
        answer: The generated answer to evaluate.
        expected_standards: List of standards expected to be referenced.

    Returns:
        Regulatory alignment score between 0 and 1.
    """
    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0,
        api_key=config.openai_api_key,
    )

    expected_str = ", ".join(expected_standards) if expected_standards else "None specified"

    messages = [
        SystemMessage(content=EVALUATION_SYSTEM_MESSAGE),
        HumanMessage(
            content=REGULATORY_ALIGNMENT_PROMPT.format(
                expected_standards=expected_str,
                question=question,
                answer=answer,
            )
        ),
    ]

    chain = llm | StrOutputParser()
    response = await chain.ainvoke(messages)
    return _parse_score(response)


@traceable(name="compute_all_domain_metrics", run_type="chain")
async def compute_all_domain_metrics(
    config: AppConfig,
    question: str,
    answer: str,
    expected_standards: list[str] | None = None,
    expected_entities: list[str] | None = None,
) -> DomainMetrics:
    """Compute all domain-specific evaluation metrics.

    Args:
        config: Application configuration.
        question: The user's question.
        answer: The generated answer.
        expected_standards: List of standards expected to be cited.
        expected_entities: List of traceability concepts expected.

    Returns:
        DomainMetrics containing all evaluation scores.
    """
    citation = await compute_citation_accuracy(config, answer, expected_standards)
    traceability = await compute_traceability_coverage(config, question, answer, expected_entities)
    precision = await compute_technical_precision(config, question, answer)
    completeness = await compute_completeness_score(config, question, answer)
    regulatory = await compute_regulatory_alignment(config, question, answer, expected_standards)

    return DomainMetrics(
        citation_accuracy=citation,
        traceability_coverage=traceability,
        technical_precision=precision,
        completeness_score=completeness,
        regulatory_alignment=regulatory,
    )


__all__ = [
    "DOMAIN_TERMS",
    "KNOWN_STANDARDS",
    "DomainMetrics",
    "compute_all_domain_metrics",
    "compute_citation_accuracy",
    "compute_completeness_score",
    "compute_regulatory_alignment",
    "compute_technical_precision",
    "compute_traceability_coverage",
    "extract_domain_terms_from_text",
    "extract_standards_from_text",
]
