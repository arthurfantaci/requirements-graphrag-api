"""Hallucination detection for RAG responses.

This module provides LLM-based grounding verification to detect when
responses contain claims not supported by the retrieved sources.

Grounding Levels:
    - FULLY_GROUNDED: All claims supported by sources
    - MOSTLY_GROUNDED: Most claims supported, minor additions
    - PARTIALLY_GROUNDED: Some claims unsupported
    - UNGROUNDED: Most claims not in sources

Usage:
    from requirements_graphrag_api.guardrails.hallucination import (
        check_hallucination,
        GroundingLevel,
    )

    result = await check_hallucination(
        response="The response text...",
        sources=[{"title": "...", "content": "..."}],
        llm=chat_llm,
    )

    if result.should_add_warning:
        response += HALLUCINATION_WARNING
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class GroundingLevel(StrEnum):
    """Level of factual grounding in sources."""

    FULLY_GROUNDED = "fully_grounded"
    MOSTLY_GROUNDED = "mostly_grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    UNGROUNDED = "ungrounded"


@dataclass(frozen=True, slots=True)
class HallucinationCheckResult:
    """Result of hallucination check.

    Attributes:
        grounding_level: How well the response is grounded in sources.
        confidence: Confidence score (0-1) in the assessment.
        unsupported_claims: List of claims not supported by sources.
        reasoning: Explanation of the analysis.
        should_add_warning: Whether to add a warning to the response.
        checked: Whether the check was actually performed.
    """

    grounding_level: GroundingLevel
    confidence: float
    unsupported_claims: tuple[str, ...]
    reasoning: str
    should_add_warning: bool
    checked: bool = True


# Confidence mapping by grounding level
_CONFIDENCE_MAP: dict[GroundingLevel, float] = {
    GroundingLevel.FULLY_GROUNDED: 0.95,
    GroundingLevel.MOSTLY_GROUNDED: 0.8,
    GroundingLevel.PARTIALLY_GROUNDED: 0.5,
    GroundingLevel.UNGROUNDED: 0.2,
}


HALLUCINATION_CHECK_PROMPT = """You are a fact-checker for a Requirements Management knowledge base.

Given the following retrieved sources and the assistant's response, analyze whether
the response is factually grounded in the sources.

## Retrieved Sources:
{sources}

## Assistant's Response:
{response}

## Instructions:
1. Identify specific factual claims made in the response
2. Check if each claim is directly supported by the sources
3. Note any claims that are NOT supported by the sources
4. Consider that general knowledge or obvious inferences may be acceptable

Respond ONLY with a JSON object in this exact format:
{{
    "grounding_level": "fully_grounded" | "mostly_grounded" | "partially_grounded" | "ungrounded",
    "unsupported_claims": ["claim1", "claim2"],
    "reasoning": "Brief explanation of your analysis"
}}

Rules for grounding_level:
- "fully_grounded": All factual claims are directly supported by sources
- "mostly_grounded": Most claims supported; minor additions are reasonable inferences
- "partially_grounded": Some significant claims are not in sources
- "ungrounded": Most claims are not supported by sources"""


async def check_hallucination(
    response: str,
    sources: list[dict[str, Any]],
    llm: ChatOpenAI,
    max_sources: int = 5,
) -> HallucinationCheckResult:
    """Check if response is grounded in sources.

    Uses LLM to analyze factual grounding of the response against
    the provided sources.

    Args:
        response: The assistant's response to check.
        sources: List of source documents with 'title' and 'content' keys.
        llm: The LLM to use for analysis.
        max_sources: Maximum number of sources to include (to fit context).

    Returns:
        HallucinationCheckResult with grounding assessment.

    Example:
        >>> result = await check_hallucination(
        ...     response="Requirements traceability links requirements...",
        ...     sources=[{"title": "Chapter 1", "content": "..."}],
        ...     llm=llm,
        ... )
        >>> result.grounding_level
        <GroundingLevel.FULLY_GROUNDED: 'fully_grounded'>
    """
    # Handle edge cases
    if not response or not response.strip():
        return HallucinationCheckResult(
            grounding_level=GroundingLevel.FULLY_GROUNDED,
            confidence=1.0,
            unsupported_claims=(),
            reasoning="Empty response has no claims to verify",
            should_add_warning=False,
            checked=False,
        )

    if not sources:
        return HallucinationCheckResult(
            grounding_level=GroundingLevel.UNGROUNDED,
            confidence=0.5,
            unsupported_claims=(),
            reasoning="No sources provided for verification",
            should_add_warning=True,
            checked=False,
        )

    # Format sources for prompt (limit to max_sources)
    limited_sources = sources[:max_sources]
    sources_text = _format_sources(limited_sources)

    # Build prompt
    prompt = HALLUCINATION_CHECK_PROMPT.format(
        sources=sources_text,
        response=response,
    )

    try:
        # Get LLM analysis
        result = await llm.ainvoke(prompt)
        content = result.content

        # Parse JSON response
        analysis = _parse_llm_response(content)

        grounding = GroundingLevel(analysis.get("grounding_level", "partially_grounded"))
        unsupported = analysis.get("unsupported_claims", [])
        reasoning = analysis.get("reasoning", "")

        confidence = _CONFIDENCE_MAP.get(grounding, 0.5)

        # Determine if warning should be added
        should_warn = grounding in (
            GroundingLevel.PARTIALLY_GROUNDED,
            GroundingLevel.UNGROUNDED,
        )

        return HallucinationCheckResult(
            grounding_level=grounding,
            confidence=confidence,
            unsupported_claims=tuple(unsupported),
            reasoning=reasoning,
            should_add_warning=should_warn,
        )

    except Exception as e:
        logger.warning("Hallucination check failed: %s", e)
        # Return conservative result on failure
        return HallucinationCheckResult(
            grounding_level=GroundingLevel.PARTIALLY_GROUNDED,
            confidence=0.5,
            unsupported_claims=(),
            reasoning=f"Unable to verify grounding: {e!s}",
            should_add_warning=True,
        )


def _format_sources(sources: list[dict[str, Any]]) -> str:
    """Format sources for the prompt.

    Args:
        sources: List of source documents.

    Returns:
        Formatted sources text.
    """
    formatted = []
    for i, source in enumerate(sources, 1):
        title = source.get("title", source.get("name", f"Source {i}"))
        content = source.get("content", source.get("text", ""))

        # Truncate long content
        if len(content) > 2000:
            content = content[:2000] + "..."

        formatted.append(f"### Source {i}: {title}\n{content}")

    return "\n\n".join(formatted)


def _parse_llm_response(content: str) -> dict[str, Any]:
    """Parse LLM response to extract JSON.

    Handles cases where LLM might include extra text around JSON.

    Args:
        content: Raw LLM response content.

    Returns:
        Parsed JSON as dictionary.
    """
    # Try direct parsing first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in the content
    import re

    json_match = re.search(r"\{[\s\S]*\}", content)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Return conservative defaults
    return {
        "grounding_level": "partially_grounded",
        "unsupported_claims": [],
        "reasoning": "Unable to parse LLM response",
    }


def check_hallucination_sync(
    response: str,
    sources: list[dict[str, Any]],
    llm: ChatOpenAI,
    max_sources: int = 5,
) -> HallucinationCheckResult:
    """Synchronous version of hallucination check.

    For use in non-async contexts. Uses the LLM's sync invoke method.

    Args:
        response: The assistant's response to check.
        sources: List of source documents.
        llm: The LLM to use for analysis.
        max_sources: Maximum number of sources to include.

    Returns:
        HallucinationCheckResult with grounding assessment.
    """
    # Handle edge cases
    if not response or not response.strip():
        return HallucinationCheckResult(
            grounding_level=GroundingLevel.FULLY_GROUNDED,
            confidence=1.0,
            unsupported_claims=(),
            reasoning="Empty response has no claims to verify",
            should_add_warning=False,
            checked=False,
        )

    if not sources:
        return HallucinationCheckResult(
            grounding_level=GroundingLevel.UNGROUNDED,
            confidence=0.5,
            unsupported_claims=(),
            reasoning="No sources provided for verification",
            should_add_warning=True,
            checked=False,
        )

    # Format sources for prompt
    limited_sources = sources[:max_sources]
    sources_text = _format_sources(limited_sources)

    prompt = HALLUCINATION_CHECK_PROMPT.format(
        sources=sources_text,
        response=response,
    )

    try:
        result = llm.invoke(prompt)
        content = result.content
        analysis = _parse_llm_response(content)

        grounding = GroundingLevel(analysis.get("grounding_level", "partially_grounded"))
        unsupported = analysis.get("unsupported_claims", [])
        reasoning = analysis.get("reasoning", "")

        confidence = _CONFIDENCE_MAP.get(grounding, 0.5)
        should_warn = grounding in (
            GroundingLevel.PARTIALLY_GROUNDED,
            GroundingLevel.UNGROUNDED,
        )

        return HallucinationCheckResult(
            grounding_level=grounding,
            confidence=confidence,
            unsupported_claims=tuple(unsupported),
            reasoning=reasoning,
            should_add_warning=should_warn,
        )

    except Exception as e:
        logger.warning("Hallucination check failed: %s", e)
        return HallucinationCheckResult(
            grounding_level=GroundingLevel.PARTIALLY_GROUNDED,
            confidence=0.5,
            unsupported_claims=(),
            reasoning=f"Unable to verify grounding: {e!s}",
            should_add_warning=True,
        )


# Standard warning message to append when hallucination is detected
HALLUCINATION_WARNING = (
    "\n\n---\n"
    "_⚠️ Note: This response may contain information not fully supported "
    "by the knowledge base. Please verify important details with authoritative sources._"
)

# Alternative shorter warning
HALLUCINATION_WARNING_SHORT = "\n\n_⚠️ Some claims may not be directly supported by sources._"
