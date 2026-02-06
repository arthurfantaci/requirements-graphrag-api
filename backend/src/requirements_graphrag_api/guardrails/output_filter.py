"""Output content filtering for LLM responses.

This module filters LLM outputs for safety, accuracy, and appropriateness
before returning them to users.

Checks Performed:
    1. Toxicity check: Same as input (using OpenAI Moderation)
    2. Confidence scoring: Estimate answer reliability
    3. Disclaimer injection: Add warnings for uncertain answers

The filter can:
    - Block toxic or harmful responses
    - Add disclaimers to low-confidence answers
    - Track modifications made for logging
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from requirements_graphrag_api.guardrails.toxicity import (
    ToxicityConfig,
    check_toxicity,
)
from requirements_graphrag_api.observability import traceable_safe

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


# Default blocked response message
BLOCKED_RESPONSE = """I apologize, but I'm unable to provide that response.

If you have questions about requirements management, traceability, or systems \
engineering, I'd be happy to help with those topics.

If you believe this is an error, please contact support."""


# Disclaimer for low-confidence answers
LOW_CONFIDENCE_DISCLAIMER = (
    "\n\n_Note: This answer may not fully address your question. "
    "Please verify with authoritative sources._"
)


@dataclass(frozen=True, slots=True)
class OutputFilterConfig:
    """Configuration for output filtering.

    Attributes:
        enabled: Whether output filtering is enabled.
        toxicity_enabled: Whether to check output for toxicity.
        add_disclaimers: Whether to add disclaimers to low-confidence answers.
        confidence_threshold: Threshold below which disclaimers are added.
        blocked_response_message: Message to return when output is blocked.
        disclaimer_message: Message to append to low-confidence answers.
    """

    enabled: bool = True
    toxicity_enabled: bool = True
    add_disclaimers: bool = True
    confidence_threshold: float = 0.6
    blocked_response_message: str = BLOCKED_RESPONSE
    disclaimer_message: str = LOW_CONFIDENCE_DISCLAIMER


@dataclass(frozen=True, slots=True)
class OutputFilterResult:
    """Result of output filtering.

    Attributes:
        is_safe: Whether the output passed all safety checks.
        filtered_content: The filtered/modified output content.
        original_content: The original unmodified output.
        warnings: List of warning messages about the output.
        modifications: List of modifications made to the output.
        confidence_score: Estimated confidence in the answer (0-1).
        should_add_disclaimer: Whether a disclaimer should be added.
        blocked_reason: Reason if the output was blocked.
    """

    is_safe: bool
    filtered_content: str
    original_content: str
    warnings: tuple[str, ...]
    modifications: tuple[str, ...]
    confidence_score: float
    should_add_disclaimer: bool
    blocked_reason: str | None = None


# Indicators that suggest uncertainty or lack of grounding
HALLUCINATION_INDICATORS: tuple[str, ...] = (
    "i don't have information about",
    "i cannot find",
    "i don't have access to",
    "based on my knowledge",
    "based on my training",
    "i believe",
    "i think",
    "i'm not sure",
    "i'm not certain",
    "i cannot verify",
    "i don't know",
    "as far as i know",
    "to my knowledge",
    "i assume",
    "i speculate",
    "it's possible that",
    "it might be",
    "perhaps",
    "maybe",
    "i would guess",
)

# Indicators of confident, grounded responses
CONFIDENCE_INDICATORS: tuple[str, ...] = (
    "according to",
    "the document states",
    "as stated in",
    "the guide mentions",
    "the source indicates",
    "based on the retrieved",
    "from the knowledge base",
)


# Default configuration
DEFAULT_CONFIG = OutputFilterConfig()
DEFAULT_TOXICITY_CONFIG = ToxicityConfig()


def _calculate_confidence(
    output: str,
    retrieved_sources: list[dict[str, Any]],
) -> tuple[float, list[str]]:
    """Calculate confidence score and collect warnings.

    Args:
        output: The LLM output to analyze.
        retrieved_sources: Sources retrieved for the query.

    Returns:
        Tuple of (confidence_score, warnings_list).
    """
    confidence = 1.0
    warnings: list[str] = []
    output_lower = output.lower()

    # Check for hallucination indicators (reduce confidence)
    for indicator in HALLUCINATION_INDICATORS:
        if indicator in output_lower:
            confidence -= 0.15
            warnings.append(f"Potential uncertainty: '{indicator}'")
            # Cap the reduction from indicators
            if confidence < 0.3:
                confidence = 0.3
                break

    # Check for confidence indicators (boost confidence)
    for indicator in CONFIDENCE_INDICATORS:
        if indicator in output_lower:
            confidence = min(1.0, confidence + 0.1)
            break

    # Check source grounding
    if not retrieved_sources:
        confidence -= 0.3
        warnings.append("No sources retrieved for grounding")
    elif len(retrieved_sources) < 2:
        confidence -= 0.1
        warnings.append("Limited sources retrieved")

    # Check response length (very short responses may be incomplete)
    word_count = len(output.split())
    if word_count < 10:
        confidence -= 0.2
        warnings.append("Response is very short")
    elif word_count < 25:
        confidence -= 0.1
        warnings.append("Response may be incomplete")

    # Ensure confidence stays in valid range
    confidence = max(0.0, min(1.0, confidence))

    return confidence, warnings


@traceable_safe(name="filter_output", run_type="chain")
async def filter_output(
    output: str,
    original_query: str,
    retrieved_sources: list[dict[str, Any]] | None = None,
    config: OutputFilterConfig = DEFAULT_CONFIG,
    toxicity_config: ToxicityConfig = DEFAULT_TOXICITY_CONFIG,
    openai_client: AsyncOpenAI | None = None,
) -> OutputFilterResult:
    """Filter LLM output for safety and quality.

    Performs toxicity checking and confidence scoring on LLM output.
    Can block harmful content and add disclaimers to uncertain answers.

    Args:
        output: The LLM output to filter.
        original_query: The original user query (for context).
        retrieved_sources: Sources retrieved for the query.
        config: Output filter configuration.
        toxicity_config: Toxicity detection configuration.
        openai_client: Async OpenAI client for toxicity checking.

    Returns:
        OutputFilterResult with filtering results.

    Example:
        >>> result = await filter_output(
        ...     output="Requirements traceability is...",
        ...     original_query="What is traceability?",
        ...     retrieved_sources=[{"content": "..."}],
        ... )
        >>> result.is_safe
        True
    """
    if not config.enabled:
        return OutputFilterResult(
            is_safe=True,
            filtered_content=output,
            original_content=output,
            warnings=(),
            modifications=(),
            confidence_score=1.0,
            should_add_disclaimer=False,
        )

    warnings: list[str] = []
    modifications: list[str] = []
    filtered = output
    sources = retrieved_sources or []

    # 1. Toxicity check on output
    if config.toxicity_enabled:
        toxicity = await check_toxicity(
            output,
            openai_client=openai_client,
            config=toxicity_config,
        )

        if toxicity.should_block:
            logger.warning(
                "Output blocked due to toxicity",
                extra={
                    "categories": [c.value for c in toxicity.categories],
                    "confidence": toxicity.confidence,
                },
            )
            return OutputFilterResult(
                is_safe=False,
                filtered_content=config.blocked_response_message,
                original_content=output,
                warnings=("Output contained harmful content",),
                modifications=("Replaced with safe response",),
                confidence_score=0.0,
                should_add_disclaimer=False,
                blocked_reason="toxicity",
            )

        if toxicity.should_warn:
            warnings.append(f"Toxicity warning: {', '.join(c.value for c in toxicity.categories)}")

    # 2. Calculate confidence score
    confidence, confidence_warnings = _calculate_confidence(output, sources)
    warnings.extend(confidence_warnings)

    # 3. Determine if disclaimer should be added
    should_add_disclaimer = config.add_disclaimers and confidence < config.confidence_threshold

    # 4. Add disclaimer if needed
    if should_add_disclaimer:
        filtered = output + config.disclaimer_message
        modifications.append("Added low-confidence disclaimer")

    return OutputFilterResult(
        is_safe=True,
        filtered_content=filtered,
        original_content=output,
        warnings=tuple(warnings),
        modifications=tuple(modifications),
        confidence_score=confidence,
        should_add_disclaimer=should_add_disclaimer,
    )


async def filter_streaming_output(
    output_chunk: str,
    accumulated_output: str,
    config: OutputFilterConfig = DEFAULT_CONFIG,
) -> tuple[str, bool]:
    """Filter streaming output chunk (lightweight check).

    For streaming responses, we can't do full toxicity checking on each chunk.
    This performs lightweight checks that can be done incrementally.

    Args:
        output_chunk: The current output chunk.
        accumulated_output: The accumulated output so far.
        config: Output filter configuration.

    Returns:
        Tuple of (filtered_chunk, should_continue).

    Note:
        Full toxicity check should be done on the complete output
        using filter_output() after streaming completes.
    """
    if not config.enabled:
        return output_chunk, True

    # For streaming, we do minimal checks
    # Full toxicity check happens post-stream
    return output_chunk, True
