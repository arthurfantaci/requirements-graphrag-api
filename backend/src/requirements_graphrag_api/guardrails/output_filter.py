"""Output content filtering for LLM responses.

Filters LLM outputs for safety and adds disclaimers for low-confidence
answers. Confidence scoring simplified to source count and response
length only (text-based heuristics removed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from requirements_graphrag_api.guardrails.toxicity import (
    ToxicityConfig,
    check_toxicity,
)
from requirements_graphrag_api.observability import traceable_safe

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
    """Configuration for output filtering."""

    enabled: bool = True
    toxicity_enabled: bool = True
    add_disclaimers: bool = True
    confidence_threshold: float = 0.6
    blocked_response_message: str = BLOCKED_RESPONSE
    disclaimer_message: str = LOW_CONFIDENCE_DISCLAIMER


@dataclass(frozen=True, slots=True)
class OutputFilterResult:
    """Result of output filtering."""

    is_safe: bool
    filtered_content: str
    original_content: str
    warnings: tuple[str, ...]
    modifications: tuple[str, ...]
    confidence_score: float
    should_add_disclaimer: bool
    blocked_reason: str | None = None


# Default configuration
DEFAULT_CONFIG = OutputFilterConfig()
DEFAULT_TOXICITY_CONFIG = ToxicityConfig()


def _calculate_confidence(
    output: str,
    retrieved_sources: list[dict[str, Any]],
) -> tuple[float, list[str]]:
    """Calculate confidence score from source count and response length."""
    confidence = 1.0
    warnings: list[str] = []

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

    confidence = max(0.0, min(1.0, confidence))
    return confidence, warnings


@traceable_safe(name="filter_output", run_type="chain")
async def filter_output(
    output: str,
    original_query: str,
    retrieved_sources: list[dict[str, Any]] | None = None,
    config: OutputFilterConfig = DEFAULT_CONFIG,
    toxicity_config: ToxicityConfig = DEFAULT_TOXICITY_CONFIG,
    **_kwargs: object,
) -> OutputFilterResult:
    """Filter LLM output for safety and quality.

    The ``openai_client`` parameter previously accepted is silently
    ignored via ``**_kwargs`` for backwards compatibility.
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
        toxicity = await check_toxicity(output, config=toxicity_config)

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
