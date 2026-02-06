"""Toxicity detection for user inputs and LLM outputs.

This module provides dual-layer toxicity detection:
1. Fast layer: Word-list based profanity check using better-profanity (~1ms)
2. Accurate layer: OpenAI Moderation API (~500ms) for comprehensive detection

Toxicity Categories (aligned with OpenAI Moderation):
    - HATE: Hateful content targeting protected characteristics
    - HATE_THREATENING: Hateful content with threats
    - HARASSMENT: Harassing content
    - HARASSMENT_THREATENING: Harassing content with threats
    - SELF_HARM: Content about self-harm
    - SELF_HARM_INTENT: Content expressing intent to self-harm
    - SELF_HARM_INSTRUCTIONS: Instructions for self-harm
    - SEXUAL: Sexual content
    - SEXUAL_MINORS: Sexual content involving minors
    - VIOLENCE: Violent content
    - VIOLENCE_GRAPHIC: Graphic violent content
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING

from requirements_graphrag_api.observability import traceable_safe

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class ToxicityCategory(StrEnum):
    """Toxicity categories aligned with OpenAI Moderation API."""

    HATE = "hate"
    HATE_THREATENING = "hate/threatening"
    HARASSMENT = "harassment"
    HARASSMENT_THREATENING = "harassment/threatening"
    SELF_HARM = "self-harm"
    SELF_HARM_INTENT = "self-harm/intent"
    SELF_HARM_INSTRUCTIONS = "self-harm/instructions"
    SEXUAL = "sexual"
    SEXUAL_MINORS = "sexual/minors"
    VIOLENCE = "violence"
    VIOLENCE_GRAPHIC = "violence/graphic"


@dataclass(frozen=True, slots=True)
class ToxicityConfig:
    """Configuration for toxicity detection.

    Attributes:
        enabled: Whether toxicity detection is enabled.
        use_full_check: Whether to use OpenAI Moderation API (slower but accurate).
        block_threshold: Confidence threshold for blocking content.
        categories_to_block: Categories that should trigger blocking.
        categories_to_warn: Categories that should trigger warnings.
    """

    enabled: bool = True
    use_full_check: bool = True
    block_threshold: float = 0.7
    categories_to_block: tuple[str, ...] = (
        "hate",
        "hate/threatening",
        "harassment",
        "harassment/threatening",
        "self-harm",
        "self-harm/intent",
        "self-harm/instructions",
        "sexual/minors",
        "violence/graphic",
    )
    categories_to_warn: tuple[str, ...] = (
        "sexual",
        "violence",
    )


@dataclass(frozen=True, slots=True)
class ToxicityResult:
    """Result of toxicity check.

    Attributes:
        is_toxic: Whether any toxicity was detected.
        categories: List of detected toxicity categories.
        category_scores: Score (0-1) for each category checked.
        confidence: Overall confidence score (max of detected scores).
        should_block: Whether the content should be blocked.
        should_warn: Whether a warning should be logged.
        check_type: Type of check performed ("fast" or "full").
    """

    is_toxic: bool
    categories: tuple[ToxicityCategory, ...]
    category_scores: dict[str, float]
    confidence: float
    should_block: bool
    should_warn: bool
    check_type: str


# Default configuration
DEFAULT_CONFIG = ToxicityConfig()


@lru_cache(maxsize=1)
def _get_profanity_filter() -> object:
    """Get initialized profanity filter (singleton).

    Returns:
        Configured profanity filter instance.

    Note:
        Uses lru_cache to avoid reinitializing on each call.
    """
    from better_profanity import profanity

    profanity.load_censor_words()
    return profanity


async def check_toxicity_fast(text: str) -> ToxicityResult:
    """Fast profanity check using word lists (~1ms).

    This is a quick first-pass check that catches obvious profanity.
    For comprehensive toxicity detection, use check_toxicity_full.

    Args:
        text: The text to check for profanity.

    Returns:
        ToxicityResult with fast check results.

    Example:
        >>> result = await check_toxicity_fast("Hello world")
        >>> result.is_toxic
        False
    """
    if not text or not text.strip():
        return ToxicityResult(
            is_toxic=False,
            categories=(),
            category_scores={},
            confidence=0.0,
            should_block=False,
            should_warn=False,
            check_type="fast",
        )

    try:
        profanity = _get_profanity_filter()
        contains_profanity = profanity.contains_profanity(text)

        return ToxicityResult(
            is_toxic=contains_profanity,
            categories=(ToxicityCategory.HARASSMENT,) if contains_profanity else (),
            category_scores={"harassment": 0.8 if contains_profanity else 0.0},
            confidence=0.8 if contains_profanity else 0.0,
            should_block=contains_profanity,
            should_warn=contains_profanity,
            check_type="fast",
        )
    except ImportError:
        logger.warning(
            "better-profanity not installed. Fast toxicity check disabled. "
            "Install with: pip install better-profanity"
        )
        return ToxicityResult(
            is_toxic=False,
            categories=(),
            category_scores={},
            confidence=0.0,
            should_block=False,
            should_warn=False,
            check_type="fast",
        )
    except Exception:
        logger.exception("Error during fast toxicity check")
        return ToxicityResult(
            is_toxic=False,
            categories=(),
            category_scores={},
            confidence=0.0,
            should_block=False,
            should_warn=False,
            check_type="fast",
        )


def _openai_category_to_enum(category_name: str) -> ToxicityCategory | None:
    """Convert OpenAI category name to ToxicityCategory enum.

    Args:
        category_name: OpenAI category name (uses underscores).

    Returns:
        Corresponding ToxicityCategory or None if not mapped.
    """
    # OpenAI uses underscores, our enum uses hyphens
    normalized = category_name.replace("_", "-").replace("/", "/")

    # Handle the slash variants
    category_map = {
        "hate": ToxicityCategory.HATE,
        "hate-threatening": ToxicityCategory.HATE_THREATENING,
        "harassment": ToxicityCategory.HARASSMENT,
        "harassment-threatening": ToxicityCategory.HARASSMENT_THREATENING,
        "self-harm": ToxicityCategory.SELF_HARM,
        "self-harm-intent": ToxicityCategory.SELF_HARM_INTENT,
        "self-harm-instructions": ToxicityCategory.SELF_HARM_INSTRUCTIONS,
        "sexual": ToxicityCategory.SEXUAL,
        "sexual-minors": ToxicityCategory.SEXUAL_MINORS,
        "violence": ToxicityCategory.VIOLENCE,
        "violence-graphic": ToxicityCategory.VIOLENCE_GRAPHIC,
    }

    return category_map.get(normalized)


async def check_toxicity_full(
    text: str,
    openai_client: AsyncOpenAI,
    config: ToxicityConfig = DEFAULT_CONFIG,
) -> ToxicityResult:
    """Full toxicity check using OpenAI Moderation API.

    Provides comprehensive toxicity detection across multiple categories
    with confidence scores.

    Args:
        text: The text to check for toxicity.
        openai_client: Async OpenAI client for API calls.
        config: Toxicity configuration.

    Returns:
        ToxicityResult with full moderation results.

    Example:
        >>> from openai import AsyncOpenAI
        >>> client = AsyncOpenAI()
        >>> result = await check_toxicity_full("Hello world", client)
        >>> result.is_toxic
        False
    """
    if not text or not text.strip():
        return ToxicityResult(
            is_toxic=False,
            categories=(),
            category_scores={},
            confidence=0.0,
            should_block=False,
            should_warn=False,
            check_type="full",
        )

    try:
        response = await openai_client.moderations.create(input=text)
        result = response.results[0]

        categories: list[ToxicityCategory] = []
        scores: dict[str, float] = {}
        should_block = False
        should_warn = False

        # Process each category from the moderation response
        categories_dict = result.categories.model_dump()
        scores_dict = result.category_scores.model_dump()

        for category_name, flagged in categories_dict.items():
            score = scores_dict.get(category_name, 0.0)
            # Normalize category name for storage
            normalized_name = category_name.replace("_", "/")
            scores[normalized_name] = score

            if flagged:
                toxicity_category = _openai_category_to_enum(category_name)
                if toxicity_category:
                    categories.append(toxicity_category)

                    # Check if this category should block
                    if normalized_name in config.categories_to_block:
                        should_block = True
                    elif normalized_name in config.categories_to_warn:
                        should_warn = True

        # Calculate confidence as max score of flagged categories
        confidence = max(scores.values()) if scores else 0.0

        # Apply threshold for blocking
        if result.flagged and confidence >= config.block_threshold:
            # Check if any flagged category is in the block list
            for cat in categories:
                if cat.value in config.categories_to_block:
                    should_block = True
                    break

        return ToxicityResult(
            is_toxic=result.flagged,
            categories=tuple(categories),
            category_scores=scores,
            confidence=confidence,
            should_block=should_block,
            should_warn=should_warn or result.flagged,
            check_type="full",
        )

    except Exception:
        logger.exception("Error during OpenAI moderation API call")
        # On error, return safe result (fail open for availability)
        return ToxicityResult(
            is_toxic=False,
            categories=(),
            category_scores={},
            confidence=0.0,
            should_block=False,
            should_warn=False,
            check_type="full",
        )


@traceable_safe(name="check_toxicity", run_type="chain")
async def check_toxicity(
    text: str,
    openai_client: AsyncOpenAI | None = None,
    use_full_check: bool = True,
    config: ToxicityConfig = DEFAULT_CONFIG,
) -> ToxicityResult:
    """Check text for toxicity with configurable depth.

    Performs a fast profanity check first, then optionally uses the
    OpenAI Moderation API for comprehensive detection.

    Args:
        text: The text to check for toxicity.
        openai_client: Async OpenAI client (required for full check).
        use_full_check: Whether to use OpenAI Moderation API if fast check passes.
        config: Toxicity configuration.

    Returns:
        ToxicityResult with combined check results.

    Example:
        >>> result = await check_toxicity("What is requirements traceability?")
        >>> result.is_toxic
        False
        >>> result.should_block
        False

        >>> result = await check_toxicity("some inappropriate text", openai_client)
        >>> result.check_type
        'fast'  # or 'full' if fast check passed
    """
    if not config.enabled:
        return ToxicityResult(
            is_toxic=False,
            categories=(),
            category_scores={},
            confidence=0.0,
            should_block=False,
            should_warn=False,
            check_type="disabled",
        )

    # Always do fast check first
    fast_result = await check_toxicity_fast(text)

    # If fast check detects toxicity, return immediately
    if fast_result.is_toxic:
        logger.debug("Fast toxicity check flagged content")
        return fast_result

    # If fast check passes and full check requested, do OpenAI moderation
    if use_full_check and openai_client is not None:
        full_result = await check_toxicity_full(text, openai_client, config)
        return full_result

    # Return fast result if no full check
    return fast_result
