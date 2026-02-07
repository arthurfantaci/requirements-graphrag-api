"""Toxicity detection for user inputs and LLM outputs.

Uses word-list based profanity check via better-profanity (~1ms).
OpenAI Moderation API removed -- profanity filter is sufficient
for a single-user authenticated API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from requirements_graphrag_api.observability import traceable_safe

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
    """Configuration for toxicity detection."""

    enabled: bool = True


@dataclass(frozen=True, slots=True)
class ToxicityResult:
    """Result of toxicity check."""

    is_toxic: bool
    categories: tuple[ToxicityCategory, ...]
    category_scores: dict[str, float]
    confidence: float
    should_block: bool
    should_warn: bool
    check_type: str


# Default configuration
DEFAULT_CONFIG = ToxicityConfig()

# Singleton safe result (reused for clean text)
_SAFE_RESULT = ToxicityResult(
    is_toxic=False,
    categories=(),
    category_scores={},
    confidence=0.0,
    should_block=False,
    should_warn=False,
    check_type="fast",
)


@lru_cache(maxsize=1)
def _get_profanity_filter() -> object:
    """Get initialized profanity filter (singleton)."""
    from better_profanity import profanity

    profanity.load_censor_words()
    return profanity


async def check_toxicity_fast(text: str) -> ToxicityResult:
    """Fast profanity check using word lists (~1ms).

    Args:
        text: The text to check for profanity.

    Returns:
        ToxicityResult with fast check results.
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
        return _SAFE_RESULT
    except Exception:
        logger.exception("Error during fast toxicity check")
        return _SAFE_RESULT


@traceable_safe(name="check_toxicity", run_type="chain")
async def check_toxicity(
    text: str,
    config: ToxicityConfig = DEFAULT_CONFIG,
    **_kwargs: object,
) -> ToxicityResult:
    """Check text for toxicity using fast profanity filter.

    The ``openai_client`` and ``use_full_check`` parameters previously
    accepted are silently ignored via ``**_kwargs`` for backwards
    compatibility.
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

    return await check_toxicity_fast(text)
