"""Hallucination detection for RAG responses.

No-op stub: LLM-based grounding verification removed (rated 2/5 justified
for a single-user authenticated API). The interface is preserved so callers
don't need changes -- functions return static FULLY_GROUNDED / UNGROUNDED
results based solely on whether sources were provided.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from requirements_graphrag_api.observability import traceable_safe

logger = logging.getLogger(__name__)


class GroundingLevel(StrEnum):
    """Level of factual grounding in sources."""

    FULLY_GROUNDED = "fully_grounded"
    MOSTLY_GROUNDED = "mostly_grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    UNGROUNDED = "ungrounded"


@dataclass(frozen=True, slots=True)
class HallucinationCheckResult:
    """Result of hallucination check."""

    grounding_level: GroundingLevel
    confidence: float
    unsupported_claims: tuple[str, ...]
    reasoning: str
    should_add_warning: bool
    checked: bool = True


# Singleton results
_SAFE_RESULT = HallucinationCheckResult(
    grounding_level=GroundingLevel.FULLY_GROUNDED,
    confidence=0.95,
    unsupported_claims=(),
    reasoning="Sources provided (LLM verification disabled)",
    should_add_warning=False,
    checked=False,
)

_NO_SOURCES_RESULT = HallucinationCheckResult(
    grounding_level=GroundingLevel.UNGROUNDED,
    confidence=0.5,
    unsupported_claims=(),
    reasoning="No sources provided for verification",
    should_add_warning=True,
    checked=False,
)

_EMPTY_RESPONSE_RESULT = HallucinationCheckResult(
    grounding_level=GroundingLevel.FULLY_GROUNDED,
    confidence=1.0,
    unsupported_claims=(),
    reasoning="Empty response has no claims to verify",
    should_add_warning=False,
    checked=False,
)


@traceable_safe(name="check_hallucination", run_type="chain")
async def check_hallucination(
    response: str,
    sources: list[dict[str, Any]],
    llm: object = None,
    max_sources: int = 5,
) -> HallucinationCheckResult:
    """Check if response is grounded in sources (no-op stub).

    Returns FULLY_GROUNDED when sources are present, UNGROUNDED otherwise.
    The ``llm`` parameter is accepted but ignored for backwards compatibility.
    """
    if not response or not response.strip():
        return _EMPTY_RESPONSE_RESULT
    if not sources:
        return _NO_SOURCES_RESULT
    return _SAFE_RESULT


@traceable_safe(name="check_hallucination_sync", run_type="chain")
def check_hallucination_sync(
    response: str,
    sources: list[dict[str, Any]],
    llm: object = None,
    max_sources: int = 5,
) -> HallucinationCheckResult:
    """Synchronous version of hallucination check (no-op stub)."""
    if not response or not response.strip():
        return _EMPTY_RESPONSE_RESULT
    if not sources:
        return _NO_SOURCES_RESULT
    return _SAFE_RESULT


# Standard warning message to append when hallucination is detected
HALLUCINATION_WARNING = (
    "\n\n---\n"
    "_Warning: This response may contain information not fully supported "
    "by the knowledge base. Please verify important details with authoritative sources._"
)

# Alternative shorter warning
HALLUCINATION_WARNING_SHORT = "\n\n_Warning: Some claims may not be directly supported by sources._"
