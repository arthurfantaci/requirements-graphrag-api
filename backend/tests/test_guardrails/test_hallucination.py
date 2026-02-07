"""Tests for hallucination detection (no-op stub)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from requirements_graphrag_api.guardrails.hallucination import (
    HALLUCINATION_WARNING,
    GroundingLevel,
    HallucinationCheckResult,
    check_hallucination,
    check_hallucination_sync,
)


class TestGroundingLevel:
    """Test GroundingLevel enum."""

    def test_levels_exist(self):
        assert GroundingLevel.FULLY_GROUNDED == "fully_grounded"
        assert GroundingLevel.MOSTLY_GROUNDED == "mostly_grounded"
        assert GroundingLevel.PARTIALLY_GROUNDED == "partially_grounded"
        assert GroundingLevel.UNGROUNDED == "ungrounded"


class TestHallucinationCheckResult:
    """Test HallucinationCheckResult dataclass."""

    def test_result_is_frozen(self):
        result = HallucinationCheckResult(
            grounding_level=GroundingLevel.FULLY_GROUNDED,
            confidence=0.95,
            unsupported_claims=(),
            reasoning="All claims verified",
            should_add_warning=False,
        )
        with pytest.raises(AttributeError):
            result.grounding_level = GroundingLevel.UNGROUNDED  # type: ignore[misc]


class TestCheckHallucination:
    """Test async hallucination check (no-op stub)."""

    @pytest.mark.asyncio
    async def test_empty_response_is_grounded(self):
        result = await check_hallucination("", [])
        assert result.grounding_level == GroundingLevel.FULLY_GROUNDED
        assert result.checked is False
        assert result.should_add_warning is False

    @pytest.mark.asyncio
    async def test_no_sources_is_ungrounded(self):
        result = await check_hallucination("Some response", [])
        assert result.grounding_level == GroundingLevel.UNGROUNDED
        assert result.checked is False
        assert result.should_add_warning is True

    @pytest.mark.asyncio
    async def test_with_sources_is_grounded(self):
        sources = [{"title": "Source", "content": "Relevant content"}]
        result = await check_hallucination("Response text", sources)
        assert result.grounding_level == GroundingLevel.FULLY_GROUNDED
        assert result.confidence == 0.95
        assert result.should_add_warning is False
        assert result.checked is False

    @pytest.mark.asyncio
    async def test_llm_param_accepted_but_ignored(self):
        """Stub accepts llm parameter for backwards compatibility."""
        llm = MagicMock()
        sources = [{"title": "Source", "content": "Content"}]
        result = await check_hallucination("Response", sources, llm)
        assert result.grounding_level == GroundingLevel.FULLY_GROUNDED
        llm.ainvoke.assert_not_called()


class TestCheckHallucinationSync:
    """Test sync hallucination check (no-op stub)."""

    def test_empty_response_is_grounded(self):
        result = check_hallucination_sync("", [])
        assert result.grounding_level == GroundingLevel.FULLY_GROUNDED
        assert result.checked is False

    def test_no_sources_is_ungrounded(self):
        result = check_hallucination_sync("Some response", [])
        assert result.grounding_level == GroundingLevel.UNGROUNDED
        assert result.should_add_warning is True

    def test_with_sources_is_grounded(self):
        sources = [{"title": "Source", "content": "Content"}]
        result = check_hallucination_sync("Response", sources)
        assert result.grounding_level == GroundingLevel.FULLY_GROUNDED
        assert result.should_add_warning is False


class TestHallucinationWarning:
    """Test the warning message constants."""

    def test_warning_exists(self):
        assert HALLUCINATION_WARNING is not None
        assert "verify" in HALLUCINATION_WARNING.lower()
