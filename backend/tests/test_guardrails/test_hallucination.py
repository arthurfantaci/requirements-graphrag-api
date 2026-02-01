"""Tests for hallucination detection."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from requirements_graphrag_api.guardrails.hallucination import (
    HALLUCINATION_WARNING,
    GroundingLevel,
    HallucinationCheckResult,
    _format_sources,
    _parse_llm_response,
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


class TestFormatSources:
    """Test source formatting for prompts."""

    def test_formats_sources_with_title_and_content(self):
        sources = [
            {"title": "Chapter 1", "content": "Traceability is important..."},
            {"title": "Chapter 2", "content": "Requirements must be tracked..."},
        ]
        result = _format_sources(sources)
        assert "Source 1: Chapter 1" in result
        assert "Source 2: Chapter 2" in result
        assert "Traceability is important" in result

    def test_uses_name_if_no_title(self):
        sources = [{"name": "Doc1", "content": "Content"}]
        result = _format_sources(sources)
        assert "Source 1: Doc1" in result

    def test_uses_text_if_no_content(self):
        sources = [{"title": "Title", "text": "Some text"}]
        result = _format_sources(sources)
        assert "Some text" in result

    def test_truncates_long_content(self):
        sources = [{"title": "Long", "content": "x" * 3000}]
        result = _format_sources(sources)
        assert len(result) < 3000 + 100  # Some buffer for formatting
        assert "..." in result


class TestParseLlmResponse:
    """Test LLM response parsing."""

    def test_parses_valid_json(self):
        content = (
            '{"grounding_level": "fully_grounded", '
            '"unsupported_claims": [], "reasoning": "All good"}'
        )
        result = _parse_llm_response(content)
        assert result["grounding_level"] == "fully_grounded"
        assert result["reasoning"] == "All good"

    def test_extracts_json_from_prose(self):
        content = (
            "Here is my analysis:\n"
            '{"grounding_level": "mostly_grounded", '
            '"unsupported_claims": ["claim1"], '
            '"reasoning": "One minor issue"}\n'
            "That's all."
        )
        result = _parse_llm_response(content)
        assert result["grounding_level"] == "mostly_grounded"

    def test_returns_defaults_on_invalid_json(self):
        content = "This is not JSON at all"
        result = _parse_llm_response(content)
        assert result["grounding_level"] == "partially_grounded"


class TestCheckHallucination:
    """Test async hallucination checking."""

    @pytest.mark.asyncio
    async def test_empty_response_is_grounded(self):
        llm = MagicMock()
        result = await check_hallucination("", [], llm)
        assert result.grounding_level == GroundingLevel.FULLY_GROUNDED
        assert result.checked is False
        assert result.should_add_warning is False

    @pytest.mark.asyncio
    async def test_no_sources_is_ungrounded(self):
        llm = MagicMock()
        result = await check_hallucination("Some response", [], llm)
        assert result.grounding_level == GroundingLevel.UNGROUNDED
        assert result.checked is False
        assert result.should_add_warning is True

    @pytest.mark.asyncio
    async def test_fully_grounded_response(self):
        llm = AsyncMock()
        llm.ainvoke.return_value = MagicMock(
            content=(
                '{"grounding_level": "fully_grounded", '
                '"unsupported_claims": [], "reasoning": "Perfect"}'
            )
        )

        sources = [{"title": "Source", "content": "Relevant content"}]
        result = await check_hallucination("Response text", sources, llm)

        assert result.grounding_level == GroundingLevel.FULLY_GROUNDED
        assert result.confidence == 0.95
        assert result.should_add_warning is False

    @pytest.mark.asyncio
    async def test_partially_grounded_adds_warning(self):
        llm = AsyncMock()
        llm.ainvoke.return_value = MagicMock(
            content=(
                '{"grounding_level": "partially_grounded", '
                '"unsupported_claims": ["claim1"], "reasoning": "Some issues"}'
            )
        )

        sources = [{"title": "Source", "content": "Content"}]
        result = await check_hallucination("Response", sources, llm)

        assert result.grounding_level == GroundingLevel.PARTIALLY_GROUNDED
        assert result.should_add_warning is True
        assert len(result.unsupported_claims) == 1

    @pytest.mark.asyncio
    async def test_ungrounded_adds_warning(self):
        llm = AsyncMock()
        llm.ainvoke.return_value = MagicMock(
            content=(
                '{"grounding_level": "ungrounded", '
                '"unsupported_claims": ["all claims"], '
                '"reasoning": "Nothing supported"}'
            )
        )

        sources = [{"title": "Source", "content": "Content"}]
        result = await check_hallucination("Response", sources, llm)

        assert result.grounding_level == GroundingLevel.UNGROUNDED
        assert result.should_add_warning is True
        assert result.confidence == 0.2

    @pytest.mark.asyncio
    async def test_llm_error_returns_conservative_result(self):
        llm = AsyncMock()
        llm.ainvoke.side_effect = Exception("LLM error")

        sources = [{"title": "Source", "content": "Content"}]
        result = await check_hallucination("Response", sources, llm)

        # Should return conservative defaults
        assert result.grounding_level == GroundingLevel.PARTIALLY_GROUNDED
        assert result.should_add_warning is True
        assert "error" in result.reasoning.lower() or "unable" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_limits_sources(self):
        llm = AsyncMock()
        llm.ainvoke.return_value = MagicMock(
            content=(
                '{"grounding_level": "fully_grounded", "unsupported_claims": [], "reasoning": "OK"}'
            )
        )

        # Create 10 sources
        sources = [{"title": f"Source {i}", "content": f"Content {i}"} for i in range(10)]

        await check_hallucination("Response", sources, llm, max_sources=3)

        # Check that prompt only contains 3 sources
        call_args = llm.ainvoke.call_args[0][0]
        # Source 4 should not be in the prompt (0-indexed means source 3)
        assert "Source 4" not in call_args or "Source 10" not in call_args


class TestCheckHallucinationSync:
    """Test sync hallucination checking."""

    def test_empty_response_is_grounded(self):
        llm = MagicMock()
        result = check_hallucination_sync("", [], llm)
        assert result.grounding_level == GroundingLevel.FULLY_GROUNDED
        assert result.checked is False

    def test_no_sources_is_ungrounded(self):
        llm = MagicMock()
        result = check_hallucination_sync("Some response", [], llm)
        assert result.grounding_level == GroundingLevel.UNGROUNDED
        assert result.should_add_warning is True

    def test_fully_grounded_sync(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content=(
                '{"grounding_level": "fully_grounded", '
                '"unsupported_claims": [], "reasoning": "Good"}'
            )
        )

        sources = [{"title": "Source", "content": "Content"}]
        result = check_hallucination_sync("Response", sources, llm)

        assert result.grounding_level == GroundingLevel.FULLY_GROUNDED
        assert result.should_add_warning is False


class TestHallucinationWarning:
    """Test the warning message constants."""

    def test_warning_exists(self):
        assert HALLUCINATION_WARNING is not None
        assert "⚠️" in HALLUCINATION_WARNING
        assert "verify" in HALLUCINATION_WARNING.lower()
