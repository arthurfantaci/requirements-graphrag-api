"""Tests for output content filtering."""

from __future__ import annotations

import pytest

from requirements_graphrag_api.guardrails.output_filter import (
    BLOCKED_RESPONSE,
    LOW_CONFIDENCE_DISCLAIMER,
    OutputFilterConfig,
    OutputFilterResult,
    filter_output,
)
from requirements_graphrag_api.guardrails.toxicity import ToxicityConfig


class TestOutputFilterConfig:
    """Test OutputFilterConfig dataclass."""

    def test_default_config(self):
        config = OutputFilterConfig()
        assert config.enabled is True
        assert config.toxicity_enabled is True
        assert config.add_disclaimers is True
        assert config.confidence_threshold == 0.6

    def test_custom_config(self):
        config = OutputFilterConfig(
            enabled=False,
            confidence_threshold=0.8,
        )
        assert config.enabled is False
        assert config.confidence_threshold == 0.8

    def test_config_is_frozen(self):
        config = OutputFilterConfig()
        with pytest.raises(AttributeError):
            config.enabled = False  # type: ignore[misc]


class TestOutputFilterResult:
    """Test OutputFilterResult dataclass."""

    def test_result_creation(self):
        result = OutputFilterResult(
            is_safe=True,
            filtered_content="Test output",
            original_content="Test output",
            warnings=(),
            modifications=(),
            confidence_score=0.9,
            should_add_disclaimer=False,
        )
        assert result.is_safe is True
        assert result.confidence_score == 0.9

    def test_result_is_frozen(self):
        result = OutputFilterResult(
            is_safe=True,
            filtered_content="Test",
            original_content="Test",
            warnings=(),
            modifications=(),
            confidence_score=0.9,
            should_add_disclaimer=False,
        )
        with pytest.raises(AttributeError):
            result.is_safe = False  # type: ignore[misc]


class TestFilterOutput:
    """Test output filtering functionality."""

    @pytest.mark.asyncio
    async def test_clean_output_passes(self):
        result = await filter_output(
            output=(
                "Requirements traceability is the ability to trace requirements "
                "throughout the development lifecycle."
            ),
            original_query="What is requirements traceability?",
            retrieved_sources=[{"content": "traceability definition"}],
        )
        assert result.is_safe is True
        assert result.confidence_score > 0.5
        assert result.filtered_content == result.original_content

    @pytest.mark.asyncio
    async def test_toxic_output_blocked(self):
        config = OutputFilterConfig()
        toxicity_config = ToxicityConfig()

        result = await filter_output(
            output="What the hell damn shit is this crap",
            original_query="What is traceability?",
            retrieved_sources=[{"content": "test"}],
            config=config,
            toxicity_config=toxicity_config,
        )
        assert result.is_safe is False
        assert result.blocked_reason == "toxicity"
        assert result.filtered_content == config.blocked_response_message

    @pytest.mark.asyncio
    async def test_disabled_filter_passes_everything(self):
        config = OutputFilterConfig(enabled=False)
        result = await filter_output(
            output="Any content",
            original_query="Any query",
            config=config,
        )
        assert result.is_safe is True
        assert result.confidence_score == 1.0

    @pytest.mark.asyncio
    async def test_no_sources_reduces_confidence(self):
        result = await filter_output(
            output="A detailed answer about requirements.",
            original_query="What is traceability?",
            retrieved_sources=[],  # No sources
        )
        assert result.confidence_score < 1.0
        assert any("No sources" in w for w in result.warnings)


class TestConfidenceScoring:
    """Test confidence score calculation."""

    @pytest.mark.asyncio
    async def test_short_response_limited_sources_reduce_confidence(self):
        config = OutputFilterConfig(toxicity_enabled=False)
        result = await filter_output(
            output="I think this might be related to requirements. I'm not sure though.",
            original_query="What is traceability?",
            retrieved_sources=[{"content": "test"}],
            config=config,
        )
        # Short response (-0.2) + limited sources (-0.1) = 0.7
        assert result.confidence_score <= 0.8
        assert any("short" in w.lower() or "limited" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_confident_language_maintains_score(self):
        config = OutputFilterConfig(toxicity_enabled=False)
        result = await filter_output(
            output=(
                "According to the document, requirements traceability is defined "
                "as the ability to trace requirements."
            ),
            original_query="What is traceability?",
            retrieved_sources=[{"content": "test"}, {"content": "test2"}],
            config=config,
        )
        assert result.confidence_score >= 0.7

    @pytest.mark.asyncio
    async def test_very_short_response_reduces_confidence(self):
        config = OutputFilterConfig(toxicity_enabled=False)
        result = await filter_output(
            output="Yes.",
            original_query="Is traceability important?",
            retrieved_sources=[{"content": "test"}],
            config=config,
        )
        assert result.confidence_score < 0.9
        assert any("short" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_limited_sources_warning(self):
        config = OutputFilterConfig(toxicity_enabled=False)
        result = await filter_output(
            output="A detailed explanation about requirements management.",
            original_query="What is requirements management?",
            retrieved_sources=[{"content": "single source"}],  # Only one source
            config=config,
        )
        assert any("limited" in w.lower() for w in result.warnings)


class TestDisclaimerInjection:
    """Test disclaimer injection for low-confidence answers."""

    @pytest.mark.asyncio
    async def test_disclaimer_added_for_low_confidence(self):
        config = OutputFilterConfig(
            toxicity_enabled=False,
            confidence_threshold=0.8,
            add_disclaimers=True,
        )
        result = await filter_output(
            output="I think the answer might be related to traceability.",
            original_query="What is traceability?",
            retrieved_sources=[],  # No sources = lower confidence
            config=config,
        )
        assert result.should_add_disclaimer is True
        assert LOW_CONFIDENCE_DISCLAIMER in result.filtered_content
        assert "Added low-confidence disclaimer" in result.modifications

    @pytest.mark.asyncio
    async def test_no_disclaimer_for_high_confidence(self):
        config = OutputFilterConfig(
            toxicity_enabled=False,
            confidence_threshold=0.6,
            add_disclaimers=True,
        )
        result = await filter_output(
            output=(
                "According to the document, requirements traceability is the ability "
                "to trace requirements throughout the lifecycle."
            ),
            original_query="What is traceability?",
            retrieved_sources=[
                {"content": "source1"},
                {"content": "source2"},
                {"content": "source3"},
            ],
            config=config,
        )
        assert result.should_add_disclaimer is False
        assert LOW_CONFIDENCE_DISCLAIMER not in result.filtered_content

    @pytest.mark.asyncio
    async def test_disclaimer_disabled_in_config(self):
        config = OutputFilterConfig(
            toxicity_enabled=False,
            add_disclaimers=False,
        )
        result = await filter_output(
            output="I'm not sure about this.",
            original_query="Question",
            retrieved_sources=[],
            config=config,
        )
        assert result.should_add_disclaimer is False
        assert LOW_CONFIDENCE_DISCLAIMER not in result.filtered_content


class TestConfidenceFactors:
    """Test that confidence depends only on source count and response length."""

    @pytest.mark.asyncio
    async def test_limited_sources_reduces_confidence(self):
        config = OutputFilterConfig(toxicity_enabled=False)
        result = await filter_output(
            output="I don't know the exact definition, but it might be related to tracking.",
            original_query="Define traceability",
            retrieved_sources=[{"content": "test"}],
            config=config,
        )
        # limited sources (-0.1) + may be incomplete (-0.1) = 0.8
        assert result.confidence_score <= 0.8
        assert any("limited" in w.lower() or "incomplete" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_hedging_language_does_not_affect_score(self):
        """Hedging language detection was removed; confidence comes from sources + length only."""
        config = OutputFilterConfig(toxicity_enabled=False)
        result = await filter_output(
            output="I believe requirements traceability is important for project success.",
            original_query="Why is traceability important?",
            retrieved_sources=[{"content": "test"}],
            config=config,
        )
        # limited sources (-0.1), may be incomplete (-0.1) = 0.8
        assert result.confidence_score <= 0.9

    @pytest.mark.asyncio
    async def test_sufficient_sources_and_length_full_confidence(self):
        config = OutputFilterConfig(toxicity_enabled=False)
        result = await filter_output(
            output=(
                "According to the retrieved documents, requirements traceability is "
                "defined as the systematic tracking of requirements throughout the "
                "entire product development lifecycle from elicitation to validation."
            ),
            original_query="Define traceability",
            retrieved_sources=[{"content": "test"}, {"content": "test2"}],
            config=config,
        )
        assert result.confidence_score == 1.0


class TestBlockedResponses:
    """Test blocked response handling."""

    def test_blocked_response_constant(self):
        assert "apologize" in BLOCKED_RESPONSE.lower()
        assert "requirements management" in BLOCKED_RESPONSE.lower()

    def test_disclaimer_constant(self):
        assert "Note" in LOW_CONFIDENCE_DISCLAIMER
        assert "verify" in LOW_CONFIDENCE_DISCLAIMER.lower()
