"""Tests for output content filtering."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client that returns clean results."""
        client = AsyncMock()

        mock_result = MagicMock()
        mock_result.flagged = False
        mock_result.categories = MagicMock()
        mock_result.categories.model_dump.return_value = {
            "hate": False,
            "harassment": False,
            "self_harm": False,
            "sexual": False,
            "violence": False,
        }
        mock_result.category_scores = MagicMock()
        mock_result.category_scores.model_dump.return_value = {
            "hate": 0.01,
            "harassment": 0.01,
            "self_harm": 0.01,
            "sexual": 0.01,
            "violence": 0.01,
        }

        mock_response = MagicMock()
        mock_response.results = [mock_result]

        client.moderations.create = AsyncMock(return_value=mock_response)
        return client

    @pytest.fixture
    def mock_toxic_client(self):
        """Create a mock client that returns toxic content."""
        client = AsyncMock()

        mock_result = MagicMock()
        mock_result.flagged = True
        mock_result.categories = MagicMock()
        mock_result.categories.model_dump.return_value = {
            "hate": True,
            "harassment": True,
            "self_harm": False,
            "sexual": False,
            "violence": False,
        }
        mock_result.category_scores = MagicMock()
        mock_result.category_scores.model_dump.return_value = {
            "hate": 0.9,
            "harassment": 0.8,
            "self_harm": 0.01,
            "sexual": 0.01,
            "violence": 0.01,
        }

        mock_response = MagicMock()
        mock_response.results = [mock_result]

        client.moderations.create = AsyncMock(return_value=mock_response)
        return client

    @pytest.mark.asyncio
    async def test_clean_output_passes(self, mock_openai_client):
        result = await filter_output(
            output=(
                "Requirements traceability is the ability to trace requirements "
                "throughout the development lifecycle."
            ),
            original_query="What is requirements traceability?",
            retrieved_sources=[{"content": "traceability definition"}],
            openai_client=mock_openai_client,
        )
        assert result.is_safe is True
        assert result.confidence_score > 0.5
        assert result.filtered_content == result.original_content

    @pytest.mark.asyncio
    async def test_toxic_output_blocked(self, mock_toxic_client):
        config = OutputFilterConfig()
        toxicity_config = ToxicityConfig(use_full_check=True)

        result = await filter_output(
            output="Some hateful content here",
            original_query="What is traceability?",
            retrieved_sources=[{"content": "test"}],
            config=config,
            toxicity_config=toxicity_config,
            openai_client=mock_toxic_client,
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
    async def test_no_sources_reduces_confidence(self, mock_openai_client):
        result = await filter_output(
            output="A detailed answer about requirements.",
            original_query="What is traceability?",
            retrieved_sources=[],  # No sources
            openai_client=mock_openai_client,
        )
        assert result.confidence_score < 1.0
        assert any("No sources" in w for w in result.warnings)


class TestConfidenceScoring:
    """Test confidence score calculation."""

    @pytest.mark.asyncio
    async def test_hallucination_indicators_reduce_confidence(self):
        config = OutputFilterConfig(toxicity_enabled=False)
        result = await filter_output(
            output="I think this might be related to requirements. I'm not sure though.",
            original_query="What is traceability?",
            retrieved_sources=[{"content": "test"}],
            config=config,
        )
        assert result.confidence_score < 0.8
        assert any("uncertainty" in w.lower() for w in result.warnings)

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


class TestHallucinationIndicators:
    """Test detection of hallucination indicators."""

    @pytest.mark.asyncio
    async def test_i_dont_know_detected(self):
        config = OutputFilterConfig(toxicity_enabled=False)
        result = await filter_output(
            output="I don't know the exact definition, but it might be related to tracking.",
            original_query="Define traceability",
            retrieved_sources=[{"content": "test"}],
            config=config,
        )
        assert result.confidence_score < 0.8
        assert any("uncertainty" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_i_believe_detected(self):
        config = OutputFilterConfig(toxicity_enabled=False)
        result = await filter_output(
            output="I believe requirements traceability is important for project success.",
            original_query="Why is traceability important?",
            retrieved_sources=[{"content": "test"}],
            config=config,
        )
        assert result.confidence_score < 1.0

    @pytest.mark.asyncio
    async def test_based_on_training_detected(self):
        config = OutputFilterConfig(toxicity_enabled=False)
        result = await filter_output(
            output="Based on my training, I would say that requirements are important.",
            original_query="Why are requirements important?",
            retrieved_sources=[{"content": "test"}],
            config=config,
        )
        assert result.confidence_score < 1.0
        assert any("uncertainty" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_according_to_boosts_confidence(self):
        config = OutputFilterConfig(toxicity_enabled=False)
        result = await filter_output(
            output=(
                "According to the retrieved documents, requirements traceability is "
                "defined as the systematic tracking of requirements."
            ),
            original_query="Define traceability",
            retrieved_sources=[{"content": "test"}, {"content": "test2"}],
            config=config,
        )
        assert result.confidence_score >= 0.8


class TestBlockedResponses:
    """Test blocked response handling."""

    def test_blocked_response_constant(self):
        assert "apologize" in BLOCKED_RESPONSE.lower()
        assert "requirements management" in BLOCKED_RESPONSE.lower()

    def test_disclaimer_constant(self):
        assert "Note" in LOW_CONFIDENCE_DISCLAIMER
        assert "verify" in LOW_CONFIDENCE_DISCLAIMER.lower()
