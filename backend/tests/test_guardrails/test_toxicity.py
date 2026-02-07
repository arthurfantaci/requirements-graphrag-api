"""Tests for toxicity detection."""

from __future__ import annotations

import pytest

from requirements_graphrag_api.guardrails.toxicity import (
    ToxicityCategory,
    ToxicityConfig,
    ToxicityResult,
    check_toxicity,
    check_toxicity_fast,
)


class TestToxicityFastCheck:
    """Test fast profanity check using word lists."""

    @pytest.mark.asyncio
    async def test_clean_text_passes(self):
        result = await check_toxicity_fast("What is requirements traceability?")
        assert result.is_toxic is False
        assert result.should_block is False
        assert result.check_type == "fast"
        assert len(result.categories) == 0

    @pytest.mark.asyncio
    async def test_profanity_detected(self):
        result = await check_toxicity_fast("This is damn annoying")
        assert result.is_toxic is True
        assert result.should_block is True
        assert result.check_type == "fast"
        assert ToxicityCategory.HARASSMENT in result.categories

    @pytest.mark.asyncio
    async def test_empty_string(self):
        result = await check_toxicity_fast("")
        assert result.is_toxic is False
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_whitespace_only(self):
        result = await check_toxicity_fast("   \n\t   ")
        assert result.is_toxic is False

    @pytest.mark.asyncio
    async def test_technical_question_clean(self):
        queries = [
            "How do I manage system requirements?",
            "Explain the V-model for product development",
            "What are the best practices for requirements documentation?",
        ]
        for query in queries:
            result = await check_toxicity_fast(query)
            assert result.is_toxic is False, f"False positive on: {query}"


class TestToxicityCheck:
    """Test combined toxicity check function."""

    @pytest.mark.asyncio
    async def test_fast_check_only(self):
        result = await check_toxicity("Clean text")
        assert result.check_type == "fast"

    @pytest.mark.asyncio
    async def test_fast_check_catches_profanity(self):
        result = await check_toxicity("This is damn annoying")
        assert result.check_type == "fast"
        assert result.is_toxic is True

    @pytest.mark.asyncio
    async def test_disabled_config_returns_clean(self):
        config = ToxicityConfig(enabled=False)
        result = await check_toxicity("Any text", config=config)
        assert result.is_toxic is False
        assert result.check_type == "disabled"

    @pytest.mark.asyncio
    async def test_extra_kwargs_ignored(self):
        # Backwards compat: openai_client= and use_full_check= should not raise
        result = await check_toxicity(
            "Clean text",
            openai_client="ignored",
            use_full_check=True,
        )
        assert result.check_type == "fast"


class TestToxicityCategory:
    """Test ToxicityCategory enum."""

    def test_all_categories_exist(self):
        expected = [
            "hate",
            "hate/threatening",
            "harassment",
            "harassment/threatening",
            "self-harm",
            "self-harm/intent",
            "self-harm/instructions",
            "sexual",
            "sexual/minors",
            "violence",
            "violence/graphic",
        ]
        for cat in expected:
            assert ToxicityCategory(cat) is not None


class TestToxicityConfig:
    """Test ToxicityConfig dataclass."""

    def test_default_config(self):
        config = ToxicityConfig()
        assert config.enabled is True

    def test_custom_config(self):
        config = ToxicityConfig(enabled=False)
        assert config.enabled is False

    def test_config_is_frozen(self):
        config = ToxicityConfig()
        with pytest.raises(AttributeError):
            config.enabled = False  # type: ignore[misc]


class TestToxicityResult:
    """Test ToxicityResult dataclass."""

    def test_result_creation(self):
        result = ToxicityResult(
            is_toxic=True,
            categories=(ToxicityCategory.HARASSMENT,),
            category_scores={"harassment": 0.8},
            confidence=0.8,
            should_block=True,
            should_warn=True,
            check_type="fast",
        )
        assert result.is_toxic is True
        assert result.confidence == 0.8

    def test_result_is_frozen(self):
        result = ToxicityResult(
            is_toxic=False,
            categories=(),
            category_scores={},
            confidence=0.0,
            should_block=False,
            should_warn=False,
            check_type="fast",
        )
        with pytest.raises(AttributeError):
            result.is_toxic = True  # type: ignore[misc]
