"""Tests for toxicity detection."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from requirements_graphrag_api.guardrails.toxicity import (
    ToxicityCategory,
    ToxicityConfig,
    ToxicityResult,
    check_toxicity,
    check_toxicity_fast,
    check_toxicity_full,
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
        # Using mild profanity that better-profanity should catch
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


class TestToxicityFullCheck:
    """Test full toxicity check using OpenAI Moderation API."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        client = AsyncMock()

        # Create mock moderation response
        mock_result = MagicMock()
        mock_result.flagged = False
        mock_result.categories = MagicMock()
        mock_result.categories.model_dump.return_value = {
            "hate": False,
            "hate_threatening": False,
            "harassment": False,
            "harassment_threatening": False,
            "self_harm": False,
            "self_harm_intent": False,
            "self_harm_instructions": False,
            "sexual": False,
            "sexual_minors": False,
            "violence": False,
            "violence_graphic": False,
        }
        mock_result.category_scores = MagicMock()
        mock_result.category_scores.model_dump.return_value = {
            "hate": 0.01,
            "hate_threatening": 0.001,
            "harassment": 0.02,
            "harassment_threatening": 0.001,
            "self_harm": 0.001,
            "self_harm_intent": 0.001,
            "self_harm_instructions": 0.001,
            "sexual": 0.01,
            "sexual_minors": 0.001,
            "violence": 0.01,
            "violence_graphic": 0.001,
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
            "hate_threatening": False,
            "harassment": True,
            "harassment_threatening": False,
            "self_harm": False,
            "self_harm_intent": False,
            "self_harm_instructions": False,
            "sexual": False,
            "sexual_minors": False,
            "violence": False,
            "violence_graphic": False,
        }
        mock_result.category_scores = MagicMock()
        mock_result.category_scores.model_dump.return_value = {
            "hate": 0.85,
            "hate_threatening": 0.1,
            "harassment": 0.75,
            "harassment_threatening": 0.05,
            "self_harm": 0.01,
            "self_harm_intent": 0.01,
            "self_harm_instructions": 0.01,
            "sexual": 0.01,
            "sexual_minors": 0.001,
            "violence": 0.02,
            "violence_graphic": 0.01,
        }

        mock_response = MagicMock()
        mock_response.results = [mock_result]

        client.moderations.create = AsyncMock(return_value=mock_response)
        return client

    @pytest.mark.asyncio
    async def test_clean_text_passes_full_check(self, mock_openai_client):
        result = await check_toxicity_full(
            "What is requirements traceability?",
            mock_openai_client,
        )
        assert result.is_toxic is False
        assert result.should_block is False
        assert result.check_type == "full"

    @pytest.mark.asyncio
    async def test_toxic_text_blocked(self, mock_toxic_client):
        result = await check_toxicity_full(
            "Some hateful content here",
            mock_toxic_client,
        )
        assert result.is_toxic is True
        assert result.should_block is True
        assert result.check_type == "full"
        assert len(result.categories) > 0

    @pytest.mark.asyncio
    async def test_empty_string_full_check(self, mock_openai_client):
        result = await check_toxicity_full("", mock_openai_client)
        assert result.is_toxic is False
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_category_scores_populated(self, mock_openai_client):
        result = await check_toxicity_full("Test content", mock_openai_client)
        assert len(result.category_scores) > 0

    @pytest.mark.asyncio
    async def test_api_error_returns_safe_result(self, mock_openai_client):
        mock_openai_client.moderations.create.side_effect = Exception("API error")
        result = await check_toxicity_full("Test content", mock_openai_client)
        assert result.is_toxic is False
        assert result.should_block is False


class TestToxicityCheck:
    """Test combined toxicity check function."""

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

    @pytest.mark.asyncio
    async def test_fast_check_only_when_no_client(self):
        result = await check_toxicity("Clean text", openai_client=None)
        assert result.check_type == "fast"

    @pytest.mark.asyncio
    async def test_full_check_when_client_provided(self, mock_openai_client):
        result = await check_toxicity(
            "Clean text",
            openai_client=mock_openai_client,
            use_full_check=True,
        )
        assert result.check_type == "full"

    @pytest.mark.asyncio
    async def test_fast_check_blocks_before_full_check(self, mock_openai_client):
        # Profanity should be caught by fast check
        result = await check_toxicity(
            "This is damn annoying",
            openai_client=mock_openai_client,
        )
        # Fast check catches it, so full check never runs
        assert result.check_type == "fast"
        assert result.is_toxic is True

    @pytest.mark.asyncio
    async def test_disabled_config_returns_clean(self):
        config = ToxicityConfig(enabled=False)
        result = await check_toxicity("Any text", config=config)
        assert result.is_toxic is False
        assert result.check_type == "disabled"


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
        assert config.use_full_check is True
        assert config.block_threshold == 0.7
        assert "hate" in config.categories_to_block

    def test_custom_config(self):
        config = ToxicityConfig(
            enabled=False,
            block_threshold=0.5,
        )
        assert config.enabled is False
        assert config.block_threshold == 0.5

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
