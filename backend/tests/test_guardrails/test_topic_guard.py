"""Tests for topic boundary enforcement."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from requirements_graphrag_api.guardrails.topic_guard import (
    TopicCheckResult,
    TopicClassification,
    TopicGuardConfig,
    check_topic_relevance,
    check_topic_relevance_fast,
)


class TestTopicClassificationEnum:
    """Test TopicClassification enum."""

    def test_all_classifications_exist(self):
        assert TopicClassification.IN_SCOPE == "in_scope"
        assert TopicClassification.OUT_OF_SCOPE == "out_of_scope"
        assert TopicClassification.BORDERLINE == "borderline"


class TestTopicGuardConfig:
    """Test TopicGuardConfig dataclass."""

    def test_default_config(self):
        config = TopicGuardConfig()
        assert config.enabled is True
        assert config.use_llm_classification is True
        assert config.allow_borderline is True
        assert "specialized assistant" in config.out_of_scope_response

    def test_custom_config(self):
        config = TopicGuardConfig(enabled=False, allow_borderline=False)
        assert config.enabled is False
        assert config.allow_borderline is False

    def test_config_is_frozen(self):
        config = TopicGuardConfig()
        with pytest.raises(AttributeError):
            config.enabled = False  # type: ignore[misc]


class TestTopicCheckResult:
    """Test TopicCheckResult dataclass."""

    def test_result_creation(self):
        result = TopicCheckResult(
            classification=TopicClassification.IN_SCOPE,
            confidence=0.9,
            suggested_response=None,
            reasoning="Contains in-scope topic",
            check_type="keyword",
        )
        assert result.classification == TopicClassification.IN_SCOPE
        assert result.confidence == 0.9

    def test_result_is_frozen(self):
        result = TopicCheckResult(
            classification=TopicClassification.IN_SCOPE,
            confidence=0.9,
            suggested_response=None,
            reasoning="Test",
            check_type="keyword",
        )
        with pytest.raises(AttributeError):
            result.classification = TopicClassification.OUT_OF_SCOPE  # type: ignore[misc]


class TestTopicRelevanceFast:
    """Test fast keyword-based topic relevance check."""

    @pytest.mark.asyncio
    async def test_in_scope_requirements_questions(self):
        queries = [
            "What is requirements traceability?",
            "How do I create a traceability matrix?",
            "Explain systems engineering best practices",
            "What are functional requirements?",
            "How does Jama Connect handle baselines?",
        ]
        for query in queries:
            result = await check_topic_relevance_fast(query)
            assert result.classification == TopicClassification.IN_SCOPE, (
                f"Should be in scope: {query}"
            )
            assert result.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_out_of_scope_politics(self):
        queries = [
            "What do you think about the election?",
            "Which political party is better?",
            "Tell me about politics",
        ]
        for query in queries:
            result = await check_topic_relevance_fast(query)
            assert result.classification == TopicClassification.OUT_OF_SCOPE, (
                f"Should be out of scope: {query}"
            )
            assert result.suggested_response is not None

    @pytest.mark.asyncio
    async def test_out_of_scope_medical(self):
        result = await check_topic_relevance_fast("What medication should I take for my headache?")
        assert result.classification == TopicClassification.OUT_OF_SCOPE
        assert result.suggested_response is not None

    @pytest.mark.asyncio
    async def test_out_of_scope_financial(self):
        result = await check_topic_relevance_fast("Should I invest in cryptocurrency?")
        assert result.classification == TopicClassification.OUT_OF_SCOPE

    @pytest.mark.asyncio
    async def test_out_of_scope_entertainment(self):
        queries = [
            "What's your favorite movie?",
            "Tell me a joke",
            "What sports team is the best?",
        ]
        for query in queries:
            result = await check_topic_relevance_fast(query)
            assert result.classification == TopicClassification.OUT_OF_SCOPE, (
                f"Should be out of scope: {query}"
            )

    @pytest.mark.asyncio
    async def test_borderline_ambiguous_query(self):
        # Query that doesn't match any keyword
        result = await check_topic_relevance_fast("How do I do this thing?")
        assert result.classification == TopicClassification.BORDERLINE
        assert result.confidence == 0.5

    @pytest.mark.asyncio
    async def test_empty_string(self):
        result = await check_topic_relevance_fast("")
        assert result.classification == TopicClassification.IN_SCOPE
        assert result.confidence == 0.5

    @pytest.mark.asyncio
    async def test_redirect_response_content(self):
        result = await check_topic_relevance_fast("Tell me about politics")
        assert result.suggested_response is not None
        assert (
            "Requirements" in result.suggested_response
            or "requirements" in result.suggested_response
        )


class TestTopicRelevanceWithLLM:
    """Test topic relevance check with LLM classification."""

    @pytest.fixture
    def mock_llm_in_scope(self):
        """Mock LLM that returns IN_SCOPE."""
        llm = AsyncMock()
        response = MagicMock()
        response.content = "IN_SCOPE"
        llm.ainvoke = AsyncMock(return_value=response)
        return llm

    @pytest.fixture
    def mock_llm_out_of_scope(self):
        """Mock LLM that returns OUT_OF_SCOPE."""
        llm = AsyncMock()
        response = MagicMock()
        response.content = "OUT_OF_SCOPE"
        llm.ainvoke = AsyncMock(return_value=response)
        return llm

    @pytest.fixture
    def mock_llm_borderline(self):
        """Mock LLM that returns BORDERLINE."""
        llm = AsyncMock()
        response = MagicMock()
        response.content = "BORDERLINE"
        llm.ainvoke = AsyncMock(return_value=response)
        return llm

    @pytest.mark.asyncio
    async def test_llm_classification_in_scope(self, mock_llm_in_scope):
        # Ambiguous query that needs LLM
        result = await check_topic_relevance(
            "How do I manage this process?",
            llm=mock_llm_in_scope,
        )
        assert result.classification == TopicClassification.IN_SCOPE
        assert result.check_type == "llm"

    @pytest.mark.asyncio
    async def test_llm_classification_out_of_scope(self, mock_llm_out_of_scope):
        # Ambiguous query that LLM classifies as out of scope
        result = await check_topic_relevance(
            "What should I do today?",
            llm=mock_llm_out_of_scope,
        )
        assert result.classification == TopicClassification.OUT_OF_SCOPE
        assert result.check_type == "llm"
        assert result.suggested_response is not None

    @pytest.mark.asyncio
    async def test_llm_classification_borderline(self, mock_llm_borderline):
        result = await check_topic_relevance(
            "How do I organize my team?",
            llm=mock_llm_borderline,
        )
        assert result.classification == TopicClassification.BORDERLINE

    @pytest.mark.asyncio
    async def test_keyword_match_skips_llm(self, mock_llm_in_scope):
        # Clear in-scope query shouldn't need LLM
        result = await check_topic_relevance(
            "What is requirements traceability?",
            llm=mock_llm_in_scope,
        )
        assert result.check_type == "keyword"
        # LLM should not have been called
        mock_llm_in_scope.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_llm_provided(self):
        # Ambiguous query without LLM falls back to keyword result
        result = await check_topic_relevance(
            "How do I manage this?",
            llm=None,
        )
        assert result.check_type == "keyword"
        assert result.classification == TopicClassification.BORDERLINE

    @pytest.mark.asyncio
    async def test_llm_error_falls_back(self, mock_llm_in_scope):
        mock_llm_in_scope.ainvoke.side_effect = Exception("LLM error")
        result = await check_topic_relevance(
            "Ambiguous question here",
            llm=mock_llm_in_scope,
        )
        # Should fall back to keyword result
        assert result.check_type == "keyword"

    @pytest.mark.asyncio
    async def test_disabled_config(self):
        config = TopicGuardConfig(enabled=False)
        result = await check_topic_relevance(
            "Tell me about politics",
            config=config,
        )
        # Should be allowed when disabled
        assert result.classification == TopicClassification.IN_SCOPE
        assert result.check_type == "disabled"


class TestKeywordMatching:
    """Test specific keyword matching behavior."""

    @pytest.mark.asyncio
    async def test_case_insensitive_matching(self):
        queries = [
            "REQUIREMENTS TRACEABILITY",
            "Requirements Traceability",
            "requirements traceability",
        ]
        for query in queries:
            result = await check_topic_relevance_fast(query)
            assert result.classification == TopicClassification.IN_SCOPE

    @pytest.mark.asyncio
    async def test_partial_word_matching(self):
        # "iso " should match but not words containing "iso"
        result = await check_topic_relevance_fast("ISO 26262 compliance")
        assert result.classification == TopicClassification.IN_SCOPE

    @pytest.mark.asyncio
    async def test_jama_software_in_scope(self):
        queries = [
            "How does Jama Software work?",
            "Jama Connect configuration",
        ]
        for query in queries:
            result = await check_topic_relevance_fast(query)
            assert result.classification == TopicClassification.IN_SCOPE


class TestRedirectResponses:
    """Test redirect response generation."""

    @pytest.mark.asyncio
    async def test_redirect_response_has_suggestions(self):
        result = await check_topic_relevance_fast("Tell me a joke")
        response = result.suggested_response
        assert response is not None
        # Should suggest in-scope topics
        assert "traceability" in response.lower() or "requirements" in response.lower()

    @pytest.mark.asyncio
    async def test_redirect_response_is_polite(self):
        result = await check_topic_relevance_fast("What's the weather?")
        response = result.suggested_response
        assert response is not None
        # Should be polite, not confrontational
        assert "specialized" in response.lower() or "help" in response.lower()


class TestMetaConversationBypass:
    """Tests for meta-conversation keyword detection in topic guard."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query",
        [
            "What was my first question?",
            "Summarize our conversation",
            "What did I ask earlier?",
            "Repeat what you said about traceability",
            "Recap our discussion",
            "What have we discussed so far?",
            "You told me about baselines",
            "Remind me what you said",
        ],
    )
    async def test_meta_conversation_returns_in_scope(self, query: str):
        """Test meta-conversation queries bypass topic guard as IN_SCOPE."""
        result = await check_topic_relevance_fast(query)
        assert result.classification == TopicClassification.IN_SCOPE
        assert result.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_meta_conversation_word_boundary(self):
        """Test partial-word overlap does not trigger meta-conversation.

        'conversations' (plural) should NOT match 'conversation' (singular)
        thanks to word-boundary regex.
        """
        result = await check_topic_relevance_fast("How do conversations work in Jama Connect?")
        # Should NOT be meta-conversation — 'conversations' != 'conversation'
        assert not (
            result.classification == TopicClassification.IN_SCOPE and result.confidence == 0.95
        ), "Should not be classified as meta-conversation"


class TestTopicGuardRegression:
    """Regression tests ensuring topic guard still works after meta-conversation bypass."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query",
        [
            "Tell me about politics",
            "What medication should I take?",
            "Should I invest in cryptocurrency?",
            "What's your favorite movie?",
        ],
    )
    async def test_out_of_scope_still_blocked(self, query: str):
        """Test out-of-scope queries still correctly classified."""
        result = await check_topic_relevance_fast(query)
        assert result.classification == TopicClassification.OUT_OF_SCOPE

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query",
        [
            "What is requirements traceability?",
            "How does Jama Connect work?",
            "Explain ISO 26262 compliance",
        ],
    )
    async def test_in_scope_still_passes(self, query: str):
        """Test in-scope queries still correctly classified."""
        result = await check_topic_relevance_fast(query)
        assert result.classification == TopicClassification.IN_SCOPE
