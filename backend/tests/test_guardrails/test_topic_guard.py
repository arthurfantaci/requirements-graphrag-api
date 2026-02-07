"""Tests for topic boundary enforcement."""

from __future__ import annotations

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


class TestTopicRelevanceKeywordOnly:
    """Test that check_topic_relevance uses keyword-only classification."""

    @pytest.mark.asyncio
    async def test_keyword_only_classification(self):
        result = await check_topic_relevance(
            "What is requirements traceability?",
        )
        assert result.check_type == "keyword"

    @pytest.mark.asyncio
    async def test_backwards_compat_llm_param_ignored(self):
        # llm= should be silently ignored via **_kwargs
        result = await check_topic_relevance(
            "What is requirements traceability?",
            llm="ignored",
        )
        assert result.check_type == "keyword"

    @pytest.mark.asyncio
    async def test_disabled_config(self):
        config = TopicGuardConfig(enabled=False)
        result = await check_topic_relevance(
            "Tell me about politics",
            config=config,
        )
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
        assert "traceability" in response.lower() or "requirements" in response.lower()

    @pytest.mark.asyncio
    async def test_redirect_response_is_polite(self):
        result = await check_topic_relevance_fast("What's the weather?")
        response = result.suggested_response
        assert response is not None
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
        result = await check_topic_relevance_fast(query)
        assert result.classification == TopicClassification.IN_SCOPE
        assert result.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_meta_conversation_word_boundary(self):
        """Test partial-word overlap does not trigger meta-conversation."""
        result = await check_topic_relevance_fast("How do conversations work in Jama Connect?")
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
        result = await check_topic_relevance_fast(query)
        assert result.classification == TopicClassification.IN_SCOPE
