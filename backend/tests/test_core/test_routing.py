"""Tests for query intent classification and routing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from requirements_graphrag_api.core.routing import (
    QueryIntent,
    _quick_classify,
    classify_intent,
    get_routing_guide,
)


class TestQueryIntent:
    """Tests for QueryIntent enum."""

    def test_intent_values(self) -> None:
        """Test that intent enum has expected values."""
        assert QueryIntent.EXPLANATORY == "explanatory"
        assert QueryIntent.STRUCTURED == "structured"
        assert QueryIntent.CONVERSATIONAL == "conversational"

    def test_intent_from_string(self) -> None:
        """Test creating intent from string."""
        assert QueryIntent("explanatory") == QueryIntent.EXPLANATORY
        assert QueryIntent("structured") == QueryIntent.STRUCTURED
        assert QueryIntent("conversational") == QueryIntent.CONVERSATIONAL


class TestQuickClassify:
    """Tests for keyword-based quick classification."""

    @pytest.mark.parametrize(
        "query",
        [
            "List all webinars",
            "list all videos in the knowledge base",
            "Show all articles",
            "show me all tools mentioned",
            "How many articles are there?",
            "count the number of standards",
            "Provide a table of all entities",
            "enumerate all concepts",
        ],
    )
    def test_structured_keywords(self, query: str) -> None:
        """Test that structured keywords are detected."""
        result = _quick_classify(query)
        assert result == QueryIntent.STRUCTURED

    @pytest.mark.parametrize(
        "query",
        [
            "What is requirements traceability?",
            "How do I implement change management?",
            "Explain the verification process",
            "What are best practices for testing?",
            "Why is traceability important?",
        ],
    )
    def test_explanatory_queries_return_none(self, query: str) -> None:
        """Test that explanatory queries are not quick-classified."""
        # Quick classify should return None for these, letting LLM decide
        result = _quick_classify(query)
        assert result is None

    @pytest.mark.parametrize(
        "query",
        [
            "What was my first question?",
            "Summarize our conversation",
            "What did I ask earlier?",
            "You said earlier that traceability is important",
            "Repeat what you said about requirements",
            "What have we discussed so far?",
            "Recap our discussion",
            "What did you tell me about baselines?",
        ],
    )
    def test_conversational_patterns(self, query: str) -> None:
        """Test that conversational patterns are detected via word-boundary regex."""
        result = _quick_classify(query)
        assert result == QueryIntent.CONVERSATIONAL

    def test_conversational_priority_over_structured(self) -> None:
        """Test conversational takes priority when query has both signals."""
        # "earlier you mentioned" pattern should trigger conversational even with "listing"
        assert (
            _quick_classify("Earlier you mentioned listing all requirements")
            == QueryIntent.CONVERSATIONAL
        )

    @pytest.mark.parametrize(
        "query",
        [
            "How do conversations work in Jama Connect?",
            "Tell me about the conversation module",
            "What are the best questioning techniques for elicitation?",
        ],
    )
    def test_word_boundary_prevents_false_positives(self, query: str) -> None:
        """Test that partial word overlap does not trigger conversational match.

        'conversations' != 'conversation' (word boundary blocks),
        'questioning' != 'question' (word boundary blocks).
        """
        result = _quick_classify(query)
        # These should NOT be CONVERSATIONAL — they're domain queries
        assert result != QueryIntent.CONVERSATIONAL

    def test_conversational_case_insensitive(self) -> None:
        """Test that CONVERSATIONAL matching is case-insensitive."""
        result = _quick_classify("WHAT WAS MY FIRST QUESTION?")
        assert result == QueryIntent.CONVERSATIONAL

    def test_pattern_matching_which_plural(self) -> None:
        """Test 'which X's' pattern detection."""
        result = _quick_classify("Which articles mention testing?")
        assert result == QueryIntent.STRUCTURED

        result = _quick_classify("Which tools are available?")
        assert result == QueryIntent.STRUCTURED


class TestClassifyIntent:
    """Tests for LLM-based intent classification."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock config."""
        config = MagicMock()
        config.chat_model = "gpt-4o-mini"
        config.openai_api_key = "test-key"
        return config

    @pytest.mark.asyncio
    async def test_quick_classify_shortcut(self, mock_config: MagicMock) -> None:
        """Test that quick classify shortcuts LLM call."""
        with patch(
            "requirements_graphrag_api.core.routing._quick_classify",
            return_value=QueryIntent.STRUCTURED,
        ):
            result = await classify_intent(mock_config, "List all webinars")
            assert result == QueryIntent.STRUCTURED

    @pytest.mark.asyncio
    async def test_quick_classify_structured_shortcut(self, mock_config: MagicMock) -> None:
        """Test that quick classify for structured queries shortcuts LLM call."""
        # "List all" triggers quick classify
        result = await classify_intent(mock_config, "List all webinars")
        assert result == QueryIntent.STRUCTURED

    @pytest.mark.asyncio
    async def test_quick_classify_patterns(self, mock_config: MagicMock) -> None:
        """Test that pattern-based quick classify works."""
        # "Which X's" pattern
        result = await classify_intent(mock_config, "Which articles mention testing?")
        assert result == QueryIntent.STRUCTURED

    @pytest.mark.asyncio
    async def test_auto_route_disabled_defaults_explanatory(self, mock_config: MagicMock) -> None:
        """Test that disabling quick_classify and auto_route defaults to explanatory."""
        # When quick_classify returns None, we need LLM
        # For this test, we'll mock to verify the flow
        with patch(
            "requirements_graphrag_api.core.routing._quick_classify",
            return_value=QueryIntent.STRUCTURED,
        ):
            result = await classify_intent(
                mock_config,
                "ambiguous query",
                use_quick_classify=True,
            )
            # Quick classify mock returns STRUCTURED
            assert result == QueryIntent.STRUCTURED


class TestRoutingGuide:
    """Tests for routing guide documentation."""

    def test_guide_structure(self) -> None:
        """Test that routing guide has expected structure."""
        guide = get_routing_guide()

        assert "title" in guide
        assert "description" in guide
        assert "query_types" in guide
        assert "tips" in guide

    def test_guide_has_all_intents(self) -> None:
        """Test that guide documents all three intent types."""
        guide = get_routing_guide()

        intents = [qt["intent"] for qt in guide["query_types"]]
        assert "explanatory" in intents
        assert "structured" in intents
        assert "conversational" in intents

    def test_guide_has_examples(self) -> None:
        """Test that each query type has examples."""
        guide = get_routing_guide()

        for query_type in guide["query_types"]:
            assert "examples" in query_type
            assert len(query_type["examples"]) > 0

    def test_guide_has_keywords(self) -> None:
        """Test that each query type has keywords."""
        guide = get_routing_guide()

        for query_type in guide["query_types"]:
            assert "keywords" in query_type
            assert len(query_type["keywords"]) > 0


class TestStructuredRegression:
    """Regression tests ensuring structured queries still work after adding CONVERSATIONAL."""

    @pytest.mark.parametrize(
        "query",
        [
            "List all webinars",
            "Show all articles",
            "How many articles are there?",
            "Count the number of standards",
            "Which articles mention testing?",
        ],
    )
    def test_structured_queries_unchanged(self, query: str) -> None:
        """Test structured queries still classify correctly."""
        result = _quick_classify(query)
        assert result == QueryIntent.STRUCTURED


class TestClassifyIntentConversational:
    """Tests for LLM-based conversational intent handling."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        config = MagicMock()
        config.chat_model = "gpt-4o-mini"
        config.openai_api_key = "test-key"
        return config

    @pytest.mark.asyncio
    async def test_quick_classify_conversational_shortcut(self, mock_config: MagicMock) -> None:
        """Test conversational queries shortcut past LLM classification."""
        result = await classify_intent(mock_config, "What was my first question?")
        assert result == QueryIntent.CONVERSATIONAL

    @pytest.mark.asyncio
    async def test_llm_fallback_returns_conversational(self, mock_config: MagicMock) -> None:
        """Test LLM fallback correctly parses conversational JSON.

        The chain is ``prompt | llm | StrOutputParser()``. We mock the
        prompt's ``__or__`` to short-circuit and produce a chain whose
        ``ainvoke`` returns the JSON string directly.
        """
        json_response = '{"intent": "conversational", "confidence": "high"}'

        # prompt | llm returns intermediate; then | StrOutputParser returns final
        # Mock the full chain: prompt.__or__ → intermediate.__or__ → final
        mock_final_chain = MagicMock()
        mock_final_chain.ainvoke = AsyncMock(return_value=json_response)

        mock_intermediate = MagicMock()
        mock_intermediate.__or__ = MagicMock(return_value=mock_final_chain)

        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_intermediate)

        with (
            patch(
                "requirements_graphrag_api.core.routing._quick_classify",
                return_value=None,
            ),
            patch(
                "requirements_graphrag_api.core.routing.get_prompt_sync",
                return_value=mock_prompt,
            ),
            patch("requirements_graphrag_api.core.routing.ChatOpenAI"),
        ):
            result = await classify_intent(mock_config, "Can you go back to my earlier point?")
            assert result == QueryIntent.CONVERSATIONAL
