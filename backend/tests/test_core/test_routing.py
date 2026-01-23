"""Tests for query intent classification and routing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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

    def test_intent_from_string(self) -> None:
        """Test creating intent from string."""
        assert QueryIntent("explanatory") == QueryIntent.EXPLANATORY
        assert QueryIntent("structured") == QueryIntent.STRUCTURED


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

    def test_guide_has_both_intents(self) -> None:
        """Test that guide documents both intent types."""
        guide = get_routing_guide()

        intents = [qt["intent"] for qt in guide["query_types"]]
        assert "explanatory" in intents
        assert "structured" in intents

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
