"""Tests for query intent classification and routing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
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


class TestMetaConversationalPatterns:
    """Tests for meta-operational CONVERSATIONAL patterns added for T11 bug fix."""

    @pytest.mark.parametrize(
        "query",
        [
            "Have you already provided a table titled 'RESULTS'?",
            "Did you already show me the standards?",
            "You already provided that information earlier",
            "You already showed me the webinars list",
            "You already gave me a list of articles",
            "Can you repeat the results from before?",
            "From your previous response, what was the total?",
            "Can you update the table you provided earlier?",
            "Are the results you showed me complete?",
            "From earlier in our chat, what standards did you mention?",
        ],
    )
    def test_meta_operational_patterns(self, query: str) -> None:
        """Test that meta-operational queries route to CONVERSATIONAL."""
        result = _quick_classify(query)
        assert result == QueryIntent.CONVERSATIONAL

    def test_have_you_already_case_insensitive(self) -> None:
        """Test case insensitivity of new patterns."""
        assert _quick_classify("HAVE YOU ALREADY provided a table?") == QueryIntent.CONVERSATIONAL
        assert _quick_classify("Can You Repeat that?") == QueryIntent.CONVERSATIONAL

    def test_t11_production_bug_regression(self) -> None:
        """Regression for T11: exact production query that was misrouted to STRUCTURED."""
        query = (
            "Have you already provided a table titled 'RESULTS' with columns: "
            "STANDARD_NAME, WEBINAR_TITLE, WEBINAR_URL?"
        )
        result = _quick_classify(query)
        assert result == QueryIntent.CONVERSATIONAL


class TestStructuralQueriesNotCaptured:
    """Verify meta-operational patterns don't false-positive on genuine structured queries."""

    @pytest.mark.parametrize(
        "query",
        [
            "Provide a table of all standards",
            "Show me all results for ISO 26262",
            "Have the results been published?",
            "Can you list all webinars?",
            "From the knowledge base, list all tools",
        ],
    )
    def test_structural_not_conversational(self, query: str) -> None:
        """Test that genuine structured/domain queries don't trigger CONVERSATIONAL."""
        result = _quick_classify(query)
        assert result != QueryIntent.CONVERSATIONAL


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

        The chain is ``prompt | llm`` (no StrOutputParser). We mock the
        prompt's ``__or__`` to short-circuit and produce a chain whose
        ``ainvoke`` returns an AIMessage-like mock with JSON content.
        """
        json_response = '{"intent": "conversational", "confidence": "high"}'

        # Build AIMessage-like response with .content and .response_metadata
        mock_ai_msg = MagicMock()
        mock_ai_msg.content = json_response
        mock_ai_msg.response_metadata = {
            "token_usage": {"prompt_tokens": 50, "completion_tokens": 20},
        }

        # prompt | llm is a 2-step chain
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=mock_ai_msg)

        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        with (
            patch(
                "requirements_graphrag_api.core.routing._quick_classify",
                return_value=None,
            ),
            patch(
                "requirements_graphrag_api.core.routing.get_prompt",
                new_callable=AsyncMock,
                return_value=mock_prompt,
            ),
            patch("requirements_graphrag_api.core.routing.ChatOpenAI"),
        ):
            result = await classify_intent(mock_config, "Can you go back to my earlier point?")
            assert result == QueryIntent.CONVERSATIONAL


class TestClassifyIntentLlmFailure:
    """Tests for fail-soft behaviour when the classification LLM raises a provider error."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        config = MagicMock()
        config.chat_model = "gpt-4o-mini"
        config.openai_api_key = "test-key"
        return config

    def _make_response(self, status_code: int, body: dict) -> httpx.Response:
        return httpx.Response(
            status_code,
            json=body,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

    @pytest.mark.asyncio
    async def test_rate_limit_insufficient_quota_defaults_explanatory(
        self, mock_config: MagicMock
    ) -> None:
        response = self._make_response(
            429,
            {"error": {"type": "insufficient_quota", "code": "insufficient_quota"}},
        )
        exc = openai.RateLimitError(
            "You exceeded your current quota",
            response=response,
            body={"error": {"type": "insufficient_quota", "code": "insufficient_quota"}},
        )

        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(side_effect=exc)
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        with (
            patch(
                "requirements_graphrag_api.core.routing._quick_classify",
                return_value=None,
            ),
            patch(
                "requirements_graphrag_api.core.routing.get_prompt",
                new_callable=AsyncMock,
                return_value=mock_prompt,
            ),
            patch("requirements_graphrag_api.core.routing.ChatOpenAI"),
        ):
            result = await classify_intent(mock_config, "What is requirements traceability?")

        assert result == QueryIntent.EXPLANATORY

    @pytest.mark.asyncio
    async def test_authentication_error_defaults_explanatory(self, mock_config: MagicMock) -> None:
        response = self._make_response(401, {"error": {"type": "invalid_api_key"}})
        exc = openai.AuthenticationError(
            "Invalid API key",
            response=response,
            body={"error": {"type": "invalid_api_key"}},
        )

        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(side_effect=exc)
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        with (
            patch(
                "requirements_graphrag_api.core.routing._quick_classify",
                return_value=None,
            ),
            patch(
                "requirements_graphrag_api.core.routing.get_prompt",
                new_callable=AsyncMock,
                return_value=mock_prompt,
            ),
            patch("requirements_graphrag_api.core.routing.ChatOpenAI"),
        ):
            result = await classify_intent(mock_config, "What is requirements traceability?")

        assert result == QueryIntent.EXPLANATORY

    @pytest.mark.asyncio
    async def test_generic_exception_defaults_explanatory(self, mock_config: MagicMock) -> None:
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(side_effect=RuntimeError("unexpected"))
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        with (
            patch(
                "requirements_graphrag_api.core.routing._quick_classify",
                return_value=None,
            ),
            patch(
                "requirements_graphrag_api.core.routing.get_prompt",
                new_callable=AsyncMock,
                return_value=mock_prompt,
            ),
            patch("requirements_graphrag_api.core.routing.ChatOpenAI"),
        ):
            result = await classify_intent(mock_config, "What is requirements traceability?")

        assert result == QueryIntent.EXPLANATORY


class TestClassifyIntentRateLimitRetry:
    """Tests for bounded retry-with-backoff on RateLimitError."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        config = MagicMock()
        config.chat_model = "gpt-4o-mini"
        config.openai_api_key = "test-key"
        return config

    def _make_rate_limit_error(self) -> openai.RateLimitError:
        response = httpx.Response(
            429,
            json={"error": {"type": "rate_limit_exceeded", "code": "rate_limit_exceeded"}},
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        return openai.RateLimitError(
            "Rate limit exceeded",
            response=response,
            body={"error": {"type": "rate_limit_exceeded", "code": "rate_limit_exceeded"}},
        )

    @pytest.mark.asyncio
    async def test_rate_limit_then_success_returns_classified_intent(
        self, mock_config: MagicMock
    ) -> None:
        """First attempt hits 429, retry succeeds — caller sees the LLM classification."""
        json_response = '{"intent": "structured", "confidence": "high"}'
        mock_ai_msg = MagicMock()
        mock_ai_msg.content = json_response
        mock_ai_msg.response_metadata = {
            "token_usage": {"prompt_tokens": 50, "completion_tokens": 20},
        }

        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(
            side_effect=[self._make_rate_limit_error(), mock_ai_msg]
        )
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        with (
            patch(
                "requirements_graphrag_api.core.routing._quick_classify",
                return_value=None,
            ),
            patch(
                "requirements_graphrag_api.core.routing.get_prompt",
                new_callable=AsyncMock,
                return_value=mock_prompt,
            ),
            patch("requirements_graphrag_api.core.routing.ChatOpenAI"),
            patch(
                "requirements_graphrag_api.core.routing.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
        ):
            result = await classify_intent(mock_config, "ambiguous query")

        assert result == QueryIntent.STRUCTURED
        # Should have slept exactly once with the first-attempt backoff (0.5s).
        assert mock_chain.ainvoke.await_count == 2
        mock_sleep.assert_awaited_once_with(0.5)

    @pytest.mark.asyncio
    async def test_rate_limit_exhausts_retries_falls_back_with_warning(
        self, mock_config: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """All attempts hit 429 — function falls back to EXPLANATORY and logs a warning."""
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(side_effect=self._make_rate_limit_error())
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        with (
            patch(
                "requirements_graphrag_api.core.routing._quick_classify",
                return_value=None,
            ),
            patch(
                "requirements_graphrag_api.core.routing.get_prompt",
                new_callable=AsyncMock,
                return_value=mock_prompt,
            ),
            patch("requirements_graphrag_api.core.routing.ChatOpenAI"),
            patch(
                "requirements_graphrag_api.core.routing.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
        ):
            result = await classify_intent(mock_config, "ambiguous query")

        assert result == QueryIntent.EXPLANATORY
        # MAX_INTENT_RETRIES=3 attempts, two intermediate sleeps (0.5s, 1.0s).
        assert mock_chain.ainvoke.await_count == 3
        assert mock_sleep.await_count == 2
        sleep_durations = [call.args[0] for call in mock_sleep.await_args_list]
        assert sleep_durations == [0.5, 1.0]

    @pytest.mark.asyncio
    async def test_non_rate_limit_exception_does_not_retry(
        self, mock_config: MagicMock
    ) -> None:
        """A non-RateLimitError exception falls back immediately without retry."""
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        with (
            patch(
                "requirements_graphrag_api.core.routing._quick_classify",
                return_value=None,
            ),
            patch(
                "requirements_graphrag_api.core.routing.get_prompt",
                new_callable=AsyncMock,
                return_value=mock_prompt,
            ),
            patch("requirements_graphrag_api.core.routing.ChatOpenAI"),
            patch(
                "requirements_graphrag_api.core.routing.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
        ):
            result = await classify_intent(mock_config, "ambiguous query")

        assert result == QueryIntent.EXPLANATORY
        assert mock_chain.ainvoke.await_count == 1
        mock_sleep.assert_not_awaited()
