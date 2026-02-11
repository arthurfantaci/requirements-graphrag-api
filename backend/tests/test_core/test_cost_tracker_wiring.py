"""Tests for CostTracker integration with production LLM calls.

Verifies that get_global_cost_tracker().record_from_response() is called
after every direct LLM invocation in the production pipeline (#144).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from requirements_graphrag_api.evaluation.cost_analysis import (
    get_global_cost_tracker,
)
from tests.conftest import create_ai_message_mock


def _make_chain_mock(content: str) -> MagicMock:
    """Create a mock chain (prompt | llm) that returns an AIMessage-like mock."""
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=create_ai_message_mock(content))
    return mock_chain


# =============================================================================
# TEXT2CYPHER
# =============================================================================


class TestCostTrackerText2Cypher:
    """Verify cost tracking in generate_cypher."""

    @pytest.mark.asyncio
    async def test_generate_cypher_records_cost(self, mock_config):
        """generate_cypher should record cost with operation='text2cypher'."""
        from requirements_graphrag_api.core.text2cypher import generate_cypher

        cypher_response = "MATCH (n:Entity) RETURN count(n) AS total"
        mock_chain = _make_chain_mock(cypher_response)
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter(
            [MagicMock(**{"__getitem__": lambda s, k: {"label": "Entity", "count": 10}.get(k)})]
        )
        mock_session.run = MagicMock(return_value=mock_result)
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_session

        with (
            patch(
                "requirements_graphrag_api.core.text2cypher.get_prompt",
                new_callable=AsyncMock,
                return_value=mock_prompt,
            ),
            patch("requirements_graphrag_api.core.text2cypher.ChatOpenAI"),
        ):
            result = await generate_cypher(mock_config, mock_driver, "How many entities?")

        assert "MATCH" in result
        tracker = get_global_cost_tracker()
        calls = tracker.get_calls()
        assert len(calls) == 1
        assert calls[0].operation == "text2cypher"
        assert calls[0].model == mock_config.chat_model
        assert calls[0].input_tokens == 100
        assert calls[0].output_tokens == 50


# =============================================================================
# ROUTING
# =============================================================================


class TestCostTrackerRouting:
    """Verify cost tracking in classify_intent (LLM path only)."""

    @pytest.mark.asyncio
    async def test_classify_intent_records_cost(self, mock_config):
        """classify_intent LLM path should record cost with operation='intent_classification'."""
        from requirements_graphrag_api.core.routing import classify_intent

        json_response = '{"intent": "explanatory", "confidence": "high"}'
        mock_chain = _make_chain_mock(json_response)
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
            result = await classify_intent(mock_config, "What is traceability?")

        assert result.value == "explanatory"
        tracker = get_global_cost_tracker()
        calls = tracker.get_calls()
        assert len(calls) == 1
        assert calls[0].operation == "intent_classification"

    @pytest.mark.asyncio
    async def test_quick_classify_does_not_record_cost(self, mock_config):
        """Quick-classify path skips LLM and should NOT record any cost."""
        from requirements_graphrag_api.core.routing import classify_intent

        result = await classify_intent(mock_config, "List all webinars")
        assert result.value == "structured"
        tracker = get_global_cost_tracker()
        assert len(tracker.get_calls()) == 0


# =============================================================================
# RAG SUBGRAPH — expand_queries
# =============================================================================


class TestCostTrackerRAG:
    """Verify cost tracking in RAG subgraph nodes."""

    @pytest.mark.asyncio
    async def test_expand_queries_records_cost(self, mock_config):
        """expand_queries node should record cost with operation='query_expansion'."""
        from requirements_graphrag_api.core.agentic.subgraphs.rag import create_rag_subgraph

        expansion_json = json.dumps(
            {
                "queries": [
                    {"query": "requirements traceability", "strategy": "original"},
                    {"query": "trace requirements software", "strategy": "synonym"},
                ]
            }
        )

        mock_chain = _make_chain_mock(expansion_json)
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        mock_driver = MagicMock()
        mock_retriever = MagicMock()

        with (
            patch(
                "requirements_graphrag_api.core.agentic.subgraphs.rag.get_prompt",
                new_callable=AsyncMock,
                return_value=mock_prompt,
            ),
            patch("requirements_graphrag_api.core.agentic.subgraphs.rag.ChatOpenAI"),
            patch(
                "requirements_graphrag_api.core.retrieval.graph_enriched_search",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            graph = create_rag_subgraph(mock_config, mock_driver, mock_retriever)
            await graph.ainvoke({"query": "What is requirements traceability?"})

        tracker = get_global_cost_tracker()
        calls = [c for c in tracker.get_calls() if c.operation == "query_expansion"]
        assert len(calls) == 1
        assert calls[0].model == mock_config.conversational_model


# =============================================================================
# SYNTHESIS SUBGRAPH
# =============================================================================


class TestCostTrackerSynthesis:
    """Verify cost tracking in Synthesis subgraph nodes."""

    @pytest.mark.asyncio
    async def test_draft_answer_records_cost(self, mock_config):
        """draft_answer node should record cost with operation='synthesis_draft'."""
        from requirements_graphrag_api.core.agentic.subgraphs.synthesis import (
            create_synthesis_subgraph,
        )

        synthesis_json = json.dumps(
            {
                "answer": "Requirements traceability is...",
                "critique": {"confidence": 0.9, "completeness": "complete"},
                "citations": ["Source 1"],
            }
        )

        mock_chain = _make_chain_mock(synthesis_json)
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        with (
            patch(
                "requirements_graphrag_api.core.agentic.subgraphs.synthesis.get_prompt",
                new_callable=AsyncMock,
                return_value=mock_prompt,
            ),
            patch("requirements_graphrag_api.core.agentic.subgraphs.synthesis.ChatOpenAI"),
        ):
            graph = create_synthesis_subgraph(mock_config)
            await graph.ainvoke(
                {
                    "query": "What is requirements traceability?",
                    "context": "Traceability is the ability to track requirements.",
                }
            )

        tracker = get_global_cost_tracker()
        calls = [c for c in tracker.get_calls() if c.operation == "synthesis_draft"]
        assert len(calls) == 1
        assert calls[0].model == mock_config.chat_model
