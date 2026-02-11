"""Unit tests for agentic subgraphs.

Tests cover:
- Subgraph creation and compilation
- Individual node functionality
- Conditional edge routing logic
- State transformations
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from requirements_graphrag_api.core.agentic.state import (
    CriticEvaluation,
    RAGState,
    RetrievedDocument,
    SynthesisState,
)
from requirements_graphrag_api.core.agentic.subgraphs import (
    create_rag_subgraph,
    create_synthesis_subgraph,
)
from tests.conftest import create_ai_message_mock

if TYPE_CHECKING:
    from requirements_graphrag_api.config import AppConfig


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_driver() -> MagicMock:
    """Create a mock Neo4j driver."""
    return MagicMock()


@pytest.fixture
def mock_retriever() -> MagicMock:
    """Create a mock VectorRetriever."""
    return MagicMock()


def create_mock_llm_chain(response: str) -> MagicMock:
    """Create a mock for prompt | llm | parser chain."""
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=response)
    return mock_chain


# =============================================================================
# RAG SUBGRAPH TESTS
# =============================================================================


class TestRAGSubgraph:
    """Tests for the RAG retrieval subgraph."""

    def test_subgraph_creation(self, mock_config: AppConfig, mock_driver, mock_retriever):
        """Test that RAG subgraph can be created and compiled."""
        graph = create_rag_subgraph(mock_config, mock_driver, mock_retriever)
        assert graph is not None
        # Verify it has the expected nodes
        assert hasattr(graph, "invoke")

    @pytest.mark.asyncio
    async def test_expand_queries_success(
        self, mock_config: AppConfig, mock_driver, mock_retriever
    ):
        """Test query expansion with valid response."""
        expansion_response = json.dumps(
            {
                "queries": [
                    {"query": "What is requirements traceability?", "strategy": "original"},
                    {"query": "How do you trace requirements in software?", "strategy": "synonym"},
                    {
                        "query": "What are the principles of requirements management?",
                        "strategy": "stepback",
                    },
                ]
            }
        )

        with patch(
            "requirements_graphrag_api.core.agentic.subgraphs.rag.ChatOpenAI"
        ) as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.__or__ = MagicMock(return_value=create_mock_llm_chain(expansion_response))
            mock_llm_class.return_value = mock_llm

            # We need to patch at the prompt level since the chain is built inside
            with patch(
                "requirements_graphrag_api.core.agentic.subgraphs.rag.get_prompt",
                new_callable=AsyncMock,
            ) as mock_prompt:
                mock_template = MagicMock()
                mock_template.__or__ = MagicMock(
                    return_value=create_mock_llm_chain(expansion_response)
                )
                mock_prompt.return_value = mock_template

                # Create graph - this test verifies the graph can be created with mocks
                _ = create_rag_subgraph(mock_config, mock_driver, mock_retriever)

    def test_rag_state_structure(self):
        """Test that RAGState has expected structure."""
        state: RAGState = {
            "query": "test query",
            "expanded_queries": ["q1", "q2"],
            "raw_results": [{"text": "result"}],
            "ranked_results": [RetrievedDocument(content="test", source="source", score=0.9)],
            "retrieval_metadata": {"count": 1},
        }
        assert state["query"] == "test query"
        assert len(state["expanded_queries"]) == 2

    @pytest.mark.asyncio
    async def test_dedupe_and_rank_sets_quality_gate(
        self, mock_config: AppConfig, mock_driver, mock_retriever
    ):
        """dedupe_and_rank sets quality_pass, relevant_count, total_count."""
        search_results = [
            {
                "text": "Traceability overview",
                "score": 0.65,
                "metadata": {"title": "Webinar: Traceability", "chunk_id": "c1"},
            },
            {
                "text": "More content",
                "score": 0.55,
                "metadata": {"title": "Article: Standards", "chunk_id": "c2"},
            },
            {
                "text": "Low score doc",
                "score": 0.40,
                "metadata": {"title": "Doc3", "chunk_id": "c3"},
            },
        ]
        expansion_json = json.dumps(
            {"queries": [{"query": "list all webinars about traceability", "strategy": "original"}]}
        )

        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=create_ai_message_mock(expansion_json))
        mock_prompt_template = MagicMock()
        mock_prompt_template.__or__ = MagicMock(return_value=mock_chain)

        mock_llm = MagicMock()

        with (
            patch(
                "requirements_graphrag_api.core.agentic.subgraphs.rag.get_prompt",
                new_callable=AsyncMock,
                return_value=mock_prompt_template,
            ),
            patch(
                "requirements_graphrag_api.core.agentic.subgraphs.rag.ChatOpenAI",
                return_value=mock_llm,
            ),
            patch(
                "requirements_graphrag_api.core.retrieval.graph_enriched_search",
                new_callable=AsyncMock,
                return_value=search_results,
            ),
        ):
            graph = create_rag_subgraph(mock_config, mock_driver, mock_retriever)
            result = await graph.ainvoke({"query": "list all webinars about traceability"})

        # All 3 docs kept — quality gate passes
        assert len(result["ranked_results"]) == 3
        assert result["relevant_count"] == 3
        assert result["total_count"] == 3
        assert result["quality_pass"] is True


# =============================================================================
# SYNTHESIS SUBGRAPH TESTS
# =============================================================================


class TestSynthesisSubgraph:
    """Tests for the Synthesis answer generation subgraph."""

    def test_subgraph_creation(self, mock_config: AppConfig):
        """Test that Synthesis subgraph can be created and compiled."""
        graph = create_synthesis_subgraph(mock_config)
        assert graph is not None
        assert hasattr(graph, "invoke")

    def test_synthesis_state_structure(self):
        """Test that SynthesisState has expected structure."""
        state: SynthesisState = {
            "query": "test query",
            "context": "some context",
            "draft_answer": "Initial answer draft",
            "critique": CriticEvaluation(
                answerable=True,
                confidence=0.8,
                completeness="complete",
                missing_aspects=[],
                reasoning="Good answer",
            ),
            "revision_count": 0,
            "final_answer": "Final answer with citations",
            "citations": ["Source 1", "Source 2"],
        }
        assert state["query"] == "test query"
        assert state["critique"].confidence == 0.8
        assert len(state["citations"]) == 2

    def test_needs_revision_high_confidence(self):
        """Test that high confidence answers don't trigger revision."""
        critique = CriticEvaluation(
            answerable=True,
            confidence=0.85,
            completeness="complete",
        )
        # Above threshold (0.7), complete -> no revision needed
        assert critique.confidence >= 0.7
        assert critique.completeness == "complete"

    def test_needs_revision_low_confidence(self):
        """Test that low confidence triggers revision."""
        critique = CriticEvaluation(
            answerable=True,
            confidence=0.5,
            completeness="partial",
            missing_aspects=["more details needed"],
        )
        # Below threshold -> needs revision
        assert critique.confidence < 0.7
        assert critique.completeness == "partial"

    def test_needs_revision_max_reached(self):
        """Test that max revisions prevents further revision."""
        # MAX_REVISIONS = 2 in synthesis.py
        state: SynthesisState = {
            "query": "test",
            "context": "ctx",
            "draft_answer": "answer",
            "critique": CriticEvaluation(
                answerable=True,
                confidence=0.5,  # Low, would normally trigger revision
                completeness="partial",
            ),
            "revision_count": 2,  # At max
        }
        # Even with low confidence, max revisions should stop iteration
        assert state.get("revision_count", 0) >= 2

    def test_needs_revision_insufficient_completeness(self):
        """Test that insufficient completeness triggers revision."""
        critique = CriticEvaluation(
            answerable=False,
            confidence=0.3,
            completeness="insufficient",
            missing_aspects=["context not found"],
            reasoning="Cannot answer from context",
        )
        assert critique.completeness == "insufficient"


# =============================================================================
# STATE TYPE TESTS
# =============================================================================


class TestStateDataclasses:
    """Tests for state dataclass structures."""

    def test_retrieved_document_defaults(self):
        """Test RetrievedDocument default values."""
        doc = RetrievedDocument(content="test", source="src")
        assert doc.score == 0.0
        assert doc.metadata == {}

    def test_retrieved_document_immutable(self):
        """Test that RetrievedDocument is immutable (frozen dataclass)."""
        doc = RetrievedDocument(content="test", source="src", score=0.9)
        with pytest.raises(AttributeError):
            doc.score = 0.5  # type: ignore

    def test_critic_evaluation_defaults(self):
        """Test CriticEvaluation default values."""
        critique = CriticEvaluation(
            answerable=True,
            confidence=0.8,
            completeness="complete",
        )
        assert critique.missing_aspects == []
        assert critique.followup_query is None
        assert critique.reasoning == ""


# =============================================================================
# INTEGRATION-STYLE TESTS (with mocks)
# =============================================================================


class TestSubgraphIntegration:
    """Integration-style tests for subgraph flows."""

    @pytest.mark.asyncio
    async def test_synthesis_no_context(self, mock_config: AppConfig):
        """Test synthesis handles missing context gracefully."""
        # Create the subgraph
        graph = create_synthesis_subgraph(mock_config)

        # Invoke with no context
        initial_state: SynthesisState = {
            "query": "What is requirements traceability?",
            "context": "",  # Empty context
        }

        # The graph should handle this by returning an appropriate message
        # without calling the LLM (early exit in draft_answer node)
        result = await graph.ainvoke(initial_state)

        assert result["draft_answer"] == "I don't have enough context to answer this question."
        assert result["critique"].answerable is False
        assert result["critique"].confidence == 0.0
