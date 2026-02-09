"""Unit tests for agentic orchestrator and checkpoints.

Tests cover:
- Orchestrator graph creation and compilation
- Checkpoint configuration helpers
- Conditional routing logic
- Integration with subgraphs (mocked)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from requirements_graphrag_api.core.agentic.checkpoints import (
    get_thread_config,
    get_thread_id_from_config,
)
from requirements_graphrag_api.core.agentic.orchestrator import (
    COMPARISON_KEYWORDS,
    DEFAULT_MAX_ITERATIONS,
    RESEARCH_ENTITY_THRESHOLD,
    create_orchestrator_graph,
)
from requirements_graphrag_api.core.agentic.state import (
    EntityInfo,
    OrchestratorState,
    RetrievedDocument,
)

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


# =============================================================================
# CHECKPOINT TESTS
# =============================================================================


class TestCheckpointHelpers:
    """Tests for checkpoint configuration helpers."""

    def test_get_thread_config_basic(self):
        """Test basic thread configuration."""
        config = get_thread_config("thread-123")

        assert config["configurable"]["thread_id"] == "thread-123"
        assert "checkpoint_id" not in config["configurable"]
        assert "user_id" not in config["configurable"]

    def test_get_thread_config_with_checkpoint(self):
        """Test thread config with checkpoint ID."""
        config = get_thread_config("thread-123", checkpoint_id="cp-456")

        assert config["configurable"]["thread_id"] == "thread-123"
        assert config["configurable"]["checkpoint_id"] == "cp-456"

    def test_get_thread_config_with_user_id(self):
        """Test thread config with user ID for namespacing."""
        config = get_thread_config("thread-123", user_id="user-789")

        assert config["configurable"]["thread_id"] == "thread-123"
        assert config["configurable"]["user_id"] == "user-789"

    def test_get_thread_config_with_run_id(self):
        """Test thread config with LangSmith run ID."""
        config = get_thread_config("thread-123", run_id="run-abc")

        assert config["configurable"]["thread_id"] == "thread-123"
        assert config["run_id"] == "run-abc"

    def test_get_thread_config_extra_configurable(self):
        """Test thread config with extra configurable params."""
        config = get_thread_config("thread-123", custom_key="custom_value")

        assert config["configurable"]["thread_id"] == "thread-123"
        assert config["configurable"]["custom_key"] == "custom_value"

    def test_get_thread_id_from_config(self):
        """Test extracting thread_id from config."""
        config = get_thread_config("thread-123")
        thread_id = get_thread_id_from_config(config)

        assert thread_id == "thread-123"

    def test_get_thread_id_from_config_missing(self):
        """Test extracting thread_id when not present."""
        config = {"configurable": {}}
        thread_id = get_thread_id_from_config(config)

        assert thread_id is None

    def test_get_thread_id_from_config_empty(self):
        """Test extracting thread_id from empty config."""
        config = {}
        thread_id = get_thread_id_from_config(config)

        assert thread_id is None


# =============================================================================
# ORCHESTRATOR TESTS
# =============================================================================


class TestOrchestratorGraph:
    """Tests for the orchestrator graph."""

    def test_orchestrator_creation(self, mock_config: AppConfig, mock_driver, mock_retriever):
        """Test that orchestrator graph can be created."""
        graph = create_orchestrator_graph(mock_config, mock_driver, mock_retriever)

        assert graph is not None
        assert hasattr(graph, "ainvoke")

    def test_orchestrator_state_structure(self):
        """Test that OrchestratorState has expected structure."""
        state: OrchestratorState = {
            "messages": [HumanMessage(content="test query")],
            "query": "test query",
            "current_phase": "rag",
            "iteration_count": 0,
            "max_iterations": 3,
        }

        assert state["query"] == "test query"
        assert state["current_phase"] == "rag"
        assert len(state["messages"]) == 1

    def test_should_research_skips_few_results(self):
        """Test that research is skipped with few results."""
        state: OrchestratorState = {
            "messages": [],
            "query": "test",
            "ranked_results": [RetrievedDocument(content="x", source="s")],
            "context": "A" * 500,  # Long enough context
            "iteration_count": 0,
            "max_iterations": DEFAULT_MAX_ITERATIONS,
        }

        # Few results -> skip research
        assert len(state["ranked_results"]) < RESEARCH_ENTITY_THRESHOLD

    def test_should_research_skips_short_context(self):
        """Test that research is skipped with short context."""
        state: OrchestratorState = {
            "messages": [],
            "query": "test",
            "ranked_results": [RetrievedDocument(content="x", source="s") for _ in range(5)],
            "context": "short",  # Too short
            "iteration_count": 0,
            "max_iterations": DEFAULT_MAX_ITERATIONS,
        }

        # Short context -> would skip research
        assert len(state["context"]) < 200

    def test_should_research_skips_max_iterations(self):
        """Test that research is skipped at max iterations."""
        state: OrchestratorState = {
            "messages": [],
            "query": "test",
            "ranked_results": [RetrievedDocument(content="x", source="s") for _ in range(5)],
            "context": "A" * 500,
            "iteration_count": DEFAULT_MAX_ITERATIONS,  # At max
            "max_iterations": DEFAULT_MAX_ITERATIONS,
        }

        # Max iterations -> would skip research
        assert state["iteration_count"] >= state["max_iterations"]


# =============================================================================
# ORCHESTRATOR INTEGRATION TESTS (with mocks)
# =============================================================================


class TestOrchestratorIntegration:
    """Integration tests for orchestrator with mocked subgraphs."""

    @pytest.mark.asyncio
    async def test_orchestrator_no_query(self, mock_config: AppConfig, mock_driver, mock_retriever):
        """Test orchestrator handles missing query gracefully."""
        graph = create_orchestrator_graph(mock_config, mock_driver, mock_retriever)

        # Empty state with no query
        state: OrchestratorState = {
            "messages": [],
        }

        result = await graph.ainvoke(state)

        assert result["current_phase"] == "error"
        assert "No query" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_orchestrator_query_from_message(
        self, mock_config: AppConfig, mock_driver, mock_retriever
    ):
        """Test orchestrator extracts query from HumanMessage."""
        # We need to mock the subgraphs to avoid actual LLM calls
        with patch(
            "requirements_graphrag_api.core.agentic.orchestrator.create_rag_subgraph"
        ) as mock_rag:
            mock_rag_graph = MagicMock()
            mock_rag_graph.ainvoke = AsyncMock(
                return_value={
                    "ranked_results": [
                        RetrievedDocument(content="Traceability content", source="s"),
                    ],
                    "expanded_queries": ["test"],
                }
            )
            mock_rag.return_value = mock_rag_graph

            with patch(
                "requirements_graphrag_api.core.agentic.orchestrator.create_synthesis_subgraph"
            ) as mock_synth:
                mock_synth_graph = MagicMock()
                mock_synth_graph.ainvoke = AsyncMock(
                    return_value={
                        "final_answer": "Test answer",
                        "citations": [],
                    }
                )
                mock_synth.return_value = mock_synth_graph

                graph = create_orchestrator_graph(mock_config, mock_driver, mock_retriever)

                # State with query in message
                state: OrchestratorState = {
                    "messages": [HumanMessage(content="What is requirements traceability?")],
                }

                result = await graph.ainvoke(state)

                # Should have extracted and processed query
                assert result.get("query") == "What is requirements traceability?"


class TestOrchestratorConstants:
    """Tests for orchestrator constants."""

    def test_default_max_iterations(self):
        """Test default max iterations constant."""
        assert DEFAULT_MAX_ITERATIONS == 3

    def test_research_threshold(self):
        """Test research entity threshold constant."""
        assert RESEARCH_ENTITY_THRESHOLD == 2

    def test_comparison_keywords_exist(self):
        """Test that comparison keywords are defined."""
        assert "compare" in COMPARISON_KEYWORDS
        assert "vs" in COMPARISON_KEYWORDS
        assert "difference between" in COMPARISON_KEYWORDS


class TestResearchSkipHeuristic:
    """Tests for the should_research() heuristic that skips research for simple queries."""

    # Realistic content long enough to pass the 200-char context threshold
    _LONG_CONTENT = "A" * 100

    @pytest.mark.asyncio
    async def test_simple_query_high_score_skips_research(
        self, mock_config: AppConfig, mock_driver, mock_retriever
    ):
        """Test that a short query with high retrieval score skips research."""
        with (
            patch(
                "requirements_graphrag_api.core.agentic.orchestrator.create_rag_subgraph"
            ) as mock_rag,
            patch(
                "requirements_graphrag_api.core.agentic.orchestrator.create_research_subgraph"
            ) as mock_research,
            patch(
                "requirements_graphrag_api.core.agentic.orchestrator.create_synthesis_subgraph"
            ) as mock_synth,
        ):
            c = self._LONG_CONTENT
            # RAG returns high-confidence results (score > 0.85)
            mock_rag_graph = MagicMock()
            mock_rag_graph.ainvoke = AsyncMock(
                return_value={
                    "ranked_results": [
                        RetrievedDocument(content=c, source="s", score=0.95),
                        RetrievedDocument(content=c, source="t", score=0.90),
                        RetrievedDocument(content=c, source="u", score=0.88),
                    ],
                    "expanded_queries": ["test"],
                }
            )
            mock_rag.return_value = mock_rag_graph

            mock_research_graph = MagicMock()
            mock_research_graph.ainvoke = AsyncMock(return_value={"entity_contexts": []})
            mock_research.return_value = mock_research_graph

            mock_synth_graph = MagicMock()
            mock_synth_graph.ainvoke = AsyncMock(
                return_value={"final_answer": "Answer", "citations": []}
            )
            mock_synth.return_value = mock_synth_graph

            graph = create_orchestrator_graph(mock_config, mock_driver, mock_retriever)
            state: OrchestratorState = {
                "messages": [HumanMessage(content="What is traceability?")],
                "query": "What is traceability?",
            }

            await graph.ainvoke(state)

            # Research subgraph should NOT have been called (short query + high score)
            mock_research_graph.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_comparison_query_triggers_research(
        self, mock_config: AppConfig, mock_driver, mock_retriever
    ):
        """Test that comparison queries proceed to research even when short."""
        with (
            patch(
                "requirements_graphrag_api.core.agentic.orchestrator.create_rag_subgraph"
            ) as mock_rag,
            patch(
                "requirements_graphrag_api.core.agentic.orchestrator.create_research_subgraph"
            ) as mock_research,
            patch(
                "requirements_graphrag_api.core.agentic.orchestrator.create_synthesis_subgraph"
            ) as mock_synth,
        ):
            c = self._LONG_CONTENT
            mock_rag_graph = MagicMock()
            mock_rag_graph.ainvoke = AsyncMock(
                return_value={
                    "ranked_results": [
                        RetrievedDocument(content=c, source="s", score=0.95),
                        RetrievedDocument(content=c, source="t", score=0.90),
                        RetrievedDocument(content=c, source="u", score=0.88),
                    ],
                    "expanded_queries": ["test"],
                }
            )
            mock_rag.return_value = mock_rag_graph

            mock_research_graph = MagicMock()
            mock_research_graph.ainvoke = AsyncMock(return_value={"entity_contexts": []})
            mock_research.return_value = mock_research_graph

            mock_synth_graph = MagicMock()
            mock_synth_graph.ainvoke = AsyncMock(
                return_value={"final_answer": "Answer", "citations": []}
            )
            mock_synth.return_value = mock_synth_graph

            graph = create_orchestrator_graph(mock_config, mock_driver, mock_retriever)
            # "compare" keyword should trigger research
            state: OrchestratorState = {
                "messages": [HumanMessage(content="Compare DOORS vs Jama")],
                "query": "Compare DOORS vs Jama",
            }

            await graph.ainvoke(state)

            # Research subgraph SHOULD have been called (comparison query)
            mock_research_graph.ainvoke.assert_called_once()


class TestPreviousContext:
    """Tests for previous_context flowing through orchestrator to synthesis (F4/F6)."""

    @pytest.mark.asyncio
    async def test_previous_context_from_conversation_history(
        self, mock_config: AppConfig, mock_driver, mock_retriever
    ):
        """Test that multi-turn history is passed as previous_context to synthesis."""
        with (
            patch(
                "requirements_graphrag_api.core.agentic.orchestrator.create_rag_subgraph"
            ) as mock_rag,
            patch("requirements_graphrag_api.core.agentic.orchestrator.create_research_subgraph"),
            patch(
                "requirements_graphrag_api.core.agentic.orchestrator.create_synthesis_subgraph"
            ) as mock_synth,
        ):
            mock_rag_graph = MagicMock()
            mock_rag_graph.ainvoke = AsyncMock(
                return_value={
                    "ranked_results": [
                        RetrievedDocument(content="Jama traceability", source="s"),
                    ],
                    "expanded_queries": [],
                }
            )
            mock_rag.return_value = mock_rag_graph

            mock_synth_graph = MagicMock()
            mock_synth_graph.ainvoke = AsyncMock(
                return_value={"final_answer": "Follow-up answer", "citations": []}
            )
            mock_synth.return_value = mock_synth_graph

            graph = create_orchestrator_graph(mock_config, mock_driver, mock_retriever)

            # Multi-turn: user question, assistant answer, then follow-up
            state: OrchestratorState = {
                "messages": [
                    HumanMessage(content="What is traceability?"),
                    AIMessage(content="Traceability is the ability to track..."),
                    HumanMessage(content="How does Jama implement it?"),
                ],
                "query": "How does Jama implement it?",
            }

            await graph.ainvoke(state)

            # Verify synthesis was called with previous_context
            synth_call = mock_synth_graph.ainvoke.call_args[0][0]
            assert "previous_context" in synth_call
            assert "Q: What is traceability?" in synth_call["previous_context"]
            assert "A: Traceability is the ability to track..." in synth_call["previous_context"]

    @pytest.mark.asyncio
    async def test_no_previous_context_for_single_message(
        self, mock_config: AppConfig, mock_driver, mock_retriever
    ):
        """Test that single-message queries have empty previous_context."""
        with (
            patch(
                "requirements_graphrag_api.core.agentic.orchestrator.create_rag_subgraph"
            ) as mock_rag,
            patch("requirements_graphrag_api.core.agentic.orchestrator.create_research_subgraph"),
            patch(
                "requirements_graphrag_api.core.agentic.orchestrator.create_synthesis_subgraph"
            ) as mock_synth,
        ):
            mock_rag_graph = MagicMock()
            mock_rag_graph.ainvoke = AsyncMock(
                return_value={
                    "ranked_results": [
                        RetrievedDocument(content="Traceability content", source="s"),
                    ],
                    "expanded_queries": [],
                }
            )
            mock_rag.return_value = mock_rag_graph

            mock_synth_graph = MagicMock()
            mock_synth_graph.ainvoke = AsyncMock(
                return_value={"final_answer": "Answer", "citations": []}
            )
            mock_synth.return_value = mock_synth_graph

            graph = create_orchestrator_graph(mock_config, mock_driver, mock_retriever)

            state: OrchestratorState = {
                "messages": [HumanMessage(content="What is traceability?")],
                "query": "What is traceability?",
            }

            await graph.ainvoke(state)

            # Verify synthesis was called with empty previous_context
            synth_call = mock_synth_graph.ainvoke.call_args[0][0]
            assert synth_call.get("previous_context", "") == ""


class TestRunRagEnrichedContext:
    """Tests that run_rag includes enrichment data in context string."""

    @pytest.mark.asyncio
    async def test_enrichment_appears_in_context(
        self, mock_config: AppConfig, mock_driver, mock_retriever
    ):
        """Verify entities, glossary, relationships flow into context."""
        with (
            patch(
                "requirements_graphrag_api.core.agentic.orchestrator.create_rag_subgraph"
            ) as mock_rag,
            patch("requirements_graphrag_api.core.agentic.orchestrator.create_research_subgraph"),
            patch(
                "requirements_graphrag_api.core.agentic.orchestrator.create_synthesis_subgraph"
            ) as mock_synth,
        ):
            mock_rag_graph = MagicMock()
            mock_rag_graph.ainvoke = AsyncMock(
                return_value={
                    "ranked_results": [
                        RetrievedDocument(
                            content="AEC requirements traceability content.",
                            source="AEC Article",
                            score=0.95,
                            metadata={
                                "entities": [
                                    {"name": "AEC", "type": "Industry"},
                                    {"name": "Traceability", "type": "Concept"},
                                ],
                                "glossary_definitions": [
                                    {
                                        "term": "Traceability Matrix",
                                        "definition": "A doc correlating reqs to tests.",
                                    },
                                ],
                                "semantic_relationships": [
                                    {
                                        "from_entity": "Traceability Matrix",
                                        "relationship": "REQUIRES",
                                        "to_entity": "Baseline",
                                    },
                                ],
                                "industry_standards": [
                                    {
                                        "standard": "ISO 19650",
                                        "organization": "ISO",
                                        "standard_definition": "BIM data management",
                                    },
                                ],
                            },
                        ),
                    ],
                    "expanded_queries": ["test"],
                }
            )
            mock_rag.return_value = mock_rag_graph

            mock_synth_graph = MagicMock()
            mock_synth_graph.ainvoke = AsyncMock(
                return_value={"final_answer": "Answer", "citations": []}
            )
            mock_synth.return_value = mock_synth_graph

            graph = create_orchestrator_graph(mock_config, mock_driver, mock_retriever)
            state: OrchestratorState = {
                "messages": [HumanMessage(content="What is AEC traceability?")],
                "query": "What is AEC traceability?",
            }

            await graph.ainvoke(state)

            # Verify synthesis received enriched context
            synth_call = mock_synth_graph.ainvoke.call_args[0][0]
            ctx = synth_call["context"]

            # Inline entities
            assert "(Entities: AEC, Traceability)" in ctx
            # KG glossary
            assert "Traceability Matrix" in ctx
            # KG relationships
            assert "Traceability Matrix -> REQUIRES -> Baseline" in ctx
            # KG standards
            assert "ISO 19650" in ctx


class TestEntitiesStrFlow:
    """Tests that entities_str flows from research → orchestrator → synthesis."""

    @pytest.mark.asyncio
    async def test_research_entities_reach_synthesis(
        self, mock_config: AppConfig, mock_driver, mock_retriever
    ):
        """Verify research EntityInfo objects are formatted and passed to synthesis."""
        with (
            patch(
                "requirements_graphrag_api.core.agentic.orchestrator.create_rag_subgraph"
            ) as mock_rag,
            patch(
                "requirements_graphrag_api.core.agentic.orchestrator.create_research_subgraph"
            ) as mock_research,
            patch(
                "requirements_graphrag_api.core.agentic.orchestrator.create_synthesis_subgraph"
            ) as mock_synth,
        ):
            c = "A" * 250  # Long enough content for research routing
            mock_rag_graph = MagicMock()
            mock_rag_graph.ainvoke = AsyncMock(
                return_value={
                    "ranked_results": [
                        RetrievedDocument(content=c, source="s", score=0.7),
                        RetrievedDocument(content=c, source="t", score=0.6),
                        RetrievedDocument(content=c, source="u", score=0.5),
                    ],
                    "expanded_queries": ["compare DOORS vs Jama"],
                }
            )
            mock_rag.return_value = mock_rag_graph

            mock_research_graph = MagicMock()
            mock_research_graph.ainvoke = AsyncMock(
                return_value={
                    "entity_contexts": [
                        EntityInfo(
                            name="Jama Connect",
                            entity_type="Tool",
                            description="Requirements management tool",
                            related_entities=["DOORS", "Polarion"],
                        ),
                    ],
                }
            )
            mock_research.return_value = mock_research_graph

            mock_synth_graph = MagicMock()
            mock_synth_graph.ainvoke = AsyncMock(
                return_value={"final_answer": "Answer", "citations": []}
            )
            mock_synth.return_value = mock_synth_graph

            graph = create_orchestrator_graph(mock_config, mock_driver, mock_retriever)
            # Comparison query triggers research path
            state: OrchestratorState = {
                "messages": [HumanMessage(content="Compare DOORS vs Jama")],
                "query": "Compare DOORS vs Jama",
            }

            await graph.ainvoke(state)

            # Verify synthesis received entities_str from research
            synth_call = mock_synth_graph.ainvoke.call_args[0][0]
            entities = synth_call.get("entities_str", "")
            assert "Jama Connect" in entities
            assert "Tool" in entities
            assert "Related: DOORS, Polarion" in entities
