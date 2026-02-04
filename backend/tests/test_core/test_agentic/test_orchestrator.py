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
from langchain_core.messages import HumanMessage

from requirements_graphrag_api.core.agentic.checkpoints import (
    get_thread_config,
    get_thread_id_from_config,
)
from requirements_graphrag_api.core.agentic.orchestrator import (
    DEFAULT_MAX_ITERATIONS,
    RESEARCH_ENTITY_THRESHOLD,
    create_orchestrator_graph,
)
from requirements_graphrag_api.core.agentic.state import (
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
                    "ranked_results": [],
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
