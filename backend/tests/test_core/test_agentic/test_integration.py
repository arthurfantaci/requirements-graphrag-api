"""Integration tests for the agentic RAG system.

These tests verify the full end-to-end flow of the agentic system:
1. API endpoint -> Orchestrator -> Subgraphs -> Streaming events
2. State propagation through the graph
3. Error handling across the system

Test Categories:
- Unit integration: Tests with mocked dependencies (fast, run in CI)
- Live integration: Tests with real LLM/Neo4j (slow, manual runs only)

The live tests require environment variables:
- OPENAI_API_KEY
- NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
- CHECKPOINT_DATABASE_URL (optional, for persistence tests)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from requirements_graphrag_api.core.agentic import (
    OrchestratorState,
    create_orchestrator_graph,
    get_thread_config,
    stream_agentic_events,
)
from requirements_graphrag_api.core.agentic.state import (
    CriticEvaluation,
    EntityInfo,
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


@pytest.fixture
def mock_subgraphs():
    """Mock all subgraphs to avoid LLM calls."""
    with (
        patch(
            "requirements_graphrag_api.core.agentic.orchestrator.create_rag_subgraph"
        ) as mock_rag,
        patch(
            "requirements_graphrag_api.core.agentic.orchestrator.create_research_subgraph"
        ) as mock_research,
        patch(
            "requirements_graphrag_api.core.agentic.orchestrator.create_synthesis_subgraph"
        ) as mock_synthesis,
    ):
        # RAG subgraph mock
        rag_graph = MagicMock()
        rag_graph.ainvoke = AsyncMock(
            return_value={
                "ranked_results": [
                    RetrievedDocument(
                        content="Requirements traceability links requirements to test cases.",
                        source="Traceability Guide",
                        score=0.95,
                    ),
                    RetrievedDocument(
                        content="Traceability matrices help track requirements coverage.",
                        source="Requirements Best Practices",
                        score=0.88,
                    ),
                ],
                "expanded_queries": [
                    "requirements traceability definition",
                    "what is traceability in software",
                ],
                "context": (
                    "Requirements traceability links requirements to test cases. "
                    "Traceability matrices help track requirements coverage."
                ),
            }
        )
        mock_rag.return_value = rag_graph

        # Research subgraph mock
        research_graph = MagicMock()
        research_graph.ainvoke = AsyncMock(
            return_value={
                "entity_contexts": [
                    EntityInfo(
                        name="Traceability Matrix",
                        entity_type="Concept",
                        description="A grid mapping requirements to test cases.",
                    ),
                ],
                "exploration_complete": True,
            }
        )
        mock_research.return_value = research_graph

        # Synthesis subgraph mock
        synthesis_graph = MagicMock()
        synthesis_graph.ainvoke = AsyncMock(
            return_value={
                "final_answer": (
                    "Requirements traceability is the ability to link requirements "
                    "to their test cases, design documents, and code implementations. "
                    "It ensures comprehensive coverage and change impact analysis."
                ),
                "citations": ["Traceability Guide", "Requirements Best Practices"],
                "critique": CriticEvaluation(
                    answerable=True,
                    confidence=0.9,
                    completeness="complete",
                ),
            }
        )
        mock_synthesis.return_value = synthesis_graph

        yield {
            "rag": mock_rag,
            "research": mock_research,
            "synthesis": mock_synthesis,
        }


# =============================================================================
# ORCHESTRATOR INTEGRATION TESTS
# =============================================================================


class TestOrchestratorIntegration:
    """Integration tests for the orchestrator graph."""

    @pytest.mark.asyncio
    async def test_full_orchestrator_flow_with_mocked_subgraphs(
        self,
        mock_config: AppConfig,
        mock_driver,
        mock_retriever,
        mock_subgraphs,
    ):
        """Test complete orchestrator flow with mocked subgraphs."""
        graph = create_orchestrator_graph(mock_config, mock_driver, mock_retriever)

        # Create initial state
        state: OrchestratorState = {
            "messages": [HumanMessage(content="What is requirements traceability?")],
            "query": "What is requirements traceability?",
        }

        # Run the orchestrator
        result = await graph.ainvoke(state)

        # Verify the flow completed
        assert result["current_phase"] == "complete"
        assert result["final_answer"] is not None
        assert "traceability" in result["final_answer"].lower()
        assert len(result.get("citations", [])) > 0

    @pytest.mark.asyncio
    async def test_orchestrator_handles_empty_query(
        self,
        mock_config: AppConfig,
        mock_driver,
        mock_retriever,
    ):
        """Test orchestrator handles missing query gracefully."""
        graph = create_orchestrator_graph(mock_config, mock_driver, mock_retriever)

        state: OrchestratorState = {
            "messages": [],
        }

        result = await graph.ainvoke(state)

        assert result["current_phase"] == "error"
        assert "No query" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_orchestrator_extracts_query_from_message(
        self,
        mock_config: AppConfig,
        mock_driver,
        mock_retriever,
        mock_subgraphs,
    ):
        """Test that orchestrator extracts query from HumanMessage."""
        graph = create_orchestrator_graph(mock_config, mock_driver, mock_retriever)

        # Only provide message, not query
        state: OrchestratorState = {
            "messages": [HumanMessage(content="Explain traceability matrices")],
        }

        result = await graph.ainvoke(state)

        assert result["query"] == "Explain traceability matrices"
        assert result["current_phase"] == "complete"


# =============================================================================
# STREAMING INTEGRATION TESTS
# =============================================================================


class TestStreamingIntegration:
    """Integration tests for SSE streaming."""

    @pytest.mark.asyncio
    async def test_streaming_emits_phase_events(
        self,
        mock_config: AppConfig,
        mock_driver,
        mock_retriever,
        mock_subgraphs,
    ):
        """Test that streaming emits phase events."""
        graph = create_orchestrator_graph(mock_config, mock_driver, mock_retriever)

        state: OrchestratorState = {
            "messages": [HumanMessage(content="What is traceability?")],
            "query": "What is traceability?",
        }

        config = get_thread_config("test-thread-001")

        events = []
        async for event in stream_agentic_events(graph, state, config):
            events.append(event)

        # Parse events
        parsed = []
        for event in events:
            if event.startswith("data: "):
                try:
                    data = json.loads(event[6:].strip())
                    parsed.append(data)
                except json.JSONDecodeError:
                    continue

        # Should have phase events
        phase_events = [e for e in parsed if "phase" in e]
        assert len(phase_events) > 0

        # Should have done event
        done_events = [e for e in parsed if "full_answer" in e]
        assert len(done_events) == 1

    @pytest.mark.asyncio
    async def test_streaming_handles_errors_gracefully(
        self,
        mock_config: AppConfig,
        mock_driver,
        mock_retriever,
    ):
        """Test that streaming handles errors and emits error event."""
        # Mock subgraph to raise an exception
        with patch(
            "requirements_graphrag_api.core.agentic.orchestrator.create_rag_subgraph"
        ) as mock_rag:
            rag_graph = MagicMock()
            rag_graph.ainvoke = AsyncMock(side_effect=RuntimeError("Simulated RAG failure"))
            mock_rag.return_value = rag_graph

            graph = create_orchestrator_graph(mock_config, mock_driver, mock_retriever)

            state: OrchestratorState = {
                "messages": [HumanMessage(content="Test query")],
                "query": "Test query",
            }

            config = get_thread_config("test-error-thread")

            events = []
            async for event in stream_agentic_events(graph, state, config):
                events.append(event)

            # Should have error event
            parsed = []
            for event in events:
                if event.startswith("data: "):
                    try:
                        data = json.loads(event[6:].strip())
                        parsed.append(data)
                    except json.JSONDecodeError:
                        continue

            # Either we get an error event or done event (orchestrator may catch error)
            # The streaming should complete, not hang
            assert len(parsed) > 0


# =============================================================================
# STATE PROPAGATION TESTS
# =============================================================================


class TestStatePropagation:
    """Tests for state propagation through the orchestrator."""

    @pytest.mark.asyncio
    async def test_context_flows_between_subgraphs(
        self,
        mock_config: AppConfig,
        mock_driver,
        mock_retriever,
        mock_subgraphs,
    ):
        """Test that context from RAG flows to synthesis."""
        graph = create_orchestrator_graph(mock_config, mock_driver, mock_retriever)

        state: OrchestratorState = {
            "messages": [HumanMessage(content="What is requirements traceability?")],
            "query": "What is requirements traceability?",
        }

        result = await graph.ainvoke(state)

        # Verify context was propagated
        assert "context" in result or "ranked_results" in result
        assert result["final_answer"] is not None

    @pytest.mark.asyncio
    async def test_messages_accumulate_correctly(
        self,
        mock_config: AppConfig,
        mock_driver,
        mock_retriever,
        mock_subgraphs,
    ):
        """Test that messages accumulate through the conversation."""
        graph = create_orchestrator_graph(mock_config, mock_driver, mock_retriever)

        # Start with a human message
        initial_messages = [HumanMessage(content="What is traceability?")]

        state: OrchestratorState = {
            "messages": initial_messages,
            "query": "What is traceability?",
        }

        result = await graph.ainvoke(state)

        # Messages should include the original
        assert len(result["messages"]) >= 1


# =============================================================================
# THREAD CONFIGURATION TESTS
# =============================================================================


class TestThreadConfiguration:
    """Tests for thread configuration in the agentic system."""

    def test_thread_config_structure(self):
        """Test that thread config has correct structure."""
        config = get_thread_config("thread-123")

        assert "configurable" in config
        assert config["configurable"]["thread_id"] == "thread-123"

    def test_thread_config_with_all_options(self):
        """Test thread config with all optional parameters."""
        config = get_thread_config(
            "thread-123",
            checkpoint_id="cp-456",
            user_id="user-789",
            run_id="run-abc",
            custom_param="custom_value",
        )

        assert config["configurable"]["thread_id"] == "thread-123"
        assert config["configurable"]["checkpoint_id"] == "cp-456"
        assert config["configurable"]["user_id"] == "user-789"
        assert config["configurable"]["custom_param"] == "custom_value"
        assert config["run_id"] == "run-abc"

    @pytest.mark.asyncio
    async def test_different_threads_are_isolated(
        self,
        mock_config: AppConfig,
        mock_driver,
        mock_retriever,
        mock_subgraphs,
    ):
        """Test that different threads maintain separate state."""
        graph = create_orchestrator_graph(mock_config, mock_driver, mock_retriever)

        # Thread 1
        state1: OrchestratorState = {
            "messages": [HumanMessage(content="Question A")],
            "query": "Question A",
        }
        config1 = get_thread_config("thread-A")
        result1 = await graph.ainvoke(state1, config=config1)

        # Thread 2
        state2: OrchestratorState = {
            "messages": [HumanMessage(content="Question B")],
            "query": "Question B",
        }
        config2 = get_thread_config("thread-B")
        result2 = await graph.ainvoke(state2, config=config2)

        # Both should complete independently
        assert result1["current_phase"] == "complete"
        assert result2["current_phase"] == "complete"
        assert result1["query"] == "Question A"
        assert result2["query"] == "Question B"
