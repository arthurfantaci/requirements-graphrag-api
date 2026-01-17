"""Tests for evaluation runner module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jama_mcp_server_graphrag.config import AppConfig
from jama_mcp_server_graphrag.evaluation.datasets import EvaluationSample
from jama_mcp_server_graphrag.evaluation.metrics import RAGMetrics
from jama_mcp_server_graphrag.evaluation.runner import (
    EvaluationReport,
    EvaluationResult,
    RAGEvaluator,
    evaluate_rag_pipeline,
)

_TEST_PASSWORD = "test"  # noqa: S105


@pytest.fixture
def mock_config() -> AppConfig:
    """Create a mock config for testing."""
    return AppConfig(
        neo4j_uri="neo4j://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password=_TEST_PASSWORD,
        openai_api_key="sk-test",
        chat_model="gpt-4o",
    )


@pytest.fixture
def mock_graph() -> MagicMock:
    """Create a mock Neo4j graph."""
    return MagicMock()


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Create a mock vector store."""
    return MagicMock()


@pytest.fixture
def sample_evaluation_sample() -> EvaluationSample:
    """Create a sample evaluation sample."""
    return EvaluationSample(
        question="What is requirements traceability?",
        ground_truth="Traceability links requirements to artifacts.",
        contexts=["Context about traceability"],
        metadata={"topic": "traceability", "difficulty": "basic"},
    )


@pytest.fixture
def sample_metrics() -> RAGMetrics:
    """Create sample metrics."""
    return RAGMetrics(
        faithfulness=0.9,
        answer_relevancy=0.85,
        context_precision=0.8,
        context_recall=0.75,
    )


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_result_creation(
        self,
        sample_evaluation_sample: EvaluationSample,
        sample_metrics: RAGMetrics,
    ) -> None:
        """Test creating an evaluation result."""
        result = EvaluationResult(
            sample=sample_evaluation_sample,
            answer="Generated answer",
            contexts=["Context 1"],
            metrics=sample_metrics,
            latency_ms=150.0,
        )

        assert result.answer == "Generated answer"
        assert result.latency_ms == 150.0

    def test_to_dict(
        self,
        sample_evaluation_sample: EvaluationSample,
        sample_metrics: RAGMetrics,
    ) -> None:
        """Test converting result to dictionary."""
        result = EvaluationResult(
            sample=sample_evaluation_sample,
            answer="Generated answer",
            contexts=["Context 1"],
            metrics=sample_metrics,
            latency_ms=150.0,
        )

        result_dict = result.to_dict()

        assert result_dict["question"] == sample_evaluation_sample.question
        assert result_dict["answer"] == "Generated answer"
        assert "metrics" in result_dict
        assert result_dict["latency_ms"] == 150.0


class TestEvaluationReport:
    """Tests for EvaluationReport dataclass."""

    def test_report_creation(
        self,
        sample_evaluation_sample: EvaluationSample,
        sample_metrics: RAGMetrics,
    ) -> None:
        """Test creating an evaluation report."""
        result = EvaluationResult(
            sample=sample_evaluation_sample,
            answer="Answer",
            contexts=[],
            metrics=sample_metrics,
            latency_ms=100.0,
        )

        report = EvaluationReport(
            results=[result],
            aggregate_metrics={"average": 0.85},
            total_samples=1,
            timestamp="2024-01-01T00:00:00",
        )

        assert report.total_samples == 1
        assert len(report.results) == 1

    def test_to_dict(
        self,
        sample_evaluation_sample: EvaluationSample,
        sample_metrics: RAGMetrics,
    ) -> None:
        """Test converting report to dictionary."""
        result = EvaluationResult(
            sample=sample_evaluation_sample,
            answer="Answer",
            contexts=[],
            metrics=sample_metrics,
            latency_ms=100.0,
        )

        report = EvaluationReport(
            results=[result],
            aggregate_metrics={"average": 0.85},
            total_samples=1,
            timestamp="2024-01-01T00:00:00",
        )

        report_dict = report.to_dict()

        assert report_dict["total_samples"] == 1
        assert "aggregate_metrics" in report_dict
        assert "results" in report_dict


class TestRAGEvaluator:
    """Tests for RAGEvaluator class."""

    def test_evaluator_creation(
        self,
        mock_config: AppConfig,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test creating an evaluator."""
        evaluator = RAGEvaluator(mock_config, mock_graph, mock_vector_store)

        assert evaluator.config == mock_config
        assert evaluator.graph == mock_graph
        assert evaluator.vector_store == mock_vector_store

    @pytest.mark.asyncio
    async def test_evaluate_sample(
        self,
        mock_config: AppConfig,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
        sample_evaluation_sample: EvaluationSample,
    ) -> None:
        """Test evaluating a single sample."""
        evaluator = RAGEvaluator(mock_config, mock_graph, mock_vector_store)

        with (
            patch(
                "jama_mcp_server_graphrag.evaluation.runner.chat",
                new_callable=AsyncMock,
            ) as mock_chat,
            patch(
                "jama_mcp_server_graphrag.evaluation.runner.compute_all_metrics",
                new_callable=AsyncMock,
            ) as mock_metrics,
        ):
            mock_chat.return_value = {
                "answer": "Generated answer",
                "sources": [{"title": "Source 1"}],
            }
            mock_metrics.return_value = RAGMetrics(
                faithfulness=0.9,
                answer_relevancy=0.85,
                context_precision=0.8,
                context_recall=0.75,
            )

            result = await evaluator.evaluate_sample(sample_evaluation_sample)

            assert result.answer == "Generated answer"
            assert result.metrics.faithfulness == 0.9
            assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_evaluate_dataset(
        self,
        mock_config: AppConfig,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
        sample_evaluation_sample: EvaluationSample,
    ) -> None:
        """Test evaluating a dataset."""
        evaluator = RAGEvaluator(mock_config, mock_graph, mock_vector_store)

        with (
            patch(
                "jama_mcp_server_graphrag.evaluation.runner.chat",
                new_callable=AsyncMock,
            ) as mock_chat,
            patch(
                "jama_mcp_server_graphrag.evaluation.runner.compute_all_metrics",
                new_callable=AsyncMock,
            ) as mock_metrics,
        ):
            mock_chat.return_value = {
                "answer": "Answer",
                "sources": [],
            }
            mock_metrics.return_value = RAGMetrics(
                faithfulness=0.9,
                answer_relevancy=0.85,
                context_precision=0.8,
                context_recall=0.75,
            )

            report = await evaluator.evaluate(
                [sample_evaluation_sample],
                max_samples=1,
            )

            assert report.total_samples == 1
            assert "faithfulness" in report.aggregate_metrics


class TestEvaluateRagPipeline:
    """Tests for evaluate_rag_pipeline convenience function."""

    @pytest.mark.asyncio
    async def test_convenience_function(
        self,
        mock_config: AppConfig,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
        sample_evaluation_sample: EvaluationSample,
    ) -> None:
        """Test the convenience function."""
        with (
            patch(
                "jama_mcp_server_graphrag.evaluation.runner.chat",
                new_callable=AsyncMock,
            ) as mock_chat,
            patch(
                "jama_mcp_server_graphrag.evaluation.runner.compute_all_metrics",
                new_callable=AsyncMock,
            ) as mock_metrics,
        ):
            mock_chat.return_value = {
                "answer": "Answer",
                "sources": [],
            }
            mock_metrics.return_value = RAGMetrics(
                faithfulness=0.9,
                answer_relevancy=0.85,
                context_precision=0.8,
                context_recall=0.75,
            )

            report = await evaluate_rag_pipeline(
                mock_config,
                mock_graph,
                mock_vector_store,
                samples=[sample_evaluation_sample],
                max_samples=1,
            )

            assert isinstance(report, EvaluationReport)
            assert report.total_samples == 1
