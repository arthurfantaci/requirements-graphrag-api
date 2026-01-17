"""Evaluation runner for GraphRAG pipeline assessment.

Orchestrates end-to-end evaluation of the RAG pipeline:
1. Runs queries through the RAG pipeline
2. Computes evaluation metrics
3. Reports results to LangSmith (if enabled)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from jama_mcp_server_graphrag.core import chat
from jama_mcp_server_graphrag.evaluation.datasets import (
    EvaluationSample,
    get_sample_evaluation_data,
)
from jama_mcp_server_graphrag.evaluation.metrics import RAGMetrics, compute_all_metrics
from jama_mcp_server_graphrag.observability import traceable

if TYPE_CHECKING:
    from langchain_neo4j import Neo4jGraph, Neo4jVector

    from jama_mcp_server_graphrag.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of a single evaluation sample.

    Attributes:
        sample: The evaluation sample that was tested.
        answer: The generated answer from the RAG pipeline.
        contexts: The retrieved contexts.
        metrics: Computed evaluation metrics.
        latency_ms: Time taken to generate answer in milliseconds.
        metadata: Additional metadata about the evaluation.
    """

    sample: EvaluationSample
    answer: str
    contexts: list[str]
    metrics: RAGMetrics
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "question": self.sample.question,
            "ground_truth": self.sample.ground_truth,
            "answer": self.answer,
            "contexts": self.contexts,
            "metrics": self.metrics.to_dict(),
            "latency_ms": self.latency_ms,
            "metadata": {**self.sample.metadata, **self.metadata},
        }


@dataclass
class EvaluationReport:
    """Aggregated evaluation report.

    Attributes:
        results: Individual evaluation results.
        aggregate_metrics: Average metrics across all samples.
        total_samples: Number of samples evaluated.
        timestamp: When the evaluation was run.
    """

    results: list[EvaluationResult]
    aggregate_metrics: dict[str, float]
    total_samples: int
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "total_samples": self.total_samples,
            "aggregate_metrics": self.aggregate_metrics,
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
        }


class RAGEvaluator:
    """Evaluator for GraphRAG pipeline quality assessment.

    Runs evaluation samples through the RAG pipeline and computes
    RAGAS-based metrics with optional LangSmith integration.
    """

    def __init__(
        self,
        config: AppConfig,
        graph: Neo4jGraph,
        vector_store: Neo4jVector,
    ) -> None:
        """Initialize the evaluator.

        Args:
            config: Application configuration.
            graph: Neo4jGraph connection.
            vector_store: Neo4jVector for retrieval.
        """
        self.config = config
        self.graph = graph
        self.vector_store = vector_store

    @traceable(name="evaluate_sample", run_type="chain")
    async def evaluate_sample(
        self,
        sample: EvaluationSample,
    ) -> EvaluationResult:
        """Evaluate a single sample through the RAG pipeline.

        Args:
            sample: The evaluation sample to test.

        Returns:
            EvaluationResult with answer, contexts, and metrics.
        """
        logger.info("Evaluating sample: %s", sample.question[:50])

        # Time the RAG pipeline
        start_time = time.perf_counter()

        # Run through RAG pipeline
        result = await chat(
            self.config,
            self.graph,
            self.vector_store,
            sample.question,
            max_sources=5,
        )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Extract answer and contexts
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        contexts = [s.get("title", "") for s in sources]

        # Compute evaluation metrics
        metrics = await compute_all_metrics(
            self.config,
            sample.question,
            answer,
            contexts,
            sample.ground_truth,
        )

        return EvaluationResult(
            sample=sample,
            answer=answer,
            contexts=contexts,
            metrics=metrics,
            latency_ms=latency_ms,
            metadata={"source_count": len(sources)},
        )

    @traceable(name="evaluate_dataset", run_type="chain")
    async def evaluate(
        self,
        samples: list[EvaluationSample] | None = None,
        *,
        max_samples: int | None = None,
    ) -> EvaluationReport:
        """Run evaluation on a dataset.

        Args:
            samples: Evaluation samples. Uses defaults if not provided.
            max_samples: Optional limit on number of samples to evaluate.

        Returns:
            EvaluationReport with all results and aggregate metrics.
        """
        if samples is None:
            samples = get_sample_evaluation_data()

        if max_samples is not None:
            samples = samples[:max_samples]

        logger.info("Starting evaluation of %d samples", len(samples))

        # Evaluate each sample
        results = []
        for sample in samples:
            try:
                result = await self.evaluate_sample(sample)
                results.append(result)
            except Exception as e:
                logger.warning("Failed to evaluate sample '%s': %s", sample.question[:30], e)

        # Compute aggregate metrics
        if results:
            aggregate_metrics = {
                "faithfulness": sum(r.metrics.faithfulness for r in results) / len(results),
                "answer_relevancy": sum(r.metrics.answer_relevancy for r in results) / len(results),
                "context_precision": sum(r.metrics.context_precision for r in results)
                / len(results),
                "context_recall": sum(r.metrics.context_recall for r in results) / len(results),
                "average_latency_ms": sum(r.latency_ms for r in results) / len(results),
            }
            aggregate_metrics["average"] = (
                aggregate_metrics["faithfulness"]
                + aggregate_metrics["answer_relevancy"]
                + aggregate_metrics["context_precision"]
                + aggregate_metrics["context_recall"]
            ) / 4
        else:
            aggregate_metrics = {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "average_latency_ms": 0.0,
                "average": 0.0,
            }

        logger.info(
            "Evaluation complete: %d samples, avg score: %.3f",
            len(results),
            aggregate_metrics["average"],
        )

        return EvaluationReport(
            results=results,
            aggregate_metrics=aggregate_metrics,
            total_samples=len(results),
            timestamp=datetime.now(tz=UTC).isoformat(),
        )


@traceable(name="evaluate_rag_pipeline", run_type="chain")
async def evaluate_rag_pipeline(
    config: AppConfig,
    graph: Neo4jGraph,
    vector_store: Neo4jVector,
    *,
    samples: list[EvaluationSample] | None = None,
    max_samples: int | None = None,
) -> EvaluationReport:
    """Convenience function to run RAG pipeline evaluation.

    Args:
        config: Application configuration.
        graph: Neo4jGraph connection.
        vector_store: Neo4jVector for retrieval.
        samples: Optional custom evaluation samples.
        max_samples: Optional limit on samples to evaluate.

    Returns:
        EvaluationReport with results and aggregate metrics.
    """
    evaluator = RAGEvaluator(config, graph, vector_store)
    return await evaluator.evaluate(samples, max_samples=max_samples)
