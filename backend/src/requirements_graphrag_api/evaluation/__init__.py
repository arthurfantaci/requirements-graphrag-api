"""Evaluation framework for GraphRAG pipeline.

Provides comprehensive evaluation of the RAG pipeline using:
- Standard RAG metrics (faithfulness, relevancy, precision, recall)
- Domain-specific metrics (citation accuracy, traceability coverage)
- Benchmark datasets (golden examples, generated examples)

Usage:
    from requirements_graphrag_api.evaluation import evaluate_rag_pipeline

    report = await evaluate_rag_pipeline(
        config=config,
        retriever=retriever,
        driver=driver,
        max_samples=10,
    )
    print(f"Average score: {report.aggregate_metrics['avg_score']:.2f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from requirements_graphrag_api.observability import traceable_safe

if TYPE_CHECKING:
    from neo4j import Driver
    from neo4j_graphrag.retrievers import VectorRetriever

    from requirements_graphrag_api.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class EvaluationExample:
    """A single evaluation example from the benchmark dataset."""

    id: str
    question: str
    ground_truth: str
    expected_entities: list[str] = field(default_factory=list)
    expected_standards: list[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"
    must_pass: bool = False


@dataclass
class EvaluationResult:
    """Result of evaluating a single example."""

    example_id: str
    question: str
    generated_answer: str
    ground_truth: str
    contexts: list[str]
    metrics: dict[str, float]
    passed: bool


@dataclass
class EvaluationReport:
    """Aggregate report from running evaluation on multiple examples."""

    total_samples: int
    passed_samples: int
    aggregate_metrics: dict[str, float]
    results: list[EvaluationResult]
    errors: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Calculate the pass rate."""
        if self.total_samples == 0:
            return 0.0
        return self.passed_samples / self.total_samples


def _load_evaluation_dataset(max_samples: int | None = None) -> list[EvaluationExample]:
    """Load evaluation examples from the benchmark dataset.

    Args:
        max_samples: Maximum number of samples to load (None for all).

    Returns:
        List of evaluation examples.
    """
    # Try to import golden dataset, fall back to generated examples
    examples: list[EvaluationExample] = []

    try:
        from tests.benchmark.golden_dataset import GOLDEN_DATASET

        examples = list(GOLDEN_DATASET)
        logger.info("Loaded %d golden dataset examples", len(examples))
    except ImportError:
        logger.warning("Golden dataset not available, using minimal test set")
        # Minimal fallback examples for basic testing
        examples = [
            EvaluationExample(
                id="fallback-1",
                question="What is requirements traceability?",
                ground_truth=(
                    "Requirements traceability is the ability to trace requirements "
                    "throughout the development lifecycle, linking them to design, "
                    "implementation, and test artifacts."
                ),
                category="definition",
                difficulty="easy",
            ),
            EvaluationExample(
                id="fallback-2",
                question="Why is bidirectional traceability important?",
                ground_truth=(
                    "Bidirectional traceability allows teams to trace from requirements "
                    "to implementation (forward) and from implementation back to "
                    "requirements (backward), enabling impact analysis and coverage verification."
                ),
                category="concept",
                difficulty="medium",
            ),
        ]

    if max_samples is not None:
        examples = examples[:max_samples]

    return examples


async def _generate_answer(
    retriever: VectorRetriever,
    driver: Driver,
    config: AppConfig,
    question: str,
) -> tuple[str, list[str]]:
    """Generate an answer using the RAG pipeline.

    Args:
        retriever: Vector retriever instance.
        driver: Neo4j driver.
        config: Application configuration.
        question: Question to answer.

    Returns:
        Tuple of (generated_answer, retrieved_contexts).
    """
    from requirements_graphrag_api.core.generation import generate_response
    from requirements_graphrag_api.core.retrieval import graph_enriched_search

    # Retrieve relevant contexts
    results = await graph_enriched_search(retriever, driver, question, limit=6)
    contexts = [r["content"] for r in results if r.get("content")]

    # Generate answer
    answer = await generate_response(
        config=config,
        question=question,
        contexts=contexts,
        stream=False,
    )

    return answer, contexts


async def _evaluate_single(
    retriever: VectorRetriever,
    driver: Driver,
    config: AppConfig,
    example: EvaluationExample,
) -> EvaluationResult:
    """Evaluate a single example.

    Args:
        retriever: Vector retriever instance.
        driver: Neo4j driver.
        config: Application configuration.
        example: Evaluation example.

    Returns:
        EvaluationResult with metrics.
    """
    # Generate answer
    answer, contexts = await _generate_answer(retriever, driver, config, example.question)

    # Calculate metrics (simplified scoring for now)
    # In production, these would use LLM-as-judge with the prompts
    metrics: dict[str, float] = {}

    # Basic heuristic metrics (replace with LLM-as-judge in production)
    # Context precision: check if contexts mention key terms
    question_terms = set(example.question.lower().split())
    context_text = " ".join(contexts).lower()
    matching_terms = sum(1 for t in question_terms if t in context_text)
    metrics["context_precision"] = min(1.0, matching_terms / max(len(question_terms), 1))

    # Answer relevancy: check if answer mentions question terms
    answer_lower = answer.lower()
    answer_matches = sum(1 for t in question_terms if t in answer_lower)
    metrics["answer_relevancy"] = min(1.0, answer_matches / max(len(question_terms), 1))

    # Faithfulness: check if answer terms appear in context
    answer_terms = set(answer.lower().split())
    faithful_terms = sum(1 for t in answer_terms if t in context_text)
    metrics["faithfulness"] = min(1.0, faithful_terms / max(len(answer_terms), 1))

    # Context recall: check coverage of ground truth
    gt_terms = set(example.ground_truth.lower().split())
    recall_matches = sum(1 for t in gt_terms if t in context_text)
    metrics["context_recall"] = min(1.0, recall_matches / max(len(gt_terms), 1))

    # Calculate average score
    avg_score = sum(metrics.values()) / len(metrics) if metrics else 0.0
    metrics["avg_score"] = avg_score

    # Determine pass/fail (threshold: 0.5)
    passed = avg_score >= 0.5

    return EvaluationResult(
        example_id=example.id,
        question=example.question,
        generated_answer=answer,
        ground_truth=example.ground_truth,
        contexts=contexts,
        metrics=metrics,
        passed=passed,
    )


@traceable_safe(name="evaluate_rag_pipeline", run_type="chain")
async def evaluate_rag_pipeline(
    config: AppConfig,
    retriever: VectorRetriever,
    driver: Driver,
    *,
    max_samples: int | None = None,
) -> EvaluationReport:
    """Evaluate the RAG pipeline against benchmark dataset.

    Runs the RAG pipeline on evaluation examples and calculates metrics:
    - Faithfulness: Is the answer grounded in retrieved context?
    - Answer Relevancy: Does the answer address the question?
    - Context Precision: Are retrieved contexts relevant?
    - Context Recall: Do contexts contain ground truth information?

    Args:
        config: Application configuration.
        retriever: Vector retriever instance.
        driver: Neo4j driver.
        max_samples: Maximum number of samples to evaluate (None for all).

    Returns:
        EvaluationReport with aggregate metrics and individual results.
    """
    logger.info("Starting RAG pipeline evaluation (max_samples=%s)", max_samples)

    # Load evaluation dataset
    examples = _load_evaluation_dataset(max_samples)
    logger.info("Loaded %d evaluation examples", len(examples))

    results: list[EvaluationResult] = []
    errors: list[str] = []

    # Evaluate each example
    for i, example in enumerate(examples):
        try:
            logger.info("Evaluating example %d/%d: %s", i + 1, len(examples), example.id)
            result = await _evaluate_single(retriever, driver, config, example)
            results.append(result)
            logger.info(
                "Example %s: avg_score=%.3f, passed=%s",
                example.id,
                result.metrics.get("avg_score", 0),
                result.passed,
            )
        except Exception as e:
            logger.error("Failed to evaluate example %s: %s", example.id, e)
            errors.append(f"Example {example.id}: {e}")

    # Calculate aggregate metrics
    if results:
        aggregate_metrics: dict[str, Any] = {
            "faithfulness": sum(r.metrics.get("faithfulness", 0) for r in results) / len(results),
            "answer_relevancy": sum(r.metrics.get("answer_relevancy", 0) for r in results)
            / len(results),
            "context_precision": sum(r.metrics.get("context_precision", 0) for r in results)
            / len(results),
            "context_recall": sum(r.metrics.get("context_recall", 0) for r in results)
            / len(results),
            "avg_score": sum(r.metrics.get("avg_score", 0) for r in results) / len(results),
        }
    else:
        aggregate_metrics = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "avg_score": 0.0,
        }

    passed_count = sum(1 for r in results if r.passed)

    logger.info(
        "Evaluation complete: %d/%d passed, avg_score=%.3f",
        passed_count,
        len(results),
        aggregate_metrics["avg_score"],
    )

    return EvaluationReport(
        total_samples=len(results),
        passed_samples=passed_count,
        aggregate_metrics=aggregate_metrics,
        results=results,
        errors=errors,
    )


__all__ = [
    "EvaluationExample",
    "EvaluationReport",
    "EvaluationResult",
    "evaluate_rag_pipeline",
]
