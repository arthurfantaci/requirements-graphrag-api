"""Evaluation framework for GraphRAG pipeline.

Provides comprehensive evaluation of the RAG pipeline using:
- Standard RAG metrics (faithfulness, relevancy, precision, recall)
- Domain-specific metrics (citation accuracy, traceability coverage)
- Agentic metrics (tool selection, iteration efficiency, critic calibration)
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

Agentic evaluators for LangSmith:
    from requirements_graphrag_api.evaluation import (
        tool_selection_evaluator,
        iteration_efficiency_evaluator,
        critic_calibration_evaluator,
        multi_hop_reasoning_evaluator,
    )

    results = evaluate(
        target=agentic_rag_chain,
        data="graphrag-agentic-eval",
        evaluators=[tool_selection_evaluator, iteration_efficiency_evaluator],
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from requirements_graphrag_api.evaluation.agentic_evaluators import (
    critic_calibration_evaluator,
    critic_calibration_evaluator_sync,
    iteration_efficiency_evaluator,
    iteration_efficiency_evaluator_sync,
    multi_hop_reasoning_evaluator,
    multi_hop_reasoning_evaluator_sync,
    tool_selection_evaluator,
    tool_selection_evaluator_sync,
)
from requirements_graphrag_api.evaluation.cost_analysis import (
    CostReport,
    CostTracker,
    LLMCall,
    estimate_cost,
    get_cost_report,
    get_global_cost_tracker,
    reset_global_cost_tracker,
)
from requirements_graphrag_api.evaluation.metrics import (
    ANSWER_RELEVANCY_PROMPT,
    CONTEXT_PRECISION_PROMPT,
    CONTEXT_RECALL_PROMPT,
    FAITHFULNESS_PROMPT,
)
from requirements_graphrag_api.evaluation.performance import (
    ExecutionMetrics,
    PerformanceTracker,
    SubgraphMetrics,
    get_global_tracker,
    get_performance_summary,
    reset_global_tracker,
    track_execution,
)
from requirements_graphrag_api.evaluation.ragas_evaluators import (
    _get_judge_llm,
    _parse_llm_score,
)
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
    # Import core functions directly to avoid decorator issues during CI evaluation
    # (the @traceable_safe decorator can have interaction issues with langsmith)
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI

    from requirements_graphrag_api.core.definitions import _build_context_from_results, search_terms
    from requirements_graphrag_api.core.retrieval import graph_enriched_search
    from requirements_graphrag_api.prompts import PromptName, get_prompt_sync

    # Search for relevant definitions/glossary terms
    definitions = await search_terms(driver, question, limit=3)

    # Retrieve relevant context from chunks
    search_results = await graph_enriched_search(
        retriever,
        driver,
        question,
        limit=6,
    )

    # Build context and extract resources
    build_result = _build_context_from_results(
        definitions,
        search_results,
        include_entities=True,
    )

    # Get prompt from catalog
    prompt_template = get_prompt_sync(PromptName.RAG_GENERATION)

    # Generate answer with LLM
    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0.1,
        api_key=config.openai_api_key,
    )

    chain = prompt_template | llm | StrOutputParser()

    answer = await chain.ainvoke(
        {
            "context": build_result.context,
            "entities": build_result.entities_str,
            "question": question,
        }
    )

    # Extract contexts from sources
    contexts = [s.get("content", "") for s in build_result.sources if s.get("content")]

    return answer, contexts


async def _llm_judge_metrics(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict[str, float]:
    """Score a single RAG example using LLM-as-judge (RAGAS-style).

    Runs 4 RAGAS metrics concurrently via gpt-4o-mini:
    faithfulness, answer_relevancy, context_precision, context_recall.

    Returns:
        Dict mapping metric name to score (0.0-1.0).
    """
    import asyncio

    contexts_str = "\n\n---\n\n".join(contexts) if contexts else ""

    prompts = {
        "faithfulness": FAITHFULNESS_PROMPT.format(
            question=question,
            context=contexts_str,
            answer=answer,
        ),
        "answer_relevancy": ANSWER_RELEVANCY_PROMPT.format(
            question=question,
            answer=answer,
        ),
        "context_precision": CONTEXT_PRECISION_PROMPT.format(
            question=question,
            contexts=contexts_str,
        ),
        "context_recall": CONTEXT_RECALL_PROMPT.format(
            question=question,
            contexts=contexts_str,
            ground_truth=ground_truth,
        ),
    }

    llm = _get_judge_llm()

    async def _score(metric_name: str, prompt: str) -> tuple[str, float]:
        try:
            response = await llm.ainvoke(prompt)
            score, reasoning = _parse_llm_score(response.content)
            logger.debug("%s: score=%.2f reason=%s", metric_name, score, reasoning)
            return metric_name, score
        except Exception as e:
            logger.warning("LLM judge failed for %s: %s", metric_name, e)
            return metric_name, 0.0

    results = await asyncio.gather(*[_score(name, prompt) for name, prompt in prompts.items()])

    return dict(results)


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

    # Score with LLM-as-judge (RAGAS-style)
    metrics = await _llm_judge_metrics(
        question=example.question,
        answer=answer,
        contexts=contexts,
        ground_truth=example.ground_truth,
    )

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
    # Cost analysis
    "CostReport",
    "CostTracker",
    # Core evaluation
    "EvaluationExample",
    "EvaluationReport",
    "EvaluationResult",
    # Performance tracking
    "ExecutionMetrics",
    "LLMCall",
    "PerformanceTracker",
    "SubgraphMetrics",
    # Agentic evaluators
    "critic_calibration_evaluator",
    "critic_calibration_evaluator_sync",
    "estimate_cost",
    "evaluate_rag_pipeline",
    "get_cost_report",
    "get_global_cost_tracker",
    "get_global_tracker",
    "get_performance_summary",
    "iteration_efficiency_evaluator",
    "iteration_efficiency_evaluator_sync",
    "multi_hop_reasoning_evaluator",
    "multi_hop_reasoning_evaluator_sync",
    "reset_global_cost_tracker",
    "reset_global_tracker",
    "tool_selection_evaluator",
    "tool_selection_evaluator_sync",
    "track_execution",
]
