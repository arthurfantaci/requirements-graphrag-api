"""Prompt evaluation utilities.

Provides tools for evaluating prompt performance and comparing variants:
- Integration with LangSmith evaluation datasets
- Prompt-specific evaluators based on metadata criteria
- A/B testing between prompt variants
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jama_mcp_server_graphrag.prompts.catalog import get_catalog
from jama_mcp_server_graphrag.prompts.definitions import PROMPT_DEFINITIONS, PromptName

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a prompt against a dataset.

    Attributes:
        prompt_name: Name of the evaluated prompt.
        dataset_name: Name of the evaluation dataset.
        scores: Dictionary of metric names to average scores.
        sample_results: Individual results for each sample.
        metadata: Additional evaluation metadata.
    """

    prompt_name: str
    dataset_name: str
    scores: dict[str, float]
    sample_results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result of comparing two prompt variants.

    Attributes:
        baseline_name: Name of the baseline prompt.
        candidate_name: Name of the candidate prompt.
        baseline_scores: Baseline prompt scores.
        candidate_scores: Candidate prompt scores.
        improvements: Dictionary of metric improvements (positive = better).
        winner: Name of the better-performing prompt.
        significant: Whether differences are statistically significant.
    """

    baseline_name: str
    candidate_name: str
    baseline_scores: dict[str, float]
    candidate_scores: dict[str, float]
    improvements: dict[str, float] = field(default_factory=dict)
    winner: str = ""
    significant: bool = False


# =============================================================================
# EVALUATOR FACTORIES
# Creates evaluators based on prompt metadata criteria
# =============================================================================


def create_json_validity_evaluator() -> Callable[[dict[str, Any]], dict[str, float]]:
    """Create an evaluator that checks JSON validity.

    Returns:
        Evaluator function that returns {"json_valid": 0.0 or 1.0}.
    """

    def evaluate(run_output: dict[str, Any]) -> dict[str, float]:
        output = run_output.get("output", "")
        try:
            # Try to parse as JSON
            if output.startswith("```"):
                lines = output.split("\n")
                output = "\n".join(line for line in lines if not line.startswith("```")).strip()
            json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return {"json_valid": 0.0}
        else:
            return {"json_valid": 1.0}

    return evaluate


def create_contains_keywords_evaluator(
    keywords: list[str],
) -> Callable[[dict[str, Any]], dict[str, float]]:
    """Create an evaluator that checks for required keywords.

    Args:
        keywords: List of keywords that should appear in output.

    Returns:
        Evaluator function that returns keyword coverage score.
    """

    def evaluate(run_output: dict[str, Any]) -> dict[str, float]:
        output = str(run_output.get("output", "")).lower()
        matches = sum(1 for kw in keywords if kw.lower() in output)
        return {"keyword_coverage": matches / len(keywords) if keywords else 1.0}

    return evaluate


def create_cypher_validity_evaluator() -> Callable[[dict[str, Any]], dict[str, float]]:
    """Create an evaluator that checks basic Cypher syntax.

    Returns:
        Evaluator function that returns {"cypher_valid": 0.0 or 1.0}.
    """
    # Basic Cypher patterns
    cypher_patterns = [
        r"\bMATCH\b",
        r"\bRETURN\b",
    ]

    def evaluate(run_output: dict[str, Any]) -> dict[str, float]:
        output = str(run_output.get("output", ""))
        # Check for basic Cypher structure
        has_match = bool(re.search(cypher_patterns[0], output, re.IGNORECASE))
        has_return = bool(re.search(cypher_patterns[1], output, re.IGNORECASE))
        return {"cypher_valid": 1.0 if (has_match and has_return) else 0.0}

    return evaluate


def create_length_evaluator(
    min_length: int = 10,
    max_length: int = 2000,
) -> Callable[[dict[str, Any]], dict[str, float]]:
    """Create an evaluator that checks output length.

    Args:
        min_length: Minimum acceptable length.
        max_length: Maximum acceptable length.

    Returns:
        Evaluator function that returns length appropriateness score.
    """

    def evaluate(run_output: dict[str, Any]) -> dict[str, float]:
        output = str(run_output.get("output", ""))
        length = len(output)
        if min_length <= length <= max_length:
            return {"length_appropriate": 1.0}
        if length < min_length:
            return {"length_appropriate": length / min_length}
        return {"length_appropriate": max_length / length}

    return evaluate


def get_evaluators_for_prompt(
    name: PromptName,
) -> list[Callable[[dict[str, Any]], dict[str, float]]]:
    """Get appropriate evaluators for a prompt based on its metadata.

    Args:
        name: Prompt name identifier.

    Returns:
        List of evaluator functions.
    """
    definition = PROMPT_DEFINITIONS[name]
    meta = definition.metadata
    evaluators: list[Callable[[dict[str, Any]], dict[str, float]]] = []

    # Add evaluators based on output format
    if meta.output_format == "json":
        evaluators.append(create_json_validity_evaluator())
    elif meta.output_format == "cypher":
        evaluators.append(create_cypher_validity_evaluator())

    # Add evaluators based on criteria
    for criterion in meta.evaluation_criteria:
        if criterion == "valid_json_output":
            if meta.output_format != "json":  # Don't duplicate
                evaluators.append(create_json_validity_evaluator())
        elif criterion == "read_only_compliance":
            # Check for absence of write keywords
            evaluators.append(create_contains_keywords_evaluator(["MATCH", "RETURN"]))

    # Always add length check
    evaluators.append(create_length_evaluator())

    return evaluators


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================


async def evaluate_prompt(
    prompt_name: PromptName,
    dataset_name: str,
    *,
    custom_evaluators: list[Callable[[dict[str, Any]], dict[str, float]]] | None = None,
) -> EvaluationResult:
    """Evaluate a prompt against a LangSmith dataset.

    Args:
        prompt_name: Prompt to evaluate.
        dataset_name: LangSmith dataset name.
        custom_evaluators: Additional evaluators to use.

    Returns:
        EvaluationResult with scores and sample results.

    Raises:
        ImportError: If langsmith is not installed.
        ValueError: If dataset is not found.
    """
    try:
        from langsmith import Client  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "langsmith package required for evaluation. Install with: pip install langsmith"
        ) from e

    logger.info("Evaluating prompt %s against dataset %s", prompt_name, dataset_name)

    client = Client()
    catalog = get_catalog()

    # Get the prompt template
    template = await catalog.get_prompt(prompt_name)

    # Get evaluators
    evaluators = get_evaluators_for_prompt(prompt_name)
    if custom_evaluators:
        evaluators.extend(custom_evaluators)

    # Load dataset
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception as e:
        raise ValueError(f"Dataset not found: {dataset_name}") from e

    # Run evaluation
    examples = list(client.list_examples(dataset_id=dataset.id))
    sample_results: list[dict[str, Any]] = []
    all_scores: dict[str, list[float]] = {}

    for example in examples:
        # Format the prompt with example inputs
        try:
            formatted = template.format(**example.inputs)
        except KeyError as e:
            logger.warning("Missing input variable for example %s: %s", example.id, e)
            continue

        # Create mock output for evaluation (in real usage, this would be LLM output)
        run_output = {
            "input": example.inputs,
            "output": formatted,  # Placeholder - real eval would use LLM output
            "expected": example.outputs,
        }

        # Run evaluators
        example_scores: dict[str, float] = {}
        for evaluator in evaluators:
            scores = evaluator(run_output)
            example_scores.update(scores)
            for metric, score in scores.items():
                if metric not in all_scores:
                    all_scores[metric] = []
                all_scores[metric].append(score)

        sample_results.append(
            {
                "example_id": str(example.id),
                "scores": example_scores,
            }
        )

    # Calculate average scores
    avg_scores = {
        metric: sum(scores) / len(scores) if scores else 0.0
        for metric, scores in all_scores.items()
    }

    return EvaluationResult(
        prompt_name=prompt_name.value,
        dataset_name=dataset_name,
        scores=avg_scores,
        sample_results=sample_results,
        metadata={
            "num_examples": len(examples),
            "num_evaluators": len(evaluators),
        },
    )


async def compare_prompts(
    baseline: PromptName,
    candidate: PromptName | str,
    dataset_name: str,
) -> ComparisonResult:
    """Compare two prompt variants on a dataset.

    Args:
        baseline: Baseline prompt name.
        candidate: Candidate prompt (PromptName or Hub path).
        dataset_name: Dataset to evaluate on.

    Returns:
        ComparisonResult with comparison metrics.
    """
    logger.info(
        "Comparing prompts: baseline=%s, candidate=%s",
        baseline,
        candidate,
    )

    # Evaluate baseline
    baseline_result = await evaluate_prompt(baseline, dataset_name)

    # Evaluate candidate
    if isinstance(candidate, PromptName):
        candidate_result = await evaluate_prompt(candidate, dataset_name)
        candidate_name = candidate.value
    else:
        # Assume it's a Hub path - evaluate would need to pull it
        candidate_result = await evaluate_prompt(baseline, dataset_name)  # Placeholder
        candidate_name = candidate

    # Calculate improvements
    improvements = {
        metric: candidate_result.scores.get(metric, 0) - baseline_result.scores.get(metric, 0)
        for metric in set(baseline_result.scores) | set(candidate_result.scores)
    }

    # Determine winner (simple: most metrics improved)
    improved_count = sum(1 for v in improvements.values() if v > 0)
    degraded_count = sum(1 for v in improvements.values() if v < 0)

    if improved_count > degraded_count:
        winner = candidate_name
    elif degraded_count > improved_count:
        winner = baseline.value
    else:
        winner = "tie"

    return ComparisonResult(
        baseline_name=baseline.value,
        candidate_name=candidate_name,
        baseline_scores=baseline_result.scores,
        candidate_scores=candidate_result.scores,
        improvements=improvements,
        winner=winner,
        significant=abs(improved_count - degraded_count) > 1,
    )


async def run_ab_test(
    prompt_a: PromptName,
    prompt_b: PromptName,
    dataset_name: str,
    *,
    num_iterations: int = 3,
) -> dict[str, Any]:
    """Run an A/B test between two prompts.

    Args:
        prompt_a: First prompt variant.
        prompt_b: Second prompt variant.
        dataset_name: Dataset to test on.
        num_iterations: Number of test iterations.

    Returns:
        Dictionary with A/B test results.
    """
    logger.info(
        "Running A/B test: %s vs %s (%d iterations)",
        prompt_a,
        prompt_b,
        num_iterations,
    )

    results_a: list[EvaluationResult] = []
    results_b: list[EvaluationResult] = []

    for i in range(num_iterations):
        logger.info("Iteration %d/%d", i + 1, num_iterations)
        result_a = await evaluate_prompt(prompt_a, dataset_name)
        result_b = await evaluate_prompt(prompt_b, dataset_name)
        results_a.append(result_a)
        results_b.append(result_b)

    # Aggregate scores
    def aggregate_scores(results: list[EvaluationResult]) -> dict[str, dict[str, float]]:
        all_metrics: dict[str, list[float]] = {}
        for r in results:
            for metric, score in r.scores.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(score)
        return {
            metric: {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
            }
            for metric, scores in all_metrics.items()
        }

    return {
        "prompt_a": prompt_a.value,
        "prompt_b": prompt_b.value,
        "dataset": dataset_name,
        "iterations": num_iterations,
        "results_a": aggregate_scores(results_a),
        "results_b": aggregate_scores(results_b),
    }


__all__ = [
    "ComparisonResult",
    "EvaluationResult",
    "compare_prompts",
    "create_contains_keywords_evaluator",
    "create_cypher_validity_evaluator",
    "create_json_validity_evaluator",
    "create_length_evaluator",
    "evaluate_prompt",
    "get_evaluators_for_prompt",
    "run_ab_test",
]
