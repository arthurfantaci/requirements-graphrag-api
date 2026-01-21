#!/usr/bin/env python
"""Run A/B comparisons between prompt variants.

This script enables prompt iteration by:
1. Comparing baseline prompts against variants
2. Running evaluations against LangSmith datasets
3. Generating comparison reports
4. Recommending promotion decisions

Usage:
    # Compare router prompt against a variant
    uv run python scripts/run_prompt_comparison.py router --variant v2

    # Run full A/B test with multiple iterations
    uv run python scripts/run_prompt_comparison.py router --variant v2 --iterations 3

    # List available prompts and datasets
    uv run python scripts/run_prompt_comparison.py --list

Requires LANGSMITH_API_KEY and evaluation datasets to be created.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Threshold for considering a metric improved or degraded
SIGNIFICANCE_THRESHOLD = 0.01


# =============================================================================
# CONFIGURATION
# =============================================================================

# Mapping of prompt names to their evaluation datasets
PROMPT_DATASET_MAP = {
    "router": {
        "prompt_name": "graphrag-router",
        "dataset_name": "graphrag-router-eval",
        "description": "Tool routing accuracy",
    },
    "critic": {
        "prompt_name": "graphrag-critic",
        "dataset_name": "graphrag-critic-eval",
        "description": "Context quality assessment",
    },
    "text2cypher": {
        "prompt_name": "graphrag-text2cypher",
        "dataset_name": "graphrag-text2cypher-eval",
        "description": "Cypher query generation",
    },
    "stepback": {
        "prompt_name": "graphrag-stepback",
        "dataset_name": None,  # No dataset yet
        "description": "Step-back query generation",
    },
    "query-updater": {
        "prompt_name": "graphrag-query-updater",
        "dataset_name": None,  # No dataset yet
        "description": "Multi-turn query refinement",
    },
    "rag-generation": {
        "prompt_name": "graphrag-rag-generation",
        "dataset_name": None,  # No dataset yet
        "description": "Answer generation with citations",
    },
}


@dataclass
class ComparisonReport:
    """Report from a prompt comparison."""

    baseline_name: str
    variant_name: str
    dataset_name: str
    timestamp: str
    iterations: int
    baseline_scores: dict[str, float]
    variant_scores: dict[str, float]
    improvements: dict[str, float]
    winner: str
    recommendation: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================


async def evaluate_prompt_on_dataset(
    client: Any,
    prompt_name: str,
    dataset_name: str,
    *,
    is_variant: bool = False,
) -> dict[str, float]:
    """Evaluate a prompt against a dataset.

    Args:
        client: LangSmith client.
        prompt_name: Name of the prompt to evaluate.
        dataset_name: Name of the evaluation dataset.
        is_variant: Whether this is a variant (affects logging).

    Returns:
        Dictionary of metric names to scores.
    """
    from jama_mcp_server_graphrag.prompts.evaluation import (
        create_cypher_validity_evaluator,
        create_json_validity_evaluator,
        create_length_evaluator,
    )
    from langchain_openai import ChatOpenAI

    logger.info(
        "Evaluating %s on dataset %s...",
        prompt_name + (" (variant)" if is_variant else " (baseline)"),
        dataset_name,
    )

    # Load dataset
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception as e:
        logger.error("Dataset not found: %s - %s", dataset_name, e)
        return {}

    examples = list(client.list_examples(dataset_id=dataset.id))
    logger.info("Found %d examples in dataset", len(examples))

    # Pull prompt from hub
    try:
        prompt_template = client.pull_prompt(prompt_name)
    except Exception as e:
        logger.error("Failed to pull prompt %s: %s", prompt_name, e)
        return {}

    # Create LLM for evaluation
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Determine evaluators based on prompt type
    evaluators = [create_length_evaluator()]
    if "router" in prompt_name or "critic" in prompt_name:
        evaluators.append(create_json_validity_evaluator())
    if "cypher" in prompt_name:
        evaluators.append(create_cypher_validity_evaluator())

    # Run evaluation
    all_scores: dict[str, list[float]] = {}
    successful = 0

    for i, example in enumerate(examples):
        try:
            # Format prompt with example inputs
            messages = prompt_template.format_messages(**example.inputs)

            # Get LLM response
            response = await llm.ainvoke(messages)
            output = response.content

            # Run evaluators
            run_output = {"output": output, "expected": example.outputs}
            for evaluator in evaluators:
                scores = evaluator(run_output)
                for metric, score in scores.items():
                    if metric not in all_scores:
                        all_scores[metric] = []
                    all_scores[metric].append(score)

            successful += 1

        except Exception as e:
            logger.warning("Failed to evaluate example %d: %s", i + 1, e)

    logger.info("Successfully evaluated %d/%d examples", successful, len(examples))

    # Calculate average scores
    return {
        metric: sum(scores) / len(scores) if scores else 0.0
        for metric, scores in all_scores.items()
    }


async def compare_prompts(
    client: Any,
    baseline_name: str,
    variant_name: str,
    dataset_name: str,
    *,
    iterations: int = 1,
) -> ComparisonReport:
    """Compare two prompt variants.

    Args:
        client: LangSmith client.
        baseline_name: Name of the baseline prompt.
        variant_name: Name of the variant prompt.
        dataset_name: Dataset to evaluate on.
        iterations: Number of evaluation iterations.

    Returns:
        ComparisonReport with results.
    """
    logger.info("=" * 60)
    logger.info("PROMPT COMPARISON")
    logger.info("=" * 60)
    logger.info("Baseline: %s", baseline_name)
    logger.info("Variant: %s", variant_name)
    logger.info("Dataset: %s", dataset_name)
    logger.info("Iterations: %d", iterations)
    logger.info("=" * 60)

    # Collect scores across iterations
    baseline_all_scores: dict[str, list[float]] = {}
    variant_all_scores: dict[str, list[float]] = {}

    for i in range(iterations):
        if iterations > 1:
            logger.info("\n--- Iteration %d/%d ---", i + 1, iterations)

        # Evaluate baseline
        baseline_scores = await evaluate_prompt_on_dataset(
            client, baseline_name, dataset_name, is_variant=False
        )
        for metric, score in baseline_scores.items():
            if metric not in baseline_all_scores:
                baseline_all_scores[metric] = []
            baseline_all_scores[metric].append(score)

        # Evaluate variant
        variant_scores = await evaluate_prompt_on_dataset(
            client, variant_name, dataset_name, is_variant=True
        )
        for metric, score in variant_scores.items():
            if metric not in variant_all_scores:
                variant_all_scores[metric] = []
            variant_all_scores[metric].append(score)

    # Calculate final averages
    baseline_final = {
        metric: sum(scores) / len(scores) for metric, scores in baseline_all_scores.items()
    }
    variant_final = {
        metric: sum(scores) / len(scores) for metric, scores in variant_all_scores.items()
    }

    # Calculate improvements
    all_metrics = set(baseline_final.keys()) | set(variant_final.keys())
    improvements = {
        metric: variant_final.get(metric, 0) - baseline_final.get(metric, 0)
        for metric in all_metrics
    }

    # Determine winner
    improved = sum(1 for v in improvements.values() if v > SIGNIFICANCE_THRESHOLD)
    degraded = sum(1 for v in improvements.values() if v < -SIGNIFICANCE_THRESHOLD)

    if improved > degraded:
        winner = "variant"
        recommendation = f"PROMOTE: Variant shows improvement in {improved} metrics"
    elif degraded > improved:
        winner = "baseline"
        recommendation = f"KEEP BASELINE: Variant degraded in {degraded} metrics"
    else:
        winner = "tie"
        recommendation = "NO CHANGE: Results are statistically similar"

    return ComparisonReport(
        baseline_name=baseline_name,
        variant_name=variant_name,
        dataset_name=dataset_name,
        timestamp=datetime.now(tz=UTC).isoformat(),
        iterations=iterations,
        baseline_scores=baseline_final,
        variant_scores=variant_final,
        improvements=improvements,
        winner=winner,
        recommendation=recommendation,
        details={
            "improved_metrics": improved,
            "degraded_metrics": degraded,
            "total_metrics": len(all_metrics),
        },
    )


def print_report(report: ComparisonReport) -> None:
    """Print a formatted comparison report."""
    print("\n" + "=" * 60)
    print("COMPARISON REPORT")
    print("=" * 60)
    print(f"Timestamp: {report.timestamp}")
    print(f"Baseline: {report.baseline_name}")
    print(f"Variant: {report.variant_name}")
    print(f"Dataset: {report.dataset_name}")
    print(f"Iterations: {report.iterations}")

    print("\n--- SCORES ---")
    print(f"{'Metric':<25} {'Baseline':>10} {'Variant':>10} {'Change':>10}")
    print("-" * 55)
    for metric in sorted(report.baseline_scores.keys()):
        baseline = report.baseline_scores.get(metric, 0)
        variant = report.variant_scores.get(metric, 0)
        change = report.improvements.get(metric, 0)
        threshold = SIGNIFICANCE_THRESHOLD
        indicator = "+" if change > threshold else ("-" if change < -threshold else "=")
        print(f"{metric:<25} {baseline:>10.3f} {variant:>10.3f} {indicator}{abs(change):>9.3f}")

    print("\n--- RESULT ---")
    print(f"Winner: {report.winner.upper()}")
    print(f"Recommendation: {report.recommendation}")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================


def list_prompts() -> None:
    """List available prompts and their datasets."""
    print("\nAvailable Prompts for Comparison:")
    print("=" * 60)
    for key, config in PROMPT_DATASET_MAP.items():
        dataset_status = config["dataset_name"] or "(no dataset)"
        print(f"  {key:<15} - {config['description']}")
        print(f"                  Dataset: {dataset_status}")
    print("\nUsage: uv run python scripts/run_prompt_comparison.py <prompt> --variant <name>")


async def main(
    prompt_key: str,
    variant_suffix: str,
    iterations: int = 1,
    output_file: str | None = None,
) -> int:
    """Run prompt comparison.

    Args:
        prompt_key: Key from PROMPT_DATASET_MAP.
        variant_suffix: Suffix for variant name (e.g., 'v2' -> prompt-name-v2).
        iterations: Number of evaluation iterations.
        output_file: Optional file to save results.

    Returns:
        Exit code.
    """
    if not os.getenv("LANGSMITH_API_KEY"):
        logger.error("LANGSMITH_API_KEY not set")
        return 1

    if prompt_key not in PROMPT_DATASET_MAP:
        logger.error("Unknown prompt: %s", prompt_key)
        list_prompts()
        return 1

    config = PROMPT_DATASET_MAP[prompt_key]
    if not config["dataset_name"]:
        logger.error(
            "No evaluation dataset for %s. Create one first with create_eval_datasets.py",
            prompt_key,
        )
        return 1

    try:
        from langsmith import Client
    except ImportError:
        logger.error("langsmith not installed. Run: pip install langsmith")
        return 1

    client = Client()

    baseline_name = config["prompt_name"]
    variant_name = f"{config['prompt_name']}-{variant_suffix}"
    dataset_name = config["dataset_name"]

    # Check if variant exists
    try:
        client.pull_prompt(variant_name)
    except Exception:
        logger.error("Variant prompt not found: %s", variant_name)
        logger.info(
            "Create a variant first by pushing a modified prompt with name: %s",
            variant_name,
        )
        logger.info("Or use the LangSmith Playground to create and save a variant.")
        return 1

    # Run comparison
    report = await compare_prompts(
        client,
        baseline_name,
        variant_name,
        dataset_name,
        iterations=iterations,
    )

    # Print report
    print_report(report)

    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.write_text(json.dumps(report.to_dict(), indent=2))
        logger.info("Report saved to: %s", output_path)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run A/B comparisons between prompt variants")
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt to compare (router, critic, text2cypher, etc.)",
    )
    parser.add_argument(
        "--variant",
        "-v",
        default="v2",
        help="Variant suffix (default: v2, creates prompt-name-v2)",
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=1,
        help="Number of evaluation iterations (default: 1)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for JSON report",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available prompts",
    )

    args = parser.parse_args()

    if args.list or not args.prompt:
        list_prompts()
        sys.exit(0)

    sys.exit(asyncio.run(main(args.prompt, args.variant, args.iterations, args.output)))
