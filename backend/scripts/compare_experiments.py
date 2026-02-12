#!/usr/bin/env python
"""Compare LangSmith experiments.

This script demonstrates LangSmith experiment comparison:
- Listing experiments for a dataset
- Comparing metrics between experiments
- Statistical comparison for decision making

Usage:
    # List experiments for the golden dataset
    uv run python scripts/compare_experiments.py --list

    # Compare two experiments
    uv run python scripts/compare_experiments.py --compare exp1 exp2

    # Show details of an experiment
    uv run python scripts/compare_experiments.py --show exp-name

Requires LANGSMITH_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
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

from requirements_graphrag_api.evaluation.golden_dataset import DATASET_NAME  # noqa: E402

# Configuration
DEFAULT_DATASET = DATASET_NAME


def list_experiments(dataset_name: str = DEFAULT_DATASET) -> list[dict[str, Any]]:
    """List all experiments for a dataset.

    Args:
        dataset_name: Name of the dataset to query.

    Returns:
        List of experiment summaries.
    """
    from langsmith import Client

    client = Client()

    print("\n" + "=" * 70)
    print(f"EXPERIMENTS FOR DATASET: {dataset_name}")
    print("=" * 70)

    try:
        # Get dataset ID
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception as e:
        logger.error("Dataset not found: %s - %s", dataset_name, e)
        return []

    experiments = []

    try:
        # List projects (experiments) that reference this dataset
        projects = list(client.list_projects())

        for project in projects:
            # Filter to projects that are experiments for this dataset
            if project.reference_dataset_id == dataset.id:
                exp_info = {
                    "name": project.name,
                    "id": str(project.id),
                    "created_at": str(project.created_at) if project.created_at else "N/A",
                    "run_count": project.run_count or 0,
                }
                experiments.append(exp_info)

        if not experiments:
            print("No experiments found for this dataset.")
            print("Run run_ragas_evaluation.py to create experiments.")
        else:
            print(f"\n{'Name':<40} {'Runs':>6} {'Created':<20}")
            print("-" * 70)
            for exp in sorted(experiments, key=lambda x: x.get("created_at", ""), reverse=True):
                created = exp.get("created_at", "N/A")[:19]
                print(f"{exp['name']:<40} {exp['run_count']:>6} {created:<20}")

        print("=" * 70)

    except Exception as e:
        logger.error("Failed to list experiments: %s", e)

    return experiments


def get_experiment_metrics(experiment_name: str) -> dict[str, float]:
    """Get aggregate metrics for an experiment.

    Args:
        experiment_name: Name of the experiment.

    Returns:
        Dict of metric names to average scores.
    """
    from langsmith import Client

    client = Client()

    metrics: dict[str, list[float]] = {}

    try:
        # Find the project
        projects = list(client.list_projects())
        project = None
        for p in projects:
            if p.name == experiment_name:
                project = p
                break

        if not project:
            logger.error("Experiment not found: %s", experiment_name)
            return {}

        # Get runs for this project
        runs = list(client.list_runs(project_id=project.id))

        for run in runs:
            # Get feedback for each run
            feedbacks = list(client.list_feedback(run_ids=[run.id]))

            for feedback in feedbacks:
                key = feedback.key
                if feedback.score is not None:
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(feedback.score)

    except Exception as e:
        logger.error("Failed to get experiment metrics: %s", e)

    # Calculate averages
    return {key: sum(scores) / len(scores) if scores else 0.0 for key, scores in metrics.items()}


def show_experiment(experiment_name: str) -> None:
    """Show details of an experiment.

    Args:
        experiment_name: Name of the experiment.
    """
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 60)

    metrics = get_experiment_metrics(experiment_name)

    if not metrics:
        print("No metrics found for this experiment.")
        print("The experiment may not have completed or may not exist.")
        return

    print(f"\n{'Metric':<25} {'Score':>10}")
    print("-" * 40)

    for metric, score in sorted(metrics.items()):
        print(f"{metric:<25} {score:>10.3f}")

    # Overall average
    avg = sum(metrics.values()) / len(metrics) if metrics else 0.0
    print("-" * 40)
    print(f"{'AVERAGE':<25} {avg:>10.3f}")
    print("=" * 60)


def compare_experiments(exp1_name: str, exp2_name: str) -> dict[str, Any]:
    """Compare two experiments.

    Args:
        exp1_name: Name of first experiment (baseline).
        exp2_name: Name of second experiment (variant).

    Returns:
        Comparison report dict.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON")
    print("=" * 70)
    print(f"Baseline: {exp1_name}")
    print(f"Variant:  {exp2_name}")
    print("=" * 70)

    exp1_metrics = get_experiment_metrics(exp1_name)
    exp2_metrics = get_experiment_metrics(exp2_name)

    if not exp1_metrics:
        logger.error("No metrics found for baseline experiment: %s", exp1_name)
        return {"error": f"No metrics for {exp1_name}"}

    if not exp2_metrics:
        logger.error("No metrics found for variant experiment: %s", exp2_name)
        return {"error": f"No metrics for {exp2_name}"}

    # Compare metrics
    all_metrics = set(exp1_metrics.keys()) | set(exp2_metrics.keys())

    print(f"\n{'Metric':<25} {'Baseline':>10} {'Variant':>10} {'Change':>10}")
    print("-" * 60)

    improvements = {}
    for metric in sorted(all_metrics):
        baseline = exp1_metrics.get(metric, 0.0)
        variant = exp2_metrics.get(metric, 0.0)
        change = variant - baseline
        improvements[metric] = change

        # Indicator
        threshold = 0.01
        if change > threshold:
            indicator = "+"
        elif change < -threshold:
            indicator = "-"
        else:
            indicator = "="

        print(f"{metric:<25} {baseline:>10.3f} {variant:>10.3f} {indicator}{abs(change):>9.3f}")

    print("-" * 60)

    # Overall assessment
    exp1_avg = sum(exp1_metrics.values()) / len(exp1_metrics) if exp1_metrics else 0.0
    exp2_avg = sum(exp2_metrics.values()) / len(exp2_metrics) if exp2_metrics else 0.0
    overall_change = exp2_avg - exp1_avg

    if overall_change > 0.01:
        indicator = "+"
    elif overall_change < -0.01:
        indicator = "-"
    else:
        indicator = "="

    avg_line = f"{'AVERAGE':<25} {exp1_avg:>10.3f} {exp2_avg:>10.3f}"
    print(f"{avg_line} {indicator}{abs(overall_change):>9.3f}")

    # Recommendation
    print("\n" + "=" * 70)
    improved = sum(1 for v in improvements.values() if v > 0.01)
    degraded = sum(1 for v in improvements.values() if v < -0.01)

    if improved > degraded:
        print(f"RECOMMENDATION: Variant is better ({improved} metrics improved)")
        winner = "variant"
    elif degraded > improved:
        print(f"RECOMMENDATION: Keep baseline ({degraded} metrics degraded)")
        winner = "baseline"
    else:
        print("RECOMMENDATION: No significant difference")
        winner = "tie"

    print("=" * 70)

    return {
        "baseline": exp1_name,
        "variant": exp2_name,
        "baseline_metrics": exp1_metrics,
        "variant_metrics": exp2_metrics,
        "improvements": improvements,
        "winner": winner,
    }


def main(
    list_flag: bool = False,
    show: str | None = None,
    compare: tuple[str, str] | None = None,
    dataset: str = DEFAULT_DATASET,
) -> int:
    """Run experiment comparison.

    Args:
        list_flag: If True, list experiments.
        show: Experiment name to show details.
        compare: Tuple of (baseline, variant) experiment names.
        dataset: Dataset name for listing.

    Returns:
        Exit code (0 for success).
    """
    if not os.getenv("LANGSMITH_API_KEY"):
        logger.error("LANGSMITH_API_KEY not set")
        return 1

    if list_flag:
        list_experiments(dataset)
        return 0

    if show:
        show_experiment(show)
        return 0

    if compare:
        result = compare_experiments(compare[0], compare[1])
        if "error" in result:
            return 1
        return 0

    # Default: list experiments
    list_experiments(dataset)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare LangSmith experiments")
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List experiments for the dataset",
    )
    parser.add_argument(
        "--show",
        "-s",
        metavar="EXPERIMENT",
        help="Show details of an experiment",
    )
    parser.add_argument(
        "--compare",
        "-c",
        nargs=2,
        metavar=("BASELINE", "VARIANT"),
        help="Compare two experiments",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default=DEFAULT_DATASET,
        help=f"Dataset name for listing (default: {DEFAULT_DATASET})",
    )

    args = parser.parse_args()

    compare_tuple = tuple(args.compare) if args.compare else None

    sys.exit(
        main(
            list_flag=args.list,
            show=args.show,
            compare=compare_tuple,
            dataset=args.dataset,
        )
    )
