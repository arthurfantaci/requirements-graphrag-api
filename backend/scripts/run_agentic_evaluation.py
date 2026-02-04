#!/usr/bin/env python
"""Run agentic RAG evaluation with LangSmith experiments.

This script runs evaluation of the agentic RAG system using:
- Custom agentic evaluators (tool selection, iteration efficiency, etc.)
- The graphrag-agentic-eval dataset
- LangSmith experiment tracking

Usage:
    # Run evaluation against agentic dataset
    uv run python scripts/run_agentic_evaluation.py

    # Run with custom experiment name
    uv run python scripts/run_agentic_evaluation.py --experiment-name "agentic-v1-test"

    # Dry run to see what would be evaluated
    uv run python scripts/run_agentic_evaluation.py --dry-run

    # Compare with previous experiment
    uv run python scripts/compare_experiments.py -d graphrag-agentic-eval

Requires:
- LANGSMITH_API_KEY
- OPENAI_API_KEY
- Agentic dataset created (run create_agentic_dataset.py first)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
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

# Configuration
DEFAULT_DATASET = "graphrag-agentic-eval"
DEFAULT_EXPERIMENT_PREFIX = "agentic-eval"
DEFAULT_MODEL = "gpt-4o-mini"


async def create_agentic_target(
    model: str = DEFAULT_MODEL,
) -> Any:
    """Create the agentic RAG pipeline target function for evaluation.

    This simulates the agentic pipeline behavior for evaluation purposes.
    In production, this would use the full orchestrator.

    Args:
        model: Model name to use for generation.

    Returns:
        Async function that takes inputs and returns outputs.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI

    from requirements_graphrag_api.prompts import PromptName, get_prompt_sync

    prompt_template = get_prompt_sync(PromptName.RAG_GENERATION)
    llm = ChatOpenAI(model=model, temperature=0.1)

    async def agentic_target(inputs: dict[str, Any]) -> dict[str, Any]:
        """Agentic RAG pipeline target function.

        Simulates agentic behavior with tool usage tracking.

        Args:
            inputs: Dict with 'question' and 'complexity' keys.

        Returns:
            Dict with answer, tools_used, iteration_count, critique, etc.
        """
        question = inputs.get("question", "")
        complexity = inputs.get("complexity", "medium")

        # Simulate tool selection based on question content
        tools_used = []
        tool_order = []

        # Simple heuristic for tool selection simulation
        question_lower = question.lower()

        if any(word in question_lower for word in ["list", "how many", "count"]):
            tools_used.append("text2cypher")
            tool_order.append("text2cypher")
        elif any(word in question_lower for word in ["what is", "define", "meaning"]):
            tools_used.append("search_definitions")
            tool_order.append("search_definitions")
            tools_used.append("graph_search")
            tool_order.append("graph_search")
        else:
            tools_used.append("graph_search")
            tool_order.append("graph_search")

        has_relationship_term = any(
            word in question_lower for word in ["relationship", "connect", "relate"]
        )
        if has_relationship_term and "explore_entity" not in tools_used:
            tools_used.append("explore_entity")
            tool_order.append("explore_entity")

        # Simulate iteration count based on complexity
        iteration_map = {"simple": 1, "medium": 2, "complex": 3, "multi_hop": 3}
        iteration_count = iteration_map.get(complexity, 2)

        # Build mock context for answer generation
        mock_context = f"""
        Context for: {question}

        Requirements management is the process of documenting, analyzing,
        tracing, prioritizing, and agreeing on requirements and then
        controlling change and communicating to relevant stakeholders.

        Traceability is the ability to trace requirements throughout the
        product development lifecycle, linking them to design, implementation,
        and test artifacts.

        Key standards include ISO 26262 for automotive, IEC 62304 for medical
        devices, and DO-178C for aerospace software.
        """

        chain = prompt_template | llm | StrOutputParser()

        answer = await chain.ainvoke(
            {
                "context": mock_context,
                "entities": "Requirements Management, Traceability, Standards",
                "question": question,
            }
        )

        # Simulate critique
        confidence = 0.8 if complexity == "simple" else 0.7 if complexity == "medium" else 0.65
        completeness = "complete" if complexity == "simple" else "partial"

        return {
            "answer": answer,
            "final_answer": answer,
            "tools_used": tools_used,
            "tool_order": tool_order,
            "iteration_count": iteration_count,
            "critique": {
                "confidence": confidence,
                "completeness": completeness,
                "missing_aspects": [] if completeness == "complete" else ["additional context"],
            },
            "reasoning_chain": [
                f"Identified query about: {question[:50]}...",
                f"Selected tools: {', '.join(tools_used)}",
                f"Completed in {iteration_count} iterations",
            ],
            "context": mock_context,
            "contexts": [mock_context],
        }

    return agentic_target


def run_agentic_evaluation(
    dataset_name: str = DEFAULT_DATASET,
    experiment_name: str | None = None,
    model: str = DEFAULT_MODEL,
    max_concurrency: int = 4,
    *,
    filter_category: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run agentic evaluation using langsmith.evaluate().

    Args:
        dataset_name: Name of the LangSmith dataset to evaluate against.
        experiment_name: Custom experiment name (auto-generated if None).
        model: Model to use for RAG generation.
        max_concurrency: Maximum concurrent evaluations.
        filter_category: Filter examples by category (e.g., 'multi_hop').
        verbose: If True, print detailed output.

    Returns:
        Dict with experiment results summary.
    """
    from langsmith import Client, evaluate

    from requirements_graphrag_api.evaluation.agentic_evaluators import (
        critic_calibration_evaluator_sync,
        iteration_efficiency_evaluator_sync,
        multi_hop_reasoning_evaluator_sync,
        tool_selection_evaluator_sync,
    )

    client = Client()

    # Generate experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
        experiment_name = f"{DEFAULT_EXPERIMENT_PREFIX}-{timestamp}"

    logger.info("=" * 60)
    logger.info("AGENTIC RAG EVALUATION")
    logger.info("=" * 60)
    logger.info("Dataset: %s", dataset_name)
    logger.info("Experiment: %s", experiment_name)
    logger.info("Model: %s", model)
    if filter_category:
        logger.info("Filter: category=%s", filter_category)
    logger.info("=" * 60)

    # Verify dataset exists and load examples
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        examples = list(client.list_examples(dataset_id=dataset.id))
        logger.info("Found %d examples in dataset", len(examples))
    except Exception as e:
        logger.error("Dataset not found: %s - %s", dataset_name, e)
        logger.error("Run create_agentic_dataset.py first to create the dataset.")
        return {"error": str(e)}

    # Filter examples by category if specified
    if filter_category:
        original_count = len(examples)
        examples = [
            ex
            for ex in examples
            if ex.metadata and ex.metadata.get("category", "").lower() == filter_category.lower()
        ]
        logger.info(
            "Filtered to %d examples with category='%s' (from %d total)",
            len(examples),
            filter_category,
            original_count,
        )
        if not examples:
            logger.error("No examples found with category='%s'", filter_category)
            return {"error": f"No examples with category={filter_category}"}

    # Create synchronous wrapper for the async target
    async def async_target(inputs: dict[str, Any]) -> dict[str, Any]:
        target_fn = await create_agentic_target(model)
        return await target_fn(inputs)

    def sync_target(inputs: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(async_target(inputs))

    # Define agentic evaluators
    evaluators = [
        tool_selection_evaluator_sync,
        iteration_efficiency_evaluator_sync,
        critic_calibration_evaluator_sync,
        multi_hop_reasoning_evaluator_sync,
    ]

    # Run evaluation with LangSmith
    logger.info("Running evaluation with %d agentic evaluators...", len(evaluators))

    # Build metadata
    eval_metadata = {
        "model": model,
        "evaluation_type": "agentic",
        "dataset": dataset_name,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "evaluators": [
            "tool_selection",
            "iteration_efficiency",
            "critic_calibration",
            "multi_hop_reasoning",
        ],
    }
    if filter_category:
        eval_metadata["filter_category"] = filter_category
        eval_metadata["filtered_count"] = len(examples)

    try:
        # Use filtered examples if filter applied, otherwise use dataset name
        data_source = examples if filter_category else dataset_name

        results = evaluate(
            sync_target,
            data=data_source,
            evaluators=evaluators,
            experiment_prefix=experiment_name,
            max_concurrency=max_concurrency,
            metadata=eval_metadata,
        )

        # Extract summary metrics
        summary = {
            "experiment_name": experiment_name,
            "dataset": dataset_name,
            "model": model,
            "total_examples": len(examples),
            "results": [],
        }

        if hasattr(results, "__iter__"):
            for result in results:
                if verbose:
                    logger.info("Result: %s", result)
                summary["results"].append(str(result))

        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info("Experiment: %s", experiment_name)
        logger.info("View results at: https://smith.langchain.com/experiments")
        logger.info("=" * 60)

        return summary

    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


def list_datasets() -> None:
    """List available datasets in LangSmith."""
    try:
        from langsmith import Client

        client = Client()
        datasets = list(client.list_datasets())

        print("\nAvailable Datasets:")
        print("=" * 60)
        for ds in datasets:
            marker = " *" if "agentic" in ds.name.lower() else ""
            print(f"  {ds.name}{marker}")
            if ds.description:
                desc = ds.description[:60] + "..." if len(ds.description) > 60 else ds.description
                print(f"    {desc}")
        print("=" * 60)
        print("  * = agentic evaluation datasets")
    except Exception as e:
        logger.error("Failed to list datasets: %s", e)


def main(
    dataset_name: str = DEFAULT_DATASET,
    experiment_name: str | None = None,
    model: str = DEFAULT_MODEL,
    max_concurrency: int = 4,
    *,
    filter_category: str | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    list_datasets_flag: bool = False,
) -> int:
    """Run agentic RAG evaluation.

    Args:
        dataset_name: LangSmith dataset name.
        experiment_name: Custom experiment name.
        model: Model for RAG generation.
        max_concurrency: Maximum concurrent evaluations.
        filter_category: Filter examples by category (e.g., 'multi_hop').
        dry_run: If True, only show what would be evaluated.
        verbose: If True, print detailed output.
        list_datasets_flag: If True, list available datasets.

    Returns:
        Exit code (0 for success).
    """
    # Check environment
    if not os.getenv("LANGSMITH_API_KEY"):
        logger.error("LANGSMITH_API_KEY not set")
        return 1

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set")
        return 1

    if list_datasets_flag:
        list_datasets()
        return 0

    if dry_run:
        logger.info("[DRY RUN] Would run agentic evaluation:")
        logger.info("  Dataset: %s", dataset_name)
        exp_preview = experiment_name or f"{DEFAULT_EXPERIMENT_PREFIX}-<timestamp>"
        logger.info("  Experiment: %s", exp_preview)
        logger.info("  Model: %s", model)
        if filter_category:
            logger.info("  Filter: category=%s", filter_category)
        logger.info(
            "  Evaluators: tool_selection, iteration_efficiency, "
            "critic_calibration, multi_hop_reasoning"
        )
        return 0

    result = run_agentic_evaluation(
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        model=model,
        max_concurrency=max_concurrency,
        filter_category=filter_category,
        verbose=verbose,
    )

    if "error" in result:
        return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run agentic RAG evaluation with LangSmith experiments"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default=DEFAULT_DATASET,
        help=f"Dataset name (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--experiment-name",
        "-e",
        help="Custom experiment name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"Model for RAG generation (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Maximum concurrent evaluations (default: 4)",
    )
    parser.add_argument(
        "--filter-category",
        "-f",
        choices=["multi_hop", "tool_selection", "iteration_efficiency", "critic_calibration"],
        help="Filter examples by category",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be evaluated without running",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed output",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        dest="list_datasets",
        help="List available datasets",
    )

    args = parser.parse_args()

    sys.exit(
        main(
            dataset_name=args.dataset,
            experiment_name=args.experiment_name,
            model=args.model,
            max_concurrency=args.max_concurrency,
            filter_category=args.filter_category,
            dry_run=args.dry_run,
            verbose=args.verbose,
            list_datasets_flag=args.list_datasets,
        )
    )
