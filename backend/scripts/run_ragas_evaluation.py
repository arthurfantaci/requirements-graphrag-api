#!/usr/bin/env python
"""Run RAGAS evaluation with LangSmith experiments.

This script demonstrates comprehensive LangSmith integration:
- Running evaluations with langsmith.evaluate()
- Creating experiments with metadata
- Using custom RAGAS evaluators
- Experiment naming conventions
- Filtering examples by intent for targeted evaluation

Usage:
    # Run evaluation against golden dataset
    uv run python scripts/run_ragas_evaluation.py

    # Run with custom experiment name
    uv run python scripts/run_ragas_evaluation.py --experiment-name "ragas-v2-test"

    # Run with specific model
    uv run python scripts/run_ragas_evaluation.py --model gpt-4o

    # Filter to only explanatory queries (recommended for RAGAS)
    uv run python scripts/run_ragas_evaluation.py --filter-intent explanatory

    # Dry run to see what would be evaluated
    uv run python scripts/run_ragas_evaluation.py --dry-run

Requires:
- LANGSMITH_API_KEY
- OPENAI_API_KEY
- Golden dataset created in LangSmith (run create_golden_dataset.py first)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

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
DEFAULT_DATASET = "graphrag-rag-golden"
DEFAULT_EXPERIMENT_PREFIX = "ragas-eval"
DEFAULT_MODEL = "gpt-4o-mini"


async def create_rag_target(
    model: str = DEFAULT_MODEL,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create the RAG pipeline target function for evaluation.

    This function will be called for each example in the dataset.
    It must accept the same input structure as the dataset examples.

    Args:
        model: Model name to use for generation.

    Returns:
        Async function that takes inputs and returns outputs.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI

    from requirements_graphrag_api.prompts import PromptName, get_prompt

    prompt_template = await get_prompt(PromptName.RAG_GENERATION)
    llm = ChatOpenAI(model=model, temperature=0.1)

    async def rag_target(inputs: dict[str, Any]) -> dict[str, Any]:
        """RAG pipeline target function.

        For evaluation, we simulate the RAG pipeline with mock context.
        In production, this would call the full retrieval pipeline.

        Args:
            inputs: Dict with 'question' key.

        Returns:
            Dict with 'answer' and 'contexts' keys.
        """
        question = inputs.get("question", "")

        # For evaluation, we use a simplified context
        # In production, this would call graph_enriched_search
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

        return {
            "answer": answer,
            "context": mock_context,
            "contexts": [mock_context],
        }

    return rag_target


def run_evaluation(
    dataset_name: str = DEFAULT_DATASET,
    experiment_name: str | None = None,
    model: str = DEFAULT_MODEL,
    max_concurrency: int = 4,
    *,
    filter_intent: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run RAGAS evaluation using langsmith.evaluate().

    Args:
        dataset_name: Name of the LangSmith dataset to evaluate against.
        experiment_name: Custom experiment name (auto-generated if None).
        model: Model to use for RAG generation.
        max_concurrency: Maximum concurrent evaluations.
        filter_intent: Filter examples by intent (e.g., 'explanatory', 'structured').
        verbose: If True, print detailed output.

    Returns:
        Dict with experiment results summary.
    """
    from langsmith import Client, evaluate

    from requirements_graphrag_api.evaluation.ragas_evaluators import (
        answer_relevancy_evaluator,
        context_precision_evaluator,
        context_recall_evaluator,
        faithfulness_evaluator,
    )

    client = Client()

    # Generate experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
        experiment_name = f"{DEFAULT_EXPERIMENT_PREFIX}-{timestamp}"

    logger.info("=" * 60)
    logger.info("RAGAS EVALUATION")
    logger.info("=" * 60)
    logger.info("Dataset: %s", dataset_name)
    logger.info("Experiment: %s", experiment_name)
    logger.info("Model: %s", model)
    if filter_intent:
        logger.info("Filter: intent=%s", filter_intent)
    logger.info("=" * 60)

    # Verify dataset exists and load examples
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        examples = list(client.list_examples(dataset_id=dataset.id))
        logger.info("Found %d examples in dataset", len(examples))
    except Exception as e:
        logger.error("Dataset not found: %s - %s", dataset_name, e)
        logger.error("Run create_golden_dataset.py first to create the dataset.")
        return {"error": str(e)}

    # Filter examples by intent if specified
    if filter_intent:
        original_count = len(examples)
        examples = [
            ex
            for ex in examples
            if ex.outputs and ex.outputs.get("intent", "").lower() == filter_intent.lower()
        ]
        logger.info(
            "Filtered to %d examples with intent='%s' (from %d total)",
            len(examples),
            filter_intent,
            original_count,
        )
        if not examples:
            logger.error("No examples found with intent='%s'", filter_intent)
            return {"error": f"No examples with intent={filter_intent}"}

    # Create synchronous wrapper for the async target
    async def async_target(inputs: dict[str, Any]) -> dict[str, Any]:
        target_fn = await create_rag_target(model)
        return await target_fn(inputs)

    def sync_target(inputs: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(async_target(inputs))

    # Define evaluators
    evaluators = [
        faithfulness_evaluator,
        answer_relevancy_evaluator,
        context_precision_evaluator,
        context_recall_evaluator,
    ]

    # Run evaluation with LangSmith
    logger.info("Running evaluation with %d evaluators...", len(evaluators))

    # Build metadata
    eval_metadata = {
        "model": model,
        "evaluation_type": "ragas",
        "dataset": dataset_name,
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }
    if filter_intent:
        eval_metadata["filter_intent"] = filter_intent
        eval_metadata["filtered_count"] = len(examples)

    try:
        # Use filtered examples if filter applied, otherwise use dataset name
        data_source = examples if filter_intent else dataset_name

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
            print(f"  {ds.name}")
            if ds.description:
                print(f"    {ds.description[:60]}...")
        print("=" * 60)
    except Exception as e:
        logger.error("Failed to list datasets: %s", e)


def main(
    dataset_name: str = DEFAULT_DATASET,
    experiment_name: str | None = None,
    model: str = DEFAULT_MODEL,
    max_concurrency: int = 4,
    *,
    filter_intent: str | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    list_datasets_flag: bool = False,
) -> int:
    """Run RAGAS evaluation.

    Args:
        dataset_name: LangSmith dataset name.
        experiment_name: Custom experiment name.
        model: Model for RAG generation.
        max_concurrency: Maximum concurrent evaluations.
        filter_intent: Filter examples by intent (e.g., 'explanatory').
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
        logger.info("[DRY RUN] Would run RAGAS evaluation:")
        logger.info("  Dataset: %s", dataset_name)
        exp_preview = experiment_name or f"{DEFAULT_EXPERIMENT_PREFIX}-<timestamp>"
        logger.info("  Experiment: %s", exp_preview)
        logger.info("  Model: %s", model)
        if filter_intent:
            logger.info("  Filter: intent=%s", filter_intent)
        logger.info("  Evaluators: faithfulness, answer_relevancy, context_precision, recall")
        return 0

    result = run_evaluation(
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        model=model,
        max_concurrency=max_concurrency,
        filter_intent=filter_intent,
        verbose=verbose,
    )

    if "error" in result:
        return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation with LangSmith experiments")
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
        "--filter-intent",
        "-f",
        choices=["explanatory", "structured"],
        help="Filter examples by intent (recommended: 'explanatory' for RAGAS)",
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
            filter_intent=args.filter_intent,
            dry_run=args.dry_run,
            verbose=args.verbose,
            list_datasets_flag=args.list_datasets,
        )
    )
