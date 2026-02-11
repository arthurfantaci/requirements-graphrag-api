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
- NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
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

    from neo4j import Driver
    from neo4j_graphrag.retrievers import VectorRetriever

    from requirements_graphrag_api.config import AppConfig

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


def _setup_infrastructure() -> tuple[AppConfig, Driver, VectorRetriever]:
    """Set up Neo4j driver and vector retriever for real retrieval.

    Returns:
        Tuple of (config, driver, retriever).
    """
    from requirements_graphrag_api.config import get_config
    from requirements_graphrag_api.core.retrieval import create_vector_retriever
    from requirements_graphrag_api.neo4j_client import create_driver

    config = get_config()
    driver = create_driver(config)
    retriever = create_vector_retriever(driver, config)
    return config, driver, retriever


async def create_rag_target(
    driver: Driver,
    retriever: VectorRetriever,
    model: str = DEFAULT_MODEL,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create the RAG pipeline target function for evaluation.

    Uses real graph_enriched_search for context retrieval.

    Args:
        driver: Neo4j driver for graph queries.
        retriever: Vector retriever for semantic search.
        model: Model name to use for generation.

    Returns:
        Async function that takes inputs and returns outputs.
    """
    from langchain_openai import ChatOpenAI

    from requirements_graphrag_api.core.retrieval import graph_enriched_search
    from requirements_graphrag_api.prompts import PromptName, get_prompt

    prompt_template = await get_prompt(PromptName.RAG_GENERATION)
    llm = ChatOpenAI(model=model, temperature=0.1)

    async def rag_target(inputs: dict[str, Any]) -> dict[str, Any]:
        """RAG pipeline target function with real retrieval.

        Args:
            inputs: Dict with 'question' key.

        Returns:
            Dict with 'answer', 'contexts', and 'intent' keys.
        """
        question = inputs.get("question", "")

        results = await graph_enriched_search(retriever, driver, question, limit=6)

        contexts = [r["content"] for r in results]
        context = "\n\n".join(contexts)

        chain = prompt_template | llm
        response = await chain.ainvoke(
            {
                "context": context,
                "entities": "",
                "question": question,
            }
        )

        return {
            "answer": response.content,
            "contexts": contexts,
            "intent": "explanatory",
        }

    return rag_target


def run_evaluation(
    dataset_name: str = DEFAULT_DATASET,
    experiment_name: str | None = None,
    model: str = DEFAULT_MODEL,
    max_concurrency: int = 2,
    *,
    filter_intent: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run RAGAS evaluation using langsmith.aevaluate().

    Args:
        dataset_name: Name of the LangSmith dataset to evaluate against.
        experiment_name: Custom experiment name (auto-generated if None).
        model: Model to use for RAG generation.
        max_concurrency: Maximum concurrent evaluations (default 2 to avoid Neo4j pool exhaustion).
        filter_intent: Filter examples by intent (e.g., 'explanatory', 'structured').
        verbose: If True, print detailed output.

    Returns:
        Dict with experiment results summary.
    """
    from langsmith import Client, aevaluate

    from requirements_graphrag_api.evaluation.ragas_evaluators import (
        answer_correctness_evaluator,
        answer_relevancy_evaluator,
        answer_semantic_similarity_evaluator,
        context_entity_recall_evaluator,
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

    # Use aevaluate() — the async-native evaluation runner (langsmith>=0.3.13).
    # It handles async targets and async evaluators on a single event loop,
    # avoiding the ThreadPoolExecutor "no current event loop" issue that
    # plagues the sync evaluate() with async evaluators.
    evaluators = [
        faithfulness_evaluator,
        answer_relevancy_evaluator,
        answer_correctness_evaluator,
        answer_semantic_similarity_evaluator,
        context_precision_evaluator,
        context_recall_evaluator,
        context_entity_recall_evaluator,
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

    # Set up Neo4j infrastructure
    _config, driver, retriever = _setup_infrastructure()

    async def _run() -> dict[str, Any]:
        # Create the async target with real retrieval
        target_fn = await create_rag_target(driver, retriever, model)

        # Use filtered examples if filter applied, otherwise use dataset name
        data_source = examples if filter_intent else dataset_name

        results = await aevaluate(
            target_fn,
            data=data_source,
            evaluators=evaluators,
            experiment_prefix=experiment_name,
            max_concurrency=max_concurrency,
            metadata=eval_metadata,
        )

        # Extract summary metrics
        summary: dict[str, Any] = {
            "experiment_name": experiment_name,
            "dataset": dataset_name,
            "model": model,
            "total_examples": len(examples),
            "results": [],
        }

        async for result in results:
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

    try:
        return asyncio.run(_run())
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        return {"error": str(e)}
    finally:
        driver.close()


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
    max_concurrency: int = 2,
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

    if not os.getenv("NEO4J_URI"):
        logger.error("NEO4J_URI not set (required for real retrieval)")
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
        logger.info(
            "  Evaluators: faithfulness, answer_relevancy, answer_correctness, "
            "answer_semantic_similarity, context_precision, context_recall, "
            "context_entity_recall"
        )
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
        default=2,
        help="Maximum concurrent evaluations (default: 2, limited by Neo4j pool)",
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
