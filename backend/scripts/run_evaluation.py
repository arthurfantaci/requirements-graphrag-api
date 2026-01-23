#!/usr/bin/env python
"""Run RAG evaluation with LangSmith tracing.

Usage:
    uv run python scripts/run_evaluation.py [--samples N]

Requires LANGSMITH_TRACING=true in .env for LangSmith integration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv(override=True)  # Override any cached env vars

from requirements_graphrag_api.config import get_config  # noqa: E402
from requirements_graphrag_api.core.retrieval import create_vector_retriever  # noqa: E402
from requirements_graphrag_api.evaluation import evaluate_rag_pipeline  # noqa: E402
from requirements_graphrag_api.neo4j_client import create_driver  # noqa: E402
from requirements_graphrag_api.observability import configure_tracing  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def save_results(output_file: Path, report_dict: dict) -> None:
    """Save evaluation results to JSON file (sync helper for async context)."""
    output_file.write_text(json.dumps(report_dict, indent=2))


async def main(max_samples: int | None = None) -> None:
    """Run the evaluation pipeline."""
    logger.info("Loading configuration...")
    config = get_config()

    # Configure LangSmith tracing BEFORE any LangChain/LangGraph calls
    tracing_enabled = configure_tracing(config)
    if tracing_enabled:
        logger.info("LangSmith tracing enabled for project: %s", config.langsmith_project)
    else:
        logger.warning("LangSmith tracing NOT enabled - traces will not be recorded")

    logger.info("Connecting to Neo4j at %s...", config.neo4j_uri)
    driver = create_driver(config)

    try:
        # Verify connection
        driver.verify_connectivity()
        logger.info("Neo4j connection verified")

        # Create retriever
        logger.info("Creating vector retriever...")
        retriever = create_vector_retriever(driver, config)

        # Run evaluation
        logger.info("Starting evaluation (max_samples=%s)...", max_samples)
        report = await evaluate_rag_pipeline(
            config,
            retriever,
            driver,
            max_samples=max_samples,
        )

        # Print results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total Samples: {report.total_samples}")
        print(f"Timestamp: {report.timestamp}")
        print("\nAggregate Metrics:")
        for metric, value in report.aggregate_metrics.items():
            print(f"  {metric}: {value:.4f}")

        print("\nIndividual Results:")
        for i, result in enumerate(report.results, 1):
            print(f"\n--- Sample {i} ---")
            print(f"Question: {result.sample.question[:60]}...")
            print(f"Latency: {result.latency_ms:.1f}ms")
            print(f"Metrics: {result.metrics.to_dict()}")

        # Save to file using sync helper
        output_file = Path("evaluation_results.json")
        save_results(output_file, report.to_dict())
        logger.info("Results saved to %s", output_file)

        print("\n" + "=" * 60)
        print("View traces in LangSmith: https://smith.langchain.com")
        print(f"Project: {config.langsmith_project or 'requirements-graphrag'}")
        print("=" * 60)

    finally:
        driver.close()
        logger.info("Neo4j connection closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Maximum number of samples to evaluate (default: 3)",
    )
    args = parser.parse_args()

    asyncio.run(main(max_samples=args.samples))
