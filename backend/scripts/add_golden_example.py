#!/usr/bin/env python
"""Add a single golden example to a per-vector LangSmith dataset.

Supports creating examples manually or pulling from a production trace.

Usage:
    # Add manually
    uv run python scripts/add_golden_example.py \
        --dataset graphrag-eval-explanatory \
        --question "What is traceability?" \
        --expected-answer "Traceability is..." \
        --intent explanatory

    # Pull from a LangSmith trace
    uv run python scripts/add_golden_example.py \
        --from-trace abc123-run-id \
        --dataset graphrag-eval-explanatory \
        --expected-answer "The corrected answer"

    # Dry run
    uv run python scripts/add_golden_example.py \
        --dataset graphrag-eval-explanatory \
        --question "What is V&V?" \
        --expected-answer "Verification and validation..." \
        --dry-run

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


def _pull_trace(run_id: str) -> dict[str, Any]:
    """Pull inputs/outputs from a LangSmith trace.

    Args:
        run_id: LangSmith run ID.

    Returns:
        Dict with 'question', 'answer', and 'metadata' keys.
    """
    from langsmith import Client

    client = Client()
    run = client.read_run(run_id)

    inputs = run.inputs or {}
    outputs = run.outputs or {}

    return {
        "question": inputs.get("question", ""),
        "answer": outputs.get("answer", outputs.get("output", "")),
        "metadata": {
            "source_run_id": run_id,
            "source_project": run.session_name if hasattr(run, "session_name") else "",
        },
    }


def add_example(
    *,
    dataset_name: str,
    question: str,
    expected_answer: str,
    intent: str = "explanatory",
    expected_entities: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> str | None:
    """Add a single example to a LangSmith dataset.

    Args:
        dataset_name: Target dataset name.
        question: The evaluation question.
        expected_answer: Ground-truth answer.
        intent: Query intent.
        expected_entities: Expected entities in context.
        metadata: Additional metadata.
        dry_run: If True, only log what would happen.

    Returns:
        Example ID if created, None otherwise.
    """
    inputs: dict[str, Any] = {"question": question}
    outputs: dict[str, Any] = {
        "expected_answer": expected_answer,
        "intent": intent,
    }
    if expected_entities:
        outputs["expected_entities"] = expected_entities

    meta = metadata or {}
    meta.setdefault("source", "manual")

    logger.info("Dataset: %s", dataset_name)
    logger.info("Question: %s", question[:80])
    logger.info("Expected answer: %s...", expected_answer[:80])
    logger.info("Intent: %s", intent)
    if expected_entities:
        logger.info("Entities: %s", expected_entities)

    if dry_run:
        logger.info("[DRY RUN] Would add example to '%s'", dataset_name)
        return None

    from langsmith import Client

    client = Client()

    # Verify dataset exists
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception as e:
        logger.error("Dataset '%s' not found: %s", dataset_name, e)
        logger.error("Run migrate_golden_datasets.py first.")
        return None

    example = client.create_example(
        inputs=inputs,
        outputs=outputs,
        dataset_id=dataset.id,
        metadata=meta,
    )

    logger.info("Added example (id=%s) to '%s'", example.id, dataset_name)
    return str(example.id)


def main() -> int:
    """Main entry point."""
    if not os.getenv("LANGSMITH_API_KEY"):
        logger.error("LANGSMITH_API_KEY not set")
        return 1

    from requirements_graphrag_api.evaluation.constants import (
        ALL_VECTOR_DATASETS,
    )

    parser = argparse.ArgumentParser(
        description="Add a golden example to a per-vector LangSmith dataset",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        choices=list(ALL_VECTOR_DATASETS),
        help="Target dataset name",
    )
    parser.add_argument(
        "--question",
        "-q",
        help="The evaluation question",
    )
    parser.add_argument(
        "--expected-answer",
        "-a",
        help="Ground-truth expected answer",
    )
    parser.add_argument(
        "--intent",
        choices=["explanatory", "structured", "conversational"],
        default="explanatory",
        help="Query intent (default: explanatory)",
    )
    parser.add_argument(
        "--entities",
        nargs="*",
        help="Expected entities in context",
    )
    parser.add_argument(
        "--from-trace",
        help="Pull question/answer from a LangSmith trace run ID",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making changes",
    )

    args = parser.parse_args()

    # Get question/answer from trace or args
    question = args.question
    expected_answer = args.expected_answer
    metadata: dict[str, Any] = {}

    if args.from_trace:
        logger.info("Pulling trace: %s", args.from_trace)
        trace_data = _pull_trace(args.from_trace)
        question = question or trace_data["question"]
        # expected_answer from args overrides trace (allows correction)
        if not expected_answer:
            expected_answer = trace_data["answer"]
        metadata = trace_data["metadata"]
        metadata["source"] = "trace"

    if not question or not expected_answer:
        logger.error("--question and --expected-answer are required (or use --from-trace)")
        return 1

    result = add_example(
        dataset_name=args.dataset,
        question=question,
        expected_answer=expected_answer,
        intent=args.intent,
        expected_entities=args.entities,
        metadata=metadata,
        dry_run=args.dry_run,
    )

    return 0 if result is not None or args.dry_run else 1


if __name__ == "__main__":
    sys.exit(main())
