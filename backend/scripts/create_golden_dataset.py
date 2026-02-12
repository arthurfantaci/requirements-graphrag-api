#!/usr/bin/env python
"""Manage the golden dataset in LangSmith (Hub-First workflow).

This script provides bidirectional sync between the local fallback
and the LangSmith dataset:

- push:   Local → LangSmith (seed or reset the remote dataset)
- export: LangSmith → Local (sync remote edits back to version control)

The source of truth is LangSmith. Users edit examples in the UI,
and this script syncs changes back to the local fallback file.

Usage:
    # Push local examples to LangSmith (initial seed or reset)
    uv run python scripts/create_golden_dataset.py --push

    # Export LangSmith dataset to local fallback (for version control)
    uv run python scripts/create_golden_dataset.py --export

    # Dry run (show what would happen)
    uv run python scripts/create_golden_dataset.py --push --dry-run

    # List datasets
    uv run python scripts/create_golden_dataset.py --list

Requires LANGSMITH_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from requirements_graphrag_api.evaluation.golden_dataset import GoldenExample

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def push_to_langsmith(*, dry_run: bool = False) -> str | None:
    """Push local golden examples to LangSmith.

    Creates the dataset if it doesn't exist, or warns if it does.
    Use --force to delete and recreate.

    Args:
        dry_run: If True, only log what would happen.

    Returns:
        Dataset ID if created, None otherwise.
    """
    from requirements_graphrag_api.evaluation.golden_dataset import (
        DATASET_NAME,
        GOLDEN_EXAMPLES,
    )

    if dry_run:
        logger.info("[DRY RUN] Would push %d examples to '%s'", len(GOLDEN_EXAMPLES), DATASET_NAME)
        _log_distribution(GOLDEN_EXAMPLES)
        return None

    from langsmith import Client

    client = Client()

    # Check if dataset already exists
    try:
        existing = client.read_dataset(dataset_name=DATASET_NAME)
        examples = list(client.list_examples(dataset_id=existing.id))
        logger.info(
            "Dataset '%s' already exists with %d examples (id=%s).",
            DATASET_NAME,
            len(examples),
            existing.id,
        )
        logger.info("To reset, delete the dataset in LangSmith UI first, then re-run --push.")
        return str(existing.id)
    except Exception:
        logger.debug("Dataset '%s' not found, creating it.", DATASET_NAME)

    # Create dataset
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Golden evaluation dataset for GraphRAG RAG pipeline.\n\n"
            "Contains curated examples with expected answers, entities, "
            "intent classification, difficulty ratings, and must-pass flags.\n\n"
            "Source of truth: LangSmith UI. Edit examples here, then run\n"
            "  uv run python scripts/create_golden_dataset.py --export\n"
            "to sync changes back to the local fallback."
        ),
    )
    logger.info("Created dataset '%s' (id=%s)", DATASET_NAME, dataset.id)

    # Add examples
    successful = 0
    for example in GOLDEN_EXAMPLES:
        ls_format = example.to_langsmith()
        try:
            client.create_example(
                inputs=ls_format["inputs"],
                outputs=ls_format["outputs"],
                dataset_id=dataset.id,
                metadata=ls_format["metadata"],
            )
            successful += 1
        except Exception as e:
            logger.error("Failed to add example %s: %s", example.id, e)

    logger.info("Added %d/%d examples to dataset", successful, len(GOLDEN_EXAMPLES))
    return str(dataset.id)


def export_from_langsmith(*, dry_run: bool = False) -> int:
    """Export LangSmith dataset to local fallback file.

    Pulls all examples from LangSmith and regenerates the local
    GOLDEN_EXAMPLES tuple in golden_dataset.py.

    Args:
        dry_run: If True, only log what would happen.

    Returns:
        Number of examples exported.
    """
    from langsmith import Client

    from requirements_graphrag_api.evaluation.golden_dataset import (
        DATASET_NAME,
        GoldenExample,
    )

    client = Client()
    dataset = client.read_dataset(dataset_name=DATASET_NAME)
    raw_examples = list(client.list_examples(dataset_id=dataset.id))

    examples = [
        GoldenExample.from_langsmith(
            inputs=ex.inputs or {},
            outputs=ex.outputs or {},
            metadata=ex.metadata or {},
        )
        for ex in raw_examples
    ]

    # Sort by ID for stable ordering
    examples.sort(key=lambda ex: ex.id)

    logger.info("Pulled %d examples from LangSmith '%s'", len(examples), DATASET_NAME)

    if dry_run:
        logger.info("[DRY RUN] Would update local fallback with %d examples", len(examples))
        for ex in examples:
            logger.info("  %s: %s", ex.id, ex.question[:60])
        return len(examples)

    # Generate the Python code for the local fallback
    target_file = (
        Path(__file__).parent.parent
        / "src"
        / "requirements_graphrag_api"
        / "evaluation"
        / "golden_dataset.py"
    )

    _update_local_fallback(target_file, examples)
    logger.info("Updated local fallback: %s (%d examples)", target_file.name, len(examples))
    return len(examples)


def _update_local_fallback(target_file: Path, examples: list[GoldenExample]) -> None:
    """Regenerate the GOLDEN_EXAMPLES section of golden_dataset.py.

    Replaces content between the LOCAL FALLBACK EXAMPLES markers.
    """
    content = target_file.read_text()

    start_marker = "GOLDEN_EXAMPLES: tuple[GoldenExample, ...] = ("
    end_marker = "\n)\n\n\n# ======"

    start_idx = content.index(start_marker)
    end_idx = content.index(end_marker, start_idx)

    # Generate new examples code
    examples_code = _generate_examples_code(examples)

    new_content = (
        content[: start_idx + len(start_marker)] + "\n" + examples_code + content[end_idx:]
    )

    target_file.write_text(new_content)


_CATEGORY_ORDER = [
    "definition",
    "concept",
    "process",
    "standard",
    "comparison",
    "tools",
    "data_query",
]


def _generate_examples_code(examples: list[GoldenExample]) -> str:
    """Generate Python code for GOLDEN_EXAMPLES entries."""
    lines: list[str] = []

    # Group by category
    categories: dict[str, list[GoldenExample]] = {}
    for ex in examples:
        categories.setdefault(ex.category, []).append(ex)

    # Sort categories by canonical order for stable git diffs
    sorted_cats = sorted(
        categories.items(),
        key=lambda item: (
            _CATEGORY_ORDER.index(item[0]) if item[0] in _CATEGORY_ORDER else 999,
            item[0],
        ),
    )

    for category, cat_examples in sorted_cats:
        lines.append(f"    # {'─' * 73}")
        lines.append(f"    # {category.upper()}")
        lines.append(f"    # {'─' * 73}")

        for ex in cat_examples:
            lines.append("    GoldenExample(")
            lines.append(f'        id="{ex.id}",')

            # Handle multi-line questions
            if len(ex.question) > 60:
                lines.append("        question=(")
                lines.append(f'            "{ex.question}"')
                lines.append("        ),")
            else:
                lines.append(f'        question="{ex.question}",')

            # Expected answer (always multi-line)
            lines.append("        expected_answer=(")
            _wrap_string(lines, ex.expected_answer, indent=12)
            lines.append("        ),")

            if ex.expected_entities:
                entities = ", ".join(f'"{e}"' for e in ex.expected_entities)
                lines.append(f"        expected_entities=[{entities}],")

            if ex.expected_standards:
                standards = ", ".join(f'"{s}"' for s in ex.expected_standards)
                lines.append(f"        expected_standards=[{standards}],")

            if ex.intent != "explanatory":
                lines.append(f'        intent="{ex.intent}",')

            lines.append(f'        category="{ex.category}",')
            lines.append(f'        difficulty="{ex.difficulty}",')

            if ex.must_pass:
                lines.append("        must_pass=True,")

            lines.append("    ),")

    return "\n".join(lines) + "\n"


def _wrap_string(lines: list[str], text: str, *, indent: int) -> None:
    """Wrap a long string into multiple quoted lines."""
    pad = " " * indent
    # Split into ~70 char segments at word boundaries
    words = text.split()
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 > 68:
            lines.append(f'{pad}"{current_line} "')
            current_line = word
        else:
            current_line = f"{current_line} {word}" if current_line else word
    if current_line:
        lines.append(f'{pad}"{current_line}"')


def _log_distribution(examples: tuple) -> None:
    """Log example distribution statistics."""
    difficulties: dict[str, int] = {}
    categories: dict[str, int] = {}
    intents: dict[str, int] = {}

    for ex in examples:
        difficulties[ex.difficulty] = difficulties.get(ex.difficulty, 0) + 1
        categories[ex.category] = categories.get(ex.category, 0) + 1
        intents[ex.intent] = intents.get(ex.intent, 0) + 1

    must_pass = sum(1 for ex in examples if ex.must_pass)
    logger.info("  Total: %d (must_pass: %d)", len(examples), must_pass)
    logger.info("  Difficulties: %s", difficulties)
    logger.info("  Categories: %s", categories)
    logger.info("  Intents: %s", intents)


def list_datasets() -> None:
    """List available datasets in LangSmith."""
    from langsmith import Client

    client = Client()
    datasets = list(client.list_datasets())

    print("\nAvailable Datasets:")
    print("=" * 60)
    for ds in datasets:
        example_count = ""
        try:
            examples = list(client.list_examples(dataset_id=ds.id))
            example_count = f" ({len(examples)} examples)"
        except Exception:
            logger.debug("Could not count examples for %s", ds.name)
        print(f"  {ds.name}{example_count}")
        if ds.description:
            first_line = ds.description.strip().split("\n")[0]
            print(f"    {first_line[:70]}")
    print("=" * 60)


def main() -> int:
    """Main entry point."""
    if not os.getenv("LANGSMITH_API_KEY"):
        logger.error("LANGSMITH_API_KEY not set")
        return 1

    parser = argparse.ArgumentParser(description="Manage golden dataset (Hub-First workflow)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--push",
        action="store_true",
        help="Push local examples to LangSmith (initial seed or reset)",
    )
    group.add_argument(
        "--export",
        action="store_true",
        help="Export LangSmith dataset to local fallback file",
    )
    group.add_argument(
        "--list",
        "-l",
        action="store_true",
        dest="list_datasets",
        help="List available LangSmith datasets",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making changes",
    )

    args = parser.parse_args()

    if args.list_datasets:
        list_datasets()
        return 0

    if args.push:
        push_to_langsmith(dry_run=args.dry_run)
    elif args.export:
        export_from_langsmith(dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
