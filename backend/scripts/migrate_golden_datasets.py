#!/usr/bin/env python
"""Migrate the mixed golden dataset into 4 per-vector LangSmith datasets.

Idempotent: checks ``client.has_dataset()`` before creating. Safe to re-run.

Target datasets (from evaluation/constants.py):
- graphrag-eval-explanatory  (explanatory GoldenExamples)
- graphrag-eval-structured   (structured GoldenExamples)
- graphrag-eval-conversational (ConversationalExamples)
- graphrag-eval-intent       (derived from all intents)

Usage:
    # Dry run — show what would happen
    uv run python scripts/migrate_golden_datasets.py --dry-run

    # Validate only — check existing datasets
    uv run python scripts/migrate_golden_datasets.py --validate-only

    # Full migration
    uv run python scripts/migrate_golden_datasets.py

Requires LANGSMITH_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _dataset_exists(client: object, name: str) -> bool:
    """Check if a LangSmith dataset exists by name."""
    try:
        client.read_dataset(dataset_name=name)  # type: ignore[union-attr]
        return True
    except Exception:
        return False


def _create_dataset_with_examples(
    client: object,
    name: str,
    description: str,
    examples: list,
    *,
    dry_run: bool = False,
) -> int:
    """Create a LangSmith dataset and populate it with examples.

    Args:
        client: LangSmith Client instance.
        name: Dataset name.
        description: Dataset description.
        examples: List of GoldenExample or ConversationalExample instances.
        dry_run: If True, only log what would happen.

    Returns:
        Number of examples added.
    """
    if _dataset_exists(client, name):
        logger.info("  SKIP: '%s' already exists", name)
        return 0

    if dry_run:
        logger.info("  [DRY RUN] Would create '%s' with %d examples", name, len(examples))
        return len(examples)

    dataset = client.create_dataset(  # type: ignore[union-attr]
        dataset_name=name,
        description=description,
    )
    logger.info("  Created dataset '%s' (id=%s)", name, dataset.id)

    added = 0
    for ex in examples:
        ls_data = ex.to_langsmith()
        try:
            client.create_example(  # type: ignore[union-attr]
                inputs=ls_data["inputs"],
                outputs=ls_data["outputs"],
                dataset_id=dataset.id,
                metadata=ls_data["metadata"],
            )
            added += 1
        except Exception as e:
            logger.error("  Failed to add %s: %s", ex.id, e)

    logger.info("  Added %d/%d examples", added, len(examples))
    return added


def _build_intent_examples(
    explanatory_examples: list,
    structured_examples: list,
    conversational_examples: list,
) -> list[dict]:
    """Build intent classification examples from all vectors.

    Each example has the question as input and the intent as expected output.
    """
    intent_examples = []

    for ex in explanatory_examples:
        intent_examples.append(
            {
                "inputs": {"question": ex.question},
                "outputs": {"intent": "explanatory"},
                "metadata": {"source_id": ex.id, "source_vector": "explanatory"},
            }
        )

    for ex in structured_examples:
        intent_examples.append(
            {
                "inputs": {"question": ex.question},
                "outputs": {"intent": "structured"},
                "metadata": {"source_id": ex.id, "source_vector": "structured"},
            }
        )

    for ex in conversational_examples:
        if ex.conversation_history:
            intent_examples.append(
                {
                    "inputs": {
                        "question": ex.question,
                        "history": ex.conversation_history,
                    },
                    "outputs": {"intent": "conversational"},
                    "metadata": {"source_id": ex.id, "source_vector": "conversational"},
                }
            )

    return intent_examples


def _create_intent_dataset(
    client: object,
    intent_examples: list[dict],
    *,
    dry_run: bool = False,
) -> int:
    """Create the intent classification dataset."""
    from requirements_graphrag_api.evaluation.constants import DATASET_INTENT

    name = DATASET_INTENT
    if _dataset_exists(client, name):
        logger.info("  SKIP: '%s' already exists", name)
        return 0

    if dry_run:
        logger.info(
            "  [DRY RUN] Would create '%s' with %d examples",
            name,
            len(intent_examples),
        )
        return len(intent_examples)

    dataset = client.create_dataset(  # type: ignore[union-attr]
        dataset_name=name,
        description=(
            "Intent classification evaluation dataset.\n\n"
            "Each example tests whether the router correctly classifies "
            "a question as explanatory, structured, or conversational."
        ),
    )
    logger.info("  Created dataset '%s' (id=%s)", name, dataset.id)

    added = 0
    for ex_data in intent_examples:
        try:
            client.create_example(  # type: ignore[union-attr]
                inputs=ex_data["inputs"],
                outputs=ex_data["outputs"],
                dataset_id=dataset.id,
                metadata=ex_data["metadata"],
            )
            added += 1
        except Exception as e:
            logger.error("  Failed to add intent example: %s", e)

    logger.info("  Added %d/%d intent examples", added, len(intent_examples))
    return added


def validate_datasets(client: object) -> bool:
    """Validate that all expected datasets exist and have correct counts."""
    from requirements_graphrag_api.evaluation.constants import (
        ALL_VECTOR_DATASETS,
    )

    all_ok = True
    for name in ALL_VECTOR_DATASETS:
        try:
            ds = client.read_dataset(dataset_name=name)  # type: ignore[union-attr]
            examples = list(
                client.list_examples(dataset_id=ds.id)  # type: ignore[union-attr]
            )
            logger.info("  OK: '%s' — %d examples", name, len(examples))
        except Exception:
            logger.warning("  MISSING: '%s'", name)
            all_ok = False

    return all_ok


def migrate(*, dry_run: bool = False) -> int:
    """Run the full migration.

    Returns:
        Total examples created across all datasets.
    """
    from requirements_graphrag_api.evaluation.constants import (
        DATASET_CONVERSATIONAL,
        DATASET_EXPLANATORY,
        DATASET_STRUCTURED,
    )
    from requirements_graphrag_api.evaluation.golden_dataset import (
        CONVERSATIONAL_EXAMPLES,
        GOLDEN_EXAMPLES,
    )

    # Split GoldenExamples by vector
    explanatory = [ex for ex in GOLDEN_EXAMPLES if ex.vector == "explanatory"]
    structured = [ex for ex in GOLDEN_EXAMPLES if ex.vector == "structured"]
    conversational = list(CONVERSATIONAL_EXAMPLES)

    logger.info(
        "Local examples: %d explanatory, %d structured, %d conversational",
        len(explanatory),
        len(structured),
        len(conversational),
    )

    if not dry_run:
        from langsmith import Client

        client = Client()
    else:
        # For dry run, create a dummy that always says "not exists"
        client = type("DryRunClient", (), {})()  # type: ignore[assignment]

    total = 0

    # 1. Explanatory dataset
    logger.info("Explanatory dataset:")
    total += _create_dataset_with_examples(
        client,
        DATASET_EXPLANATORY,
        (
            "Explanatory vector evaluation dataset.\n\n"
            "RAG pipeline evaluation: faithfulness, relevancy, "
            "context precision/recall, answer correctness, entity recall."
        ),
        explanatory,
        dry_run=dry_run,
    )

    # 2. Structured dataset
    logger.info("Structured dataset:")
    total += _create_dataset_with_examples(
        client,
        DATASET_STRUCTURED,
        (
            "Structured (Text2Cypher) vector evaluation dataset.\n\n"
            "Evaluates Cypher generation: parse validity, schema adherence, "
            "execution success, result shape, safety, result correctness."
        ),
        structured,
        dry_run=dry_run,
    )

    # 3. Conversational dataset
    logger.info("Conversational dataset:")
    total += _create_dataset_with_examples(
        client,
        DATASET_CONVERSATIONAL,
        (
            "Conversational vector evaluation dataset.\n\n"
            "Multi-turn conversation evaluation: coherence, context retention, "
            "hallucination detection."
        ),
        conversational,
        dry_run=dry_run,
    )

    # 4. Intent dataset (derived from all vectors)
    logger.info("Intent dataset:")
    intent_examples = _build_intent_examples(explanatory, structured, conversational)
    logger.info("  Derived %d intent examples from all vectors", len(intent_examples))
    total += _create_intent_dataset(client, intent_examples, dry_run=dry_run)

    return total


def main() -> int:
    """Main entry point."""
    if not os.getenv("LANGSMITH_API_KEY"):
        logger.error("LANGSMITH_API_KEY not set")
        return 1

    parser = argparse.ArgumentParser(
        description="Migrate golden dataset to 4 per-vector LangSmith datasets",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making changes",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only check if datasets exist and show counts",
    )

    args = parser.parse_args()

    if args.validate_only:
        from langsmith import Client

        client = Client()
        logger.info("Validating datasets:")
        ok = validate_datasets(client)
        return 0 if ok else 1

    total = migrate(dry_run=args.dry_run)
    logger.info("Migration complete: %d total examples created", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
