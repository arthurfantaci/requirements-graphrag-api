#!/usr/bin/env python
"""Generate benchmark evaluation datasets.

Creates comprehensive evaluation datasets by combining templates
with domain concepts from the requirements management knowledge base.

Usage:
    # Generate default 250-example dataset
    uv run python scripts/generate_benchmark_dataset.py

    # Generate specific count
    uv run python scripts/generate_benchmark_dataset.py --count 100

    # Generate specific categories
    uv run python scripts/generate_benchmark_dataset.py --categories definitional procedural

    # Output to JSON file
    uv run python scripts/generate_benchmark_dataset.py --output benchmark_data.json

    # Show statistics only
    uv run python scripts/generate_benchmark_dataset.py --stats-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src and tests to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.benchmark.generator import (
    generate_evaluation_dataset,
    get_examples_by_category,
    get_examples_by_difficulty,
)
from tests.benchmark.golden_dataset import GOLDEN_DATASET
from tests.benchmark.schemas import (
    BenchmarkExample,
    DifficultyLevel,
    QueryCategory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def print_statistics(
    dataset: list[BenchmarkExample],
    title: str = "Dataset Statistics",
) -> None:
    """Print dataset statistics."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"\n  Total Examples: {len(dataset)}")

    # By category
    print("\n  By Category:")
    for category in QueryCategory:
        examples = get_examples_by_category(dataset, category)
        pct = (len(examples) / len(dataset) * 100) if dataset else 0
        print(f"    {category.value:15} : {len(examples):3} ({pct:5.1f}%)")

    # By difficulty
    print("\n  By Difficulty:")
    for difficulty in DifficultyLevel:
        examples = get_examples_by_difficulty(dataset, difficulty)
        pct = (len(examples) / len(dataset) * 100) if dataset else 0
        print(f"    {difficulty.value:10} : {len(examples):3} ({pct:5.1f}%)")

    # Count unique expected tools
    all_tools: set[str] = set()
    for ex in dataset:
        all_tools.update(t.value for t in ex.expected_tools)
    print(f"\n  Unique Tools Referenced: {len(all_tools)}")

    # Count examples with standards
    with_standards = [ex for ex in dataset if ex.expected_standards]
    print(f"  Examples with Standards: {len(with_standards)}")

    # Count examples with entities
    with_entities = [ex for ex in dataset if ex.expected_entities]
    print(f"  Examples with Entities: {len(with_entities)}")

    print(f"\n{'=' * 60}\n")


def save_to_json(dataset: list[BenchmarkExample], output_path: Path) -> None:
    """Save dataset to JSON file."""
    data = [ex.to_dict() for ex in dataset]
    output_path.write_text(json.dumps(data, indent=2))
    logger.info("Saved %d examples to %s", len(dataset), output_path)


def save_to_langsmith_format(
    dataset: list[BenchmarkExample],
    output_path: Path,
) -> None:
    """Save dataset in LangSmith-compatible format."""
    langsmith_data = []
    for ex in dataset:
        langsmith_data.append(
            {
                "inputs": {
                    "question": ex.question,
                },
                "outputs": {
                    "ground_truth": ex.ground_truth,
                    "expected_tools": [t.value for t in ex.expected_tools],
                },
                "metadata": {
                    "id": ex.id,
                    "category": ex.category.value,
                    "difficulty": ex.difficulty.value,
                    "expected_entities": ex.expected_entities,
                    "expected_standards": ex.expected_standards,
                    "tags": ex.tags,
                },
            }
        )

    output_path.write_text(json.dumps(langsmith_data, indent=2))
    logger.info("Saved %d examples in LangSmith format to %s", len(dataset), output_path)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark evaluation datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--count",
        type=int,
        default=250,
        help="Number of examples to generate (default: 250)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=[c.value for c in QueryCategory],
        help="Specific categories to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--langsmith-format",
        action="store_true",
        help="Output in LangSmith dataset format",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print statistics without generating new data",
    )
    parser.add_argument(
        "--include-golden",
        action="store_true",
        help="Include golden dataset in output",
    )
    parser.add_argument(
        "--golden-only",
        action="store_true",
        help="Output only the golden dataset",
    )

    args = parser.parse_args()

    # Golden dataset only
    if args.golden_only:
        print_statistics(GOLDEN_DATASET, "Golden Dataset Statistics")

        if args.output:
            output_path = Path(args.output)
            if args.langsmith_format:
                save_to_langsmith_format(GOLDEN_DATASET, output_path)
            else:
                save_to_json(GOLDEN_DATASET, output_path)

        return 0

    # Stats only for existing data
    if args.stats_only:
        print_statistics(GOLDEN_DATASET, "Golden Dataset Statistics")

        generated = generate_evaluation_dataset(total_examples=args.count, seed=args.seed)
        print_statistics(generated, f"Generated Dataset Statistics (n={args.count})")

        return 0

    # Parse categories if specified
    categories = None
    if args.categories:
        categories = [QueryCategory(c) for c in args.categories]
        logger.info("Generating for categories: %s", args.categories)

    # Generate dataset
    logger.info("Generating %d evaluation examples...", args.count)
    dataset = generate_evaluation_dataset(
        total_examples=args.count,
        categories=categories,
        seed=args.seed,
    )

    # Optionally include golden dataset
    if args.include_golden:
        # Remove duplicates by ID
        existing_ids = {ex.id for ex in dataset}
        golden_to_add = [ex for ex in GOLDEN_DATASET if ex.id not in existing_ids]
        dataset = golden_to_add + dataset
        logger.info("Added %d golden examples", len(golden_to_add))

    # Print statistics
    print_statistics(dataset, "Generated Dataset Statistics")

    # Save if output specified
    if args.output:
        output_path = Path(args.output)
        if args.langsmith_format:
            save_to_langsmith_format(dataset, output_path)
        else:
            save_to_json(dataset, output_path)

    # Print sample examples
    print("Sample Examples:")
    print("-" * 60)
    for ex in dataset[:3]:
        print(f"\n[{ex.id}] {ex.category.value} / {ex.difficulty.value}")
        print(f"  Q: {ex.question}")
        print(f"  Tools: {[t.value for t in ex.expected_tools]}")
        if ex.expected_standards:
            print(f"  Standards: {ex.expected_standards}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
