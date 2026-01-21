#!/usr/bin/env python3
"""Update golden dataset with new evaluation examples from human feedback.

This script integrates verified human feedback into the benchmark golden
dataset for continuous evaluation improvement.

Workflow:
1. Load existing golden dataset
2. Load new examples from feedback import
3. Deduplicate and validate new examples
4. Merge into golden dataset with appropriate categorization
5. Generate updated dataset file

Usage:
    # Update golden dataset with new examples
    python scripts/update_datasets.py --examples data/feedback/new_examples.json

    # Preview changes without writing
    python scripts/update_datasets.py --examples new_examples.json --dry-run

    # Update with custom output path
    python scripts/update_datasets.py --examples new_examples.json --output custom_golden.py

    # Filter examples by confidence
    python scripts/update_datasets.py --examples new_examples.json --min-confidence 0.9
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# Constants for difficulty inference
EASY_QUESTION_WORDS = 8
EASY_ANSWER_WORDS = 50
MEDIUM_QUESTION_WORDS = 15
MEDIUM_ANSWER_WORDS = 100

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))


@dataclass
class DatasetExample:
    """An example for the evaluation dataset.

    Attributes:
        id: Unique identifier.
        question: The question.
        ground_truth: Expected answer.
        category: Query category.
        difficulty: Difficulty level.
        expected_tools: Expected tool routing.
        expected_entities: Entities that should be mentioned.
        expected_standards: Standards that should be referenced.
        tags: Tags for filtering.
        metadata: Additional metadata.
    """

    id: str
    question: str
    ground_truth: str
    category: str = "factual"
    difficulty: str = "medium"
    expected_tools: list[str] = field(default_factory=list)
    expected_entities: list[str] = field(default_factory=list)
    expected_standards: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "category": self.category,
            "difficulty": self.difficulty,
            "expected_tools": self.expected_tools,
            "expected_entities": self.expected_entities,
            "expected_standards": self.expected_standards,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetExample:
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            question=data.get("question", ""),
            ground_truth=data.get("ground_truth", ""),
            category=data.get("category", "factual"),
            difficulty=data.get("difficulty", "medium"),
            expected_tools=data.get("expected_tools", []),
            expected_entities=data.get("expected_entities", []),
            expected_standards=data.get("expected_standards", []),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class UpdateStats:
    """Statistics from dataset update.

    Attributes:
        existing_examples: Number of examples in original dataset.
        new_examples_loaded: Number of new examples loaded.
        duplicates_found: Number of duplicate questions found.
        low_confidence_skipped: Number skipped due to low confidence.
        examples_added: Number of examples added.
        final_total: Final dataset size.
    """

    existing_examples: int = 0
    new_examples_loaded: int = 0
    duplicates_found: int = 0
    low_confidence_skipped: int = 0
    examples_added: int = 0
    final_total: int = 0

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            "existing_examples": self.existing_examples,
            "new_examples_loaded": self.new_examples_loaded,
            "duplicates_found": self.duplicates_found,
            "low_confidence_skipped": self.low_confidence_skipped,
            "examples_added": self.examples_added,
            "final_total": self.final_total,
        }


def load_golden_dataset() -> list[DatasetExample]:
    """Load existing golden dataset.

    Returns:
        List of existing examples.
    """
    try:
        from benchmark.golden_dataset import GOLDEN_DATASET
        from benchmark.schemas import BenchmarkExample

        examples = []
        for ex in GOLDEN_DATASET:
            if isinstance(ex, BenchmarkExample):
                examples.append(
                    DatasetExample(
                        id=ex.id,
                        question=ex.question,
                        ground_truth=ex.ground_truth,
                        category=ex.category.value,
                        difficulty=ex.difficulty.value,
                        expected_tools=[t.value for t in ex.expected_tools],
                        expected_entities=ex.expected_entities,
                        expected_standards=ex.expected_standards,
                        tags=ex.tags,
                        metadata=ex.metadata,
                    )
                )

        logger.info("Loaded %d examples from golden dataset", len(examples))
        return examples

    except ImportError as e:
        logger.warning("Could not import golden dataset: %s", e)
        return []


def load_new_examples(input_path: Path) -> list[DatasetExample]:
    """Load new examples from feedback import.

    Args:
        input_path: Path to JSON file with new examples.

    Returns:
        List of new examples.
    """
    logger.info("Loading new examples from %s", input_path)

    with input_path.open() as f:
        data = json.load(f)

    # Handle nested structure
    examples_data = data.get("examples", []) if isinstance(data, dict) else data

    examples = []
    for item in examples_data:
        try:
            example = DatasetExample.from_dict(item)
            examples.append(example)
        except Exception as e:
            logger.warning("Failed to parse example: %s", e)

    logger.info("Loaded %d new examples", len(examples))
    return examples


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts.

    Args:
        text1: First text.
        text2: Second text.

    Returns:
        Similarity ratio (0-1).
    """
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def find_duplicates(
    new_examples: list[DatasetExample],
    existing_examples: list[DatasetExample],
    similarity_threshold: float = 0.85,
) -> set[str]:
    """Find duplicate examples based on question similarity.

    Args:
        new_examples: New examples to check.
        existing_examples: Existing dataset examples.
        similarity_threshold: Minimum similarity to consider duplicate.

    Returns:
        Set of new example IDs that are duplicates.
    """
    duplicates = set()
    existing_questions = [ex.question for ex in existing_examples]

    for new_ex in new_examples:
        for existing_q in existing_questions:
            similarity = calculate_similarity(new_ex.question, existing_q)
            if similarity >= similarity_threshold:
                duplicates.add(new_ex.id)
                logger.debug(
                    "Duplicate found: '%s' similar to existing (%.2f)",
                    new_ex.question[:50],
                    similarity,
                )
                break

    return duplicates


def infer_category(question: str) -> str:
    """Infer query category from question text.

    Args:
        question: The question text.

    Returns:
        Inferred category string.
    """
    question_lower = question.lower()

    if question_lower.startswith(("what is", "what are", "define")):
        return "definitional"
    if "how" in question_lower and ("relate" in question_lower or "connection" in question_lower):
        return "relational"
    if question_lower.startswith("how") or "steps" in question_lower:
        return "procedural"
    if "compare" in question_lower or "difference" in question_lower or "vs" in question_lower:
        return "comparison"
    if "why" in question_lower or "analyze" in question_lower:
        return "analytical"
    if any(word in question_lower for word in ["how many", "count", "list", "which"]):
        return "factual"

    return "factual"


def infer_difficulty(question: str, ground_truth: str) -> str:
    """Infer difficulty level from question and answer.

    Args:
        question: The question text.
        ground_truth: The ground truth answer.

    Returns:
        Inferred difficulty level.
    """
    # Simple heuristics based on complexity
    question_words = len(question.split())
    answer_words = len(ground_truth.split())

    if question_words <= EASY_QUESTION_WORDS and answer_words <= EASY_ANSWER_WORDS:
        return "easy"
    if question_words <= MEDIUM_QUESTION_WORDS and answer_words <= MEDIUM_ANSWER_WORDS:
        return "medium"
    if "and" in question.lower() or answer_words > MEDIUM_ANSWER_WORDS:
        return "hard"

    return "medium"


def extract_standards(text: str) -> list[str]:
    """Extract standard references from text.

    Args:
        text: Text to search.

    Returns:
        List of found standards.
    """
    import re

    standards = []

    # Common standard patterns
    patterns = [
        r"ISO\s*\d+(?:[-/]\d+)?",
        r"IEC\s*\d+(?:[-/]\d+)?",
        r"DO-\d+[A-Z]?",
        r"ASPICE",
        r"CMMI",
        r"FDA\s*21\s*CFR",
        r"AUTOSAR",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        standards.extend(matches)

    return list(set(standards))


def extract_entities(text: str) -> list[str]:
    """Extract domain entities from text.

    Args:
        text: Text to search.

    Returns:
        List of found entities.
    """
    # Domain-specific terms to look for
    domain_terms = [
        "traceability",
        "requirements",
        "verification",
        "validation",
        "V-model",
        "ASIL",
        "safety",
        "hazard analysis",
        "FMEA",
        "FTA",
        "impact analysis",
        "change management",
        "test cases",
        "compliance",
        "audit",
    ]

    text_lower = text.lower()
    found = []

    for term in domain_terms:
        if term.lower() in text_lower:
            found.append(term)

    return found


def enrich_example(example: DatasetExample) -> DatasetExample:
    """Enrich an example with inferred metadata.

    Args:
        example: Example to enrich.

    Returns:
        Enriched example.
    """
    # Infer category if not set
    if example.category == "factual" and not example.metadata.get("category_set"):
        example.category = infer_category(example.question)

    # Infer difficulty if not set
    if example.difficulty == "medium" and not example.metadata.get("difficulty_set"):
        example.difficulty = infer_difficulty(example.question, example.ground_truth)

    # Extract standards if not set
    if not example.expected_standards:
        combined_text = f"{example.question} {example.ground_truth}"
        example.expected_standards = extract_standards(combined_text)

    # Extract entities if not set
    if not example.expected_entities:
        combined_text = f"{example.question} {example.ground_truth}"
        example.expected_entities = extract_entities(combined_text)

    # Add default tags
    if "human-feedback" not in example.tags:
        example.tags.append("human-feedback")
    if "auto-enriched" not in example.tags:
        example.tags.append("auto-enriched")

    return example


def merge_datasets(
    existing: list[DatasetExample],
    new_examples: list[DatasetExample],
    min_confidence: float = 0.8,
) -> tuple[list[DatasetExample], UpdateStats]:
    """Merge new examples into existing dataset.

    Args:
        existing: Existing dataset examples.
        new_examples: New examples to add.
        min_confidence: Minimum confidence threshold.

    Returns:
        Tuple of (merged dataset, update statistics).
    """
    stats = UpdateStats(
        existing_examples=len(existing),
        new_examples_loaded=len(new_examples),
    )

    # Find duplicates
    duplicates = find_duplicates(new_examples, existing)
    stats.duplicates_found = len(duplicates)

    # Process new examples
    merged = list(existing)
    next_id = len(existing) + 1

    for example in new_examples:
        # Skip duplicates
        if example.id in duplicates:
            continue

        # Check confidence
        confidence = example.metadata.get("confidence", 1.0)
        if confidence < min_confidence:
            stats.low_confidence_skipped += 1
            continue

        # Assign new ID if needed
        if not example.id or example.id.startswith("feedback_"):
            example.id = f"hf_{next_id:04d}"
            next_id += 1

        # Enrich example
        enriched = enrich_example(example)

        merged.append(enriched)
        stats.examples_added += 1

    stats.final_total = len(merged)

    return merged, stats


def generate_python_code(examples: list[DatasetExample]) -> str:
    """Generate Python code for updated dataset.

    Args:
        examples: List of examples.

    Returns:
        Python code as string.
    """
    lines = [
        '"""Updated golden dataset with human feedback examples.',
        "",
        f"Generated: {datetime.now(tz=UTC).isoformat()}",
        f"Total examples: {len(examples)}",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Final",
        "",
        "from tests.benchmark.schemas import (",
        "    BenchmarkExample,",
        "    DifficultyLevel,",
        "    ExpectedMetrics,",
        "    ExpectedRouting,",
        "    QueryCategory,",
        ")",
        "",
        "",
        "UPDATED_GOLDEN_DATASET: Final[list[BenchmarkExample]] = [",
    ]

    for ex in examples:
        lines.append("    BenchmarkExample(")
        lines.append(f'        id="{ex.id}",')
        lines.append(f'        question="{_escape_string(ex.question)}",')
        lines.append(f"        category=QueryCategory.{ex.category.upper()},")
        lines.append(f"        difficulty=DifficultyLevel.{ex.difficulty.upper()},")

        # Expected tools
        if ex.expected_tools:
            tools_str = ", ".join(
                f"ExpectedRouting.{t.upper().replace('GRAPHRAG_', '')}" for t in ex.expected_tools
            )
            lines.append(f"        expected_tools=[{tools_str}],")
        else:
            lines.append("        expected_tools=[ExpectedRouting.VECTOR_SEARCH],")

        # Ground truth (handle multiline)
        lines.append("        ground_truth=(")
        gt_lines = _wrap_string(ex.ground_truth, 70)
        for gt_line in gt_lines:
            lines.append(f'            "{_escape_string(gt_line)}"')
        lines.append("        ),")

        # Expected entities
        if ex.expected_entities:
            entities_str = ", ".join(f'"{e}"' for e in ex.expected_entities)
            lines.append(f"        expected_entities=[{entities_str}],")

        # Expected standards
        if ex.expected_standards:
            standards_str = ", ".join(f'"{s}"' for s in ex.expected_standards)
            lines.append(f"        expected_standards=[{standards_str}],")

        # Tags
        if ex.tags:
            tags_str = ", ".join(f'"{t}"' for t in ex.tags)
            lines.append(f"        tags=[{tags_str}],")

        lines.append("    ),")

    lines.append("]")
    lines.append("")

    return "\n".join(lines)


def _escape_string(s: str) -> str:
    """Escape string for Python code."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _wrap_string(s: str, width: int) -> list[str]:
    """Wrap string into multiple lines."""
    words = s.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > width and current_line:
            lines.append(" ".join(current_line) + " ")
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def save_json_dataset(
    examples: list[DatasetExample],
    output_path: Path,
) -> None:
    """Save dataset as JSON.

    Args:
        examples: List of examples.
        output_path: Output file path.
    """
    data = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "total_examples": len(examples),
        "examples": [ex.to_dict() for ex in examples],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info("Saved JSON dataset to %s", output_path)


def save_python_dataset(
    examples: list[DatasetExample],
    output_path: Path,
) -> None:
    """Save dataset as Python module.

    Args:
        examples: List of examples.
        output_path: Output file path.
    """
    code = generate_python_code(examples)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        f.write(code)

    logger.info("Saved Python dataset to %s", output_path)


def main() -> int:
    """Main entry point for dataset update.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Update golden dataset with human feedback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--examples",
        "-e",
        type=Path,
        required=True,
        help="Input JSON file with new examples",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/datasets/updated_golden_dataset.json"),
        help="Output file for updated dataset (JSON)",
    )
    parser.add_argument(
        "--python-output",
        type=Path,
        help="Also output as Python module",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.8,
        help="Minimum confidence threshold for new examples",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for duplicate detection",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load existing dataset
        existing = load_golden_dataset()

        # Load new examples
        new_examples = load_new_examples(args.examples)

        if not new_examples:
            logger.info("No new examples to add")
            return 0

        # Merge datasets
        merged, stats = merge_datasets(
            existing,
            new_examples,
            min_confidence=args.min_confidence,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("DATASET UPDATE SUMMARY")
        print("=" * 60)
        print(f"Existing examples:      {stats.existing_examples}")
        print(f"New examples loaded:    {stats.new_examples_loaded}")
        print(f"Duplicates found:       {stats.duplicates_found}")
        print(f"Low confidence skipped: {stats.low_confidence_skipped}")
        print(f"Examples added:         {stats.examples_added}")
        print(f"Final dataset size:     {stats.final_total}")
        print("=" * 60 + "\n")

        if args.dry_run:
            print("DRY RUN - No files written")

            # Show preview of new examples
            if stats.examples_added > 0:
                print("\nNew examples to be added:")
                print("-" * 40)
                new_ids = {ex.id for ex in existing}
                for ex in merged:
                    if ex.id not in new_ids:
                        print(f"  [{ex.id}] {ex.question[:60]}...")
            return 0

        # Save outputs
        save_json_dataset(merged, args.output)
        print(f"Saved JSON dataset to {args.output}")

        if args.python_output:
            save_python_dataset(merged, args.python_output)
            print(f"Saved Python dataset to {args.python_output}")

        return 0

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 1
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON: %s", e)
        return 1
    except Exception:
        logger.exception("Update failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
