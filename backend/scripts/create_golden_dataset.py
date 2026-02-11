#!/usr/bin/env python
"""Create a golden dataset in LangSmith for RAGAS evaluation.

This script demonstrates LangSmith dataset management concepts:
- Creating datasets programmatically with Client.create_dataset()
- Adding examples with structured inputs/outputs
- Using metadata for categorization and filtering
- Dataset versioning best practices

Usage:
    uv run python scripts/create_golden_dataset.py [--dry-run]

The golden dataset contains curated question-answer pairs with:
- Expected answers for ground truth comparison
- Expected entities for context entity recall evaluation
- Intent classification for routing evaluation
- Difficulty and domain metadata for analysis

Requires LANGSMITH_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

if TYPE_CHECKING:
    from langsmith import Client

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Dataset configuration
DATASET_NAME = "graphrag-rag-golden"
DATASET_DESCRIPTION = """Golden evaluation dataset for GraphRAG RAG pipeline.

Contains curated examples with:
- Questions spanning core requirements management concepts
- Human-verified expected answers
- Expected entities for context entity recall evaluation
- Intent classification labels
- Difficulty ratings for stratified analysis

Use with langsmith.evaluate() and RAGAS evaluators for comprehensive
pipeline quality assessment.
"""

# =============================================================================
# GOLDEN DATASET EXAMPLES
# Each example includes:
# - inputs: question (required for RAG pipeline)
# - outputs: expected_answer, expected_entities, intent
# - metadata: difficulty, domain for filtering/analysis
# =============================================================================

GOLDEN_EXAMPLES: list[dict[str, Any]] = [
    # Core Concepts - Easy
    {
        "inputs": {"question": "What is requirements traceability?"},
        "outputs": {
            "expected_answer": (
                "Requirements traceability is the ability to trace requirements "
                "throughout the product development lifecycle. It establishes and "
                "maintains links between requirements, design elements, implementation, "
                "and test cases. This enables teams to track requirement origins, "
                "understand dependencies, and verify that all requirements are "
                "properly implemented and tested."
            ),
            "expected_entities": [
                "requirements traceability",
                "traceability matrix",
            ],
            "intent": "explanatory",
        },
        "metadata": {"difficulty": "easy", "domain": "core_concepts"},
    },
    {
        "inputs": {"question": "What is a requirements management tool?"},
        "outputs": {
            "expected_answer": (
                "A requirements management tool is software that helps teams capture, "
                "organize, track, and manage requirements throughout a product's lifecycle. "
                "These tools provide capabilities like requirements authoring, traceability "
                "matrices, change management, collaboration features, and reporting. "
                "Examples include Jama Connect, IBM DOORS, and Helix RM."
            ),
            "expected_entities": [
                "requirements management",
                "Jama Connect",
                "IBM DOORS",
            ],
            "intent": "explanatory",
        },
        "metadata": {"difficulty": "easy", "domain": "tools"},
    },
    {
        "inputs": {"question": "What is the difference between verification and validation?"},
        "outputs": {
            "expected_answer": (
                "Verification confirms that the product is built correctly according to "
                "specifications ('Are we building the product right?'). Validation confirms "
                "that the right product is being built to meet user needs ('Are we building "
                "the right product?'). Verification involves reviews, inspections, and "
                "testing against requirements. Validation involves user acceptance testing "
                "and ensuring the product solves the intended problem."
            ),
            "expected_entities": [
                "verification and validation",
            ],
            "intent": "explanatory",
        },
        "metadata": {"difficulty": "easy", "domain": "core_concepts"},
    },
    # Core Concepts - Medium
    {
        "inputs": {"question": "How do I implement bidirectional traceability in my project?"},
        "outputs": {
            "expected_answer": (
                "To implement bidirectional traceability: 1) Define trace link types "
                "(derives from, satisfies, verifies, implements). 2) Establish links "
                "from high-level requirements to derived requirements (forward traceability) "
                "and from implementation/tests back to requirements (backward traceability). "
                "3) Use a requirements management tool to maintain links. 4) Create "
                "traceability matrices to visualize coverage. 5) Regularly review and "
                "update links during change management. 6) Use traceability reports for "
                "impact analysis and coverage verification."
            ),
            "expected_entities": [
                "bidirectional traceability",
                "coverage analysis",
            ],
            "intent": "explanatory",
        },
        "metadata": {"difficulty": "medium", "domain": "processes"},
    },
    {
        "inputs": {"question": "What are best practices for writing good requirements?"},
        "outputs": {
            "expected_answer": (
                "Best practices for writing requirements include: 1) Be atomic - one "
                "requirement per statement. 2) Be unambiguous - use precise language. "
                "3) Be verifiable - each requirement must be testable. 4) Be complete - "
                "include all necessary information. 5) Be consistent - no contradictions. "
                "6) Use 'shall' for mandatory requirements. 7) Avoid implementation details "
                "in functional requirements. 8) Include acceptance criteria. 9) Assign "
                "unique identifiers for traceability. 10) Review with stakeholders."
            ),
            "expected_entities": [
                "requirements",
                "acceptance criteria",
            ],
            "intent": "explanatory",
        },
        "metadata": {"difficulty": "medium", "domain": "processes"},
    },
    {
        "inputs": {"question": "Why is change management important in requirements?"},
        "outputs": {
            "expected_answer": (
                "Change management is critical because requirements evolve throughout "
                "a project. Without proper change management: 1) Scope creep goes "
                "uncontrolled. 2) Impact of changes isn't analyzed. 3) Stakeholders "
                "aren't informed. 4) Traceability breaks down. 5) Testing coverage gaps "
                "emerge. Effective change management includes change request processes, "
                "impact analysis, approval workflows, version control, and communication "
                "to affected parties."
            ),
            "expected_entities": [
                "change management",
                "impact analysis",
            ],
            "intent": "explanatory",
        },
        "metadata": {"difficulty": "medium", "domain": "processes"},
    },
    # Standards - Medium
    {
        "inputs": {"question": "What is ISO 26262 and how does it relate to requirements?"},
        "outputs": {
            "expected_answer": (
                "ISO 26262 is the international standard for functional safety of "
                "electrical and electronic systems in road vehicles. It defines "
                "Automotive Safety Integrity Levels (ASIL A-D) and requires rigorous "
                "requirements management including: functional safety requirements, "
                "technical safety requirements, hardware/software safety requirements, "
                "bidirectional traceability, verification and validation evidence, "
                "and formal safety analysis documentation."
            ),
            "expected_entities": [
                "ISO 26262",
                "ASIL",
                "functional safety",
            ],
            "intent": "explanatory",
        },
        "metadata": {"difficulty": "medium", "domain": "standards"},
    },
    {
        "inputs": {"question": "What standards apply to medical device software development?"},
        "outputs": {
            "expected_answer": (
                "Key standards for medical device software include: IEC 62304 "
                "(Medical device software lifecycle), which defines software safety "
                "classes A/B/C and required documentation. ISO 14971 for risk "
                "management. FDA 21 CFR Part 820 (Quality System Regulation) for "
                "US market. IEC 62366 for usability. These require comprehensive "
                "requirements documentation, traceability, risk analysis, and "
                "verification/validation evidence."
            ),
            "expected_entities": [
                "IEC 62304",
                "ISO 14971",
                "medical device",
            ],
            "intent": "explanatory",
        },
        "metadata": {"difficulty": "medium", "domain": "standards"},
    },
    # Advanced Topics - Hard
    {
        "inputs": {
            "question": (
                "How do I set up a requirements management process for a regulated industry?"
            )
        },
        "outputs": {
            "expected_answer": (
                "Setting up RM for regulated industries requires: 1) Identify applicable "
                "standards (ISO 26262, IEC 62304, DO-178C, etc.). 2) Define requirements "
                "types and hierarchy (stakeholder, system, software, hardware). "
                "3) Establish traceability strategy covering all lifecycle phases. "
                "4) Implement formal review and approval workflows. 5) Select compliant "
                "tooling with audit trails. 6) Define change control process with impact "
                "analysis. 7) Create verification/validation strategy. 8) Plan for "
                "regulatory audits with documentation. 9) Train team on processes and tools."
            ),
            "expected_entities": [
                "ISO 26262",
                "IEC 62304",
                "DO-178C",
            ],
            "intent": "explanatory",
        },
        "metadata": {"difficulty": "hard", "domain": "processes"},
    },
    {
        "inputs": {
            "question": (
                "What is Model-Based Systems Engineering and how does it help with requirements?"
            )
        },
        "outputs": {
            "expected_answer": (
                "Model-Based Systems Engineering (MBSE) uses models as the primary "
                "artifact for systems engineering instead of documents. For requirements, "
                "MBSE provides: visual requirements representation (SysML diagrams), "
                "formal consistency checking, automated traceability through model "
                "relationships, simulation for early validation, and better stakeholder "
                "communication. Tools like Cameo, Rhapsody, and Capella support MBSE. "
                "Challenges include learning curve and tool adoption."
            ),
            "expected_entities": [
                "MBSE",
                "SysML",
            ],
            "intent": "explanatory",
        },
        "metadata": {"difficulty": "hard", "domain": "methodologies"},
    },
    # Structured Queries - for intent classification testing
    {
        "inputs": {"question": "List all webinars in the knowledge base."},
        "outputs": {
            "expected_answer": "[List of webinars from database]",
            "expected_entities": [],
            "intent": "structured",
        },
        "metadata": {"difficulty": "easy", "domain": "data_query"},
    },
    {
        "inputs": {"question": "How many articles mention requirements traceability?"},
        "outputs": {
            "expected_answer": "[Count from database]",
            "expected_entities": [],
            "intent": "structured",
        },
        "metadata": {"difficulty": "easy", "domain": "data_query"},
    },
    {
        "inputs": {"question": "Which standards are covered in the knowledge base?"},
        "outputs": {
            "expected_answer": "[List of standards from database]",
            "expected_entities": [],
            "intent": "structured",
        },
        "metadata": {"difficulty": "easy", "domain": "data_query"},
    },
    {
        "inputs": {"question": "What are the top 5 most mentioned tools?"},
        "outputs": {
            "expected_answer": "[Ranked list from database]",
            "expected_entities": [],
            "intent": "structured",
        },
        "metadata": {"difficulty": "medium", "domain": "data_query"},
    },
    # Edge Cases
    {
        "inputs": {"question": "Tell me about Jama Connect."},
        "outputs": {
            "expected_answer": (
                "Jama Connect is a requirements management and product development "
                "platform. It provides requirements authoring, live traceability, "
                "review and approval workflows, risk management integration, "
                "test management, and collaboration features. It's used across "
                "industries including automotive, aerospace, medical devices, and "
                "industrial equipment for managing complex product development."
            ),
            "expected_entities": [
                "Jama Connect",
                "requirements management",
            ],
            "intent": "explanatory",
        },
        "metadata": {"difficulty": "medium", "domain": "tools"},
    },
    {
        "inputs": {
            "question": "I need to improve our requirements review process. Any suggestions?"
        },
        "outputs": {
            "expected_answer": (
                "To improve requirements reviews: 1) Use structured review checklists "
                "covering completeness, consistency, and testability. 2) Conduct reviews "
                "in small batches rather than large documents. 3) Include diverse "
                "stakeholders (developers, testers, domain experts). 4) Use collaborative "
                "review tools for asynchronous input. 5) Track review metrics (defects "
                "found, review time). 6) Implement different review types: peer review, "
                "formal inspection, walkthrough. 7) Review traceability links along with "
                "requirements. 8) Document decisions and action items."
            ),
            "expected_entities": [
                "review status",
                "approval workflow",
            ],
            "intent": "explanatory",
        },
        "metadata": {"difficulty": "medium", "domain": "processes"},
    },
]


def create_golden_dataset(
    client: Client | None,
    *,
    dry_run: bool = False,
) -> str | None:
    """Create the golden dataset in LangSmith.

    Args:
        client: LangSmith client.
        dry_run: If True, only log what would be created.

    Returns:
        Dataset ID if created, None if dry run or error.
    """
    if dry_run:
        logger.info("[DRY RUN] Would create dataset: %s", DATASET_NAME)
        logger.info("  Description: %s", DATASET_DESCRIPTION[:100] + "...")
        logger.info("  Examples: %d", len(GOLDEN_EXAMPLES))

        # Show distribution
        difficulties = {}
        domains = {}
        intents = {}
        for ex in GOLDEN_EXAMPLES:
            meta = ex.get("metadata", {})
            difficulties[meta.get("difficulty", "unknown")] = (
                difficulties.get(meta.get("difficulty", "unknown"), 0) + 1
            )
            domains[meta.get("domain", "unknown")] = (
                domains.get(meta.get("domain", "unknown"), 0) + 1
            )
            intent = ex.get("outputs", {}).get("intent", "unknown")
            intents[intent] = intents.get(intent, 0) + 1

        logger.info("\n  Distribution:")
        logger.info("    Difficulties: %s", difficulties)
        logger.info("    Domains: %s", domains)
        logger.info("    Intents: %s", intents)
        return None

    if client is None:
        return None

    # Check if dataset already exists
    try:
        existing = client.read_dataset(dataset_name=DATASET_NAME)
        logger.info(
            "Dataset '%s' already exists (id=%s). Skipping creation.",
            DATASET_NAME,
            existing.id,
        )
        return str(existing.id)
    except Exception:
        logger.debug("Dataset '%s' not found, will create it.", DATASET_NAME)

    # Create dataset with metadata
    logger.info("Creating dataset: %s", DATASET_NAME)
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=DATASET_DESCRIPTION,
    )
    logger.info("Created dataset with ID: %s", dataset.id)

    # Add examples with metadata
    successful = 0
    for i, example in enumerate(GOLDEN_EXAMPLES):
        try:
            client.create_example(
                inputs=example["inputs"],
                outputs=example["outputs"],
                dataset_id=dataset.id,
                metadata=example.get("metadata"),
            )
            successful += 1
            logger.debug(
                "Added example %d/%d: %s",
                i + 1,
                len(GOLDEN_EXAMPLES),
                example["inputs"]["question"][:50],
            )
        except Exception as e:
            logger.error("Failed to add example %d: %s", i + 1, e)

    logger.info("Added %d/%d examples to dataset", successful, len(GOLDEN_EXAMPLES))
    logger.info("View at: https://smith.langchain.com/datasets")

    return str(dataset.id)


def main(dry_run: bool = False) -> int:
    """Create the golden dataset.

    Args:
        dry_run: If True, only log what would be created.

    Returns:
        Exit code (0 for success).
    """
    if not os.getenv("LANGSMITH_API_KEY"):
        logger.error("LANGSMITH_API_KEY not set")
        return 1

    client: Client | None = None
    if not dry_run:
        try:
            from langsmith import Client as LangSmithClient

            client = LangSmithClient()
        except ImportError:
            logger.error("langsmith package not installed. Run: pip install langsmith")
            return 1

    logger.info("=" * 60)
    logger.info("Creating Golden Dataset for RAGAS Evaluation")
    logger.info("=" * 60)

    dataset_id = create_golden_dataset(client, dry_run=dry_run)

    if dry_run:
        logger.info("[DRY RUN] Would create dataset with %d examples", len(GOLDEN_EXAMPLES))
    elif dataset_id:
        logger.info("Dataset ready: %s", DATASET_NAME)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create golden dataset in LangSmith for RAGAS evaluation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be created without actually creating",
    )
    args = parser.parse_args()

    sys.exit(main(dry_run=args.dry_run))
