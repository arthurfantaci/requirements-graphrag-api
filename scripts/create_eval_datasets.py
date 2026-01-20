#!/usr/bin/env python
"""Create evaluation datasets in LangSmith for prompt testing.

This script creates datasets for evaluating:
- Router: Tool selection accuracy
- Critic: Context evaluation quality
- Text2Cypher: Cypher query generation

Usage:
    uv run python scripts/create_eval_datasets.py [--dry-run]

Requires LANGSMITH_API_KEY and LANGSMITH_WORKSPACE_ID in environment.
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

# Constants
PREVIEW_EXAMPLE_COUNT = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# ROUTER EVALUATION DATASET
# Tests tool routing accuracy based on question characteristics
# =============================================================================

ROUTER_EXAMPLES: list[dict[str, Any]] = [
    # Simple concept lookups -> vector_search
    {
        "inputs": {
            "tools": """Available tools:
- graphrag_vector_search: Semantic search for general knowledge questions
- graphrag_hybrid_search: Combined keyword + semantic search for specific terms
- graphrag_graph_enriched_search: Search with related entities for complex topics
- graphrag_explore_entity: Deep dive into specific entities
- graphrag_lookup_standard: Look up regulatory standards
- graphrag_lookup_term: Look up glossary definitions
- graphrag_text2cypher: Convert questions to Cypher for aggregations
- graphrag_chat: Multi-turn conversation with synthesis""",
            "question": "What is requirements traceability?",
        },
        "outputs": {
            "expected_tools": ["graphrag_vector_search"],
            "reasoning": "Simple concept definition question",
        },
    },
    {
        "inputs": {
            "tools": """[same tools list]""",
            "question": "Explain the basics of requirements management.",
        },
        "outputs": {
            "expected_tools": ["graphrag_vector_search"],
            "reasoning": "General knowledge question about fundamentals",
        },
    },
    # Specific term lookups -> hybrid_search or lookup_term
    {
        "inputs": {
            "tools": """[same tools list]""",
            "question": "What does ASIL stand for in automotive?",
        },
        "outputs": {
            "expected_tools": ["graphrag_lookup_term", "graphrag_hybrid_search"],
            "reasoning": "Acronym/term definition lookup",
        },
    },
    {
        "inputs": {
            "tools": """[same tools list]""",
            "question": "Define 'traceability matrix'.",
        },
        "outputs": {
            "expected_tools": ["graphrag_lookup_term"],
            "reasoning": "Direct term definition request",
        },
    },
    # Regulatory/standards questions -> lookup_standard
    {
        "inputs": {
            "tools": """[same tools list]""",
            "question": "What are the requirements of ISO 26262?",
        },
        "outputs": {
            "expected_tools": ["graphrag_lookup_standard", "graphrag_hybrid_search"],
            "reasoning": "Standards-specific question",
        },
    },
    {
        "inputs": {
            "tools": """[same tools list]""",
            "question": "Which standards apply to medical device development?",
        },
        "outputs": {
            "expected_tools": ["graphrag_lookup_standard"],
            "reasoning": "Industry-specific standards lookup",
        },
    },
    # Relationship/connection questions -> graph_enriched_search
    {
        "inputs": {
            "tools": """[same tools list]""",
            "question": "How does requirements traceability relate to change management?",
        },
        "outputs": {
            "expected_tools": ["graphrag_graph_enriched_search"],
            "reasoning": "Question about relationships between concepts",
        },
    },
    {
        "inputs": {
            "tools": """[same tools list]""",
            "question": "What tools support both requirements management and test management?",
        },
        "outputs": {
            "expected_tools": ["graphrag_graph_enriched_search", "graphrag_explore_entity"],
            "reasoning": "Cross-domain relationship question",
        },
    },
    # Entity exploration -> explore_entity
    {
        "inputs": {
            "tools": """[same tools list]""",
            "question": "Tell me everything about Jama Connect.",
        },
        "outputs": {
            "expected_tools": ["graphrag_explore_entity"],
            "reasoning": "Deep dive into specific entity",
        },
    },
    {
        "inputs": {
            "tools": """[same tools list]""",
            "question": "What features does DOORS have?",
        },
        "outputs": {
            "expected_tools": ["graphrag_explore_entity", "graphrag_hybrid_search"],
            "reasoning": "Specific product/entity exploration",
        },
    },
    # Aggregation/counting questions -> text2cypher
    {
        "inputs": {
            "tools": """[same tools list]""",
            "question": "How many articles mention ISO 26262?",
        },
        "outputs": {
            "expected_tools": ["graphrag_text2cypher"],
            "reasoning": "Counting/aggregation query",
        },
    },
    {
        "inputs": {
            "tools": """[same tools list]""",
            "question": "Which industries have the most standards?",
        },
        "outputs": {
            "expected_tools": ["graphrag_text2cypher"],
            "reasoning": "Comparative aggregation query",
        },
    },
    # Multi-part/complex questions -> chat
    {
        "inputs": {
            "tools": """[same tools list]""",
            "question": (
                "I'm implementing requirements management for automotive. "
                "What standards should I follow and what tools would you recommend?"
            ),
        },
        "outputs": {
            "expected_tools": ["graphrag_chat"],
            "reasoning": "Multi-faceted question requiring synthesis",
        },
    },
    {
        "inputs": {
            "tools": """[same tools list]""",
            "question": (
                "Compare the traceability features of different RM tools "
                "and explain which is best for regulated industries."
            ),
        },
        "outputs": {
            "expected_tools": ["graphrag_chat", "graphrag_graph_enriched_search"],
            "reasoning": "Comparison requiring multiple lookups and synthesis",
        },
    },
]

# =============================================================================
# CRITIC EVALUATION DATASET
# Tests context quality assessment accuracy
# =============================================================================

CRITIC_EXAMPLES: list[dict[str, Any]] = [
    # Complete context - should be answerable
    {
        "inputs": {
            "context": """Requirements traceability is the ability to trace requirements
throughout the product lifecycle. It enables teams to track the origin, development,
and allocation of requirements from inception to implementation and testing.
A traceability matrix is commonly used to document these relationships.""",
            "question": "What is requirements traceability?",
        },
        "outputs": {
            "expected_answerable": True,
            "expected_completeness": "complete",
            "expected_confidence": 0.9,
        },
    },
    {
        "inputs": {
            "context": """ISO 26262 is the international standard for functional safety
of electrical and electronic systems in road vehicles. It defines ASIL levels
(Automotive Safety Integrity Levels) from A to D, with D being the most stringent.
The standard covers the entire product development lifecycle.""",
            "question": "What is ISO 26262 and what are ASIL levels?",
        },
        "outputs": {
            "expected_answerable": True,
            "expected_completeness": "complete",
            "expected_confidence": 0.95,
        },
    },
    # Partial context - can answer but incomplete
    {
        "inputs": {
            "context": """Requirements management tools help teams capture and organize
requirements. Popular tools include Jama Connect, IBM DOORS, and Helix RM.""",
            "question": "What are the key features to look for in a requirements management tool?",
        },
        "outputs": {
            "expected_answerable": True,
            "expected_completeness": "partial",
            "expected_confidence": 0.5,
            "missing_aspects": ["feature comparison", "evaluation criteria"],
        },
    },
    {
        "inputs": {
            "context": """Change management in requirements engineering involves tracking
modifications to requirements over time. Impact analysis is performed to understand
how changes affect other requirements.""",
            "question": "How do you implement a complete change management process?",
        },
        "outputs": {
            "expected_answerable": True,
            "expected_completeness": "partial",
            "expected_confidence": 0.6,
            "missing_aspects": ["workflow details", "approval process", "tooling"],
        },
    },
    # Insufficient context - cannot answer reliably
    {
        "inputs": {
            "context": """Requirements come in different types including functional
and non-functional requirements.""",
            "question": "How do I set up bi-directional traceability in my project?",
        },
        "outputs": {
            "expected_answerable": False,
            "expected_completeness": "insufficient",
            "expected_confidence": 0.2,
            "missing_aspects": ["traceability setup", "tooling", "process"],
        },
    },
    {
        "inputs": {
            "context": """Quality assurance is important in software development.""",
            "question": (
                "What are the specific requirements for medical device software under IEC 62304?"
            ),
        },
        "outputs": {
            "expected_answerable": False,
            "expected_completeness": "insufficient",
            "expected_confidence": 0.1,
            "missing_aspects": [
                "IEC 62304 details",
                "software classes",
                "documentation requirements",
            ],
        },
    },
    # Irrelevant context
    {
        "inputs": {
            "context": """Marketing strategies for B2B software companies include
content marketing, trade shows, and direct sales approaches.""",
            "question": "What is the V-model in software development?",
        },
        "outputs": {
            "expected_answerable": False,
            "expected_completeness": "insufficient",
            "expected_confidence": 0.0,
            "missing_aspects": ["V-model definition", "development phases", "verification"],
        },
    },
    # Complex multi-aspect question with good context
    {
        "inputs": {
            "context": """Requirements traceability enables tracking from stakeholder needs
through system requirements, software requirements, design, implementation, and testing.
Forward traceability tracks from requirements to implementation, while backward
traceability tracks from implementation back to requirements. Bi-directional
traceability combines both approaches. Benefits include impact analysis, coverage
analysis, and compliance demonstration.""",
            "question": "Explain bi-directional traceability and its benefits.",
        },
        "outputs": {
            "expected_answerable": True,
            "expected_completeness": "complete",
            "expected_confidence": 0.95,
        },
    },
    # Context with conflicting information
    {
        "inputs": {
            "context": """Some sources recommend document-based requirements management
for simplicity. However, modern best practices suggest database-driven tools provide
better traceability. The choice depends on project complexity.""",
            "question": "Should I use documents or a database for requirements?",
        },
        "outputs": {
            "expected_answerable": True,
            "expected_completeness": "complete",
            "expected_confidence": 0.8,
            "note": "Context presents both perspectives",
        },
    },
    # Very short context
    {
        "inputs": {
            "context": """DOORS is a requirements management tool by IBM.""",
            "question": "Compare DOORS to other requirements management tools.",
        },
        "outputs": {
            "expected_answerable": False,
            "expected_completeness": "insufficient",
            "expected_confidence": 0.2,
            "missing_aspects": ["other tools", "comparison criteria", "feature details"],
        },
    },
]

# =============================================================================
# TEXT2CYPHER EVALUATION DATASET
# Tests Cypher query generation accuracy
# =============================================================================

TEXT2CYPHER_EXAMPLES: list[dict[str, Any]] = [
    # Simple count queries
    {
        "inputs": {
            "schema": """Node types: Article, Chunk, Entity, Definition, Standard, Image
Relationships: (Chunk)-[:FROM_ARTICLE]->(Article), (Entity)-[:MENTIONED_IN]->(Chunk)""",
            "examples": """Q: How many articles are there?
Cypher: MATCH (a:Article) RETURN count(a) AS article_count""",
            "question": "How many articles are in the knowledge base?",
        },
        "outputs": {
            "expected_cypher": "MATCH (a:Article) RETURN count(a)",
            "expected_patterns": ["MATCH", "Article", "count", "RETURN"],
        },
    },
    {
        "inputs": {
            "schema": """[same schema]""",
            "examples": """[same examples]""",
            "question": "Count the number of entities.",
        },
        "outputs": {
            "expected_cypher": "MATCH (e:Entity) RETURN count(e)",
            "expected_patterns": ["MATCH", "Entity", "count", "RETURN"],
        },
    },
    # Filter queries
    {
        "inputs": {
            "schema": """[same schema]""",
            "examples": """Q: Find articles about traceability
Cypher: MATCH (a:Article) WHERE toLower(a.title) CONTAINS 'traceability' RETURN a.title""",
            "question": "Find all articles about ISO 26262.",
        },
        "outputs": {
            "expected_cypher": (
                "MATCH (a:Article) WHERE toLower(a.title) CONTAINS 'iso 26262' RETURN a"
            ),
            "expected_patterns": ["MATCH", "Article", "WHERE", "CONTAINS", "iso 26262"],
        },
    },
    {
        "inputs": {
            "schema": """[same schema]""",
            "examples": """[same examples]""",
            "question": "Which standards are from ISO?",
        },
        "outputs": {
            "expected_cypher": "MATCH (s:Standard) WHERE s.organization = 'ISO' RETURN s",
            "expected_patterns": ["MATCH", "Standard", "WHERE", "RETURN"],
        },
    },
    # Relationship queries
    {
        "inputs": {
            "schema": """[same schema]""",
            "examples": """Q: Which entities are mentioned in an article?
Cypher: MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)-[:FROM_ARTICLE]->(a:Article)
WHERE a.title = 'Article Title' RETURN DISTINCT e.name""",
            "question": "What entities are mentioned in articles about requirements?",
        },
        "outputs": {
            "expected_cypher": (
                "MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)-[:FROM_ARTICLE]->(a:Article)"
            ),
            "expected_patterns": [
                "MATCH",
                "Entity",
                "MENTIONED_IN",
                "Chunk",
                "FROM_ARTICLE",
                "Article",
            ],
        },
    },
    {
        "inputs": {
            "schema": """[same schema]""",
            "examples": """[same examples]""",
            "question": "Which articles mention the term 'traceability matrix'?",
        },
        "outputs": {
            "expected_cypher": (
                "MATCH (d:Definition)-[:MENTIONED_IN]->(c:Chunk)-[:FROM_ARTICLE]->(a:Article)"
            ),
            "expected_patterns": ["Definition", "MENTIONED_IN", "FROM_ARTICLE", "Article"],
        },
    },
    # Aggregation queries
    {
        "inputs": {
            "schema": """[same schema]""",
            "examples": """Q: How many entities per article?
Cypher: MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)-[:FROM_ARTICLE]->(a:Article)
RETURN a.title, count(DISTINCT e) AS entity_count ORDER BY entity_count DESC""",
            "question": "Which articles have the most entities mentioned?",
        },
        "outputs": {
            "expected_cypher": "ORDER BY",
            "expected_patterns": ["MATCH", "count", "ORDER BY", "DESC"],
        },
    },
    {
        "inputs": {
            "schema": """[same schema]""",
            "examples": """[same examples]""",
            "question": "What are the top 5 most mentioned standards?",
        },
        "outputs": {
            "expected_cypher": "LIMIT 5",
            "expected_patterns": ["MATCH", "Standard", "count", "ORDER BY", "LIMIT"],
        },
    },
    # Pattern matching queries
    {
        "inputs": {
            "schema": """[same schema]""",
            "examples": """[same examples]""",
            "question": "Find all definitions that contain the word 'requirement'.",
        },
        "outputs": {
            "expected_cypher": (
                "MATCH (d:Definition) WHERE toLower(d.definition) CONTAINS 'requirement'"
            ),
            "expected_patterns": ["Definition", "WHERE", "CONTAINS", "requirement"],
        },
    },
    # Existence check queries
    {
        "inputs": {
            "schema": """[same schema]""",
            "examples": """[same examples]""",
            "question": "Are there any images in the knowledge base?",
        },
        "outputs": {
            "expected_cypher": "MATCH (i:Image) RETURN count(i) > 0",
            "expected_patterns": ["MATCH", "Image", "RETURN"],
        },
    },
    # Multi-hop relationship queries
    {
        "inputs": {
            "schema": """[same schema with: (Standard)-[:APPLIES_TO]->(Industry)]""",
            "examples": """[same examples]""",
            "question": "Which standards apply to the automotive industry?",
        },
        "outputs": {
            "expected_cypher": "MATCH (s:Standard)-[:APPLIES_TO]->(i:Industry)",
            "expected_patterns": ["Standard", "APPLIES_TO", "Industry", "automotive"],
        },
    },
]


def create_dataset(
    client: Client | None,
    name: str,
    description: str,
    examples: list[dict[str, Any]],
    *,
    dry_run: bool = False,
) -> str | None:
    """Create a dataset in LangSmith and populate with examples.

    Args:
        client: LangSmith client.
        name: Dataset name.
        description: Dataset description.
        examples: List of example dictionaries with 'inputs' and 'outputs'.
        dry_run: If True, only log what would be created.

    Returns:
        Dataset ID if created, None if dry run.
    """
    if dry_run:
        logger.info("[DRY RUN] Would create dataset: %s", name)
        logger.info("  Description: %s", description)
        logger.info("  Examples: %d", len(examples))
        for i, ex in enumerate(examples[:PREVIEW_EXAMPLE_COUNT]):
            logger.info("  Example %d inputs: %s", i + 1, list(ex["inputs"].keys()))
        if len(examples) > PREVIEW_EXAMPLE_COUNT:
            remaining = len(examples) - PREVIEW_EXAMPLE_COUNT
            logger.info("  ... and %d more examples", remaining)
        return None

    if client is None:
        return None

    # Check if dataset already exists
    try:
        existing = client.read_dataset(dataset_name=name)
        logger.info(
            "Dataset '%s' already exists (id=%s). Skipping creation.",
            name,
            existing.id,
        )
        return str(existing.id)
    except Exception:
        logger.debug("Dataset '%s' not found, will create it.", name)

    # Create dataset
    logger.info("Creating dataset: %s", name)
    dataset = client.create_dataset(
        dataset_name=name,
        description=description,
    )
    logger.info("Created dataset with ID: %s", dataset.id)

    # Add examples
    for i, example in enumerate(examples):
        try:
            client.create_example(
                inputs=example["inputs"],
                outputs=example["outputs"],
                dataset_id=dataset.id,
            )
            logger.debug("Added example %d/%d", i + 1, len(examples))
        except Exception as e:
            logger.error("Failed to add example %d: %s", i + 1, e)

    logger.info("Added %d examples to dataset '%s'", len(examples), name)
    return str(dataset.id)


def main(dry_run: bool = False) -> int:
    """Create all evaluation datasets.

    Args:
        dry_run: If True, only log what would be created.

    Returns:
        Exit code (0 for success).
    """
    # Check environment
    if not os.getenv("LANGSMITH_API_KEY"):
        logger.error("LANGSMITH_API_KEY not set")
        return 1

    client: Client | None = None
    if not dry_run:
        try:
            from langsmith import Client as LangSmithClient  # noqa: PLC0415

            client = LangSmithClient()
        except ImportError:
            logger.error("langsmith package not installed. Run: pip install langsmith")
            return 1

    logger.info("=" * 60)
    logger.info("Creating Evaluation Datasets for GraphRAG Prompts")
    logger.info("=" * 60)

    datasets = [
        {
            "name": "graphrag-router-eval",
            "description": (
                "Evaluation dataset for the GraphRAG router prompt. "
                "Tests tool selection accuracy based on question characteristics."
            ),
            "examples": ROUTER_EXAMPLES,
        },
        {
            "name": "graphrag-critic-eval",
            "description": (
                "Evaluation dataset for the GraphRAG critic prompt. "
                "Tests context quality assessment accuracy."
            ),
            "examples": CRITIC_EXAMPLES,
        },
        {
            "name": "graphrag-text2cypher-eval",
            "description": (
                "Evaluation dataset for the GraphRAG text2cypher prompt. "
                "Tests Cypher query generation accuracy."
            ),
            "examples": TEXT2CYPHER_EXAMPLES,
        },
    ]

    created_ids = []
    for ds in datasets:
        logger.info("--- %s ---", ds["name"])
        dataset_id = create_dataset(
            client,
            ds["name"],
            ds["description"],
            ds["examples"],
            dry_run=dry_run,
        )
        if dataset_id:
            created_ids.append(dataset_id)

    logger.info("%s", "=" * 60)
    if dry_run:
        logger.info("[DRY RUN] Would create %d datasets", len(datasets))
    else:
        logger.info("Created/verified %d datasets", len(created_ids))
        logger.info("View datasets at: https://smith.langchain.com/datasets")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create evaluation datasets in LangSmith")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be created without actually creating",
    )
    args = parser.parse_args()

    sys.exit(main(dry_run=args.dry_run))
