"""Programmatic benchmark dataset generator.

Generates evaluation examples by combining query templates
with domain concepts from the knowledge base.

Usage:
    from tests.benchmark.generator import generate_evaluation_dataset

    # Generate 250 examples across all categories
    dataset = generate_evaluation_dataset(total_examples=250)

    # Generate examples for specific category
    dataset = generate_evaluation_dataset(
        total_examples=50,
        categories=[QueryCategory.DEFINITIONAL],
    )
"""

from __future__ import annotations

import itertools
import random
from typing import TYPE_CHECKING

from tests.benchmark.schemas import (
    BenchmarkExample,
    DifficultyLevel,
    ExpectedMetrics,
    ExpectedRouting,
    QueryCategory,
)
from tests.benchmark.templates import (
    ANALYTICAL_TEMPLATES,
    COMPARISON_TEMPLATES,
    CORE_CONCEPTS,
    DEFINITIONAL_TEMPLATES,
    EDGE_CASE_TEMPLATES,
    GROUND_TRUTH_HINTS,
    INDUSTRIES,
    INDUSTRY_STANDARDS,
    PROCEDURAL_TEMPLATES,
    RELATIONAL_TEMPLATES,
    STANDARD_SPECIFIC_TEMPLATES,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# =============================================================================
# ROUTING RULES
# =============================================================================

# Map query patterns to expected tool routing
ROUTING_RULES: dict[QueryCategory, list[ExpectedRouting]] = {
    QueryCategory.DEFINITIONAL: [
        ExpectedRouting.VECTOR_SEARCH,
        ExpectedRouting.LOOKUP_TERM,
    ],
    QueryCategory.RELATIONAL: [
        ExpectedRouting.GRAPH_ENRICHED,
        ExpectedRouting.EXPLORE_ENTITY,
    ],
    QueryCategory.PROCEDURAL: [
        ExpectedRouting.VECTOR_SEARCH,
        ExpectedRouting.HYBRID_SEARCH,
    ],
    QueryCategory.COMPARISON: [
        ExpectedRouting.HYBRID_SEARCH,
        ExpectedRouting.GRAPH_ENRICHED,
    ],
    QueryCategory.FACTUAL: [
        ExpectedRouting.TEXT2CYPHER,
        ExpectedRouting.LOOKUP_STANDARD,
        ExpectedRouting.STANDARDS_BY_INDUSTRY,
    ],
    QueryCategory.ANALYTICAL: [
        ExpectedRouting.GRAPH_ENRICHED,
        ExpectedRouting.CHAT,
    ],
    QueryCategory.EDGE_CASE: [
        ExpectedRouting.VECTOR_SEARCH,
        ExpectedRouting.CHAT,
    ],
}

# =============================================================================
# DIFFICULTY ASSIGNMENT
# =============================================================================


def _assign_difficulty(
    category: QueryCategory,
    question: str,
) -> DifficultyLevel:
    """Assign difficulty based on category and question complexity."""
    # Edge cases are typically hard
    if category == QueryCategory.EDGE_CASE:
        return DifficultyLevel.HARD

    # Multi-part questions are harder
    if " and " in question.lower() and "?" in question:
        word_count = len(question.split())
        if word_count > 20:
            return DifficultyLevel.EXPERT
        return DifficultyLevel.HARD

    # Standard-specific questions require more knowledge
    if any(std.lower() in question.lower() for std in INDUSTRY_STANDARDS):
        return DifficultyLevel.MEDIUM

    # Simple definitional questions are easy
    if category == QueryCategory.DEFINITIONAL:
        if question.startswith(("What is", "Define")):
            return DifficultyLevel.EASY

    # Analytical questions require reasoning
    if category == QueryCategory.ANALYTICAL:
        return DifficultyLevel.MEDIUM

    # Default based on category
    category_difficulty: dict[QueryCategory, DifficultyLevel] = {
        QueryCategory.DEFINITIONAL: DifficultyLevel.EASY,
        QueryCategory.RELATIONAL: DifficultyLevel.MEDIUM,
        QueryCategory.PROCEDURAL: DifficultyLevel.MEDIUM,
        QueryCategory.COMPARISON: DifficultyLevel.MEDIUM,
        QueryCategory.FACTUAL: DifficultyLevel.EASY,
        QueryCategory.ANALYTICAL: DifficultyLevel.HARD,
        QueryCategory.EDGE_CASE: DifficultyLevel.HARD,
    }
    return category_difficulty.get(category, DifficultyLevel.MEDIUM)


# =============================================================================
# GROUND TRUTH GENERATION
# =============================================================================


def _generate_ground_truth(
    question: str,
    category: QueryCategory,
    concepts: list[str],
) -> str:
    """Generate ground truth based on question and concepts."""
    # Check if we have a pre-defined hint for any concept
    for concept in concepts:
        if concept.lower() in GROUND_TRUTH_HINTS:
            return GROUND_TRUTH_HINTS[concept.lower()]

    # Generate generic ground truth based on category
    if category == QueryCategory.DEFINITIONAL:
        concept = concepts[0] if concepts else "this concept"
        return (
            f"The answer should define {concept} and explain its role in requirements management."
        )

    if category == QueryCategory.RELATIONAL:
        if len(concepts) >= 2:
            return f"The answer should explain how {concepts[0]} and {concepts[1]} are related."
        return "The answer should explain the relationship between the concepts."

    if category == QueryCategory.PROCEDURAL:
        concept = concepts[0] if concepts else "this process"
        return f"The answer should provide steps or best practices for implementing {concept}."

    if category == QueryCategory.COMPARISON:
        if len(concepts) >= 2:
            return (
                f"The answer should compare {concepts[0]} and {concepts[1]}, "
                "highlighting differences."
            )
        return "The answer should compare the concepts and highlight key differences."

    if category == QueryCategory.FACTUAL:
        return "The answer should provide specific factual information from the knowledge base."

    if category == QueryCategory.ANALYTICAL:
        return "The answer should provide analysis and reasoning about the topic."

    if category == QueryCategory.EDGE_CASE:
        return (
            "The system should handle this gracefully, either providing "
            "a best-effort answer or explaining limitations."
        )

    return "The answer should be relevant and accurate based on the knowledge base."


# =============================================================================
# EXAMPLE GENERATORS BY CATEGORY
# =============================================================================


def _generate_definitional_examples(count: int, seed: int = 42) -> list[BenchmarkExample]:
    """Generate definitional query examples."""
    rng = random.Random(seed)
    examples = []

    # Combine templates with concepts
    combinations = list(itertools.product(DEFINITIONAL_TEMPLATES, CORE_CONCEPTS))
    rng.shuffle(combinations)

    for i, (template, concept) in enumerate(combinations[:count]):
        question = template.format(concept=concept)
        examples.append(
            BenchmarkExample(
                id=f"def_{i:03d}",
                question=question,
                category=QueryCategory.DEFINITIONAL,
                difficulty=_assign_difficulty(QueryCategory.DEFINITIONAL, question),
                expected_tools=ROUTING_RULES[QueryCategory.DEFINITIONAL],
                ground_truth=_generate_ground_truth(
                    question, QueryCategory.DEFINITIONAL, [concept]
                ),
                expected_entities=[concept],
                tags=["definitional", "concept"],
            )
        )

    return examples


def _generate_relational_examples(count: int, seed: int = 42) -> list[BenchmarkExample]:
    """Generate relational query examples."""
    rng = random.Random(seed)
    examples = []

    # Create concept pairs
    concept_pairs = list(itertools.combinations(CORE_CONCEPTS[:15], 2))
    rng.shuffle(concept_pairs)

    for i, (concept1, concept2) in enumerate(concept_pairs[:count]):
        template = rng.choice(RELATIONAL_TEMPLATES)
        question = template.format(concept1=concept1, concept2=concept2)
        examples.append(
            BenchmarkExample(
                id=f"rel_{i:03d}",
                question=question,
                category=QueryCategory.RELATIONAL,
                difficulty=_assign_difficulty(QueryCategory.RELATIONAL, question),
                expected_tools=ROUTING_RULES[QueryCategory.RELATIONAL],
                ground_truth=_generate_ground_truth(
                    question, QueryCategory.RELATIONAL, [concept1, concept2]
                ),
                expected_entities=[concept1, concept2],
                tags=["relational", "multi-concept"],
            )
        )

    return examples


def _generate_procedural_examples(count: int, seed: int = 42) -> list[BenchmarkExample]:
    """Generate procedural query examples."""
    rng = random.Random(seed)
    examples = []

    combinations = list(itertools.product(PROCEDURAL_TEMPLATES, CORE_CONCEPTS))
    rng.shuffle(combinations)

    for i, (template, concept) in enumerate(combinations[:count]):
        question = template.format(concept=concept)
        examples.append(
            BenchmarkExample(
                id=f"proc_{i:03d}",
                question=question,
                category=QueryCategory.PROCEDURAL,
                difficulty=_assign_difficulty(QueryCategory.PROCEDURAL, question),
                expected_tools=ROUTING_RULES[QueryCategory.PROCEDURAL],
                ground_truth=_generate_ground_truth(question, QueryCategory.PROCEDURAL, [concept]),
                expected_entities=[concept],
                tags=["procedural", "how-to"],
            )
        )

    return examples


def _generate_comparison_examples(count: int, seed: int = 42) -> list[BenchmarkExample]:
    """Generate comparison query examples."""
    rng = random.Random(seed)
    examples = []

    # Meaningful comparison pairs
    comparison_pairs = [
        ("verification", "validation"),
        ("functional requirements", "non-functional requirements"),
        ("forward traceability", "backward traceability"),
        ("agile", "waterfall"),
        ("system requirements", "software requirements"),
        ("ISO 26262", "IEC 62304"),
        ("ASPICE", "CMMI"),
        ("requirements elicitation", "requirements analysis"),
    ]

    for i, (concept1, concept2) in enumerate(comparison_pairs[:count]):
        template = rng.choice(COMPARISON_TEMPLATES)
        question = template.format(concept1=concept1, concept2=concept2)
        examples.append(
            BenchmarkExample(
                id=f"comp_{i:03d}",
                question=question,
                category=QueryCategory.COMPARISON,
                difficulty=_assign_difficulty(QueryCategory.COMPARISON, question),
                expected_tools=ROUTING_RULES[QueryCategory.COMPARISON],
                ground_truth=_generate_ground_truth(
                    question, QueryCategory.COMPARISON, [concept1, concept2]
                ),
                expected_entities=[concept1, concept2],
                tags=["comparison", "analysis"],
            )
        )

    return examples


def _generate_factual_examples(count: int, seed: int = 42) -> list[BenchmarkExample]:
    """Generate factual query examples."""
    rng = random.Random(seed)
    examples = []

    # Standard-specific questions
    for i, standard in enumerate(INDUSTRY_STANDARDS[:count]):
        template = rng.choice(STANDARD_SPECIFIC_TEMPLATES)
        industry = rng.choice(INDUSTRIES)
        question = template.format(standard=standard, industry=industry)
        examples.append(
            BenchmarkExample(
                id=f"fact_{i:03d}",
                question=question,
                category=QueryCategory.FACTUAL,
                difficulty=_assign_difficulty(QueryCategory.FACTUAL, question),
                expected_tools=[
                    ExpectedRouting.LOOKUP_STANDARD,
                    ExpectedRouting.TEXT2CYPHER,
                ],
                ground_truth=_generate_ground_truth(question, QueryCategory.FACTUAL, [standard]),
                expected_standards=[standard],
                tags=["factual", "standards"],
            )
        )

    # Industry-specific questions
    for _j, industry in enumerate(INDUSTRIES[: count - len(examples)]):
        question = f"What standards apply to {industry}?"
        examples.append(
            BenchmarkExample(
                id=f"fact_{len(examples):03d}",
                question=question,
                category=QueryCategory.FACTUAL,
                difficulty=DifficultyLevel.EASY,
                expected_tools=[ExpectedRouting.STANDARDS_BY_INDUSTRY],
                ground_truth=f"The answer should list standards applicable to {industry}.",
                tags=["factual", "industry"],
                metadata={"industry": industry},
            )
        )

    return examples[:count]


def _generate_analytical_examples(count: int, seed: int = 42) -> list[BenchmarkExample]:
    """Generate analytical query examples."""
    rng = random.Random(seed)
    examples = []

    combinations = list(itertools.product(ANALYTICAL_TEMPLATES, CORE_CONCEPTS[:12]))
    rng.shuffle(combinations)

    for i, (template, concept) in enumerate(combinations[:count]):
        question = template.format(concept=concept)
        examples.append(
            BenchmarkExample(
                id=f"anal_{i:03d}",
                question=question,
                category=QueryCategory.ANALYTICAL,
                difficulty=_assign_difficulty(QueryCategory.ANALYTICAL, question),
                expected_tools=ROUTING_RULES[QueryCategory.ANALYTICAL],
                ground_truth=_generate_ground_truth(question, QueryCategory.ANALYTICAL, [concept]),
                expected_entities=[concept],
                tags=["analytical", "reasoning"],
            )
        )

    return examples


def _generate_edge_case_examples(count: int, seed: int = 42) -> list[BenchmarkExample]:
    """Generate edge case examples."""
    rng = random.Random(seed)
    examples = []

    # Fill in templates that have placeholders
    filled_templates = []
    for template in EDGE_CASE_TEMPLATES:
        if "{concept" in template or "{standard}" in template:
            # Fill with random values
            filled = template.format(
                concept=rng.choice(CORE_CONCEPTS),
                concept1=rng.choice(CORE_CONCEPTS),
                concept2=rng.choice(CORE_CONCEPTS),
                standard=rng.choice(INDUSTRY_STANDARDS),
            )
            filled_templates.append(filled)
        else:
            filled_templates.append(template)

    rng.shuffle(filled_templates)

    for i, question in enumerate(filled_templates[:count]):
        # Determine if this is out-of-domain
        out_of_domain = any(
            phrase in question.lower()
            for phrase in ["weather", "cook", "pasta", "machine learning"]
        )

        examples.append(
            BenchmarkExample(
                id=f"edge_{i:03d}",
                question=question,
                category=QueryCategory.EDGE_CASE,
                difficulty=DifficultyLevel.HARD,
                expected_tools=ROUTING_RULES[QueryCategory.EDGE_CASE],
                ground_truth=_generate_ground_truth(question, QueryCategory.EDGE_CASE, []),
                expected_metrics=ExpectedMetrics(
                    min_faithfulness=0.5,  # Lower threshold for edge cases
                    min_relevancy=0.5,
                    min_precision=0.4,
                    min_recall=0.4,
                ),
                tags=["edge-case", "out-of-domain" if out_of_domain else "ambiguous"],
                metadata={"out_of_domain": out_of_domain},
            )
        )

    return examples


# =============================================================================
# MAIN GENERATOR
# =============================================================================


def generate_evaluation_dataset(
    total_examples: int = 250,
    categories: Sequence[QueryCategory] | None = None,
    seed: int = 42,
    distribution: dict[QueryCategory, float] | None = None,
) -> list[BenchmarkExample]:
    """Generate a complete evaluation dataset.

    Args:
        total_examples: Total number of examples to generate.
        categories: Categories to include. If None, includes all.
        seed: Random seed for reproducibility.
        distribution: Custom distribution of examples per category.
            Values should sum to 1.0.

    Returns:
        List of BenchmarkExample instances.

    Example:
        # Generate 250 examples with default distribution
        dataset = generate_evaluation_dataset(total_examples=250)

        # Generate only definitional examples
        dataset = generate_evaluation_dataset(
            total_examples=50,
            categories=[QueryCategory.DEFINITIONAL],
        )

        # Custom distribution
        dataset = generate_evaluation_dataset(
            total_examples=100,
            distribution={
                QueryCategory.DEFINITIONAL: 0.3,
                QueryCategory.PROCEDURAL: 0.3,
                QueryCategory.FACTUAL: 0.4,
            },
        )
    """
    if categories is None:
        categories = list(QueryCategory)

    # Default distribution emphasizes common query types
    if distribution is None:
        distribution = {
            QueryCategory.DEFINITIONAL: 0.25,
            QueryCategory.RELATIONAL: 0.10,
            QueryCategory.PROCEDURAL: 0.20,
            QueryCategory.COMPARISON: 0.10,
            QueryCategory.FACTUAL: 0.15,
            QueryCategory.ANALYTICAL: 0.10,
            QueryCategory.EDGE_CASE: 0.10,
        }

    # Filter distribution to requested categories
    filtered_dist = {k: v for k, v in distribution.items() if k in categories}

    # Normalize distribution
    total_weight = sum(filtered_dist.values())
    normalized_dist = {k: v / total_weight for k, v in filtered_dist.items()}

    # Calculate examples per category
    examples_per_category = {
        cat: max(1, int(total_examples * weight)) for cat, weight in normalized_dist.items()
    }

    # Generator functions by category
    generators = {
        QueryCategory.DEFINITIONAL: _generate_definitional_examples,
        QueryCategory.RELATIONAL: _generate_relational_examples,
        QueryCategory.PROCEDURAL: _generate_procedural_examples,
        QueryCategory.COMPARISON: _generate_comparison_examples,
        QueryCategory.FACTUAL: _generate_factual_examples,
        QueryCategory.ANALYTICAL: _generate_analytical_examples,
        QueryCategory.EDGE_CASE: _generate_edge_case_examples,
    }

    # Generate examples
    all_examples: list[BenchmarkExample] = []
    for category, count in examples_per_category.items():
        if category in generators:
            examples = generators[category](count, seed)
            all_examples.extend(examples)

    # Shuffle for variety in test runs
    rng = random.Random(seed)
    rng.shuffle(all_examples)

    return all_examples[:total_examples]


def get_examples_by_category(
    dataset: list[BenchmarkExample],
    category: QueryCategory,
) -> list[BenchmarkExample]:
    """Filter dataset to specific category."""
    return [ex for ex in dataset if ex.category == category]


def get_examples_by_difficulty(
    dataset: list[BenchmarkExample],
    difficulty: DifficultyLevel,
) -> list[BenchmarkExample]:
    """Filter dataset to specific difficulty."""
    return [ex for ex in dataset if ex.difficulty == difficulty]


def get_examples_by_tag(
    dataset: list[BenchmarkExample],
    tag: str,
) -> list[BenchmarkExample]:
    """Filter dataset by tag."""
    return [ex for ex in dataset if tag in ex.tags]


__all__ = [
    "generate_evaluation_dataset",
    "get_examples_by_category",
    "get_examples_by_difficulty",
    "get_examples_by_tag",
]
