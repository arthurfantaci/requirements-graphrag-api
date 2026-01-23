"""Evaluation dataset generator.

Generates synthetic evaluation examples by combining templates with
domain-specific entities and variations. Used to augment the golden
dataset for comprehensive benchmark coverage.

Usage:
    from tests.benchmark.generator import generate_evaluation_dataset

    examples = generate_evaluation_dataset(total_examples=100)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Question templates with placeholders
QUESTION_TEMPLATES: tuple[tuple[str, str, str], ...] = (
    # (template, category, difficulty)
    ("What is {entity}?", "definition", "easy"),
    ("Define {entity} in requirements management.", "definition", "easy"),
    ("How does {entity} work?", "concept", "medium"),
    ("Why is {entity} important?", "concept", "medium"),
    ("What are the benefits of {entity}?", "concept", "medium"),
    ("How do you implement {entity}?", "process", "medium"),
    ("What tools support {entity}?", "tool", "medium"),
    ("How does {standard} address {entity}?", "standard", "hard"),
    ("What are best practices for {entity}?", "best_practice", "hard"),
    ("How do {entity1} and {entity2} differ?", "comparison", "hard"),
)

# Domain entities for template substitution
ENTITIES: tuple[str, ...] = (
    "requirements traceability",
    "traceability matrix",
    "impact analysis",
    "change management",
    "baseline management",
    "requirements decomposition",
    "derived requirements",
    "bidirectional traceability",
    "coverage analysis",
    "suspect links",
    "requirements attributes",
    "verification and validation",
    "requirements allocation",
    "test coverage",
    "compliance artifacts",
)

STANDARDS: tuple[str, ...] = (
    "ISO 26262",
    "IEC 62304",
    "DO-178C",
    "ASPICE",
    "CMMI",
    "ISO 29148",
    "IEEE 830",
)

# Answer template components
ANSWER_COMPONENTS: dict[str, list[str]] = {
    "definition": [
        "{entity} is a key concept in requirements management that enables",
        "{entity} refers to the practice of",
        "In requirements engineering, {entity} means",
    ],
    "benefits": [
        "helps teams manage complexity",
        "supports regulatory compliance",
        "enables change impact analysis",
        "improves quality assurance",
        "facilitates stakeholder communication",
    ],
    "processes": [
        "establish clear procedures",
        "maintain documentation",
        "perform regular reviews",
        "use appropriate tooling",
        "train team members",
    ],
}


@dataclass
class GeneratedExample:
    """A generated evaluation example."""

    id: str
    question: str
    ground_truth: str
    expected_entities: list[str] = field(default_factory=list)
    expected_standards: list[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"
    must_pass: bool = False


def _generate_ground_truth(entity: str, category: str) -> str:
    """Generate a synthetic ground truth answer.

    Args:
        entity: The main entity being discussed.
        category: Question category.

    Returns:
        Generated ground truth text.
    """
    templates = ANSWER_COMPONENTS.get("definition", [])
    if not templates:
        return f"{entity} is an important concept in requirements management."

    intro = random.choice(templates).format(entity=entity)
    benefits = random.sample(ANSWER_COMPONENTS["benefits"], k=2)
    processes = random.sample(ANSWER_COMPONENTS["processes"], k=2)

    return (
        f"{intro} {benefits[0]} and {benefits[1]}. "
        f"Key practices include: {processes[0]} and {processes[1]}."
    )


def generate_evaluation_dataset(
    total_examples: int = 50,
    seed: int | None = None,
) -> list[GeneratedExample]:
    """Generate synthetic evaluation examples.

    Creates evaluation examples by combining question templates with
    domain entities. Used to augment the golden dataset.

    Args:
        total_examples: Number of examples to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of generated evaluation examples.
    """
    if seed is not None:
        random.seed(seed)

    examples: list[GeneratedExample] = []

    for i in range(total_examples):
        # Select template and entity
        template, category, difficulty = random.choice(QUESTION_TEMPLATES)
        entity = random.choice(ENTITIES)

        # Handle different template types
        if "{entity1}" in template and "{entity2}" in template:
            # Comparison template
            entities = random.sample(ENTITIES, k=2)
            question = template.format(entity1=entities[0], entity2=entities[1])
            expected_entities = list(entities)
        elif "{standard}" in template:
            # Standard template
            standard = random.choice(STANDARDS)
            question = template.format(standard=standard, entity=entity)
            expected_entities = [entity]
        else:
            # Simple template
            question = template.format(entity=entity)
            expected_entities = [entity]

        # Generate ground truth
        ground_truth = _generate_ground_truth(entity, category)

        examples.append(
            GeneratedExample(
                id=f"gen-{i + 1:04d}",
                question=question,
                ground_truth=ground_truth,
                expected_entities=expected_entities,
                category=category,
                difficulty=difficulty,
                must_pass=False,
            )
        )

    return examples
