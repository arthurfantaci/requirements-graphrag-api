"""Evaluation datasets for GraphRAG quality assessment.

Provides sample evaluation data specific to requirements management
domain for testing retrieval and generation quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationSample:
    """A single evaluation sample for RAG testing.

    Attributes:
        question: The user's question.
        ground_truth: The expected/reference answer.
        contexts: Optional ground truth contexts for retrieval evaluation.
        metadata: Additional metadata about the sample.
    """

    question: str
    ground_truth: str
    contexts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def get_sample_evaluation_data() -> list[EvaluationSample]:
    """Get sample evaluation data for requirements management domain.

    Returns a curated set of Q&A pairs covering key topics from the
    Jama Software requirements management guide.

    Returns:
        List of EvaluationSample instances.
    """
    return [
        EvaluationSample(
            question="What is requirements traceability?",
            ground_truth=(
                "Requirements traceability is the ability to trace a requirement "
                "throughout its lifecycle, from origin through implementation and "
                "verification. It links requirements to their sources, design elements, "
                "test cases, and other artifacts to ensure complete coverage and "
                "impact analysis capability."
            ),
            contexts=[
                "Requirements traceability enables tracking requirements from origin "
                "to implementation, ensuring complete coverage."
            ],
            metadata={"topic": "traceability", "difficulty": "basic"},
        ),
        EvaluationSample(
            question="What are the key components of the V-model?",
            ground_truth=(
                "The V-model consists of development phases on the left side "
                "(requirements, design, implementation) and corresponding testing "
                "phases on the right side (unit testing, integration testing, "
                "system testing, acceptance testing). Each development phase has "
                "a corresponding verification/validation phase."
            ),
            contexts=[
                "The V-model shows the relationship between development and "
                "testing phases in a V-shaped diagram."
            ],
            metadata={"topic": "methodology", "difficulty": "intermediate"},
        ),
        EvaluationSample(
            question="How does ISO 26262 relate to requirements management?",
            ground_truth=(
                "ISO 26262 is the functional safety standard for automotive systems. "
                "It requires rigorous requirements management including hazard analysis, "
                "ASIL (Automotive Safety Integrity Level) classification, requirements "
                "traceability, and verification/validation activities to ensure "
                "safety-critical systems meet their safety goals."
            ),
            contexts=[
                "ISO 26262 defines ASIL levels and safety requirements for automotive "
                "electronic systems."
            ],
            metadata={"topic": "standards", "difficulty": "advanced"},
        ),
        EvaluationSample(
            question="What are the best practices for writing requirements?",
            ground_truth=(
                "Best practices for writing requirements include: making them atomic "
                "(one requirement per statement), unambiguous, testable/verifiable, "
                "complete, consistent, prioritized, and traceable. Requirements should "
                "use clear language, avoid implementation details unless necessary, "
                "and be reviewed by stakeholders."
            ),
            contexts=["Good requirements are atomic, unambiguous, testable, and traceable."],
            metadata={"topic": "best_practices", "difficulty": "basic"},
        ),
        EvaluationSample(
            question="What is the difference between verification and validation?",
            ground_truth=(
                "Verification answers 'Are we building the product right?' - it checks "
                "that the product meets specified requirements through reviews, testing, "
                "and analysis. Validation answers 'Are we building the right product?' - "
                "it confirms the product meets user needs and intended use through "
                "user acceptance testing and stakeholder feedback."
            ),
            contexts=[
                "Verification checks conformance to requirements; validation confirms "
                "fitness for intended use."
            ],
            metadata={"topic": "v_and_v", "difficulty": "intermediate"},
        ),
        EvaluationSample(
            question="What is impact analysis in requirements management?",
            ground_truth=(
                "Impact analysis is the process of identifying the potential consequences "
                "of a change to a requirement. It uses traceability links to determine "
                "which downstream artifacts (design, code, tests) would be affected by "
                "a proposed change, helping teams assess risk, effort, and make "
                "informed decisions about changes."
            ),
            contexts=[
                "Impact analysis uses traceability to identify affected artifacts "
                "when requirements change."
            ],
            metadata={"topic": "traceability", "difficulty": "intermediate"},
        ),
        EvaluationSample(
            question="What is DO-178C and when is it applicable?",
            ground_truth=(
                "DO-178C is the software certification standard for airborne systems "
                "and equipment. It is applicable to software used in aircraft and "
                "avionics systems. The standard defines development and verification "
                "processes based on Design Assurance Levels (DAL A through E), with "
                "DAL A being the most stringent for catastrophic failure conditions."
            ),
            contexts=[
                "DO-178C governs software development for airborne systems with "
                "DAL levels determining rigor."
            ],
            metadata={"topic": "standards", "difficulty": "advanced"},
        ),
        EvaluationSample(
            question="What are ASPICE process areas?",
            ground_truth=(
                "ASPICE (Automotive SPICE) defines process areas including: "
                "System Requirements Analysis (SYS.2), Software Requirements Analysis "
                "(SWE.1), Software Architectural Design (SWE.2), Software Detailed "
                "Design (SWE.3), Software Unit Verification (SWE.4), Software "
                "Integration (SWE.5), and Software Qualification Test (SWE.6). "
                "Each process area has capability levels from 0 to 5."
            ),
            contexts=[
                "ASPICE defines engineering process areas for automotive software "
                "development with capability levels."
            ],
            metadata={"topic": "standards", "difficulty": "advanced"},
        ),
    ]


def create_evaluation_dataset(
    samples: list[EvaluationSample] | None = None,
    *,
    filter_topic: str | None = None,
    filter_difficulty: str | None = None,
) -> list[dict[str, Any]]:
    """Create an evaluation dataset in RAGAS-compatible format.

    Args:
        samples: List of evaluation samples. Uses default if not provided.
        filter_topic: Optional filter by topic metadata.
        filter_difficulty: Optional filter by difficulty metadata.

    Returns:
        List of dictionaries in RAGAS evaluation format.
    """
    if samples is None:
        samples = get_sample_evaluation_data()

    # Apply filters
    filtered = samples
    if filter_topic:
        filtered = [s for s in filtered if s.metadata.get("topic") == filter_topic]
    if filter_difficulty:
        filtered = [s for s in filtered if s.metadata.get("difficulty") == filter_difficulty]

    # Convert to RAGAS format
    return [
        {
            "question": sample.question,
            "ground_truth": sample.ground_truth,
            "contexts": sample.contexts,
            "metadata": sample.metadata,
        }
        for sample in filtered
    ]
