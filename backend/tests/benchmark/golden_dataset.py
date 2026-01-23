"""Golden dataset for RAG evaluation.

Contains curated question-answer pairs for evaluating the GraphRAG pipeline
on requirements management and traceability topics.

The dataset is organized by:
- Category: definition, concept, process, standard, comparison
- Difficulty: easy, medium, hard
- Must-pass: Critical examples that must score above threshold

Usage:
    from tests.benchmark.golden_dataset import GOLDEN_DATASET, get_must_pass_examples

    for example in GOLDEN_DATASET:
        print(f"{example.id}: {example.question}")

    must_pass = get_must_pass_examples()
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GoldenExample:
    """A curated evaluation example from the golden dataset."""

    id: str
    question: str
    ground_truth: str
    expected_entities: list[str] = field(default_factory=list)
    expected_standards: list[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"
    must_pass: bool = False


# =============================================================================
# GOLDEN DATASET
# =============================================================================

GOLDEN_DATASET: tuple[GoldenExample, ...] = (
    # -------------------------------------------------------------------------
    # DEFINITIONS (Easy)
    # -------------------------------------------------------------------------
    GoldenExample(
        id="def-001",
        question="What is requirements traceability?",
        ground_truth=(
            "Requirements traceability is the ability to describe and follow the life "
            "of a requirement in both a forward and backward direction through the "
            "development lifecycle. It links requirements to their origins, design "
            "elements, implementation, and verification activities."
        ),
        expected_entities=["requirements traceability", "traceability matrix"],
        category="definition",
        difficulty="easy",
        must_pass=True,
    ),
    GoldenExample(
        id="def-002",
        question="What is a traceability matrix?",
        ground_truth=(
            "A traceability matrix is a document that correlates requirements to "
            "their origins, design elements, test cases, and other artifacts. It "
            "provides a visual representation of relationships between items across "
            "the development lifecycle."
        ),
        expected_entities=["traceability matrix", "requirements"],
        category="definition",
        difficulty="easy",
        must_pass=True,
    ),
    GoldenExample(
        id="def-003",
        question="What are derived requirements?",
        ground_truth=(
            "Derived requirements are requirements that are not explicitly stated "
            "in the original specification but are implied by or inferred from other "
            "requirements. They often emerge during the design process when system "
            "requirements are decomposed into lower-level specifications."
        ),
        expected_entities=["derived requirements", "requirements decomposition"],
        category="definition",
        difficulty="easy",
        must_pass=True,
    ),
    GoldenExample(
        id="def-004",
        question="What is impact analysis?",
        ground_truth=(
            "Impact analysis is the process of identifying the potential consequences "
            "of a change and estimating what needs to be modified to accomplish that "
            "change. In requirements management, it uses traceability links to assess "
            "how changes to one requirement affect related items."
        ),
        expected_entities=["impact analysis", "change management"],
        category="definition",
        difficulty="easy",
        must_pass=True,
    ),
    GoldenExample(
        id="def-005",
        question="What is a suspect link?",
        ground_truth=(
            "A suspect link is a traceability link that may be invalid or outdated "
            "due to changes in one of the linked items. When a requirement changes, "
            "its trace links are marked as suspect until they are reviewed and verified."
        ),
        expected_entities=["suspect links", "trace links"],
        category="definition",
        difficulty="easy",
        must_pass=True,
    ),
    # -------------------------------------------------------------------------
    # CONCEPTS (Medium)
    # -------------------------------------------------------------------------
    GoldenExample(
        id="con-001",
        question="Why is bidirectional traceability important?",
        ground_truth=(
            "Bidirectional traceability is important because it enables both forward "
            "and backward tracing. Forward tracing verifies that all requirements are "
            "implemented, while backward tracing confirms that all implementation "
            "artifacts tie back to requirements. This supports impact analysis, "
            "coverage analysis, and regulatory compliance."
        ),
        expected_entities=["bidirectional traceability", "coverage analysis"],
        category="concept",
        difficulty="medium",
        must_pass=True,
    ),
    GoldenExample(
        id="con-002",
        question="How does traceability support change management?",
        ground_truth=(
            "Traceability supports change management by enabling impact analysis - "
            "when a requirement changes, trace links reveal all affected design "
            "elements, code modules, and test cases. This helps teams assess change "
            "scope, estimate effort, and ensure all impacted items are updated."
        ),
        expected_entities=["change management", "impact analysis"],
        category="concept",
        difficulty="medium",
        must_pass=True,
    ),
    GoldenExample(
        id="con-003",
        question="What is requirements decomposition?",
        ground_truth=(
            "Requirements decomposition is the process of breaking down high-level "
            "requirements into more detailed, lower-level requirements. This creates "
            "a hierarchy where system requirements flow down to subsystem and "
            "component requirements, maintaining traceability throughout."
        ),
        expected_entities=["requirements decomposition", "requirements allocation"],
        category="concept",
        difficulty="medium",
        must_pass=True,
    ),
    GoldenExample(
        id="con-004",
        question="What is coverage analysis in requirements management?",
        ground_truth=(
            "Coverage analysis examines whether all requirements have been adequately "
            "addressed in design, implementation, and testing. It uses traceability "
            "to identify gaps where requirements lack downstream artifacts and ensures "
            "complete verification of the system."
        ),
        expected_entities=["coverage analysis", "verification and validation"],
        category="concept",
        difficulty="medium",
        must_pass=True,
    ),
    GoldenExample(
        id="con-005",
        question="What role do requirements attributes play in traceability?",
        ground_truth=(
            "Requirements attributes provide metadata that supports traceability and "
            "management. Common attributes include ID, status, priority, owner, "
            "rationale, and verification method. These enable filtering, reporting, "
            "and tracking requirements throughout the lifecycle."
        ),
        expected_entities=["requirements attributes", "requirements management"],
        category="concept",
        difficulty="medium",
        must_pass=True,
    ),
    # -------------------------------------------------------------------------
    # PROCESSES (Medium)
    # -------------------------------------------------------------------------
    GoldenExample(
        id="pro-001",
        question="What are the key steps in establishing traceability?",
        ground_truth=(
            "Key steps include: 1) Define traceability strategy and scope, "
            "2) Identify trace link types (derives from, satisfies, verifies), "
            "3) Establish baselines for requirements, "
            "4) Create and maintain trace links, "
            "5) Review and validate links regularly, "
            "6) Generate traceability reports for stakeholders."
        ),
        expected_entities=["traceability matrix", "baseline management"],
        category="process",
        difficulty="medium",
        must_pass=True,
    ),
    GoldenExample(
        id="pro-002",
        question="How do you maintain traceability over time?",
        ground_truth=(
            "Maintaining traceability requires: regular reviews of trace links, "
            "updating links when requirements change, marking suspect links for "
            "review, using tools to automate link management, establishing change "
            "control processes, and training team members on traceability practices."
        ),
        expected_entities=["suspect links", "change management"],
        category="process",
        difficulty="medium",
        must_pass=True,
    ),
    GoldenExample(
        id="pro-003",
        question="What is the baseline management process?",
        ground_truth=(
            "Baseline management involves capturing a snapshot of requirements at "
            "a specific point in time, controlling changes through formal review "
            "and approval, maintaining history of baselines, and comparing current "
            "state against baselines to track evolution."
        ),
        expected_entities=["baseline management", "change management"],
        category="process",
        difficulty="medium",
        must_pass=True,
    ),
    GoldenExample(
        id="pro-004",
        question="How do verification and validation use traceability?",
        ground_truth=(
            "Verification uses traceability to confirm each requirement has "
            "corresponding test cases and that tests map back to requirements. "
            "Validation uses traceability to ensure the implemented system meets "
            "stakeholder needs by tracing from user requirements through to "
            "acceptance tests."
        ),
        expected_entities=["verification and validation", "test coverage"],
        category="process",
        difficulty="medium",
        must_pass=True,
    ),
    GoldenExample(
        id="pro-005",
        question="What is the requirements review process?",
        ground_truth=(
            "Requirements review involves examining requirements for completeness, "
            "correctness, consistency, and testability. Reviews may be informal "
            "(walkthroughs) or formal (inspections). Traceability helps reviewers "
            "check that all source requirements have been addressed."
        ),
        expected_entities=["review status", "approval workflow"],
        category="process",
        difficulty="medium",
        must_pass=True,
    ),
    # -------------------------------------------------------------------------
    # STANDARDS (Hard)
    # -------------------------------------------------------------------------
    GoldenExample(
        id="std-001",
        question="How does ISO 26262 address requirements traceability?",
        ground_truth=(
            "ISO 26262 requires bidirectional traceability between safety goals, "
            "functional safety requirements, technical safety requirements, and "
            "verification results. It mandates traceability matrices demonstrating "
            "complete coverage and impact analysis for safety-related changes."
        ),
        expected_entities=["ISO 26262", "safety-critical"],
        expected_standards=["ISO 26262"],
        category="standard",
        difficulty="hard",
        must_pass=True,
    ),
    GoldenExample(
        id="std-002",
        question="What traceability requirements exist in DO-178C?",
        ground_truth=(
            "DO-178C requires traceability from system requirements to software "
            "requirements to design to code and to test cases. It emphasizes "
            "complete coverage, bidirectional tracing, and verification of derived "
            "requirements. Higher design assurance levels require more rigorous "
            "traceability practices."
        ),
        expected_entities=["DO-178C", "safety-critical"],
        expected_standards=["DO-178C"],
        category="standard",
        difficulty="hard",
        must_pass=True,
    ),
    GoldenExample(
        id="std-003",
        question="How does IEC 62304 handle traceability for medical devices?",
        ground_truth=(
            "IEC 62304 requires traceability between system requirements, software "
            "requirements, software architecture, software units, and testing. "
            "The standard mandates maintaining traceability throughout the product "
            "lifecycle and during maintenance activities."
        ),
        expected_entities=["IEC 62304", "medical device"],
        expected_standards=["IEC 62304"],
        category="standard",
        difficulty="hard",
        must_pass=True,
    ),
    GoldenExample(
        id="std-004",
        question="What does Automotive SPICE say about traceability?",
        ground_truth=(
            "Automotive SPICE (ASPICE) defines traceability requirements in its "
            "process reference model. It requires bidirectional traceability "
            "between customer requirements, system requirements, and software "
            "requirements, with consistency and coverage verification at each level."
        ),
        expected_entities=["ASPICE", "automotive"],
        expected_standards=["ASPICE"],
        category="standard",
        difficulty="hard",
        must_pass=True,
    ),
    GoldenExample(
        id="std-005",
        question="How does CMMI address requirements traceability?",
        ground_truth=(
            "CMMI addresses traceability through the Requirements Management (REQM) "
            "process area, which requires maintaining bidirectional traceability "
            "among requirements and work products. Higher maturity levels require "
            "more comprehensive traceability practices and metrics."
        ),
        expected_entities=["CMMI", "requirements management"],
        expected_standards=["CMMI"],
        category="standard",
        difficulty="hard",
        must_pass=True,
    ),
    # -------------------------------------------------------------------------
    # COMPARISONS (Hard)
    # -------------------------------------------------------------------------
    GoldenExample(
        id="cmp-001",
        question="What is the difference between forward and backward traceability?",
        ground_truth=(
            "Forward traceability tracks requirements from origin through design, "
            "implementation, and testing to verify complete implementation. "
            "Backward traceability traces from deliverables back to requirements "
            "to confirm all work products tie to valid requirements and identify "
            "gold plating."
        ),
        expected_entities=["bidirectional traceability", "coverage analysis"],
        category="comparison",
        difficulty="hard",
        must_pass=True,
    ),
    GoldenExample(
        id="cmp-002",
        question="How do horizontal and vertical traceability differ?",
        ground_truth=(
            "Vertical traceability links requirements across abstraction levels - "
            "from user requirements to system requirements to component requirements. "
            "Horizontal traceability links artifacts at the same level - such as "
            "requirements to test cases or design to code at the same detail level."
        ),
        expected_entities=["requirements decomposition", "traceability matrix"],
        category="comparison",
        difficulty="hard",
        must_pass=False,
    ),
    GoldenExample(
        id="cmp-003",
        question="What distinguishes verification from validation in requirements?",
        ground_truth=(
            "Verification confirms that the product is built correctly according "
            "to specifications ('Are we building the product right?'). Validation "
            "confirms the product meets stakeholder needs and intended use "
            "('Are we building the right product?'). Both use traceability but "
            "focus on different aspects."
        ),
        expected_entities=["verification and validation"],
        category="comparison",
        difficulty="hard",
        must_pass=True,
    ),
    GoldenExample(
        id="cmp-004",
        question="How do derived and allocated requirements differ?",
        ground_truth=(
            "Derived requirements emerge from analysis or design decisions and are "
            "not directly traceable to source requirements. Allocated requirements "
            "are explicitly assigned from higher-level requirements to specific "
            "components or subsystems, maintaining direct trace links."
        ),
        expected_entities=["derived requirements", "requirements allocation"],
        category="comparison",
        difficulty="hard",
        must_pass=False,
    ),
    GoldenExample(
        id="cmp-005",
        question="What is the difference between a requirement and a specification?",
        ground_truth=(
            "A requirement expresses a need or constraint that a system must satisfy. "
            "A specification is a detailed description of how the requirement will "
            "be met, including technical details and design decisions. Requirements "
            "focus on 'what' while specifications focus on 'how'."
        ),
        expected_entities=["requirements", "specifications"],
        category="comparison",
        difficulty="hard",
        must_pass=False,
    ),
)


def get_must_pass_examples() -> list[GoldenExample]:
    """Get examples that must pass evaluation threshold.

    Returns:
        List of critical examples that must score above threshold.
    """
    return [ex for ex in GOLDEN_DATASET if ex.must_pass]


def get_examples_by_category(category: str) -> list[GoldenExample]:
    """Get examples filtered by category.

    Args:
        category: Category to filter by (definition, concept, process, standard, comparison).

    Returns:
        List of examples in the specified category.
    """
    return [ex for ex in GOLDEN_DATASET if ex.category == category]


def get_examples_by_difficulty(difficulty: str) -> list[GoldenExample]:
    """Get examples filtered by difficulty.

    Args:
        difficulty: Difficulty to filter by (easy, medium, hard).

    Returns:
        List of examples with the specified difficulty.
    """
    return [ex for ex in GOLDEN_DATASET if ex.difficulty == difficulty]
