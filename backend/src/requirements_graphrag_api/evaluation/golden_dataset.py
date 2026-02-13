"""Hub-First golden dataset for RAGAS evaluation.

Follows the same pattern as the prompt catalog (prompts/catalog.py):
- LangSmith is the source of truth (users edit in the UI)
- Local examples serve as a versioned fallback when LangSmith is unreachable
- get_golden_examples() tries LangSmith first, then falls back to local

Usage:
    from requirements_graphrag_api.evaluation.golden_dataset import (
        get_golden_examples,
        DATASET_NAME,
    )

    # Hub-First: tries LangSmith, falls back to local
    examples = await get_golden_examples()

    # Force local-only (for CI Tier 1 / offline)
    examples = await get_golden_examples(use_langsmith=False)
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Centralized dataset name — used by all scripts and CI.
DATASET_NAME = "graphrag-rag-golden"


@dataclass
class GoldenExample:
    """A curated evaluation example.

    Attributes:
        id: Unique identifier (e.g., "def-001", "str-002").
        question: The user question to evaluate.
        expected_answer: Ground-truth answer for comparison.
        expected_entities: Entities expected in retrieved context.
        expected_standards: Standards expected to be referenced.
        intent: Query intent ("explanatory" or "structured").
        category: Topic category for stratified analysis.
        difficulty: Difficulty rating ("easy", "medium", "hard").
        must_pass: Whether this example must score above threshold.
    """

    id: str
    question: str
    expected_answer: str
    expected_entities: list[str] = field(default_factory=list)
    expected_standards: list[str] = field(default_factory=list)
    intent: str = "explanatory"
    category: str = "general"
    difficulty: str = "medium"
    must_pass: bool = False

    @property
    def vector(self) -> str:
        """Return the evaluation vector for this example."""
        return self.intent if self.intent in ("explanatory", "structured") else "explanatory"

    def to_langsmith_inputs(self) -> dict[str, Any]:
        """Return just the inputs dict for LangSmith dataset creation."""
        return {"question": self.question}

    def to_langsmith(self) -> dict[str, Any]:
        """Convert to LangSmith example format (inputs/outputs/metadata)."""
        return {
            "inputs": {"question": self.question},
            "outputs": {
                "expected_answer": self.expected_answer,
                "expected_entities": self.expected_entities,
                "intent": self.intent,
            },
            "metadata": {
                "id": self.id,
                "difficulty": self.difficulty,
                "category": self.category,
                "must_pass": self.must_pass,
                "expected_standards": self.expected_standards,
            },
        }

    @classmethod
    def from_langsmith(
        cls,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> GoldenExample:
        """Create from LangSmith example data."""
        meta = metadata or {}
        return cls(
            id=meta.get("id", ""),
            question=inputs.get("question", ""),
            expected_answer=(
                outputs.get("expected_answer")
                or outputs.get("ground_truth")
                or outputs.get("answer", "")
            ),
            expected_entities=outputs.get("expected_entities", []),
            expected_standards=meta.get("expected_standards", []),
            intent=outputs.get("intent", "explanatory"),
            category=meta.get("category", meta.get("domain", "general")),
            difficulty=meta.get("difficulty", "medium"),
            must_pass=meta.get("must_pass", False),
        )


@dataclass
class ConversationalExample:
    """A curated conversational evaluation example with multi-turn history.

    Unlike GoldenExample (single-turn RAG), this captures conversation
    context: prior history, a follow-up question, and expected references
    the model should retain from earlier turns.

    Attributes:
        id: Unique identifier (e.g., "conv-001").
        conversation_history: Prior turns as a formatted string.
        question: The follow-up question in this turn.
        expected_answer: Ground-truth answer for comparison.
        expected_references: Topics/facts from history the answer should retain.
        category: Topic category for stratified analysis.
        difficulty: Difficulty rating ("easy", "medium", "hard").
        must_pass: Whether this example must score above threshold.
    """

    id: str
    conversation_history: str
    question: str
    expected_answer: str
    expected_references: list[str] = field(default_factory=list)
    category: str = "conversation"
    difficulty: str = "medium"
    must_pass: bool = False

    @property
    def vector(self) -> str:
        """Return the evaluation vector."""
        return "conversational"

    def to_langsmith(self) -> dict[str, Any]:
        """Convert to LangSmith example format."""
        return {
            "inputs": {
                "question": self.question,
                "history": self.conversation_history,
            },
            "outputs": {
                "expected_answer": self.expected_answer,
                "expected_references": self.expected_references,
            },
            "metadata": {
                "id": self.id,
                "difficulty": self.difficulty,
                "category": self.category,
                "must_pass": self.must_pass,
            },
        }

    @classmethod
    def from_langsmith(
        cls,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> ConversationalExample:
        """Create from LangSmith example data."""
        meta = metadata or {}
        return cls(
            id=meta.get("id", ""),
            conversation_history=inputs.get("history", ""),
            question=inputs.get("question", ""),
            expected_answer=(outputs.get("expected_answer") or outputs.get("answer", "")),
            expected_references=outputs.get("expected_references", []),
            category=meta.get("category", "conversation"),
            difficulty=meta.get("difficulty", "medium"),
            must_pass=meta.get("must_pass", False),
        )


# =============================================================================
# LOCAL FALLBACK EXAMPLES
#
# These serve as the offline fallback when LangSmith is unreachable.
# To sync from LangSmith: uv run python scripts/create_golden_dataset.py --export
# =============================================================================

GOLDEN_EXAMPLES: tuple[GoldenExample, ...] = (
    # -------------------------------------------------------------------------
    # DEFINITIONS (Easy)
    # -------------------------------------------------------------------------
    GoldenExample(
        id="def-001",
        question="What is requirements traceability?",
        expected_answer=(
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
        expected_answer=(
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
        expected_answer=(
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
        expected_answer=(
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
        expected_answer=(
            "A suspect link is a traceability link that may be invalid or outdated "
            "due to changes in one of the linked items. When a requirement changes, "
            "its trace links are marked as suspect until they are reviewed and verified."
        ),
        expected_entities=["suspect links", "trace links"],
        category="definition",
        difficulty="easy",
        must_pass=True,
    ),
    GoldenExample(
        id="def-006",
        question="What is a requirements management tool?",
        expected_answer=(
            "A requirements management tool is software that helps teams capture, "
            "organize, track, and manage requirements throughout a product's lifecycle. "
            "These tools provide capabilities like requirements authoring, traceability "
            "matrices, change management, collaboration features, and reporting. "
            "Examples include Jama Connect, IBM DOORS, and Helix RM."
        ),
        expected_entities=["requirements management", "Jama Connect", "IBM DOORS"],
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
        expected_answer=(
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
        expected_answer=(
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
        expected_answer=(
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
        expected_answer=(
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
        expected_answer=(
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
    GoldenExample(
        id="con-006",
        question="Why is change management important in requirements?",
        expected_answer=(
            "Change management is critical because requirements evolve throughout "
            "a project. Without proper change management: scope creep goes "
            "uncontrolled, impact of changes isn't analyzed, stakeholders "
            "aren't informed, traceability breaks down, and testing coverage gaps "
            "emerge. Effective change management includes change request processes, "
            "impact analysis, approval workflows, version control, and communication "
            "to affected parties."
        ),
        expected_entities=["change management", "impact analysis"],
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
        expected_answer=(
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
        expected_answer=(
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
        expected_answer=(
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
        expected_answer=(
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
        expected_answer=(
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
    GoldenExample(
        id="pro-006",
        question="What are best practices for writing good requirements?",
        expected_answer=(
            "Best practices for writing requirements include: 1) Be atomic - one "
            "requirement per statement. 2) Be unambiguous - use precise language. "
            "3) Be verifiable - each requirement must be testable. 4) Be complete - "
            "include all necessary information. 5) Be consistent - no contradictions. "
            "6) Use 'shall' for mandatory requirements. 7) Avoid implementation details "
            "in functional requirements. 8) Include acceptance criteria. 9) Assign "
            "unique identifiers for traceability. 10) Review with stakeholders."
        ),
        expected_entities=["requirements", "acceptance criteria"],
        category="process",
        difficulty="medium",
        must_pass=True,
    ),
    GoldenExample(
        id="pro-007",
        question="How do I implement bidirectional traceability in my project?",
        expected_answer=(
            "To implement bidirectional traceability: 1) Define trace link types "
            "(derives from, satisfies, verifies, implements). 2) Establish links "
            "from high-level requirements to derived requirements (forward traceability) "
            "and from implementation/tests back to requirements (backward traceability). "
            "3) Use a requirements management tool to maintain links. 4) Create "
            "traceability matrices to visualize coverage. 5) Regularly review and "
            "update links during change management. 6) Use traceability reports for "
            "impact analysis and coverage verification."
        ),
        expected_entities=["bidirectional traceability", "coverage analysis"],
        category="process",
        difficulty="medium",
        must_pass=True,
    ),
    GoldenExample(
        id="pro-008",
        question=("How do I set up a requirements management process for a regulated industry?"),
        expected_answer=(
            "Setting up RM for regulated industries requires: 1) Identify applicable "
            "standards (ISO 26262, IEC 62304, DO-178C, etc.). 2) Define requirements "
            "types and hierarchy (stakeholder, system, software, hardware). "
            "3) Establish traceability strategy covering all lifecycle phases. "
            "4) Implement formal review and approval workflows. 5) Select compliant "
            "tooling with audit trails. 6) Define change control process with impact "
            "analysis. 7) Create verification/validation strategy. 8) Plan for "
            "regulatory audits with documentation. 9) Train team on processes and tools."
        ),
        expected_entities=["ISO 26262", "IEC 62304", "DO-178C"],
        expected_standards=["ISO 26262", "IEC 62304", "DO-178C"],
        category="process",
        difficulty="hard",
        must_pass=True,
    ),
    GoldenExample(
        id="pro-009",
        question="I need to improve our requirements review process. Any suggestions?",
        expected_answer=(
            "To improve requirements reviews: 1) Use structured review checklists "
            "covering completeness, consistency, and testability. 2) Conduct reviews "
            "in small batches rather than large documents. 3) Include diverse "
            "stakeholders (developers, testers, domain experts). 4) Use collaborative "
            "review tools for asynchronous input. 5) Track review metrics (defects "
            "found, review time). 6) Implement different review types: peer review, "
            "formal inspection, walkthrough. 7) Review traceability links along with "
            "requirements. 8) Document decisions and action items."
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
        expected_answer=(
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
        expected_answer=(
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
        expected_answer=(
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
        expected_answer=(
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
        expected_answer=(
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
    GoldenExample(
        id="std-006",
        question="What is ISO 26262 and how does it relate to requirements?",
        expected_answer=(
            "ISO 26262 is the international standard for functional safety of "
            "electrical and electronic systems in road vehicles. It defines "
            "Automotive Safety Integrity Levels (ASIL A-D) and requires rigorous "
            "requirements management including: functional safety requirements, "
            "technical safety requirements, hardware/software safety requirements, "
            "bidirectional traceability, verification and validation evidence, "
            "and formal safety analysis documentation."
        ),
        expected_entities=["ISO 26262", "ASIL", "functional safety"],
        expected_standards=["ISO 26262"],
        category="standard",
        difficulty="medium",
        must_pass=True,
    ),
    GoldenExample(
        id="std-007",
        question="What standards apply to medical device software development?",
        expected_answer=(
            "Key standards for medical device software include: IEC 62304 "
            "(Medical device software lifecycle), which defines software safety "
            "classes A/B/C and required documentation. ISO 14971 for risk "
            "management. FDA 21 CFR Part 820 (Quality System Regulation) for "
            "US market. IEC 62366 for usability. These require comprehensive "
            "requirements documentation, traceability, risk analysis, and "
            "verification/validation evidence."
        ),
        expected_entities=["IEC 62304", "ISO 14971", "medical device"],
        expected_standards=["IEC 62304", "ISO 14971"],
        category="standard",
        difficulty="medium",
        must_pass=True,
    ),
    # -------------------------------------------------------------------------
    # COMPARISONS (Hard)
    # -------------------------------------------------------------------------
    GoldenExample(
        id="cmp-001",
        question="What is the difference between forward and backward traceability?",
        expected_answer=(
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
        expected_answer=(
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
        question="What is the difference between verification and validation?",
        expected_answer=(
            "Verification confirms that the product is built correctly according "
            "to specifications ('Are we building the product right?'). Validation "
            "confirms the product meets stakeholder needs and intended use "
            "('Are we building the right product?'). Both use traceability but "
            "focus on different aspects."
        ),
        expected_entities=["verification and validation"],
        category="comparison",
        difficulty="easy",
        must_pass=True,
    ),
    GoldenExample(
        id="cmp-004",
        question="How do derived and allocated requirements differ?",
        expected_answer=(
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
        expected_answer=(
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
    # -------------------------------------------------------------------------
    # TOOLS (Medium)
    # -------------------------------------------------------------------------
    GoldenExample(
        id="tool-001",
        question="Tell me about Jama Connect.",
        expected_answer=(
            "Jama Connect is a requirements management and product development "
            "platform. It provides requirements authoring, live traceability, "
            "review and approval workflows, risk management integration, "
            "test management, and collaboration features. It's used across "
            "industries including automotive, aerospace, medical devices, and "
            "industrial equipment for managing complex product development."
        ),
        expected_entities=["Jama Connect", "requirements management"],
        category="tools",
        difficulty="medium",
        must_pass=True,
    ),
    GoldenExample(
        id="tool-002",
        question=(
            "What is Model-Based Systems Engineering and how does it help with requirements?"
        ),
        expected_answer=(
            "Model-Based Systems Engineering (MBSE) uses models as the primary "
            "artifact for systems engineering instead of documents. For requirements, "
            "MBSE provides: visual requirements representation (SysML diagrams), "
            "formal consistency checking, automated traceability through model "
            "relationships, simulation for early validation, and better stakeholder "
            "communication. Tools like Cameo, Rhapsody, and Capella support MBSE."
        ),
        expected_entities=["MBSE", "SysML"],
        category="tools",
        difficulty="hard",
        must_pass=False,
    ),
    # -------------------------------------------------------------------------
    # STRUCTURED QUERIES (for intent classification testing)
    # -------------------------------------------------------------------------
    GoldenExample(
        id="str-001",
        question="List all webinars in the knowledge base.",
        expected_answer="[List of webinars from database]",
        intent="structured",
        category="data_query",
        difficulty="easy",
        must_pass=False,
    ),
    GoldenExample(
        id="str-002",
        question="How many articles mention requirements traceability?",
        expected_answer="[Count from database]",
        intent="structured",
        category="data_query",
        difficulty="easy",
        must_pass=False,
    ),
    GoldenExample(
        id="str-003",
        question="Which standards are covered in the knowledge base?",
        expected_answer="[List of standards from database]",
        intent="structured",
        category="data_query",
        difficulty="easy",
        must_pass=False,
    ),
    GoldenExample(
        id="str-004",
        question="What are the top 5 most mentioned tools?",
        expected_answer="[Ranked list from database]",
        intent="structured",
        category="data_query",
        difficulty="medium",
        must_pass=False,
    ),
)


# =============================================================================
# LOCAL CONVERSATIONAL EXAMPLES
# =============================================================================

CONVERSATIONAL_EXAMPLES: tuple[ConversationalExample, ...] = (
    ConversationalExample(
        id="conv-001",
        conversation_history=(
            "User: What is requirements traceability?\n"
            "Assistant: Requirements traceability is the ability to describe "
            "and follow the life of a requirement in both a forward and "
            "backward direction through the development lifecycle."
        ),
        question="How does it help with change management?",
        expected_answer=(
            "Traceability supports change management by enabling impact "
            "analysis — when a requirement changes, trace links reveal all "
            "affected design elements, code modules, and test cases."
        ),
        expected_references=["requirements traceability", "impact analysis"],
        category="follow_up",
        difficulty="medium",
        must_pass=True,
    ),
    ConversationalExample(
        id="conv-002",
        conversation_history=(
            "User: Tell me about ISO 26262.\n"
            "Assistant: ISO 26262 is the international standard for "
            "functional safety of electrical and electronic systems in "
            "road vehicles. It defines ASIL levels A through D."
        ),
        question="What ASIL level requires the most rigorous traceability?",
        expected_answer=(
            "ASIL D requires the most rigorous traceability. At this level, "
            "bidirectional traceability between all lifecycle artifacts is "
            "mandatory, including safety goals, functional safety requirements, "
            "technical safety requirements, and verification evidence."
        ),
        expected_references=["ISO 26262", "ASIL"],
        category="follow_up",
        difficulty="hard",
        must_pass=True,
    ),
    ConversationalExample(
        id="conv-003",
        conversation_history=(
            "User: What tools support requirements management?\n"
            "Assistant: Common tools include Jama Connect, IBM DOORS, "
            "and Helix RM. Each provides requirements authoring, "
            "traceability, and collaboration features."
        ),
        question="Which one is best for medical device development?",
        expected_answer=(
            "Jama Connect is widely regarded as strong for medical device "
            "development due to its built-in support for IEC 62304 workflows, "
            "risk management integration (ISO 14971), and FDA audit trail "
            "capabilities. IBM DOORS is also used in regulated environments."
        ),
        expected_references=["Jama Connect", "IEC 62304", "medical device"],
        category="follow_up",
        difficulty="medium",
        must_pass=True,
    ),
    ConversationalExample(
        id="conv-004",
        conversation_history=(
            "User: What is a suspect link?\n"
            "Assistant: A suspect link is a traceability link that may be "
            "invalid or outdated due to changes in one of the linked items.\n"
            "User: How do I identify them?\n"
            "Assistant: Most RM tools automatically flag links as suspect "
            "when a linked item changes. You can also run periodic audits."
        ),
        question="What should I do when I find suspect links?",
        expected_answer=(
            "When you find suspect links: 1) Review the linked items to "
            "determine if the relationship is still valid. 2) Update or "
            "remove invalid links. 3) Re-verify affected test cases. "
            "4) Document the review decision. 5) Clear the suspect flag."
        ),
        expected_references=["suspect links", "trace links"],
        category="multi_turn",
        difficulty="medium",
        must_pass=True,
    ),
    ConversationalExample(
        id="conv-005",
        conversation_history="",
        question="What is requirements traceability?",
        expected_answer=(
            "Requirements traceability is the ability to describe and follow "
            "the life of a requirement in both a forward and backward "
            "direction through the development lifecycle."
        ),
        expected_references=[],
        category="first_turn",
        difficulty="easy",
        must_pass=True,
    ),
    ConversationalExample(
        id="conv-006",
        conversation_history=(
            "User: Compare forward and backward traceability.\n"
            "Assistant: Forward traceability tracks from requirements to "
            "implementation and tests. Backward traceability traces from "
            "deliverables back to requirements."
        ),
        question="Which one helps identify gold plating?",
        expected_answer=(
            "Backward traceability helps identify gold plating — features "
            "or code that don't trace back to any requirement. By tracing "
            "from implementation artifacts back to requirements, you can "
            "find work that was added without a requirement justification."
        ),
        expected_references=[
            "backward traceability",
            "gold plating",
        ],
        category="follow_up",
        difficulty="hard",
        must_pass=False,
    ),
    ConversationalExample(
        id="conv-007",
        conversation_history=(
            "User: What is DO-178C?\n"
            "Assistant: DO-178C is the standard for software considerations "
            "in airborne systems. It defines 5 Design Assurance Levels.\n"
            "User: What are the levels?\n"
            "Assistant: Level A (catastrophic), B (hazardous), C (major), "
            "D (minor), E (no effect). Higher levels require more rigorous "
            "development and verification."
        ),
        question="How does Level A differ from Level C for traceability?",
        expected_answer=(
            "Level A requires complete bidirectional traceability with "
            "independence of verification activities. Level C requires "
            "traceability but with less rigor — independence is not "
            "required and some objectives can be satisfied with less "
            "formal evidence."
        ),
        expected_references=["DO-178C", "design assurance levels"],
        category="multi_turn",
        difficulty="hard",
        must_pass=False,
    ),
    ConversationalExample(
        id="conv-008",
        conversation_history=(
            "User: What is MBSE?\n"
            "Assistant: Model-Based Systems Engineering uses models as "
            "the primary artifact instead of documents. Tools like Cameo, "
            "Rhapsody, and Capella support it."
        ),
        question="Can MBSE replace traditional traceability matrices?",
        expected_answer=(
            "MBSE can complement but not fully replace traceability matrices. "
            "Model relationships provide automatic traceability within the "
            "model, but you may still need matrices for cross-tool "
            "traceability and regulatory compliance documentation."
        ),
        expected_references=["MBSE", "traceability matrix"],
        category="follow_up",
        difficulty="hard",
        must_pass=False,
    ),
    ConversationalExample(
        id="conv-009",
        conversation_history=(
            "User: What is requirements decomposition?\n"
            "Assistant: Requirements decomposition breaks down high-level "
            "requirements into lower-level, more detailed requirements."
        ),
        question=("How do I know when I've decomposed enough?"),
        expected_answer=(
            "You've decomposed enough when each requirement is: "
            "1) Atomic — addresses a single concern. "
            "2) Testable — you can write a verification procedure. "
            "3) Implementable — a developer can build it without "
            "further clarification. "
            "4) Allocatable — it maps to a single component or subsystem."
        ),
        expected_references=[
            "requirements decomposition",
            "atomic requirements",
        ],
        category="follow_up",
        difficulty="medium",
        must_pass=True,
    ),
    ConversationalExample(
        id="conv-010",
        conversation_history=(
            "User: What is the difference between V&V?\n"
            "Assistant: Verification confirms the product is built "
            "correctly per specifications. Validation confirms it meets "
            "stakeholder needs.\n"
            "User: Which uses traceability more?\n"
            "Assistant: Both use traceability, but verification relies on "
            "it more heavily to demonstrate requirement-to-test coverage."
        ),
        question="Give me an example of each in practice.",
        expected_answer=(
            "Verification example: Tracing each software requirement to "
            "its unit test to confirm 100% test coverage. "
            "Validation example: Tracing user needs to acceptance tests "
            "and conducting user acceptance testing to confirm the system "
            "solves the right problem."
        ),
        expected_references=[
            "verification and validation",
            "test coverage",
        ],
        category="multi_turn",
        difficulty="medium",
        must_pass=True,
    ),
    ConversationalExample(
        id="conv-011",
        conversation_history=(
            "User: Tell me about Jama Connect.\n"
            "Assistant: Jama Connect is a requirements management platform "
            "with live traceability, review workflows, and risk management."
        ),
        question="How does its live traceability work?",
        expected_answer=(
            "Jama Connect's live traceability automatically maintains "
            "relationships between items across the lifecycle. When an item "
            "changes, linked items are flagged as suspect. The trace graph "
            "updates in real-time, showing coverage and impact analysis "
            "without manual matrix updates."
        ),
        expected_references=["Jama Connect", "live traceability"],
        category="follow_up",
        difficulty="medium",
        must_pass=False,
    ),
    ConversationalExample(
        id="conv-012",
        conversation_history=(
            "User: What are derived requirements?\n"
            "Assistant: Derived requirements emerge from design decisions "
            "and are not directly traceable to source requirements."
        ),
        question="Are derived requirements a problem?",
        expected_answer=(
            "Derived requirements are not inherently a problem — they're "
            "normal in systems engineering. However, they need special "
            "attention: 1) They must be documented and justified. "
            "2) Safety analyses may be needed (especially in DO-178C). "
            "3) They should be reviewed for unintended scope expansion. "
            "4) Their origin rationale should be captured."
        ),
        expected_references=["derived requirements"],
        category="follow_up",
        difficulty="medium",
        must_pass=False,
    ),
    ConversationalExample(
        id="conv-013",
        conversation_history=(
            "User: How does IEC 62304 handle traceability?\n"
            "Assistant: IEC 62304 requires traceability between system "
            "requirements, software requirements, architecture, units, "
            "and testing throughout the product lifecycle."
        ),
        question="What happens if we skip traceability for Class A software?",
        expected_answer=(
            "IEC 62304 Class A (no injury possible) has reduced "
            "requirements — you still need basic software development "
            "planning but full traceability is not mandatory. However, "
            "if your risk analysis is wrong and the class should be B or C, "
            "you'll need to retroactively establish traceability, which "
            "is far more expensive."
        ),
        expected_references=["IEC 62304", "software safety class"],
        category="follow_up",
        difficulty="hard",
        must_pass=False,
    ),
    ConversationalExample(
        id="conv-014",
        conversation_history=(
            "User: What is baseline management?\n"
            "Assistant: Baseline management captures a snapshot of "
            "requirements at a specific point and controls changes "
            "through formal review.\n"
            "User: When should I baseline?\n"
            "Assistant: Baseline at key milestones: after requirements "
            "review approval, before design starts, at design review, "
            "and before testing."
        ),
        question="What if stakeholders want changes after a baseline?",
        expected_answer=(
            "Changes after a baseline go through the change control "
            "process: 1) Submit a change request. 2) Perform impact "
            "analysis using traceability. 3) Review and approve/reject. "
            "4) If approved, update requirements, trace links, and "
            "downstream artifacts. 5) Create a new baseline version."
        ),
        expected_references=["baseline management", "change management"],
        category="multi_turn",
        difficulty="medium",
        must_pass=True,
    ),
    ConversationalExample(
        id="conv-015",
        conversation_history=(
            "User: What is ASPICE?\n"
            "Assistant: Automotive SPICE defines process assessment levels "
            "for automotive software development, covering engineering "
            "and management processes."
        ),
        question="How does ASPICE level 2 differ from level 3?",
        expected_answer=(
            "At ASPICE Level 2 (Managed), processes are planned, monitored, "
            "and adjusted at the project level. Work products are "
            "appropriately established and controlled. At Level 3 "
            "(Established), standard processes are defined at the "
            "organization level, and projects tailor them. The key "
            "difference is organizational standardization vs. "
            "project-level management."
        ),
        expected_references=["ASPICE", "capability levels"],
        category="follow_up",
        difficulty="hard",
        must_pass=False,
    ),
)


# =============================================================================
# QUERY HELPERS
# =============================================================================


def get_must_pass_examples() -> list[GoldenExample]:
    """Get examples that must score above threshold."""
    return [ex for ex in GOLDEN_EXAMPLES if ex.must_pass]


def get_must_pass_conversational() -> list[ConversationalExample]:
    """Get conversational examples that must score above threshold."""
    return [ex for ex in CONVERSATIONAL_EXAMPLES if ex.must_pass]


def get_examples_by_category(category: str) -> list[GoldenExample]:
    """Get examples filtered by category."""
    return [ex for ex in GOLDEN_EXAMPLES if ex.category == category]


def get_examples_by_intent(intent: str) -> list[GoldenExample]:
    """Get examples filtered by intent."""
    return [ex for ex in GOLDEN_EXAMPLES if ex.intent == intent]


def get_examples_by_vector(
    vector: str,
) -> list[GoldenExample] | list[ConversationalExample]:
    """Get examples for a specific evaluation vector.

    Args:
        vector: One of "explanatory", "structured", "conversational".

    Returns:
        List of GoldenExample or ConversationalExample instances.
    """
    if vector == "conversational":
        return list(CONVERSATIONAL_EXAMPLES)
    return [ex for ex in GOLDEN_EXAMPLES if ex.vector == vector]


# =============================================================================
# HUB-FIRST DATASET ACCESS
# =============================================================================


async def get_golden_examples(
    *,
    dataset_name: str = DATASET_NAME,
    use_langsmith: bool | None = None,
) -> list[GoldenExample]:
    """Get golden examples, trying LangSmith first then local fallback.

    Follows the same Hub-First pattern as prompts/catalog.py.

    Args:
        dataset_name: LangSmith dataset name.
        use_langsmith: Override Hub lookup. None = auto-detect from
            LANGSMITH_API_KEY. False = local only (for CI Tier 1).

    Returns:
        List of GoldenExample instances.
    """
    if use_langsmith is None:
        use_langsmith = bool(os.getenv("LANGSMITH_API_KEY"))

    if use_langsmith:
        examples = await _pull_from_langsmith(dataset_name)
        if examples is not None:
            return examples

    logger.debug("Using local fallback dataset (%d examples)", len(GOLDEN_EXAMPLES))
    return list(GOLDEN_EXAMPLES)


async def _pull_from_langsmith(dataset_name: str) -> list[GoldenExample] | None:
    """Pull examples from LangSmith dataset.

    Returns:
        List of GoldenExample if successful, None on failure.
    """
    try:
        from langsmith import Client

        client = Client()
        dataset = await asyncio.to_thread(client.read_dataset, dataset_name=dataset_name)
        raw_examples = await asyncio.to_thread(
            lambda: list(client.list_examples(dataset_id=dataset.id))
        )

        examples = [
            GoldenExample.from_langsmith(
                inputs=ex.inputs or {},
                outputs=ex.outputs or {},
                metadata=ex.metadata or {},
            )
            for ex in raw_examples
        ]

        logger.info(
            "Pulled %d examples from LangSmith dataset '%s'",
            len(examples),
            dataset_name,
        )
        return examples

    except Exception as e:
        logger.debug("Failed to pull from LangSmith (falling back to local): %s", e)
        return None


__all__ = [
    "CONVERSATIONAL_EXAMPLES",
    "DATASET_NAME",
    "GOLDEN_EXAMPLES",
    "ConversationalExample",
    "GoldenExample",
    "get_examples_by_category",
    "get_examples_by_intent",
    "get_examples_by_vector",
    "get_golden_examples",
    "get_must_pass_conversational",
    "get_must_pass_examples",
]
