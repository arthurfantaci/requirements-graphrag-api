"""Golden dataset of hand-curated benchmark examples.

These are critical examples that must always be evaluated:
- Cover key functionality
- Include known edge cases
- Have verified ground truth
- Represent real user queries

This dataset runs on every CI evaluation (Tier 2+).
"""

from __future__ import annotations

from typing import Final

from tests.benchmark.schemas import (
    BenchmarkExample,
    DifficultyLevel,
    ExpectedMetrics,
    ExpectedRouting,
    QueryCategory,
)

# =============================================================================
# GOLDEN DATASET - 30 HAND-CURATED EXAMPLES
# =============================================================================

GOLDEN_DATASET: Final[list[BenchmarkExample]] = [
    # =========================================================================
    # DEFINITIONAL - Core Concepts
    # =========================================================================
    BenchmarkExample(
        id="gold_001",
        question="What is requirements traceability?",
        category=QueryCategory.DEFINITIONAL,
        difficulty=DifficultyLevel.EASY,
        expected_tools=[ExpectedRouting.VECTOR_SEARCH, ExpectedRouting.LOOKUP_TERM],
        ground_truth=(
            "Requirements traceability is the ability to trace requirements through "
            "their entire lifecycle, linking them to their sources (e.g., stakeholder "
            "needs), related requirements, design elements, implementation, and test "
            "cases. It enables impact analysis and ensures complete coverage."
        ),
        expected_entities=["traceability", "requirements"],
        tags=["golden", "core-concept", "must-pass"],
    ),
    BenchmarkExample(
        id="gold_002",
        question="What is the difference between verification and validation?",
        category=QueryCategory.DEFINITIONAL,
        difficulty=DifficultyLevel.EASY,
        expected_tools=[ExpectedRouting.VECTOR_SEARCH],
        ground_truth=(
            "Verification confirms that the product is built correctly according to "
            "specifications ('Are we building the product right?'). Validation confirms "
            "that the product meets user needs and intended use ('Are we building the "
            "right product?'). Both are essential for quality assurance."
        ),
        expected_entities=["verification", "validation"],
        tags=["golden", "core-concept", "must-pass"],
    ),
    BenchmarkExample(
        id="gold_003",
        question="Define a traceability matrix.",
        category=QueryCategory.DEFINITIONAL,
        difficulty=DifficultyLevel.EASY,
        expected_tools=[ExpectedRouting.VECTOR_SEARCH, ExpectedRouting.LOOKUP_TERM],
        ground_truth=(
            "A traceability matrix is a document that maps requirements to other "
            "artifacts (test cases, design elements, code) to ensure complete coverage "
            "and enable impact analysis. It shows relationships between requirements "
            "and downstream/upstream artifacts."
        ),
        expected_entities=["traceability matrix"],
        tags=["golden", "core-concept", "must-pass"],
    ),
    # =========================================================================
    # STANDARDS - ISO 26262 (Automotive)
    # =========================================================================
    BenchmarkExample(
        id="gold_004",
        question="What is ISO 26262?",
        category=QueryCategory.FACTUAL,
        difficulty=DifficultyLevel.MEDIUM,
        expected_tools=[ExpectedRouting.LOOKUP_STANDARD, ExpectedRouting.VECTOR_SEARCH],
        ground_truth=(
            "ISO 26262 is the international standard for functional safety of "
            "electrical and electronic systems in road vehicles. It defines ASIL "
            "(Automotive Safety Integrity Level) classifications from A to D, with "
            "D being the most stringent. The standard covers the entire product "
            "development lifecycle."
        ),
        expected_standards=["ISO 26262"],
        tags=["golden", "standards", "automotive", "must-pass"],
    ),
    BenchmarkExample(
        id="gold_005",
        question="What are the ASIL levels in ISO 26262?",
        category=QueryCategory.FACTUAL,
        difficulty=DifficultyLevel.MEDIUM,
        expected_tools=[ExpectedRouting.LOOKUP_STANDARD, ExpectedRouting.VECTOR_SEARCH],
        ground_truth=(
            "ASIL (Automotive Safety Integrity Level) has four levels: ASIL A "
            "(lowest), ASIL B, ASIL C, and ASIL D (highest/most stringent). There's "
            "also QM (Quality Management) for non-safety-relevant functions. ASIL is "
            "determined by severity, exposure probability, and controllability."
        ),
        expected_standards=["ISO 26262"],
        expected_entities=["ASIL"],
        tags=["golden", "standards", "automotive", "must-pass"],
    ),
    # =========================================================================
    # STANDARDS - IEC 62304 (Medical)
    # =========================================================================
    BenchmarkExample(
        id="gold_006",
        question="What is IEC 62304?",
        category=QueryCategory.FACTUAL,
        difficulty=DifficultyLevel.MEDIUM,
        expected_tools=[ExpectedRouting.LOOKUP_STANDARD, ExpectedRouting.VECTOR_SEARCH],
        ground_truth=(
            "IEC 62304 is the international standard for medical device software "
            "lifecycle processes. It defines software safety classes (A, B, C) based "
            "on potential harm to patients. Class C requires the most rigorous "
            "development and documentation practices."
        ),
        expected_standards=["IEC 62304"],
        tags=["golden", "standards", "medical", "must-pass"],
    ),
    # =========================================================================
    # STANDARDS - ASPICE (Automotive Process)
    # =========================================================================
    BenchmarkExample(
        id="gold_007",
        question="What is ASPICE?",
        category=QueryCategory.FACTUAL,
        difficulty=DifficultyLevel.MEDIUM,
        expected_tools=[ExpectedRouting.LOOKUP_STANDARD, ExpectedRouting.VECTOR_SEARCH],
        ground_truth=(
            "ASPICE (Automotive SPICE) is a process assessment model for automotive "
            "software development based on ISO/IEC 15504. It defines process areas "
            "(e.g., SWE.1-6 for software engineering) and capability levels from 0 "
            "(Incomplete) to 5 (Optimizing). It's widely used for supplier assessment."
        ),
        expected_standards=["ASPICE"],
        tags=["golden", "standards", "automotive", "must-pass"],
    ),
    # =========================================================================
    # PROCEDURAL - How-To Questions
    # =========================================================================
    BenchmarkExample(
        id="gold_008",
        question="How do I implement bidirectional traceability?",
        category=QueryCategory.PROCEDURAL,
        difficulty=DifficultyLevel.MEDIUM,
        expected_tools=[ExpectedRouting.VECTOR_SEARCH, ExpectedRouting.HYBRID_SEARCH],
        ground_truth=(
            "Bidirectional traceability requires: 1) Establishing trace links from "
            "requirements to downstream artifacts (forward) and from artifacts back "
            "to requirements (backward). 2) Using a requirements management tool like "
            "Jama Connect. 3) Defining link types and maintaining them throughout "
            "the lifecycle. 4) Validating coverage through traceability reports."
        ),
        expected_entities=["bidirectional traceability", "traceability"],
        tags=["golden", "procedural", "must-pass"],
    ),
    BenchmarkExample(
        id="gold_009",
        question="What are best practices for requirements elicitation?",
        category=QueryCategory.PROCEDURAL,
        difficulty=DifficultyLevel.MEDIUM,
        expected_tools=[ExpectedRouting.VECTOR_SEARCH, ExpectedRouting.HYBRID_SEARCH],
        ground_truth=(
            "Best practices include: 1) Identify and involve all stakeholders. "
            "2) Use multiple techniques (interviews, workshops, prototypes). "
            "3) Document requirements in a clear, testable format. "
            "4) Validate requirements with stakeholders. "
            "5) Prioritize requirements based on value and risk."
        ),
        expected_entities=["requirements elicitation", "stakeholder"],
        tags=["golden", "procedural", "must-pass"],
    ),
    # =========================================================================
    # RELATIONAL - Concept Relationships
    # =========================================================================
    BenchmarkExample(
        id="gold_010",
        question="How does change management relate to impact analysis?",
        category=QueryCategory.RELATIONAL,
        difficulty=DifficultyLevel.MEDIUM,
        expected_tools=[ExpectedRouting.GRAPH_ENRICHED, ExpectedRouting.EXPLORE_ENTITY],
        ground_truth=(
            "Change management and impact analysis are closely related: impact analysis "
            "is a key activity within change management. When a change is proposed, "
            "impact analysis identifies all affected requirements, designs, tests, and "
            "other artifacts through traceability links. This informs the change "
            "decision and implementation scope."
        ),
        expected_entities=["change management", "impact analysis"],
        tags=["golden", "relational", "must-pass"],
    ),
    BenchmarkExample(
        id="gold_011",
        question="What is the relationship between requirements and test cases?",
        category=QueryCategory.RELATIONAL,
        difficulty=DifficultyLevel.MEDIUM,
        expected_tools=[ExpectedRouting.GRAPH_ENRICHED, ExpectedRouting.VECTOR_SEARCH],
        ground_truth=(
            "Requirements and test cases have a verification relationship: test cases "
            "are derived from requirements to verify that the implementation meets "
            "the specified requirements. Traceability between them ensures complete "
            "test coverage and supports impact analysis when requirements change."
        ),
        expected_entities=["requirements", "test cases", "verification"],
        tags=["golden", "relational", "must-pass"],
    ),
    # =========================================================================
    # ANALYTICAL - Why/Analysis Questions
    # =========================================================================
    BenchmarkExample(
        id="gold_012",
        question="Why is requirements traceability important in regulated industries?",
        category=QueryCategory.ANALYTICAL,
        difficulty=DifficultyLevel.MEDIUM,
        expected_tools=[ExpectedRouting.GRAPH_ENRICHED, ExpectedRouting.CHAT],
        ground_truth=(
            "In regulated industries, traceability is critical because: "
            "1) Standards (ISO 26262, IEC 62304, DO-178C) mandate it. "
            "2) It demonstrates compliance during audits. "
            "3) It enables impact analysis for safe change management. "
            "4) It ensures all safety requirements are verified and validated. "
            "5) It provides evidence for product liability defense."
        ),
        expected_standards=["ISO 26262", "IEC 62304"],
        tags=["golden", "analytical", "must-pass"],
    ),
    BenchmarkExample(
        id="gold_013",
        question="What are the challenges with requirements management?",
        category=QueryCategory.ANALYTICAL,
        difficulty=DifficultyLevel.MEDIUM,
        expected_tools=[ExpectedRouting.VECTOR_SEARCH, ExpectedRouting.CHAT],
        ground_truth=(
            "Common challenges include: 1) Requirements volatility and scope creep. "
            "2) Ambiguous or incomplete requirements. 3) Lack of stakeholder "
            "involvement. 4) Poor traceability maintenance. 5) Tool limitations. "
            "6) Communication gaps between teams. 7) Balancing detail with agility."
        ),
        expected_entities=["requirements management"],
        tags=["golden", "analytical", "must-pass"],
    ),
    # =========================================================================
    # COMPARISON - Direct Comparisons
    # =========================================================================
    BenchmarkExample(
        id="gold_014",
        question="Compare functional and non-functional requirements.",
        category=QueryCategory.COMPARISON,
        difficulty=DifficultyLevel.MEDIUM,
        expected_tools=[ExpectedRouting.HYBRID_SEARCH, ExpectedRouting.VECTOR_SEARCH],
        ground_truth=(
            "Functional requirements describe what the system should do (features, "
            "behaviors, functions). Non-functional requirements describe how the "
            "system should perform (quality attributes like performance, security, "
            "usability, reliability). Both are essential: functional requirements "
            "define capabilities, non-functional requirements define constraints."
        ),
        expected_entities=["functional requirements", "non-functional requirements"],
        tags=["golden", "comparison", "must-pass"],
    ),
    BenchmarkExample(
        id="gold_015",
        question="What's the difference between forward and backward traceability?",
        category=QueryCategory.COMPARISON,
        difficulty=DifficultyLevel.EASY,
        expected_tools=[ExpectedRouting.VECTOR_SEARCH],
        ground_truth=(
            "Forward traceability traces from requirements to downstream artifacts "
            "(design, code, tests) - ensuring requirements are implemented. "
            "Backward traceability traces from artifacts back to requirements - "
            "ensuring all artifacts have a justified origin. Both together provide "
            "bidirectional traceability."
        ),
        expected_entities=["forward traceability", "backward traceability"],
        tags=["golden", "comparison", "must-pass"],
    ),
    # =========================================================================
    # V-MODEL and SDLC
    # =========================================================================
    BenchmarkExample(
        id="gold_016",
        question="What is the V-model?",
        category=QueryCategory.DEFINITIONAL,
        difficulty=DifficultyLevel.MEDIUM,
        expected_tools=[ExpectedRouting.VECTOR_SEARCH, ExpectedRouting.LOOKUP_TERM],
        ground_truth=(
            "The V-model is a software development methodology that emphasizes "
            "verification and validation at each development stage. The left side "
            "of the V shows decomposition (requirements → design → implementation), "
            "while the right side shows integration and testing. Each development "
            "phase has a corresponding test phase."
        ),
        expected_entities=["V-model"],
        tags=["golden", "methodology", "must-pass"],
    ),
    # =========================================================================
    # INDUSTRY-SPECIFIC
    # =========================================================================
    BenchmarkExample(
        id="gold_017",
        question="What standards apply to automotive software development?",
        category=QueryCategory.FACTUAL,
        difficulty=DifficultyLevel.EASY,
        expected_tools=[ExpectedRouting.STANDARDS_BY_INDUSTRY],
        ground_truth=(
            "Key automotive standards include: ISO 26262 (functional safety), "
            "ASPICE (process assessment), ISO 21434 (cybersecurity), "
            "AUTOSAR (software architecture), and ISO 9001 (quality management). "
            "These ensure safety, quality, and compliance in vehicle software."
        ),
        expected_standards=["ISO 26262", "ASPICE"],
        tags=["golden", "standards", "automotive", "must-pass"],
        metadata={"industry": "automotive"},
    ),
    BenchmarkExample(
        id="gold_018",
        question="What standards apply to medical device software?",
        category=QueryCategory.FACTUAL,
        difficulty=DifficultyLevel.EASY,
        expected_tools=[ExpectedRouting.STANDARDS_BY_INDUSTRY],
        ground_truth=(
            "Key medical device standards include: IEC 62304 (software lifecycle), "
            "ISO 14971 (risk management), FDA 21 CFR Part 11 (electronic records), "
            "ISO 13485 (quality management), and IEC 60601 (safety). "
            "These ensure patient safety and regulatory compliance."
        ),
        expected_standards=["IEC 62304", "FDA 21 CFR Part 11", "ISO 13485"],
        tags=["golden", "standards", "medical", "must-pass"],
        metadata={"industry": "medical device"},
    ),
    # =========================================================================
    # TEXT2CYPHER - Graph Queries
    # =========================================================================
    BenchmarkExample(
        id="gold_019",
        question="How many articles are in the knowledge base?",
        category=QueryCategory.FACTUAL,
        difficulty=DifficultyLevel.EASY,
        expected_tools=[ExpectedRouting.TEXT2CYPHER],
        ground_truth=(
            "The answer should return the count of Article nodes in the knowledge "
            "graph. The expected Cypher: MATCH (a:Article) RETURN count(a)"
        ),
        tags=["golden", "text2cypher", "must-pass"],
    ),
    BenchmarkExample(
        id="gold_020",
        question="Which articles mention ISO 26262?",
        category=QueryCategory.FACTUAL,
        difficulty=DifficultyLevel.MEDIUM,
        expected_tools=[ExpectedRouting.TEXT2CYPHER, ExpectedRouting.LOOKUP_STANDARD],
        ground_truth=(
            "The answer should list articles that reference ISO 26262, found via "
            "the MENTIONED_IN relationship from the Standard node to Chunks to Articles."
        ),
        expected_standards=["ISO 26262"],
        tags=["golden", "text2cypher", "must-pass"],
    ),
    # =========================================================================
    # EDGE CASES - Must Handle Gracefully
    # =========================================================================
    BenchmarkExample(
        id="gold_021",
        question="What is the weather like today?",
        category=QueryCategory.EDGE_CASE,
        difficulty=DifficultyLevel.HARD,
        expected_tools=[ExpectedRouting.CHAT],
        ground_truth=(
            "This is an out-of-domain question. The system should recognize it's "
            "outside the requirements management domain and politely explain that "
            "it can only answer questions about requirements management and related topics."
        ),
        expected_metrics=ExpectedMetrics(
            min_faithfulness=0.3,
            min_relevancy=0.3,
            min_precision=0.2,
            min_recall=0.2,
        ),
        tags=["golden", "edge-case", "out-of-domain", "must-pass"],
        metadata={"out_of_domain": True},
    ),
    BenchmarkExample(
        id="gold_022",
        question="Tell me about requirements.",
        category=QueryCategory.EDGE_CASE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_tools=[ExpectedRouting.VECTOR_SEARCH, ExpectedRouting.CHAT],
        ground_truth=(
            "This is an ambiguous query. The system should either ask for "
            "clarification or provide a general overview of requirements in the "
            "context of requirements management."
        ),
        expected_metrics=ExpectedMetrics(
            min_faithfulness=0.5,
            min_relevancy=0.5,
            min_precision=0.4,
            min_recall=0.4,
        ),
        tags=["golden", "edge-case", "ambiguous", "must-pass"],
    ),
    BenchmarkExample(
        id="gold_023",
        question="What is requirments traceability?",  # Intentional typo
        category=QueryCategory.EDGE_CASE,
        difficulty=DifficultyLevel.EASY,
        expected_tools=[ExpectedRouting.VECTOR_SEARCH],
        ground_truth=(
            "The system should handle the misspelling gracefully and answer about "
            "requirements traceability. Semantic search should find relevant content "
            "despite the typo."
        ),
        expected_entities=["traceability"],
        tags=["golden", "edge-case", "typo", "must-pass"],
    ),
    # =========================================================================
    # JAMA CONNECT SPECIFIC
    # =========================================================================
    BenchmarkExample(
        id="gold_024",
        question="What is Jama Connect?",
        category=QueryCategory.DEFINITIONAL,
        difficulty=DifficultyLevel.EASY,
        expected_tools=[ExpectedRouting.VECTOR_SEARCH],
        ground_truth=(
            "Jama Connect is a requirements management and traceability platform "
            "that helps teams capture, manage, and trace requirements throughout "
            "the product development lifecycle. It supports collaboration, "
            "workflow automation, and regulatory compliance."
        ),
        expected_entities=["Jama Connect"],
        tags=["golden", "tool", "must-pass"],
    ),
    # =========================================================================
    # MULTI-PART QUESTIONS
    # =========================================================================
    BenchmarkExample(
        id="gold_025",
        question=(
            "What is requirements traceability and why is it important for ISO 26262 compliance?"
        ),
        category=QueryCategory.EDGE_CASE,
        difficulty=DifficultyLevel.HARD,
        expected_tools=[
            ExpectedRouting.VECTOR_SEARCH,
            ExpectedRouting.LOOKUP_STANDARD,
        ],
        ground_truth=(
            "The answer should cover: 1) Definition of requirements traceability. "
            "2) ISO 26262's specific requirements for traceability. "
            "3) How traceability supports safety case documentation and audits."
        ),
        expected_standards=["ISO 26262"],
        expected_entities=["traceability"],
        tags=["golden", "multi-part", "must-pass"],
    ),
    # =========================================================================
    # RISK AND HAZARD ANALYSIS
    # =========================================================================
    BenchmarkExample(
        id="gold_026",
        question="What is hazard analysis in safety-critical development?",
        category=QueryCategory.DEFINITIONAL,
        difficulty=DifficultyLevel.MEDIUM,
        expected_tools=[ExpectedRouting.VECTOR_SEARCH, ExpectedRouting.HYBRID_SEARCH],
        ground_truth=(
            "Hazard analysis is the systematic identification and assessment of "
            "potential hazards in a system. Common techniques include FMEA (Failure "
            "Mode and Effects Analysis), FTA (Fault Tree Analysis), and HAZOP. "
            "Results drive safety requirements and mitigation measures."
        ),
        expected_entities=["hazard analysis", "FMEA", "FTA"],
        tags=["golden", "safety", "must-pass"],
    ),
    # =========================================================================
    # COMPLIANCE AND AUDITS
    # =========================================================================
    BenchmarkExample(
        id="gold_027",
        question="How do I prepare for a regulatory audit?",
        category=QueryCategory.PROCEDURAL,
        difficulty=DifficultyLevel.HARD,
        expected_tools=[ExpectedRouting.VECTOR_SEARCH, ExpectedRouting.CHAT],
        ground_truth=(
            "Audit preparation includes: 1) Ensure complete traceability from "
            "requirements to verification evidence. 2) Maintain up-to-date "
            "documentation. 3) Conduct internal audits first. 4) Review applicable "
            "standards and checklists. 5) Prepare traceability reports and coverage "
            "matrices. 6) Brief team members on audit process."
        ),
        tags=["golden", "compliance", "must-pass"],
    ),
    # =========================================================================
    # AEROSPACE
    # =========================================================================
    BenchmarkExample(
        id="gold_028",
        question="What is DO-178C?",
        category=QueryCategory.FACTUAL,
        difficulty=DifficultyLevel.MEDIUM,
        expected_tools=[ExpectedRouting.LOOKUP_STANDARD, ExpectedRouting.VECTOR_SEARCH],
        ground_truth=(
            "DO-178C is the standard for software considerations in airborne "
            "systems and equipment certification. It defines five Design Assurance "
            "Levels (DAL A-E) based on failure severity. It's recognized by FAA, "
            "EASA, and other aviation authorities."
        ),
        expected_standards=["DO-178C"],
        tags=["golden", "standards", "aerospace", "must-pass"],
    ),
    # =========================================================================
    # PROCESS IMPROVEMENT
    # =========================================================================
    BenchmarkExample(
        id="gold_029",
        question="What is CMMI?",
        category=QueryCategory.FACTUAL,
        difficulty=DifficultyLevel.MEDIUM,
        expected_tools=[ExpectedRouting.LOOKUP_STANDARD, ExpectedRouting.VECTOR_SEARCH],
        ground_truth=(
            "CMMI (Capability Maturity Model Integration) is a process improvement "
            "framework that defines maturity levels from 1 (Initial) to 5 "
            "(Optimizing). It covers development, services, and acquisition. "
            "Organizations use it to improve process capability and performance."
        ),
        expected_standards=["CMMI"],
        tags=["golden", "standards", "process", "must-pass"],
    ),
    # =========================================================================
    # NEGATIVE TEST - Should NOT Hallucinate
    # =========================================================================
    BenchmarkExample(
        id="gold_030",
        question="What is ISO 99999?",  # Non-existent standard
        category=QueryCategory.EDGE_CASE,
        difficulty=DifficultyLevel.HARD,
        expected_tools=[ExpectedRouting.LOOKUP_STANDARD, ExpectedRouting.VECTOR_SEARCH],
        ground_truth=(
            "The system should indicate that ISO 99999 is not found in the knowledge "
            "base and should NOT hallucinate a fake standard. It may offer to search "
            "for related standards."
        ),
        expected_metrics=ExpectedMetrics(
            min_faithfulness=0.8,  # High because it should not make things up
            min_relevancy=0.3,
            min_precision=0.3,
            min_recall=0.3,
        ),
        tags=["golden", "edge-case", "no-hallucination", "must-pass"],
        metadata={"test_type": "negative"},
    ),
]


def get_golden_dataset() -> list[BenchmarkExample]:
    """Return the golden dataset."""
    return GOLDEN_DATASET.copy()


def get_must_pass_examples() -> list[BenchmarkExample]:
    """Return only examples tagged as must-pass."""
    return [ex for ex in GOLDEN_DATASET if "must-pass" in ex.tags]


def get_golden_by_category(category: QueryCategory) -> list[BenchmarkExample]:
    """Filter golden dataset by category."""
    return [ex for ex in GOLDEN_DATASET if ex.category == category]


__all__ = [
    "GOLDEN_DATASET",
    "get_golden_by_category",
    "get_golden_dataset",
    "get_must_pass_examples",
]
