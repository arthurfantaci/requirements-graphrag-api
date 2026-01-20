"""Query templates for programmatic benchmark generation.

Templates are organized by query category and combined with
domain concepts from the requirements management knowledge base.
"""

from __future__ import annotations

from typing import Final

# =============================================================================
# DOMAIN CONCEPTS
# =============================================================================

# Core concepts from the requirements management knowledge base
CORE_CONCEPTS: Final[list[str]] = [
    "requirements traceability",
    "requirements management",
    "verification",
    "validation",
    "change management",
    "impact analysis",
    "baseline",
    "requirements elicitation",
    "stakeholder analysis",
    "requirements specification",
    "functional requirements",
    "non-functional requirements",
    "system requirements",
    "software requirements",
    "user requirements",
    "business requirements",
    "traceability matrix",
    "bidirectional traceability",
    "forward traceability",
    "backward traceability",
    "requirements decomposition",
    "requirements allocation",
    "requirements prioritization",
    "requirements review",
    "requirements sign-off",
]

# Industry standards referenced in the knowledge base
INDUSTRY_STANDARDS: Final[list[str]] = [
    "ISO 26262",
    "IEC 62304",
    "DO-178C",
    "DO-254",
    "ISO 13485",
    "FDA 21 CFR Part 11",
    "ASPICE",
    "CMMI",
    "ISO 9001",
    "MIL-STD-498",
    "IEEE 830",
    "IEEE 29148",
]

# Industries covered
INDUSTRIES: Final[list[str]] = [
    "automotive",
    "medical device",
    "aerospace",
    "defense",
    "rail",
    "pharmaceutical",
    "software",
]

# Tools and methodologies
TOOLS_AND_METHODS: Final[list[str]] = [
    "Jama Connect",
    "V-model",
    "agile",
    "waterfall",
    "SDLC",
    "ALM",
    "risk management",
    "hazard analysis",
    "FMEA",
    "FTA",
]

# ASIL levels for automotive
ASIL_LEVELS: Final[list[str]] = ["ASIL A", "ASIL B", "ASIL C", "ASIL D", "QM"]

# =============================================================================
# QUERY TEMPLATES BY CATEGORY
# =============================================================================

DEFINITIONAL_TEMPLATES: Final[list[str]] = [
    "What is {concept}?",
    "Define {concept}.",
    "Explain {concept} in requirements management.",
    "What does {concept} mean?",
    "Can you explain what {concept} is?",
    "What is the definition of {concept}?",
    "Describe {concept}.",
    "What is meant by {concept}?",
]

RELATIONAL_TEMPLATES: Final[list[str]] = [
    "How does {concept1} relate to {concept2}?",
    "What is the relationship between {concept1} and {concept2}?",
    "How are {concept1} and {concept2} connected?",
    "What's the difference between {concept1} and {concept2}?",
    "How does {concept1} support {concept2}?",
    "Why is {concept1} important for {concept2}?",
]

PROCEDURAL_TEMPLATES: Final[list[str]] = [
    "How do I implement {concept}?",
    "What are the steps to perform {concept}?",
    "How do you set up {concept}?",
    "What's the process for {concept}?",
    "How should I approach {concept}?",
    "What are best practices for {concept}?",
    "How do I create a {concept}?",
    "What's the workflow for {concept}?",
]

COMPARISON_TEMPLATES: Final[list[str]] = [
    "Compare {concept1} and {concept2}.",
    "{concept1} vs {concept2}: what are the differences?",
    "What are the pros and cons of {concept1} versus {concept2}?",
    "When should I use {concept1} instead of {concept2}?",
    "How do {concept1} and {concept2} differ?",
]

FACTUAL_TEMPLATES: Final[list[str]] = [
    "What standards apply to {industry}?",
    "What are the ASIL levels in {standard}?",
    "Which articles discuss {concept}?",
    "How many {entity_type} are in the knowledge base?",
    "List the main topics covered in {concept}.",
    "What industries use {standard}?",
]

ANALYTICAL_TEMPLATES: Final[list[str]] = [
    "Why is {concept} important in regulated industries?",
    "What are the challenges with {concept}?",
    "What happens if {concept} is not implemented correctly?",
    "How does {concept} improve product quality?",
    "What are the benefits of {concept}?",
    "Why do organizations struggle with {concept}?",
]

STANDARD_SPECIFIC_TEMPLATES: Final[list[str]] = [
    "What is {standard} and what does it cover?",
    "What are the key requirements of {standard}?",
    "How does {standard} apply to {industry}?",
    "What is the scope of {standard}?",
    "What are the compliance requirements for {standard}?",
    "How do I achieve compliance with {standard}?",
]

# =============================================================================
# EDGE CASE TEMPLATES
# =============================================================================

EDGE_CASE_TEMPLATES: Final[list[str]] = [
    # Ambiguous queries
    "Tell me about requirements.",
    "What should I know?",
    "Help me with traceability.",
    # Multi-part questions
    "What is {concept1} and how does it relate to {concept2}, and what standards cover it?",
    "Explain {concept}, list its benefits, and describe how to implement it.",
    # Out-of-domain (should gracefully handle)
    "What is the weather like today?",
    "How do I cook pasta?",
    "What is machine learning?",
    # Adversarial/tricky
    "Is {standard} better than all other standards?",
    "Why is {concept} useless?",
    "Prove that {concept} doesn't work.",
    # Very specific (may not have exact answer)
    "What is the exact page number where {concept} is defined in {standard}?",
    "Who invented {concept}?",
    # Misspellings and variations
    "What is requirments traceability?",
    "Explain tracability matrix.",
    "What is ISO26262?",  # No space
]

# =============================================================================
# GROUND TRUTH TEMPLATES
# =============================================================================

# Partial ground truth templates - generator fills in specifics
GROUND_TRUTH_HINTS: Final[dict[str, str]] = {
    "requirements traceability": (
        "Requirements traceability is the ability to trace requirements through "
        "their lifecycle, linking them to their sources, related requirements, "
        "design elements, test cases, and other artifacts."
    ),
    "verification": (
        "Verification confirms that the product is built correctly according to "
        "specifications. It answers 'Are we building the product right?'"
    ),
    "validation": (
        "Validation confirms that the product meets user needs and intended use. "
        "It answers 'Are we building the right product?'"
    ),
    "ISO 26262": (
        "ISO 26262 is the international standard for functional safety of "
        "electrical and electronic systems in road vehicles. It defines ASIL "
        "levels (A through D) based on severity, exposure, and controllability."
    ),
    "IEC 62304": (
        "IEC 62304 is the international standard for medical device software "
        "lifecycle processes. It defines software safety classes (A, B, C) "
        "and required development activities."
    ),
    "ASPICE": (
        "ASPICE (Automotive SPICE) is a process assessment model for automotive "
        "software development. It defines process areas and capability levels "
        "from 0 to 5."
    ),
    "traceability matrix": (
        "A traceability matrix is a document that maps and traces requirements "
        "to other artifacts such as test cases, design documents, and code. "
        "It helps ensure coverage and supports impact analysis."
    ),
    "V-model": (
        "The V-model is a software development methodology that emphasizes "
        "verification and validation at each stage. The left side shows "
        "decomposition/definition, the right side shows integration/testing."
    ),
}

__all__ = [
    "ANALYTICAL_TEMPLATES",
    "ASIL_LEVELS",
    "COMPARISON_TEMPLATES",
    "CORE_CONCEPTS",
    "DEFINITIONAL_TEMPLATES",
    "EDGE_CASE_TEMPLATES",
    "FACTUAL_TEMPLATES",
    "GROUND_TRUTH_HINTS",
    "INDUSTRIES",
    "INDUSTRY_STANDARDS",
    "PROCEDURAL_TEMPLATES",
    "RELATIONAL_TEMPLATES",
    "STANDARD_SPECIFIC_TEMPLATES",
    "TOOLS_AND_METHODS",
]
