"""Domain-specific evaluation metrics for requirements management.

Provides specialized evaluation prompts and constants for the requirements
management and traceability domain, including:
- Citation accuracy for industry standards
- Traceability coverage evaluation
- Technical precision metrics
- Regulatory alignment assessment

These metrics complement standard RAG metrics with domain expertise.
"""

from __future__ import annotations

from typing import Final

# =============================================================================
# DOMAIN CONSTANTS
# =============================================================================

# Known industry standards in requirements management
KNOWN_STANDARDS: Final[tuple[str, ...]] = (
    "ISO 26262",  # Automotive functional safety
    "IEC 62304",  # Medical device software
    "DO-178C",  # Avionics software
    "ISO 29148",  # Requirements engineering
    "IEEE 830",  # Software requirements specifications
    "CMMI",  # Capability Maturity Model Integration
    "ASPICE",  # Automotive SPICE
    "FDA 21 CFR Part 11",  # FDA electronic records
    "IEC 61508",  # Functional safety
    "MIL-STD-498",  # Military software development
    "ISO 13485",  # Medical devices QMS
    "AS9100",  # Aerospace quality
    "IATF 16949",  # Automotive quality
    "ISO 9001",  # Quality management
    "INCOSE",  # Systems engineering
)

# Domain-specific terminology for requirements management
DOMAIN_TERMS: Final[tuple[str, ...]] = (
    "requirements traceability",
    "impact analysis",
    "change management",
    "baseline management",
    "verification and validation",
    "requirements decomposition",
    "traceability matrix",
    "bidirectional traceability",
    "coverage analysis",
    "requirements allocation",
    "derived requirements",
    "parent-child relationships",
    "trace links",
    "suspect links",
    "requirements attributes",
    "safety-critical",
    "compliance artifacts",
    "test coverage",
    "review status",
    "approval workflow",
)

# =============================================================================
# DOMAIN EVALUATION PROMPTS
# =============================================================================

CITATION_ACCURACY_PROMPT = """You are evaluating citation accuracy for industry standards.

Given the following:
- Expected Standards: {expected_standards}
- Answer: {answer}

Evaluate whether the answer correctly cites and references the relevant standards.
Check that standard names, numbers, and descriptions are accurate.

Score from 0.0 to 1.0:
- 1.0: All standard citations are accurate and complete
- 0.5: Some citations are accurate, others are missing or incorrect
- 0.0: Citations are incorrect or missing entirely

Respond with only a JSON object:
{{"score": <float>, "reasoning": "<brief explanation>"}}
"""

TRACEABILITY_COVERAGE_PROMPT = """You are evaluating traceability coverage in an answer.

Given the following:
- Question: {question}
- Answer: {answer}
- Expected Entities: {expected_entities}

Evaluate whether the answer properly addresses traceability concepts and covers
the expected entities (requirements, tests, design elements, etc.).

Score from 0.0 to 1.0:
- 1.0: Answer comprehensively covers traceability with all expected entities
- 0.5: Answer partially covers traceability concepts
- 0.0: Answer does not address traceability adequately

Respond with only a JSON object:
{{"score": <float>, "reasoning": "<brief explanation>"}}
"""

TECHNICAL_PRECISION_PROMPT = """You are evaluating technical precision in a \
requirements management answer.

Given the following:
- Question: {question}
- Answer: {answer}

Evaluate the technical accuracy of the answer in the requirements management domain.
Check for correct use of terminology, accurate descriptions of processes, and
proper understanding of requirements engineering concepts.

Score from 0.0 to 1.0:
- 1.0: Technically precise with correct terminology and concepts
- 0.5: Generally accurate but with some imprecision
- 0.0: Contains technical errors or misunderstandings

Respond with only a JSON object:
{{"score": <float>, "reasoning": "<brief explanation>"}}
"""

COMPLETENESS_SCORE_PROMPT = """You are evaluating the completeness of an answer.

Given the following:
- Question: {question}
- Answer: {answer}

Evaluate whether the answer comprehensively addresses all aspects of the question.
Consider whether key points, examples, and caveats are included.

Score from 0.0 to 1.0:
- 1.0: Answer is comprehensive and complete
- 0.5: Answer addresses main points but misses some aspects
- 0.0: Answer is incomplete or superficial

Respond with only a JSON object:
{{"score": <float>, "reasoning": "<brief explanation>"}}
"""

REGULATORY_ALIGNMENT_PROMPT = """You are evaluating regulatory alignment in an answer.

Given the following:
- Expected Standards: {expected_standards}
- Question: {question}
- Answer: {answer}

Evaluate whether the answer correctly aligns with regulatory requirements
and industry standards applicable to the question context.

Score from 0.0 to 1.0:
- 1.0: Answer fully aligns with relevant regulations and standards
- 0.5: Answer partially addresses regulatory considerations
- 0.0: Answer ignores or contradicts regulatory requirements

Respond with only a JSON object:
{{"score": <float>, "reasoning": "<brief explanation>"}}
"""
