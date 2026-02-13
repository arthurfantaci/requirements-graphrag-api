"""Centralized evaluation constants.

Single source of truth for dataset names, queue names, experiment naming
patterns, schema validation sets, and Cypher safety patterns used across
evaluation scripts, CI pipelines, and regression gates.
"""

from __future__ import annotations

import re
from typing import Final

# =============================================================================
# DATASET NAMES (LangSmith)
# =============================================================================

# Legacy mixed dataset — keep read-only until Phase 4 CI stable (B5).
DATASET_LEGACY: Final[str] = "graphrag-rag-golden"

# Per-vector evaluation datasets (Phase 2)
DATASET_EXPLANATORY: Final[str] = "graphrag-eval-explanatory"
DATASET_STRUCTURED: Final[str] = "graphrag-eval-structured"
DATASET_CONVERSATIONAL: Final[str] = "graphrag-eval-conversational"
DATASET_INTENT: Final[str] = "graphrag-eval-intent"

ALL_VECTOR_DATASETS: Final[tuple[str, ...]] = (
    DATASET_EXPLANATORY,
    DATASET_STRUCTURED,
    DATASET_CONVERSATIONAL,
    DATASET_INTENT,
)

# Stale datasets to archive (W8)
DATASET_CRITIC_EVAL: Final[str] = "graphrag-critic-eval"

# =============================================================================
# ANNOTATION QUEUE NAMES (Phase 5)
# =============================================================================

QUEUE_USER_REPORTED: Final[str] = "user-reported-issues"

# =============================================================================
# EXPERIMENT NAMING
# =============================================================================

# Pattern: {vector}-{prompt_tag}-{YYYYMMDD-HHMMSS}
EXPERIMENT_PREFIX: Final[str] = "graphrag"


def experiment_name(vector: str, prompt_tag: str = "production") -> str:
    """Generate a timestamped experiment name.

    Args:
        vector: Vector name (explanatory, structured, conversational, intent).
        prompt_tag: Prompt tag (production, staging).

    Returns:
        Formatted experiment name like ``graphrag-explanatory-production-20260212-143000``.
    """
    from datetime import UTC, datetime

    ts = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    return f"{EXPERIMENT_PREFIX}-{vector}-{prompt_tag}-{ts}"


# =============================================================================
# GRAPH SCHEMA — VALID LABELS & RELATIONSHIPS
# =============================================================================

VALID_LABELS: Final[frozenset[str]] = frozenset(
    {
        "Article",
        "Artifact",
        "Bestpractice",
        "Challenge",
        "Chapter",
        "Chunk",
        "Concept",
        "Definition",
        "Image",
        "Industry",
        "Methodology",
        "Processstage",
        "Role",
        "Standard",
        "Tool",
        "Video",
        "Webinar",
    }
)

VALID_RELATIONSHIPS: Final[frozenset[str]] = frozenset(
    {
        "ADDRESSES",
        "ALTERNATIVE_TO",
        "APPLIES_TO",
        "COMPONENT_OF",
        "DEFINES",
        "FROM_ARTICLE",
        "HAS_IMAGE",
        "HAS_VIDEO",
        "HAS_WEBINAR",
        "MENTIONED_IN",
        "NEXT_CHUNK",
        "PREREQUISITE_FOR",
        "PRODUCES",
        "REFERENCES",
        "RELATED_TO",
        "REQUIRES",
        "USED_BY",
    }
)

# Labels that exist in the schema but have 0 nodes — treat as deprecated.
DEPRECATED_LABELS: Final[frozenset[str]] = frozenset({"Entity", "GlossaryTerm"})

# Internal neo4j-graphrag labels — never surface to users.
INTERNAL_LABELS: Final[frozenset[str]] = frozenset({"__Entity__", "__KGBuilder__"})


# =============================================================================
# CYPHER SAFETY PATTERNS
# =============================================================================

# Tier 1: Write operations (already blocked in production by text2cypher.py)
_TIER1_KEYWORDS: Final[tuple[str, ...]] = (
    "DELETE",
    "MERGE",
    "CREATE",
    "SET",
    "REMOVE",
    "DROP",
)

TIER1_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = tuple(
    (kw, re.compile(rf"\b{kw}\b", re.IGNORECASE)) for kw in _TIER1_KEYWORDS
)

# Tier 2: Procedure & subquery abuse
TIER2_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("CALL subquery", re.compile(r"\bCALL\s*\{", re.IGNORECASE)),
    ("CALL db.*", re.compile(r"\bCALL\s+db\.", re.IGNORECASE)),
    ("CALL dbms.*", re.compile(r"\bCALL\s+dbms\.", re.IGNORECASE)),
    ("CALL apoc.*", re.compile(r"\bCALL\s+apoc\.", re.IGNORECASE)),
    ("CALL gds.*", re.compile(r"\bCALL\s+gds\.", re.IGNORECASE)),
    ("LOAD CSV", re.compile(r"\bLOAD\s+CSV\b", re.IGNORECASE)),
    ("PERIODIC COMMIT", re.compile(r"\bUSING\s+PERIODIC\s+COMMIT\b", re.IGNORECASE)),
    ("FOREACH", re.compile(r"\bFOREACH\b", re.IGNORECASE)),
    ("DETACH", re.compile(r"\bDETACH\b", re.IGNORECASE)),
    ("file:// protocol", re.compile(r"file:///", re.IGNORECASE)),
)

# Tier 3: DDL / admin operations
TIER3_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("CREATE INDEX", re.compile(r"\bCREATE\s+INDEX\b", re.IGNORECASE)),
    ("CREATE CONSTRAINT", re.compile(r"\bCREATE\s+CONSTRAINT\b", re.IGNORECASE)),
    ("DROP INDEX", re.compile(r"\bDROP\s+INDEX\b", re.IGNORECASE)),
    ("DROP CONSTRAINT", re.compile(r"\bDROP\s+CONSTRAINT\b", re.IGNORECASE)),
    ("CREATE DATABASE", re.compile(r"\bCREATE\s+DATABASE\b", re.IGNORECASE)),
    ("DROP DATABASE", re.compile(r"\bDROP\s+DATABASE\b", re.IGNORECASE)),
    ("ALTER DATABASE", re.compile(r"\bALTER\s+DATABASE\b", re.IGNORECASE)),
    ("ALTER", re.compile(r"\bALTER\b", re.IGNORECASE)),
    ("RENAME", re.compile(r"\bRENAME\b", re.IGNORECASE)),
    ("TERMINATE", re.compile(r"\bTERMINATE\b", re.IGNORECASE)),
    ("STOP", re.compile(r"\bSTOP\b", re.IGNORECASE)),
    ("GRANT", re.compile(r"\bGRANT\b", re.IGNORECASE)),
    ("REVOKE", re.compile(r"\bREVOKE\b", re.IGNORECASE)),
    ("DENY", re.compile(r"\bDENY\b", re.IGNORECASE)),
)


def validate_cypher_comprehensive(cypher: str) -> str | None:
    """Extended safety validation for LLM-generated Cypher.

    Checks Tier 1 (write keywords) + Tier 2 (procedures/subqueries) +
    Tier 3 (DDL/admin). Only applied to LLM-generated queries, not to
    hardcoded schema introspection queries in ``_fetch_schema()``.

    Args:
        cypher: The Cypher query string to validate.

    Returns:
        Error message if unsafe, ``None`` if safe.
    """
    for name, pat in TIER1_PATTERNS:
        if pat.search(cypher):
            return f"Forbidden write keyword: {name}"

    for name, pat in TIER2_PATTERNS:
        if pat.search(cypher):
            return f"Forbidden pattern: {name}"

    for name, pat in TIER3_PATTERNS:
        if pat.search(cypher):
            return f"Forbidden admin operation: {name}"

    return None


# =============================================================================
# REGRESSION THRESHOLDS (per-vector)
# =============================================================================

# Minimum acceptable scores per vector before triggering regression failure.
# These are intentionally conservative to avoid false-alarm fatigue.
REGRESSION_THRESHOLDS: Final[dict[str, dict[str, float]]] = {
    "explanatory": {
        "faithfulness": 0.70,
        "answer_relevancy": 0.65,
        "context_precision": 0.60,
        "context_recall": 0.55,
        "answer_correctness": 0.50,
        "context_entity_recall": 0.40,
        "groundedness": 0.65,
    },
    "structured": {
        "cypher_parse_valid": 0.85,
        "cypher_schema_adherence": 0.75,
        "cypher_execution_success": 0.80,
        "cypher_safety": 1.0,
        "result_correctness": 0.50,
    },
    "conversational": {
        "conv_coherence": 0.65,
        "conv_context_retention": 0.55,
        "conv_hallucination": 0.80,
    },
    "intent": {
        "intent_accuracy": 0.85,
    },
}

# LLM model for judge evaluators
JUDGE_MODEL: Final[str] = "gpt-4o-mini"

__all__ = [
    "ALL_VECTOR_DATASETS",
    "DATASET_CONVERSATIONAL",
    "DATASET_CRITIC_EVAL",
    "DATASET_EXPLANATORY",
    "DATASET_INTENT",
    "DATASET_LEGACY",
    "DATASET_STRUCTURED",
    "DEPRECATED_LABELS",
    "EXPERIMENT_PREFIX",
    "INTERNAL_LABELS",
    "JUDGE_MODEL",
    "QUEUE_USER_REPORTED",
    "REGRESSION_THRESHOLDS",
    "TIER1_PATTERNS",
    "TIER2_PATTERNS",
    "TIER3_PATTERNS",
    "VALID_LABELS",
    "VALID_RELATIONSHIPS",
    "experiment_name",
    "validate_cypher_comprehensive",
]
