"""Structured vector evaluators for Text2Cypher evaluation.

Provides 6 evaluators compatible with ``langsmith.evaluate()``:
- 5 deterministic (zero LLM cost): parse validity, schema adherence,
  execution success, result shape, safety
- 1 LLM-as-judge: result correctness

All evaluators follow the LangSmith convention:
    async def evaluator(run, example) -> dict[str, Any]
    Returns {"key": str, "score": float, "comment": str}
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from requirements_graphrag_api.evaluation.constants import (
    DEPRECATED_LABELS,
    INTERNAL_LABELS,
    JUDGE_MODEL,
    TIER1_PATTERNS,
    VALID_LABELS,
    VALID_RELATIONSHIPS,
    validate_cypher_comprehensive,
)

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run


# =============================================================================
# Deterministic evaluators
# =============================================================================


async def cypher_parse_validity(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Check if Cypher query is syntactically valid for execution.

    Binary: 1 if valid read-only Cypher, 0 otherwise.
    """
    outputs = run.outputs or {}
    cypher = outputs.get("cypher", "") or outputs.get("output", "")

    score, comment = _check_parse_validity(cypher)
    return {"key": "cypher_parse_valid", "score": score, "comment": comment}


def _check_parse_validity(cypher: str) -> tuple[int, str]:
    if not cypher or not cypher.strip():
        return 0, "Empty Cypher query"

    first_word = cypher.strip().split()[0].upper()
    valid_starters = {"MATCH", "OPTIONAL", "RETURN", "WITH", "UNWIND", "CALL"}
    if first_word not in valid_starters:
        return 0, f"Invalid starter keyword: {first_word}"

    if not re.search(r"\bRETURN\b", cypher, re.IGNORECASE):
        return 0, "Missing RETURN clause"

    for name, pat in TIER1_PATTERNS:
        if pat.search(cypher):
            return 0, f"Contains forbidden write keyword: {name}"

    # Balanced brackets (skip inside string literals)
    stack: list[str] = []
    openers = {"(": ")", "[": "]", "{": "}"}
    closers = {")", "]", "}"}
    in_string = False
    string_char: str | None = None
    for char in cypher:
        if in_string:
            if char == string_char:
                in_string = False
            continue
        if char in ('"', "'"):
            in_string = True
            string_char = char
            continue
        if char in openers:
            stack.append(openers[char])
        elif char in closers:
            if not stack or stack[-1] != char:
                return 0, f"Unbalanced bracket: unexpected '{char}'"
            stack.pop()
    if stack:
        return 0, f"Unbalanced brackets: {len(stack)} unclosed"

    return 1, "Valid read-only Cypher"


async def cypher_schema_adherence(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Check if Cypher uses valid labels and relationship types.

    Scores 0.0-1.0 based on label/relationship validity and coverage.
    """
    outputs = run.outputs or {}
    cypher = outputs.get("cypher", "") or outputs.get("output", "")
    expected_labels = (example.outputs or {}).get("expected_labels", []) if example else []

    score, comment = _check_schema_adherence(cypher, expected_labels)
    return {"key": "cypher_schema_adherence", "score": score, "comment": comment}


def _check_schema_adherence(
    cypher: str, expected_labels: list[str] | None = None
) -> tuple[float, str]:
    if not cypher or not cypher.strip():
        return 0.0, "Empty Cypher"

    found_labels = set(re.findall(r":(\w+)", cypher))
    found_rels = set(re.findall(r"\[:(\w+)", cypher))

    invalid_labels = found_labels - VALID_LABELS - INTERNAL_LABELS - DEPRECATED_LABELS
    if invalid_labels:
        return 0.0, f"Invalid/hallucinated labels: {invalid_labels}"

    deprecated_used = found_labels & DEPRECATED_LABELS
    if deprecated_used:
        return 0.2, f"Deprecated labels used: {deprecated_used} — have 0 nodes"

    internal_used = found_labels & INTERNAL_LABELS
    if internal_used:
        return 0.3, f"Internal labels used: {internal_used} — should use concrete types"

    invalid_rels = found_rels - VALID_RELATIONSHIPS
    if invalid_rels:
        return 0.3, f"Invalid/hallucinated relationships: {invalid_rels}"

    if not expected_labels:
        return 1.0, "No expected labels to check"

    expected_set = {ln.strip() for ln in expected_labels}
    matched = found_labels & expected_set
    score = len(matched) / len(expected_set) if expected_set else 1.0
    return score, f"Matched {len(matched)}/{len(expected_set)} expected labels"


async def cypher_execution_success(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Check if Cypher query executed successfully (no errors).

    Binary: 1 if executed without error, 0 otherwise.
    An empty result set (0 rows) with no error is still SUCCESS.
    """
    outputs = run.outputs or {}
    cypher = outputs.get("cypher", "") or outputs.get("output", "")
    error = outputs.get("error", "")
    row_count = outputs.get("row_count", 0)

    key = "cypher_execution_success"

    if not cypher or not cypher.strip():
        return {"key": key, "score": 0, "comment": "Empty Cypher — generation failed"}

    if error:
        error_lower = error.lower()
        snippet = error[:200]
        if "timeout" in error_lower or "timed out" in error_lower:
            return {"key": key, "score": 0, "comment": f"Execution timeout: {snippet}"}
        if "syntax" in error_lower:
            return {"key": key, "score": 0, "comment": f"Cypher syntax error: {snippet}"}
        if "unavailable" in error_lower or "connection" in error_lower:
            return {"key": key, "score": 0, "comment": f"Neo4j connection error: {snippet}"}
        return {"key": key, "score": 0, "comment": f"Execution error: {snippet}"}

    return {"key": key, "score": 1, "comment": f"Executed OK, {row_count} rows"}


async def result_shape_accuracy(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Check if query results have the expected shape (columns, non-empty).

    Penalizes empty results when the golden example expects non-empty output.
    """
    outputs = run.outputs or {}
    results = outputs.get("results", [])
    error = outputs.get("error", "")

    key = "result_shape_accuracy"

    if error:
        return {"key": key, "score": 0.0, "comment": "Execution error — no results"}

    expected = (example.outputs or {}) if example else {}
    expects_results = expected.get("expects_results", True)

    if not results:
        if expects_results:
            return {"key": key, "score": 0.0, "comment": "Empty results but expected non-empty"}
        return {"key": key, "score": 1.0, "comment": "Empty results as expected"}

    return {"key": key, "score": 1.0, "comment": f"{len(results)} rows returned"}


async def cypher_safety(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Comprehensive safety check using Tier 1-3 patterns.

    Binary: 1 if safe, 0 if any dangerous pattern found.
    """
    outputs = run.outputs or {}
    cypher = outputs.get("cypher", "") or outputs.get("output", "")

    if not cypher or not cypher.strip():
        return {"key": "cypher_safety", "score": 0, "comment": "Empty Cypher"}

    error = validate_cypher_comprehensive(cypher)
    if error:
        return {"key": "cypher_safety", "score": 0, "comment": error}

    return {"key": "cypher_safety", "score": 1, "comment": "Safe read-only query"}


# =============================================================================
# LLM-as-judge evaluator
# =============================================================================


async def result_correctness(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """LLM-as-judge: evaluates whether query results correctly answer the question.

    Uses EVAL_RESULT_CORRECTNESS prompt. Returns 0.0-1.0 score.
    """
    from langchain_openai import ChatOpenAI

    from requirements_graphrag_api.prompts import PromptName, get_prompt

    inputs = run.inputs or {}
    outputs = run.outputs or {}

    question = inputs.get("question", "")
    cypher = outputs.get("cypher", "") or outputs.get("output", "")
    results = str(outputs.get("results", ""))

    if not question or not cypher:
        return {"key": "result_correctness", "score": 0.0, "comment": "Missing question or cypher"}

    template = await get_prompt(PromptName.EVAL_RESULT_CORRECTNESS)
    messages = template.format_messages(question=question, cypher=cypher, results=results)

    llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)
    response = await llm.ainvoke(messages)

    from requirements_graphrag_api.evaluation.ragas_evaluators import _parse_llm_score

    score, reasoning = _parse_llm_score(response.content)
    return {"key": "result_correctness", "score": score, "comment": reasoning}


__all__ = [
    "cypher_execution_success",
    "cypher_parse_validity",
    "cypher_safety",
    "cypher_schema_adherence",
    "result_correctness",
    "result_shape_accuracy",
]
