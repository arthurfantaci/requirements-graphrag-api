"""Evaluator functions for prompt A/B testing.

This module provides evaluator functions used by the prompt comparison script
to assess prompt outputs during A/B testing experiments.

Evaluators are designed to work with the run_prompt_comparison.py script and
return dictionaries of metric names to scores (0.0 to 1.0).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

# Type alias for evaluator functions
EvaluatorFn = Callable[[dict[str, Any]], dict[str, float]]


def create_json_validity_evaluator() -> EvaluatorFn:
    """Create an evaluator that checks if output is valid JSON.

    Returns:
        Evaluator function that scores JSON validity.
    """

    def evaluate(run_output: dict[str, Any]) -> dict[str, float]:
        """Evaluate JSON validity of output.

        Args:
            run_output: Dict with 'output' key containing the LLM response.

        Returns:
            Dict with 'json_valid' score (1.0 if valid, 0.0 if invalid).
        """
        output = run_output.get("output", "")
        try:
            # Handle markdown code blocks
            if "```json" in output:
                match = re.search(r"```json\s*(.*?)\s*```", output, re.DOTALL)
                if match:
                    output = match.group(1)
            elif "```" in output:
                match = re.search(r"```\s*(.*?)\s*```", output, re.DOTALL)
                if match:
                    output = match.group(1)

            json.loads(output.strip())
            return {"json_valid": 1.0}
        except (json.JSONDecodeError, AttributeError):
            return {"json_valid": 0.0}

    return evaluate


def create_cypher_validity_evaluator() -> EvaluatorFn:
    """Create an evaluator that checks if output is valid Cypher syntax.

    This is a basic syntax check, not a full Cypher parser.

    Returns:
        Evaluator function that scores Cypher validity.
    """
    # Common Cypher keywords that indicate a valid query structure
    required_patterns = [
        r"\bMATCH\b",
        r"\bRETURN\b",
    ]
    valid_patterns = [
        r"\bWHERE\b",
        r"\bWITH\b",
        r"\bORDER\s+BY\b",
        r"\bLIMIT\b",
        r"\bCOUNT\s*\(",
        r"\bCOLLECT\s*\(",
        r"\bDISTINCT\b",
        r":\w+",  # Node labels like :Article
        r"\[:\w+\]",  # Relationship types like [:MENTIONED_IN]
        r"\(\w+\)",  # Node variables like (a)
    ]

    def evaluate(run_output: dict[str, Any]) -> dict[str, float]:
        """Evaluate Cypher syntax validity of output.

        Args:
            run_output: Dict with 'output' key containing the Cypher query.

        Returns:
            Dict with 'cypher_valid' score (0.0 to 1.0).
        """
        output = run_output.get("output", "").strip()

        # Handle markdown code blocks
        if "```cypher" in output:
            match = re.search(r"```cypher\s*(.*?)\s*```", output, re.DOTALL)
            if match:
                output = match.group(1)
        elif "```" in output:
            match = re.search(r"```\s*(.*?)\s*```", output, re.DOTALL)
            if match:
                output = match.group(1)

        if not output:
            return {"cypher_valid": 0.0}

        # Check for required patterns (MATCH and RETURN)
        has_required = all(
            re.search(pattern, output, re.IGNORECASE) for pattern in required_patterns
        )
        if not has_required:
            return {"cypher_valid": 0.0}

        # Count valid patterns for a quality score
        valid_count = sum(
            1 for pattern in valid_patterns if re.search(pattern, output, re.IGNORECASE)
        )
        # Base score of 0.5 for having MATCH and RETURN, plus bonus for valid patterns
        score = min(1.0, 0.5 + (valid_count * 0.1))

        return {"cypher_valid": score}

    return evaluate


def create_length_evaluator(min_length: int = 10, max_length: int = 2000) -> EvaluatorFn:
    """Create an evaluator that checks output length.

    Args:
        min_length: Minimum acceptable length.
        max_length: Maximum acceptable length.

    Returns:
        Evaluator function that scores length appropriateness.
    """

    def evaluate(run_output: dict[str, Any]) -> dict[str, float]:
        """Evaluate output length appropriateness.

        Args:
            run_output: Dict with 'output' key containing the response.

        Returns:
            Dict with 'length_appropriate' score.
        """
        output = run_output.get("output", "")
        length = len(output)

        if length < min_length:
            # Too short - proportional penalty
            score = length / min_length
        elif length > max_length:
            # Too long - proportional penalty
            score = max(0.0, 1.0 - (length - max_length) / max_length)
        else:
            score = 1.0

        return {"length_appropriate": score}

    return evaluate


def create_intent_accuracy_evaluator() -> EvaluatorFn:
    """Create an evaluator that checks intent classification accuracy.

    Compares the classified intent against expected intent in the run output.

    Returns:
        Evaluator function that scores intent classification accuracy.
    """

    def evaluate(run_output: dict[str, Any]) -> dict[str, float]:
        """Evaluate intent classification accuracy.

        Args:
            run_output: Dict with 'output' (LLM response) and 'expected' (ground truth).

        Returns:
            Dict with 'intent_accuracy' score (1.0 if correct, 0.0 if wrong).
        """
        output = run_output.get("output", "")
        expected = run_output.get("expected", {})
        expected_intent = expected.get("intent", "").lower()

        if not expected_intent:
            # No ground truth to compare against
            return {"intent_accuracy": 1.0}

        # Try to parse JSON output
        try:
            if "```json" in output:
                match = re.search(r"```json\s*(.*?)\s*```", output, re.DOTALL)
                if match:
                    output = match.group(1)
            parsed = json.loads(output.strip())
            classified_intent = parsed.get("intent", "").lower()
        except (json.JSONDecodeError, AttributeError):
            # Fall back to keyword matching
            output_lower = output.lower()
            if "explanatory" in output_lower:
                classified_intent = "explanatory"
            elif "structured" in output_lower:
                classified_intent = "structured"
            else:
                classified_intent = ""

        if classified_intent == expected_intent:
            return {"intent_accuracy": 1.0}
        return {"intent_accuracy": 0.0}

    return evaluate


__all__ = [
    "EvaluatorFn",
    "create_cypher_validity_evaluator",
    "create_intent_accuracy_evaluator",
    "create_json_validity_evaluator",
    "create_length_evaluator",
]
