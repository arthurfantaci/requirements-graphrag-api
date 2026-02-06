"""Tests to verify all guardrail functions have @traceable_safe decorators.

Uses __wrapped__ attribute (set by functools.wraps) to verify decoration
at runtime — more reliable than source inspection with aliased imports.
"""

from __future__ import annotations

import importlib

import pytest

EXPECTED_TRACES = [
    ("validate_conversation_history", "conversation", "chain"),
    ("check_topic_relevance", "topic_guard", "llm"),
    ("check_prompt_injection", "prompt_injection", "chain"),
    ("check_toxicity", "toxicity", "chain"),
    ("detect_and_redact_pii", "pii_detection", "chain"),
    ("filter_output", "output_filter", "chain"),
    ("check_hallucination", "hallucination", "llm"),
    ("check_hallucination_sync", "hallucination", "llm"),
]


@pytest.mark.parametrize("func_name,module,run_type", EXPECTED_TRACES)
def test_guardrail_has_traceable_decorator(func_name: str, module: str, run_type: str) -> None:
    """Verify each guardrail function is decorated with @traceable_safe."""
    mod = importlib.import_module(f"requirements_graphrag_api.guardrails.{module}")
    func = getattr(mod, func_name)
    assert hasattr(func, "__wrapped__"), (
        f"{func_name} in {module} is not decorated with @traceable_safe"
    )
