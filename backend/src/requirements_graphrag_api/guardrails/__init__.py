"""Guardrails module for input/output safety checks.

This module provides security guardrails for the GraphRAG API:
- Prompt injection detection and blocking
- PII detection and redaction
- Structured event logging for security monitoring

Usage:
    from requirements_graphrag_api.guardrails import (
        check_prompt_injection,
        detect_and_redact_pii,
        log_guardrail_event,
    )
"""

from __future__ import annotations

from requirements_graphrag_api.guardrails.events import (
    GuardrailEvent,
    GuardrailEventType,
    log_guardrail_event,
)
from requirements_graphrag_api.guardrails.pii_detection import (
    PIICheckResult,
    detect_and_redact_pii,
)
from requirements_graphrag_api.guardrails.prompt_injection import (
    InjectionCheckResult,
    InjectionRisk,
    check_prompt_injection,
)

__all__ = [
    "GuardrailEvent",
    "GuardrailEventType",
    "InjectionCheckResult",
    "InjectionRisk",
    "PIICheckResult",
    "check_prompt_injection",
    "detect_and_redact_pii",
    "log_guardrail_event",
]
