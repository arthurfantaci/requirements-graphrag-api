"""Guardrails module for input/output safety checks.

This module provides security guardrails for the GraphRAG API:
- Prompt injection detection and blocking (Phase 1)
- PII detection and redaction (Phase 1)
- Toxicity detection (Phase 2)
- Topic boundary enforcement (Phase 2)
- Output content filtering (Phase 2)
- Structured event logging for security monitoring

Usage:
    from requirements_graphrag_api.guardrails import (
        # Phase 1 - Critical Security
        check_prompt_injection,
        detect_and_redact_pii,
        # Phase 2 - Content Safety
        check_toxicity,
        check_topic_relevance,
        filter_output,
        # Events
        log_guardrail_event,
    )
"""

from __future__ import annotations

from requirements_graphrag_api.guardrails.events import (
    ActionTaken,
    GuardrailEvent,
    GuardrailEventType,
    create_output_filter_event,
    create_topic_event,
    create_toxicity_event,
    log_guardrail_event,
)
from requirements_graphrag_api.guardrails.output_filter import (
    OutputFilterConfig,
    OutputFilterResult,
    filter_output,
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
from requirements_graphrag_api.guardrails.topic_guard import (
    TopicCheckResult,
    TopicClassification,
    TopicGuardConfig,
    check_topic_relevance,
)
from requirements_graphrag_api.guardrails.toxicity import (
    ToxicityCategory,
    ToxicityConfig,
    ToxicityResult,
    check_toxicity,
)

__all__ = [
    "ActionTaken",
    "GuardrailEvent",
    "GuardrailEventType",
    "InjectionCheckResult",
    "InjectionRisk",
    "OutputFilterConfig",
    "OutputFilterResult",
    "PIICheckResult",
    "TopicCheckResult",
    "TopicClassification",
    "TopicGuardConfig",
    "ToxicityCategory",
    "ToxicityConfig",
    "ToxicityResult",
    "check_prompt_injection",
    "check_topic_relevance",
    "check_toxicity",
    "create_output_filter_event",
    "create_topic_event",
    "create_toxicity_event",
    "detect_and_redact_pii",
    "filter_output",
    "log_guardrail_event",
]
