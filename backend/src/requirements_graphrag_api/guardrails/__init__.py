"""Guardrails module for input/output safety checks.

This module provides security guardrails for the GraphRAG API:
- Prompt injection detection and blocking (Phase 1)
- PII detection and redaction (Phase 1)
- Toxicity detection (Phase 2)
- Topic boundary enforcement (Phase 2)
- Output content filtering (Phase 2)
- Conversation history validation (Phase 4)
- Hallucination detection (Phase 4)
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
        # Phase 4 - Advanced Features
        validate_conversation_history,
        check_hallucination,
        # Events
        log_guardrail_event,
    )
"""

from __future__ import annotations

from requirements_graphrag_api.guardrails.conversation import (
    ConversationValidationResult,
    create_validated_history,
    validate_conversation_history,
)
from requirements_graphrag_api.guardrails.events import (
    ActionTaken,
    GuardrailEvent,
    GuardrailEventType,
    create_output_filter_event,
    create_topic_event,
    create_toxicity_event,
    log_guardrail_event,
)
from requirements_graphrag_api.guardrails.hallucination import (
    HALLUCINATION_WARNING,
    HALLUCINATION_WARNING_SHORT,
    GroundingLevel,
    HallucinationCheckResult,
    check_hallucination,
    check_hallucination_sync,
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
    "HALLUCINATION_WARNING",
    "HALLUCINATION_WARNING_SHORT",
    # Events
    "ActionTaken",
    # Phase 4 - Conversation Validation
    "ConversationValidationResult",
    # Phase 4 - Hallucination Detection
    "GroundingLevel",
    "GuardrailEvent",
    "GuardrailEventType",
    "HallucinationCheckResult",
    # Phase 1 - Prompt Injection
    "InjectionCheckResult",
    "InjectionRisk",
    # Phase 2 - Output Filtering
    "OutputFilterConfig",
    "OutputFilterResult",
    # Phase 1 - PII Detection
    "PIICheckResult",
    # Phase 2 - Topic Guard
    "TopicCheckResult",
    "TopicClassification",
    "TopicGuardConfig",
    # Phase 2 - Toxicity
    "ToxicityCategory",
    "ToxicityConfig",
    "ToxicityResult",
    "check_hallucination",
    "check_hallucination_sync",
    "check_prompt_injection",
    "check_topic_relevance",
    "check_toxicity",
    "create_output_filter_event",
    "create_topic_event",
    "create_toxicity_event",
    "create_validated_history",
    "detect_and_redact_pii",
    "filter_output",
    "log_guardrail_event",
    "validate_conversation_history",
]
