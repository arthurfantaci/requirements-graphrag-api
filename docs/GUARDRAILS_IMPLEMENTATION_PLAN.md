# Enterprise Guardrails Implementation Plan

## Executive Summary

This document outlines a comprehensive plan to bring the Requirements GraphRAG API up to enterprise-grade guardrail standards for customer-facing chatbot applications. The plan is organized by priority tier based on risk assessment and compliance requirements.

## Current State Assessment

| Category | Current Status | Gap |
|----------|---------------|-----|
| Input Validation | Basic Pydantic constraints | No content filtering |
| Output Safety | Prompt-level grounding only | No runtime moderation |
| Rate Limiting | None | Full implementation needed |
| Authentication | None | Full implementation needed |
| PII Protection | Log redaction only | No input/output detection |
| Prompt Injection | None | Full implementation needed |
| Toxicity Detection | None | Full implementation needed |

---

## Tier 1: Critical Security (Implement First)

### 1.1 Prompt Injection Protection

**Risk**: Critical - Attackers can manipulate LLM behavior, extract system prompts, or bypass safety measures.

**Recommended Approach**: Multi-layer defense

```python
# backend/src/requirements_graphrag_api/guardrails/prompt_injection.py

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

class InjectionRisk(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass(frozen=True, slots=True)
class InjectionCheckResult:
    risk_level: InjectionRisk
    detected_patterns: list[str]
    sanitized_input: str
    should_block: bool

# Known prompt injection patterns
INJECTION_PATTERNS = [
    # Direct instruction override attempts
    r"ignore\s+(previous|all|above)\s+(instructions?|prompts?)",
    r"disregard\s+(your|the)\s+(rules?|guidelines?|instructions?)",
    r"forget\s+(everything|all)\s+(above|before)",
    # Role manipulation
    r"you\s+are\s+now\s+(?:a|an)\s+\w+",
    r"act\s+as\s+(?:a|an)?\s*(?:different|new)",
    r"pretend\s+(?:you|to\s+be)",
    r"roleplay\s+as",
    # System prompt extraction
    r"(reveal|show|display|print|output)\s+(your|the)\s+(system|initial|original)\s+(prompt|instructions?)",
    r"what\s+(?:is|are)\s+your\s+(?:system|initial)\s+(?:prompt|instructions?)",
    # Delimiter injection
    r"```\s*system",
    r"\[INST\]",
    r"<\|im_start\|>",
    r"<\|system\|>",
    # Jailbreak patterns
    r"do\s+anything\s+now",
    r"DAN\s+mode",
    r"developer\s+mode",
    r"bypass\s+(?:your|the)\s+(?:filters?|restrictions?|safety)",
]

def check_prompt_injection(user_input: str) -> InjectionCheckResult:
    """Check user input for prompt injection attempts."""
    detected = []
    input_lower = user_input.lower()

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, input_lower, re.IGNORECASE):
            detected.append(pattern)

    # Determine risk level
    if len(detected) >= 3:
        risk = InjectionRisk.CRITICAL
    elif len(detected) >= 2:
        risk = InjectionRisk.HIGH
    elif len(detected) == 1:
        risk = InjectionRisk.MEDIUM
    else:
        risk = InjectionRisk.NONE

    return InjectionCheckResult(
        risk_level=risk,
        detected_patterns=detected,
        sanitized_input=user_input,  # Could add sanitization here
        should_block=risk in (InjectionRisk.HIGH, InjectionRisk.CRITICAL),
    )
```

**Integration Point**: Add as middleware before chat endpoint processing.

**Tools to Consider**:
- [Rebuff](https://github.com/protectai/rebuff) - Prompt injection detection
- [LangKit](https://github.com/whylabs/langkit) - LLM security toolkit
- [NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) - Comprehensive guardrails

---

### 1.2 Rate Limiting

**Risk**: High - API vulnerable to abuse, cost attacks, and denial of service.

**Recommended Approach**: SlowAPI with Redis backend for distributed rate limiting.

```python
# backend/src/requirements_graphrag_api/middleware/rate_limit.py

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse

# Initialize limiter
limiter = Limiter(key_func=get_remote_address)

# Rate limit configurations by endpoint type
RATE_LIMITS = {
    "chat": "20/minute",      # LLM calls are expensive
    "search": "60/minute",    # Vector search is cheaper
    "feedback": "30/minute",  # Prevent feedback spam
    "health": "120/minute",   # Health checks can be frequent
    "default": "100/minute",
}

def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": f"Rate limit exceeded. Try again in {exc.detail}",
            "retry_after": exc.detail,
        },
    )
```

**Integration in api.py**:
```python
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

@router.post("/chat")
@limiter.limit("20/minute")
async def chat_endpoint(request: Request, body: ChatRequest):
    ...
```

**Dependencies**: `slowapi`, `redis` (for distributed deployments)

---

### 1.3 PII Detection & Redaction

**Risk**: High - Privacy violations, GDPR/CCPA compliance issues.

**Recommended Approach**: Microsoft Presidio for comprehensive PII detection.

```python
# backend/src/requirements_graphrag_api/guardrails/pii_detection.py

from __future__ import annotations

from dataclasses import dataclass
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine

# Initialize engines (singleton pattern recommended)
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# PII types to detect
PII_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "US_BANK_NUMBER",
    "IP_ADDRESS",
    "LOCATION",
    "DATE_TIME",
    "NRP",  # Nationality, Religion, Political group
]

@dataclass(frozen=True, slots=True)
class PIICheckResult:
    contains_pii: bool
    detected_entities: list[dict]
    anonymized_text: str
    original_text: str

def detect_and_redact_pii(text: str) -> PIICheckResult:
    """Detect PII in text and return anonymized version."""
    # Analyze text for PII
    results = analyzer.analyze(
        text=text,
        entities=PII_ENTITIES,
        language="en",
    )

    if not results:
        return PIICheckResult(
            contains_pii=False,
            detected_entities=[],
            anonymized_text=text,
            original_text=text,
        )

    # Anonymize detected PII
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
    )

    # Convert results to serializable format
    entities = [
        {
            "type": r.entity_type,
            "start": r.start,
            "end": r.end,
            "score": r.score,
        }
        for r in results
    ]

    return PIICheckResult(
        contains_pii=True,
        detected_entities=entities,
        anonymized_text=anonymized.text,
        original_text=text,
    )
```

**Dependencies**: `presidio-analyzer`, `presidio-anonymizer`

---

## Tier 2: Content Safety (Implement Second)

### 2.1 Toxicity Detection

**Risk**: Medium-High - Reputational damage, user harm, compliance issues.

**Recommended Approach**: Dual-layer with fast heuristics + LLM-based classification.

```python
# backend/src/requirements_graphrag_api/guardrails/toxicity.py

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

class ToxicityCategory(StrEnum):
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    SELF_HARM = "self_harm"
    PROFANITY = "profanity"

@dataclass(frozen=True, slots=True)
class ToxicityResult:
    is_toxic: bool
    categories: list[ToxicityCategory]
    confidence: float
    should_block: bool

# Option 1: Use better-profanity for fast filtering
from better_profanity import profanity

def quick_toxicity_check(text: str) -> bool:
    """Fast profanity check using word lists."""
    return profanity.contains_profanity(text)

# Option 2: Use OpenAI Moderation API (more accurate)
async def openai_moderation_check(text: str, client) -> ToxicityResult:
    """Use OpenAI's moderation endpoint for comprehensive check."""
    response = await client.moderations.create(input=text)
    result = response.results[0]

    categories = []
    if result.categories.hate:
        categories.append(ToxicityCategory.HATE_SPEECH)
    if result.categories.harassment:
        categories.append(ToxicityCategory.HARASSMENT)
    if result.categories.violence:
        categories.append(ToxicityCategory.VIOLENCE)
    if result.categories.sexual:
        categories.append(ToxicityCategory.SEXUAL)
    if result.categories.self_harm:
        categories.append(ToxicityCategory.SELF_HARM)

    return ToxicityResult(
        is_toxic=result.flagged,
        categories=categories,
        confidence=max(
            result.category_scores.hate,
            result.category_scores.harassment,
            result.category_scores.violence,
        ),
        should_block=result.flagged,
    )

# Option 3: LlamaGuard for on-premise (no external API)
# Requires hosting LlamaGuard model locally
```

**Recommended**: OpenAI Moderation API for cloud deployments (free, fast), LlamaGuard for on-premise requirements.

**Dependencies**: `better-profanity` (fast), `openai` (moderation API)

---

### 2.2 Output Content Filtering

**Risk**: Medium - LLM could generate harmful, misleading, or off-topic content.

```python
# backend/src/requirements_graphrag_api/guardrails/output_filter.py

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class OutputFilterResult:
    is_safe: bool
    filtered_content: str
    warnings: list[str]
    confidence_score: float

async def filter_llm_output(
    output: str,
    context_sources: list[str],
    question: str,
) -> OutputFilterResult:
    """Filter LLM output for safety and accuracy."""
    warnings = []

    # Check 1: Toxicity in output
    toxicity = await openai_moderation_check(output)
    if toxicity.is_toxic:
        return OutputFilterResult(
            is_safe=False,
            filtered_content="I apologize, but I cannot provide that response.",
            warnings=["Output contained harmful content"],
            confidence_score=0.0,
        )

    # Check 2: Off-topic detection (output should relate to requirements management)
    # Could use embedding similarity to check relevance

    # Check 3: Hallucination detection (basic - check for unsupported claims)
    # More sophisticated: use AlignScore or similar

    return OutputFilterResult(
        is_safe=True,
        filtered_content=output,
        warnings=warnings,
        confidence_score=0.9,
    )
```

---

### 2.3 Topic Boundary Enforcement

**Risk**: Medium - Users could try to use the chatbot for unintended purposes.

```python
# backend/src/requirements_graphrag_api/guardrails/topic_guard.py

from __future__ import annotations

# Topics that are IN scope for this chatbot
IN_SCOPE_TOPICS = [
    "requirements management",
    "requirements traceability",
    "systems engineering",
    "product development",
    "software requirements",
    "compliance",
    "standards",
    "Jama Software",
    "requirements documentation",
]

# Topics that are explicitly OUT of scope
OUT_OF_SCOPE_TOPICS = [
    "politics",
    "religion",
    "personal advice",
    "medical advice",
    "legal advice",
    "financial advice",
    "competitor products",  # Unless comparing features
]

TOPIC_GUARD_PROMPT = """
You are a topic classifier for a Requirements Management knowledge base chatbot.

Determine if the following user query is:
1. IN_SCOPE - Related to requirements management, traceability, systems engineering, or product development
2. OUT_OF_SCOPE - Unrelated to the knowledge base domain
3. POTENTIALLY_HARMFUL - Contains requests for harmful, illegal, or inappropriate content

User Query: {query}

Respond with only one of: IN_SCOPE, OUT_OF_SCOPE, POTENTIALLY_HARMFUL
"""

async def check_topic_relevance(query: str, llm) -> str:
    """Classify if query is within the chatbot's intended scope."""
    response = await llm.ainvoke(TOPIC_GUARD_PROMPT.format(query=query))
    return response.content.strip()
```

---

## Tier 3: Access Control (Implement Third)

### 3.1 API Key Authentication

**Risk**: High - No access control means anyone can use the API.

```python
# backend/src/requirements_graphrag_api/middleware/auth.py

from __future__ import annotations

import hashlib
import secrets
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# In production, store hashed keys in database
VALID_API_KEYS = {
    # hash: {"name": "client_name", "tier": "standard|premium", "rate_limit": "100/min"}
}

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> dict:
    """Verify API key and return client metadata."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include X-API-Key header.",
        )

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    if key_hash not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return VALID_API_KEYS[key_hash]

def generate_api_key() -> str:
    """Generate a new API key."""
    return f"rgapi_{secrets.token_urlsafe(32)}"
```

**Integration**:
```python
@router.post("/chat")
async def chat_endpoint(
    request: Request,
    body: ChatRequest,
    client: dict = Depends(verify_api_key),  # Add authentication
):
    ...
```

---

### 3.2 Request Signing (Optional - High Security)

For enterprise customers requiring request integrity verification.

---

## Tier 4: Observability & Compliance

### 4.1 Guardrail Event Logging

```python
# backend/src/requirements_graphrag_api/guardrails/logging.py

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import StrEnum

logger = logging.getLogger("guardrails")

class GuardrailEventType(StrEnum):
    PROMPT_INJECTION_DETECTED = "prompt_injection_detected"
    PII_DETECTED = "pii_detected"
    TOXICITY_DETECTED = "toxicity_detected"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    TOPIC_OUT_OF_SCOPE = "topic_out_of_scope"
    OUTPUT_FILTERED = "output_filtered"

@dataclass
class GuardrailEvent:
    event_type: GuardrailEventType
    timestamp: datetime
    user_ip: str
    api_key_id: str | None
    input_hash: str  # Hash of input for correlation without storing PII
    details: dict
    action_taken: str  # "blocked", "warned", "allowed"

def log_guardrail_event(event: GuardrailEvent) -> None:
    """Log guardrail event for audit and analysis."""
    logger.warning(
        "Guardrail triggered: %s",
        event.event_type,
        extra={"guardrail_event": asdict(event)},
    )

    # Also send to LangSmith as feedback for analysis
    # This enables building datasets of edge cases
```

---

### 4.2 Compliance Dashboard Metrics

Track and report on:
- Guardrail trigger rates by type
- Blocked request percentage
- PII detection frequency
- Toxicity detection frequency
- False positive rates (via user feedback)

---

## Implementation Roadmap

### Phase 1: Critical Security (Week 1-2)
1. [ ] Implement prompt injection detection
2. [ ] Add rate limiting with SlowAPI
3. [ ] Integrate PII detection with Presidio
4. [ ] Add guardrail event logging

### Phase 2: Content Safety (Week 3-4)
1. [ ] Integrate OpenAI Moderation API for toxicity
2. [ ] Add output content filtering
3. [ ] Implement topic boundary enforcement
4. [ ] Add confidence scoring to responses

### Phase 3: Access Control (Week 5-6)
1. [ ] Implement API key authentication
2. [ ] Add per-key rate limiting
3. [ ] Create API key management endpoints
4. [ ] Add request audit logging

### Phase 4: Advanced Features (Week 7-8)
1. [ ] Integrate NeMo Guardrails for comprehensive protection
2. [ ] Add hallucination detection (AlignScore or similar)
3. [ ] Implement conversation history validation
4. [ ] Add request/response size limits
5. [ ] Create compliance dashboard

---

## Recommended Third-Party Tools

| Tool | Purpose | License | Notes |
|------|---------|---------|-------|
| [NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) | Comprehensive guardrails | Apache 2.0 | Best for full-featured implementation |
| [Presidio](https://github.com/microsoft/presidio) | PII detection | MIT | Microsoft-backed, production-ready |
| [SlowAPI](https://github.com/laurentS/slowapi) | Rate limiting | MIT | FastAPI-native |
| [better-profanity](https://github.com/snguyenthanh/better_profanity) | Fast profanity filter | MIT | Lightweight |
| [OpenAI Moderation](https://platform.openai.com/docs/guides/moderation) | Toxicity detection | API | Free, fast, accurate |
| [LlamaGuard](https://ai.meta.com/research/publications/llama-guard/) | On-premise safety | Llama License | For air-gapped deployments |

---

## Configuration Schema

```python
# backend/src/requirements_graphrag_api/config.py (additions)

@dataclass(frozen=True, slots=True)
class GuardrailConfig:
    """Configuration for guardrail features."""

    # Feature flags
    enable_prompt_injection_check: bool = True
    enable_pii_detection: bool = True
    enable_toxicity_check: bool = True
    enable_topic_guard: bool = True
    enable_output_filter: bool = True

    # Thresholds
    prompt_injection_block_threshold: str = "medium"  # none, low, medium, high
    toxicity_block_threshold: float = 0.7
    pii_redact_in_logs: bool = True

    # Rate limiting
    rate_limit_chat: str = "20/minute"
    rate_limit_search: str = "60/minute"
    rate_limit_default: str = "100/minute"

    # Authentication
    require_api_key: bool = False  # Enable in production
    api_key_header: str = "X-API-Key"
```

---

## Testing Strategy

### Unit Tests
- Test each guardrail function in isolation
- Test edge cases (unicode, long inputs, special characters)
- Test false positive/negative rates

### Integration Tests
- Test guardrail middleware with actual endpoints
- Test rate limiting under load
- Test authentication flows

### Red Team Testing
- Attempt prompt injection attacks
- Test PII bypass techniques
- Test toxicity filter evasion
- Document and address failures

---

## References

- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [NVIDIA NeMo Guardrails Documentation](https://docs.nvidia.com/nemo/guardrails/latest/)
- [LangChain Security Best Practices](https://docs.langchain.com/docs/security)
- [Microsoft Presidio Documentation](https://microsoft.github.io/presidio/)
