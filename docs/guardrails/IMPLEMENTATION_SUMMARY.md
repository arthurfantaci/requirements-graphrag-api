# Guardrails Implementation Summary

## Quick Reference for Implementation

This document provides a quick reference for implementing the guardrails system. Use this to resume implementation after context clear.

---

## GitHub Issues

| Phase | Issue | Priority | Status |
|-------|-------|----------|--------|
| Phase 1: Critical Security | [#89](https://github.com/arthurfantaci/requirements-graphrag-api/issues/89) | P0 | Not Started |
| Phase 2: Content Safety | [#90](https://github.com/arthurfantaci/requirements-graphrag-api/issues/90) | P1 | Not Started |
| Phase 3: Access Control | [#91](https://github.com/arthurfantaci/requirements-graphrag-api/issues/91) | P1 | Not Started |
| Phase 4: Advanced Features | [#92](https://github.com/arthurfantaci/requirements-graphrag-api/issues/92) | P2 | Not Started |

---

## Directory Structure to Create

```
backend/src/requirements_graphrag_api/
├── guardrails/                    # NEW - Phase 1-4
│   ├── __init__.py
│   ├── prompt_injection.py        # Phase 1
│   ├── pii_detection.py           # Phase 1
│   ├── events.py                  # Phase 1
│   ├── toxicity.py                # Phase 2
│   ├── output_filter.py           # Phase 2
│   ├── topic_guard.py             # Phase 2
│   ├── hallucination.py           # Phase 4
│   ├── conversation.py            # Phase 4
│   ├── metrics.py                 # Phase 4
│   └── nemo/                      # Phase 4 (optional)
│       ├── __init__.py
│       ├── config.py
│       └── config/
│           ├── config.yml
│           └── prompts.yml
├── auth/                          # NEW - Phase 3
│   ├── __init__.py
│   ├── api_key.py
│   ├── middleware.py
│   ├── scopes.py
│   └── audit.py
├── middleware/                    # NEW - Phase 1, 4
│   ├── __init__.py
│   ├── rate_limit.py              # Phase 1
│   ├── size_limit.py              # Phase 4
│   └── timeout.py                 # Phase 4
└── routes/
    └── admin.py                   # NEW - Phase 4
```

---

## Dependencies to Add

### Phase 1
```toml
slowapi = ">=0.1.9"
presidio-analyzer = ">=2.2.0"
presidio-anonymizer = ">=2.2.0"
```

### Phase 2
```toml
better-profanity = ">=0.7.0"
```

### Phase 4 (Optional)
```toml
nemoguardrails = ">=0.10.0"
```

---

## Implementation Order

### Phase 1 - Critical Security (Start Here)

1. **Create guardrails module structure**
   - `backend/src/requirements_graphrag_api/guardrails/__init__.py`

2. **Implement prompt injection detection**
   - File: `guardrails/prompt_injection.py`
   - Key function: `check_prompt_injection(text: str) -> InjectionCheckResult`
   - See: `docs/guardrails/PHASE1_CRITICAL_SECURITY.md`

3. **Implement PII detection**
   - File: `guardrails/pii_detection.py`
   - Key function: `detect_and_redact_pii(text: str) -> PIICheckResult`
   - Uses: Microsoft Presidio

4. **Implement guardrail event logging**
   - File: `guardrails/events.py`
   - Key function: `log_guardrail_event(event: GuardrailEvent)`

5. **Add rate limiting**
   - File: `middleware/rate_limit.py`
   - Uses: SlowAPI
   - Limits: 20/min for /chat, 60/min for /search

6. **Update config.py**
   - Add `GuardrailConfig` dataclass

7. **Integrate into routes/chat.py**
   - Add guardrail checks before processing
   - Log all guardrail events

8. **Write tests**
   - `tests/test_guardrails/test_prompt_injection.py`
   - `tests/test_guardrails/test_pii_detection.py`
   - `tests/test_guardrails/test_rate_limiting.py`

---

## Key Code Patterns

### Guardrail Check Pattern
```python
# In routes/chat.py
from requirements_graphrag_api.guardrails import (
    check_prompt_injection,
    detect_and_redact_pii,
    log_guardrail_event,
)

@router.post("/chat")
async def chat_endpoint(request: Request, body: ChatRequest):
    # 1. Check for prompt injection
    injection_result = check_prompt_injection(body.message)
    if injection_result.should_block:
        log_guardrail_event(GuardrailEvent(...))
        raise HTTPException(400, "Request blocked by safety filter")

    # 2. Detect and redact PII
    pii_result = detect_and_redact_pii(body.message)
    safe_message = pii_result.anonymized_text

    # 3. Continue with normal processing
    ...
```

### Rate Limiting Pattern
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/chat")
@limiter.limit("20/minute")
async def chat_endpoint(request: Request, body: ChatRequest):
    ...
```

---

## Testing Commands

```bash
# Run all guardrail tests
cd backend
uv run pytest tests/test_guardrails/ -v

# Run with coverage
uv run pytest tests/test_guardrails/ -v --cov=requirements_graphrag_api.guardrails

# Run specific test file
uv run pytest tests/test_guardrails/test_prompt_injection.py -v
```

---

## Verification Checklist

### Phase 1 Complete When:
- [ ] `check_prompt_injection()` detects known patterns
- [ ] `detect_and_redact_pii()` identifies and redacts PII
- [ ] Rate limiting returns 429 when exceeded
- [ ] Guardrail events logged to structured logger
- [ ] All tests passing
- [ ] `/chat` endpoint integrates all guardrails

### Phase 2 Complete When:
- [ ] Toxicity detection working (fast + OpenAI Moderation)
- [ ] Topic guard redirects off-topic queries
- [ ] Output filter catches toxic responses
- [ ] Confidence scoring added to responses

### Phase 3 Complete When:
- [ ] API key authentication working
- [ ] Per-key rate limiting by tier
- [ ] Audit logging captures all requests
- [ ] Admin endpoints for key management

### Phase 4 Complete When:
- [ ] Hallucination detection identifies ungrounded claims
- [ ] Conversation history validated
- [ ] Request size/timeout limits enforced
- [ ] Metrics dashboard available

---

## Documentation Files

| File | Description |
|------|-------------|
| `docs/guardrails/GUARDRAILS_IMPLEMENTATION_PLAN.md` | Executive overview |
| `docs/guardrails/PHASE1_CRITICAL_SECURITY.md` | Phase 1 detailed spec |
| `docs/guardrails/PHASE2_CONTENT_SAFETY.md` | Phase 2 detailed spec |
| `docs/guardrails/PHASE3_ACCESS_CONTROL.md` | Phase 3 detailed spec |
| `docs/guardrails/PHASE4_ADVANCED_FEATURES.md` | Phase 4 detailed spec |
| `docs/guardrails/IMPLEMENTATION_SUMMARY.md` | This file |

---

## Resume Implementation Command

To resume implementation after context clear:

```
Please implement Phase 1 of the guardrails system as specified in
docs/guardrails/PHASE1_CRITICAL_SECURITY.md. The GitHub issue is #89.
Start by creating the guardrails module structure and implementing
prompt injection detection.
```
