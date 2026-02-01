# Phase 2: Content Safety Implementation

## Overview
Implement content moderation guardrails: toxicity detection, output filtering, and topic boundary enforcement.

**Timeline**: Week 3-4
**Priority**: P1 (High)
**Prerequisites**: Phase 1 complete

---

## 2.1 Toxicity Detection

### Files to Create

#### `backend/src/requirements_graphrag_api/guardrails/toxicity.py`

**Purpose**: Detect toxic, harmful, or inappropriate content in user inputs and LLM outputs.

**Approach**: Dual-layer detection
1. **Fast layer**: Word-list based profanity check (< 1ms)
2. **Accurate layer**: OpenAI Moderation API (< 500ms)

**Dependencies**: `better-profanity>=0.7.0`, `openai>=1.0.0` (already installed)

**Toxicity Categories** (aligned with OpenAI Moderation):
```python
class ToxicityCategory(StrEnum):
    HATE = "hate"                    # Hateful content
    HATE_THREATENING = "hate/threatening"
    HARASSMENT = "harassment"        # Harassing content
    HARASSMENT_THREATENING = "harassment/threatening"
    SELF_HARM = "self-harm"         # Self-harm content
    SELF_HARM_INTENT = "self-harm/intent"
    SELF_HARM_INSTRUCTIONS = "self-harm/instructions"
    SEXUAL = "sexual"               # Sexual content
    SEXUAL_MINORS = "sexual/minors"
    VIOLENCE = "violence"           # Violent content
    VIOLENCE_GRAPHIC = "violence/graphic"
```

**Key Components**:

```python
@dataclass(frozen=True, slots=True)
class ToxicityResult:
    """Result of toxicity check."""
    is_toxic: bool
    categories: list[ToxicityCategory]
    category_scores: dict[str, float]
    confidence: float
    should_block: bool
    check_type: str  # "fast" or "full"

async def check_toxicity_fast(text: str) -> ToxicityResult:
    """Fast profanity check using word lists (~1ms)."""
    from better_profanity import profanity
    contains_profanity = profanity.contains_profanity(text)
    return ToxicityResult(
        is_toxic=contains_profanity,
        categories=[ToxicityCategory.HARASSMENT] if contains_profanity else [],
        category_scores={},
        confidence=0.8 if contains_profanity else 0.0,
        should_block=contains_profanity,
        check_type="fast",
    )

async def check_toxicity_full(
    text: str,
    openai_client: AsyncOpenAI,
) -> ToxicityResult:
    """Full toxicity check using OpenAI Moderation API."""
    response = await openai_client.moderations.create(input=text)
    result = response.results[0]

    categories = []
    scores = {}
    for category, flagged in result.categories.model_dump().items():
        score = getattr(result.category_scores, category)
        scores[category] = score
        if flagged:
            categories.append(ToxicityCategory(category.replace("_", "-")))

    return ToxicityResult(
        is_toxic=result.flagged,
        categories=categories,
        category_scores=scores,
        confidence=max(scores.values()) if scores else 0.0,
        should_block=result.flagged,
        check_type="full",
    )

async def check_toxicity(
    text: str,
    openai_client: AsyncOpenAI | None = None,
    use_full_check: bool = True,
) -> ToxicityResult:
    """Check text for toxicity with configurable depth."""
    # Always do fast check first
    fast_result = await check_toxicity_fast(text)
    if fast_result.is_toxic:
        return fast_result

    # If fast check passes and full check requested, do OpenAI moderation
    if use_full_check and openai_client:
        return await check_toxicity_full(text, openai_client)

    return fast_result
```

**Configuration**:
```python
@dataclass(frozen=True, slots=True)
class ToxicityConfig:
    enabled: bool = True
    use_full_check: bool = True  # Use OpenAI Moderation API
    block_threshold: float = 0.7  # Confidence threshold for blocking
    categories_to_block: tuple[str, ...] = (
        "hate", "harassment", "self-harm", "sexual/minors", "violence/graphic"
    )
    categories_to_warn: tuple[str, ...] = (
        "sexual", "violence"
    )
```

---

## 2.2 Output Content Filtering

### Files to Create

#### `backend/src/requirements_graphrag_api/guardrails/output_filter.py`

**Purpose**: Filter LLM outputs for safety, accuracy, and appropriateness before returning to users.

**Checks Performed**:
1. **Toxicity check**: Same as input (using OpenAI Moderation)
2. **Off-topic detection**: Ensure response relates to requirements management
3. **Confidence scoring**: Estimate answer reliability
4. **Disclaimer injection**: Add warnings for uncertain answers

**Key Components**:

```python
@dataclass(frozen=True, slots=True)
class OutputFilterResult:
    """Result of output filtering."""
    is_safe: bool
    filtered_content: str
    original_content: str
    warnings: list[str]
    modifications: list[str]  # What was changed
    confidence_score: float
    should_add_disclaimer: bool

async def filter_output(
    output: str,
    original_query: str,
    retrieved_sources: list[dict],
    config: OutputFilterConfig,
    openai_client: AsyncOpenAI,
) -> OutputFilterResult:
    """Filter LLM output for safety and quality."""
    warnings = []
    modifications = []
    filtered = output

    # 1. Toxicity check on output
    toxicity = await check_toxicity(output, openai_client)
    if toxicity.should_block:
        return OutputFilterResult(
            is_safe=False,
            filtered_content=config.blocked_response_message,
            original_content=output,
            warnings=["Output contained harmful content"],
            modifications=["Replaced with safe response"],
            confidence_score=0.0,
            should_add_disclaimer=False,
        )

    # 2. Check for hallucination indicators
    hallucination_indicators = [
        "I don't have information about",
        "I cannot find",
        "Based on my knowledge",  # Not grounded in sources
        "I believe",
        "I think",
    ]
    confidence = 1.0
    for indicator in hallucination_indicators:
        if indicator.lower() in output.lower():
            confidence -= 0.2
            warnings.append(f"Potential uncertainty: '{indicator}'")

    # 3. Check source grounding (basic)
    if not retrieved_sources:
        confidence -= 0.3
        warnings.append("No sources retrieved for grounding")

    # 4. Add disclaimer if confidence is low
    should_add_disclaimer = confidence < 0.6

    return OutputFilterResult(
        is_safe=True,
        filtered_content=filtered,
        original_content=output,
        warnings=warnings,
        modifications=modifications,
        confidence_score=max(0.0, confidence),
        should_add_disclaimer=should_add_disclaimer,
    )
```

**Blocked Response Template**:
```python
BLOCKED_RESPONSE = """I apologize, but I'm unable to provide that response.

If you have questions about requirements management, traceability, or systems engineering, I'd be happy to help with those topics.

If you believe this is an error, please contact support."""
```

---

## 2.3 Topic Boundary Enforcement

### Files to Create

#### `backend/src/requirements_graphrag_api/guardrails/topic_guard.py`

**Purpose**: Ensure queries and responses stay within the intended domain (requirements management).

**Approach**: Two methods
1. **Keyword/embedding similarity**: Fast, for obvious cases
2. **LLM classification**: Accurate, for ambiguous cases

**In-Scope Topics**:
```python
IN_SCOPE_TOPICS = [
    "requirements management",
    "requirements traceability",
    "systems engineering",
    "product development",
    "software requirements",
    "hardware requirements",
    "compliance",
    "standards (ISO, IEC, FDA, etc.)",
    "verification and validation",
    "change management",
    "configuration management",
    "risk management",
    "Jama Software",
    "requirements documentation",
    "specification writing",
    "test case management",
]
```

**Out-of-Scope Topics**:
```python
OUT_OF_SCOPE_TOPICS = [
    "politics",
    "religion",
    "personal relationships",
    "medical diagnosis or treatment",
    "legal advice",
    "financial investment advice",
    "competitor product recommendations",
    "current events / news",
    "entertainment / pop culture",
    "cooking / recipes",
    "sports",
]
```

**Classification Prompt**:
```python
TOPIC_CLASSIFIER_PROMPT = """You are a topic classifier for a Requirements Management knowledge base chatbot.

The chatbot should ONLY answer questions about:
- Requirements management and traceability
- Systems engineering and product development
- Compliance standards (ISO, IEC, FDA, etc.)
- Jama Software products and features
- Related technical documentation practices

Classify the following user query as one of:
- IN_SCOPE: Related to the topics above
- OUT_OF_SCOPE: Unrelated to requirements management
- BORDERLINE: Could be related depending on context

User Query: {query}

Classification (respond with only IN_SCOPE, OUT_OF_SCOPE, or BORDERLINE):"""
```

**Key Components**:

```python
class TopicClassification(StrEnum):
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"
    BORDERLINE = "borderline"

@dataclass(frozen=True, slots=True)
class TopicCheckResult:
    classification: TopicClassification
    confidence: float
    suggested_response: str | None  # For out-of-scope, suggest redirect
    reasoning: str | None

async def check_topic_relevance(
    query: str,
    llm: ChatOpenAI,
    use_llm_classification: bool = True,
) -> TopicCheckResult:
    """Check if query is within the chatbot's intended scope."""

    # Fast check: keyword matching
    query_lower = query.lower()
    for topic in OUT_OF_SCOPE_TOPICS:
        if topic.lower() in query_lower:
            return TopicCheckResult(
                classification=TopicClassification.OUT_OF_SCOPE,
                confidence=0.9,
                suggested_response=get_redirect_response(topic),
                reasoning=f"Query contains out-of-scope topic: {topic}",
            )

    # LLM classification for ambiguous cases
    if use_llm_classification:
        response = await llm.ainvoke(
            TOPIC_CLASSIFIER_PROMPT.format(query=query)
        )
        classification = parse_classification(response.content)
        return TopicCheckResult(
            classification=classification,
            confidence=0.85,
            suggested_response=None if classification == TopicClassification.IN_SCOPE else get_redirect_response(None),
            reasoning=f"LLM classified as: {classification.value}",
        )

    # Default: allow
    return TopicCheckResult(
        classification=TopicClassification.IN_SCOPE,
        confidence=0.5,
        suggested_response=None,
        reasoning="Passed keyword filter, no LLM check",
    )

def get_redirect_response(detected_topic: str | None) -> str:
    """Get a polite redirect response for out-of-scope queries."""
    return f"""I'm a specialized assistant for Requirements Management topics.

I can help you with:
- Requirements traceability and management
- Systems engineering best practices
- Compliance with standards (ISO, IEC, FDA)
- Jama Software features and workflows

Would you like to ask about any of these topics instead?"""
```

---

### Integration Points

#### Update `backend/src/requirements_graphrag_api/routes/chat.py`

```python
from requirements_graphrag_api.guardrails import (
    check_prompt_injection,
    detect_and_redact_pii,
    check_toxicity,
    check_topic_relevance,
    filter_output,
    log_guardrail_event,
)

async def _generate_sse_events(...):
    # === INPUT GUARDRAILS ===

    # 1. Prompt injection (from Phase 1)
    injection_result = check_prompt_injection(request.message)
    if injection_result.should_block:
        yield error_event("Request blocked by safety filter")
        return

    # 2. PII detection (from Phase 1)
    pii_result = detect_and_redact_pii(request.message)
    safe_message = pii_result.anonymized_text

    # 3. Toxicity check (Phase 2)
    toxicity_result = await check_toxicity(safe_message, openai_client)
    if toxicity_result.should_block:
        yield error_event("Request contains inappropriate content")
        return

    # 4. Topic relevance (Phase 2)
    topic_result = await check_topic_relevance(safe_message, llm)
    if topic_result.classification == TopicClassification.OUT_OF_SCOPE:
        yield token_event(topic_result.suggested_response)
        yield done_event(topic_result.suggested_response)
        return

    # === PROCESS QUERY ===
    # ... existing generation logic ...

    # === OUTPUT GUARDRAILS ===

    # 5. Filter output (Phase 2)
    filter_result = await filter_output(
        output=full_answer,
        original_query=safe_message,
        retrieved_sources=sources,
        config=config.guardrails.output_filter,
        openai_client=openai_client,
    )

    if not filter_result.is_safe:
        yield error_event("Unable to generate appropriate response")
        return

    # Add disclaimer if needed
    final_answer = filter_result.filtered_content
    if filter_result.should_add_disclaimer:
        final_answer += "\n\n_Note: This answer may not fully address your question. Please verify with authoritative sources._"

    yield done_event(final_answer)
```

---

### Tests to Create

#### `backend/tests/test_guardrails/test_toxicity.py`

```python
"""Tests for toxicity detection."""

import pytest
from requirements_graphrag_api.guardrails.toxicity import (
    check_toxicity_fast,
    check_toxicity,
    ToxicityCategory,
)

class TestToxicityDetection:
    """Test toxicity detection functionality."""

    @pytest.mark.asyncio
    async def test_fast_check_detects_profanity(self):
        result = await check_toxicity_fast("This is a damn test")
        assert result.is_toxic is True
        assert result.check_type == "fast"

    @pytest.mark.asyncio
    async def test_fast_check_passes_clean_text(self):
        result = await check_toxicity_fast("What is requirements traceability?")
        assert result.is_toxic is False

    @pytest.mark.asyncio
    async def test_full_check_with_mock_openai(self, mock_openai_client):
        # Mock OpenAI moderation response
        result = await check_toxicity(
            "Some potentially harmful text",
            openai_client=mock_openai_client,
            use_full_check=True,
        )
        assert result.check_type == "full"
```

#### `backend/tests/test_guardrails/test_topic_guard.py`

```python
"""Tests for topic boundary enforcement."""

import pytest
from requirements_graphrag_api.guardrails.topic_guard import (
    check_topic_relevance,
    TopicClassification,
)

class TestTopicGuard:
    """Test topic boundary enforcement."""

    @pytest.mark.asyncio
    async def test_in_scope_query(self, mock_llm):
        result = await check_topic_relevance(
            "What is requirements traceability?",
            llm=mock_llm,
        )
        assert result.classification == TopicClassification.IN_SCOPE

    @pytest.mark.asyncio
    async def test_out_of_scope_politics(self, mock_llm):
        result = await check_topic_relevance(
            "What do you think about the election?",
            llm=mock_llm,
        )
        assert result.classification == TopicClassification.OUT_OF_SCOPE
        assert result.suggested_response is not None

    @pytest.mark.asyncio
    async def test_out_of_scope_medical(self, mock_llm):
        result = await check_topic_relevance(
            "What medication should I take for my headache?",
            llm=mock_llm,
        )
        assert result.classification == TopicClassification.OUT_OF_SCOPE
```

---

### Dependencies to Add

**`backend/pyproject.toml`**:
```toml
[project.dependencies]
# ... existing deps ...
better-profanity = ">=0.7.0"
# openai already included
```

---

### Configuration Additions

```python
@dataclass(frozen=True, slots=True)
class ContentSafetyConfig:
    """Configuration for content safety features."""
    # Toxicity
    toxicity_enabled: bool = True
    toxicity_use_full_check: bool = True
    toxicity_block_threshold: float = 0.7

    # Topic guard
    topic_guard_enabled: bool = True
    topic_guard_use_llm: bool = True

    # Output filter
    output_filter_enabled: bool = True
    output_filter_add_disclaimers: bool = True
    output_filter_confidence_threshold: float = 0.6

    # Messages
    blocked_response_message: str = "I'm unable to provide that response."
    out_of_scope_message: str = "I'm a specialized assistant for Requirements Management."
```

---

### Acceptance Criteria

- [ ] Toxicity detection blocks harmful content with >90% accuracy
- [ ] OpenAI Moderation API integration working
- [ ] Topic guard redirects out-of-scope queries politely
- [ ] Output filter catches toxic LLM responses
- [ ] Confidence scoring provides meaningful estimates
- [ ] Disclaimers added to low-confidence responses
- [ ] All checks complete in <1 second total
- [ ] Feature flags allow disabling each check
- [ ] 100% test coverage on new modules
