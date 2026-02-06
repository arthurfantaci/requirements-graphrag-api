# Phase B: Conversation Layer & Streaming Synthesis

> **Status**: Planning (not started)
> **Prerequisite**: Phase A merged (PR #125, 2026-02-05)
> **Pre-planning**: Arthur running production profiling and LangSmith trace analysis

---

## 1. Deferred Failures from Phase A

These three failures were identified in thread `56a1e0c5` during Phase A analysis. They require new features (not bug fixes) and were deferred to Phase B.

### F1: Topic Guard Blocks Meta-Conversation Questions

**Problem**: When a user asks "what was my first question?" or "can you summarize our conversation?", the topic guard classifies this as OUT_OF_SCOPE because it doesn't contain domain keywords (requirements, traceability, etc.). The user gets a polite redirect instead of a useful answer.

**Root cause**: Topic guard only checks if the query relates to requirements management domain. Meta-conversation questions (about the conversation itself) are a valid use case that falls outside domain keyword matching.

**Solution direction**: Either (a) add a meta-conversation allowlist to topic guard, or (b) detect CONVERSATIONAL intent before topic guard runs and bypass it.

### F3: No CONVERSATIONAL Intent in Router

**Problem**: The intent classifier only knows EXPLANATORY and STRUCTURED. Questions like "summarize what we discussed", "what did you say about traceability?", or "can you rephrase your last answer?" have no matching intent. They get routed to the agentic RAG orchestrator, which runs a full retrieval pipeline for a question that should be answered from conversation history alone.

**Root cause**: `QueryIntent` enum in `core/routing.py` only has two values. The `_quick_classify()` and LLM classifier don't know about conversational patterns.

**Solution direction**: Add `QueryIntent.CONVERSATIONAL` that routes to a lightweight handler querying the checkpointer for conversation history instead of running the full orchestrator.

### F5: Checkpointer Never Queried for Conversation Context

**Problem**: The PostgreSQL checkpointer (`AsyncPostgresSaver`) stores conversation state via LangGraph, but it's only used for *writing* (persisting state after each turn). It's never *read* to retrieve prior conversation context for a new session or for meta-conversation questions.

**Root cause**: The checkpointer is wired for persistence but the application never calls `checkpointer.aget()` or equivalent to load prior state. Each request starts fresh with only the `conversation_history` array from the frontend.

**Solution direction**: For CONVERSATIONAL intent, query the checkpointer by thread_id to retrieve full conversation history, then answer from that context without RAG retrieval.

---

## 2. Current Architecture (Post-Phase A)

### What exists and works

- **Intent classifier** (`core/routing.py`): Two-stage (keyword + LLM), returns EXPLANATORY or STRUCTURED
- **Agentic orchestrator** (`core/agentic/orchestrator.py`): LangGraph StateGraph with RAG → Research → Synthesis subgraphs
- **Checkpointer** (`core/agentic/checkpoints.py`): `AsyncPostgresSaver` wired in `_generate_explanatory_events()`, creates thread configs with `get_thread_config(thread_id)`
- **Conversation history**: Frontend sends `conversation_history` array, backend converts to `HumanMessage`/`AIMessage` list (F2 fix in Phase A), passes to orchestrator state
- **previous_context**: Built from messages in `run_synthesis()`, passed to SYNTHESIS prompt (F4/F6 fix in Phase A)
- **Topic guard**: Keyword + LLM classification, runs before intent classification (parallelized in Phase A)
- **Streaming**: Post-synthesis JSON parsed and chunked into SSE token events (50-char chunks, 5ms delay)

### What doesn't exist yet

- No CONVERSATIONAL intent or routing
- No checkpointer read path (only write)
- No meta-conversation detection
- No streaming LLM output (synthesis waits for full JSON before chunking)

---

## 3. Candidate Phase B Features

### Tier 1: Conversation Layer (Core)

| Feature | Complexity | Dependencies |
|---------|-----------|--------------|
| Add `QueryIntent.CONVERSATIONAL` to router | Low | None |
| Add conversational keyword patterns to `_quick_classify()` | Low | CONVERSATIONAL intent |
| Add conversational LLM classification examples to INTENT_CLASSIFIER prompt | Low | CONVERSATIONAL intent |
| Build CONVERSATIONAL handler that queries checkpointer | Medium | CONVERSATIONAL intent, checkpointer read |
| Bypass topic guard for CONVERSATIONAL intent | Low | CONVERSATIONAL intent |
| Add meta-conversation detection to topic guard allowlist | Low | None |

### Tier 2: Streaming Synthesis (Performance)

| Feature | Complexity | Dependencies |
|---------|-----------|--------------|
| Switch SYNTHESIS prompt from JSON to streaming Markdown | High | Prompt redesign |
| Stream LLM tokens directly instead of post-synthesis chunking | Medium | Markdown synthesis |
| Move self-critique to separate non-blocking step | Medium | Markdown synthesis |
| Update frontend SSE handler for true token streaming | Low | Backend streaming |

### Tier 3: Quality & Observability

| Feature | Complexity | Dependencies |
|---------|-----------|--------------|
| Add "feedback/complaint" intent to classifier | Low | None |
| Build LangSmith adversarial evaluation dataset | Medium | None |
| Automated quality regression tests for optimizations | Medium | Eval dataset |

---

## 4. Performance Context (Phase A Results)

| Path | Avg TTFT | Avg Total | Target |
|------|----------|-----------|--------|
| Structured (Text2Cypher) | 0.5s | 1.5s | < 2s (met) |
| Explanatory (Agentic RAG) | 16.1s | 18.1s | < 3s (not met) |

**Root cause**: 90%+ of explanatory latency is LLM inference (sequential call chain of 2-7 LLM calls). Code-level optimizations in Phase A saved ~7s of overhead but hit a ceiling.

**Key insight**: The next meaningful TTFT improvement requires streaming synthesis output (Tier 2), which is an architectural change, not a code optimization.

### Arthur's Production Profiling Results

> **TODO**: Arthur to fill in after running production traces and LangSmith analysis post-merge. Key data to capture:
> - Per-component latency breakdown from LangSmith trace spans
> - Which LLM calls dominate (query expansion? synthesis? revision?)
> - Frequency of research skip heuristic firing
> - Frequency of synthesis revision triggering
> - Any quality regressions observed

---

## 5. Open Questions for Planning Session

1. **Scope**: Is Phase B just Tier 1 (conversation layer), or does it include Tier 2 (streaming synthesis)?
2. **Priority**: Should CONVERSATIONAL intent or streaming synthesis come first? (Conversation layer is lower risk but lower impact on latency; streaming is higher impact but higher risk)
3. **Frontend changes**: Does the frontend need updates for CONVERSATIONAL responses? (Different rendering than RAG answers?)
4. **Checkpointer schema**: Does the current LangGraph checkpointer schema support the queries we need, or do we need custom tables?
5. **Evaluation**: Should we build the quality regression dataset before or after making architectural changes?

---

## 6. Key Files for Phase B

| File | Relevance |
|------|-----------|
| `core/routing.py` | Add CONVERSATIONAL intent, update classifiers |
| `core/agentic/orchestrator.py` | May need CONVERSATIONAL bypass route |
| `core/agentic/checkpoints.py` | Add read path for conversation retrieval |
| `core/agentic/streaming.py` | Streaming synthesis changes |
| `core/agentic/subgraphs/synthesis.py` | Markdown streaming, self-critique separation |
| `routes/chat.py` | CONVERSATIONAL handler, topic guard bypass |
| `guardrails/topic_guard.py` | Meta-conversation allowlist |
| `prompts/definitions.py` | INTENT_CLASSIFIER prompt updates, possible CONVERSATIONAL prompt |
| `docs/performance/PHASE_A_PERFORMANCE_ANALYSIS.md` | Performance baseline and training rubric |
