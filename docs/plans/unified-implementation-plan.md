# Unified Implementation Plan: LangSmith Platform-First Architecture

> Integrates: Master Implementation Plan + Plan Review Team (4 reports) + Prompt Engineering Team (6 deliverables) + Day 2 Operator Workflows
> Date: 2026-02-12
> Status: IN PROGRESS — Phase 0 ✅, Phase 1 ✅

---

## Executive Summary

This document is the **single source of truth** for implementing the LangSmith Platform-First Architecture. It reconciles findings from two independent agent teams and incorporates Day 2 operational readiness requirements:

- **Plan Review Team** (4 agents): Identified 9 consolidated BLOCKERs and 14 WARNINGs through independent codebase analysis, SDK verification, and cross-phase integration risk assessment.
- **Prompt Engineering Team** (6 agents): Audited all 14 prompts, revised 5 application prompts, revised 6 evaluation prompts, created 10 new judge/evaluator prompts, and validated all 23 prompts against pipeline code.
- **Day 2 Operations Analysis**: Evaluated post-deployment operator workflows for prompt engineering, evaluation, tracing, monitoring, and optimization. Identified 6 operational gaps and integrated fixes into the phase schedule.

**Design principle**: Every phase must leave the system not just *production-ready* but *operator-ready* — meaning a practitioner can run, diagnose, and iterate without building ad-hoc workarounds.

**Total effort**: ~27 hours across 8 phases (adjusted from original 22 hrs)
**Total monthly cost**: ~$29/mo (adjusted from original $22/mo)
**Files**: 16 new, 24 modified (19 backend + 4 frontend + 1 docs), 1 prompt deleted + 7 stale/orphaned Hub prompts cleaned
**Prompts**: 23 validated (8 unchanged, 5 revised, 10 new)
**Zero technical debt**: Every phase leaves the system production-ready AND operator-ready

---

## Consolidated BLOCKER Registry

All BLOCKERs from both teams, deduplicated and with prescribed fixes. Each must be resolved before or during the specified phase.

| # | Phase | Description | Source(s) | Fix |
|---|-------|-------------|-----------|-----|
| **B1** | 0 | Removing `RAG_GENERATION` breaks `run_ragas_evaluation.py:112` and `ci_evaluation.py:394`. Both scripts use it as the generation prompt. SYNTHESIS uses different variables (`{context}`,`{previous_context}`,`{question}` vs `{context}`,`{entities}`,`{question}`) and returns JSON instead of plain text. | Review: B1, PE: B1+B2 | In Phase 0, atomically: (1) replace `PromptName.RAG_GENERATION` with `PromptName.SYNTHESIS` in both scripts, (2) change `ainvoke` args: drop `{entities}`, add `{previous_context: ""}`, (3) add `json.loads()` + `parsed["answer"]` extraction, (4) remove `rag-generation` from `PROMPT_DATASET_MAP`, (5) update stale comment in `core/definitions.py:335` |
| **B2** | 5 | `client.add_runs_to_annotation_queue()` takes `queue_id` (UUID), NOT `queue_name` (str). Commented code at `feedback.py:188` will fail. | Review: B2 | Cache queue name-to-ID mapping at startup via `client.list_annotation_queues(name=...)`. Update `feedback.py` to resolve queue ID by name. |
| **B3** | 5 | `client.create_annotation_queue()` has no `rubric_items` parameter; only `rubric_instructions` (str). `setup_annotation_queues.py` will fail with `TypeError`. | Review: B3 | Create queues via SDK with `rubric_instructions` as descriptive text. Configure individual rubric feedback keys via LangSmith UI after queue creation. |
| **B4** | 2 | Dataset migration script is not idempotent. Partial failure leaves system in undefined state. No `--dry-run`, no rollback. | Review: B4 | Add `client.has_dataset()` checks, `--dry-run` flag, `--validate-only` flag. Split into two steps: (a) migrate existing 39 examples, (b) add new examples (separately reviewable). New examples must be PR-reviewed as code before pushing to LangSmith. |
| **B5** | 2 | 30-day auto-archival of old `graphrag-rag-golden` dataset creates a non-reversible cliff. | Review: B5 | Do NOT auto-archive. Archive manually only after Phase 4 CI is confirmed stable over 2+ nightly cycles. Keep local `GOLDEN_EXAMPLES` tuple intact as escape hatch. |
| **B6** | 3 | `BaseHTTPMiddleware` buffers `StreamingResponse` headers, which can break SSE streaming on `/chat`. This is the core user-facing feature. Note: existing `SizeLimitMiddleware` uses `BaseHTTPMiddleware` but doesn't modify response headers (the specific pattern that causes buffering). | Review: B6, NEW-1, NEW-3 | Use raw ASGI middleware for `TraceCorrelationMiddleware` (it needs to set the `X-Trace-ID` response header). **Updated**: The middleware extracts the existing OTel trace ID from `trace.get_current_span().get_span_context()` rather than generating a new UUID. W3C `traceparent` propagation is already handled by `FastAPIInstrumentor` + Sentry `browserTracingIntegration()` (see "Existing Observability Infrastructure"). |
| **B7** | Pre-0 | PromptName enum additions overlap between phases. Phase 2 adds `EVAL_CONV_*` (offline), Phase 5 adds `JUDGE_*` (online). Naming inconsistent across plan documents. | Review: B7, PE: W1 | Create unified PromptName addition plan. `EVAL_*` prefix = offline evaluators (Phase 2), `JUDGE_*` prefix = online judges (Phase 5). These are distinct prompts with different rubrics. Resolve `EVAL_RESULT_CORRECTNESS` name collision: adopt structured vector version (7-point calibration). |
| **B8** | 3 | `_generate_sse_events()` receives `ChatRequest` (Pydantic), not FastAPI `Request`. The `trace_id` from middleware has no path to the SSE generator without threading through 4 function signatures. | Review: B8 | Thread `trace_id` as new parameter: (1) `chat_endpoint` reads `request.state.trace_id`, (2) `_generate_sse_events` accepts `trace_id`, (3) all 3 handler calls pass `trace_id`, (4) all 3 handler signatures accept `trace_id`. Single atomic commit. |
| **B9** | Pre-0 | Queue name mismatch: 3-vector doc says `review-intent-errors`, master plan says `user-reported-issues`. | Review: B9 | Consolidate to `user-reported-issues` in ALL documents. Mark 3-vector doc queue names as superseded. |
| **B10** | 0/1 | 4 stale prompts in Hub (`graphrag-critic`, `graphrag-stepback`, `graphrag-router`, `graphrag-router-v2`). Must be cleaned before Phase 1 tagging. | Review: BLOCKER-04 | ✅ Deleted 4 stale prompts. Also discovered and deleted 3 additional orphans (`graphrag-entity-selector`, `graphrag-rag-generation`, `graphrag-agent-reasoning`) not in PromptName enum (B10b). Hub now has exactly 13 prompts = PromptName enum. |
| **B11** | 2 | `openevals` package referenced for groundedness evaluator but not installed. Import is `RAG_GROUNDEDNESS_PROMPT`, uses `context=` param (non-standard). | Review: SA recommendation | Add `openevals>=0.1.0` to `pyproject.toml` dev dependencies. Verify import path and parameter names before implementation. |

---

## Consolidated WARNING Registry (Priority-Ranked)

| # | Priority | Phase | Description | Fix |
|---|----------|-------|-------------|-----|
| W1 | HIGH | 2 | Do NOT rename `golden_dataset.py` — affects 9 import sites. | Restructure internally. Add new dataclasses, keep file name. |
| W2 | HIGH | All | Dataset, queue, and prompt names are hardcoded strings scattered across 10+ files. | Create `evaluation/constants.py` with all name constants. Import everywhere. |
| W3 | HIGH | Cost | Online evaluator traces auto-upgrade to extended retention (~$7.20/mo additional). Plan underestimates by ~33%. | Updated cost: $22/mo -> ~$29/mo. |
| W4 | MEDIUM | 3 | LangGraph `astream_events()` metadata propagation to child runs is unverified. | Write focused integration test as Phase 3 acceptance criteria. |
| W5 | MEDIUM | 1 | `push_prompt()` has `commit_tags` parameter for atomic push+tag. | Use `commit_tags=["production", "staging"]` in push script. Eliminates manual UI tagging step. |
| W6 | MEDIUM | All | 7+ test files need updates not listed in file manifest. | Add to manifest: `test_chat.py` (mock signature), `tests/benchmark/golden_dataset.py` (re-export), new evaluator test files. |
| W7 | MEDIUM | 5 | Batch Cypher evaluator running every 15 min will miss traces from failed windows. | Track last successfully evaluated trace timestamp in state file. |
| W8 | MEDIUM | 2 | 3 existing eval datasets (`graphrag-agentic-eval`, `graphrag-intent-classifier-eval`, `graphrag-text2cypher-eval`) not addressed. `graphrag-critic-eval` is stale. | Archive `graphrag-critic-eval`. Document disposition of other 3: merge into new vector datasets or keep read-only with deprecation note. |
| W9 | MEDIUM | 2 | Existing `create_intent_accuracy_evaluator` in `prompts/evaluation.py` handles only 2 intents. | Extend to handle `conversational` intent, don't create parallel `evaluation/intent_evaluator.py`. |
| W10 | MEDIUM | 0 | `_fetch_schema()` returns internal labels (`__KGBuilder__`, `__Entity__`), deprecated labels with 0 nodes. Confuses LLM. | Filter: `[lbl for lbl in labels if not lbl.startswith("__") and count > 0]` in `text2cypher.py`. Ship alongside TEXT2CYPHER prompt revision. |
| W11 | MEDIUM | 0 | `CYPHER_STARTERS` includes `EXPLAIN` and `PROFILE` — should be blocked per revised TEXT2CYPHER safety guidelines. | Remove `"EXPLAIN"` and `"PROFILE"` from `CYPHER_STARTERS` in `text2cypher.py`. Ship alongside TEXT2CYPHER prompt revision. |
| W12 | MEDIUM | 2 | `EVAL_CONV_CONTEXT_RETENTION` and `CONV_COMBINED_JUDGE` require `{expected_references}` variable not in current `GoldenExample`. | Add `expected_references: list[str] = field(default_factory=list)` to conversational dataset schema when creating `graphrag-eval-conversational` in Phase 2. |
| W13 | MEDIUM | 0 | SYNTHESIS JSON parse fallback (`synthesis.py:126-139`) loses critique data when `json.loads()` fails. | Add markdown-stripping before `json.loads()` in `synthesis.py:98`. Ship alongside SYNTHESIS v2.0 prompt change. |
| W14 | LOW | 3 | `request_id` (8-char) and `trace_id` (OTel trace ID) serve overlapping correlation purposes. | Keep both. Document: `request_id` = short log correlation, `trace_id` = OTel trace ID for cross-system linking (Sentry ↔ LangSmith OTLP ↔ LangSmith SDK). |
| W15 | LOW | 6 | FeedbackModal rubric UI is significant scope (93 -> 200+ lines). | Phase the UI: thumbs up/down + intent tag first, full rubric sliders in follow-up. |
| W16 | LOW | 3 | Starlette middleware execution order is counterintuitive (reverse registration). | Add code comment in `api.py` documenting actual execution order. |
| W17 | LOW | 5 | No alert for online evaluator failures (e.g., OpenAI outage). | Add "Evaluator Health" alert in Phase 6: success rate < 80% over 1 hour -> warning. |

---

## Pre-Implementation Checklist

Before any phase begins, resolve these items:

- [ ] **Create unified PromptName enum plan** (B7): Table of all additions across Phases 0, 2, 5
  - `EVAL_*` prefix for offline evaluators (Phase 2): `EVAL_RESULT_CORRECTNESS` (structured version), `EVAL_GROUNDEDNESS`, `EVAL_CONV_COHERENCE`, `EVAL_CONV_CONTEXT_RETENTION`, `EVAL_CONV_HALLUCINATION`, `EVAL_CONV_COMBINED`
  - `JUDGE_*` prefix for online judges (Phase 5): `JUDGE_HALLUCINATION`, `JUDGE_COHERENCE`
  - Remove the explanatory vector's duplicate `EVAL_RESULT_CORRECTNESS` — use structured version (7-point calibration)
- [ ] **Consolidate queue names** (B9): Standardize on `user-reported-issues` in all documents
- [ ] **Create `evaluation/constants.py`** (W2) with all dataset names, queue names, experiment naming patterns
- [ ] **Add `openevals>=0.1.0`** to `pyproject.toml` dev dependencies (B11)
- [ ] **Correct 3-vector doc** Section 2.4: Change `RAG_GENERATION` reference to `SYNTHESIS`

---

## Existing Observability Infrastructure

> **IMPORTANT**: The following observability stack was implemented prior to this plan and is already in the codebase. All phases below must build **on top of** this infrastructure, not alongside it.

### Frontend (shipped)

| Component | Package | File | What It Does |
|-----------|---------|------|-------------|
| Sentry SDK | `@sentry/react` v10.38.0 | `main.jsx` | Browser tracing, session replay (10% normal / 100% on error), React error handlers, trace propagation to backend via `tracePropagationTargets` |
| Vercel Analytics | `@vercel/analytics` v1.6.1 | `App.jsx` | Page views, web vitals (automatic) |
| Vercel Speed Insights | `@vercel/speed-insights` v1.3.1 | `App.jsx` | Core Web Vitals (LCP, FID, CLS) |
| SSE Metrics Hook | custom | `useSSEMetrics.js` | TTFT, TPS, token count, duration, errors → Sentry spans + breadcrumbs |
| Security Headers | - | `vercel.json` | X-Content-Type-Options, X-Frame-Options, Referrer-Policy, Permissions-Policy, immutable cache |

### Backend (shipped)

| Component | Package | File | What It Does |
|-----------|---------|------|-------------|
| Sentry SDK | `sentry-sdk[fastapi,langchain,langgraph]` v2.52+ | `observability.py:304-331` | Error tracking + performance monitoring. Called at `api.py:66` (module level, before app creation). |
| OpenTelemetry SDK | `opentelemetry-api/sdk` v1.39+ | `observability.py:334-383` | Dual-export TracerProvider: spans flow to **both** Sentry and LangSmith OTLP. `configure_otel()` reuses Sentry's TracerProvider when available. Called at `api.py:70`. |
| FastAPI OTel Instrumentation | `opentelemetry-instrumentation-fastapi` | `api.py:220-228` | Extracts W3C `traceparent` headers from frontend Sentry, creates per-request spans. |
| LangSmith OTLP Export | `opentelemetry-exporter-otlp-proto-http` | `observability.py:358-364` | Exports OTel spans to `https://api.smith.langchain.com/otel/v1/traces` with project header. |
| Traceable Decorator | custom | `observability.py:123-269` | `traceable_safe()` wraps `langsmith.traceable` with automatic input sanitization + config parameter handling. |
| Thread Metadata | custom | `observability.py:272-301` | `create_thread_metadata()` groups traces into LangSmith Threads via `thread_id`. |

### Architectural Consequence

The shared `TracerProvider` pattern means:
- **One OTel trace ID** flows through: Frontend Sentry → Backend FastAPI → LangSmith OTLP → Sentry Performance
- **LangSmith SDK traces** (LangChain/LangGraph auto-tracing) are a *separate* trace system with their own `run_id`
- **Cross-system linking** requires storing the OTel trace ID in LangSmith SDK trace metadata (Phase 3) so operators can navigate: `Sentry error → OTel trace_id → LangSmith metadata filter → LangChain trace tree`

This means Phase 3's `TraceCorrelationMiddleware` **must extract the existing OTel trace ID** from the current span context, NOT generate a new UUID. Generating a new UUID would create a parallel correlation system.

---

## Phase 0: Cleanup Dead Code + Prompt Revisions (1.5 hrs)

> Original: 30 min | Adjusted: 1.5 hrs (+1 hr for prompt revisions + code fixes)

### Goal
Remove orphaned code, revise TEXT2CYPHER and SYNTHESIS prompts in lockstep with their companion code changes, and clean Hub of stale prompts.

### Changes

| File | Action | Detail | Source |
|------|--------|--------|--------|
| `prompts/definitions.py` | MODIFY | (1) Remove `RAG_GENERATION` from enum + `PROMPT_DEFINITIONS`. (2) Replace SYNTHESIS template with v2.0 (remove unused `MessagesPlaceholder`, add JSON enforcement, add confidence calibration). (3) Replace TEXT2CYPHER template with v2.0 (16 node types, 16 relationships, 15 examples, `labels()` bug fix, expanded safety). | Master plan, PE: explanatory + structured |
| `core/text2cypher.py` | MODIFY | (1) Filter internal labels from `_fetch_schema()`: `[lbl for lbl in labels if not lbl.startswith("__") and count > 0]`. (2) Remove `"EXPLAIN"` and `"PROFILE"` from `CYPHER_STARTERS`. | PE: neo4j-engineer W4, W5 |
| `core/agentic/subgraphs/synthesis.py` | MODIFY | Add markdown-stripping before `json.loads()` at line ~98. | PE: pipeline-architect W3 |
| `scripts/run_ragas_evaluation.py` | MODIFY | Replace `PromptName.RAG_GENERATION` with `PromptName.SYNTHESIS`. Change `ainvoke` args: drop `{entities}`, add `{previous_context: ""}`. Parse JSON response (`parsed["answer"]`). | Review: B1, PE: B1+B2 |
| `scripts/ci_evaluation.py` | MODIFY | Same RAG_GENERATION -> SYNTHESIS migration as above. | Review: B1, PE: B1+B2 |
| `scripts/run_prompt_comparison.py` | MODIFY | Remove `rag-generation`, `critic`, `stepback` from `PROMPT_DATASET_MAP`. Add `synthesis`, `conversational`, `coreference-resolver`, `query-expansion`. | Master plan + Review: BLOCKER-02, INFO-01 |
| `core/definitions.py` | MODIFY | Update stale comment at line ~335 referencing `RAG_GENERATION` to say `SYNTHESIS`. | Review: B1 |

### Hub Cleanup (B10 + B10b)

Delete 4 stale Hub prompts before Phase 1 (B10 — identified by code review):
- `graphrag-critic` ✅ deleted
- `graphrag-stepback` ✅ deleted
- `graphrag-router` ✅ deleted
- `graphrag-router-v2` ✅ deleted

Delete 3 orphaned Hub prompts missed by original plan (B10b — identified by Hub audit post-Phase 1):
- `graphrag-entity-selector` ✅ deleted (superseded by `graphrag-query-expansion`)
- `graphrag-rag-generation` ✅ deleted (superseded by `graphrag-synthesis`)
- `graphrag-agent-reasoning` ✅ deleted (superseded by `graphrag-intent-classifier`)

> **Plan gap note**: The original review teams audited the *codebase* for references to deleted `PromptName` enum values but did not audit the *LangSmith Hub itself* for pre-existing orphans that predated the current naming convention. Future plans should include a Hub inventory step: `list_prompts() → diff against PromptName enum`.

### Prompt Content References

| Prompt | Version | Source Document | Key Changes |
|--------|---------|----------------|-------------|
| SYNTHESIS | 1.1.0 -> 2.0.0 | `prompts-explanatory-vector.md` Section 1.1 | Remove unused `MessagesPlaceholder("history")`, add JSON enforcement instruction, add confidence calibration (0.9+/0.7-0.9/<0.7) |
| TEXT2CYPHER | 1.0.0 -> 2.0.0 | `prompts-structured-vector.md` Section 1.1 | Add 8 missing node types, 10 missing relationships, fix `labels()` bug in examples 4+7, 4 new few-shot examples, expanded safety guidelines |
| CONVERSATIONAL | 1.0.0 -> 1.1.0 | `prompts-conversational-vector.md` Section 1.1 | Add history length guidance (20 messages), markdown handling, injection defense reinforcement |

### Acceptance Criteria
- [x] `RAG_GENERATION` not in `PromptName` enum
- [x] Both eval scripts use `PromptName.SYNTHESIS` with correct variable mapping
- [x] `_fetch_schema()` does not return `__KGBuilder__` or `__Entity__`
- [x] `CYPHER_STARTERS` does not contain `EXPLAIN` or `PROFILE`
- [x] `synthesis.py` strips markdown before `json.loads()`
- [x] 7 stale/orphaned Hub prompts deleted (4 original B10 + 3 discovered B10b)
- [x] `run_prompt_comparison.py` PROMPT_DATASET_MAP has only valid entries
- [x] 786 tests passing
- [x] `ruff check` clean

---

## Phase 1: Prompt Hub Seeding (1 hr)

> Unchanged from master plan. Enhanced with W5 (commit_tags).

### Goal
Push all 13 active prompts to LangSmith Hub with `:production` and `:staging` tags.

### Changes

| File | Action | Detail |
|------|--------|--------|
| `scripts/sync_prompts_to_hub.py` | NEW | CLI wrapping `catalog.push_all()` with `--dry-run` and `--only` flags. Use `commit_tags=["production", "staging"]` to tag atomically during push (W5). |
| `scripts/check_prompt_sync.py` | NEW | Compare Hub `:production` templates against local `definitions.py`. |
| `.github/workflows/prompt-sync-check.yml` | NEW | Weekly CI check (Monday 9am, warn only). |

### Acceptance Criteria
- [x] All 13 prompts visible in LangSmith Hub with `:production` and `:staging` tags
- [x] Hub-first loading works: `get_prompt(PromptName.SYNTHESIS)` pulls from Hub in production
- [x] Local fallback still works when `LANGSMITH_API_KEY` is unset
- [x] `sync_prompts_to_hub.py` supports `--dry-run`, `--only`, `--tags`
- [x] `check_prompt_sync.py` reports 13/13 in sync
- [x] Weekly CI workflow runs on Mondays (warn-only)

---

## Phase 2: 3-Vector Datasets + Evaluators (8 hrs)

> Original: 6 hrs | Adjusted: 8 hrs (+1.5 hr for idempotent migration, PE prompts, extend evaluator + 0.5 hr golden dataset mgmt)

### Goal
Split the mixed 39-example dataset into 4 intent-specific datasets. Register all evaluation prompts (revised + new). Build purpose-built evaluators.

### 2a. Dataset Migration (2.5 hrs)

| File | Action | Detail |
|------|--------|--------|
| `evaluation/constants.py` | NEW | All dataset names, queue names, experiment naming patterns as constants (W2). |
| `evaluation/golden_dataset.py` | MODIFY | **Do NOT rename file** (W1). Add `ConversationalExample` dataclass with `conversation_history`, `question`, `expected_answer`, `expected_references` (W12). Keep `GoldenExample` and `GOLDEN_EXAMPLES` intact. |
| `scripts/migrate_golden_datasets.py` | NEW | Idempotent migration (B4): `client.has_dataset()` checks, `--dry-run`, `--validate-only`. Two-step: (a) migrate existing 39 examples, (b) add new examples. |
| `scripts/create_golden_dataset.py` | MODIFY | Support `--dataset` flag for per-vector targeting. |

**4 LangSmith Datasets**:
- `graphrag-eval-explanatory` (46 examples: 35 migrated + 11 new)
- `graphrag-eval-structured` (18 examples: 4 migrated + 14 new)
- `graphrag-eval-conversational` (15 examples: all new, include `expected_references`)
- `graphrag-eval-intent` (30 examples: derived from all intents)

**Existing dataset disposition** (W8):
- `graphrag-rag-golden`: Keep read-only. Archive manually after Phase 4 CI stable (B5).
- `graphrag-critic-eval`: Archive immediately (critic removed).
- `graphrag-intent-classifier-eval` (18 examples): Merge useful examples into `graphrag-eval-intent`.
- `graphrag-text2cypher-eval` (11 examples): Merge useful examples into `graphrag-eval-structured`.
- `graphrag-agentic-eval` (16 examples): Keep for now; evaluate inclusion in future phases.

### 2b. Prompt Registration (1 hr)

Register all new evaluation prompts in `definitions.py`:

| PromptName Entry | Hub Name | Version | Source Document | Phase |
|-----------------|----------|---------|-----------------|-------|
| `EVAL_RESULT_CORRECTNESS` | `graphrag-eval-result-correctness` | 1.0.0 | `prompts-structured-vector.md` Section 2.1 (7-point calibration) | 2 |
| `EVAL_GROUNDEDNESS` | `graphrag-eval-groundedness` | 1.0.0 | `prompts-explanatory-vector.md` Section 3.2 | 2 |
| `EVAL_CONV_COHERENCE` | `graphrag-eval-conv-coherence` | 1.0.0 | `prompts-conversational-vector.md` Section 2.1 | 2 |
| `EVAL_CONV_CONTEXT_RETENTION` | `graphrag-eval-conv-context-retention` | 1.0.0 | `prompts-conversational-vector.md` Section 2.2 | 2 |
| `EVAL_CONV_HALLUCINATION` | `graphrag-eval-conv-hallucination` | 1.0.0 | `prompts-conversational-vector.md` Section 2.3 | 2 |
| `EVAL_CONV_COMBINED` | `graphrag-eval-conv-combined` | 1.0.0 | `prompts-conversational-vector.md` Section 2.4 | 2 |

Also register revised eval prompts (already in enum, update templates):

| PromptName Entry | Version Change | Source Document |
|-----------------|----------------|-----------------|
| `EVAL_FAITHFULNESS` | 2.0.0 -> 3.0.0 | `prompts-explanatory-vector.md` Section 2.1 |
| `EVAL_ANSWER_RELEVANCY` | 2.0.0 -> 3.0.0 | `prompts-explanatory-vector.md` Section 2.2 |
| `EVAL_CONTEXT_PRECISION` | 2.0.0 -> 3.0.0 | `prompts-explanatory-vector.md` Section 2.3 |
| `EVAL_CONTEXT_RECALL` | 2.0.0 -> 3.0.0 | `prompts-explanatory-vector.md` Section 2.4 |
| `EVAL_ANSWER_CORRECTNESS` | 1.0.0 -> 2.0.0 | `prompts-explanatory-vector.md` Section 2.5 |
| `EVAL_CONTEXT_ENTITY_RECALL` | 1.0.0 -> 2.0.0 | `prompts-explanatory-vector.md` Section 2.6 |

**Key changes across all 6 revised EVAL_* prompts**: Data variables (`{context}`, `{answer}`, `{ground_truth}`) moved from system message to human message per QS-5. Added calibration rubrics per QS-3.

### 2c. New Evaluators (3 hrs)

| File | Action | Detail |
|------|--------|--------|
| `evaluation/structured_evaluators.py` | NEW | 6 evaluators: `cypher_parse_validity`, `cypher_schema_adherence` (corrected VALID_LABELS: 17 labels, no `Entity`/`Topic`), `cypher_execution_success`, `result_shape_accuracy`, `cypher_safety` (enhanced: 22 dangerous patterns from neo4j-engineer), `result_correctness` (LLM judge). |
| `evaluation/conversational_evaluators.py` | NEW | 4 evaluators: `conversation_coherence`, `context_retention`, `coreference_accuracy`, `conversation_hallucination`. Offline judges use `reference_outputs`. |
| `evaluation/intent_evaluator.py` | MODIFY (existing `prompts/evaluation.py`) | Extend existing `create_intent_accuracy_evaluator()` to handle `conversational` intent (currently only handles 2 intents) (W9). Move to `evaluation/` module. |
| `evaluation/regression.py` | NEW | Regression gate logic: per-vector thresholds, must_pass enforcement. |
| `evaluation/__init__.py` | MODIFY | Export new evaluator modules. |

### 2d. Unified Evaluation Runner (1 hr)

| File | Action | Detail |
|------|--------|--------|
| `scripts/run_vector_evaluation.py` | NEW | Unified runner with `--vector {explanatory,structured,conversational,intent,all}` flag and `--prompt-tag {production,staging}` flag for single-prompt A/B testing. Replaces `run_ragas_evaluation.py` for multi-vector use. |

### 2e. Golden Dataset Management Pipeline (Day 2) (30 min)

| File | Action | Detail |
|------|--------|--------|
| `scripts/add_golden_example.py` | NEW | CLI to add a single example to a vector dataset. Accepts `--dataset`, `--question`, `--expected-answer`, `--intent`, `--from-trace <run_id>` (pulls inputs/outputs from LangSmith trace), `--dry-run`. Writes to both local code (`golden_dataset.py`) and LangSmith dataset. |
| `scripts/create_golden_dataset.py` | MODIFY | Add `--sync-from-langsmith` flag to pull examples added via LangSmith UI "Add to Dataset" back to local code for version control. Extends existing `--export` capability. |
| `scripts/import_feedback.py` | MODIFY | Add `--push-to-langsmith` flag so generated examples from human feedback can be pushed directly to the appropriate vector dataset (currently only writes to local JSON). |

**Day 2 operator workflow — "I found a bad trace, I want to improve the golden dataset":**
```bash
# 1. From annotation queue in LangSmith UI: review trace, click "Add to Dataset"
# 2. Sync back to local code for version control:
uv run python scripts/create_golden_dataset.py --sync-from-langsmith --dataset graphrag-eval-explanatory
# 3. Or add directly from a production trace:
uv run python scripts/add_golden_example.py \
  --from-trace abc123-run-id \
  --dataset graphrag-eval-explanatory \
  --expected-answer "The corrected answer" \
  --dry-run
# 4. Commit the updated golden_dataset.py to git
# 5. Re-run evaluation to see impact:
uv run python scripts/run_vector_evaluation.py --vector explanatory
```

### Acceptance Criteria
- [ ] 4 datasets created in LangSmith with correct example counts
- [ ] Migration script is idempotent (running twice produces same result)
- [ ] `--vector explanatory` runs 8 evaluators on 46 examples
- [ ] `--vector structured` runs 6 evaluators on 18 examples (5 deterministic + 1 LLM)
- [ ] `--vector conversational` runs 4 evaluators on 15 examples
- [ ] `--vector intent` runs 1 evaluator on 30 examples
- [ ] `--prompt-tag staging` evaluates against staging-tagged Hub prompts
- [ ] `add_golden_example.py --from-trace` correctly pulls trace inputs/outputs
- [ ] `create_golden_dataset.py --sync-from-langsmith` syncs UI-added examples to local code
- [ ] Full run cost < $0.70
- [ ] `VALID_LABELS` contains 17 entries (no `Entity`, `Topic`, `__KGBuilder__`, `__Entity__`)
- [ ] `cypher_safety` blocks all 22 dangerous patterns
- [ ] Old `graphrag-rag-golden` preserved (NOT archived)

---

## Phase 3: Trace Enrichment + Frontend Correlation (3 hrs)

> Original: 2 hrs | Adjusted: 3 hrs (+0.5 hr for raw ASGI middleware, +0.5 hr for frontend trace capture). Reduced from 3.5 hrs after discovering existing OTel + Sentry infrastructure.

### Goal
Bridge the gap between the **OTel trace system** (Sentry + LangSmith OTLP, already shipping) and the **LangSmith SDK trace system** (LangChain/LangGraph auto-tracing). Tag every LangSmith SDK trace with the OTel trace ID so operators can navigate between systems. Expose the OTel trace ID to the frontend via the SSE `done` event for feedback correlation.

### Existing Infrastructure (already deployed — see "Existing Observability Infrastructure")
- `FastAPIInstrumentor.instrument_app(app)` creates OTel spans for every request (`api.py:220-228`)
- `configure_otel()` exports spans to LangSmith OTLP + Sentry via shared `TracerProvider` (`observability.py:334-383`)
- Sentry `browserTracingIntegration()` sends W3C `traceparent` headers to backend (`main.jsx:10`)
- `tracePropagationTargets` includes the API URL (`main.jsx:16-19`)
- `useSSEMetrics.js` reports TTFT, TPS, token count, duration, errors to Sentry spans

### What This Phase Adds
The existing OTel infrastructure provides **request-level** tracing. What's missing is:
1. The OTel trace ID stored in **LangSmith SDK trace metadata** (so you can search by trace_id in LangSmith)
2. The OTel trace ID in the **SSE `done` event** (so the frontend can include it in feedback POST and Sentry attributes)
3. An `X-Trace-ID` response header for **debugging** (visible in browser DevTools)
4. **Intent, session_id, model** metadata enrichment on LangSmith traces

### 3a. Backend Trace Enrichment (2 hrs)

| File | Action | Detail |
|------|--------|--------|
| `middleware/tracing.py` | NEW | **Thin raw ASGI middleware** (B6), NOT `BaseHTTPMiddleware`. Reads OTel trace ID from `trace.get_current_span().get_span_context().trace_id` (NOT a new UUID). Falls back to `X-Trace-ID` request header if OTel span is not active, falls back to `uuid4()` as last resort. Stores in `scope["state"]["trace_id"]`, injects `X-Trace-ID` response header via `send` wrapper. ~25 lines of middleware code. |
| `middleware/__init__.py` | MODIFY (not NEW per INFO-06) | Add `tracing` module export. |
| `api.py` | MODIFY | Register `TraceCorrelationMiddleware` between CORS and Auth. Must be registered **after** `FastAPIInstrumentor.instrument_app(app)` so OTel span context is available. Add code comment documenting actual execution order (W16). |
| `routes/chat.py` | MODIFY | (1) Read `request.state.trace_id` in `chat_endpoint`. (2) Pass `trace_id` to `_generate_sse_events`. (3) Pass to all 3 handler calls. (4) Include `trace_id` in the SSE `done` event payload alongside existing `run_id`. (B8) |
| `routes/handlers.py` | MODIFY | All 3 handler signatures accept `trace_id`. Merge into `RunnableConfig` metadata (explanatory) and `langsmith_extra` (structured, conversational) alongside existing `create_thread_metadata()`. This stores the OTel trace_id in LangSmith SDK trace metadata, completing the cross-system link. (B8) |

### 3b. Frontend Trace Capture (Day 2) (1 hr)

**Why here, not Phase 6**: Trace correlation is the backbone of every Day 2 workflow — diagnostics, feedback analysis, Sentry-to-LangSmith linking. Deferring it to Phase 6 means Phases 4-5 are deployed without operator correlation, creating a blind spot during the most critical stabilization period.

| File | Action | Detail |
|------|--------|--------|
| `frontend/src/hooks/useSSEChat.js` | MODIFY | Capture `trace_id` from SSE `done` event payload (backend sends it alongside existing `run_id`). Store as `msg.traceId` on the assistant message object. Note: `useSSEChat.js` already captures `run_id` from the `done` event (line 125) — `trace_id` is added to the same payload. |
| `frontend/src/components/feedback/ResponseActions.jsx` | MODIFY | (1) Accept `traceId` prop (alongside existing `runId`). (2) Include `trace_id` in the `/feedback` POST body so backend can link feedback to LangSmith traces. |
| `frontend/src/hooks/useSSEMetrics.js` | MODIFY | Accept `traceId` from the SSE done event and add as `sse.trace_id` span attribute on the existing Sentry span. This enables "Sentry Performance → click span → see trace_id → search in LangSmith" workflow. Note: the OTel-level trace context already flows automatically via `browserTracingIntegration()` — this attribute is an *explicit* link to the LangSmith trace tree, not a duplicate. |

**Note on existing instrumentation**: Sentry `browserTracingIntegration()` already propagates W3C `traceparent` to the backend, creating automatic request-level correlation. The SSE `done` event `trace_id` serves a different purpose: it's the OTel trace ID for the *specific graph execution*, enabling per-message (not per-request) cross-system linking.

### Acceptance Criteria
- [ ] `X-Trace-ID` header echoed in backend response matches OTel span context trace ID
- [ ] SSE `done` event payload includes `trace_id` alongside `run_id`
- [ ] SSE streaming NOT broken (verify chat endpoint streams correctly)
- [ ] LangSmith SDK trace metadata contains `otel_trace_id`, `intent`, `session_id`, `model`
- [ ] Integration test: invoke graph with metadata in config, verify child traces inherit it (W4)
- [ ] `request_id` (8-char) and `trace_id` (OTel UUID) both documented and preserved (W14)
- [ ] Frontend: `ResponseActions` sends `trace_id` in feedback POST body
- [ ] Frontend: Sentry SSE span includes `sse.trace_id` attribute
- [ ] Cross-system link verified: Sentry span `sse.trace_id` → LangSmith filter `metadata.otel_trace_id` → trace tree

---

## Phase 4: CI Evaluation Refactor (3 hrs)

> Unchanged effort from master plan.

### Goal
Update CI tiers to run per-vector evaluations with regression gating.

### Changes

| File | Action | Detail |
|------|--------|--------|
| `scripts/ci_evaluation.py` | MODIFY | Refactor tier runner to accept `--vector` flag. Use per-vector datasets/evaluators from `evaluation/constants.py`. Add regression gate logic. |
| `scripts/compare_experiments.py` | MODIFY | Support per-vector comparison. Import dataset names from `evaluation/constants.py`. |
| `.github/workflows/evaluation.yml` | MODIFY | Update tier 2-4 steps to run multiple vectors. |

### Updated CI Tiers

| Tier | Trigger | Vectors | Gate |
|------|---------|---------|------|
| 1 | Every PR | None (prompt validation) | Format + parse |
| 2 | Merge to main | Intent + Explanatory(smoke) + Structured(smoke) | Intent >= 95%, safety 100% |
| 3 | Release | All 4 vectors (full) | Per-vector thresholds |
| 4 | Nightly | All 4 + regression comparison | Per-vector + no regression |

### Regression Gate Thresholds

| Vector | Hard Gates | Soft Gates (max drop from baseline) |
|--------|-----------|-------------------------------------|
| Intent | accuracy >= 95% | N/A |
| Explanatory | must_pass: correctness >= 0.5 AND faithfulness >= 0.5 | faithfulness: -0.05, correctness: -0.05 |
| Structured | cypher_parse: 100%, cypher_safety: 100% | execution_success: >= 90% |
| Conversational | conv_hallucination: 100% | coherence: -0.10 |

### Acceptance Criteria
- [ ] `ci_evaluation.py --tier 2` runs Intent + Explanatory(smoke) + Structured(smoke)
- [ ] `ci_evaluation.py --tier 3` runs all 4 vectors in parallel
- [ ] `ci_evaluation.py --tier 4` includes regression comparison
- [ ] Regression gate blocks CI when thresholds violated
- [ ] All dataset/queue names imported from `evaluation/constants.py`

---

## Phase 4.5: Day 2 Operator Workflows (2.5 hrs)

> NEW PHASE — Builds on Phases 2-4 to deliver complete operator workflows before annotation queues and online evaluation go live.

### Goal
Ensure every practitioner workflow is executable end-to-end: single-prompt A/B testing, ad-hoc trace evaluation, evaluation diagnostics, and prompt iteration with Hub tags.

### Rationale
Without this phase, Phases 5-6 deploy online evaluation and monitoring, but operators have no way to **diagnose** what the dashboards show them. An alert fires ("hallucination score dropped") — then what? This phase answers that question.

### 4.5a. Single-Prompt A/B Testing via Hub Tags (1 hr)

| File | Action | Detail |
|------|--------|--------|
| `scripts/run_prompt_comparison.py` | MODIFY | Refactor to support Hub tag comparison (not just variant suffixes). Add `--baseline-tag production --variant-tag staging` flags. Internally: pull both prompt versions from Hub, create separate targets, run `aevaluate()` on both against the same dataset, output side-by-side comparison. Update `PROMPT_DATASET_MAP` to use per-vector dataset constants. |
| `scripts/run_vector_evaluation.py` | MODIFY | Add `--prompt-tag` flag. When set, overrides `PROMPT_ENVIRONMENT` for the evaluation run only, allowing "evaluate explanatory vector with staging SYNTHESIS but production everything else." Internally: temporarily patches `catalog.environment` before `get_prompt()` calls for the target prompt only. |

**Day 2 operator workflow — "I revised SYNTHESIS and want to test it before promoting":**
```bash
# 1. Edit SYNTHESIS locally in definitions.py
# 2. Push to Hub as staging
uv run python scripts/sync_prompts_to_hub.py --only synthesis --tag staging

# 3. A/B test staging vs production
uv run python scripts/run_prompt_comparison.py synthesis \
  --baseline-tag production --variant-tag staging --iterations 2

# 4. Or run full evaluation with staging prompt
uv run python scripts/run_vector_evaluation.py --vector explanatory --prompt-tag staging

# 5. If scores improve, promote to production
uv run python scripts/sync_prompts_to_hub.py --only synthesis --tag production

# 6. Verify sync
uv run python scripts/check_prompt_sync.py
```

### 4.5b. Ad-Hoc Trace Evaluation (45 min)

| File | Action | Detail |
|------|--------|--------|
| `scripts/evaluate_trace.py` | NEW | Evaluate a single production trace against all evaluators for its vector. Accepts `--run-id <uuid>` or `--trace-id <uuid>`. Fetches trace from LangSmith via `client.read_run(run_id)`, extracts `inputs` and `outputs`, determines vector from `metadata.intent`, runs the appropriate evaluator suite, and outputs a diagnostic report. Supports `--evaluators` flag to run specific evaluators only. |

**Day 2 operator workflow — "A user reported a bad answer, I want to diagnose it":**
```bash
# 1. Get the run_id from user feedback or LangSmith UI
# 2. Run all evaluators for that trace
uv run python scripts/evaluate_trace.py --run-id abc123-def456

# Output:
# Trace: abc123-def456
# Intent: explanatory
# Vector: explanatory (8 evaluators)
# ─────────────────────────────────
# faithfulness:           0.42  ← BELOW THRESHOLD (0.5)
# answer_relevancy:       0.81  ✓
# answer_correctness:     0.35  ← BELOW THRESHOLD (0.5)
# context_precision:      0.90  ✓
# context_recall:         0.28  ← LOW (context missing key info)
# context_entity_recall:  0.50  ✓
# groundedness:           0.38  ← BELOW THRESHOLD (0.5)
# semantic_similarity:    0.65  ✓
# ─────────────────────────────────
# DIAGNOSIS: Low context_recall (0.28) is the root cause.
#   Retrieved context is missing key entities.
#   → Check retrieval quality for this question.
#   → Consider adding this trace to golden dataset.

# 3. Optionally add to golden dataset
uv run python scripts/add_golden_example.py \
  --from-trace abc123-def456 \
  --dataset graphrag-eval-explanatory \
  --expected-answer "The corrected answer here"
```

### 4.5c. Diagnostic Runbook (45 min)

| File | Action | Detail |
|------|--------|--------|
| `docs/runbooks/evaluation-degradation.md` | NEW | Step-by-step diagnostic guide for: (1) "Nightly eval scores dropped" — how to compare experiments, identify which evaluator degraded, trace to root cause. (2) "Online eval alert fired" — how to find the flagged traces in annotation queue, diagnose, and remediate. (3) "Prompt change caused regression" — how to A/B test, rollback to previous Hub tag, re-run CI. (4) "New golden examples needed" — how to add from traces, from annotation queue, or from scratch. |
| `docs/runbooks/prompt-iteration.md` | MODIFY | Update existing `PROMPT_ITERATION_WORKFLOW.md` with Hub-tag-based A/B testing workflow (replaces variant-suffix pattern). Add section on `run_prompt_comparison.py --baseline-tag --variant-tag` usage. |

### Acceptance Criteria
- [ ] `run_prompt_comparison.py synthesis --baseline-tag production --variant-tag staging` compares two Hub versions
- [ ] `run_vector_evaluation.py --vector explanatory --prompt-tag staging` evaluates with staging prompt
- [ ] `evaluate_trace.py --run-id <uuid>` outputs per-evaluator scores with threshold comparison
- [ ] `evaluate_trace.py` auto-detects vector from trace metadata and runs correct evaluator suite
- [ ] `evaluation-degradation.md` covers all 4 diagnostic scenarios
- [ ] Updated `prompt-iteration.md` documents Hub-tag A/B workflow

---

## Phase 5: Annotation Queues + Online Evaluation (4 hrs)

> Original: 3 hrs | Adjusted: 4 hrs (+1 hr for queue ID resolution, API signature fixes)

### Goal
Create annotation queues for human review. Deploy online judges for production monitoring.

### 5a. Annotation Queues (1.5 hrs)

| File | Action | Detail |
|------|--------|--------|
| `scripts/setup_annotation_queues.py` | NEW | Create 4 queues: `review-explanatory`, `review-structured`, `review-conversational`, `user-reported-issues`. Use `rubric_instructions` (str), NOT `rubric_items` (B3). Include name-to-ID cache utility. |
| `routes/feedback.py` | MODIFY | Fix: use `queue_id` (B2) not `queue_name`. Add intent-aware routing. Extend `FeedbackRequest` with `intent` and `rubric_scores` fields. |

### 5b. Online Evaluation (2.5 hrs)

| File | Action | Detail |
|------|--------|--------|
| `prompts/definitions.py` | MODIFY | Register `JUDGE_HALLUCINATION` and `JUDGE_COHERENCE` prompt entries. |
| `scripts/online_eval_cypher.py` | NEW | Batch evaluator for structured traces (runs every 15 min). Track last-processed timestamp in state file (W7). |

**Prompt Content References for Online Judges**:

| PromptName | Hub Name | Version | Source Document |
|------------|----------|---------|-----------------|
| `JUDGE_HALLUCINATION` | `graphrag-judge-hallucination` | 1.0.0 | `prompts-conversational-vector.md` Section 3.1 |
| `JUDGE_COHERENCE` | `graphrag-judge-coherence` | 1.0.0 | `prompts-conversational-vector.md` Section 3.2 |

**Key distinction**: Online judges do NOT have `reference_outputs` (run on production traces with sampling). Offline `EVAL_CONV_*` judges DO have `reference_outputs` (run against golden dataset).

**Online evaluators** (configured in LangSmith UI):
1. Hallucination detector: all traces, 20% sampling, score < 0.7 -> queue
2. Coherence evaluator: conversational traces, 30% sampling, score < 0.6 -> queue
3. Answer relevancy: all traces, 10% sampling
4. Cypher validity: structured traces, 100% via batch script

### Acceptance Criteria
- [ ] 4 annotation queues visible in LangSmith UI
- [ ] `feedback.py` uses `queue_id` (UUID), not `queue_name`
- [ ] Online evaluators run on production traces
- [ ] Low scores trigger queue routing
- [ ] Batch evaluator tracks last-processed timestamp

---

## Phase 6: Monitoring, Dashboards + Frontend Feedback (3.5 hrs)

> Original: 3 hrs | Adjusted: 3.5 hrs (+0.5 hr for cost-by-intent dashboard). Reduced from 4 hrs — Sentry SDK and OTel already deployed (see "Existing Observability Infrastructure"), cross-system correlation largely solved by shared TracerProvider (Phase 3 completes the link).

### Existing Infrastructure This Phase Builds On
- **Sentry** (frontend + backend): Already initialized, capturing errors, performance spans, and session replays. No installation needed.
- **OTel → LangSmith OTLP**: Already exporting spans. LangSmith already receives request-level traces.
- **SSE Metrics → Sentry**: `useSSEMetrics.js` already reports TTFT, TPS, token count, duration, errors as Sentry span attributes.
- **Cross-system correlation**: Phase 3 stores OTel trace_id in LangSmith SDK metadata and Sentry span attributes. The shared TracerProvider means OTel-level trace IDs already match across Sentry and LangSmith OTLP.

### What This Phase Adds
The existing infrastructure provides **data collection**. What's missing is **actionable surfacing**: LangSmith dashboards, alert rules, Sentry alert rules, and the feedback UI for intent-aware human review.

### 6a. Monitoring + Dashboards (1.5 hrs)

| File | Action | Detail |
|------|--------|--------|
| `scripts/quality_check.py` | NEW | Daily degradation detection (7-day vs 30-day baseline). Outputs structured JSON for CI integration. References `evaluate_trace.py` for deep-dive instructions. |

**LangSmith dashboards** (configured in UI):
1. **Production Overview**: trace volume, latency P50/P95, error rate, cost — grouped by intent
2. **Evaluation Health**: online eval scores trending over 30 days
3. **Intent Deep-Dive**: per-intent latency, token usage, eval scores
4. **Cost Attribution** (Day 2): cost per intent per day, cost per evaluator type, monthly trend. Uses `metadata.intent` grouping on traces. This surfaces "which intent is most expensive?" for optimization decisions.

**LangSmith alerts** (webhooks):
1. Error rate > 5% for 10 min → Critical
2. Latency P95 > 10s for 5 min → Warning
3. Hallucination avg < 0.65 over 24h → Warning
4. Cost > $5 in 1 hour → Critical
5. Online evaluator success rate < 80% over 1 hour → Warning (W17)

**Sentry alerts** (configured in Sentry UI — no code, SDK already deployed):
1. TTFT P95 > 3s → Warning (query: `sse.ttft_ms` span attribute from `useSSEMetrics.js`)
2. TPS < 5 sustained for 5 min → Warning (query: `sse.tokens_per_second` span attribute)
3. SSE stream error rate > 10% → Critical (query: `sse.success = false` span attribute)

**SSE Performance Metrics — Already Captured**:

| Metric | Sentry Span Attribute | Surfaced Where | Phase 6 Action |
|--------|----------------------|----------------|----------------|
| TTFT | `sse.ttft_ms` | Sentry Performance | Create alert rule (see above) |
| TPS | `sse.tokens_per_second` | Sentry Performance | Create alert rule (see above) |
| Token count | `sse.token_count` | Sentry Performance | Already visible |
| Duration | `sse.total_duration_ms` | Sentry Performance | Already visible |
| Error rate | `sse.success` | Sentry Issues | Create alert rule (see above) |
| OTel trace link | `sse.trace_id` (Phase 3b) | Sentry Performance | Navigate to LangSmith via metadata filter |

**Cross-system correlation** (largely solved by existing shared TracerProvider + Phase 3):
- **Automatic**: OTel trace IDs match across Sentry and LangSmith OTLP (shared `TracerProvider`)
- **Phase 3 addition**: OTel trace_id stored in LangSmith SDK metadata (`metadata.otel_trace_id`) and Sentry SSE span (`sse.trace_id`)
- **Operator workflow**: "Sentry error → copy `sse.trace_id` from span → LangSmith filter `metadata.otel_trace_id` → full LangChain trace tree"
- Document this workflow in `docs/runbooks/evaluation-degradation.md` (created in Phase 4.5c)

### 6b. Frontend Feedback Enhancement (2 hrs)

| File | Action | Detail |
|------|--------|--------|
| `FeedbackModal.jsx` | MODIFY | Initial scope: thumbs up/down + intent tag (W15). Follow-up scope: full rubric sliders per intent type. |
| `ResponseActions.jsx` | MODIFY | Pass `intent` prop from routing SSE event to modal. Note: `traceId` and `runId` props already added in Phase 3b. |
| `useSSEMetrics.js` | MODIFY | Add Sentry replay ID (`Sentry.getReplay()?.getReplayId()`) to span attributes alongside existing `sse.trace_id`. Enables "session replay → trace → LangSmith" three-way linking. |

**Note on trace_id**: Frontend capture of `trace_id` from SSE `done` event was moved to **Phase 3b** so correlation is available from the start. Phase 6 only adds the `intent` prop threading for the feedback modal and Sentry replay ID for session replay linking.

### Acceptance Criteria
- [ ] 4 dashboards created in LangSmith (including Cost Attribution)
- [ ] 5 LangSmith alerts configured and tested
- [ ] 3 Sentry alerts created for TTFT P95, TPS, and SSE error rate
- [ ] Frontend shows intent tag on feedback modal
- [ ] Cross-system diagnostic workflow documented in runbook
- [ ] `quality_check.py` references `evaluate_trace.py` for follow-up diagnostics
- [ ] Three-way link verified: Sentry replay → `sse.trace_id` → LangSmith `metadata.otel_trace_id`

---

## File Manifest

### New Files (16)

| File | Phase | Purpose |
|------|-------|---------|
| `evaluation/constants.py` | Pre-0 | Centralized name constants |
| `scripts/sync_prompts_to_hub.py` | 1 | Push prompts to Hub with tags |
| `scripts/check_prompt_sync.py` | 1 | Weekly sync verification |
| `.github/workflows/prompt-sync-check.yml` | 1 | Weekly CI workflow |
| `evaluation/structured_evaluators.py` | 2 | 6 Cypher evaluators |
| `evaluation/conversational_evaluators.py` | 2 | 4 conversation evaluators |
| `evaluation/regression.py` | 2 | Regression gate logic |
| `scripts/migrate_golden_datasets.py` | 2 | Idempotent dataset migration |
| `scripts/run_vector_evaluation.py` | 2, 4.5a | Unified multi-vector runner (created Phase 2, `--prompt-tag` added Phase 4.5a) |
| `scripts/add_golden_example.py` | 2e | Add single example from trace or manual input |
| `middleware/tracing.py` | 3 | Thin raw ASGI middleware: extracts OTel trace ID, sets X-Trace-ID header |
| `scripts/evaluate_trace.py` | 4.5b | Ad-hoc single-trace evaluation |
| `docs/runbooks/evaluation-degradation.md` | 4.5c | Diagnostic guide for 4 degradation scenarios |
| `scripts/setup_annotation_queues.py` | 5 | Queue creation with name-to-ID cache |
| `scripts/online_eval_cypher.py` | 5 | Batch Cypher evaluator with state tracking |
| `scripts/quality_check.py` | 6 | Daily degradation detection |

### Modified Files — Backend (19)

| File | Phase(s) | Change Summary |
|------|----------|---------------|
| `prompts/definitions.py` | 0, 2, 5 | Remove RAG_GENERATION, update SYNTHESIS v2.0 + TEXT2CYPHER v2.0 + CONVERSATIONAL v1.1 + all 6 EVAL_* v3.0/2.0, add 6 new eval prompts (Phase 2), add 2 online judge prompts (Phase 5) |
| `core/text2cypher.py` | 0 | Filter `_fetch_schema()`, remove EXPLAIN/PROFILE from CYPHER_STARTERS |
| `core/agentic/subgraphs/synthesis.py` | 0 | Add markdown-stripping before `json.loads()` |
| `core/definitions.py` | 0 | Update stale RAG_GENERATION comment |
| `scripts/run_ragas_evaluation.py` | 0 | RAG_GENERATION -> SYNTHESIS migration |
| `scripts/ci_evaluation.py` | 0, 4 | RAG_GENERATION -> SYNTHESIS (Phase 0), per-vector refactor (Phase 4) |
| `scripts/run_prompt_comparison.py` | 0, 4.5a | Fix stale PROMPT_DATASET_MAP (Phase 0), Hub-tag A/B testing (Phase 4.5a) |
| `scripts/create_golden_dataset.py` | 2, 2e | Add `--dataset` flag (Phase 2), `--sync-from-langsmith` (Phase 2e) |
| `scripts/import_feedback.py` | 2e | Add `--push-to-langsmith` flag for syncing feedback-derived examples to vector datasets |
| `evaluation/golden_dataset.py` | 2 | Add `ConversationalExample`, keep file name unchanged |
| `prompts/evaluation.py` | 2 | Extract `create_intent_accuracy_evaluator` to `evaluation/intent_evaluator.py`, extend for `conversational` intent (W9) |
| `evaluation/__init__.py` | 2 | Export new evaluator modules |
| `middleware/__init__.py` | 3 | Add tracing export (MODIFY, not NEW) |
| `api.py` | 3 | Register TraceCorrelationMiddleware + execution order comment |
| `routes/chat.py` | 3 | Thread trace_id, enrich root trace |
| `routes/handlers.py` | 3 | Thread trace_id to all 3 handlers |
| `scripts/compare_experiments.py` | 4 | Per-vector comparison |
| `.github/workflows/evaluation.yml` | 4 | Multi-vector CI steps |
| `routes/feedback.py` | 5 | Queue ID resolution + rubric scores |

### Modified Files — Frontend (4)

| File | Phase(s) | Change |
|------|----------|--------|
| `frontend/src/hooks/useSSEChat.js` | 3b | Capture `trace_id` from SSE `done` event payload, store as `msg.traceId` |
| `frontend/src/components/feedback/ResponseActions.jsx` | 3b, 6 | Accept `traceId` prop + include in feedback POST (Phase 3b), pass `intent` prop (Phase 6) |
| `frontend/src/hooks/useSSEMetrics.js` | 3b, 6 | Add `sse.trace_id` to existing Sentry span (Phase 3b), add Sentry replay ID (Phase 6) |
| `frontend/src/components/feedback/FeedbackModal.jsx` | 6 | Intent tag + rubric UI (phased: thumbs up/down first, full sliders later) |

### Modified Files — Docs (1)

| File | Phase | Change |
|------|-------|--------|
| `docs/runbooks/prompt-iteration.md` | 4.5c | Update existing workflow with Hub-tag A/B testing pattern (replaces variant-suffix) |

---

## Dependency Graph

```
Pre-Implementation (constants, naming, openevals dep)
  |
  v
Phase 0 (cleanup + prompt revisions + Hub cleanup)
  |
  +---> Phase 1 (Hub seeding)
  |
  +---> Phase 2 (datasets + evaluators + golden mgmt) --+----+
  |                                                      |    |
  +---> Phase 3 (trace enrichment + frontend corr.) ----+|    |
                                                        ||    |
                                                        vv    |
                                                 Phase 4 (CI) |
                                                        |     |
                                                        v     v
                                                 Phase 4.5 (Day 2 workflows)
                                                        |
                                                        v
                                                 Phase 5 (queues + online eval)
                                                        |
                                                        v
                                                 Phase 6 (monitoring + frontend feedback)
```

- Phases 2 and 3 can run in parallel (no file overlap after Phase 0).
- Phase 4.5 depends on **both** Phase 4 (CI infrastructure) **and** Phase 2 (evaluators + datasets + run_vector_evaluation.py).

---

## Prompt Inventory (23 Validated)

### Application Prompts (8)

| # | Prompt | Version | Status | Phase |
|---|--------|---------|--------|-------|
| 1 | INTENT_CLASSIFIER | 1.1.0 | Unchanged | - |
| 2 | QUERY_UPDATER | 1.0.0 | Unchanged | - |
| 3 | QUERY_EXPANSION | 1.0.0 | Unchanged | - |
| 4 | COREFERENCE_RESOLVER | 1.0.0 | Unchanged | - |
| 5 | SYNTHESIS | 1.1.0 -> **2.0.0** | Revised | 0 |
| 6 | TEXT2CYPHER | 1.0.0 -> **2.0.0** | Revised | 0 |
| 7 | CONVERSATIONAL | 1.0.0 -> **1.1.0** | Revised | 0 |
| 8 | ~~RAG_GENERATION~~ | ~~1.1.0~~ | **Deleted** | 0 |

### Evaluation Prompts — Revised (6)

| # | Prompt | Version | Key Change | Phase |
|---|--------|---------|------------|-------|
| 9 | EVAL_FAITHFULNESS | 2.0.0 -> **3.0.0** | Data moved to human message, calibration rubric | 2 |
| 10 | EVAL_ANSWER_RELEVANCY | 2.0.0 -> **3.0.0** | Operationalized scoring formula | 2 |
| 11 | EVAL_CONTEXT_PRECISION | 2.0.0 -> **3.0.0** | Data moved to human message | 2 |
| 12 | EVAL_CONTEXT_RECALL | 2.0.0 -> **3.0.0** | Data moved to human message | 2 |
| 13 | EVAL_ANSWER_CORRECTNESS | 1.0.0 -> **2.0.0** | Minor calibration | 2 |
| 14 | EVAL_CONTEXT_ENTITY_RECALL | 1.0.0 -> **2.0.0** | Minor calibration | 2 |

### Evaluation Prompts — New (9)

| # | Prompt | Version | Vector | Phase |
|---|--------|---------|--------|-------|
| 15 | EVAL_RESULT_CORRECTNESS | 1.0.0 | Structured | 2 |
| 16 | EVAL_GROUNDEDNESS | 1.0.0 | Explanatory | 2 |
| 17 | EVAL_CONV_COHERENCE | 1.0.0 | Conversational | 2 |
| 18 | EVAL_CONV_CONTEXT_RETENTION | 1.0.0 | Conversational | 2 |
| 19 | EVAL_CONV_HALLUCINATION | 1.0.0 | Conversational | 2 |
| 20 | EVAL_CONV_COMBINED | 1.0.0 | Conversational | 2 |
| 21 | JUDGE_HALLUCINATION | 1.0.0 | Online (all) | 5 |
| 22 | JUDGE_COHERENCE | 1.0.0 | Online (conv) | 5 |

Note: Prompt #15 uses the **structured vector version** (7-point calibration) per B7 resolution. The explanatory vector's duplicate version is discarded.

---

## Cost Analysis (Updated)

| Category | Monthly Cost | Notes |
|----------|-------------|-------|
| Online evaluation (LLM calls) | ~$1.70 | 48 evals/day at gpt-4o-mini rates |
| Extended retention (online eval traces) | ~$7.20 | ~1,440 traces/mo at $5/1k (W3) |
| Full eval run (all vectors) | ~$0.65 per run | 431 LLM calls across 109 examples |
| CI evaluation (nightly) | ~$19.50/mo | $0.65/run x 30 nights |
| LangSmith traces (production) | Free | Within 5k free tier |
| **Total incremental** | **~$29/mo** | +$7/mo vs original estimate |

---

## Effort Summary (Updated)

| Phase | Original | Adjusted | Delta | Reason |
|-------|----------|----------|-------|--------|
| Pre-0 | - | 30 min | +30 min | Constants module, naming consolidation, openevals dep |
| 0 | 30 min | 1.5 hrs | +1 hr | Prompt revisions + code fixes + Hub cleanup |
| 1 | 1 hr | 1 hr | - | commit_tags simplifies |
| 2 | 6 hrs | 8 hrs | +2 hrs | Idempotent migration, PE prompts, extend intent evaluator, golden dataset mgmt (2e) |
| 3 | 2 hrs | 3 hrs | +1 hr | OTel trace ID extraction middleware, LangSmith metadata enrichment, frontend trace capture (3b). Reduced from 3.5 hrs — OTel + Sentry already deployed. |
| 4 | 3 hrs | 3 hrs | - | Unchanged |
| 4.5 | - | 2.5 hrs | +2.5 hrs | **NEW**: Day 2 operator workflows (A/B testing, ad-hoc eval, runbook) |
| 5 | 3 hrs | 4 hrs | +1 hr | Queue ID resolution, API fixes |
| 6 | 3 hrs | 3.5 hrs | +0.5 hr | Cost attribution dashboard, Sentry alert rules. Reduced from 4 hrs — Sentry SDK + OTel already deployed, cross-system correlation solved in Phase 3. |
| **Total** | **22 hrs** | **~27 hrs** | **+5 hrs (+23%)** | Review findings + Day 2 readiness. OTel/Sentry pre-deployment saved ~1 hr. |

---

## Top 6 Execution Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | SSE streaming broken by middleware | Medium | Critical | Raw ASGI middleware (B6). End-to-end test after Phase 3. |
| 2 | Dataset migration partial failure | Medium | High | Idempotent script with dry-run (B4). Staging test first. |
| 3 | LangGraph metadata propagation failure | Low-Medium | Medium | Integration test before full Phase 3 (W4). Fallback: per-handler metadata. |
| 4 | SYNTHESIS JSON parse change breaks eval | Medium | Medium | Phase 0 atomic commit updates both prompt + eval scripts together. |
| 5 | Day 2 workflows not exercised before Phase 5 | Medium | Medium | Phase 4.5 includes acceptance criteria that require end-to-end execution of A/B testing, ad-hoc trace eval, and golden dataset sync. Block Phase 5 start until all 4.5 criteria pass. |
| 6 | Online eval costs exceed budget | Low | Low | Monitor first month. Adjust sampling rates. ~$29/mo well within range. |

---

## Document References

| Document | Location | Team |
|----------|----------|------|
| Master Implementation Plan (superseded by this doc) | `docs/plans/master-implementation-plan.md` | Planning |
| LangSmith Platform Research | `docs/plans/langsmith-platform-research.md` | Planning |
| 3-Vector Evaluation Architecture | `docs/plans/3-vector-eval-architecture.md` | Planning |
| Prompt Management Design | `docs/plans/prompt-management-design.md` | Planning |
| Observability Architecture | `docs/plans/observability-architecture.md` | Planning |
| Codebase Impact Review | `docs/plans/review-codebase-impact.md` | Plan Review |
| Platform Validation Review | `docs/plans/review-platform-validation.md` | Plan Review |
| Integration Risk Review | `docs/plans/review-integration-risks.md` | Plan Review |
| Senior Architect Review | `docs/plans/review-senior-architect.md` | Plan Review |
| Prompt Audit & Standards | `docs/plans/prompt-audit-and-standards.md` | Prompt Engineering |
| Explanatory Vector Prompts | `docs/plans/prompts-explanatory-vector.md` | Prompt Engineering |
| Structured Vector Prompts | `docs/plans/prompts-structured-vector.md` | Prompt Engineering |
| Conversational Vector Prompts | `docs/plans/prompts-conversational-vector.md` | Prompt Engineering |
| Neo4j Schema Validation | `docs/plans/neo4j-schema-and-cypher-validation.md` | Prompt Engineering |
| Pipeline Validation Report | `docs/plans/prompt-pipeline-validation.md` | Prompt Engineering |
