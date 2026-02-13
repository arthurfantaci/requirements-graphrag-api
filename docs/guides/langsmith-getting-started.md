# LangSmith Getting Started — Quick Reference

> Quick reference card for the GraphRAG LangSmith integration.
> For detailed setup, see [langsmith-ui-setup.md](langsmith-ui-setup.md).
> For use cases and workflows, see [langsmith-user-manual.md](langsmith-user-manual.md).

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| LangSmith account | Admin access to org **Norfolk AI\|BI**, workspace **graphrag-api** |
| LangSmith project | `graphrag-api-prod` (within graphrag-api workspace) |
| Environment variables | `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT=graphrag-api-prod` |
| Neo4j (for Cypher eval only) | `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` |
| Python environment | `uv` installed, backend dependencies available |

### LangSmith Hierarchy

```
Norfolk AI|BI (Org: 91793c67-6068-4615-8a2f-7de948bc8f68)
  └── graphrag-api (Workspace: 33c08fbf-3b88-4b49-ae32-9677043ebed2)
        ├── Tracing Projects
        │     ├── graphrag-api-prod   ← production traces
        │     ├── graphrag-api-dev    ← development traces
        │     └── evaluators          ← evaluation experiment runs
        ├── Datasets & Experiments    ← workspace-level (shared)
        ├── Annotation Queues         ← workspace-level (shared)
        ├── Prompts (19)              ← workspace-level (shared)
        ├── Monitoring                ← workspace-level
        └── Custom Dashboards         ← workspace-level
```

### Environment Setup

```bash
# Required for all LangSmith features
export LANGSMITH_API_KEY=lsv2_pt_...
export LANGSMITH_TRACING=true

# Project selection (determines which tracing project receives traces)
export LANGSMITH_PROJECT=graphrag-api-prod   # production
# export LANGSMITH_PROJECT=graphrag-api-dev  # development (default in config.py)

# Required for online Cypher evaluation and offline eval
export NEO4J_URI=neo4j+s://...
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=...

# Prompt environment (controls which Hub tag is pulled)
export PROMPT_ENVIRONMENT=production   # or "staging"
```

---

## Architecture Overview

```
Norfolk AI|BI org → graphrag-api workspace
┌─────────────────────────────────────────────────────────────────┐
│  PRODUCTION TRAFFIC → project: graphrag-api-prod                │
│                                                                 │
│  User → Chat API → Intent Router → Handler (3 vectors)         │
│           │                            │                        │
│           │  X-Trace-ID               │  metadata: intent,     │
│           │  correlation              │  correlation_id,       │
│           │                            │  session_id            │
│           ▼                            ▼                        │
│      ┌─────────────────────────────────────┐                    │
│      │    LangSmith Traces (graphrag-api-  │                    │
│      │    prod, enriched with metadata)    │                    │
│      └────┬───────────┬───────────┬────────┘                    │
│           │           │           │                              │
│           ▼           ▼           ▼                              │
│    ┌──────────┐ ┌──────────┐ ┌──────────┐                       │
│    │ Online   │ │ Online   │ │ Online   │   ◄── UI-configured  │
│    │ Halluci- │ │ Coherence│ │ Cypher   │       evaluators     │
│    │ nation   │ │ (15%)    │ │ Parse    │                       │
│    │ (10%)    │ │          │ │ (100%)   │                       │
│    └────┬─────┘ └────┬─────┘ └────┬─────┘                       │
│         │            │            │                              │
│         ▼            ▼            ▼                              │
│    ┌──────────────────────────────────┐                          │
│    │      Automation Rules            │  ◄── Score threshold    │
│    │  (route low scores to queues)    │      triggers routing   │
│    └────┬────────┬────────┬───────────┘                          │
│         │        │        │                                      │
│         ▼        ▼        ▼                                      │
│    ┌────────┐┌────────┐┌────────┐┌──────────────┐               │
│    │review- ││review- ││review- ││user-reported- │              │
│    │explan. ││convers.││struct. ││issues         │              │
│    └────────┘└────────┘└────────┘└──────┬───────┘               │
│                                          │                       │
│                         Frontend ────────┘                       │
│                         feedback.py routes                       │
│                         negative feedback                        │
│                         by intent                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  SCHEDULED / MANUAL (target: graphrag-api-prod)                 │
│                                                                 │
│  online_eval_cypher.py ──► Batch Cypher execution against Neo4j │
│  quality_check.py ───────► 7-day vs 30-day degradation check    │
│  run_vector_evaluation.py ► Full offline eval → evaluators proj │
│  ci_evaluation.py ────────► CI tier runner (1-4)                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Daily Operations — Command Cheat Sheet

### Check production quality

```bash
# Quick degradation check (dry run — no exit code failure)
uv run python scripts/quality_check.py --project graphrag-api-prod --dry-run

# Full degradation check (exits 1 on degradation)
uv run python scripts/quality_check.py --project graphrag-api-prod
```

### Run online Cypher evaluation

```bash
# Evaluate last 24h of structured traces (posts feedback to LangSmith)
uv run python scripts/online_eval_cypher.py --project graphrag-api-prod

# Dry run — evaluate but don't post
uv run python scripts/online_eval_cypher.py --project graphrag-api-prod --dry-run

# Custom window
uv run python scripts/online_eval_cypher.py --project graphrag-api-prod \
    --hours-back 48 --limit 100
```

### Run offline evaluations

```bash
# Single vector
uv run python scripts/run_vector_evaluation.py --vector explanatory
uv run python scripts/run_vector_evaluation.py --vector structured
uv run python scripts/run_vector_evaluation.py --vector conversational
uv run python scripts/run_vector_evaluation.py --vector intent

# All vectors
uv run python scripts/run_vector_evaluation.py --vector all
```

### Run CI evaluations

```bash
# Tier 1: Prompt validation (every PR)
uv run python scripts/ci_evaluation.py --tier 1

# Tier 2: Smoke tests (merge to main)
uv run python scripts/ci_evaluation.py --tier 2

# Tier 3: Full evaluation (releases)
uv run python scripts/ci_evaluation.py --tier 3

# Tier 4: Full + regression comparison (manual)
uv run python scripts/ci_evaluation.py --tier 4
```

### Manage prompts

```bash
# Push all prompts to Hub
uv run python scripts/sync_prompts_to_hub.py

# Dry run (preview changes)
uv run python scripts/sync_prompts_to_hub.py --dry-run

# Push specific prompt
uv run python scripts/sync_prompts_to_hub.py --only synthesis

# Check sync status
uv run python scripts/check_prompt_sync.py
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `evaluation/constants.py` | All magic strings: dataset names, queue names, thresholds, safety patterns |
| `routes/feedback.py` | Feedback endpoint: PII redaction, rubric scores, queue routing |
| `routes/chat.py` | Trace enrichment: intent, correlation_id, session_id metadata |
| `middleware/tracing.py` | X-Trace-ID correlation middleware |
| `scripts/quality_check.py` | Daily 7d vs 30d degradation detection |
| `scripts/online_eval_cypher.py` | Batch Cypher validation against Neo4j |
| `scripts/run_vector_evaluation.py` | Unified offline evaluation runner |
| `scripts/ci_evaluation.py` | CI tier runner with regression gates |
| `scripts/sync_prompts_to_hub.py` | Push prompts to LangSmith Hub |
| `scripts/check_prompt_sync.py` | Verify Hub sync status |
| `FeedbackModal.jsx` | Intent-specific rubric UI (star ratings) |
| `ResponseActions.jsx` | Thumbs up/down + intent threading |
| `AssistantMessage.jsx` | Passes intent to ResponseActions |

---

## 4 Evaluation Datasets

| Dataset | Examples | Evaluators | Intent |
|---------|----------|------------|--------|
| `graphrag-eval-explanatory` | 46 | 8 (faithfulness, relevancy, precision, recall, correctness, entity recall, groundedness, hallucination) | explanatory |
| `graphrag-eval-structured` | 18 | 6 (cypher parse, schema, execution, safety, result shape, result correctness) | structured |
| `graphrag-eval-conversational` | 15 | 4 (coherence, context retention, coreference, hallucination) | conversational |
| `graphrag-eval-intent` | 30 | 1 (intent accuracy) | all |

---

## Regression Thresholds (CI Tier 3+)

| Vector | Hard Gates | Soft Gates |
|--------|-----------|------------|
| Intent | accuracy >= 95% | N/A |
| Explanatory | correctness >= 0.5, faithfulness >= 0.5 | faithfulness drop <= 0.05 |
| Structured | cypher_parse: 100%, cypher_safety: 100% | execution_success >= 90% |
| Conversational | hallucination: 100% | coherence drop <= 0.10 |

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Feedback not reaching queue | Queue name mismatch | Verify names match `constants.py` exactly |
| `_resolve_queue_id` returns None | Queue doesn't exist in LangSmith | Create the queue (Section A of setup guide) |
| Online evaluator not scoring traces | Filter mismatch | Check `metadata.intent` is set on traces |
| No intent on traces | Handler not enriching metadata | Check `routes/handlers.py` metadata enrichment |
| Rubric scores missing | Frontend not sending `rubric_scores` | Check FeedbackModal → ResponseActions intent prop chain |
| quality_check.py shows N/A | Insufficient feedback data | Need ≥5 samples per key in both windows |
