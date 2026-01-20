# Evaluation & Quality Assurance Implementation Handoff

## Overview

This document tracks the implementation of **Evaluation & Quality Assurance** competencies for the jama-mcp-server-graphrag project.

## Phase Status

| Phase | Description | Status | PR |
|-------|-------------|--------|-----|
| **Phase 1** | Benchmark Suite & Datasets | 🟢 Complete | Pending |
| **Phase 2** | Custom Domain Metrics | ⬜ Not Started | — |
| **Phase 3** | CI/CD Integration | ⬜ Not Started | — |
| **Phase 4** | MLflow Comparison | ⬜ Not Started | — |
| **Phase 5** | Cost/Token Tracking | ⬜ Not Started | — |
| **Phase 6** | Human Feedback Loop | ⬜ Not Started | — |

---

## Phase 1: Benchmark Suite (Complete)

### Files Created

```
tests/benchmark/
├── __init__.py                    ✅ Package exports
├── schemas.py                     ✅ Pydantic models (BenchmarkExample, etc.)
├── templates.py                   ✅ Query templates + domain concepts
├── generator.py                   ✅ Programmatic dataset generation
├── golden_dataset.py              ✅ 30 hand-curated critical examples
├── conftest.py                    ✅ Pytest fixtures
├── test_retrieval_accuracy.py     ✅ Retrieval quality tests
├── test_answer_quality.py         ✅ Answer generation tests
├── test_agentic_routing.py        ✅ Router decision tests
└── test_latency_performance.py    ✅ Performance benchmarks

scripts/
└── generate_benchmark_dataset.py  ✅ CLI for dataset generation
```

### Key Components

#### Schemas (`schemas.py`)
- `QueryCategory`: DEFINITIONAL, RELATIONAL, PROCEDURAL, COMPARISON, FACTUAL, ANALYTICAL, EDGE_CASE
- `DifficultyLevel`: EASY, MEDIUM, HARD, EXPERT
- `ExpectedRouting`: All MCP tools mapped
- `BenchmarkExample`: Complete example with metadata

#### Golden Dataset (`golden_dataset.py`)
- 30 hand-curated examples
- Covers all query categories
- Includes critical edge cases (out-of-domain, typos, multi-part)
- All tagged as `must-pass` for CI

#### Generator (`generator.py`)
- Programmatic generation from templates
- Combines 8 template sets × 25+ domain concepts
- Produces 250+ diverse examples
- Deterministic with seed parameter

### Verification Commands

```bash
# Run benchmark tests
uv run pytest tests/benchmark/ -v

# Generate dataset statistics
uv run python scripts/generate_benchmark_dataset.py --stats-only

# Generate and save dataset
uv run python scripts/generate_benchmark_dataset.py --output benchmark_data.json

# Generate LangSmith-compatible format
uv run python scripts/generate_benchmark_dataset.py --langsmith-format --output langsmith_eval.json

# View golden dataset only
uv run python scripts/generate_benchmark_dataset.py --golden-only
```

### Test Coverage

| Test File | Test Count | Description |
|-----------|------------|-------------|
| `test_retrieval_accuracy.py` | ~20 | Precision@K, Recall@K, MRR |
| `test_answer_quality.py` | ~20 | Faithfulness, relevancy checks |
| `test_agentic_routing.py` | ~15 | Tool selection accuracy |
| `test_latency_performance.py` | ~15 | Latency thresholds, throughput |

---

## Phase 2: Custom Domain Metrics (Planned)

### Deliverables

```
src/jama_mcp_server_graphrag/evaluation/
└── domain_metrics.py    # NEW
```

### Metrics to Implement

| Metric | Description | Formula |
|--------|-------------|---------|
| Citation Accuracy | Standards correctly cited | correct_citations / total_citations |
| Traceability Coverage | Links mentioned when relevant | traced_refs / expected_refs |
| Technical Precision | Domain terms used correctly | correct_terms / total_terms |
| Completeness Score | All aspects of query addressed | aspects_covered / aspects_asked |
| Regulatory Alignment | ISO/ASPICE/FDA refs accurate | aligned_refs / regulatory_refs |

---

## Phase 3: CI/CD Integration (Planned)

### CI Tiers

| Tier | Trigger | Scope | Time | Cost |
|------|---------|-------|------|------|
| 1 | Every PR | Unit tests, prompt validation | ~1 min | $0 |
| 2 | Merge to main | Smoke eval (10 queries) | ~5 min | ~$0.50 |
| 3 | Release tag | Full benchmark (250 queries) | ~20 min | ~$15 |
| 4 | Nightly | Deep eval + A/B tests | ~45 min | ~$20 |

### Files to Create/Modify

```
.github/workflows/
├── ci.yml               # MODIFY: Add Tier 1-2
└── evaluation.yml       # NEW: Tier 3-4

scripts/
└── ci_evaluation.py     # NEW: CI-friendly runner
```

---

## Phase 4: MLflow Comparison (Planned)

### Deliverables

```
src/jama_mcp_server_graphrag/
├── mlflow_tracking.py           # NEW
└── observability_comparison.py  # NEW

docs/
└── PLATFORM_COMPARISON.md       # NEW

scripts/
└── compare_platforms.py         # NEW
```

### Comparison Dimensions

1. Setup complexity
2. Evaluation features
3. Visualization
4. Prompt versioning
5. Self-hosting options

---

## Phase 5: Cost/Token Tracking (Planned)

### Cost Thresholds

```python
COST_THRESHOLDS = {
    "query_budget_target": 0.015,
    "query_budget_warning": 0.025,
    "query_budget_alert": 0.040,
    "query_budget_hard_limit": 0.100,
}
```

### Deliverables

```
src/jama_mcp_server_graphrag/evaluation/
└── cost_metrics.py      # NEW

src/jama_mcp_server_graphrag/
└── token_counter.py     # NEW
```

---

## Phase 6: Human Feedback Loop (Planned)

### Workflow

1. Export low-confidence runs to annotation queues
2. Human reviewers annotate
3. Import annotations as evaluation examples
4. Re-run evaluations

### Deliverables

```
docs/
└── FEEDBACK_WORKFLOW.md         # NEW

scripts/
├── export_for_annotation.py     # NEW
├── import_feedback.py           # NEW
└── update_datasets.py           # NEW
```

---

## Quick Reference

### Run All Benchmark Tests
```bash
uv run pytest tests/benchmark/ -v
```

### Run Specific Category
```bash
uv run pytest tests/benchmark/test_retrieval_accuracy.py -v
```

### Generate Full Dataset
```bash
uv run python scripts/generate_benchmark_dataset.py \
    --count 250 \
    --include-golden \
    --langsmith-format \
    --output eval_dataset.json
```

### Check Dataset Statistics
```bash
uv run python scripts/generate_benchmark_dataset.py --stats-only
```

---

## Dependencies Added

None - all dependencies already in pyproject.toml:
- pytest
- pytest-asyncio
- pytest-benchmark (optional, for performance tracking)

---

## Next Steps

1. **Verify Phase 1**: Run tests, check for issues
2. **Create PR**: Commit Phase 1 changes
3. **Start Phase 2**: Implement domain metrics
4. **Iterate**: Complete remaining phases

---

## Contact

This handoff document enables continuation in a new Claude session. Reference this file along with the project's SPECIFICATION.md and CLAUDE.md for full context.
