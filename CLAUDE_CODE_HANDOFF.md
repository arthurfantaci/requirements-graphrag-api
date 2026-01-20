# Claude Code Handoff: Evaluation & Quality Assurance

## Quick Start for Claude Code

```bash
# Current state
git branch  # Should show: feature/phase2-domain-metrics
git log --oneline -3  # Verify Phase 1 merged to main
```

## Project Context

**Project:** jama-mcp-server-graphrag  
**Goal:** Production-ready GraphRAG MCP Server for requirements management  
**Current Focus:** Evaluation & Quality Assurance competencies (Phases 2-6)

## Completed Work

### Phase 1: Benchmark Suite ✅ MERGED (PR #3)

```
tests/benchmark/
├── __init__.py           # Package exports
├── schemas.py            # BenchmarkExample, QueryCategory, DifficultyLevel, ExpectedRouting
├── templates.py          # Query patterns + 25 domain concepts
├── generator.py          # Programmatic dataset generation (250+ examples)
├── golden_dataset.py     # 30 hand-curated critical examples
├── conftest.py           # Pytest fixtures, MetricAssertions helper
├── test_retrieval_accuracy.py   # Precision@K, Recall@K, MRR (20 tests)
├── test_answer_quality.py       # Faithfulness, relevancy (20 tests)
├── test_agentic_routing.py      # Tool selection accuracy (15 tests)
└── test_latency_performance.py  # Latency thresholds (15 tests)

scripts/
└── generate_benchmark_dataset.py  # CLI for dataset generation
```

**Key Commands:**
```bash
uv run pytest tests/benchmark/ -v              # Run all 409 benchmark tests
uv run python scripts/generate_benchmark_dataset.py --stats-only  # View dataset stats
```

---

## Phase 2: Custom Domain Metrics (CURRENT)

### Objective
Implement 5 domain-specific evaluation metrics for requirements management content.

### Deliverables

```
src/jama_mcp_server_graphrag/evaluation/
├── metrics.py           # EXISTS - RAGAS metrics
└── domain_metrics.py    # NEW - Domain-specific metrics
```

### Metrics to Implement

| Metric | Description | Formula | Use Case |
|--------|-------------|---------|----------|
| **Citation Accuracy** | Standards correctly cited | `correct_citations / total_citations` | Verify ISO 26262, IEC 62304 refs |
| **Traceability Coverage** | Links mentioned when relevant | `traced_refs / expected_refs` | Ensure traceability concepts included |
| **Technical Precision** | Domain terms used correctly | `correct_terms / total_terms` | Validate ASIL, FMEA, V-model usage |
| **Completeness Score** | All aspects of query addressed | `aspects_covered / aspects_asked` | Multi-part question coverage |
| **Regulatory Alignment** | Standard refs are accurate | `aligned_refs / regulatory_refs` | FDA, ISO compliance accuracy |

### Implementation Pattern

Follow existing `metrics.py` pattern:
```python
# src/jama_mcp_server_graphrag/evaluation/domain_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jama_mcp_server_graphrag.config import AppConfig

@dataclass(frozen=True)
class DomainMetrics:
    """Domain-specific evaluation metrics."""
    citation_accuracy: float
    traceability_coverage: float
    technical_precision: float
    completeness_score: float
    regulatory_alignment: float

async def compute_citation_accuracy(
    config: AppConfig,
    answer: str,
    expected_standards: list[str],
) -> float:
    """Compute citation accuracy for standards references."""
    ...
```

### Tests to Add

```
tests/test_evaluation/
└── test_domain_metrics.py  # NEW - Tests for domain metrics
```

---

## Phase 3: CI/CD Evaluation Integration

### Objective
Add tiered evaluation to CI/CD pipeline.

### CI Tiers

| Tier | Trigger | Scope | Time | Cost |
|------|---------|-------|------|------|
| 1 | Every PR | Unit tests, prompt validation | ~1 min | $0 |
| 2 | Merge to main | Smoke eval (10 queries) | ~5 min | ~$0.50 |
| 3 | Release tag (v*) | Full benchmark (250 queries) | ~20 min | ~$15 |
| 4 | Nightly | Deep eval + A/B tests | ~45 min | ~$20 |

### Deliverables

```
.github/workflows/
├── ci.yml              # MODIFY - Add Tier 1-2
└── evaluation.yml      # NEW - Tier 3-4

scripts/
└── ci_evaluation.py    # NEW - CI-friendly evaluation runner
```

---

## Phase 4: MLflow Comparison

### Objective
Compare LangSmith vs MLflow across 5 dimensions.

### Deliverables

```
src/jama_mcp_server_graphrag/
├── mlflow_tracking.py           # NEW - MLflow integration
└── observability_comparison.py  # NEW - Side-by-side comparison

docs/
└── PLATFORM_COMPARISON.md       # NEW - Detailed comparison report

scripts/
└── compare_platforms.py         # NEW - Run comparison experiments
```

### Comparison Dimensions
1. Setup complexity
2. Evaluation features
3. Visualization capabilities
4. Prompt versioning
5. Self-hosting options

---

## Phase 5: Cost & Token Tracking

### Objective
Track and budget token usage per query.

### Cost Thresholds (from planning)

```python
COST_THRESHOLDS = {
    "query_budget_target": 0.015,      # Target per query
    "query_budget_warning": 0.025,     # Warning threshold
    "query_budget_alert": 0.040,       # Alert threshold
    "query_budget_hard_limit": 0.100,  # Hard limit
    "smoke_test_budget": 0.50,         # CI Tier 2
    "benchmark_budget": 5.00,          # CI Tier 3
    "full_eval_budget": 15.00,         # CI Tier 4
}
```

### Deliverables

```
src/jama_mcp_server_graphrag/
├── token_counter.py              # NEW - Token counting utilities
└── evaluation/
    └── cost_metrics.py           # NEW - Cost tracking metrics
```

---

## Phase 6: Human Feedback Loop

### Objective
Enable human annotation workflow for continuous improvement.

### Workflow
1. Export low-confidence runs to LangSmith annotation queues
2. Human reviewers annotate (correct/incorrect, improvements)
3. Import annotations as new evaluation examples
4. Re-run evaluations to measure improvement

### Deliverables

```
scripts/
├── export_for_annotation.py     # NEW - Export to annotation queue
├── import_feedback.py           # NEW - Import human annotations
└── update_datasets.py           # NEW - Update golden dataset

docs/
└── FEEDBACK_WORKFLOW.md         # NEW - Process documentation
```

---

## Key Files to Reference

| File | Purpose |
|------|---------|
| `SPECIFICATION.md` | Full project specification |
| `CLAUDE.md` | Claude-specific instructions |
| `EVALUATION_HANDOFF.md` | Phase tracking (update as you go) |
| `src/jama_mcp_server_graphrag/evaluation/metrics.py` | Existing RAGAS metrics pattern |
| `tests/benchmark/schemas.py` | BenchmarkExample schema |
| `tests/benchmark/golden_dataset.py` | 30 curated examples with expected_standards |

---

## Coding Standards Reminder

1. **Type Hints**: `from __future__ import annotations` + full typing
2. **Docstrings**: Google-style with Args, Returns, Raises
3. **Structured Logging**: Module-level loggers with `structlog`
4. **Error Handling**: Custom exceptions with chaining
5. **Testing**: pytest fixtures, mocks, ≥80% coverage target
6. **Linting**: `uv run ruff check .` before commits

---

## Git Workflow

```bash
# Current branch
git checkout feature/phase2-domain-metrics

# Conventional commits
git commit -m "feat(eval): add citation accuracy metric"
git commit -m "test(eval): add domain metrics unit tests"

# Push and create PR
git push -u origin feature/phase2-domain-metrics
gh pr create --title "feat(eval): Add custom domain metrics" --body "..."

# After merge
git checkout main && git pull && git branch -d feature/phase2-domain-metrics
```

---

## Ready to Start Phase 2

Begin with:
1. Read `src/jama_mcp_server_graphrag/evaluation/metrics.py` for pattern
2. Create `src/jama_mcp_server_graphrag/evaluation/domain_metrics.py`
3. Create `tests/test_evaluation/test_domain_metrics.py`
4. Run tests: `uv run pytest tests/test_evaluation/test_domain_metrics.py -v`
