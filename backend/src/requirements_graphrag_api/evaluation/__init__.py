"""Evaluation framework for GraphRAG pipeline.

Provides:
- Hub-First golden dataset (LangSmith source of truth, local fallback)
- LLM-as-judge evaluators via LangSmith (RAGAS-style)
- Cost tracking and analysis

Usage:
    from requirements_graphrag_api.evaluation import (
        CostTracker,
        get_global_cost_tracker,
    )

    from requirements_graphrag_api.evaluation.golden_dataset import (
        DATASET_NAME,
        get_golden_examples,
    )
"""

from __future__ import annotations

from requirements_graphrag_api.evaluation.cost_analysis import (
    CostReport,
    CostTracker,
    LLMCall,
    estimate_cost,
    get_cost_report,
    get_global_cost_tracker,
    reset_global_cost_tracker,
)
from requirements_graphrag_api.evaluation.golden_dataset import (
    DATASET_NAME,
    GOLDEN_EXAMPLES,
    GoldenExample,
    get_golden_examples,
    get_must_pass_examples,
)
from requirements_graphrag_api.evaluation.ragas_evaluators import (
    _get_judge_llm,
    _parse_llm_score,
)

__all__ = [
    "DATASET_NAME",
    "GOLDEN_EXAMPLES",
    "CostReport",
    "CostTracker",
    "GoldenExample",
    "LLMCall",
    "_get_judge_llm",
    "_parse_llm_score",
    "estimate_cost",
    "get_cost_report",
    "get_global_cost_tracker",
    "get_golden_examples",
    "get_must_pass_examples",
    "reset_global_cost_tracker",
]
