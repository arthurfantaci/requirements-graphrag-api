"""Evaluation framework for GraphRAG pipeline.

Provides:
- Hub-First golden dataset (LangSmith source of truth, local fallback)
- Per-vector evaluators: explanatory (RAGAS), structured (Cypher), conversational
- Regression gate with per-vector thresholds
- Cost tracking and analysis

Usage:
    from requirements_graphrag_api.evaluation import (
        CostTracker,
        get_global_cost_tracker,
    )

    from requirements_graphrag_api.evaluation.golden_dataset import (
        DATASET_NAME,
        get_golden_examples,
        get_examples_by_vector,
    )

    from requirements_graphrag_api.evaluation.constants import (
        DATASET_EXPLANATORY,
        DATASET_STRUCTURED,
        VALID_LABELS,
    )
"""

from __future__ import annotations

from requirements_graphrag_api.evaluation.constants import (
    DATASET_CONVERSATIONAL,
    DATASET_EXPLANATORY,
    DATASET_INTENT,
    DATASET_LEGACY,
    DATASET_STRUCTURED,
    JUDGE_MODEL,
    QUEUE_CONVERSATIONAL,
    QUEUE_EXPLANATORY,
    QUEUE_INTENT_MAP,
    QUEUE_STRUCTURED,
    QUEUE_USER_REPORTED,
    REGRESSION_THRESHOLDS,
    VALID_LABELS,
    VALID_RELATIONSHIPS,
    experiment_name,
)
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
    CONVERSATIONAL_EXAMPLES,
    DATASET_NAME,
    GOLDEN_EXAMPLES,
    ConversationalExample,
    GoldenExample,
    get_examples_by_vector,
    get_golden_examples,
    get_must_pass_examples,
)
from requirements_graphrag_api.evaluation.ragas_evaluators import (
    _get_judge_llm,
    _parse_llm_score,
)
from requirements_graphrag_api.evaluation.regression import (
    MetricResult,
    RegressionReport,
    check_all_vectors,
    check_regression,
)

__all__ = [
    "CONVERSATIONAL_EXAMPLES",
    "DATASET_CONVERSATIONAL",
    "DATASET_EXPLANATORY",
    "DATASET_INTENT",
    "DATASET_LEGACY",
    "DATASET_NAME",
    "DATASET_STRUCTURED",
    "GOLDEN_EXAMPLES",
    "JUDGE_MODEL",
    "QUEUE_CONVERSATIONAL",
    "QUEUE_EXPLANATORY",
    "QUEUE_INTENT_MAP",
    "QUEUE_STRUCTURED",
    "QUEUE_USER_REPORTED",
    "REGRESSION_THRESHOLDS",
    "VALID_LABELS",
    "VALID_RELATIONSHIPS",
    "ConversationalExample",
    "CostReport",
    "CostTracker",
    "GoldenExample",
    "LLMCall",
    "MetricResult",
    "RegressionReport",
    "_get_judge_llm",
    "_parse_llm_score",
    "check_all_vectors",
    "check_regression",
    "estimate_cost",
    "experiment_name",
    "get_cost_report",
    "get_examples_by_vector",
    "get_global_cost_tracker",
    "get_golden_examples",
    "get_must_pass_examples",
    "reset_global_cost_tracker",
]
