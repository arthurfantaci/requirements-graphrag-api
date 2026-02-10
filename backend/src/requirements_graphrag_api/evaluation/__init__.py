"""Evaluation framework for GraphRAG pipeline.

Provides comprehensive evaluation of the RAG pipeline using:
- Standard RAG metrics (faithfulness, relevancy, precision, recall)
- LLM-as-judge evaluators via LangSmith (RAGAS-style)
- Cost tracking and analysis

Usage:
    from requirements_graphrag_api.evaluation import (
        CostTracker,
        get_global_cost_tracker,
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
from requirements_graphrag_api.evaluation.metrics import (
    ANSWER_RELEVANCY_PROMPT,
    CONTEXT_PRECISION_PROMPT,
    CONTEXT_RECALL_PROMPT,
    FAITHFULNESS_PROMPT,
)
from requirements_graphrag_api.evaluation.ragas_evaluators import (
    _get_judge_llm,
    _parse_llm_score,
)

__all__ = [
    "ANSWER_RELEVANCY_PROMPT",
    "CONTEXT_PRECISION_PROMPT",
    "CONTEXT_RECALL_PROMPT",
    "FAITHFULNESS_PROMPT",
    "CostReport",
    "CostTracker",
    "LLMCall",
    "_get_judge_llm",
    "_parse_llm_score",
    "estimate_cost",
    "get_cost_report",
    "get_global_cost_tracker",
    "reset_global_cost_tracker",
]
