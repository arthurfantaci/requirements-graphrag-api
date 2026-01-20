"""Evaluation module for GraphRAG quality assessment.

Provides RAGAS-based evaluation metrics with LangSmith integration
for measuring retrieval and generation quality, plus domain-specific
metrics for requirements management content.

Usage:
    from jama_mcp_server_graphrag.evaluation import (
        evaluate_rag_pipeline,
        create_evaluation_dataset,
        RAGEvaluator,
        DomainMetrics,
        compute_all_domain_metrics,
    )

    # Create evaluator
    evaluator = RAGEvaluator(config)

    # Run evaluation
    results = await evaluator.evaluate(dataset)

    # Compute domain-specific metrics
    domain_metrics = await compute_all_domain_metrics(
        config, question, answer, expected_standards
    )
"""

from jama_mcp_server_graphrag.evaluation.datasets import (
    EvaluationSample,
    create_evaluation_dataset,
    get_sample_evaluation_data,
)
from jama_mcp_server_graphrag.evaluation.domain_metrics import (
    DOMAIN_TERMS,
    KNOWN_STANDARDS,
    DomainMetrics,
    compute_all_domain_metrics,
    compute_citation_accuracy,
    compute_completeness_score,
    compute_regulatory_alignment,
    compute_technical_precision,
    compute_traceability_coverage,
    extract_domain_terms_from_text,
    extract_standards_from_text,
)
from jama_mcp_server_graphrag.evaluation.metrics import (
    RAGMetrics,
    compute_all_metrics,
    compute_answer_relevancy,
    compute_context_precision,
    compute_context_recall,
    compute_faithfulness,
)
from jama_mcp_server_graphrag.evaluation.runner import (
    EvaluationReport,
    EvaluationResult,
    RAGEvaluator,
    evaluate_rag_pipeline,
)

__all__ = [
    "DOMAIN_TERMS",
    "KNOWN_STANDARDS",
    "DomainMetrics",
    "EvaluationReport",
    "EvaluationResult",
    "EvaluationSample",
    "RAGEvaluator",
    "RAGMetrics",
    "compute_all_domain_metrics",
    "compute_all_metrics",
    "compute_answer_relevancy",
    "compute_citation_accuracy",
    "compute_completeness_score",
    "compute_context_precision",
    "compute_context_recall",
    "compute_faithfulness",
    "compute_regulatory_alignment",
    "compute_technical_precision",
    "compute_traceability_coverage",
    "create_evaluation_dataset",
    "evaluate_rag_pipeline",
    "extract_domain_terms_from_text",
    "extract_standards_from_text",
    "get_sample_evaluation_data",
]
