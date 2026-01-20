"""Benchmark test suite for GraphRAG evaluation.

This package provides comprehensive benchmarking for the RAG pipeline:
- Retrieval accuracy metrics (Precision@K, Recall@K, MRR)
- Answer quality evaluation (RAGAS + domain metrics)
- Agentic routing accuracy
- Latency and performance benchmarks

Usage:
    # Run all benchmarks
    uv run pytest tests/benchmark/ -v

    # Run specific benchmark category
    uv run pytest tests/benchmark/test_retrieval_accuracy.py -v

    # Run with performance tracking
    uv run pytest tests/benchmark/ -v --benchmark-only
"""

from tests.benchmark.generator import generate_evaluation_dataset
from tests.benchmark.golden_dataset import GOLDEN_DATASET
from tests.benchmark.schemas import (
    BenchmarkExample,
    DifficultyLevel,
    ExpectedRouting,
    QueryCategory,
)

__all__ = [
    "GOLDEN_DATASET",
    "BenchmarkExample",
    "DifficultyLevel",
    "ExpectedRouting",
    "QueryCategory",
    "generate_evaluation_dataset",
]
