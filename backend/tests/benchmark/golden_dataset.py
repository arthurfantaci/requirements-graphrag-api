"""Golden dataset re-export from the evaluation module.

The canonical golden dataset lives in:
    requirements_graphrag_api.evaluation.golden_dataset

This module re-exports for backward compatibility with existing imports.
"""

from requirements_graphrag_api.evaluation.golden_dataset import (
    GOLDEN_EXAMPLES,
    GoldenExample,
    get_examples_by_category,
    get_must_pass_examples,
)

# Re-export the tuple as GOLDEN_DATASET for backward compat
GOLDEN_DATASET = GOLDEN_EXAMPLES

__all__ = [
    "GOLDEN_DATASET",
    "GOLDEN_EXAMPLES",
    "GoldenExample",
    "get_examples_by_category",
    "get_must_pass_examples",
]
