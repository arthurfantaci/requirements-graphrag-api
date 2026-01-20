"""Pydantic schemas for benchmark examples.

Defines the data structures for evaluation examples with
full type safety and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryCategory(str, Enum):
    """Categories of benchmark queries."""

    DEFINITIONAL = "definitional"  # What is X?
    RELATIONAL = "relational"  # How does X relate to Y?
    PROCEDURAL = "procedural"  # How do I do X?
    COMPARISON = "comparison"  # X vs Y
    FACTUAL = "factual"  # Specific facts (counts, names, dates)
    ANALYTICAL = "analytical"  # Why/analysis questions
    EDGE_CASE = "edge_case"  # Ambiguous, out-of-domain, adversarial


class DifficultyLevel(str, Enum):
    """Difficulty levels for benchmark examples."""

    EASY = "easy"  # Direct lookup, single concept
    MEDIUM = "medium"  # Multiple concepts, some reasoning
    HARD = "hard"  # Complex relationships, multi-hop
    EXPERT = "expert"  # Domain expertise required, edge cases


class ExpectedRouting(str, Enum):
    """Expected tool routing decisions."""

    VECTOR_SEARCH = "graphrag_vector_search"
    HYBRID_SEARCH = "graphrag_hybrid_search"
    GRAPH_ENRICHED = "graphrag_graph_enriched_search"
    EXPLORE_ENTITY = "graphrag_explore_entity"
    LOOKUP_TERM = "graphrag_lookup_term"
    LOOKUP_STANDARD = "graphrag_lookup_standard"
    SEARCH_STANDARDS = "graphrag_search_standards"
    STANDARDS_BY_INDUSTRY = "graphrag_standards_by_industry"
    TEXT2CYPHER = "graphrag_text2cypher"
    CHAT = "graphrag_chat"


@dataclass
class ExpectedMetrics:
    """Expected metric ranges for evaluation.

    Attributes:
        min_faithfulness: Minimum acceptable faithfulness score.
        min_relevancy: Minimum acceptable answer relevancy score.
        min_precision: Minimum acceptable context precision.
        min_recall: Minimum acceptable context recall.
    """

    min_faithfulness: float = 0.7
    min_relevancy: float = 0.7
    min_precision: float = 0.6
    min_recall: float = 0.6


@dataclass
class BenchmarkExample:
    """A single benchmark evaluation example.

    Attributes:
        id: Unique identifier for the example.
        question: The user query to evaluate.
        category: Query category for grouping.
        difficulty: Difficulty level.
        expected_tools: Expected tool routing decisions.
        ground_truth: Expected answer or key points.
        expected_entities: Entities that should be mentioned.
        expected_standards: Standards that should be referenced.
        expected_metrics: Target metric thresholds.
        tags: Additional tags for filtering.
        metadata: Additional metadata.
    """

    id: str
    question: str
    category: QueryCategory
    difficulty: DifficultyLevel
    expected_tools: list[ExpectedRouting]
    ground_truth: str
    expected_entities: list[str] = field(default_factory=list)
    expected_standards: list[str] = field(default_factory=list)
    expected_metrics: ExpectedMetrics = field(default_factory=ExpectedMetrics)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "question": self.question,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "expected_tools": [t.value for t in self.expected_tools],
            "ground_truth": self.ground_truth,
            "expected_entities": self.expected_entities,
            "expected_standards": self.expected_standards,
            "expected_metrics": {
                "min_faithfulness": self.expected_metrics.min_faithfulness,
                "min_relevancy": self.expected_metrics.min_relevancy,
                "min_precision": self.expected_metrics.min_precision,
                "min_recall": self.expected_metrics.min_recall,
            },
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkExample:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            question=data["question"],
            category=QueryCategory(data["category"]),
            difficulty=DifficultyLevel(data["difficulty"]),
            expected_tools=[ExpectedRouting(t) for t in data["expected_tools"]],
            ground_truth=data["ground_truth"],
            expected_entities=data.get("expected_entities", []),
            expected_standards=data.get("expected_standards", []),
            expected_metrics=ExpectedMetrics(**data.get("expected_metrics", {})),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


__all__ = [
    "BenchmarkExample",
    "DifficultyLevel",
    "ExpectedMetrics",
    "ExpectedRouting",
    "QueryCategory",
]
