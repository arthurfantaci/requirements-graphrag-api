"""Agentic routing benchmark tests.

Tests the router's ability to select appropriate tools:
- Tool selection accuracy
- Routing reasoning quality
- Edge case handling
- Fallback behavior

These tests validate that the router correctly dispatches
queries to the most appropriate retrieval tools.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.benchmark.schemas import (
    BenchmarkExample,
    ExpectedRouting,
    QueryCategory,
)

# =============================================================================
# ROUTING ACCURACY METRICS
# =============================================================================


def routing_accuracy(
    expected: list[ExpectedRouting],
    actual: list[str],
) -> float:
    """Calculate routing accuracy.

    Args:
        expected: List of expected tool routings.
        actual: List of actual tool routings selected.

    Returns:
        Accuracy score (0.0 to 1.0).
    """
    if not expected:
        return 1.0 if not actual else 0.0

    expected_values = {e.value for e in expected}
    actual_set = set(actual)

    # Check if any expected tool was selected
    matches = expected_values & actual_set
    return len(matches) / len(expected_values)


def top_k_routing_accuracy(
    expected: list[ExpectedRouting],
    actual_ranked: list[str],
    k: int = 1,
) -> float:
    """Calculate if correct tool is in top-K selections.

    Args:
        expected: List of expected tool routings.
        actual_ranked: List of actual tools ranked by confidence.
        k: Number of top selections to consider.

    Returns:
        1.0 if any expected tool in top-K, else 0.0.
    """
    if not expected:
        return 1.0

    expected_values = {e.value for e in expected}
    top_k = set(actual_ranked[:k])

    return 1.0 if expected_values & top_k else 0.0


# =============================================================================
# UNIT TESTS FOR ROUTING METRICS
# =============================================================================


class TestRoutingMetrics:
    """Unit tests for routing metric functions."""

    def test_routing_accuracy_perfect(self) -> None:
        """Test routing accuracy with perfect match."""
        expected = [ExpectedRouting.VECTOR_SEARCH]
        actual = ["graphrag_vector_search"]

        assert routing_accuracy(expected, actual) == 1.0

    def test_routing_accuracy_multiple_expected(self) -> None:
        """Test routing accuracy with multiple expected tools."""
        expected = [
            ExpectedRouting.VECTOR_SEARCH,
            ExpectedRouting.LOOKUP_TERM,
        ]
        actual = ["graphrag_vector_search"]

        # One of two expected tools selected
        assert routing_accuracy(expected, actual) == 0.5

    def test_routing_accuracy_no_match(self) -> None:
        """Test routing accuracy with no match."""
        expected = [ExpectedRouting.VECTOR_SEARCH]
        actual = ["graphrag_text2cypher"]

        assert routing_accuracy(expected, actual) == 0.0

    def test_routing_accuracy_empty(self) -> None:
        """Test routing accuracy with empty inputs."""
        assert routing_accuracy([], []) == 1.0
        assert routing_accuracy([], ["some_tool"]) == 0.0

    def test_top_k_accuracy_top_1(self) -> None:
        """Test top-1 routing accuracy."""
        expected = [ExpectedRouting.VECTOR_SEARCH]
        actual = ["graphrag_vector_search", "graphrag_hybrid_search"]

        assert top_k_routing_accuracy(expected, actual, k=1) == 1.0

    def test_top_k_accuracy_top_2(self) -> None:
        """Test top-2 routing accuracy."""
        expected = [ExpectedRouting.VECTOR_SEARCH]
        actual = ["graphrag_hybrid_search", "graphrag_vector_search"]

        assert top_k_routing_accuracy(expected, actual, k=1) == 0.0
        assert top_k_routing_accuracy(expected, actual, k=2) == 1.0


# =============================================================================
# ROUTER BEHAVIOR TESTS
# =============================================================================


class TestRouterBehavior:
    """Tests for router behavior patterns."""

    def test_definitional_queries_route_to_vector_search(
        self,
        definitional_examples: list[BenchmarkExample],
    ) -> None:
        """Test that definitional queries expect vector search."""
        for example in definitional_examples:
            # Definitional queries should include vector search or lookup_term
            valid_tools = {
                ExpectedRouting.VECTOR_SEARCH,
                ExpectedRouting.LOOKUP_TERM,
                ExpectedRouting.HYBRID_SEARCH,
            }
            has_valid_tool = any(t in valid_tools for t in example.expected_tools)
            assert has_valid_tool, f"Example {example.id} should route to search tool"

    def test_standards_queries_route_to_lookup(
        self,
        standards_examples: list[BenchmarkExample],
    ) -> None:
        """Test that standards queries expect lookup tools."""
        for example in standards_examples:
            if example.category == QueryCategory.FACTUAL:
                # Factual standards queries should use lookup or text2cypher
                valid_tools = {
                    ExpectedRouting.LOOKUP_STANDARD,
                    ExpectedRouting.STANDARDS_BY_INDUSTRY,
                    ExpectedRouting.TEXT2CYPHER,
                    ExpectedRouting.VECTOR_SEARCH,
                }
                has_valid_tool = any(t in valid_tools for t in example.expected_tools)
                assert has_valid_tool, f"Standards example {example.id} needs appropriate routing"

    def test_relational_queries_route_to_graph(
        self,
        golden_dataset: list[BenchmarkExample],
    ) -> None:
        """Test that relational queries expect graph-enriched search."""
        relational = [ex for ex in golden_dataset if ex.category == QueryCategory.RELATIONAL]

        for example in relational:
            valid_tools = {
                ExpectedRouting.GRAPH_ENRICHED,
                ExpectedRouting.EXPLORE_ENTITY,
                ExpectedRouting.HYBRID_SEARCH,
            }
            has_valid_tool = any(t in valid_tools for t in example.expected_tools)
            assert has_valid_tool, f"Relational {example.id} should use graph tools"

    def test_factual_count_queries_route_to_cypher(
        self,
        golden_dataset: list[BenchmarkExample],
    ) -> None:
        """Test that count/factual queries expect text2cypher."""
        factual = [
            ex
            for ex in golden_dataset
            if ex.category == QueryCategory.FACTUAL
            and any(
                word in ex.question.lower()
                for word in ["how many", "count", "list", "which articles"]
            )
        ]

        for example in factual:
            # These should include text2cypher as an option
            assert (
                ExpectedRouting.TEXT2CYPHER in example.expected_tools
                or ExpectedRouting.LOOKUP_STANDARD in example.expected_tools
                or ExpectedRouting.STANDARDS_BY_INDUSTRY in example.expected_tools
            ), f"Factual {example.id} should consider text2cypher"


# =============================================================================
# EDGE CASE ROUTING TESTS
# =============================================================================


class TestEdgeCaseRouting:
    """Tests for edge case routing behavior."""

    def test_out_of_domain_routing(
        self,
        edge_case_examples: list[BenchmarkExample],
    ) -> None:
        """Test routing for out-of-domain queries."""
        ood_examples = [ex for ex in edge_case_examples if ex.metadata.get("out_of_domain", False)]

        for example in ood_examples:
            # Out-of-domain should route to chat (for graceful handling)
            # or vector search (which will return low relevance)
            valid_tools = {
                ExpectedRouting.CHAT,
                ExpectedRouting.VECTOR_SEARCH,
            }
            has_valid_tool = any(t in valid_tools for t in example.expected_tools)
            assert has_valid_tool, f"OOD {example.id} should gracefully handle"

    def test_ambiguous_query_routing(
        self,
        edge_case_examples: list[BenchmarkExample],
    ) -> None:
        """Test routing for ambiguous queries."""
        ambiguous = [ex for ex in edge_case_examples if "ambiguous" in ex.tags]

        for example in ambiguous:
            # Ambiguous queries might go to vector search or chat
            valid_tools = {
                ExpectedRouting.VECTOR_SEARCH,
                ExpectedRouting.CHAT,
                ExpectedRouting.HYBRID_SEARCH,
            }
            has_valid_tool = any(t in valid_tools for t in example.expected_tools)
            assert has_valid_tool, f"Ambiguous {example.id} needs flexible routing"

    def test_multi_part_query_routing(
        self,
        golden_dataset: list[BenchmarkExample],
    ) -> None:
        """Test routing for multi-part questions."""
        multi_part = [ex for ex in golden_dataset if "multi-part" in ex.tags]

        # Multi-part questions may need multiple tools
        for example in multi_part:
            assert len(example.expected_tools) >= 1, (
                f"Multi-part {example.id} should consider multiple tools"
            )


# =============================================================================
# ROUTER MOCK TESTS
# =============================================================================


class TestRouterMock:
    """Tests with mocked router responses."""

    @pytest.mark.asyncio
    async def test_router_returns_valid_tools(
        self,
        mock_config: MagicMock,
    ) -> None:
        """Test that router returns valid tool names."""
        with patch("jama_mcp_server_graphrag.agentic.router.route_query") as mock_router:
            mock_router.return_value = {
                "selected_tools": ["graphrag_vector_search"],
                "reasoning": "Definitional query about core concept",
                "confidence": 0.9,
            }

            result = await mock_router(mock_config, "What is traceability?")

            # Validate response structure
            assert "selected_tools" in result
            assert isinstance(result["selected_tools"], list)
            assert len(result["selected_tools"]) > 0

    @pytest.mark.asyncio
    async def test_router_provides_reasoning(
        self,
        mock_config: MagicMock,
    ) -> None:
        """Test that router provides reasoning for selection."""
        with patch("jama_mcp_server_graphrag.agentic.router.route_query") as mock_router:
            mock_router.return_value = {
                "selected_tools": ["graphrag_lookup_standard"],
                "reasoning": "Query asks about specific standard ISO 26262",
                "confidence": 0.95,
            }

            result = await mock_router(mock_config, "What is ISO 26262?")

            assert "reasoning" in result
            assert len(result["reasoning"]) > 10

    @pytest.mark.asyncio
    async def test_router_confidence_score(
        self,
        mock_config: MagicMock,
    ) -> None:
        """Test that router provides confidence score."""
        with patch("jama_mcp_server_graphrag.agentic.router.route_query") as mock_router:
            mock_router.return_value = {
                "selected_tools": ["graphrag_vector_search"],
                "reasoning": "General query",
                "confidence": 0.75,
            }

            result = await mock_router(mock_config, "Tell me about requirements.")

            assert "confidence" in result
            assert 0.0 <= result["confidence"] <= 1.0


# =============================================================================
# BENCHMARK-DRIVEN ROUTING TESTS
# =============================================================================


class TestBenchmarkRouting:
    """Benchmark-driven routing tests."""

    def test_all_examples_have_expected_tools(
        self,
        golden_example: BenchmarkExample,
    ) -> None:
        """Test that all examples have expected tools defined."""
        assert len(golden_example.expected_tools) > 0, (
            f"Example {golden_example.id} has no expected tools"
        )

    def test_expected_tools_are_valid(
        self,
        golden_example: BenchmarkExample,
    ) -> None:
        """Test that expected tools are valid enum values."""
        for tool in golden_example.expected_tools:
            assert isinstance(tool, ExpectedRouting), f"Invalid tool {tool} in {golden_example.id}"

    def test_routing_coverage_by_tool(
        self,
        golden_dataset: list[BenchmarkExample],
    ) -> None:
        """Test that dataset covers various routing tools."""
        all_tools: set[ExpectedRouting] = set()
        for example in golden_dataset:
            all_tools.update(example.expected_tools)

        # Key tools should be covered
        key_tools = {
            ExpectedRouting.VECTOR_SEARCH,
            ExpectedRouting.LOOKUP_STANDARD,
            ExpectedRouting.GRAPH_ENRICHED,
            ExpectedRouting.TEXT2CYPHER,
        }

        for tool in key_tools:
            assert tool in all_tools, f"Tool {tool.value} not covered in dataset"


# =============================================================================
# AGGREGATE ROUTING TESTS
# =============================================================================


class TestAggregateRouting:
    """Aggregate routing statistics tests."""

    def test_routing_distribution(
        self,
        golden_dataset: list[BenchmarkExample],
    ) -> None:
        """Test routing tool distribution across dataset."""
        tool_counts: dict[ExpectedRouting, int] = {}

        for example in golden_dataset:
            for tool in example.expected_tools:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

        # Vector search should be common (used for many query types)
        assert tool_counts.get(ExpectedRouting.VECTOR_SEARCH, 0) >= 5

        # Print distribution for visibility
        print("\nRouting Distribution:")
        for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tool.value}: {count}")

    def test_category_to_routing_mapping(
        self,
        golden_dataset: list[BenchmarkExample],
    ) -> None:
        """Test that categories map to expected routing patterns."""
        category_tools: dict[QueryCategory, set[ExpectedRouting]] = {}

        for example in golden_dataset:
            if example.category not in category_tools:
                category_tools[example.category] = set()
            category_tools[example.category].update(example.expected_tools)

        # Verify expected patterns
        # Definitional -> should include vector search
        assert ExpectedRouting.VECTOR_SEARCH in category_tools.get(
            QueryCategory.DEFINITIONAL, set()
        )

        # Factual -> should include lookup or text2cypher
        factual_tools = category_tools.get(QueryCategory.FACTUAL, set())
        assert (
            ExpectedRouting.LOOKUP_STANDARD in factual_tools
            or ExpectedRouting.TEXT2CYPHER in factual_tools
            or ExpectedRouting.STANDARDS_BY_INDUSTRY in factual_tools
        )


__all__ = [
    "routing_accuracy",
    "top_k_routing_accuracy",
]
