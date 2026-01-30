"""Tests for prompts/evaluation.py module."""

from __future__ import annotations

from requirements_graphrag_api.prompts.evaluation import (
    create_cypher_validity_evaluator,
    create_intent_accuracy_evaluator,
    create_json_validity_evaluator,
    create_length_evaluator,
)


class TestJsonValidityEvaluator:
    """Tests for JSON validity evaluator."""

    def test_valid_json(self) -> None:
        """Test that valid JSON scores 1.0."""
        evaluator = create_json_validity_evaluator()
        result = evaluator({"output": '{"intent": "explanatory"}'})
        assert result == {"json_valid": 1.0}

    def test_valid_json_with_markdown(self) -> None:
        """Test that JSON in markdown code blocks is parsed."""
        evaluator = create_json_validity_evaluator()
        result = evaluator({"output": '```json\n{"intent": "structured"}\n```'})
        assert result == {"json_valid": 1.0}

    def test_invalid_json(self) -> None:
        """Test that invalid JSON scores 0.0."""
        evaluator = create_json_validity_evaluator()
        result = evaluator({"output": "not valid json"})
        assert result == {"json_valid": 0.0}

    def test_empty_output(self) -> None:
        """Test that empty output scores 0.0."""
        evaluator = create_json_validity_evaluator()
        result = evaluator({"output": ""})
        assert result == {"json_valid": 0.0}

    def test_missing_output(self) -> None:
        """Test that missing output key scores 0.0."""
        evaluator = create_json_validity_evaluator()
        result = evaluator({})
        assert result == {"json_valid": 0.0}


class TestCypherValidityEvaluator:
    """Tests for Cypher validity evaluator."""

    def test_valid_cypher_basic(self) -> None:
        """Test that basic valid Cypher scores > 0."""
        evaluator = create_cypher_validity_evaluator()
        result = evaluator({"output": "MATCH (n) RETURN n"})
        assert result["cypher_valid"] >= 0.5

    def test_valid_cypher_with_patterns(self) -> None:
        """Test that Cypher with more patterns scores higher."""
        evaluator = create_cypher_validity_evaluator()
        cypher = (
            "MATCH (a:Article)-[:HAS]->(b) WHERE a.title CONTAINS 'test' "
            "RETURN a ORDER BY a.title LIMIT 10"
        )
        result = evaluator({"output": cypher})
        assert result["cypher_valid"] > 0.5

    def test_valid_cypher_with_markdown(self) -> None:
        """Test that Cypher in markdown code blocks is parsed."""
        evaluator = create_cypher_validity_evaluator()
        result = evaluator({"output": "```cypher\nMATCH (n) RETURN count(n)\n```"})
        assert result["cypher_valid"] >= 0.5

    def test_invalid_cypher_no_match(self) -> None:
        """Test that Cypher without MATCH scores 0.0."""
        evaluator = create_cypher_validity_evaluator()
        result = evaluator({"output": "SELECT * FROM table"})
        assert result == {"cypher_valid": 0.0}

    def test_invalid_cypher_no_return(self) -> None:
        """Test that Cypher without RETURN scores 0.0."""
        evaluator = create_cypher_validity_evaluator()
        result = evaluator({"output": "MATCH (n)"})
        assert result == {"cypher_valid": 0.0}

    def test_empty_output(self) -> None:
        """Test that empty output scores 0.0."""
        evaluator = create_cypher_validity_evaluator()
        result = evaluator({"output": ""})
        assert result == {"cypher_valid": 0.0}


class TestLengthEvaluator:
    """Tests for length evaluator."""

    def test_appropriate_length(self) -> None:
        """Test that appropriate length scores 1.0."""
        evaluator = create_length_evaluator(min_length=10, max_length=100)
        result = evaluator({"output": "This is a good length response."})
        assert result == {"length_appropriate": 1.0}

    def test_too_short(self) -> None:
        """Test that too short output scores < 1.0."""
        evaluator = create_length_evaluator(min_length=100, max_length=1000)
        result = evaluator({"output": "Short"})
        assert result["length_appropriate"] < 1.0
        assert result["length_appropriate"] > 0.0

    def test_too_long(self) -> None:
        """Test that too long output scores < 1.0."""
        evaluator = create_length_evaluator(min_length=10, max_length=50)
        result = evaluator({"output": "x" * 100})
        assert result["length_appropriate"] < 1.0

    def test_empty_output(self) -> None:
        """Test that empty output scores 0.0."""
        evaluator = create_length_evaluator(min_length=10, max_length=100)
        result = evaluator({"output": ""})
        assert result == {"length_appropriate": 0.0}

    def test_custom_bounds(self) -> None:
        """Test custom min/max bounds."""
        evaluator = create_length_evaluator(min_length=5, max_length=10)
        # Exactly at min
        result = evaluator({"output": "12345"})
        assert result == {"length_appropriate": 1.0}
        # Exactly at max
        result = evaluator({"output": "1234567890"})
        assert result == {"length_appropriate": 1.0}


class TestIntentAccuracyEvaluator:
    """Tests for intent accuracy evaluator."""

    def test_correct_intent_explanatory(self) -> None:
        """Test correct classification of explanatory intent."""
        evaluator = create_intent_accuracy_evaluator()
        result = evaluator(
            {
                "output": '{"intent": "explanatory"}',
                "expected": {"intent": "explanatory"},
            }
        )
        assert result == {"intent_accuracy": 1.0}

    def test_correct_intent_structured(self) -> None:
        """Test correct classification of structured intent."""
        evaluator = create_intent_accuracy_evaluator()
        result = evaluator(
            {
                "output": '{"intent": "structured"}',
                "expected": {"intent": "structured"},
            }
        )
        assert result == {"intent_accuracy": 1.0}

    def test_incorrect_intent(self) -> None:
        """Test incorrect classification scores 0.0."""
        evaluator = create_intent_accuracy_evaluator()
        result = evaluator(
            {
                "output": '{"intent": "explanatory"}',
                "expected": {"intent": "structured"},
            }
        )
        assert result == {"intent_accuracy": 0.0}

    def test_case_insensitive(self) -> None:
        """Test that comparison is case insensitive."""
        evaluator = create_intent_accuracy_evaluator()
        result = evaluator(
            {
                "output": '{"intent": "EXPLANATORY"}',
                "expected": {"intent": "explanatory"},
            }
        )
        assert result == {"intent_accuracy": 1.0}

    def test_no_expected_intent(self) -> None:
        """Test that missing expected intent defaults to 1.0."""
        evaluator = create_intent_accuracy_evaluator()
        result = evaluator(
            {
                "output": '{"intent": "explanatory"}',
                "expected": {},
            }
        )
        assert result == {"intent_accuracy": 1.0}

    def test_keyword_fallback_explanatory(self) -> None:
        """Test keyword fallback for non-JSON output."""
        evaluator = create_intent_accuracy_evaluator()
        result = evaluator(
            {
                "output": "The intent is explanatory because...",
                "expected": {"intent": "explanatory"},
            }
        )
        assert result == {"intent_accuracy": 1.0}

    def test_keyword_fallback_structured(self) -> None:
        """Test keyword fallback for structured."""
        evaluator = create_intent_accuracy_evaluator()
        result = evaluator(
            {
                "output": "This is a structured query request",
                "expected": {"intent": "structured"},
            }
        )
        assert result == {"intent_accuracy": 1.0}

    def test_json_in_markdown(self) -> None:
        """Test JSON in markdown code blocks."""
        evaluator = create_intent_accuracy_evaluator()
        result = evaluator(
            {
                "output": '```json\n{"intent": "explanatory"}\n```',
                "expected": {"intent": "explanatory"},
            }
        )
        assert result == {"intent_accuracy": 1.0}
