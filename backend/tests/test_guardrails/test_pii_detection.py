"""Tests for PII detection and redaction.

These tests use mocking to avoid requiring spaCy NLP models,
which makes them faster and more reliable in CI environments.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from requirements_graphrag_api.guardrails.pii_detection import (
    DEFAULT_PII_ENTITIES,
    DetectedEntity,
    PIICheckResult,
    detect_and_redact_pii,
)


@pytest.fixture
def mock_presidio():
    """Fixture to mock Presidio analyzer and anonymizer."""
    with (
        patch(
            "requirements_graphrag_api.guardrails.pii_detection.get_pii_analyzer"
        ) as mock_get_analyzer,
        patch(
            "requirements_graphrag_api.guardrails.pii_detection.get_pii_anonymizer"
        ) as mock_get_anonymizer,
    ):
        mock_analyzer = MagicMock()
        mock_anonymizer = MagicMock()
        mock_get_analyzer.return_value = mock_analyzer
        mock_get_anonymizer.return_value = mock_anonymizer

        yield {
            "analyzer": mock_analyzer,
            "anonymizer": mock_anonymizer,
            "get_analyzer": mock_get_analyzer,
            "get_anonymizer": mock_get_anonymizer,
        }


class TestPIIDetectionBasics:
    """Test basic PII detection functionality."""

    def test_empty_string_returns_empty(self):
        # Empty string doesn't call analyzer
        result = detect_and_redact_pii("")
        assert result.contains_pii is False
        assert result.anonymized_text == ""

    def test_whitespace_only_returns_original(self):
        result = detect_and_redact_pii("   ")
        assert result.contains_pii is False
        assert result.anonymized_text == "   "

    def test_no_pii_returns_original_text(self, mock_presidio):
        mock_presidio["analyzer"].analyze.return_value = []

        text = "What is requirements traceability?"
        result = detect_and_redact_pii(text)

        assert result.contains_pii is False
        assert result.anonymized_text == text
        assert result.entity_count == 0
        assert len(result.detected_entities) == 0


class TestEmailDetection:
    """Test email address detection and redaction."""

    def test_detects_email_address(self, mock_presidio):
        # Mock analyzer to return email entity
        mock_result = MagicMock()
        mock_result.entity_type = "EMAIL_ADDRESS"
        mock_result.start = 14
        mock_result.end = 30
        mock_result.score = 0.95
        mock_presidio["analyzer"].analyze.return_value = [mock_result]

        # Mock anonymizer
        anon_result = MagicMock()
        anon_result.text = "Contact me at <EMAIL_ADDRESS>"
        mock_presidio["anonymizer"].anonymize.return_value = anon_result

        result = detect_and_redact_pii("Contact me at john@example.com")

        assert result.contains_pii is True
        assert any(e.entity_type == "EMAIL_ADDRESS" for e in result.detected_entities)

    def test_redacts_email_address(self, mock_presidio):
        mock_result = MagicMock()
        mock_result.entity_type = "EMAIL_ADDRESS"
        mock_result.start = 6
        mock_result.end = 22
        mock_result.score = 0.95
        mock_presidio["analyzer"].analyze.return_value = [mock_result]

        anon_result = MagicMock()
        anon_result.text = "Email <EMAIL_ADDRESS> for more info"
        mock_presidio["anonymizer"].anonymize.return_value = anon_result

        result = detect_and_redact_pii("Email john@example.com for more info")

        assert "<EMAIL_ADDRESS>" in result.anonymized_text
        assert "john@example.com" not in result.anonymized_text

    def test_multiple_emails(self, mock_presidio):
        mock_results = [
            MagicMock(entity_type="EMAIL_ADDRESS", start=8, end=24, score=0.95),
            MagicMock(entity_type="EMAIL_ADDRESS", start=28, end=41, score=0.95),
        ]
        mock_presidio["analyzer"].analyze.return_value = mock_results

        anon_result = MagicMock()
        anon_result.text = "Contact <EMAIL_ADDRESS> or <EMAIL_ADDRESS>"
        mock_presidio["anonymizer"].anonymize.return_value = anon_result

        result = detect_and_redact_pii("Contact john@example.com or jane@test.org")

        assert result.entity_count == 2
        email_entities = result.get_entities_by_type("EMAIL_ADDRESS")
        assert len(email_entities) == 2


class TestPhoneNumberDetection:
    """Test phone number detection and redaction."""

    def test_detects_phone_number(self, mock_presidio):
        mock_result = MagicMock()
        mock_result.entity_type = "PHONE_NUMBER"
        mock_result.start = 11
        mock_result.end = 23
        mock_result.score = 0.85
        mock_presidio["analyzer"].analyze.return_value = [mock_result]

        anon_result = MagicMock()
        anon_result.text = "Call me at <PHONE_NUMBER>"
        mock_presidio["anonymizer"].anonymize.return_value = anon_result

        result = detect_and_redact_pii("Call me at 555-123-4567")

        assert result.contains_pii is True
        assert any(e.entity_type == "PHONE_NUMBER" for e in result.detected_entities)


class TestNameDetection:
    """Test person name detection."""

    def test_detects_full_name(self, mock_presidio):
        mock_result = MagicMock()
        mock_result.entity_type = "PERSON"
        mock_result.start = 15
        mock_result.end = 25
        mock_result.score = 0.85
        mock_presidio["analyzer"].analyze.return_value = [mock_result]

        anon_result = MagicMock()
        anon_result.text = "Please contact <PERSON> for details"
        mock_presidio["anonymizer"].anonymize.return_value = anon_result

        result = detect_and_redact_pii("Please contact John Smith for details")

        assert result.contains_pii is True
        person_entities = result.get_entities_by_type("PERSON")
        assert len(person_entities) == 1


class TestAnonymizationTypes:
    """Test different anonymization methods."""

    def test_replace_anonymization(self, mock_presidio):
        mock_result = MagicMock()
        mock_result.entity_type = "EMAIL_ADDRESS"
        mock_result.start = 6
        mock_result.end = 22
        mock_result.score = 0.95
        mock_presidio["analyzer"].analyze.return_value = [mock_result]

        anon_result = MagicMock()
        anon_result.text = "Email <EMAIL_ADDRESS>"
        mock_presidio["anonymizer"].anonymize.return_value = anon_result

        result = detect_and_redact_pii("Email john@example.com", anonymize_type="replace")

        assert "<EMAIL_ADDRESS>" in result.anonymized_text

    def test_hash_anonymization_calls_anonymizer(self, mock_presidio):
        mock_result = MagicMock()
        mock_result.entity_type = "EMAIL_ADDRESS"
        mock_result.start = 6
        mock_result.end = 22
        mock_result.score = 0.95
        mock_presidio["analyzer"].analyze.return_value = [mock_result]

        anon_result = MagicMock()
        anon_result.text = "Email abc123hash"
        mock_presidio["anonymizer"].anonymize.return_value = anon_result

        result = detect_and_redact_pii("Email john@example.com", anonymize_type="hash")

        assert "john@example.com" not in result.anonymized_text


class TestScoreThreshold:
    """Test confidence score threshold filtering."""

    def test_threshold_passed_to_analyzer(self, mock_presidio):
        mock_presidio["analyzer"].analyze.return_value = []

        detect_and_redact_pii("Test text", score_threshold=0.9)

        # Verify score_threshold was passed to analyzer
        mock_presidio["analyzer"].analyze.assert_called_once()
        call_kwargs = mock_presidio["analyzer"].analyze.call_args[1]
        assert call_kwargs["score_threshold"] == 0.9


class TestEntityFiltering:
    """Test filtering by entity types."""

    def test_only_specified_entities_passed(self, mock_presidio):
        mock_presidio["analyzer"].analyze.return_value = []

        detect_and_redact_pii("Test", entities=("EMAIL_ADDRESS",))

        call_kwargs = mock_presidio["analyzer"].analyze.call_args[1]
        assert call_kwargs["entities"] == ["EMAIL_ADDRESS"]

    def test_default_entities_list(self):
        assert len(DEFAULT_PII_ENTITIES) > 0
        assert "EMAIL_ADDRESS" in DEFAULT_PII_ENTITIES
        assert "PHONE_NUMBER" in DEFAULT_PII_ENTITIES
        assert "PERSON" in DEFAULT_PII_ENTITIES


class TestResultDataclass:
    """Test PIICheckResult dataclass properties."""

    def test_result_structure(self):
        result = PIICheckResult(
            contains_pii=True,
            detected_entities=(
                DetectedEntity(
                    entity_type="EMAIL_ADDRESS",
                    text="test@example.com",
                    start=0,
                    end=16,
                    score=0.95,
                ),
            ),
            anonymized_text="<EMAIL_ADDRESS>",
            original_text="test@example.com",
            entity_count=1,
        )
        assert result.contains_pii is True
        assert result.entity_count == 1

    def test_get_entities_by_type(self):
        entities = (
            DetectedEntity("EMAIL_ADDRESS", "a@b.com", 0, 7, 0.9),
            DetectedEntity("EMAIL_ADDRESS", "c@d.com", 10, 17, 0.9),
            DetectedEntity("PHONE_NUMBER", "555-1234", 20, 28, 0.85),
        )
        result = PIICheckResult(
            contains_pii=True,
            detected_entities=entities,
            anonymized_text="redacted",
            original_text="original",
            entity_count=3,
        )

        emails = result.get_entities_by_type("EMAIL_ADDRESS")
        phones = result.get_entities_by_type("PHONE_NUMBER")

        assert len(emails) == 2
        assert len(phones) == 1


class TestDomainAllowList:
    """Test that domain-specific terms are not flagged as PII."""

    def test_cypher_not_flagged_as_person(self, mock_presidio):
        text = "The previous Cypher query was incorrect"
        mock_result = MagicMock()
        mock_result.entity_type = "PERSON"
        mock_result.start = 13
        mock_result.end = 19
        mock_result.score = 0.85
        mock_presidio["analyzer"].analyze.return_value = [mock_result]

        result = detect_and_redact_pii(text)

        assert result.contains_pii is False
        assert result.anonymized_text == text

    def test_neo4j_not_flagged_as_person(self, mock_presidio):
        text = "Use Neo4j for graph databases"
        mock_result = MagicMock()
        mock_result.entity_type = "PERSON"
        mock_result.start = 4
        mock_result.end = 9
        mock_result.score = 0.80
        mock_presidio["analyzer"].analyze.return_value = [mock_result]

        result = detect_and_redact_pii(text)

        assert result.contains_pii is False
        assert result.anonymized_text == text

    def test_graphrag_not_flagged(self, mock_presidio):
        text = "GraphRAG combines graph and RAG"
        mock_result = MagicMock()
        mock_result.entity_type = "PERSON"
        mock_result.start = 0
        mock_result.end = 8
        mock_result.score = 0.75
        mock_presidio["analyzer"].analyze.return_value = [mock_result]

        result = detect_and_redact_pii(text)

        assert result.contains_pii is False
        assert result.anonymized_text == text

    def test_allow_list_case_insensitive(self, mock_presidio):
        text = "CYPHER is a query language"
        mock_result = MagicMock()
        mock_result.entity_type = "PERSON"
        mock_result.start = 0
        mock_result.end = 6
        mock_result.score = 0.85
        mock_presidio["analyzer"].analyze.return_value = [mock_result]

        result = detect_and_redact_pii(text)

        assert result.contains_pii is False
        assert result.anonymized_text == text

    def test_real_person_still_detected(self, mock_presidio):
        text = "Contact John Smith about Cypher"
        mock_results = [
            MagicMock(entity_type="PERSON", start=8, end=18, score=0.90),
            MagicMock(entity_type="PERSON", start=25, end=31, score=0.85),
        ]
        mock_presidio["analyzer"].analyze.return_value = mock_results

        anon_result = MagicMock()
        anon_result.text = "Contact <PERSON> about Cypher"
        mock_presidio["anonymizer"].anonymize.return_value = anon_result

        result = detect_and_redact_pii(text)

        assert result.contains_pii is True
        assert result.entity_count == 1
        assert result.detected_entities[0].text == "John Smith"


class TestErrorHandling:
    """Test error handling and fallback behavior."""

    def test_returns_original_on_import_error(self):
        with patch(
            "requirements_graphrag_api.guardrails.pii_detection.get_pii_analyzer",
            side_effect=ImportError("Presidio not installed"),
        ):
            result = detect_and_redact_pii("Email john@example.com")

            assert result.anonymized_text == "Email john@example.com"
            assert result.contains_pii is False

    def test_returns_original_on_analysis_error(self, mock_presidio):
        mock_presidio["analyzer"].analyze.side_effect = Exception("Analysis failed")

        result = detect_and_redact_pii("Email john@example.com")

        assert result.anonymized_text == "Email john@example.com"
        assert result.contains_pii is False


class TestDetectedEntityDataclass:
    """Test DetectedEntity dataclass."""

    def test_entity_creation(self):
        entity = DetectedEntity(
            entity_type="EMAIL_ADDRESS",
            text="test@example.com",
            start=0,
            end=16,
            score=0.95,
        )

        assert entity.entity_type == "EMAIL_ADDRESS"
        assert entity.text == "test@example.com"
        assert entity.start == 0
        assert entity.end == 16
        assert entity.score == 0.95

    def test_entity_is_frozen(self):
        entity = DetectedEntity("EMAIL_ADDRESS", "test", 0, 4, 0.9)
        with pytest.raises(AttributeError):
            entity.entity_type = "PHONE_NUMBER"  # type: ignore[misc]
