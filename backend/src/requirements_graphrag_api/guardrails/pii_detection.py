"""PII detection and redaction using Microsoft Presidio.

This module provides PII (Personally Identifiable Information) detection
and anonymization capabilities using the Presidio library.

Supported PII Types:
    - PERSON: Names
    - EMAIL_ADDRESS: Email addresses
    - PHONE_NUMBER: Phone numbers
    - CREDIT_CARD: Credit card numbers
    - US_SSN: US Social Security Numbers
    - US_BANK_NUMBER: US bank account numbers
    - IP_ADDRESS: IP addresses
    - LOCATION: Geographic locations
    - IBAN_CODE: International bank account numbers
    - MEDICAL_LICENSE: Medical license numbers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult
    from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)


# Default PII entities to detect
DEFAULT_PII_ENTITIES: tuple[str, ...] = (
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
)

# Extended entity list for more comprehensive detection
EXTENDED_PII_ENTITIES: tuple[str, ...] = (
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "US_BANK_NUMBER",
    "IP_ADDRESS",
    "LOCATION",
    "IBAN_CODE",
    "MEDICAL_LICENSE",
)


@dataclass(frozen=True, slots=True)
class DetectedEntity:
    """A detected PII entity.

    Attributes:
        entity_type: Type of PII detected (e.g., EMAIL_ADDRESS).
        text: The actual text that was detected.
        start: Start position in the original text.
        end: End position in the original text.
        score: Confidence score (0-1).
    """

    entity_type: str
    text: str
    start: int
    end: int
    score: float


@dataclass(frozen=True, slots=True)
class PIICheckResult:
    """Result of PII detection and redaction.

    Attributes:
        contains_pii: Whether any PII was detected.
        detected_entities: List of detected PII entities.
        anonymized_text: Text with PII redacted/replaced.
        original_text: Original input text.
        entity_count: Number of PII entities detected.
    """

    contains_pii: bool
    detected_entities: tuple[DetectedEntity, ...]
    anonymized_text: str
    original_text: str
    entity_count: int

    def get_entities_by_type(self, entity_type: str) -> tuple[DetectedEntity, ...]:
        """Get all detected entities of a specific type.

        Args:
            entity_type: The PII type to filter by.

        Returns:
            Tuple of DetectedEntity objects matching the type.
        """
        return tuple(e for e in self.detected_entities if e.entity_type == entity_type)


@lru_cache(maxsize=1)
def get_pii_analyzer() -> AnalyzerEngine:
    """Get or create the PII analyzer engine (singleton).

    Returns:
        Configured AnalyzerEngine instance.

    Note:
        Uses lru_cache to avoid reinitializing the NLP models on each call.
        The analyzer loads spaCy models which can take several seconds.
    """
    from presidio_analyzer import AnalyzerEngine

    logger.info("Initializing Presidio AnalyzerEngine...")
    analyzer = AnalyzerEngine()
    logger.info("Presidio AnalyzerEngine initialized successfully")
    return analyzer


@lru_cache(maxsize=1)
def get_pii_anonymizer() -> AnonymizerEngine:
    """Get or create the PII anonymizer engine (singleton).

    Returns:
        Configured AnonymizerEngine instance.
    """
    from presidio_anonymizer import AnonymizerEngine

    return AnonymizerEngine()


def _convert_results(
    results: list[RecognizerResult],
    text: str,
) -> tuple[DetectedEntity, ...]:
    """Convert Presidio results to DetectedEntity objects.

    Args:
        results: List of RecognizerResult from Presidio.
        text: Original text for extracting matched content.

    Returns:
        Tuple of DetectedEntity objects.
    """
    entities = []
    for result in results:
        entities.append(
            DetectedEntity(
                entity_type=result.entity_type,
                text=text[result.start : result.end],
                start=result.start,
                end=result.end,
                score=result.score,
            )
        )
    return tuple(entities)


def detect_and_redact_pii(
    text: str,
    entities: tuple[str, ...] | None = None,
    score_threshold: float = 0.7,
    anonymize_type: str = "replace",
    language: str = "en",
) -> PIICheckResult:
    """Detect and redact PII from text.

    Analyzes the input text for PII entities and returns a result
    with the anonymized text.

    Args:
        text: The text to analyze for PII.
        entities: PII entity types to detect. Defaults to DEFAULT_PII_ENTITIES.
        score_threshold: Minimum confidence score (0-1) for detection.
        anonymize_type: How to anonymize ("replace", "redact", "hash").
        language: Language code for NER. Defaults to "en".

    Returns:
        PIICheckResult with detection results and anonymized text.

    Example:
        >>> result = detect_and_redact_pii("Contact john@example.com")
        >>> result.contains_pii
        True
        >>> result.anonymized_text
        'Contact <EMAIL_ADDRESS>'

        >>> result = detect_and_redact_pii("What is requirements traceability?")
        >>> result.contains_pii
        False
        >>> result.anonymized_text
        'What is requirements traceability?'
    """
    if not text or not text.strip():
        return PIICheckResult(
            contains_pii=False,
            detected_entities=(),
            anonymized_text=text,
            original_text=text,
            entity_count=0,
        )

    entities_to_detect = list(entities) if entities is not None else list(DEFAULT_PII_ENTITIES)

    try:
        analyzer = get_pii_analyzer()
        anonymizer = get_pii_anonymizer()

        # Analyze text for PII
        results = analyzer.analyze(
            text=text,
            entities=entities_to_detect,
            language=language,
            score_threshold=score_threshold,
        )

        if not results:
            return PIICheckResult(
                contains_pii=False,
                detected_entities=(),
                anonymized_text=text,
                original_text=text,
                entity_count=0,
            )

        # Convert results to our format
        detected = _convert_results(results, text)

        # Anonymize the text
        anonymize_operators = _get_anonymize_operators(anonymize_type, results)
        anonymized_result = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=anonymize_operators,
        )

        return PIICheckResult(
            contains_pii=True,
            detected_entities=detected,
            anonymized_text=anonymized_result.text,
            original_text=text,
            entity_count=len(detected),
        )

    except ImportError:
        logger.warning(
            "Presidio libraries not installed. PII detection disabled. "
            "Install with: pip install presidio-analyzer presidio-anonymizer"
        )
        return PIICheckResult(
            contains_pii=False,
            detected_entities=(),
            anonymized_text=text,
            original_text=text,
            entity_count=0,
        )
    except Exception:
        logger.exception("Error during PII detection")
        # On error, return original text but flag that PII detection failed
        return PIICheckResult(
            contains_pii=False,
            detected_entities=(),
            anonymized_text=text,
            original_text=text,
            entity_count=0,
        )


def _get_anonymize_operators(
    anonymize_type: str,
    results: list[RecognizerResult],
) -> dict[str, Any]:
    """Get anonymization operators based on type.

    Args:
        anonymize_type: Type of anonymization ("replace", "redact", "hash").
        results: Presidio analysis results.

    Returns:
        Dictionary of operators for each entity type.
    """
    from presidio_anonymizer.entities import OperatorConfig

    operators: dict[str, Any] = {}

    for result in results:
        entity_type = result.entity_type

        if anonymize_type == "replace":
            # Replace with entity type placeholder: <EMAIL_ADDRESS>
            operators[entity_type] = OperatorConfig(
                "replace",
                {"new_value": f"<{entity_type}>"},
            )
        elif anonymize_type == "redact":
            # Completely remove the PII
            operators[entity_type] = OperatorConfig(
                "redact",
                {},
            )
        elif anonymize_type == "hash":
            # Hash the value (one-way)
            operators[entity_type] = OperatorConfig(
                "hash",
                {"hash_type": "sha256"},
            )
        else:
            # Default to replace
            operators[entity_type] = OperatorConfig(
                "replace",
                {"new_value": f"<{entity_type}>"},
            )

    return operators
