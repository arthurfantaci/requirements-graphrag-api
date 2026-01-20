"""Tests for domain-specific evaluation metrics module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jama_mcp_server_graphrag.config import AppConfig
from jama_mcp_server_graphrag.evaluation.domain_metrics import (
    DOMAIN_TERMS,
    KNOWN_STANDARDS,
    DomainMetrics,
    _parse_score,
    compute_all_domain_metrics,
    compute_citation_accuracy,
    compute_completeness_score,
    compute_regulatory_alignment,
    compute_technical_precision,
    compute_traceability_coverage,
    extract_domain_terms_from_text,
    extract_standards_from_text,
)

_TEST_PASSWORD = "test"


@pytest.fixture
def mock_config() -> AppConfig:
    """Create a mock config for testing."""
    return AppConfig(
        neo4j_uri="neo4j://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password=_TEST_PASSWORD,
        openai_api_key="sk-test",
        chat_model="gpt-4o",
    )


def _create_mock_llm(score: str) -> MagicMock:
    """Create a mock LLM that returns the specified score.

    Args:
        score: The score string to return from the mock chain.

    Returns:
        Mock LLM configured for LCEL chaining.
    """
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=score)

    mock_llm = MagicMock()
    mock_llm.__or__ = MagicMock(return_value=mock_chain)
    return mock_llm


class TestDomainMetrics:
    """Tests for DomainMetrics dataclass."""

    def test_metrics_creation(self) -> None:
        """Test creating domain metrics object."""
        metrics = DomainMetrics(
            citation_accuracy=0.9,
            traceability_coverage=0.85,
            technical_precision=0.8,
            completeness_score=0.75,
            regulatory_alignment=0.95,
        )

        assert metrics.citation_accuracy == 0.9
        assert metrics.traceability_coverage == 0.85
        assert metrics.technical_precision == 0.8
        assert metrics.completeness_score == 0.75
        assert metrics.regulatory_alignment == 0.95

    def test_average_calculation(self) -> None:
        """Test average score calculation."""
        metrics = DomainMetrics(
            citation_accuracy=1.0,
            traceability_coverage=0.8,
            technical_precision=0.6,
            completeness_score=0.4,
            regulatory_alignment=0.7,
        )

        expected_avg = (1.0 + 0.8 + 0.6 + 0.4 + 0.7) / 5
        assert metrics.average == expected_avg

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = DomainMetrics(
            citation_accuracy=0.9,
            traceability_coverage=0.85,
            technical_precision=0.8,
            completeness_score=0.75,
            regulatory_alignment=0.95,
        )

        result = metrics.to_dict()

        assert result["citation_accuracy"] == 0.9
        assert result["traceability_coverage"] == 0.85
        assert result["technical_precision"] == 0.8
        assert result["completeness_score"] == 0.75
        assert result["regulatory_alignment"] == 0.95
        assert "average" in result


class TestParseScore:
    """Tests for _parse_score function."""

    def test_parse_valid_score(self) -> None:
        """Test parsing a valid score."""
        assert _parse_score("0.85") == 0.85
        assert _parse_score("1.0") == 1.0
        assert _parse_score("0") == 0.0

    def test_parse_with_whitespace(self) -> None:
        """Test parsing score with whitespace."""
        assert _parse_score("  0.75  ") == 0.75

    def test_parse_with_text_prefix(self) -> None:
        """Test parsing score with text prefix."""
        assert _parse_score("Score: 0.85") == 0.85
        assert _parse_score("The score is 0.9") == 0.9

    def test_parse_clamps_above_one(self) -> None:
        """Test that scores above 1 are clamped."""
        assert _parse_score("1.5") == 1.0

    def test_parse_clamps_below_zero(self) -> None:
        """Test that scores below 0 are clamped."""
        assert _parse_score("-0.5") == 0.0

    def test_parse_invalid_returns_default(self) -> None:
        """Test that invalid input returns 0.5 default."""
        assert _parse_score("not a number") == 0.5
        assert _parse_score("") == 0.5


class TestKnownStandards:
    """Tests for the KNOWN_STANDARDS constant."""

    def test_contains_major_standards(self) -> None:
        """Test that major standards are included."""
        assert "ISO 26262" in KNOWN_STANDARDS
        assert "IEC 62304" in KNOWN_STANDARDS
        assert "DO-178C" in KNOWN_STANDARDS
        assert "ASPICE" in KNOWN_STANDARDS
        assert "CMMI" in KNOWN_STANDARDS

    def test_is_frozenset(self) -> None:
        """Test that KNOWN_STANDARDS is immutable."""
        assert isinstance(KNOWN_STANDARDS, frozenset)


class TestDomainTerms:
    """Tests for the DOMAIN_TERMS constant."""

    def test_contains_key_terms(self) -> None:
        """Test that key domain terms are included."""
        assert "traceability" in DOMAIN_TERMS
        assert "ASIL" in DOMAIN_TERMS
        assert "V-model" in DOMAIN_TERMS
        assert "FMEA" in DOMAIN_TERMS
        assert "verification" in DOMAIN_TERMS

    def test_is_frozenset(self) -> None:
        """Test that DOMAIN_TERMS is immutable."""
        assert isinstance(DOMAIN_TERMS, frozenset)


class TestExtractStandardsFromText:
    """Tests for extract_standards_from_text function."""

    def test_extracts_single_standard(self) -> None:
        """Test extracting a single standard."""
        text = "The project follows ISO 26262 guidelines."
        result = extract_standards_from_text(text)
        assert "ISO 26262" in result

    def test_extracts_multiple_standards(self) -> None:
        """Test extracting multiple standards."""
        text = "Automotive projects use ISO 26262 and ASPICE for compliance."
        result = extract_standards_from_text(text)
        assert "ISO 26262" in result
        assert "ASPICE" in result

    def test_case_insensitive(self) -> None:
        """Test that extraction is case insensitive."""
        text = "The standard iso 26262 defines safety requirements."
        result = extract_standards_from_text(text)
        assert "ISO 26262" in result

    def test_returns_empty_for_no_standards(self) -> None:
        """Test that empty list returned when no standards found."""
        text = "This text contains no standard references."
        result = extract_standards_from_text(text)
        assert result == []


class TestExtractDomainTermsFromText:
    """Tests for extract_domain_terms_from_text function."""

    def test_extracts_single_term(self) -> None:
        """Test extracting a single domain term."""
        text = "Requirements traceability is important for compliance."
        result = extract_domain_terms_from_text(text)
        assert "traceability" in result
        assert "requirements" in result

    def test_extracts_multiple_terms(self) -> None:
        """Test extracting multiple domain terms."""
        text = "The V-model includes verification and validation phases."
        result = extract_domain_terms_from_text(text)
        assert "V-model" in result
        assert "verification" in result
        assert "validation" in result

    def test_case_insensitive(self) -> None:
        """Test that extraction is case insensitive."""
        text = "FMEA analysis is required for ASIL B components."
        result = extract_domain_terms_from_text(text)
        assert "FMEA" in result
        assert "ASIL" in result

    def test_returns_empty_for_no_terms(self) -> None:
        """Test that empty list returned when no terms found."""
        text = "This is a generic text without domain terms."
        result = extract_domain_terms_from_text(text)
        assert result == []


class TestComputeCitationAccuracy:
    """Tests for compute_citation_accuracy function."""

    @pytest.mark.asyncio
    async def test_returns_score(self, mock_config: AppConfig) -> None:
        """Test that compute_citation_accuracy returns a score."""
        with patch("jama_mcp_server_graphrag.evaluation.domain_metrics.ChatOpenAI") as mock_chat:
            mock_chat.return_value = _create_mock_llm("0.85")

            score = await compute_citation_accuracy(
                mock_config,
                "ISO 26262 defines ASIL levels from A to D.",
                ["ISO 26262"],
            )

            assert 0.0 <= score <= 1.0
            assert score == 0.85

    @pytest.mark.asyncio
    async def test_handles_none_expected_standards(self, mock_config: AppConfig) -> None:
        """Test handling of None expected_standards."""
        with patch("jama_mcp_server_graphrag.evaluation.domain_metrics.ChatOpenAI") as mock_chat:
            mock_chat.return_value = _create_mock_llm("0.9")

            score = await compute_citation_accuracy(
                mock_config,
                "This answer mentions no standards.",
                None,
            )

            assert 0.0 <= score <= 1.0


class TestComputeTraceabilityCoverage:
    """Tests for compute_traceability_coverage function."""

    @pytest.mark.asyncio
    async def test_returns_score(self, mock_config: AppConfig) -> None:
        """Test that compute_traceability_coverage returns a score."""
        with patch("jama_mcp_server_graphrag.evaluation.domain_metrics.ChatOpenAI") as mock_chat:
            mock_chat.return_value = _create_mock_llm("0.9")

            score = await compute_traceability_coverage(
                mock_config,
                "How do I implement bidirectional traceability?",
                "Bidirectional traceability requires forward and backward trace links.",
                ["traceability", "bidirectional traceability"],
            )

            assert 0.0 <= score <= 1.0
            assert score == 0.9

    @pytest.mark.asyncio
    async def test_handles_none_expected_entities(self, mock_config: AppConfig) -> None:
        """Test handling of None expected_entities."""
        with patch("jama_mcp_server_graphrag.evaluation.domain_metrics.ChatOpenAI") as mock_chat:
            mock_chat.return_value = _create_mock_llm("1.0")

            score = await compute_traceability_coverage(
                mock_config,
                "What is the weather?",
                "I cannot answer weather questions.",
                None,
            )

            assert 0.0 <= score <= 1.0


class TestComputeTechnicalPrecision:
    """Tests for compute_technical_precision function."""

    @pytest.mark.asyncio
    async def test_returns_score(self, mock_config: AppConfig) -> None:
        """Test that compute_technical_precision returns a score."""
        with patch("jama_mcp_server_graphrag.evaluation.domain_metrics.ChatOpenAI") as mock_chat:
            mock_chat.return_value = _create_mock_llm("0.95")

            score = await compute_technical_precision(
                mock_config,
                "What are ASIL levels?",
                "ASIL (Automotive Safety Integrity Level) ranges from A to D, "
                "with D being the most stringent.",
            )

            assert 0.0 <= score <= 1.0
            assert score == 0.95


class TestComputeCompletenessScore:
    """Tests for compute_completeness_score function."""

    @pytest.mark.asyncio
    async def test_returns_score(self, mock_config: AppConfig) -> None:
        """Test that compute_completeness_score returns a score."""
        with patch("jama_mcp_server_graphrag.evaluation.domain_metrics.ChatOpenAI") as mock_chat:
            mock_chat.return_value = _create_mock_llm("0.8")

            score = await compute_completeness_score(
                mock_config,
                "What is traceability and why is it important?",
                "Traceability is tracking requirements. It is important for compliance.",
            )

            assert 0.0 <= score <= 1.0
            assert score == 0.8

    @pytest.mark.asyncio
    async def test_multi_part_question(self, mock_config: AppConfig) -> None:
        """Test completeness for multi-part questions."""
        with patch("jama_mcp_server_graphrag.evaluation.domain_metrics.ChatOpenAI") as mock_chat:
            mock_chat.return_value = _create_mock_llm("1.0")

            score = await compute_completeness_score(
                mock_config,
                "Compare verification and validation. What's the difference?",
                "Verification checks if we built the product right. "
                "Validation checks if we built the right product. "
                "They differ in focus: correctness vs. needs.",
            )

            assert score == 1.0


class TestComputeRegulatoryAlignment:
    """Tests for compute_regulatory_alignment function."""

    @pytest.mark.asyncio
    async def test_returns_score(self, mock_config: AppConfig) -> None:
        """Test that compute_regulatory_alignment returns a score."""
        with patch("jama_mcp_server_graphrag.evaluation.domain_metrics.ChatOpenAI") as mock_chat:
            mock_chat.return_value = _create_mock_llm("0.9")

            score = await compute_regulatory_alignment(
                mock_config,
                "What standards apply to automotive software?",
                "Automotive software must comply with ISO 26262 for functional safety.",
                ["ISO 26262"],
            )

            assert 0.0 <= score <= 1.0
            assert score == 0.9

    @pytest.mark.asyncio
    async def test_handles_none_expected_standards(self, mock_config: AppConfig) -> None:
        """Test handling of None expected_standards."""
        with patch("jama_mcp_server_graphrag.evaluation.domain_metrics.ChatOpenAI") as mock_chat:
            mock_chat.return_value = _create_mock_llm("1.0")

            score = await compute_regulatory_alignment(
                mock_config,
                "What is traceability?",
                "Traceability is the ability to trace requirements.",
                None,
            )

            assert 0.0 <= score <= 1.0


class TestComputeAllDomainMetrics:
    """Tests for compute_all_domain_metrics function."""

    @pytest.mark.asyncio
    async def test_returns_all_metrics(self, mock_config: AppConfig) -> None:
        """Test that compute_all_domain_metrics returns all scores."""
        with patch("jama_mcp_server_graphrag.evaluation.domain_metrics.ChatOpenAI") as mock_chat:
            # Mock will be called multiple times, return different scores
            mock_llm = MagicMock()
            mock_chain = MagicMock()

            scores = iter(["0.9", "0.85", "0.95", "0.8", "0.88"])
            mock_chain.ainvoke = AsyncMock(side_effect=lambda _: next(scores))
            mock_llm.__or__ = MagicMock(return_value=mock_chain)
            mock_chat.return_value = mock_llm

            metrics = await compute_all_domain_metrics(
                mock_config,
                "What is ISO 26262 and how does it relate to traceability?",
                "ISO 26262 is the automotive functional safety standard. "
                "It requires bidirectional traceability.",
                expected_standards=["ISO 26262"],
                expected_entities=["traceability"],
            )

            assert isinstance(metrics, DomainMetrics)
            assert metrics.citation_accuracy == 0.9
            assert metrics.traceability_coverage == 0.85
            assert metrics.technical_precision == 0.95
            assert metrics.completeness_score == 0.8
            assert metrics.regulatory_alignment == 0.88

    @pytest.mark.asyncio
    async def test_handles_none_optionals(self, mock_config: AppConfig) -> None:
        """Test handling of None optional parameters."""
        with patch("jama_mcp_server_graphrag.evaluation.domain_metrics.ChatOpenAI") as mock_chat:
            mock_chat.return_value = _create_mock_llm("0.75")

            metrics = await compute_all_domain_metrics(
                mock_config,
                "What is requirements management?",
                "Requirements management is the process of managing requirements.",
                expected_standards=None,
                expected_entities=None,
            )

            assert isinstance(metrics, DomainMetrics)
            # All scores should be 0.75 since we're using the same mock
            assert all(v == 0.75 for k, v in metrics.to_dict().items() if k != "average")
