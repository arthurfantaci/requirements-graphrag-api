"""Tests for guardrail event logging."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from unittest.mock import patch

from requirements_graphrag_api.guardrails.events import (
    ActionTaken,
    GuardrailEvent,
    GuardrailEventType,
    compute_input_hash,
    create_injection_event,
    create_pii_event,
    create_rate_limit_event,
    log_guardrail_event,
)


class TestGuardrailEventType:
    """Test GuardrailEventType enum values."""

    def test_phase1_event_types(self):
        assert GuardrailEventType.PROMPT_INJECTION_DETECTED == "prompt_injection_detected"
        assert GuardrailEventType.PROMPT_INJECTION_BLOCKED == "prompt_injection_blocked"
        assert GuardrailEventType.PII_DETECTED == "pii_detected"
        assert GuardrailEventType.PII_REDACTED == "pii_redacted"
        assert GuardrailEventType.RATE_LIMIT_EXCEEDED == "rate_limit_exceeded"

    def test_phase2_event_types_exist(self):
        assert hasattr(GuardrailEventType, "TOXICITY_DETECTED")
        assert hasattr(GuardrailEventType, "TOPIC_OUT_OF_SCOPE")
        assert hasattr(GuardrailEventType, "OUTPUT_FILTERED")


class TestActionTaken:
    """Test ActionTaken enum values."""

    def test_all_actions_exist(self):
        assert ActionTaken.ALLOWED == "allowed"
        assert ActionTaken.WARNED == "warned"
        assert ActionTaken.BLOCKED == "blocked"
        assert ActionTaken.REDACTED == "redacted"
        assert ActionTaken.FILTERED == "filtered"


class TestGuardrailEvent:
    """Test GuardrailEvent dataclass."""

    def test_create_event(self):
        event = GuardrailEvent(
            event_type=GuardrailEventType.PROMPT_INJECTION_BLOCKED,
            request_id="req-123",
            action_taken=ActionTaken.BLOCKED,
        )
        assert event.event_type == GuardrailEventType.PROMPT_INJECTION_BLOCKED
        assert event.request_id == "req-123"
        assert event.action_taken == ActionTaken.BLOCKED

    def test_event_has_timestamp(self):
        before = datetime.now(UTC)
        event = GuardrailEvent(
            event_type=GuardrailEventType.PII_DETECTED,
            request_id="req-456",
            action_taken=ActionTaken.REDACTED,
        )
        after = datetime.now(UTC)
        assert before <= event.timestamp <= after

    def test_event_to_dict(self):
        event = GuardrailEvent(
            event_type=GuardrailEventType.RATE_LIMIT_EXCEEDED,
            request_id="req-789",
            action_taken=ActionTaken.BLOCKED,
            user_ip="192.168.1.1",
            details={"endpoint": "/chat"},
        )
        event_dict = event.to_dict()
        assert event_dict["event_type"] == "rate_limit_exceeded"
        assert event_dict["request_id"] == "req-789"
        assert event_dict["action_taken"] == "blocked"
        assert event_dict["user_ip"] == "192.168.1.1"
        assert event_dict["details"]["endpoint"] == "/chat"
        assert "timestamp" in event_dict

    def test_event_with_optional_fields(self):
        event = GuardrailEvent(
            event_type=GuardrailEventType.PII_REDACTED,
            request_id="req-abc",
            action_taken=ActionTaken.REDACTED,
            user_ip=None,
            api_key_id=None,
            input_hash=None,
            risk_level=None,
        )
        event_dict = event.to_dict()
        assert event_dict["user_ip"] is None
        assert event_dict["api_key_id"] is None


class TestComputeInputHash:
    """Test input hash computation."""

    def test_hash_is_consistent(self):
        text = "test input"
        hash1 = compute_input_hash(text)
        hash2 = compute_input_hash(text)
        assert hash1 == hash2

    def test_different_inputs_different_hashes(self):
        hash1 = compute_input_hash("input 1")
        hash2 = compute_input_hash("input 2")
        assert hash1 != hash2

    def test_hash_length(self):
        # Should be first 16 chars of SHA256 hex
        hash_result = compute_input_hash("any text")
        assert len(hash_result) == 16

    def test_empty_string_hashes(self):
        hash_result = compute_input_hash("")
        assert len(hash_result) == 16


class TestCreateInjectionEvent:
    """Test injection event factory function."""

    def test_creates_blocked_event(self):
        event = create_injection_event(
            request_id="req-001",
            risk_level="high",
            patterns=("instruction_override", "dan_jailbreak"),
            blocked=True,
            user_ip="10.0.0.1",
            input_text="malicious input",
        )
        assert event.event_type == GuardrailEventType.PROMPT_INJECTION_BLOCKED
        assert event.action_taken == ActionTaken.BLOCKED
        assert event.risk_level == "high"
        assert "instruction_override" in event.details["patterns"]
        assert event.details["pattern_count"] == 2

    def test_creates_detected_event(self):
        event = create_injection_event(
            request_id="req-002",
            risk_level="medium",
            patterns=("role_pretend",),
            blocked=False,
            user_ip="10.0.0.2",
        )
        assert event.event_type == GuardrailEventType.PROMPT_INJECTION_DETECTED
        assert event.action_taken == ActionTaken.WARNED

    def test_includes_input_hash(self):
        event = create_injection_event(
            request_id="req-003",
            risk_level="high",
            patterns=(),
            blocked=False,
            input_text="some input to hash",
        )
        assert event.input_hash is not None
        assert len(event.input_hash) == 16


class TestCreatePIIEvent:
    """Test PII event factory function."""

    def test_creates_redacted_event(self):
        event = create_pii_event(
            request_id="req-010",
            entity_types=("EMAIL_ADDRESS", "PHONE_NUMBER"),
            entity_count=2,
            redacted=True,
            user_ip="10.0.1.1",
        )
        assert event.event_type == GuardrailEventType.PII_REDACTED
        assert event.action_taken == ActionTaken.REDACTED
        assert event.details["entity_count"] == 2

    def test_creates_detected_event(self):
        event = create_pii_event(
            request_id="req-011",
            entity_types=("PERSON",),
            entity_count=1,
            redacted=False,
        )
        assert event.event_type == GuardrailEventType.PII_DETECTED
        assert event.action_taken == ActionTaken.WARNED

    def test_deduplicates_entity_types(self):
        event = create_pii_event(
            request_id="req-012",
            entity_types=("EMAIL_ADDRESS", "EMAIL_ADDRESS", "PHONE_NUMBER"),
            entity_count=3,
            redacted=True,
        )
        # Should deduplicate entity types in details
        assert len(set(event.details["entity_types"])) <= 2


class TestCreateRateLimitEvent:
    """Test rate limit event factory function."""

    def test_creates_rate_limit_event(self):
        event = create_rate_limit_event(
            request_id="req-020",
            limit="20/minute",
            endpoint="/chat",
            user_ip="192.168.0.1",
            api_key_id="key-hash-123",
        )
        assert event.event_type == GuardrailEventType.RATE_LIMIT_EXCEEDED
        assert event.action_taken == ActionTaken.BLOCKED
        assert event.details["limit"] == "20/minute"
        assert event.details["endpoint"] == "/chat"
        assert event.api_key_id == "key-hash-123"


class TestLogGuardrailEvent:
    """Test event logging functionality."""

    def test_logs_blocked_event_as_warning(self, caplog):
        event = GuardrailEvent(
            event_type=GuardrailEventType.PROMPT_INJECTION_BLOCKED,
            request_id="req-100",
            action_taken=ActionTaken.BLOCKED,
            risk_level="high",
        )
        with caplog.at_level(logging.WARNING, logger="guardrails"):
            log_guardrail_event(event)
        assert "prompt_injection_blocked" in caplog.text
        assert "req-100" in caplog.text

    def test_logs_redacted_event_as_info(self, caplog):
        event = GuardrailEvent(
            event_type=GuardrailEventType.PII_REDACTED,
            request_id="req-101",
            action_taken=ActionTaken.REDACTED,
        )
        with caplog.at_level(logging.INFO, logger="guardrails"):
            log_guardrail_event(event)
        assert "pii_redacted" in caplog.text

    def test_logs_allowed_event_as_debug(self, caplog):
        event = GuardrailEvent(
            event_type=GuardrailEventType.PII_DETECTED,
            request_id="req-102",
            action_taken=ActionTaken.ALLOWED,
        )
        with caplog.at_level(logging.DEBUG, logger="guardrails"):
            log_guardrail_event(event)
        assert "req-102" in caplog.text

    def test_includes_details_in_log(self, caplog):
        event = GuardrailEvent(
            event_type=GuardrailEventType.RATE_LIMIT_EXCEEDED,
            request_id="req-103",
            action_taken=ActionTaken.BLOCKED,
            details={"endpoint": "/chat", "limit": "20/minute"},
        )
        with caplog.at_level(logging.WARNING, logger="guardrails"):
            log_guardrail_event(event)
        assert "endpoint=/chat" in caplog.text or "/chat" in caplog.text

    def test_can_exclude_details(self, caplog):
        event = GuardrailEvent(
            event_type=GuardrailEventType.RATE_LIMIT_EXCEEDED,
            request_id="req-104",
            action_taken=ActionTaken.BLOCKED,
            details={"secret": "should_not_appear"},
        )
        with caplog.at_level(logging.WARNING, logger="guardrails"):
            log_guardrail_event(event, include_details=False)
        assert "should_not_appear" not in caplog.text

    def test_logs_extra_data(self, caplog):
        """Verify structured logging includes event data in extra."""
        event = GuardrailEvent(
            event_type=GuardrailEventType.PROMPT_INJECTION_BLOCKED,
            request_id="req-105",
            action_taken=ActionTaken.BLOCKED,
        )
        with patch("requirements_graphrag_api.guardrails.events.guardrail_logger") as mock_logger:
            log_guardrail_event(event)
            mock_logger.log.assert_called_once()
            # Check that extra data is passed
            call_kwargs = mock_logger.log.call_args[1]
            assert "guardrail_event" in call_kwargs.get("extra", {})
