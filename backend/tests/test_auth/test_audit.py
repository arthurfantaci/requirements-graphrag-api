"""Tests for audit logging."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from requirements_graphrag_api.auth.api_key import APIKeyInfo, APIKeyTier
from requirements_graphrag_api.auth.audit import (
    AuditEvent,
    AuditEventType,
    AuditHandler,
    AuditLogger,
    LoggingAuditHandler,
    log_api_error,
    log_api_request,
    log_api_response,
    log_auth_failure,
    log_auth_success,
    log_guardrail_event,
    log_rate_limit_event,
)


class TestAuditEventType:
    """Tests for AuditEventType enum."""

    def test_auth_event_types(self):
        """Verify auth event type values."""
        assert AuditEventType.AUTH_SUCCESS == "auth.success"
        assert AuditEventType.AUTH_FAILURE == "auth.failure"
        assert AuditEventType.AUTH_KEY_CREATED == "auth.key_created"
        assert AuditEventType.AUTH_KEY_REVOKED == "auth.key_revoked"

    def test_api_event_types(self):
        """Verify API event type values."""
        assert AuditEventType.API_REQUEST == "api.request"
        assert AuditEventType.API_RESPONSE == "api.response"
        assert AuditEventType.API_ERROR == "api.error"

    def test_guardrail_event_types(self):
        """Verify guardrail event type values."""
        assert AuditEventType.GUARDRAIL_TRIGGERED == "guardrail.triggered"
        assert AuditEventType.GUARDRAIL_BLOCKED == "guardrail.blocked"

    def test_rate_limit_event_types(self):
        """Verify rate limit event type values."""
        assert AuditEventType.RATE_LIMIT_WARNING == "rate_limit.warning"
        assert AuditEventType.RATE_LIMIT_EXCEEDED == "rate_limit.exceeded"


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    @pytest.fixture
    def sample_event(self) -> AuditEvent:
        """Create a sample audit event."""
        return AuditEvent(
            event_type=AuditEventType.API_REQUEST,
            timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
            request_id="test-request-id",
            client_id="test-client-id",
            client_ip="192.168.1.1",
            endpoint="/chat",
            method="POST",
            status_code=200,
            duration_ms=150.5,
            details={"key": "value"},
        )

    def test_event_is_immutable(self, sample_event: AuditEvent):
        """AuditEvent should be immutable."""
        with pytest.raises(AttributeError):
            sample_event.request_id = "new-id"  # type: ignore[misc]

    def test_to_dict(self, sample_event: AuditEvent):
        """to_dict should convert event to dictionary."""
        data = sample_event.to_dict()

        assert data["event_type"] == "api.request"
        assert data["timestamp"] == "2024-01-15T12:00:00+00:00"
        assert data["request_id"] == "test-request-id"
        assert data["client_id"] == "test-client-id"
        assert data["endpoint"] == "/chat"
        assert data["method"] == "POST"
        assert data["status_code"] == 200
        assert data["duration_ms"] == 150.5
        assert data["details"]["key"] == "value"

    def test_to_json(self, sample_event: AuditEvent):
        """to_json should return valid JSON string."""
        json_str = sample_event.to_json()
        import json

        data = json.loads(json_str)
        assert data["event_type"] == "api.request"


class TestAuditLogger:
    """Tests for AuditLogger."""

    @pytest.fixture
    def logger(self) -> AuditLogger:
        """Create a fresh audit logger."""
        return AuditLogger()

    @pytest.fixture
    def sample_event(self) -> AuditEvent:
        """Create a sample audit event."""
        return AuditEvent(
            event_type=AuditEventType.API_REQUEST,
            timestamp=datetime.now(UTC),
            request_id="test-request-id",
            client_id=None,
            client_ip="127.0.0.1",
            endpoint="/test",
            method="GET",
        )

    @pytest.mark.asyncio
    async def test_log_with_no_handlers(self, logger: AuditLogger, sample_event: AuditEvent):
        """Should handle logging with no handlers gracefully."""
        # Should not raise
        await logger.log(sample_event)

    @pytest.mark.asyncio
    async def test_log_calls_all_handlers(self, logger: AuditLogger, sample_event: AuditEvent):
        """Should call all registered handlers."""
        handler1 = AsyncMock(spec=AuditHandler)
        handler2 = AsyncMock(spec=AuditHandler)

        logger.add_handler(handler1)
        logger.add_handler(handler2)

        await logger.log(sample_event)

        handler1.handle.assert_called_once_with(sample_event)
        handler2.handle.assert_called_once_with(sample_event)

    @pytest.mark.asyncio
    async def test_handler_error_doesnt_stop_others(
        self, logger: AuditLogger, sample_event: AuditEvent
    ):
        """Error in one handler shouldn't prevent others from running."""
        handler1 = AsyncMock(spec=AuditHandler)
        handler1.handle.side_effect = Exception("Handler 1 error")
        handler2 = AsyncMock(spec=AuditHandler)

        logger.add_handler(handler1)
        logger.add_handler(handler2)

        await logger.log(sample_event)

        # Handler 2 should still be called
        handler2.handle.assert_called_once_with(sample_event)

    def test_add_and_remove_handler(self, logger: AuditLogger):
        """Should add and remove handlers."""
        handler = AsyncMock(spec=AuditHandler)

        logger.add_handler(handler)
        assert handler in logger._handlers

        logger.remove_handler(handler)
        assert handler not in logger._handlers

    @pytest.mark.asyncio
    async def test_close_calls_all_handlers(self, logger: AuditLogger):
        """close should call close on all handlers."""
        handler1 = AsyncMock(spec=AuditHandler)
        handler2 = AsyncMock(spec=AuditHandler)

        logger.add_handler(handler1)
        logger.add_handler(handler2)

        await logger.close()

        handler1.close.assert_called_once()
        handler2.close.assert_called_once()


class TestLoggingAuditHandler:
    """Tests for LoggingAuditHandler."""

    @pytest.fixture
    def mock_logger(self) -> MagicMock:
        """Create a mock Python logger."""
        return MagicMock()

    @pytest.fixture
    def handler(self, mock_logger: MagicMock) -> LoggingAuditHandler:
        """Create handler with mock logger."""
        return LoggingAuditHandler(logger=mock_logger)

    @pytest.mark.asyncio
    async def test_logs_api_request(self, handler: LoggingAuditHandler, mock_logger: MagicMock):
        """Should log API request events."""
        event = AuditEvent(
            event_type=AuditEventType.API_REQUEST,
            timestamp=datetime.now(UTC),
            request_id="req-123",
            client_id=None,
            client_ip="127.0.0.1",
            endpoint="/chat",
            method="POST",
        )

        await handler.handle(event)

        mock_logger.log.assert_called_once()
        args, _kwargs = mock_logger.log.call_args
        assert args[0] == 20  # INFO level
        # Check that event type is passed as argument (will be formatted by logger)
        assert args[2] == "api.request"

    @pytest.mark.asyncio
    async def test_logs_auth_failure_as_warning(
        self, handler: LoggingAuditHandler, mock_logger: MagicMock
    ):
        """Should log auth failure events at WARNING level."""
        event = AuditEvent(
            event_type=AuditEventType.AUTH_FAILURE,
            timestamp=datetime.now(UTC),
            request_id="req-123",
            client_id=None,
            client_ip="127.0.0.1",
            endpoint="/chat",
            method="POST",
        )

        await handler.handle(event)

        args, _kwargs = mock_logger.log.call_args
        assert args[0] == 30  # WARNING level


class TestConvenienceFunctions:
    """Tests for audit convenience functions."""

    @pytest.fixture
    def mock_request(self) -> MagicMock:
        """Create a mock FastAPI request."""
        request = MagicMock()
        request.url.path = "/chat"
        request.method = "POST"
        request.client.host = "192.168.1.1"
        request.query_params = {"limit": "10"}
        request.headers = {
            "content-type": "application/json",
            "user-agent": "test-agent",
        }
        return request

    @pytest.fixture
    def sample_client(self) -> APIKeyInfo:
        """Create a sample client."""
        return APIKeyInfo(
            key_id="test-key-id",
            name="Test Key",
            tier=APIKeyTier.STANDARD,
            organization="Test Org",
            created_at=datetime.now(UTC),
            expires_at=None,
            rate_limit="50/minute",
            is_active=True,
            scopes=("chat", "search"),
            metadata={},
        )

    @pytest.mark.asyncio
    async def test_log_api_request(self, mock_request: MagicMock, sample_client: APIKeyInfo):
        """log_api_request should create correct event."""
        # Use a fresh logger to capture the event
        logger = AuditLogger()
        captured_events: list[AuditEvent] = []

        class CaptureHandler(AuditHandler):
            async def handle(self, event: AuditEvent) -> None:
                captured_events.append(event)

        logger.add_handler(CaptureHandler())

        # Temporarily replace global logger
        from requirements_graphrag_api.auth import audit

        original_logger = audit.audit_logger
        audit.audit_logger = logger

        try:
            await log_api_request(mock_request, sample_client, "req-123")

            assert len(captured_events) == 1
            event = captured_events[0]
            assert event.event_type == AuditEventType.API_REQUEST
            assert event.client_id == "test-key-id"
            assert event.endpoint == "/chat"
            assert event.method == "POST"
        finally:
            audit.audit_logger = original_logger

    @pytest.mark.asyncio
    async def test_log_api_response(self, mock_request: MagicMock, sample_client: APIKeyInfo):
        """log_api_response should create correct event."""
        logger = AuditLogger()
        captured_events: list[AuditEvent] = []

        class CaptureHandler(AuditHandler):
            async def handle(self, event: AuditEvent) -> None:
                captured_events.append(event)

        logger.add_handler(CaptureHandler())

        from requirements_graphrag_api.auth import audit

        original_logger = audit.audit_logger
        audit.audit_logger = logger

        try:
            await log_api_response(mock_request, sample_client, "req-123", 200, 150.5)

            assert len(captured_events) == 1
            event = captured_events[0]
            assert event.event_type == AuditEventType.API_RESPONSE
            assert event.status_code == 200
            assert event.duration_ms == 150.5
        finally:
            audit.audit_logger = original_logger

    @pytest.mark.asyncio
    async def test_log_api_error(self, mock_request: MagicMock):
        """log_api_error should create correct event."""
        logger = AuditLogger()
        captured_events: list[AuditEvent] = []

        class CaptureHandler(AuditHandler):
            async def handle(self, event: AuditEvent) -> None:
                captured_events.append(event)

        logger.add_handler(CaptureHandler())

        from requirements_graphrag_api.auth import audit

        original_logger = audit.audit_logger
        audit.audit_logger = logger

        try:
            await log_api_error(
                mock_request,
                None,
                "req-123",
                500,
                "internal_error",
                "Something went wrong",
            )

            assert len(captured_events) == 1
            event = captured_events[0]
            assert event.event_type == AuditEventType.API_ERROR
            assert event.status_code == 500
            assert event.details["error_type"] == "internal_error"
        finally:
            audit.audit_logger = original_logger

    @pytest.mark.asyncio
    async def test_log_auth_success(self, mock_request: MagicMock, sample_client: APIKeyInfo):
        """log_auth_success should create correct event."""
        logger = AuditLogger()
        captured_events: list[AuditEvent] = []

        class CaptureHandler(AuditHandler):
            async def handle(self, event: AuditEvent) -> None:
                captured_events.append(event)

        logger.add_handler(CaptureHandler())

        from requirements_graphrag_api.auth import audit

        original_logger = audit.audit_logger
        audit.audit_logger = logger

        try:
            await log_auth_success(mock_request, sample_client, "req-123")

            assert len(captured_events) == 1
            event = captured_events[0]
            assert event.event_type == AuditEventType.AUTH_SUCCESS
            assert event.client_id == "test-key-id"
        finally:
            audit.audit_logger = original_logger

    @pytest.mark.asyncio
    async def test_log_auth_failure(self, mock_request: MagicMock):
        """log_auth_failure should create correct event."""
        logger = AuditLogger()
        captured_events: list[AuditEvent] = []

        class CaptureHandler(AuditHandler):
            async def handle(self, event: AuditEvent) -> None:
                captured_events.append(event)

        logger.add_handler(CaptureHandler())

        from requirements_graphrag_api.auth import audit

        original_logger = audit.audit_logger
        audit.audit_logger = logger

        try:
            await log_auth_failure(mock_request, "req-123", "invalid_key")

            assert len(captured_events) == 1
            event = captured_events[0]
            assert event.event_type == AuditEventType.AUTH_FAILURE
            assert event.details["reason"] == "invalid_key"
        finally:
            audit.audit_logger = original_logger

    @pytest.mark.asyncio
    async def test_log_guardrail_event_blocked(self, mock_request: MagicMock):
        """log_guardrail_event should use BLOCKED type when blocked."""
        logger = AuditLogger()
        captured_events: list[AuditEvent] = []

        class CaptureHandler(AuditHandler):
            async def handle(self, event: AuditEvent) -> None:
                captured_events.append(event)

        logger.add_handler(CaptureHandler())

        from requirements_graphrag_api.auth import audit

        original_logger = audit.audit_logger
        audit.audit_logger = logger

        try:
            await log_guardrail_event(
                mock_request,
                None,
                "req-123",
                "prompt_injection",
                triggered=True,
                blocked=True,
                details={"risk_level": "high"},
            )

            assert len(captured_events) == 1
            event = captured_events[0]
            assert event.event_type == AuditEventType.GUARDRAIL_BLOCKED
            assert event.details["guardrail"] == "prompt_injection"
        finally:
            audit.audit_logger = original_logger

    @pytest.mark.asyncio
    async def test_log_rate_limit_exceeded(self, mock_request: MagicMock):
        """log_rate_limit_event should use EXCEEDED type when exceeded."""
        logger = AuditLogger()
        captured_events: list[AuditEvent] = []

        class CaptureHandler(AuditHandler):
            async def handle(self, event: AuditEvent) -> None:
                captured_events.append(event)

        logger.add_handler(CaptureHandler())

        from requirements_graphrag_api.auth import audit

        original_logger = audit.audit_logger
        audit.audit_logger = logger

        try:
            await log_rate_limit_event(
                mock_request,
                None,
                "req-123",
                exceeded=True,
                limit="50/minute",
                current_count=51,
            )

            assert len(captured_events) == 1
            event = captured_events[0]
            assert event.event_type == AuditEventType.RATE_LIMIT_EXCEEDED
            assert event.details["limit"] == "50/minute"
            assert event.details["current_count"] == 51
        finally:
            audit.audit_logger = original_logger
