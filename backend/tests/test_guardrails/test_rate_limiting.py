"""Tests for rate limiting middleware."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from starlette.requests import Request

from requirements_graphrag_api.middleware.rate_limit import (
    CHAT_RATE_LIMIT,
    FEEDBACK_RATE_LIMIT,
    HEALTH_RATE_LIMIT,
    SEARCH_RATE_LIMIT,
    _parse_retry_after,
    get_rate_limit_key,
    get_rate_limiter,
)


class TestRateLimitKeyFunction:
    """Test rate limit key generation."""

    def test_uses_ip_by_default(self):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.query_params = {}
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"

        with patch(
            "requirements_graphrag_api.middleware.rate_limit.get_remote_address",
            return_value="192.168.1.1",
        ):
            key = get_rate_limit_key(mock_request)
            assert key == "192.168.1.1"

    def test_uses_bearer_token_hash_when_present(self):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"Authorization": "Bearer test-api-key-123"}
        mock_request.query_params = {}

        key = get_rate_limit_key(mock_request)
        assert key.startswith("api:")
        assert len(key) == 20  # "api:" + 16 hex chars

    def test_uses_query_param_api_key_when_present(self):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.query_params = {"api_key": "test-key"}

        key = get_rate_limit_key(mock_request)
        assert key.startswith("api:")

    def test_bearer_token_takes_precedence(self):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"Authorization": "Bearer header-key"}
        mock_request.query_params = {"api_key": "query-key"}

        key = get_rate_limit_key(mock_request)
        assert key.startswith("api:")
        # Should use header key, not query key
        # Different keys should produce different hashes


class TestRateLimiterConfiguration:
    """Test rate limiter initialization."""

    def test_limiter_singleton(self):
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        assert limiter1 is limiter2

    def test_limiter_has_key_func(self):
        limiter = get_rate_limiter()
        assert limiter._key_func is not None


class TestEndpointLimits:
    """Test endpoint-specific rate limits."""

    def test_chat_limit(self):
        assert "20" in CHAT_RATE_LIMIT
        assert "minute" in CHAT_RATE_LIMIT

    def test_search_limit(self):
        assert "60" in SEARCH_RATE_LIMIT
        assert "minute" in SEARCH_RATE_LIMIT

    def test_feedback_limit(self):
        assert "30" in FEEDBACK_RATE_LIMIT
        assert "minute" in FEEDBACK_RATE_LIMIT

    def test_health_limit(self):
        assert "120" in HEALTH_RATE_LIMIT
        assert "minute" in HEALTH_RATE_LIMIT


class TestRetryAfterParsing:
    """Test retry-after header value parsing."""

    def test_parse_per_second(self):
        result = _parse_retry_after("10 per second")
        assert result == 1

    def test_parse_per_minute(self):
        result = _parse_retry_after("20 per minute")
        assert result == 60

    def test_parse_per_hour(self):
        result = _parse_retry_after("100 per hour")
        assert result == 3600

    def test_parse_per_day(self):
        result = _parse_retry_after("1000 per day")
        assert result == 86400

    def test_parse_unknown_defaults_to_60(self):
        result = _parse_retry_after("unknown format")
        assert result == 60


@pytest.fixture
def mock_rate_limit_exception():
    """Create a mock RateLimitExceeded exception."""
    exc = MagicMock()
    exc.detail = "Rate limit exceeded: 20 per minute"
    return exc


class TestRateLimitExceededHandler:
    """Test the rate limit exceeded response handler."""

    @pytest.mark.asyncio
    async def test_handler_returns_429(self, mock_rate_limit_exception):
        from requirements_graphrag_api.middleware.rate_limit import (
            rate_limit_exceeded_handler,
        )

        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/chat"
        mock_request.state = MagicMock()
        mock_request.state.request_id = "test-123"

        with patch(
            "requirements_graphrag_api.middleware.rate_limit.get_remote_address",
            return_value="192.168.1.1",
        ):
            response = await rate_limit_exceeded_handler(mock_request, mock_rate_limit_exception)

            assert response.status_code == 429
            assert "Retry-After" in response.headers
            assert "X-RateLimit-Limit" in response.headers

    @pytest.mark.asyncio
    async def test_handler_includes_retry_info(self, mock_rate_limit_exception):
        from requirements_graphrag_api.middleware.rate_limit import (
            rate_limit_exceeded_handler,
        )

        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/chat"
        mock_request.state = MagicMock()
        mock_request.state.request_id = "test-456"

        with patch(
            "requirements_graphrag_api.middleware.rate_limit.get_remote_address",
            return_value="192.168.1.1",
        ):
            response = await rate_limit_exceeded_handler(mock_request, mock_rate_limit_exception)

            import json

            body = json.loads(response.body)
            assert "error" in body
            assert body["error"] == "rate_limit_exceeded"
            assert "retry_after" in body
            assert "message" in body


class TestRateLimitEventLogging:
    """Test that rate limit events are properly logged."""

    @pytest.mark.asyncio
    async def test_logs_rate_limit_event(self, mock_rate_limit_exception):
        from requirements_graphrag_api.middleware.rate_limit import (
            rate_limit_exceeded_handler,
        )

        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/chat"
        mock_request.state = MagicMock()
        mock_request.state.request_id = "test-789"

        with (
            patch(
                "requirements_graphrag_api.middleware.rate_limit.get_remote_address",
                return_value="192.168.1.1",
            ),
            patch(
                "requirements_graphrag_api.middleware.rate_limit.log_guardrail_event"
            ) as mock_log,
        ):
            await rate_limit_exceeded_handler(mock_request, mock_rate_limit_exception)

            mock_log.assert_called_once()
            event = mock_log.call_args[0][0]
            assert event.event_type.value == "rate_limit_exceeded"
