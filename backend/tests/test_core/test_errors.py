"""Unit tests for core/errors.py — LLM provider error classifier."""

from __future__ import annotations

import httpx
import openai

from requirements_graphrag_api.core.errors import classify_llm_error


def _make_response(status_code: int, body: dict) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=body,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )


class TestClassifyLlmError:
    """Tests for the classify_llm_error classifier mapping table."""

    def test_rate_limit_insufficient_quota_returns_capacity(self) -> None:
        response = _make_response(
            429,
            {"error": {"type": "insufficient_quota", "code": "insufficient_quota"}},
        )
        exc = openai.RateLimitError(
            "You exceeded your current quota",
            response=response,
            body={"error": {"type": "insufficient_quota", "code": "insufficient_quota"}},
        )
        message, category = classify_llm_error(exc)

        assert category == "capacity"
        assert "insufficient_quota" not in message
        assert "Error code" not in message
        assert "429" not in message
        assert message == "The assistant is temporarily at capacity. Please try again shortly."

    def test_rate_limit_generic_returns_capacity(self) -> None:
        response = _make_response(
            429, {"error": {"type": "requests", "code": "rate_limit_exceeded"}}
        )
        exc = openai.RateLimitError(
            "Rate limit exceeded",
            response=response,
            body={"error": {"type": "requests", "code": "rate_limit_exceeded"}},
        )
        message, category = classify_llm_error(exc)

        assert category == "capacity"
        assert "rate_limit_exceeded" not in message
        assert "Rate limit" not in message

    def test_authentication_error_returns_config(self) -> None:
        response = _make_response(401, {"error": {"type": "invalid_api_key"}})
        exc = openai.AuthenticationError(
            "Invalid API key",
            response=response,
            body={"error": {"type": "invalid_api_key"}},
        )
        message, category = classify_llm_error(exc)

        assert category == "config"
        assert "key" not in message.lower()
        assert "api" not in message.lower()
        assert message == "The assistant is temporarily unavailable."

    def test_permission_denied_returns_config(self) -> None:
        response = _make_response(403, {"error": {"type": "permission_denied"}})
        exc = openai.PermissionDeniedError(
            "Permission denied",
            response=response,
            body={"error": {"type": "permission_denied"}},
        )
        message, category = classify_llm_error(exc)

        assert category == "config"
        assert message == "The assistant is temporarily unavailable."

    def test_unknown_exception_returns_unknown(self) -> None:
        exc = RuntimeError("something unexpected")
        message, category = classify_llm_error(exc)

        assert category == "unknown"
        assert "unexpected" not in message
        assert message == "Something went wrong generating a response."

    def test_connection_error_returns_unknown(self) -> None:
        response = _make_response(500, {"error": {"type": "server_error"}})
        exc = openai.InternalServerError(
            "Internal server error",
            response=response,
            body={"error": {"type": "server_error"}},
        )
        message, category = classify_llm_error(exc)

        assert category == "unknown"
        assert "Internal" not in message
        assert "server" not in message.lower()

    def test_safe_messages_never_contain_raw_error_text(self) -> None:
        raw_text = "You exceeded your current quota, please check your plan"
        response = _make_response(
            429,
            {"error": {"message": raw_text, "type": "insufficient_quota"}},
        )
        exc = openai.RateLimitError(
            raw_text,
            response=response,
            body={"error": {"message": raw_text, "type": "insufficient_quota"}},
        )
        message, _category = classify_llm_error(exc)

        assert raw_text not in message
        assert "quota" not in message
