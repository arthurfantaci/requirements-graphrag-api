"""Regression tests: raw LLM provider errors must never reach end users via SSE.

Covers the streaming.py catch-all (the critical net) and the routing fail-soft.
"""

from __future__ import annotations

import contextlib
import json
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from requirements_graphrag_api.core.agentic.streaming import stream_agentic_events

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(status_code: int, body: dict) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=body,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )


async def _collect_sse(generator) -> list[dict]:
    """Collect SSE payloads from an async SSE generator into a list of parsed dicts."""
    events = []
    async for raw_line in generator:
        stripped = raw_line.strip()
        if stripped.startswith("data: "):
            payload = stripped[len("data: ") :]
            with contextlib.suppress(json.JSONDecodeError):
                events.append(json.loads(payload))
    return events


def _make_raising_graph(exc: BaseException) -> MagicMock:
    """Return a mock graph whose astream_events raises the given exception."""

    async def _raising(*_args, **_kwargs):
        # astream_events is an async generator; raise on first iteration
        raise exc
        yield  # makes this an async generator

    graph = MagicMock()
    graph.astream_events = _raising
    return graph


def _make_initial_state(query: str = "What is requirements traceability?") -> dict:
    return {"query": query, "messages": []}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStreamingCatchAllSafety:
    """stream_agentic_events must never forward raw provider error text as SSE."""

    @pytest.mark.asyncio
    async def test_insufficient_quota_error_emits_safe_message(self) -> None:
        response = _make_response(
            429,
            {"error": {"type": "insufficient_quota", "code": "insufficient_quota"}},
        )
        exc = openai.RateLimitError(
            "You exceeded your current quota, please check your plan and billing details",
            response=response,
            body={"error": {"type": "insufficient_quota", "code": "insufficient_quota"}},
        )

        graph = _make_raising_graph(exc)
        events = await _collect_sse(
            stream_agentic_events(
                graph,
                _make_initial_state(),
                config={"configurable": {"thread_id": "t1"}},
            )
        )

        error_events = [e for e in events if "error" in e]
        assert error_events, "Expected at least one error event"

        for event in error_events:
            error_text = event["error"]
            assert "insufficient_quota" not in error_text, f"Raw quota error leaked: {error_text!r}"
            assert "Error code: 429" not in error_text, f"Raw error code leaked: {error_text!r}"
            assert "exceeded your current quota" not in error_text.lower(), (
                f"Raw quota message leaked: {error_text!r}"
            )
            assert "billing" not in error_text.lower(), f"Billing detail leaked: {error_text!r}"
            assert "check your plan" not in error_text.lower(), (
                f"Plan detail leaked: {error_text!r}"
            )

    @pytest.mark.asyncio
    async def test_insufficient_quota_error_message_is_user_friendly(self) -> None:
        response = _make_response(
            429,
            {"error": {"type": "insufficient_quota", "code": "insufficient_quota"}},
        )
        exc = openai.RateLimitError(
            "You exceeded your current quota",
            response=response,
            body={"error": {"type": "insufficient_quota", "code": "insufficient_quota"}},
        )

        graph = _make_raising_graph(exc)
        events = await _collect_sse(
            stream_agentic_events(
                graph,
                _make_initial_state(),
                config={"configurable": {"thread_id": "t2"}},
            )
        )

        error_events = [e for e in events if "error" in e]
        assert error_events

        error_text = error_events[-1]["error"]
        is_friendly = (
            "temporarily at capacity" in error_text.lower()
            or "unavailable" in error_text.lower()
            or "went wrong" in error_text.lower()
        )
        assert is_friendly, f"Expected user-friendly error message, got: {error_text!r}"

    @pytest.mark.asyncio
    async def test_authentication_error_emits_safe_message(self) -> None:
        body = {"error": {"type": "invalid_api_key", "code": "invalid_api_key"}}
        response = _make_response(401, body)
        exc = openai.AuthenticationError(
            "Incorrect API key provided: sk-real****key",
            response=response,
            body=body,
        )

        graph = _make_raising_graph(exc)
        events = await _collect_sse(
            stream_agentic_events(
                graph,
                _make_initial_state(),
                config={"configurable": {"thread_id": "t3"}},
            )
        )

        error_events = [e for e in events if "error" in e]
        assert error_events

        for event in error_events:
            error_text = event["error"]
            assert "sk-" not in error_text, f"API key fragment leaked: {error_text!r}"
            assert "invalid_api_key" not in error_text, f"Auth detail leaked: {error_text!r}"
            assert "Incorrect" not in error_text, f"Raw auth message leaked: {error_text!r}"

    @pytest.mark.asyncio
    async def test_generic_exception_emits_safe_message(self) -> None:
        exc = RuntimeError("internal connection pool exhausted at host neo4j://prod:7687")

        graph = _make_raising_graph(exc)
        events = await _collect_sse(
            stream_agentic_events(
                graph,
                _make_initial_state(),
                config={"configurable": {"thread_id": "t4"}},
            )
        )

        error_events = [e for e in events if "error" in e]
        assert error_events

        for event in error_events:
            error_text = event["error"]
            assert "neo4j://prod" not in error_text, f"Internal host leaked: {error_text!r}"
            assert "connection pool" not in error_text.lower(), (
                f"Internal detail leaked: {error_text!r}"
            )
