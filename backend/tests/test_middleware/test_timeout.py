"""Tests for request timeout middleware."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from requirements_graphrag_api.middleware.timeout import (
    TIMEOUTS,
    get_timeout_for_endpoint,
    run_with_timeout,
    with_timeout,
)


class TestTimeoutConstants:
    """Test timeout configuration constants."""

    def test_timeouts_dict_exists(self):
        assert TIMEOUTS is not None
        assert isinstance(TIMEOUTS, dict)

    def test_expected_keys_exist(self):
        expected = ["chat", "search", "cypher", "feedback", "health", "default"]
        for key in expected:
            assert key in TIMEOUTS

    def test_chat_timeout_is_longest(self):
        assert TIMEOUTS["chat"] >= TIMEOUTS["search"]
        assert TIMEOUTS["chat"] >= TIMEOUTS["health"]


class TestGetTimeoutForEndpoint:
    """Test endpoint timeout lookup."""

    def test_chat_endpoint(self):
        assert get_timeout_for_endpoint("/chat") == TIMEOUTS["chat"]
        assert get_timeout_for_endpoint("/api/chat") == TIMEOUTS["chat"]

    def test_search_endpoint(self):
        assert get_timeout_for_endpoint("/search") == TIMEOUTS["search"]
        assert get_timeout_for_endpoint("/search/vector") == TIMEOUTS["search"]
        assert get_timeout_for_endpoint("/search/hybrid") == TIMEOUTS["search"]

    def test_cypher_endpoint(self):
        assert get_timeout_for_endpoint("/cypher") == TIMEOUTS["cypher"]
        assert get_timeout_for_endpoint("/graph/query") == TIMEOUTS["cypher"]

    def test_feedback_endpoint(self):
        assert get_timeout_for_endpoint("/feedback") == TIMEOUTS["feedback"]

    def test_health_endpoint(self):
        assert get_timeout_for_endpoint("/health") == TIMEOUTS["health"]

    def test_unknown_endpoint_returns_default(self):
        assert get_timeout_for_endpoint("/unknown") == TIMEOUTS["default"]
        assert get_timeout_for_endpoint("/api/v1/something") == TIMEOUTS["default"]


class TestWithTimeoutDecorator:
    """Test the with_timeout decorator."""

    @pytest.mark.asyncio
    async def test_fast_function_completes(self):
        @with_timeout(1.0)
        async def fast_func():
            return "done"

        result = await fast_func()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_slow_function_times_out(self):
        @with_timeout(0.1)
        async def slow_func():
            await asyncio.sleep(1.0)
            return "done"

        with pytest.raises(HTTPException) as exc_info:
            await slow_func()

        assert exc_info.value.status_code == 504
        assert exc_info.value.detail["error"] == "request_timeout"
        assert exc_info.value.detail["timeout_seconds"] == 0.1

    @pytest.mark.asyncio
    async def test_preserves_function_name(self):
        @with_timeout(1.0)
        async def my_named_function():
            return "done"

        assert my_named_function.__name__ == "my_named_function"

    @pytest.mark.asyncio
    async def test_passes_arguments(self):
        @with_timeout(1.0)
        async def func_with_args(a, b, *, c=None):
            return (a, b, c)

        result = await func_with_args(1, 2, c=3)
        assert result == (1, 2, 3)

    @pytest.mark.asyncio
    async def test_exception_propagates(self):
        @with_timeout(1.0)
        async def raising_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await raising_func()


class TestRunWithTimeout:
    """Test the run_with_timeout utility function."""

    @pytest.mark.asyncio
    async def test_fast_coroutine_completes(self):
        async def fast_coro():
            return "done"

        result = await run_with_timeout(fast_coro(), timeout=1.0)
        assert result == "done"

    @pytest.mark.asyncio
    async def test_slow_coroutine_times_out(self):
        async def slow_coro():
            await asyncio.sleep(1.0)
            return "done"

        with pytest.raises(HTTPException) as exc_info:
            await run_with_timeout(slow_coro(), timeout=0.1, operation="test_op")

        assert exc_info.value.status_code == 504
        assert "test_op" in exc_info.value.detail["message"]

    @pytest.mark.asyncio
    async def test_custom_operation_name_in_error(self):
        async def slow():
            await asyncio.sleep(1.0)

        with pytest.raises(HTTPException) as exc_info:
            await run_with_timeout(slow(), timeout=0.05, operation="my_custom_operation")

        assert "my_custom_operation" in exc_info.value.detail["message"]

    @pytest.mark.asyncio
    async def test_returns_correct_type(self):
        async def return_int():
            return 42

        result = await run_with_timeout(return_int(), timeout=1.0)
        assert result == 42
        assert isinstance(result, int)


class TestTimeoutEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_zero_timeout_immediately_times_out(self):
        @with_timeout(0.0)
        async def instant():
            return "done"

        # Zero timeout should time out immediately (or nearly so)
        with pytest.raises(HTTPException) as exc_info:
            await instant()
        assert exc_info.value.status_code == 504

    @pytest.mark.asyncio
    async def test_very_short_timeout(self):
        @with_timeout(0.001)
        async def short_work():
            await asyncio.sleep(0.1)
            return "done"

        with pytest.raises(HTTPException):
            await short_work()

    @pytest.mark.asyncio
    async def test_exactly_at_timeout_boundary(self):
        # A task that takes just slightly less than the timeout should succeed
        @with_timeout(0.2)
        async def boundary_work():
            await asyncio.sleep(0.05)
            return "done"

        result = await boundary_work()
        assert result == "done"
