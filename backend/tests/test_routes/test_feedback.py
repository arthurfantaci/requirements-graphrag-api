"""Tests for feedback endpoint with annotation queue routing.

Tests cover: model validation, queue ID resolution/caching,
intent-based queue routing, and per-dimension rubric scores.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from requirements_graphrag_api.routes.feedback import (
    _queue_id_cache,
    _resolve_queue_id,
    router,
)


@pytest.fixture(autouse=True)
def _clear_queue_cache():
    """Clear the module-level queue ID cache between tests."""
    _queue_id_cache.clear()
    yield
    _queue_id_cache.clear()


@pytest.fixture
def mock_app() -> FastAPI:
    """Create a test FastAPI app with feedback router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def mock_langsmith_client():
    """Create a mock LangSmith Client."""
    client = MagicMock()
    # Default: create_feedback returns an object with .id
    feedback_obj = MagicMock()
    feedback_obj.id = "fb-123"
    client.create_feedback.return_value = feedback_obj

    # Default: list_annotation_queues returns empty
    client.list_annotation_queues.return_value = []

    return client


@pytest.fixture
def client(mock_app: FastAPI, mock_langsmith_client: MagicMock) -> TestClient:
    """Create a test client with mocked LangSmith."""
    with (
        patch.dict("os.environ", {"LANGSMITH_API_KEY": "test-key"}),
        patch("requirements_graphrag_api.routes.feedback.detect_and_redact_pii") as mock_pii,
        patch(
            "langsmith.Client",
            return_value=mock_langsmith_client,
        ),
    ):
        # PII detection returns clean text by default
        pii_result = MagicMock()
        pii_result.contains_pii = False
        mock_pii.return_value = pii_result

        yield TestClient(mock_app)


# =========================================================================
# A. Model Validation
# =========================================================================


class TestModelValidation:
    """Tests for FeedbackRequest model fields."""

    def test_new_fields_default_to_none(self, client: TestClient) -> None:
        """Intent and rubric_scores default to None (backward compat)."""
        response = client.post(
            "/feedback",
            json={"run_id": "run-1", "score": 1.0},
        )
        assert response.status_code == 200

    def test_backward_compatible_without_intent(self, client: TestClient) -> None:
        """Existing payloads without intent/rubric_scores still work."""
        response = client.post(
            "/feedback",
            json={
                "run_id": "run-1",
                "score": 0.0,
                "category": "incorrect",
                "comment": "Wrong answer",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "received"

    def test_intent_field_accepted(self, client: TestClient) -> None:
        """Intent field is accepted and processed."""
        response = client.post(
            "/feedback",
            json={"run_id": "run-1", "score": 1.0, "intent": "explanatory"},
        )
        assert response.status_code == 200

    def test_rubric_scores_field_accepted(self, client: TestClient) -> None:
        """Rubric scores field is accepted and processed."""
        response = client.post(
            "/feedback",
            json={
                "run_id": "run-1",
                "score": 1.0,
                "rubric_scores": {"completeness": 0.8, "accuracy": 0.9},
            },
        )
        assert response.status_code == 200

    def test_all_new_fields_together(self, client: TestClient) -> None:
        """All new fields work together in a single request."""
        response = client.post(
            "/feedback",
            json={
                "run_id": "run-1",
                "score": 0.0,
                "intent": "structured",
                "rubric_scores": {"completeness": 0.3},
                "comment": "Bad query",
            },
        )
        assert response.status_code == 200


# =========================================================================
# B. Queue ID Resolution
# =========================================================================


class TestQueueIdResolution:
    """Tests for _resolve_queue_id helper."""

    def test_resolves_name_to_id(self) -> None:
        """Resolves queue name to UUID via list_annotation_queues."""
        client = MagicMock()
        queue_obj = MagicMock()
        queue_obj.id = "uuid-abc"
        client.list_annotation_queues.return_value = [queue_obj]

        result = _resolve_queue_id(client, "review-explanatory")

        assert result == "uuid-abc"
        client.list_annotation_queues.assert_called_once_with(name="review-explanatory")

    def test_caches_resolved_id(self) -> None:
        """Second call returns cached ID without API call."""
        client = MagicMock()
        queue_obj = MagicMock()
        queue_obj.id = "uuid-abc"
        client.list_annotation_queues.return_value = [queue_obj]

        # First call — hits API
        _resolve_queue_id(client, "review-explanatory")
        # Second call — from cache
        result = _resolve_queue_id(client, "review-explanatory")

        assert result == "uuid-abc"
        assert client.list_annotation_queues.call_count == 1

    def test_returns_none_for_missing_queue(self) -> None:
        """Returns None when queue doesn't exist."""
        client = MagicMock()
        client.list_annotation_queues.return_value = []

        result = _resolve_queue_id(client, "nonexistent-queue")

        assert result is None

    def test_returns_none_on_api_error(self) -> None:
        """Returns None and logs warning on API error."""
        client = MagicMock()
        client.list_annotation_queues.side_effect = Exception("API error")

        result = _resolve_queue_id(client, "review-explanatory")

        assert result is None


# =========================================================================
# C. Queue Routing
# =========================================================================


class TestQueueRouting:
    """Tests for annotation queue routing on negative feedback."""

    def test_negative_with_intent_routes_to_correct_queue(
        self, mock_app: FastAPI, mock_langsmith_client: MagicMock
    ) -> None:
        """Negative feedback with intent routes to intent-specific queue."""
        # Set up queue resolution
        queue_obj = MagicMock()
        queue_obj.id = "uuid-explanatory"
        mock_langsmith_client.list_annotation_queues.return_value = [queue_obj]

        with (
            patch.dict("os.environ", {"LANGSMITH_API_KEY": "test-key"}),
            patch("requirements_graphrag_api.routes.feedback.detect_and_redact_pii") as mock_pii,
            patch(
                "langsmith.Client",
                return_value=mock_langsmith_client,
            ),
        ):
            pii_result = MagicMock()
            pii_result.contains_pii = False
            mock_pii.return_value = pii_result

            test_client = TestClient(mock_app)
            response = test_client.post(
                "/feedback",
                json={"run_id": "run-1", "score": 0.0, "intent": "explanatory"},
            )

        assert response.status_code == 200
        mock_langsmith_client.add_runs_to_annotation_queue.assert_called_once_with(
            queue_id="uuid-explanatory",
            run_ids=["run-1"],
        )

    def test_negative_without_intent_routes_to_user_reported(
        self, mock_app: FastAPI, mock_langsmith_client: MagicMock
    ) -> None:
        """Negative feedback without intent routes to user-reported queue."""
        queue_obj = MagicMock()
        queue_obj.id = "uuid-user-reported"
        mock_langsmith_client.list_annotation_queues.return_value = [queue_obj]

        with (
            patch.dict("os.environ", {"LANGSMITH_API_KEY": "test-key"}),
            patch("requirements_graphrag_api.routes.feedback.detect_and_redact_pii") as mock_pii,
            patch(
                "langsmith.Client",
                return_value=mock_langsmith_client,
            ),
        ):
            pii_result = MagicMock()
            pii_result.contains_pii = False
            mock_pii.return_value = pii_result

            test_client = TestClient(mock_app)
            response = test_client.post(
                "/feedback",
                json={"run_id": "run-1", "score": 0.0},
            )

        assert response.status_code == 200
        mock_langsmith_client.list_annotation_queues.assert_called_once_with(
            name="user-reported-issues"
        )

    def test_positive_feedback_no_queue_call(
        self, mock_app: FastAPI, mock_langsmith_client: MagicMock
    ) -> None:
        """Positive feedback does NOT trigger queue routing."""
        with (
            patch.dict("os.environ", {"LANGSMITH_API_KEY": "test-key"}),
            patch("requirements_graphrag_api.routes.feedback.detect_and_redact_pii") as mock_pii,
            patch(
                "langsmith.Client",
                return_value=mock_langsmith_client,
            ),
        ):
            pii_result = MagicMock()
            pii_result.contains_pii = False
            mock_pii.return_value = pii_result

            test_client = TestClient(mock_app)
            response = test_client.post(
                "/feedback",
                json={"run_id": "run-1", "score": 1.0, "intent": "explanatory"},
            )

        assert response.status_code == 200
        mock_langsmith_client.add_runs_to_annotation_queue.assert_not_called()
        mock_langsmith_client.list_annotation_queues.assert_not_called()


# =========================================================================
# D. Rubric Scores
# =========================================================================


class TestRubricScores:
    """Tests for per-dimension rubric score submission."""

    def test_rubric_scores_create_separate_feedback(
        self, mock_app: FastAPI, mock_langsmith_client: MagicMock
    ) -> None:
        """Each rubric dimension creates a separate feedback entry."""
        with (
            patch.dict("os.environ", {"LANGSMITH_API_KEY": "test-key"}),
            patch("requirements_graphrag_api.routes.feedback.detect_and_redact_pii") as mock_pii,
            patch(
                "langsmith.Client",
                return_value=mock_langsmith_client,
            ),
        ):
            pii_result = MagicMock()
            pii_result.contains_pii = False
            mock_pii.return_value = pii_result

            test_client = TestClient(mock_app)
            response = test_client.post(
                "/feedback",
                json={
                    "run_id": "run-1",
                    "score": 1.0,
                    "rubric_scores": {"completeness": 0.8, "accuracy": 0.9},
                },
            )

        assert response.status_code == 200
        # 1 main feedback + 2 rubric dimensions = 3 total create_feedback calls
        assert mock_langsmith_client.create_feedback.call_count == 3

        # Check rubric calls were made with correct keys
        call_keys = [
            call.kwargs.get("key") or call[1].get("key", "")
            for call in mock_langsmith_client.create_feedback.call_args_list
        ]
        assert "user-completeness" in call_keys
        assert "user-accuracy" in call_keys

    def test_rubric_failure_does_not_break_endpoint(
        self, mock_app: FastAPI, mock_langsmith_client: MagicMock
    ) -> None:
        """Rubric score failure doesn't break the overall endpoint."""
        # First call succeeds (main feedback), subsequent calls fail
        feedback_obj = MagicMock()
        feedback_obj.id = "fb-123"
        mock_langsmith_client.create_feedback.side_effect = [
            feedback_obj,  # main feedback
            Exception("Rubric failed"),  # first rubric dimension
        ]

        with (
            patch.dict("os.environ", {"LANGSMITH_API_KEY": "test-key"}),
            patch("requirements_graphrag_api.routes.feedback.detect_and_redact_pii") as mock_pii,
            patch(
                "langsmith.Client",
                return_value=mock_langsmith_client,
            ),
        ):
            pii_result = MagicMock()
            pii_result.contains_pii = False
            mock_pii.return_value = pii_result

            test_client = TestClient(mock_app)
            response = test_client.post(
                "/feedback",
                json={
                    "run_id": "run-1",
                    "score": 1.0,
                    "rubric_scores": {"completeness": 0.8},
                },
            )

        # Endpoint still succeeds despite rubric failure
        assert response.status_code == 200
        assert response.json()["status"] == "received"
