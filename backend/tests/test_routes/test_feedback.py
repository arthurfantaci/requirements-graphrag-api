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


# =========================================================================
# E. Comment / Source-Info Separation (Phase 4 — LangSmith feedback capture)
#
# NOTE on test scope: these mocked tests assert what the route passes to
# `client.create_feedback`. They cannot detect that the LangSmith server
# silently drops the `extra=` kwarg (it does — confirmed 2026-05-22 via a
# round-trip probe in production). The metadata must travel via
# `source_info=` paired with `feedback_source_type="api"`, which lands on
# `feedback_source.metadata` on the stored Feedback record.
# =========================================================================


class TestCommentAndSourceInfo:
    """User free-text comment passes AS-IS; system metadata lives in `source_info`."""

    def test_user_comment_passes_through_unchanged(
        self, client: TestClient, mock_langsmith_client: MagicMock
    ) -> None:
        """The user's comment text reaches LangSmith's `comment` kwarg unchanged."""
        user_text = "Sources panel disagreed with the citation list in the response."
        response = client.post(
            "/feedback",
            json={"run_id": "run-1", "score": 0.0, "comment": user_text},
        )

        assert response.status_code == 200
        call = mock_langsmith_client.create_feedback.call_args
        assert call.kwargs["comment"] == user_text
        # No metadata bleed
        assert "|" not in (call.kwargs["comment"] or "")
        assert "Category:" not in (call.kwargs["comment"] or "")
        assert "Intent:" not in (call.kwargs["comment"] or "")

    def test_metadata_goes_to_source_info_not_comment(
        self, client: TestClient, mock_langsmith_client: MagicMock
    ) -> None:
        """All system metadata lands in `source_info`, not in `comment`."""
        response = client.post(
            "/feedback",
            json={
                "run_id": "run-1",
                "score": 0.0,
                "comment": "Free text only",
                "category": "incorrect",
                "intent": "explanatory",
                "message_id": "msg-abc",
                "conversation_id": "conv-xyz",
                "trace_id": "trace-123",
            },
        )

        assert response.status_code == 200
        call = mock_langsmith_client.create_feedback.call_args
        assert call.kwargs["comment"] == "Free text only"
        assert call.kwargs["feedback_source_type"] == "api"
        assert call.kwargs["source_info"] == {
            "category": "incorrect",
            "intent": "explanatory",
            "message_id": "msg-abc",
            "conversation_id": "conv-xyz",
            "trace_id": "trace-123",
        }

    def test_empty_comment_with_metadata_yields_none_comment(
        self, client: TestClient, mock_langsmith_client: MagicMock
    ) -> None:
        """No user text + metadata → `comment=None`, metadata still flows via `source_info`."""
        response = client.post(
            "/feedback",
            json={
                "run_id": "run-1",
                "score": 0.0,
                "category": "incorrect",
                "intent": "structured",
            },
        )

        assert response.status_code == 200
        call = mock_langsmith_client.create_feedback.call_args
        assert call.kwargs["comment"] is None
        assert call.kwargs["source_info"] == {"category": "incorrect", "intent": "structured"}

    def test_no_metadata_no_source_info(
        self, client: TestClient, mock_langsmith_client: MagicMock
    ) -> None:
        """Comment-only submission produces `source_info=None` (not an empty dict)."""
        response = client.post(
            "/feedback",
            json={"run_id": "run-1", "score": 1.0, "comment": "Great response"},
        )

        assert response.status_code == 200
        call = mock_langsmith_client.create_feedback.call_args
        assert call.kwargs["comment"] == "Great response"
        assert call.kwargs["source_info"] is None

    def test_correction_remains_structured(
        self, client: TestClient, mock_langsmith_client: MagicMock
    ) -> None:
        """Correction text stays in the structured `correction` kwarg, not folded into comment."""
        response = client.post(
            "/feedback",
            json={
                "run_id": "run-1",
                "score": 0.0,
                "comment": "Wrong answer",
                "correction": "The correct answer is X",
            },
        )

        assert response.status_code == 200
        call = mock_langsmith_client.create_feedback.call_args
        assert call.kwargs["comment"] == "Wrong answer"
        assert call.kwargs["correction"] == {"text": "The correct answer is X"}
        # Correction must NOT leak into comment
        assert "The correct answer is X" not in call.kwargs["comment"]

    def test_pii_redacted_comment_is_what_reaches_langsmith(
        self, mock_app: FastAPI, mock_langsmith_client: MagicMock
    ) -> None:
        """When PII is detected, the anonymized text (not the raw comment) is sent to LangSmith."""
        raw_comment = "Email me at user@example.com about this."
        redacted_comment = "Email me at <EMAIL> about this."

        with (
            patch.dict("os.environ", {"LANGSMITH_API_KEY": "test-key"}),
            patch("requirements_graphrag_api.routes.feedback.detect_and_redact_pii") as mock_pii,
            patch("langsmith.Client", return_value=mock_langsmith_client),
        ):
            pii_result = MagicMock()
            pii_result.contains_pii = True
            pii_result.anonymized_text = redacted_comment
            pii_result.entity_count = 1
            mock_pii.return_value = pii_result

            test_client = TestClient(mock_app)
            response = test_client.post(
                "/feedback",
                json={"run_id": "run-1", "score": 0.0, "comment": raw_comment},
            )

        assert response.status_code == 200
        call = mock_langsmith_client.create_feedback.call_args
        assert call.kwargs["comment"] == redacted_comment
        assert raw_comment not in (call.kwargs["comment"] or "")
