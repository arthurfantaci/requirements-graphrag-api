"""User feedback endpoint for collecting response quality ratings.

This endpoint allows the frontend to submit user feedback (thumbs up/down)
which is then forwarded to LangSmith for tracking and analysis.

Feedback is correlated with LangSmith runs via run_id, enabling:
- Quality monitoring and dashboards
- Annotation queue population for human review
- Dataset generation for prompt optimization
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from requirements_graphrag_api.guardrails import detect_and_redact_pii
from requirements_graphrag_api.middleware.timeout import TIMEOUTS, with_timeout

logger = logging.getLogger(__name__)

router = APIRouter()


class FeedbackRequest(BaseModel):
    """Request body for feedback submission."""

    run_id: str = Field(
        ...,
        min_length=1,
        description="LangSmith run ID to associate feedback with",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Feedback score: 0 for negative (thumbs down), 1 for positive (thumbs up)",
    )
    category: str | None = Field(
        default=None,
        description="Optional feedback category (e.g., 'incorrect', 'missing', 'irrelevant')",
    )
    correction: str | None = Field(
        default=None,
        description="Optional user-provided correction for the answer",
    )
    comment: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional additional comment",
    )
    message_id: str | None = Field(
        default=None,
        description="Optional frontend message ID for correlation",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Optional conversation ID for context",
    )


class FeedbackResponse(BaseModel):
    """Response body for feedback submission."""

    status: str = Field(
        default="received",
        description="Status of feedback submission",
    )
    feedback_id: str | None = Field(
        default=None,
        description="LangSmith feedback ID if successfully created",
    )


@router.post("/feedback", response_model=FeedbackResponse)
@with_timeout(TIMEOUTS["feedback"])
async def submit_feedback(
    request: Request,
    body: FeedbackRequest,
) -> FeedbackResponse:
    """Submit user feedback for a chat response.

    Feedback is sent to LangSmith and associated with the original run.
    This enables quality monitoring, annotation queues, and prompt optimization.

    **Feedback Scores:**
    - `1.0` = Thumbs up (positive/helpful)
    - `0.0` = Thumbs down (negative/not helpful)

    **Optional Fields:**
    - `category`: Type of issue (for negative feedback)
    - `correction`: User-provided correct answer
    - `comment`: Additional context

    Args:
        request: FastAPI request object.
        body: Feedback request body.

    Returns:
        FeedbackResponse with status and optional feedback ID.

    Raises:
        HTTPException: If LangSmith is not configured or feedback submission fails.
    """
    # Check if LangSmith is configured
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        logger.warning("Feedback received but LangSmith not configured - storing locally only")
        # Still return success - we don't want to break the UX if tracing is disabled
        return FeedbackResponse(status="received_local")

    try:
        from langsmith import Client

        client = Client()

        # Redact PII from free-text fields before sending to LangSmith
        safe_comment = body.comment
        safe_correction = body.correction
        if body.comment:
            pii_result = detect_and_redact_pii(body.comment)
            if pii_result.contains_pii:
                safe_comment = pii_result.anonymized_text
                logger.info(
                    "PII redacted from feedback comment: %d entities", pii_result.entity_count
                )
        if body.correction:
            pii_result = detect_and_redact_pii(body.correction)
            if pii_result.contains_pii:
                safe_correction = pii_result.anonymized_text
                logger.info(
                    "PII redacted from feedback correction: %d entities", pii_result.entity_count
                )

        # Determine feedback value as a simple string for charting/filtering
        # LangSmith UI cannot display dict values in charts (shows as [object Object])
        feedback_value = (
            body.category if body.category else ("positive" if body.score >= 0.5 else "negative")
        )

        # Build comment string with all metadata for human review
        comment_parts: list[str] = []
        if safe_comment:
            comment_parts.append(safe_comment)
        if body.category:
            comment_parts.append(f"Category: {body.category}")
        if safe_correction:
            comment_parts.append(f"Correction: {safe_correction}")
        if body.message_id:
            comment_parts.append(f"Message ID: {body.message_id}")
        if body.conversation_id:
            comment_parts.append(f"Conversation ID: {body.conversation_id}")

        comment = " | ".join(comment_parts) if comment_parts else None

        # Create feedback in LangSmith
        # - score: numeric value (0.0-1.0) for quantitative metrics
        # - value: simple string for categorical filtering in charts
        # - comment: detailed metadata for human review
        # - correction: structured correction dict (if provided)
        feedback = client.create_feedback(
            run_id=body.run_id,
            key="user-feedback",
            score=body.score,
            comment=comment,
            value=feedback_value,
            correction={"text": safe_correction} if safe_correction else None,
        )

        feedback_id = str(feedback.id) if feedback else None

        # Log for monitoring
        feedback_type = "positive" if body.score >= 0.5 else "negative"
        logger.info(
            "User feedback submitted: run_id=%s, type=%s, category=%s",
            body.run_id,
            feedback_type,
            body.category,
        )

        # If negative feedback, consider adding to annotation queue
        # This is commented out for now - enable when annotation queue is set up
        # if body.score < 0.5:
        #     try:
        #         client.add_runs_to_annotation_queue(
        #             queue_name="user-reported-issues",
        #             run_ids=[body.run_id],
        #         )
        #     except Exception as e:
        #         logger.warning("Failed to add to annotation queue: %s", e)

        return FeedbackResponse(status="received", feedback_id=feedback_id)

    except ImportError:
        logger.error("langsmith package not installed")
        raise HTTPException(
            status_code=500,
            detail="Feedback system not available - langsmith not installed",
        ) from None
    except Exception:
        logger.exception("Failed to submit feedback to LangSmith")
        # Don't expose internal errors to the client
        raise HTTPException(
            status_code=500,
            detail="Failed to submit feedback",
        ) from None
