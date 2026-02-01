"""Admin endpoints for guardrail management and monitoring.

This module provides administrative endpoints for:
- Compliance dashboard metrics
- Guardrail configuration status
- Metrics period rotation

All endpoints require ADMIN scope for access.

Usage:
    GET /admin/guardrails/metrics - Get current guardrail metrics
    GET /admin/guardrails/history - Get historical metrics
    POST /admin/guardrails/rotate-metrics - Rotate to new metrics period
    GET /admin/guardrails/config - Get current guardrail configuration
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from requirements_graphrag_api.auth import APIKeyInfo, Scope, require_scopes
from requirements_graphrag_api.config import get_guardrail_config
from requirements_graphrag_api.guardrails.metrics import GuardrailMetrics, metrics

router = APIRouter(prefix="/admin", tags=["Admin"])

# Module-level dependency for admin scope checking
AdminClient = Annotated[APIKeyInfo | None, Depends(require_scopes(Scope.ADMIN))]


@router.get("/guardrails/metrics")
async def get_guardrail_metrics(
    client: AdminClient,
) -> dict[str, Any]:
    """Get current guardrail metrics for compliance dashboard.

    Returns aggregated metrics for the current period including:
    - Request totals (total, blocked, warned)
    - Detection counts by guardrail type
    - Performance metrics (latency)

    Requires ADMIN scope.

    Returns:
        Current metrics in structured format.
    """
    _ = client  # Dependency ensures ADMIN scope
    current = metrics.get_current_metrics()
    return current.to_dict()


@router.get("/guardrails/history")
async def get_guardrail_history(
    client: AdminClient,
    limit: int = 24,
) -> dict[str, Any]:
    """Get historical guardrail metrics.

    Returns metrics from previous periods for trend analysis.

    Args:
        client: Authenticated client info (injected by dependency).
        limit: Maximum number of periods to return (default: 24).

    Requires ADMIN scope.

    Returns:
        List of historical metrics periods.
    """
    _ = client  # Dependency ensures ADMIN scope
    history = metrics.get_history()

    # Apply limit
    if limit > 0:
        history = history[-limit:]

    return {
        "periods": [m.to_dict() for m in history],
        "count": len(history),
    }


@router.post("/guardrails/rotate-metrics")
async def rotate_metrics(
    client: AdminClient,
) -> dict[str, Any]:
    """Rotate to a new metrics period.

    Archives the current period and starts a fresh one.
    Use this for scheduled rotation (e.g., hourly via cron).

    Requires ADMIN scope.

    Returns:
        Status and the completed period's summary.
    """
    _ = client  # Dependency ensures ADMIN scope
    completed = metrics.rotate_period()
    return {
        "status": "rotated",
        "completed_period": {
            "start": completed.period_start.isoformat(),
            "end": completed.period_end.isoformat() if completed.period_end else None,
            "total_requests": completed.total_requests,
            "requests_blocked": completed.requests_blocked,
        },
    }


@router.get("/guardrails/config")
async def get_guardrail_configuration(
    client: AdminClient,
) -> dict[str, Any]:
    """Get current guardrail configuration.

    Returns the active guardrail configuration including
    feature flags and thresholds.

    Requires ADMIN scope.

    Returns:
        Current guardrail configuration.
    """
    _ = client  # Dependency ensures ADMIN scope
    config = get_guardrail_config()

    return {
        "features": {
            "prompt_injection": {
                "enabled": config.prompt_injection_enabled,
                "block_threshold": config.injection_block_threshold,
            },
            "pii_detection": {
                "enabled": config.pii_detection_enabled,
                "entities": list(config.pii_entities),
                "score_threshold": config.pii_score_threshold,
                "anonymize_type": config.pii_anonymize_type,
            },
            "rate_limiting": {
                "enabled": config.rate_limiting_enabled,
                "chat_limit": config.rate_limit_chat,
                "search_limit": config.rate_limit_search,
                "default_limit": config.rate_limit_default,
            },
        },
    }


@router.get("/guardrails/summary")
async def get_guardrail_summary(
    client: AdminClient,
) -> dict[str, Any]:
    """Get a quick summary of guardrail health.

    Returns a simplified view suitable for status dashboards.

    Requires ADMIN scope.

    Returns:
        Health summary with key indicators.
    """
    _ = client  # Dependency ensures ADMIN scope
    current = metrics.get_current_metrics()

    # Calculate health indicators
    block_rate = (
        current.requests_blocked / current.total_requests if current.total_requests > 0 else 0.0
    )

    # Health status based on block rate
    if block_rate > 0.2:
        health_status = "elevated"  # >20% blocked is concerning
    elif block_rate > 0.05:
        health_status = "normal"  # 5-20% is normal
    else:
        health_status = "good"  # <5% is great

    # Performance status based on latency
    if current.avg_guardrail_latency_ms > 200:
        performance_status = "slow"
    elif current.avg_guardrail_latency_ms > 100:
        performance_status = "moderate"
    else:
        performance_status = "fast"

    return {
        "health_status": health_status,
        "performance_status": performance_status,
        "period_start": current.period_start.isoformat(),
        "totals": {
            "requests": current.total_requests,
            "blocked": current.requests_blocked,
            "warned": current.requests_warned,
        },
        "block_rate": round(block_rate, 4),
        "avg_latency_ms": round(current.avg_guardrail_latency_ms, 2),
        "top_issues": _get_top_issues(current),
    }


def _get_top_issues(m: GuardrailMetrics) -> list[dict[str, Any]]:
    """Extract top issues from metrics for dashboard.

    Args:
        m: GuardrailMetrics instance.

    Returns:
        List of top issue types sorted by count.
    """
    issues = [
        {"type": "prompt_injection", "count": m.prompt_injection_blocked},
        {"type": "pii_detection", "count": m.pii_detected},
        {"type": "toxicity", "count": m.toxicity_blocked},
        {"type": "rate_limit", "count": m.rate_limit_exceeded},
        {"type": "topic_violation", "count": m.topic_out_of_scope},
        {"type": "hallucination", "count": m.hallucination_warnings},
        {"type": "size_exceeded", "count": m.request_size_exceeded},
        {"type": "timeout", "count": m.request_timeout},
    ]

    # Filter out zero counts and sort by count descending
    issues = [i for i in issues if i["count"] > 0]
    issues.sort(key=lambda x: x["count"], reverse=True)

    return issues[:5]  # Top 5 issues
