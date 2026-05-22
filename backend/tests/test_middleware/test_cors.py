"""Tests for CORS configuration — exact-list + regex pattern matching.

Covers the dual-source CORS origin allowlist used in api.py: `CORS_ORIGINS`
(comma-separated exact-match list) and `CORS_ORIGIN_REGEX` (regex pattern
for dynamic origins like Vercel preview branches). FastAPI's CORSMiddleware
applies the two with OR semantics — either source can authorize a request.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.testclient import TestClient


@pytest.fixture
def app_with_cors():
    """Build a FastAPI app mirroring api.py's CORS wiring."""
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://graphrag.norfolkaibi.com"],
        allow_origin_regex=r"^https://frontend-[a-zA-Z0-9-]+-norfolk-ai-bi\.vercel\.app$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    return app


@pytest.fixture
def client(app_with_cors):
    return TestClient(app_with_cors)


class TestCORSOriginAllowlist:
    """Verify exact-match list and regex pattern both authorize origins."""

    def test_exact_match_origin_allowed(self, client):
        """Origin in `allow_origins` list passes preflight."""
        response = client.options(
            "/ping",
            headers={
                "Origin": "https://graphrag.norfolkaibi.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "https://graphrag.norfolkaibi.com"

    def test_regex_match_vercel_preview_branch_alias(self, client):
        """Vercel branch-alias preview URL matches the regex."""
        origin = "https://frontend-git-feat-sidebar-ux-persistence-norfolk-ai-bi.vercel.app"
        response = client.options(
            "/ping",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin

    def test_regex_match_vercel_deployment_url(self, client):
        """Vercel per-deployment URL (no `git-` prefix) matches the regex."""
        origin = "https://frontend-325st6mmx-norfolk-ai-bi.vercel.app"
        response = client.options(
            "/ping",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin

    def test_unknown_origin_rejected(self, client):
        """Origin matching neither list nor regex receives no CORS headers."""
        response = client.options(
            "/ping",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" not in response.headers

    def test_different_org_vercel_url_rejected(self, client):
        """Vercel URL outside the `norfolk-ai-bi` org is not matched by the regex."""
        response = client.options(
            "/ping",
            headers={
                "Origin": "https://frontend-abc123-other-org.vercel.app",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" not in response.headers


class TestCORSRegexUnsetDefault:
    """When CORS_ORIGIN_REGEX is unset, middleware falls back to exact-list only."""

    def test_no_regex_means_only_exact_list_allowed(self):
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["https://graphrag.norfolkaibi.com"],
            allow_origin_regex=None,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/ping")
        async def ping():
            return {"ok": True}

        client = TestClient(app)
        response = client.options(
            "/ping",
            headers={
                "Origin": "https://frontend-git-test-norfolk-ai-bi.vercel.app",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" not in response.headers
