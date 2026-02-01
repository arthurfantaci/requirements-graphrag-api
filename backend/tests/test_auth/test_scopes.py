"""Tests for scope-based authorization."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from requirements_graphrag_api.auth.api_key import APIKeyInfo, APIKeyTier
from requirements_graphrag_api.auth.scopes import (
    DEFAULT_TIER_SCOPES,
    ENDPOINT_SCOPES,
    Scope,
    ScopeChecker,
    check_scopes,
    require_scopes,
)


class TestScopeEnum:
    """Tests for the Scope enum."""

    def test_scope_values(self):
        """Verify scope string values."""
        assert Scope.CHAT == "chat"
        assert Scope.SEARCH == "search"
        assert Scope.FEEDBACK == "feedback"
        assert Scope.ADMIN == "admin"

    def test_all_scopes_defined(self):
        """Ensure all expected scopes are defined."""
        scopes = list(Scope)
        assert len(scopes) == 4


class TestDefaultTierScopes:
    """Tests for default tier scope mappings."""

    def test_free_tier_scopes(self):
        """Free tier should have basic scopes."""
        scopes = DEFAULT_TIER_SCOPES["free"]
        assert Scope.CHAT in scopes
        assert Scope.SEARCH in scopes
        assert Scope.FEEDBACK not in scopes
        assert Scope.ADMIN not in scopes

    def test_standard_tier_scopes(self):
        """Standard tier should have chat, search, feedback."""
        scopes = DEFAULT_TIER_SCOPES["standard"]
        assert Scope.CHAT in scopes
        assert Scope.SEARCH in scopes
        assert Scope.FEEDBACK in scopes
        assert Scope.ADMIN not in scopes

    def test_enterprise_tier_scopes(self):
        """Enterprise tier should have all scopes."""
        scopes = DEFAULT_TIER_SCOPES["enterprise"]
        assert Scope.CHAT in scopes
        assert Scope.SEARCH in scopes
        assert Scope.FEEDBACK in scopes
        assert Scope.ADMIN in scopes


class TestEndpointScopes:
    """Tests for endpoint scope mappings."""

    def test_chat_endpoint_requires_chat_scope(self):
        """Chat endpoint should require chat scope."""
        assert Scope.CHAT in ENDPOINT_SCOPES["/chat"]

    def test_search_endpoints_require_search_scope(self):
        """Search endpoints should require search scope."""
        assert Scope.SEARCH in ENDPOINT_SCOPES["/search/hybrid"]
        assert Scope.SEARCH in ENDPOINT_SCOPES["/search/vector"]
        assert Scope.SEARCH in ENDPOINT_SCOPES["/search/graph"]

    def test_feedback_endpoint_requires_feedback_scope(self):
        """Feedback endpoint should require feedback scope."""
        assert Scope.FEEDBACK in ENDPOINT_SCOPES["/feedback"]


class TestCheckScopes:
    """Tests for the check_scopes function."""

    def test_has_all_required_scopes(self):
        """Should return True when client has all required scopes."""
        client_scopes = ("chat", "search", "feedback")
        required = (Scope.CHAT, Scope.SEARCH)
        has_scopes, missing = check_scopes(client_scopes, required)
        assert has_scopes is True
        assert missing == set()

    def test_missing_required_scopes(self):
        """Should return False and list missing scopes."""
        client_scopes = ("chat",)
        required = (Scope.CHAT, Scope.SEARCH)
        has_scopes, missing = check_scopes(client_scopes, required)
        assert has_scopes is False
        assert "search" in missing

    def test_empty_required_scopes(self):
        """Should return True when no scopes required."""
        client_scopes = ("chat",)
        required: tuple[Scope, ...] = ()
        has_scopes, missing = check_scopes(client_scopes, required)
        assert has_scopes is True
        assert missing == set()

    def test_empty_client_scopes(self):
        """Should return False when client has no scopes."""
        client_scopes: tuple[str, ...] = ()
        required = (Scope.CHAT,)
        has_scopes, missing = check_scopes(client_scopes, required)
        assert has_scopes is False
        assert "chat" in missing


class TestScopeChecker:
    """Tests for the ScopeChecker dependency."""

    @pytest.fixture
    def mock_request_with_client(self) -> MagicMock:
        """Create a mock request with authenticated client."""
        request = MagicMock()
        request.state.client = APIKeyInfo(
            key_id="test-key",
            name="Test Key",
            tier=APIKeyTier.STANDARD,
            organization="Test Org",
            created_at=datetime.now(UTC),
            expires_at=None,
            rate_limit="50/minute",
            is_active=True,
            scopes=("chat", "search", "feedback"),
            metadata={},
        )
        return request

    @pytest.fixture
    def mock_request_anonymous(self) -> MagicMock:
        """Create a mock request with no client."""
        request = MagicMock()
        request.state.client = None
        return request

    @pytest.mark.asyncio
    async def test_allows_when_has_scope(self, mock_request_with_client: MagicMock):
        """Should allow when client has required scope."""
        checker = ScopeChecker(Scope.CHAT)
        result = await checker(mock_request_with_client)
        assert result is not None
        assert result.key_id == "test-key"

    @pytest.mark.asyncio
    async def test_allows_multiple_scopes(self, mock_request_with_client: MagicMock):
        """Should allow when client has all required scopes."""
        checker = ScopeChecker(Scope.CHAT, Scope.SEARCH)
        result = await checker(mock_request_with_client)
        assert result is not None

    @pytest.mark.asyncio
    async def test_denies_missing_scope(self, mock_request_with_client: MagicMock):
        """Should deny when client lacks required scope."""
        checker = ScopeChecker(Scope.ADMIN)  # Client doesn't have admin
        with pytest.raises(HTTPException) as exc_info:
            await checker(mock_request_with_client)
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail["error"] == "insufficient_scope"
        assert "admin" in exc_info.value.detail["message"]

    @pytest.mark.asyncio
    async def test_denies_anonymous_when_required(self, mock_request_anonymous: MagicMock):
        """Should deny anonymous when scopes required."""
        checker = ScopeChecker(Scope.CHAT)
        with pytest.raises(HTTPException) as exc_info:
            await checker(mock_request_anonymous)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"] == "authentication_required"

    @pytest.mark.asyncio
    async def test_allows_anonymous_when_configured(self, mock_request_anonymous: MagicMock):
        """Should allow anonymous when allow_anonymous=True."""
        checker = ScopeChecker(Scope.CHAT, allow_anonymous=True)
        result = await checker(mock_request_anonymous)
        assert result is None


class TestRequireScopesDependency:
    """Tests for the require_scopes convenience function."""

    @pytest.fixture
    def mock_request_with_client(self) -> MagicMock:
        """Create a mock request with authenticated client."""
        request = MagicMock()
        request.state.client = APIKeyInfo(
            key_id="test-key",
            name="Test Key",
            tier=APIKeyTier.STANDARD,
            organization=None,
            created_at=datetime.now(UTC),
            expires_at=None,
            rate_limit="50/minute",
            is_active=True,
            scopes=("chat", "search"),
            metadata={},
        )
        return request

    @pytest.mark.asyncio
    async def test_returns_scope_checker(self, mock_request_with_client: MagicMock):
        """Should return a callable ScopeChecker."""
        checker = require_scopes(Scope.CHAT)
        assert isinstance(checker, ScopeChecker)

    @pytest.mark.asyncio
    async def test_checker_works_correctly(self, mock_request_with_client: MagicMock):
        """Returned checker should work correctly."""
        checker = require_scopes(Scope.CHAT, Scope.SEARCH)
        result = await checker(mock_request_with_client)
        assert result is not None
        assert result.key_id == "test-key"
