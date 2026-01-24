"""Prompt Catalog with LangSmith Hub integration.

Provides centralized prompt management with:
- LangSmith Hub integration for version control and collaboration
- Local fallback when Hub is unavailable
- Caching for performance optimization
- Environment-based prompt selection (dev/staging/production)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Final

from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from langsmith import Client

from requirements_graphrag_api.observability import traceable_safe
from requirements_graphrag_api.prompts.definitions import (
    PROMPT_DEFINITIONS,
    PromptDefinition,
    PromptName,
)

logger = logging.getLogger(__name__)

# Environment variables for LangSmith Hub configuration
LANGSMITH_ORG_ENV: Final[str] = "LANGSMITH_ORG"
LANGSMITH_API_KEY_ENV: Final[str] = "LANGSMITH_API_KEY"
LANGSMITH_TENANT_ID_ENV: Final[str] = "LANGSMITH_TENANT_ID"
PROMPT_ENVIRONMENT_ENV: Final[str] = "PROMPT_ENVIRONMENT"

# Default configuration
# Empty string means workspace-scoped prompts (no org prefix)
DEFAULT_ORG: Final[str] = ""
DEFAULT_CACHE_TTL: Final[int] = 300  # 5 minutes


def _create_langsmith_client() -> Client:
    """Create LangSmith Client with workspace_id for org-scoped API keys.

    Reads LANGSMITH_TENANT_ID from environment and passes it explicitly
    since the SDK doesn't auto-detect it.

    Returns:
        Configured LangSmith Client instance.
    """
    from langsmith import Client

    # SDK doesn't auto-read workspace from env, must pass explicitly
    workspace_id = os.getenv(LANGSMITH_TENANT_ID_ENV)
    if workspace_id:
        logger.info("Creating LangSmith client with workspace_id")
        return Client(workspace_id=workspace_id)
    return Client()


@dataclass
class CacheEntry:
    """Cache entry for a prompt template.

    Attributes:
        template: The cached ChatPromptTemplate.
        timestamp: When the entry was cached.
        source: Where the prompt came from ("hub" or "local").
    """

    template: ChatPromptTemplate
    timestamp: float
    source: str


@dataclass
class PromptCatalog:
    """Centralized prompt management with LangSmith Hub integration.

    The catalog provides:
    1. **Hub Integration**: Pull prompts from LangSmith Hub for version control
    2. **Local Fallback**: Use local definitions when Hub is unavailable
    3. **Caching**: Reduce Hub API calls with configurable TTL
    4. **Environment Selection**: Use different prompts per environment

    Attributes:
        organization: LangSmith organization name.
        environment: Environment tag for prompt selection.
        cache_ttl: Cache time-to-live in seconds.
        use_hub: Whether to attempt Hub lookups.
    """

    organization: str = field(default_factory=lambda: os.getenv(LANGSMITH_ORG_ENV, DEFAULT_ORG))
    environment: str = field(
        default_factory=lambda: os.getenv(PROMPT_ENVIRONMENT_ENV, "development")
    )
    cache_ttl: int = DEFAULT_CACHE_TTL
    use_hub: bool = field(default_factory=lambda: bool(os.getenv(LANGSMITH_API_KEY_ENV)))
    _cache: dict[str, CacheEntry] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Initialize the catalog and log configuration."""
        logger.info(
            "PromptCatalog initialized: org=%s, env=%s, use_hub=%s, cache_ttl=%d",
            self.organization,
            self.environment,
            self.use_hub,
            self.cache_ttl,
        )

    def _get_hub_path(self, name: PromptName) -> str:
        """Get the full LangSmith Hub path for a prompt.

        Args:
            name: Prompt name identifier.

        Returns:
            Full hub path. If organization is set, returns organization/prompt-name.
            If organization is empty, returns just prompt-name (workspace-scoped).
        """
        if self.organization:
            return f"{self.organization}/{name.value}"
        return name.value

    def _is_cache_valid(self, key: str) -> bool:
        """Check if a cache entry is still valid.

        Args:
            key: Cache key to check.

        Returns:
            True if entry exists and hasn't expired.
        """
        if key not in self._cache:
            return False
        entry = self._cache[key]
        return (time.time() - entry.timestamp) < self.cache_ttl

    async def _pull_from_hub(self, name: PromptName) -> ChatPromptTemplate | None:
        """Pull a prompt from LangSmith Hub.

        Args:
            name: Prompt name identifier.

        Returns:
            ChatPromptTemplate if found, None otherwise.
        """
        if not self.use_hub:
            return None

        hub_path = self._get_hub_path(name)

        try:
            # Pull with environment tag if not development
            if self.environment != "development":
                hub_path = f"{hub_path}:{self.environment}"

            logger.debug("Pulling prompt from hub: %s", hub_path)
            client = _create_langsmith_client()
            prompt = await asyncio.to_thread(client.pull_prompt, hub_path)

            if isinstance(prompt, ChatPromptTemplate):
                logger.info("Successfully pulled prompt from hub: %s", hub_path)
                return prompt

            logger.warning(
                "Hub prompt is not a ChatPromptTemplate: %s (type: %s)",
                hub_path,
                type(prompt).__name__,
            )
        except Exception as e:
            logger.debug("Failed to pull from hub (falling back to local): %s - %s", hub_path, e)

        return None

    def _get_local_fallback(self, name: PromptName) -> ChatPromptTemplate:
        """Get the local fallback prompt template.

        Args:
            name: Prompt name identifier.

        Returns:
            ChatPromptTemplate from local definitions.

        Raises:
            KeyError: If prompt name is not found in definitions.
        """
        if name not in PROMPT_DEFINITIONS:
            raise KeyError(f"Unknown prompt name: {name}")
        return PROMPT_DEFINITIONS[name].template

    @traceable_safe(name="prompt_catalog.get_prompt", run_type="retriever")
    async def get_prompt(self, name: PromptName) -> ChatPromptTemplate:
        """Get a prompt template, checking cache and hub first.

        The lookup order is:
        1. Check cache (if valid)
        2. Try LangSmith Hub (if enabled and API key available)
        3. Fall back to local definition

        Args:
            name: Prompt name identifier.

        Returns:
            ChatPromptTemplate for the requested prompt.
        """
        cache_key = f"{name.value}:{self.environment}"
        definition = PROMPT_DEFINITIONS.get(name)
        version = definition.metadata.version if definition else "unknown"

        # Check cache first
        if self._is_cache_valid(cache_key):
            logger.debug("Cache hit for prompt: %s", name)
            cached = self._cache[cache_key]
            # Log prompt metadata for tracing
            self._log_prompt_metadata(name, cached.source, version, from_cache=True)
            return cached.template

        # Try hub lookup
        template = await self._pull_from_hub(name)
        source = "hub"

        if template is None:
            # Fall back to local
            template = self._get_local_fallback(name)
            source = "local"

        # Update cache
        self._cache[cache_key] = CacheEntry(
            template=template,
            timestamp=time.time(),
            source=source,
        )
        logger.debug("Cached prompt: %s (source: %s)", name, source)

        # Log prompt metadata for tracing
        self._log_prompt_metadata(name, source, version, from_cache=False)

        return template

    def _log_prompt_metadata(
        self,
        name: PromptName,
        source: str,
        version: str,
        *,
        from_cache: bool,
    ) -> None:
        """Log prompt metadata for LangSmith tracing.

        Args:
            name: Prompt name identifier.
            source: Where the prompt came from ("hub" or "local").
            version: Prompt version string.
            from_cache: Whether the prompt was served from cache.
        """
        logger.debug(
            "Prompt loaded: name=%s, version=%s, source=%s, environment=%s, cached=%s",
            name.value,
            version,
            source,
            self.environment,
            from_cache,
        )

    def get_prompt_sync(self, name: PromptName) -> ChatPromptTemplate:
        """Synchronous version of get_prompt.

        Uses local definitions only to avoid blocking on async Hub calls.
        For Hub integration, use the async get_prompt method.

        Args:
            name: Prompt name identifier.

        Returns:
            ChatPromptTemplate from local definitions or cache.
        """
        cache_key = f"{name.value}:{self.environment}"

        # Check cache first (might have Hub version from previous async call)
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key].template

        # Use local fallback (don't block on Hub)
        template = self._get_local_fallback(name)

        # Update cache with local version
        self._cache[cache_key] = CacheEntry(
            template=template,
            timestamp=time.time(),
            source="local",
        )

        return template

    def get_definition(self, name: PromptName) -> PromptDefinition:
        """Get the full prompt definition including metadata.

        Args:
            name: Prompt name identifier.

        Returns:
            PromptDefinition with template and metadata.

        Raises:
            KeyError: If prompt name is not found.
        """
        if name not in PROMPT_DEFINITIONS:
            raise KeyError(f"Unknown prompt name: {name}")
        return PROMPT_DEFINITIONS[name]

    def list_prompts(self) -> list[PromptName]:
        """List all available prompt names.

        Returns:
            List of PromptName values.
        """
        return list(PROMPT_DEFINITIONS.keys())

    def get_cache_status(self) -> dict[str, dict[str, str | float | bool]]:
        """Get the current cache status for all entries.

        Returns:
            Dictionary mapping cache keys to their status.
        """
        now = time.time()
        status = {}
        for key, entry in self._cache.items():
            age = now - entry.timestamp
            status[key] = {
                "source": entry.source,
                "age_seconds": round(age, 2),
                "valid": age < self.cache_ttl,
            }
        return status

    def invalidate_cache(self, name: PromptName | None = None) -> int:
        """Invalidate cached prompts.

        Args:
            name: Specific prompt to invalidate, or None for all.

        Returns:
            Number of cache entries invalidated.
        """
        if name is None:
            count = len(self._cache)
            self._cache.clear()
            logger.info("Invalidated all %d cache entries", count)
            return count

        # Invalidate specific prompt across all environments
        invalidated = 0
        keys_to_remove = [k for k in self._cache if k.startswith(f"{name.value}:")]
        for key in keys_to_remove:
            del self._cache[key]
            invalidated += 1
        logger.info("Invalidated %d cache entries for prompt: %s", invalidated, name)
        return invalidated

    async def push(self, name: PromptName) -> str:
        """Push a prompt to LangSmith Hub.

        Args:
            name: Prompt name identifier.

        Returns:
            URL of the pushed prompt.

        Raises:
            KeyError: If prompt name is not found.
            RuntimeError: If LangSmith API key is not configured.
        """
        if not self.use_hub:
            raise RuntimeError(
                "LangSmith Hub is not enabled. Set LANGSMITH_API_KEY environment variable."
            )

        if name not in PROMPT_DEFINITIONS:
            raise KeyError(f"Unknown prompt name: {name}")

        definition = PROMPT_DEFINITIONS[name]
        hub_path = self._get_hub_path(name)

        try:
            client = _create_langsmith_client()
            url = await asyncio.to_thread(
                client.push_prompt,
                hub_path,
                object=definition.template,
                description=definition.metadata.description,
                tags=[*definition.metadata.tags, "ChatPromptTemplate"],
            )
            logger.info("Pushed prompt to hub: %s -> %s", name.value, url)
            return str(url)
        except Exception as e:
            # Handle 409 Conflict when prompt hasn't changed - treat as success
            error_str = str(e)
            if "409" in error_str and "Nothing to commit" in error_str:
                logger.info("Prompt unchanged, no commit needed: %s", name.value)
                return f"[UNCHANGED] {hub_path}"
            logger.error("Failed to push prompt %s: %s", name.value, e)
            raise

    async def push_all(self) -> dict[str, str]:
        """Push all prompts to LangSmith Hub.

        Returns:
            Dictionary mapping prompt names to URLs or error messages.
        """
        results: dict[str, str] = {}

        for name in PROMPT_DEFINITIONS:
            try:
                url = await self.push(name)
                results[name.value] = url
            except Exception as e:
                results[name.value] = f"ERROR: {e}"

        return results


# =============================================================================
# MODULE-LEVEL SINGLETON AND CONVENIENCE FUNCTIONS
# =============================================================================

_catalog: PromptCatalog | None = None


def get_catalog() -> PromptCatalog:
    """Get the global prompt catalog singleton.

    Returns:
        The global PromptCatalog instance.
    """
    global _catalog
    if _catalog is None:
        _catalog = PromptCatalog()
    return _catalog


def initialize_catalog(
    *,
    organization: str | None = None,
    environment: str | None = None,
    cache_ttl: int | None = None,
    use_hub: bool | None = None,
) -> PromptCatalog:
    """Initialize or reconfigure the global prompt catalog.

    Args:
        organization: LangSmith organization name.
        environment: Environment tag for prompt selection.
        cache_ttl: Cache time-to-live in seconds.
        use_hub: Whether to attempt Hub lookups.

    Returns:
        The configured PromptCatalog instance.
    """
    global _catalog

    kwargs: dict[str, str | int | bool] = {}
    if organization is not None:
        kwargs["organization"] = organization
    if environment is not None:
        kwargs["environment"] = environment
    if cache_ttl is not None:
        kwargs["cache_ttl"] = cache_ttl
    if use_hub is not None:
        kwargs["use_hub"] = use_hub

    _catalog = PromptCatalog(**kwargs)  # type: ignore[arg-type]
    return _catalog


async def get_prompt(name: PromptName) -> ChatPromptTemplate:
    """Get a prompt template asynchronously.

    Convenience function that uses the global catalog.

    Args:
        name: Prompt name identifier.

    Returns:
        ChatPromptTemplate for the requested prompt.
    """
    return await get_catalog().get_prompt(name)


@lru_cache(maxsize=32)
def get_prompt_sync(name: PromptName) -> ChatPromptTemplate:
    """Get a prompt template synchronously.

    Convenience function that uses the global catalog.
    Uses LRU cache for additional performance optimization.

    Args:
        name: Prompt name identifier.

    Returns:
        ChatPromptTemplate for the requested prompt.
    """
    return get_catalog().get_prompt_sync(name)


__all__ = [
    "CacheEntry",
    "PromptCatalog",
    "get_catalog",
    "get_prompt",
    "get_prompt_sync",
    "initialize_catalog",
]
