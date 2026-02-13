"""Prompt Catalog with LangSmith Hub integration.

Provides centralized prompt management with:
- LangSmith Hub integration for version control and collaboration
- Local fallback when Hub is unavailable
- Environment-based prompt selection (dev/staging/production)
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
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
LANGSMITH_WORKSPACE_ID_ENV: Final[str] = "LANGSMITH_WORKSPACE_ID"
PROMPT_ENVIRONMENT_ENV: Final[str] = "PROMPT_ENVIRONMENT"

# Default configuration
# Empty string means workspace-scoped prompts (no org prefix)
DEFAULT_ORG: Final[str] = ""


def _create_langsmith_client() -> Client:
    """Create LangSmith Client with workspace_id for org-scoped API keys.

    Reads LANGSMITH_WORKSPACE_ID from environment and passes it explicitly
    since the SDK doesn't auto-detect it.

    Returns:
        Configured LangSmith Client instance.
    """
    from langsmith import Client

    # SDK doesn't auto-read workspace from env, must pass explicitly
    workspace_id = os.getenv(LANGSMITH_WORKSPACE_ID_ENV)
    if workspace_id:
        logger.info("Creating LangSmith client with workspace_id")
        return Client(workspace_id=workspace_id)
    return Client()


@dataclass
class PromptCatalog:
    """Centralized prompt management with LangSmith Hub integration.

    The catalog provides:
    1. **Hub Integration**: Pull prompts from LangSmith Hub for version control
    2. **Local Fallback**: Use local definitions when Hub is unavailable
    3. **Environment Selection**: Use different prompts per environment

    Attributes:
        organization: LangSmith organization name.
        environment: Environment tag for prompt selection.
        use_hub: Whether to attempt Hub lookups.
    """

    organization: str = field(default_factory=lambda: os.getenv(LANGSMITH_ORG_ENV, DEFAULT_ORG))
    environment: str = field(
        default_factory=lambda: os.getenv(PROMPT_ENVIRONMENT_ENV, "development")
    )
    use_hub: bool = field(default_factory=lambda: bool(os.getenv(LANGSMITH_API_KEY_ENV)))

    def __post_init__(self) -> None:
        """Initialize the catalog and log configuration."""
        logger.info(
            "PromptCatalog initialized: org=%s, env=%s, use_hub=%s",
            self.organization,
            self.environment,
            self.use_hub,
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
        """Get a prompt template, trying Hub first then local fallback.

        Args:
            name: Prompt name identifier.

        Returns:
            ChatPromptTemplate for the requested prompt.
        """
        # Try hub lookup (SDK handles caching internally)
        template = await self._pull_from_hub(name)
        source = "hub"

        if template is None:
            template = self._get_local_fallback(name)
            source = "local"

        definition = PROMPT_DEFINITIONS.get(name)
        version = definition.metadata.version if definition else "unknown"
        logger.debug(
            "Prompt loaded: name=%s, version=%s, source=%s, environment=%s",
            name.value,
            version,
            source,
            self.environment,
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

    async def _tag_latest_commit(
        self,
        client: Client,
        prompt_id: str,
        tags: list[str],
    ) -> None:
        """Apply commit tags to the latest commit of a prompt.

        Used when push_prompt returns 409 (content unchanged) since
        commit_tags are only applied to new commits. If a tag already
        exists (on any commit), it is deleted and recreated on the latest.
        """
        try:
            commits = await asyncio.to_thread(
                lambda: list(client.list_prompt_commits(prompt_id, limit=1))
            )
            if not commits:
                logger.warning("No commits found for %s", prompt_id)
                return
            commit_id = str(commits[0].id)
            repo_path = f"-/{prompt_id}"
            for tag in tags:
                try:
                    await asyncio.to_thread(client._create_commit_tags, repo_path, commit_id, [tag])
                except Exception as tag_err:
                    if "409" not in str(tag_err):
                        raise
                    # Tag exists (possibly on old commit) — delete and recreate
                    await asyncio.to_thread(
                        client.request_with_retries,
                        "DELETE",
                        f"/repos/{repo_path}/tags/{tag}",
                    )
                    await asyncio.to_thread(
                        client.request_with_retries,
                        "POST",
                        f"/repos/{repo_path}/tags",
                        json={"tag_name": tag, "commit_id": commit_id},
                    )
            logger.info("Tagged %s latest commit with %s", prompt_id, tags)
        except Exception as e:
            logger.warning("Failed to tag %s: %s", prompt_id, e)

    async def push(
        self,
        name: PromptName,
        *,
        commit_tags: list[str] | None = None,
    ) -> str:
        """Push a prompt to LangSmith Hub.

        Args:
            name: Prompt name identifier.
            commit_tags: Optional version tags (e.g. ["production", "staging"])
                applied atomically during push.

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
        # Use bare prompt name for push (workspace scoping is handled by
        # the Client's workspace_id header, not by path prefix).
        prompt_id = name.value

        try:
            client = _create_langsmith_client()
            push_kwargs: dict[str, object] = {
                "object": definition.template,
                "description": definition.metadata.description,
                "tags": [*definition.metadata.tags, "ChatPromptTemplate"],
            }
            if commit_tags:
                push_kwargs["commit_tags"] = commit_tags
            url = await asyncio.to_thread(
                client.push_prompt,
                prompt_id,
                **push_kwargs,
            )
            logger.info("Pushed prompt to hub: %s -> %s", name.value, url)
            return str(url)
        except Exception as e:
            error_str = str(e)
            # 409 "Nothing to commit" — content unchanged, apply tags separately
            if "409" in error_str and "Nothing to commit" in error_str:
                logger.info("Prompt unchanged, no commit needed: %s", name.value)
                if commit_tags:
                    await self._tag_latest_commit(client, prompt_id, commit_tags)
                return f"[UNCHANGED] {prompt_id}"
            # 409 "already exists" — push succeeded but tag move failed;
            # content is on Hub, just need to fix the tags
            if "409" in error_str and "already exists" in error_str:
                logger.info("Push succeeded but tag conflict: %s", name.value)
                if commit_tags:
                    await self._tag_latest_commit(client, prompt_id, commit_tags)
                return f"https://smith.langchain.com/hub/{prompt_id}"
            logger.error("Failed to push prompt %s: %s", name.value, e)
            raise

    async def push_all(
        self,
        *,
        commit_tags: list[str] | None = None,
    ) -> dict[str, str]:
        """Push all prompts to LangSmith Hub.

        Args:
            commit_tags: Optional version tags applied to all pushed prompts.

        Returns:
            Dictionary mapping prompt names to URLs or error messages.
        """
        results: dict[str, str] = {}

        for name in PROMPT_DEFINITIONS:
            try:
                url = await self.push(name, commit_tags=commit_tags)
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
    use_hub: bool | None = None,
) -> PromptCatalog:
    """Initialize or reconfigure the global prompt catalog.

    Args:
        organization: LangSmith organization name.
        environment: Environment tag for prompt selection.
        use_hub: Whether to attempt Hub lookups.

    Returns:
        The configured PromptCatalog instance.
    """
    global _catalog

    kwargs: dict[str, str | bool] = {}
    if organization is not None:
        kwargs["organization"] = organization
    if environment is not None:
        kwargs["environment"] = environment
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


__all__ = [
    "PromptCatalog",
    "get_catalog",
    "get_prompt",
    "initialize_catalog",
]
