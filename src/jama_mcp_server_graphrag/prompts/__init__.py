"""Centralized prompt catalog for GraphRAG.

This package provides:
- Centralized prompt definitions with metadata
- LangSmith Hub integration for version control
- Caching for performance optimization
- CLI tools for prompt management
- Evaluation utilities for A/B testing

Quick Start:
    from jama_mcp_server_graphrag.prompts import PromptName, get_prompt_sync

    # Get a prompt synchronously
    prompt = get_prompt_sync(PromptName.ROUTER)

    # Get a prompt asynchronously (supports Hub lookup)
    prompt = await get_prompt(PromptName.ROUTER)

CLI Usage:
    # List available prompts
    python -m jama_mcp_server_graphrag.prompts.cli list

    # Push prompts to LangSmith Hub
    python -m jama_mcp_server_graphrag.prompts.cli push --all

    # Validate prompt definitions
    python -m jama_mcp_server_graphrag.prompts.cli validate
"""

from __future__ import annotations

from jama_mcp_server_graphrag.prompts.catalog import (
    CacheEntry,
    PromptCatalog,
    get_catalog,
    get_prompt,
    get_prompt_sync,
    initialize_catalog,
)
from jama_mcp_server_graphrag.prompts.definitions import (
    PROMPT_DEFINITIONS,
    TEXT2CYPHER_EXAMPLES,
    PromptDefinition,
    PromptMetadata,
    PromptName,
)

__all__ = [
    "PROMPT_DEFINITIONS",
    "TEXT2CYPHER_EXAMPLES",
    "CacheEntry",
    "PromptCatalog",
    "PromptDefinition",
    "PromptMetadata",
    "PromptName",
    "get_catalog",
    "get_prompt",
    "get_prompt_sync",
    "initialize_catalog",
]
