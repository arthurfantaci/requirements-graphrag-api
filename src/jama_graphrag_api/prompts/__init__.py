"""Centralized prompt catalog for GraphRAG.

This package provides:
- Centralized prompt definitions with metadata
- LangSmith Hub integration for version control
- Caching for performance optimization
- Evaluation utilities for A/B testing

Quick Start:
    from jama_graphrag_api.prompts import PromptName, get_prompt_sync

    # Get a prompt synchronously
    prompt = get_prompt_sync(PromptName.ROUTER)

    # Get a prompt asynchronously (supports Hub lookup)
    prompt = await get_prompt(PromptName.ROUTER)
"""

from __future__ import annotations

from jama_graphrag_api.prompts.catalog import (
    CacheEntry,
    PromptCatalog,
    get_catalog,
    get_prompt,
    get_prompt_sync,
    initialize_catalog,
)
from jama_graphrag_api.prompts.definitions import (
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
