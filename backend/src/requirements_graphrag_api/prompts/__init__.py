"""Centralized prompt catalog for GraphRAG.

This package provides:
- Centralized prompt definitions with metadata
- LangSmith Hub integration for version control
- Environment-based prompt selection

Quick Start:
    from requirements_graphrag_api.prompts import PromptName, get_prompt

    # Get a prompt asynchronously (supports Hub lookup)
    prompt = await get_prompt(PromptName.INTENT_CLASSIFIER)
"""

from __future__ import annotations

from requirements_graphrag_api.prompts.catalog import (
    PromptCatalog,
    get_catalog,
    get_prompt,
    initialize_catalog,
)
from requirements_graphrag_api.prompts.definitions import (
    PROMPT_DEFINITIONS,
    PromptDefinition,
    PromptMetadata,
    PromptName,
)

__all__ = [
    "PROMPT_DEFINITIONS",
    "PromptCatalog",
    "PromptDefinition",
    "PromptMetadata",
    "PromptName",
    "get_catalog",
    "get_prompt",
    "initialize_catalog",
]
