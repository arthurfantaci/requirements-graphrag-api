"""Agentic-specific prompt definitions and utilities.

This module extends the centralized prompt catalog with agentic-specific
prompts and provides utilities for prompt composition.

New Prompts:
- AGENT_REASONING: Main agent loop system prompt with tool descriptions
- QUERY_EXPANSION: Generates multiple search queries from user question
- SYNTHESIS: Combines CRITIC self-evaluation with answer generation
- ENTITY_SELECTOR: Selects entities for deep exploration

Integration with Existing Prompts:
- STEPBACK: Used within QUERY_EXPANSION for broader context queries
- QUERY_UPDATER: Used for multi-turn conversation context
- CRITIC: Used within SYNTHESIS for self-evaluation
- RAG_GENERATION: Used as base format for final answers

Prompt Design Principles:
1. Compose prompts hierarchically (don't duplicate logic)
2. Use the centralized catalog for version control
3. Push new prompts to LangSmith Hub for A/B testing
"""

from __future__ import annotations

__all__: list[str] = []
