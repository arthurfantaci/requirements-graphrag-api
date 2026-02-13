#!/usr/bin/env python
"""Sync all prompts from local definitions to LangSmith Hub.

Pushes each prompt in PROMPT_DEFINITIONS to LangSmith Hub with
`:production` and `:staging` commit tags for version pinning.

Usage:
    # Dry run (show what would be pushed)
    uv run python scripts/sync_prompts_to_hub.py --dry-run

    # Push all prompts
    uv run python scripts/sync_prompts_to_hub.py

    # Push only specific prompts
    uv run python scripts/sync_prompts_to_hub.py --only synthesis text2cypher

    # Push with custom tags
    uv run python scripts/sync_prompts_to_hub.py --tags production staging

Requires LANGSMITH_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def sync_prompts(
    *,
    only: list[str] | None = None,
    tags: list[str] | None = None,
    dry_run: bool = False,
) -> int:
    """Push prompts to LangSmith Hub.

    Args:
        only: If set, only push prompts whose value matches these names.
        tags: Commit tags to apply (default: ["production", "staging"]).
        dry_run: If True, only show what would be pushed.

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    from requirements_graphrag_api.prompts.catalog import PromptCatalog
    from requirements_graphrag_api.prompts.definitions import PROMPT_DEFINITIONS, PromptName

    commit_tags = tags or ["production", "staging"]

    # Filter prompts if --only specified
    prompts_to_push: list[PromptName] = []
    for name in PromptName:
        if only and name.value not in only:
            continue
        prompts_to_push.append(name)

    if only:
        # Validate --only values
        valid_names = {n.value for n in PromptName}
        invalid = set(only) - valid_names
        if invalid:
            logger.error("Unknown prompt names: %s", ", ".join(sorted(invalid)))
            logger.info("Valid names: %s", ", ".join(sorted(valid_names)))
            return 1

    logger.info("=" * 60)
    logger.info("PROMPT HUB SYNC")
    logger.info("=" * 60)
    logger.info("Prompts: %d / %d", len(prompts_to_push), len(PROMPT_DEFINITIONS))
    logger.info("Commit tags: %s", ", ".join(commit_tags))
    logger.info("Dry run: %s", dry_run)
    logger.info("=" * 60)

    if dry_run:
        for name in prompts_to_push:
            defn = PROMPT_DEFINITIONS[name]
            logger.info(
                "  [DRY RUN] %s (v%s) -> Hub with tags %s",
                name.value,
                defn.metadata.version,
                commit_tags,
            )
        logger.info("Dry run complete. %d prompts would be pushed.", len(prompts_to_push))
        return 0

    catalog = PromptCatalog()
    errors = 0

    for name in prompts_to_push:
        defn = PROMPT_DEFINITIONS[name]
        try:
            url = await catalog.push(name, commit_tags=commit_tags)
            status = "UNCHANGED" if "[UNCHANGED]" in url else "PUSHED"
            logger.info("  [%s] %s (v%s) -> %s", status, name.value, defn.metadata.version, url)
        except Exception as e:
            logger.error("  [ERROR] %s: %s", name.value, e)
            errors += 1

    logger.info("=" * 60)
    logger.info(
        "SYNC COMPLETE: %d pushed, %d errors",
        len(prompts_to_push) - errors,
        errors,
    )
    logger.info("=" * 60)

    return 1 if errors else 0


def main() -> int:
    """Entry point."""
    if not os.getenv("LANGSMITH_API_KEY"):
        logger.error("LANGSMITH_API_KEY not set")
        return 1

    parser = argparse.ArgumentParser(description="Sync prompts to LangSmith Hub")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be pushed without pushing",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Only push these prompt names (e.g. synthesis text2cypher)",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["production", "staging"],
        help="Commit tags to apply (default: production staging)",
    )

    args = parser.parse_args()
    return asyncio.run(sync_prompts(only=args.only, tags=args.tags, dry_run=args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
