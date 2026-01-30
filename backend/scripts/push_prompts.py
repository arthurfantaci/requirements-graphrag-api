#!/usr/bin/env python
"""Script to push all prompts to LangSmith Hub.

This script initializes the prompt catalog with configuration from
environment variables and pushes all prompt definitions to LangSmith Hub.

Usage:
    # Set environment variables
    export LANGSMITH_API_KEY=your_api_key
    export LANGSMITH_ORG=requirements-graphrag

    # Push all prompts
    uv run python scripts/push_prompts.py

    # Push with custom organization
    uv run python scripts/push_prompts.py --org my-organization

    # Push specific prompt
    uv run python scripts/push_prompts.py --prompt graphrag-intent-classifier

    # Dry run (list prompts without pushing)
    uv run python scripts/push_prompts.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path for local development
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from requirements_graphrag_api.prompts import (  # noqa: E402
    PROMPT_DEFINITIONS,
    PromptCatalog,
    PromptName,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def push_prompts(
    *,
    organization: str,
    environment: str,
    prompt_name: str | None = None,
    dry_run: bool = False,
) -> dict[str, str]:
    """Push prompts to LangSmith Hub.

    Args:
        organization: LangSmith organization name.
        environment: Target environment.
        prompt_name: Specific prompt to push (None for all).
        dry_run: If True, list prompts without pushing.

    Returns:
        Dictionary mapping prompt names to URLs or status.
    """
    logger.info("Initializing prompt catalog")
    logger.info("  Organization: %s", organization)
    logger.info("  Environment: %s", environment)

    # Initialize catalog
    catalog = PromptCatalog(
        organization=organization,
        environment=environment,
        use_hub=True,
    )

    if dry_run:
        logger.info("\n🔍 DRY RUN - Listing prompts without pushing\n")
        results = {}
        for name, definition in PROMPT_DEFINITIONS.items():
            hub_path = catalog._get_hub_path(name)
            logger.info("  📝 %s", hub_path)
            logger.info("     Version: %s", definition.metadata.version)
            logger.info("     Description: %s", definition.metadata.description)
            logger.info("     Variables: %s", ", ".join(definition.metadata.input_variables))
            logger.info("     Tags: %s", ", ".join(definition.metadata.tags))
            results[name.value] = f"[DRY RUN] Would push to {hub_path}"
        return results

    results = {}

    if prompt_name:
        # Push specific prompt
        try:
            name = PromptName(prompt_name)
        except ValueError:
            logger.error("❌ Unknown prompt: %s", prompt_name)
            logger.error("   Available: %s", ", ".join(p.value for p in PromptName))
            sys.exit(1)

        logger.info("\n🚀 Pushing prompt: %s\n", name.value)
        try:
            url = await catalog.push(name)
            logger.info("✅ Success: %s", url)
            results[name.value] = url
        except Exception as e:
            logger.error("❌ Failed: %s", e)
            results[name.value] = f"ERROR: {e}"
            sys.exit(1)
    else:
        # Push all prompts
        logger.info("\n🚀 Pushing all prompts to LangSmith Hub\n")
        results = await catalog.push_all()

        pushed_count = sum(
            1 for url in results.values() if not str(url).startswith(("ERROR", "[UNCHANGED]"))
        )
        unchanged_count = sum(1 for url in results.values() if str(url).startswith("[UNCHANGED]"))
        error_count = sum(1 for url in results.values() if str(url).startswith("ERROR"))

        logger.info("\n📊 Results:")
        for name, url in results.items():
            if str(url).startswith("ERROR"):
                logger.error("   ❌ %s: %s", name, url)
            elif str(url).startswith("[UNCHANGED]"):
                logger.info("   ⏭️  %s: %s", name, url)
            else:
                logger.info("   ✅ %s: %s", name, url)

        logger.info(
            "\n📈 Summary: %d pushed, %d unchanged, %d failed",
            pushed_count,
            unchanged_count,
            error_count,
        )

        if error_count > 0:
            sys.exit(1)

    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Push prompts to LangSmith Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--org",
        default=os.getenv("LANGSMITH_ORG", ""),
        help="LangSmith organization name (default: empty for workspace-scoped prompts)",
    )
    parser.add_argument(
        "--env",
        default=os.getenv("PROMPT_ENVIRONMENT", "development"),
        help="Target environment (default: development)",
    )
    parser.add_argument(
        "--prompt",
        help="Specific prompt to push (default: push all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List prompts without pushing",
    )

    args = parser.parse_args()

    # Check for API key
    if not args.dry_run and not os.getenv("LANGSMITH_API_KEY"):
        logger.error("❌ LANGSMITH_API_KEY environment variable not set")
        logger.error("   Set it with: export LANGSMITH_API_KEY=your_api_key")
        sys.exit(1)

    asyncio.run(
        push_prompts(
            organization=args.org,
            environment=args.env,
            prompt_name=args.prompt,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
