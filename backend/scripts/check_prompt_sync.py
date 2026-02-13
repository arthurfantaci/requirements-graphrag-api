#!/usr/bin/env python
"""Check whether LangSmith Hub prompts match local definitions.

Pulls the :production-tagged version of each prompt from Hub and compares
its template against the local PROMPT_DEFINITIONS. Reports any drift.

Usage:
    # Check all prompts
    uv run python scripts/check_prompt_sync.py

    # Verbose output (show matching prompts too)
    uv run python scripts/check_prompt_sync.py --verbose

Exit codes:
    0 - All prompts in sync
    1 - Drift detected (Hub differs from local)
    2 - Configuration or runtime error

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


def _templates_match(local_template: object, hub_template: object) -> bool:
    """Compare two ChatPromptTemplate instances by their message structure.

    Compares the message template strings and types without formatting
    (which would fail on templates with unfilled variables).
    """
    from langchain_core.prompts import ChatPromptTemplate

    if not isinstance(local_template, ChatPromptTemplate) or not isinstance(
        hub_template, ChatPromptTemplate
    ):
        return False

    if len(local_template.messages) != len(hub_template.messages):
        return False

    for local_msg, hub_msg in zip(local_template.messages, hub_template.messages, strict=True):
        if type(local_msg) is not type(hub_msg):
            return False
        # Compare template strings (handles MessagePromptTemplate subclasses)
        local_text = getattr(getattr(local_msg, "prompt", None), "template", None)
        hub_text = getattr(getattr(hub_msg, "prompt", None), "template", None)
        if local_text != hub_text:
            return False

    return True


async def check_sync(*, verbose: bool = False) -> int:
    """Check if Hub prompts match local definitions.

    Args:
        verbose: If True, print status for all prompts (not just mismatches).

    Returns:
        Exit code (0 = in sync, 1 = drift detected, 2 = error).
    """
    from langsmith import Client

    from requirements_graphrag_api.prompts.definitions import PROMPT_DEFINITIONS, PromptName

    logger.info("=" * 60)
    logger.info("PROMPT SYNC CHECK")
    logger.info("=" * 60)

    workspace_id = os.getenv("LANGSMITH_WORKSPACE_ID")
    client = Client(workspace_id=workspace_id) if workspace_id else Client()

    in_sync = 0
    drifted = 0
    missing = 0
    errors = 0

    for name in PromptName:
        defn = PROMPT_DEFINITIONS[name]
        hub_path = f"{name.value}:production"

        try:
            hub_template = await asyncio.to_thread(client.pull_prompt, hub_path)
        except Exception as e:
            error_str = str(e)
            if "not found" in error_str.lower() or "404" in error_str:
                logger.warning("  [MISSING] %s — not found on Hub", name.value)
                missing += 1
            else:
                logger.error("  [ERROR] %s — %s", name.value, e)
                errors += 1
            continue

        # Compare templates
        try:
            if _templates_match(defn.template, hub_template):
                in_sync += 1
                if verbose:
                    logger.info("  [OK] %s (v%s)", name.value, defn.metadata.version)
            else:
                drifted += 1
                logger.warning(
                    "  [DRIFT] %s — Hub :production differs from local v%s",
                    name.value,
                    defn.metadata.version,
                )
        except Exception as e:
            logger.error("  [ERROR] %s — comparison failed: %s", name.value, e)
            errors += 1

    total = len(list(PromptName))
    logger.info("=" * 60)
    logger.info(
        "RESULT: %d in sync, %d drifted, %d missing, %d errors (of %d total)",
        in_sync,
        drifted,
        missing,
        errors,
        total,
    )
    logger.info("=" * 60)

    if drifted > 0 or missing > 0:
        logger.warning(
            "Run 'uv run python scripts/sync_prompts_to_hub.py' to push local prompts to Hub."
        )
        return 1
    if errors > 0:
        return 2
    return 0


def main() -> int:
    """Entry point."""
    if not os.getenv("LANGSMITH_API_KEY"):
        logger.error("LANGSMITH_API_KEY not set")
        return 2

    parser = argparse.ArgumentParser(description="Check LangSmith Hub prompt sync status")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show status for all prompts (not just mismatches)",
    )

    args = parser.parse_args()
    return asyncio.run(check_sync(verbose=args.verbose))


if __name__ == "__main__":
    sys.exit(main())
