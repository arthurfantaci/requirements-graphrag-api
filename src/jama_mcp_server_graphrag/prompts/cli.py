"""CLI for prompt catalog management.

Provides commands for:
- Pushing local prompts to LangSmith Hub
- Pulling prompts from Hub to local cache
- Listing available prompts and their status
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Final

from jama_mcp_server_graphrag.prompts.catalog import (
    LANGSMITH_API_KEY_ENV,
    LANGSMITH_ORG_ENV,
    get_catalog,
    initialize_catalog,
)
from jama_mcp_server_graphrag.prompts.definitions import (
    PROMPT_DEFINITIONS,
    PromptName,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Exit codes
EXIT_SUCCESS: Final[int] = 0
EXIT_ERROR: Final[int] = 1
EXIT_CONFIG_ERROR: Final[int] = 2


def _check_hub_config() -> bool:
    """Check if LangSmith Hub is configured.

    Returns:
        True if API key is set, False otherwise.
    """
    if not os.getenv(LANGSMITH_API_KEY_ENV):
        logger.error(
            "LangSmith API key not set. Set %s environment variable.",
            LANGSMITH_API_KEY_ENV,
        )
        return False
    return True


async def cmd_push(args: argparse.Namespace) -> int:
    """Push prompts to LangSmith Hub.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    if not _check_hub_config():
        return EXIT_CONFIG_ERROR

    try:
        from langchain import hub  # noqa: PLC0415
    except ImportError:
        logger.error("langchain package not installed. Run: pip install langchain")
        return EXIT_ERROR

    org = os.getenv(LANGSMITH_ORG_ENV, "jama-graphrag")

    # Determine which prompts to push
    if args.all:
        prompt_names = list(PromptName)
    elif args.prompt:
        try:
            prompt_names = [PromptName(args.prompt)]
        except ValueError:
            logger.error("Unknown prompt name: %s", args.prompt)
            logger.info("Available prompts: %s", ", ".join(p.value for p in PromptName))
            return EXIT_ERROR
    else:
        logger.error("Specify --all or --prompt <name>")
        return EXIT_ERROR

    # Push each prompt
    success_count = 0
    for name in prompt_names:
        definition = PROMPT_DEFINITIONS[name]
        hub_path = f"{org}/{name.value}"

        try:
            logger.info("Pushing %s to %s...", name.value, hub_path)
            hub.push(
                hub_path,
                definition.template,
                new_repo_description=definition.metadata.description,
                new_repo_is_public=False,
                tags=definition.metadata.tags,
            )
            logger.info("✓ Pushed %s", name.value)
            success_count += 1
        except Exception as e:
            logger.error("✗ Failed to push %s: %s", name.value, e)

    logger.info(
        "Pushed %d/%d prompts successfully",
        success_count,
        len(prompt_names),
    )
    return EXIT_SUCCESS if success_count == len(prompt_names) else EXIT_ERROR


async def cmd_pull(args: argparse.Namespace) -> int:
    """Pull prompts from LangSmith Hub.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    if not _check_hub_config():
        return EXIT_CONFIG_ERROR

    catalog = get_catalog()

    # Determine which prompts to pull
    if args.all:
        prompt_names = list(PromptName)
    elif args.prompt:
        try:
            prompt_names = [PromptName(args.prompt)]
        except ValueError:
            logger.error("Unknown prompt name: %s", args.prompt)
            logger.info("Available prompts: %s", ", ".join(p.value for p in PromptName))
            return EXIT_ERROR
    else:
        logger.error("Specify --all or --prompt <name>")
        return EXIT_ERROR

    # Pull each prompt
    success_count = 0
    for name in prompt_names:
        try:
            logger.info("Pulling %s...", name.value)
            template = await catalog.get_prompt(name)
            if template:
                logger.info("✓ Pulled %s", name.value)
                success_count += 1
        except Exception as e:
            logger.error("✗ Failed to pull %s: %s", name.value, e)

    logger.info(
        "Pulled %d/%d prompts successfully",
        success_count,
        len(prompt_names),
    )
    return EXIT_SUCCESS if success_count == len(prompt_names) else EXIT_ERROR


def cmd_list(args: argparse.Namespace) -> int:  # noqa: ARG001
    """List available prompts and their metadata.

    Args:
        args: Parsed command-line arguments (unused).

    Returns:
        Exit code (always 0).
    """
    catalog = get_catalog()

    print("\n📚 Available Prompts")
    print("=" * 60)

    for name in PromptName:
        definition = PROMPT_DEFINITIONS[name]
        meta = definition.metadata

        print(f"\n🏷️  {name.value}")
        print(f"   Version: {meta.version}")
        print(f"   Description: {meta.description}")
        print(f"   Input Variables: {', '.join(meta.input_variables)}")
        print(f"   Output Format: {meta.output_format}")
        print(f"   Tags: {', '.join(meta.tags)}")

    # Show cache status
    cache_status = catalog.get_cache_status()
    if cache_status:
        print("\n📦 Cache Status")
        print("-" * 60)
        for key, status in cache_status.items():
            valid_icon = "✓" if status["valid"] else "✗"
            print(f"   {valid_icon} {key}: {status['source']} ({status['age_seconds']:.1f}s old)")

    print()
    return EXIT_SUCCESS


def cmd_validate(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Validate all prompt definitions.

    Args:
        args: Parsed command-line arguments (unused).

    Returns:
        Exit code (0 if all valid, 1 if any invalid).
    """
    print("\n🔍 Validating Prompts")
    print("=" * 60)

    all_valid = True
    for name in PromptName:
        definition = PROMPT_DEFINITIONS[name]
        template = definition.template
        meta = definition.metadata

        errors: list[str] = []

        # Check input variables match
        template_vars = set(template.input_variables)
        meta_vars = set(meta.input_variables)
        if template_vars != meta_vars:
            errors.append(f"Variable mismatch: template={template_vars}, metadata={meta_vars}")

        # Check version format (should be semver-like)
        version_parts = meta.version.split(".")
        if len(version_parts) != 3 or not all(p.isdigit() for p in version_parts):  # noqa: PLR2004
            errors.append(f"Invalid version format: {meta.version}")

        # Report results
        if errors:
            print(f"\n✗ {name.value}")
            for error in errors:
                print(f"   - {error}")
            all_valid = False
        else:
            print(f"✓ {name.value}")

    print()
    return EXIT_SUCCESS if all_valid else EXIT_ERROR


def cmd_clear_cache(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Clear the prompt cache.

    Args:
        args: Parsed command-line arguments (unused).

    Returns:
        Exit code (always 0).
    """
    catalog = get_catalog()
    count = catalog.invalidate_cache()
    print(f"Cleared {count} cache entries")
    return EXIT_SUCCESS


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Prompt Catalog CLI - Manage GraphRAG prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Push command
    push_parser = subparsers.add_parser("push", help="Push prompts to LangSmith Hub")
    push_group = push_parser.add_mutually_exclusive_group()
    push_group.add_argument("--all", action="store_true", help="Push all prompts")
    push_group.add_argument("--prompt", type=str, help="Specific prompt to push")

    # Pull command
    pull_parser = subparsers.add_parser("pull", help="Pull prompts from LangSmith Hub")
    pull_group = pull_parser.add_mutually_exclusive_group()
    pull_group.add_argument("--all", action="store_true", help="Pull all prompts")
    pull_group.add_argument("--prompt", type=str, help="Specific prompt to pull")

    # List command
    subparsers.add_parser("list", help="List available prompts")

    # Validate command
    subparsers.add_parser("validate", help="Validate prompt definitions")

    # Clear cache command
    subparsers.add_parser("clear-cache", help="Clear the prompt cache")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return EXIT_SUCCESS

    # Initialize catalog
    initialize_catalog()

    # Command dispatch table
    sync_commands = {
        "list": cmd_list,
        "validate": cmd_validate,
        "clear-cache": cmd_clear_cache,
    }
    async_commands = {
        "push": cmd_push,
        "pull": cmd_pull,
    }

    if args.command in sync_commands:
        return sync_commands[args.command](args)
    if args.command in async_commands:
        return asyncio.run(async_commands[args.command](args))

    parser.print_help()
    return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
