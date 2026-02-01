#!/usr/bin/env python3
"""Script to create API keys for the GraphRAG API.

This script creates API keys in the PostgreSQL database for production use.
The raw API key is displayed only once - store it securely!

Usage:
    # Set the database URL from Railway
    export AUTH_DATABASE_URL="postgresql://..."

    # Create a standard key
    python scripts/create_api_key.py --name "My App"

    # Create an enterprise key with admin scope
    python scripts/create_api_key.py --name "Admin App" --tier enterprise

    # Create a key for a specific organization
    python scripts/create_api_key.py --name "Acme Corp" --tier premium --org "Acme"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add backend/src to path for imports
backend_src = Path(__file__).parent.parent / "backend" / "src"
sys.path.insert(0, str(backend_src))

from requirements_graphrag_api.auth import (
    APIKeyInfo,
    APIKeyTier,
    generate_api_key,
)
from requirements_graphrag_api.auth.postgres_store import PostgresAPIKeyStore


# Rate limits by tier
TIER_RATE_LIMITS = {
    "free": "10/minute",
    "standard": "50/minute",
    "premium": "200/minute",
    "enterprise": "1000/minute",
}

# Default scopes by tier
TIER_SCOPES = {
    "free": ("chat", "search"),
    "standard": ("chat", "search", "feedback"),
    "premium": ("chat", "search", "feedback"),
    "enterprise": ("chat", "search", "feedback", "admin"),
}


async def create_key(
    name: str,
    tier: str,
    organization: str | None,
    scopes: tuple[str, ...] | None = None,
) -> None:
    """Create a new API key.

    Args:
        name: Human-readable name for the key.
        tier: API key tier (free, standard, premium, enterprise).
        organization: Organization name (optional).
        scopes: Custom scopes (uses tier defaults if not specified).
    """
    database_url = os.environ.get("AUTH_DATABASE_URL")
    if not database_url:
        print("ERROR: AUTH_DATABASE_URL environment variable not set")
        print()
        print("Set it from your Railway PostgreSQL connection string:")
        print('  export AUTH_DATABASE_URL="postgresql://..."')
        sys.exit(1)

    # Generate key
    raw_key = generate_api_key()
    key_id = f"key_{raw_key[6:14]}"  # Use part of key as ID

    # Determine scopes
    final_scopes = scopes if scopes else TIER_SCOPES.get(tier, ("chat", "search"))

    info = APIKeyInfo(
        key_id=key_id,
        name=name,
        tier=APIKeyTier(tier),
        organization=organization,
        created_at=datetime.now(UTC),
        expires_at=None,
        rate_limit=TIER_RATE_LIMITS.get(tier, "10/minute"),
        is_active=True,
        scopes=final_scopes,
        metadata={"created_by": "create_api_key.py"},
    )

    # Store in database
    try:
        async with PostgresAPIKeyStore(database_url) as store:
            await store.create(raw_key, info)
    except Exception as e:
        print(f"ERROR: Failed to create key: {e}")
        sys.exit(1)

    # Display results
    print()
    print("=" * 60)
    print("  API KEY CREATED SUCCESSFULLY")
    print("=" * 60)
    print()
    print(f"  Name:         {name}")
    print(f"  Key ID:       {key_id}")
    print(f"  Tier:         {tier}")
    print(f"  Organization: {organization or '(none)'}")
    print(f"  Rate Limit:   {info.rate_limit}")
    print(f"  Scopes:       {', '.join(final_scopes)}")
    print()
    print("-" * 60)
    print("  RAW API KEY (save this - shown only once!):")
    print("-" * 60)
    print()
    print(f"  {raw_key}")
    print()
    print("-" * 60)
    print("  Usage example:")
    print("-" * 60)
    print()
    print(f'  curl -H "X-API-Key: {raw_key}" \\')
    print('       -H "Content-Type: application/json" \\')
    print("       -X POST https://your-api.railway.app/chat \\")
    print('       -d \'{"query": "What is requirements traceability?"}\'')
    print()


async def list_keys(organization: str | None = None) -> None:
    """List all API keys.

    Args:
        organization: Filter by organization (optional).
    """
    database_url = os.environ.get("AUTH_DATABASE_URL")
    if not database_url:
        print("ERROR: AUTH_DATABASE_URL environment variable not set")
        sys.exit(1)

    try:
        async with PostgresAPIKeyStore(database_url) as store:
            keys = await store.list_keys(organization=organization, include_inactive=True)
    except Exception as e:
        print(f"ERROR: Failed to list keys: {e}")
        sys.exit(1)

    if not keys:
        print("No API keys found.")
        return

    print()
    print(f"{'Key ID':<20} {'Name':<25} {'Tier':<12} {'Active':<8} {'Organization'}")
    print("-" * 90)
    for key in keys:
        active = "Yes" if key.is_active else "No"
        org = key.organization or "-"
        print(f"{key.key_id:<20} {key.name:<25} {key.tier.value:<12} {active:<8} {org}")
    print()
    print(f"Total: {len(keys)} key(s)")


async def revoke_key(key_id: str) -> None:
    """Revoke an API key.

    Args:
        key_id: The key ID to revoke.
    """
    database_url = os.environ.get("AUTH_DATABASE_URL")
    if not database_url:
        print("ERROR: AUTH_DATABASE_URL environment variable not set")
        sys.exit(1)

    try:
        async with PostgresAPIKeyStore(database_url) as store:
            success = await store.revoke(key_id)
    except Exception as e:
        print(f"ERROR: Failed to revoke key: {e}")
        sys.exit(1)

    if success:
        print(f"Successfully revoked key: {key_id}")
    else:
        print(f"Key not found: {key_id}")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Manage API keys for the GraphRAG API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a standard key
  python scripts/create_api_key.py create --name "My App"

  # Create an enterprise key
  python scripts/create_api_key.py create --name "Admin" --tier enterprise

  # List all keys
  python scripts/create_api_key.py list

  # Revoke a key
  python scripts/create_api_key.py revoke --key-id key_abc12345
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new API key")
    create_parser.add_argument("--name", required=True, help="Key name")
    create_parser.add_argument(
        "--tier",
        default="standard",
        choices=["free", "standard", "premium", "enterprise"],
        help="API key tier (default: standard)",
    )
    create_parser.add_argument("--org", default=None, help="Organization name")

    # List command
    list_parser = subparsers.add_parser("list", help="List all API keys")
    list_parser.add_argument("--org", default=None, help="Filter by organization")

    # Revoke command
    revoke_parser = subparsers.add_parser("revoke", help="Revoke an API key")
    revoke_parser.add_argument("--key-id", required=True, help="Key ID to revoke")

    args = parser.parse_args()

    if args.command == "create":
        asyncio.run(create_key(args.name, args.tier, args.org))
    elif args.command == "list":
        asyncio.run(list_keys(args.org))
    elif args.command == "revoke":
        asyncio.run(revoke_key(args.key_id))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
