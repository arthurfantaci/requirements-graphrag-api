"""Vercel serverless entry point.

This module provides the entry point for Vercel serverless deployment.
It imports the FastAPI app from the main API module.

Usage:
    Vercel automatically uses this file as the serverless function handler.
    The app is exposed as the default export for the @vercel/python runtime.

Note:
    The PYTHONPATH is set to 'src' in vercel.json to allow imports
    from the jama_graphrag_api package.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for local package imports
# This ensures the package is importable when running on Vercel
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import the FastAPI app
# This import triggers the app creation and lifespan setup
from jama_graphrag_api.api import app  # noqa: E402, F401

# Export the app for Vercel's @vercel/python runtime
# Vercel expects either 'app' or 'handler' as the default export
