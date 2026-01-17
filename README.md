# Jama MCP Server GraphRAG

GraphRAG MCP Server for Requirements Management Knowledge Graph.

## Overview

Production-ready GraphRAG backend for the **Jama Software "Essential Guide to Requirements Management and Traceability"** knowledge base.

## Installation

```bash
uv sync --extra dev
```

## Usage

### MCP Server (for Claude Desktop)

```bash
uv run jama-mcp-server-graphrag
```

### REST API (for React frontend)

```bash
uv run jama-graphrag-api
```

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check src/

# Format
uv run ruff format src/
```

## License

MIT
