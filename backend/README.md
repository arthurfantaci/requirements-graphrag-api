# Requirements GraphRAG API - Backend

GraphRAG REST API for the Jama Software "Essential Guide to Requirements Management and Traceability" knowledge base.

## Features

- **Semantic Search**: Vector similarity search using OpenAI embeddings
- **Hybrid Search**: Combined vector and full-text retrieval
- **Graph-Enriched RAG**: Cypher traversals for context expansion
- **Text2Cypher**: Natural language to Cypher query generation
- **LangSmith Integration**: Observability and tracing

## Quick Start

```bash
# Install dependencies
uv sync

# Run locally
uv run uvicorn requirements_graphrag_api.api:app --reload

# Run with Docker
docker build -t requirements-graphrag-api .
docker run -p 8000:8000 --env-file .env requirements-graphrag-api
```

## Environment Variables

See `.env.example` for required configuration.

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health
