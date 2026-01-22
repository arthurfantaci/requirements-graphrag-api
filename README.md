# Requirements GraphRAG API

GraphRAG-powered API and chat interface for the **Jama Software "Essential Guide to Requirements Management and Traceability"** knowledge base.

## Architecture

```
requirements-graphrag-api/
├── backend/                    # Python FastAPI + GraphRAG
│   ├── src/requirements_graphrag_api/
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/                   # React + Vite + Tailwind
│   ├── src/
│   ├── Dockerfile
│   └── package.json
├── docker-compose.yml          # Local development
└── railway.toml                # Railway deployment config
```

## Quick Start

### Local Development with Docker

```bash
# Copy environment template
cp .env.example .env
# Edit .env with your credentials

# Start both services
docker-compose up

# Access:
# - Backend API: http://localhost:8000
# - Frontend: http://localhost:5173
# - API Docs: http://localhost:8000/docs
```

### Local Development without Docker

```bash
# Backend
cd backend
uv sync --extra dev
uv run uvicorn requirements_graphrag_api.api:app --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

## Deployment

| Component | Platform | Configuration |
|-----------|----------|---------------|
| Backend | Railway | `railway.toml` |
| Frontend | Vercel | `frontend/vercel.json` |

### Railway (Backend)

1. Create a new Railway project
2. Connect your GitHub repo
3. Set build command: Uses `railway.toml` automatically
4. Add environment variables from `.env.example`

### Vercel (Frontend)

1. Create new Vercel project
2. Set root directory to `frontend`
3. Add environment variable: `VITE_API_URL=https://your-railway-app.railway.app`

## Features

- **Semantic Search**: Vector similarity using OpenAI embeddings
- **Hybrid Search**: Combined vector + full-text retrieval
- **Graph-Enriched RAG**: Neo4j Cypher traversals for context
- **Text2Cypher**: Natural language to Cypher queries
- **LangSmith Integration**: Observability and tracing

## Environment Variables

See `.env.example` for full configuration. Key variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `NEO4J_URI` | Yes | Neo4j connection URI |
| `NEO4J_PASSWORD` | Yes | Neo4j password |
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `LANGSMITH_API_KEY` | No | LangSmith for tracing |
| `VITE_API_URL` | Yes (frontend) | Backend API URL |

## License

MIT
