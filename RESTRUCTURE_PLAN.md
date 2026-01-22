# Project Restructuring Plan: requirements-graphrag-api

## Overview

Transform `jama-mcp-server-graphrag` into `requirements-graphrag-api` - a monorepo with Docker-based deployment using Railway (backend) + Vercel (frontend).

## Current State → Target State

```
CURRENT                              TARGET
jama-mcp-server-graphrag/            requirements-graphrag-api/
├── src/                             ├── backend/
│   └── jama_graphrag_api/           │   ├── src/requirements_graphrag_api/
├── api/                             │   ├── Dockerfile
│   └── index.py (Vercel)            │   ├── pyproject.toml
├── vercel.json                      │   └── .env.example
├── pyproject.toml                   ├── frontend/
└── ...                              │   ├── src/
                                     │   ├── Dockerfile
                                     │   ├── package.json
                                     │   └── vite.config.js
                                     ├── docker-compose.yml
                                     ├── docker-compose.prod.yml
                                     ├── railway.toml
                                     └── CLAUDE.md
```

## Phase 1: Backend Restructuring

### 1.1 Directory Changes
- [ ] Create `backend/` directory
- [ ] Move `src/jama_graphrag_api/` → `backend/src/requirements_graphrag_api/`
- [ ] Move `pyproject.toml` → `backend/pyproject.toml`
- [ ] Rename package from `jama_graphrag_api` to `requirements_graphrag_api`
- [ ] Update all imports in Python files
- [ ] Move `tests/` → `backend/tests/`

### 1.2 Docker Configuration (Backend)
- [ ] Create `backend/Dockerfile`
- [ ] Create `backend/.dockerignore`
- [ ] Remove Vercel-specific files (`api/`, `vercel.json`)

### 1.3 Railway Configuration
- [ ] Create `railway.toml` in root
- [ ] Configure environment variables template
- [ ] Set up health check endpoint

## Phase 2: Frontend Scaffolding

### 2.1 Create React + Vite + Tailwind Frontend
- [ ] `npm create vite@latest frontend -- --template react`
- [ ] Install Tailwind CSS
- [ ] Create basic chat interface components
- [ ] Configure API client for backend

### 2.2 Docker Configuration (Frontend)
- [ ] Create `frontend/Dockerfile` (multi-stage build)
- [ ] Create `frontend/.dockerignore`
- [ ] Create `frontend/nginx.conf` for production serving

### 2.3 Vercel Configuration
- [ ] Create `frontend/vercel.json`
- [ ] Configure environment variables for API URL

## Phase 3: Local Development Setup

### 3.1 Docker Compose
- [ ] Create root `docker-compose.yml` for local dev
- [ ] Backend service with hot reload
- [ ] Frontend service with hot reload
- [ ] Optional: Local Neo4j container for testing

### 3.2 Environment Configuration
- [ ] Create root `.env.example` with all variables
- [ ] Document local vs production env vars

## Phase 4: Documentation & Cleanup

### 4.1 Update Documentation
- [ ] Update root `CLAUDE.md` with new structure
- [ ] Update root `README.md`
- [ ] Create `backend/README.md`
- [ ] Create `frontend/README.md`
- [ ] Update `SPECIFICATION.md`

### 4.2 GitHub Repository
- [ ] Rename repository to `requirements-graphrag-api`
- [ ] Update all references in code and docs
- [ ] Update GitHub description and topics

### 4.3 Cleanup Old Deployment
- [ ] User: Delete Vercel project `jama-mcp-server-graphrag` via dashboard
- [ ] Remove old Vercel configuration files

## Phase 5: Deployment Setup

### 5.1 Railway (Backend)
- [ ] Create Railway project
- [ ] Connect to GitHub repo
- [ ] Configure build settings (Dockerfile in `/backend`)
- [ ] Set environment variables
- [ ] Configure custom domain (optional)

### 5.2 Vercel (Frontend)
- [ ] Create new Vercel project
- [ ] Connect to GitHub repo
- [ ] Set root directory to `/frontend`
- [ ] Configure environment variables (VITE_API_URL)
- [ ] Configure custom domain (optional)

## File Templates

### backend/Dockerfile
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ src/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn
CMD ["uv", "run", "uvicorn", "requirements_graphrag_api.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml
```yaml
services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - NEO4J_URI=${NEO4J_URI}
      - NEO4J_USERNAME=${NEO4J_USERNAME}
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LANGSMITH_API_KEY=${LANGSMITH_API_KEY}
      - LANGSMITH_TRACING=${LANGSMITH_TRACING:-false}
    volumes:
      - ./backend/src:/app/src:ro  # Hot reload in dev

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      target: development
    ports:
      - "5173:5173"
    environment:
      - VITE_API_URL=http://localhost:8000
    volumes:
      - ./frontend/src:/app/src:ro  # Hot reload in dev
    depends_on:
      - backend
```

### railway.toml
```toml
[build]
builder = "dockerfile"
dockerfilePath = "backend/Dockerfile"

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 3
```

## Estimated Effort

| Phase | Tasks | Complexity |
|-------|-------|------------|
| Phase 1 | Backend restructuring | Medium |
| Phase 2 | Frontend scaffolding | Medium |
| Phase 3 | Docker Compose setup | Low |
| Phase 4 | Documentation | Low |
| Phase 5 | Deployment setup | Medium |

## Pre-Implementation Checklist

- [ ] User approves this plan
- [ ] User deletes old Vercel project
- [ ] User has Railway account (or creates one)
- [ ] User confirms new name: `requirements-graphrag-api`

## Post-Implementation Verification

- [ ] `docker-compose up` runs both services locally
- [ ] Backend health check passes: `curl http://localhost:8000/health`
- [ ] Frontend loads: `http://localhost:5173`
- [ ] Frontend can call backend API
- [ ] Railway deployment successful
- [ ] Vercel deployment successful
- [ ] End-to-end test passes
