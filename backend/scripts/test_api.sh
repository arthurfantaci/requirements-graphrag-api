#!/bin/bash
# =============================================================================
# Manual API Testing Script
# =============================================================================
# Prerequisites:
#   1. Configure .env file with valid credentials:
#      - NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
#      - OPENAI_API_KEY
#   2. Start the server: uv run uvicorn requirements_graphrag_api.api:app --reload
#
# Usage: ./scripts/test_api.sh
# =============================================================================

BASE_URL="${API_URL:-http://localhost:8000}"

echo "Testing Requirements GraphRAG API at $BASE_URL"
echo "========================================="

# Test root endpoint
echo -e "\n1. Root Endpoint"
curl -s "$BASE_URL/" | jq .

# Test health endpoint
echo -e "\n2. Health Check"
curl -s "$BASE_URL/health" | jq .

# Test vector search
echo -e "\n3. Vector Search"
curl -s -X POST "$BASE_URL/api/v1/search/vector" \
  -H "Content-Type: application/json" \
  -d '{"query": "requirements traceability", "limit": 5}' | jq .

# Test hybrid search
echo -e "\n4. Hybrid Search"
curl -s -X POST "$BASE_URL/api/v1/search/hybrid" \
  -H "Content-Type: application/json" \
  -d '{"query": "ISO 26262", "limit": 5, "keyword_weight": 0.3}' | jq .

# Test graph-enriched search
echo -e "\n5. Graph-Enriched Search"
curl -s -X POST "$BASE_URL/api/v1/search/graph" \
  -H "Content-Type: application/json" \
  -d '{"query": "functional safety", "limit": 5, "traversal_depth": 2}' | jq .

# Test glossary lookup
echo -e "\n6. Glossary Term Lookup"
curl -s "$BASE_URL/api/v1/glossary/terms/baseline" | jq .

# Test glossary search
echo -e "\n7. Glossary Search"
curl -s -X POST "$BASE_URL/api/v1/glossary/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "requirements", "limit": 5}' | jq .

# Test standards lookup
echo -e "\n8. Standards Lookup"
curl -s "$BASE_URL/api/v1/standards/lookup/ISO%2026262" | jq .

# Test chat endpoint
echo -e "\n9. Chat (RAG Q&A)"
curl -s -X POST "$BASE_URL/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is requirements traceability and why is it important?"}' | jq .

# Test schema endpoint
echo -e "\n10. Graph Schema"
curl -s "$BASE_URL/api/v1/schema" | jq .

echo -e "\n========================================="
echo "API testing complete!"
echo "For interactive testing, visit: $BASE_URL/docs"
