#!/usr/bin/env bash
# Verification script for requirements-graphrag-api
#
# Usage:
#   ./scripts/verify.sh pre-pr       # Tier 1: lint + format + tests
#   ./scripts/verify.sh post-deploy   # Tier 3: live endpoint health checks
#   ./scripts/verify.sh all           # Both tiers
#
# Environment variables:
#   BACKEND_URL   — Railway backend URL (default: http://localhost:8000)
#   FRONTEND_URL  — Vercel frontend URL (default: http://localhost:5173)
#
# See docs/internal/VERIFICATION_RUNBOOK.md for the full checklist.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"

BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:5173}"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }

# ---------------------------------------------------------------------------
# Tier 1: Pre-PR (lint + format + tests)
# ---------------------------------------------------------------------------
pre_pr() {
    echo ""
    echo "=== TIER 1: Pre-PR Verification ==="
    echo ""

    cd "$BACKEND_DIR"

    echo "[1/3] Ruff lint..."
    if uv run ruff check . 2>&1; then
        pass "Lint clean"
    else
        fail "Lint errors found"
        return 1
    fi

    echo "[2/3] Ruff format..."
    if uv run ruff format --check . 2>&1; then
        pass "Format clean"
    else
        fail "Format issues found (run: uv run ruff format .)"
        return 1
    fi

    echo "[3/3] Pytest..."
    if uv run pytest -q 2>&1; then
        pass "All tests passed"
    else
        fail "Test failures"
        return 1
    fi

    echo ""
    echo -e "${GREEN}=== Tier 1 PASSED ===${NC}"
}

# ---------------------------------------------------------------------------
# Tier 3: Post-Deploy (live endpoint checks)
# ---------------------------------------------------------------------------
post_deploy() {
    echo ""
    echo "=== TIER 3: Post-Deploy Verification ==="
    echo "Backend:  $BACKEND_URL"
    echo "Frontend: $FRONTEND_URL"
    echo ""

    local failures=0

    # 1. Health check
    echo "[1/5] Health check..."
    if HEALTH=$(curl -sf --max-time 10 "$BACKEND_URL/health" 2>/dev/null); then
        STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null)
        NEO4J=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('neo4j','unknown'))" 2>/dev/null)
        if [ "$STATUS" = "healthy" ] && [ "$NEO4J" = "connected" ]; then
            pass "healthy, neo4j connected"
        else
            warn "status=$STATUS, neo4j=$NEO4J"
            failures=$((failures + 1))
        fi
    else
        fail "Health endpoint unreachable"
        failures=$((failures + 1))
    fi

    # 2. Schema endpoint
    echo "[2/5] Schema endpoint..."
    if SCHEMA=$(curl -sf --max-time 10 "$BACKEND_URL/schema" 2>/dev/null); then
        LABEL_COUNT=$(echo "$SCHEMA" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('node_labels',[])))" 2>/dev/null)
        if [ "$LABEL_COUNT" -gt 0 ] 2>/dev/null; then
            pass "$LABEL_COUNT node labels"
        else
            fail "No node labels returned"
            failures=$((failures + 1))
        fi
    else
        fail "Schema endpoint unreachable"
        failures=$((failures + 1))
    fi

    # 3. Vector search
    echo "[3/5] Vector search..."
    if SEARCH=$(curl -sf --max-time 15 -X POST "$BACKEND_URL/search/vector" \
        -H "Content-Type: application/json" \
        -d '{"query":"traceability","top_k":3}' 2>/dev/null); then
        RESULT_COUNT=$(echo "$SEARCH" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('results',[])))" 2>/dev/null)
        if [ "$RESULT_COUNT" -gt 0 ] 2>/dev/null; then
            pass "$RESULT_COUNT results"
        else
            fail "No search results"
            failures=$((failures + 1))
        fi
    else
        fail "Search endpoint unreachable or errored"
        failures=$((failures + 1))
    fi

    # 4. Definitions endpoint
    echo "[4/5] Definitions endpoint..."
    if DEFS=$(curl -sf --max-time 10 "$BACKEND_URL/definitions" 2>/dev/null); then
        TERM_COUNT=$(echo "$DEFS" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('terms',[])))" 2>/dev/null)
        if [ "$TERM_COUNT" -gt 0 ] 2>/dev/null; then
            pass "$TERM_COUNT terms"
        else
            warn "No terms returned (may be expected if glossary is empty)"
        fi
    else
        fail "Definitions endpoint unreachable"
        failures=$((failures + 1))
    fi

    # 5. Frontend loads
    echo "[5/5] Frontend..."
    if HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" --max-time 10 "$FRONTEND_URL" 2>/dev/null); then
        if [ "$HTTP_CODE" = "200" ]; then
            pass "HTTP 200"
        else
            fail "HTTP $HTTP_CODE"
            failures=$((failures + 1))
        fi
    else
        fail "Frontend unreachable"
        failures=$((failures + 1))
    fi

    echo ""
    if [ "$failures" -eq 0 ]; then
        echo -e "${GREEN}=== Tier 3 PASSED ===${NC}"
    else
        echo -e "${RED}=== Tier 3 FAILED ($failures check(s)) ===${NC}"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "${1:-help}" in
    pre-pr)
        pre_pr
        ;;
    post-deploy)
        post_deploy
        ;;
    all)
        pre_pr
        echo ""
        post_deploy
        ;;
    *)
        echo "Usage: $0 {pre-pr|post-deploy|all}"
        echo ""
        echo "  pre-pr       Tier 1: lint + format + tests (local)"
        echo "  post-deploy  Tier 3: live endpoint health checks"
        echo "  all          Both tiers"
        echo ""
        echo "Environment:"
        echo "  BACKEND_URL   Railway URL (default: http://localhost:8000)"
        echo "  FRONTEND_URL  Vercel URL  (default: http://localhost:5173)"
        ;;
esac
