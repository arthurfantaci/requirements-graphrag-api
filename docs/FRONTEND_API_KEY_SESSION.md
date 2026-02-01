# Claude Code Session: Frontend API Key Integration

## Context

The backend now requires API key authentication (`REQUIRE_API_KEY=true`). The React frontend is failing because it doesn't include the `X-API-Key` header in API requests.

**Issue**: https://github.com/arthurfantaci/requirements-graphrag-api/issues/102
**Branch to create**: `fix/frontend-api-key`

## Project Git Workflow

This project follows a standard PR-based workflow:
1. Create feature branch from `main`
2. Make changes and commit with conventional commits
3. Push branch and create PR
4. Wait for CI (lint + tests) to pass
5. Squash merge via `gh pr merge --squash --delete-branch`
6. Clean up with `git checkout main && git pull && git fetch --prune`

## Implementation Tasks

### Task 1: Create feature branch
```bash
git checkout main
git pull origin main
git checkout -b fix/frontend-api-key
```

### Task 2: Create API utility module

**File**: `frontend/src/utils/api.js`

```javascript
/**
 * API utilities for making authenticated requests to the backend.
 */

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const API_KEY = import.meta.env.VITE_API_KEY || ''

/**
 * Get headers for API requests, including authentication if configured.
 * @param {Object} additionalHeaders - Additional headers to include
 * @returns {Object} Headers object
 */
export const getApiHeaders = (additionalHeaders = {}) => {
  const headers = {
    'Content-Type': 'application/json',
    ...additionalHeaders,
  }

  if (API_KEY) {
    headers['X-API-Key'] = API_KEY
  }

  return headers
}

/**
 * Make an authenticated fetch request to the API.
 * @param {string} endpoint - API endpoint (e.g., '/chat')
 * @param {Object} options - Fetch options
 * @returns {Promise<Response>}
 */
export const apiFetch = async (endpoint, options = {}) => {
  const { headers: customHeaders, ...restOptions } = options

  return fetch(`${API_URL}${endpoint}`, {
    ...restOptions,
    headers: getApiHeaders(customHeaders),
  })
}

export { API_URL, API_KEY }
```

### Task 3: Update useSSEChat.js

**File**: `frontend/src/hooks/useSSEChat.js`

Find the fetch call (around line 199) and add the API key header:

```javascript
// Add import at top
import { API_URL, API_KEY } from '../utils/api'

// Update the fetch call to include the header
const response = await fetch(`${API_URL}/chat`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    ...(API_KEY && { 'X-API-Key': API_KEY }),
  },
  body: JSON.stringify(payload),
})
```

Note: SSE/streaming requests may need special handling - check if the existing code uses EventSource or fetch with streaming.

### Task 4: Update FeedbackBar.jsx

**File**: `frontend/src/components/feedback/FeedbackBar.jsx`

```javascript
// Replace the API_URL constant with import
import { apiFetch } from '../../utils/api'

// Update the fetch call (around line 65)
const response = await apiFetch('/feedback', {
  method: 'POST',
  body: JSON.stringify(feedbackData),
})
```

### Task 5: Update ResponseActions.jsx

**File**: `frontend/src/components/feedback/ResponseActions.jsx`

```javascript
// Replace the API_URL constant with import
import { apiFetch } from '../../utils/api'

// Update the fetch call (around line 81)
await apiFetch('/feedback', {
  method: 'POST',
  body: JSON.stringify(feedbackPayload),
})
```

### Task 6: Update .env.example

**File**: `frontend/.env.example`

Add:
```
VITE_API_KEY=your_api_key_here
```

### Task 7: Test locally

```bash
cd frontend
echo "VITE_API_KEY=rgapi_YOUR_KEY_HERE" >> .env.local
npm run dev
```

Verify:
- Chat works
- Feedback submission works
- No console errors

### Task 8: Commit and create PR

```bash
git add -A
git commit -m "fix(frontend): add X-API-Key header to all API requests

- Create shared API utility with authentication headers
- Update useSSEChat.js to include API key
- Update FeedbackBar.jsx to use apiFetch
- Update ResponseActions.jsx to use apiFetch
- Add VITE_API_KEY to .env.example

Closes #102

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

git push -u origin fix/frontend-api-key

gh pr create --title "fix(frontend): add X-API-Key header to all API requests" --body "..."
```

### Task 9: After PR merges

User needs to add `VITE_API_KEY` environment variable in Vercel dashboard:
1. Go to Vercel project settings
2. Go to Environment Variables
3. Add: `VITE_API_KEY` = `<the API key created earlier>`
4. Redeploy

## Security Considerations

- The API key will be visible in browser DevTools (this is expected for frontend keys)
- Create a **separate key** for the frontend with limited scopes (chat, search, feedback)
- Never expose a key with `admin` scope to the frontend
- The existing enterprise key should be replaced with a standard-tier frontend key

## Recommended: Create dedicated frontend key

After fixing the code, create a frontend-specific key:

```bash
export AUTH_DATABASE_URL="postgresql://..."
python scripts/create_api_key.py create --name "Frontend App" --tier standard
```

Use this key (not the enterprise admin key) for `VITE_API_KEY`.
