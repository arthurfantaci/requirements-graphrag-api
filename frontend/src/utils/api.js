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
